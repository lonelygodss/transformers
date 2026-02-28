# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Calibration utility for MSD-first time-domain truncated dot-product simulation.

Runs a calibration forward pass over a small dataset, records per-channel
activation/weight statistics, and computes optimal B_base per channel.
Results are stored as `msd_calibration_data` in the model config.

Usage:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3.calibration_msd import calibrate_channel_budgets

    model = AutoModelForCausalLM.from_pretrained("path/to/model")
    tokenizer = AutoTokenizer.from_pretrained("path/to/model")

    calibrate_channel_budgets(model, tokenizer, texts, target_snr_db=30.0)
    model.config.save_pretrained("path/to/output")
"""

import math
from collections import defaultdict

import torch

from ...utils import logging


logger = logging.get_logger(__name__)


def _safe_log2_cal(x):
    """log2 that maps 0 and negative values to a large negative number."""
    return torch.where(x > 0, torch.log2(x), torch.tensor(-60.0, dtype=x.dtype, device=x.device))


def _compute_block_delay_stats(x_q, x_scales, w_q, w_scales, block_size, format_max):
    """
    Compute delay statistics for a single linear layer's forward pass.

    Returns per-output-channel statistics:
        - mean inter-block delay across blocks
        - max inter-block delay
        - mean intra-block delay
        - max intra-block delay
        - fraction of blocks with delay >= various thresholds
    """
    N = x_q.shape[0]
    nb = x_q.shape[1]
    bs = x_q.shape[2]
    out = w_q.shape[0]

    # Inter-block delays: (N, out, nb)
    log2_x = _safe_log2_cal(x_scales).unsqueeze(1)
    log2_w = _safe_log2_cal(w_scales).unsqueeze(0)
    combined_e = torch.floor(log2_x + log2_w)
    e_max = combined_e.amax(dim=-1)  # (N, out)
    inter_delays = e_max.unsqueeze(-1) - combined_e  # (N, out, nb)

    # Intra-block delays: (N, nb, bs)
    abs_vals = x_q.abs()
    elem_log2 = torch.where(
        abs_vals > 0,
        torch.floor(torch.log2(abs_vals)),
        torch.tensor(-60.0, dtype=x_q.dtype, device=x_q.device),
    )
    e_max_block = elem_log2.amax(dim=-1, keepdim=True)
    intra_delays = e_max_block - elem_log2  # (N, nb, bs)

    # Per-output-channel statistics (aggregate over N)
    # inter_delays: (N, out, nb) -> mean over N and nb
    mean_inter = inter_delays.mean(dim=(0, 2))  # (out,)
    max_inter = inter_delays.amax(dim=(0, 2))  # (out,)

    # intra_delays: (N, nb, bs) -> same for all output channels
    mean_intra = intra_delays.mean()
    max_intra = intra_delays.amax()

    return {
        "mean_inter": mean_inter,
        "max_inter": max_inter,
        "mean_intra": float(mean_intra),
        "max_intra": float(max_intra),
        "inter_delays": inter_delays,  # full tensor for budget search
        "intra_delays": intra_delays,
    }


def _find_budget_for_snr(
    x_q,
    x_scales,
    w_q,
    w_scales,
    delay_stats,
    online_delay,
    target_snr_db=30.0,
    budget_range=(4, 48),
):
    """
    Binary search for the minimum per-channel budget B_base such that
    the SNR of the truncated dot-product vs exact stays above target_snr_db.

    Returns: budget tensor of shape (out_features,)
    """
    from .modular_qwen3 import _msd_truncate

    N = x_q.shape[0]
    out = w_q.shape[0]
    nb = x_q.shape[1]
    bs = x_q.shape[2]

    inter_delays = delay_stats["inter_delays"]  # (N, out, nb)
    intra_delays = delay_stats["intra_delays"]  # (N, nb, bs)

    # Exact products (no truncation)
    prods_exact = x_q.unsqueeze(1) * w_q.unsqueeze(0)  # (N, out, nb, bs)
    combined_scales = x_scales.unsqueeze(1) * w_scales.unsqueeze(0)  # (N, out, nb)
    exact_block_dots = prods_exact.sum(dim=-1)  # (N, out, nb)
    exact_result = (exact_block_dots * combined_scales).sum(dim=-1)  # (N, out)

    # Per-channel binary search
    budgets = torch.full((out,), float(budget_range[1]), dtype=torch.float32)
    lo = torch.full((out,), float(budget_range[0]), dtype=torch.float32)
    hi = torch.full((out,), float(budget_range[1]), dtype=torch.float32)

    for _ in range(12):  # ~12 iterations for 4-48 range
        mid = torch.floor((lo + hi) / 2.0)
        # Compute truncated result at budget=mid for each channel
        b_expanded = mid.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, out, 1, 1)
        total_delay = (
            inter_delays.unsqueeze(-1)
            + intra_delays.unsqueeze(1)
            + online_delay
        )  # (N, out, nb, bs)
        p_eff = torch.clamp(b_expanded - total_delay, min=0.0)
        prods_trunc = _msd_truncate(prods_exact, p_eff)
        block_dots_trunc = prods_trunc.sum(dim=-1)  # (N, out, nb)
        result_trunc = (block_dots_trunc * combined_scales).sum(dim=-1)  # (N, out)

        # SNR per channel: SNR = 10 * log10(signal_power / noise_power)
        signal_power = (exact_result**2).mean(dim=0)  # (out,)
        noise_power = ((exact_result - result_trunc) ** 2).mean(dim=0) + 1e-30
        snr = 10.0 * torch.log10(signal_power / noise_power)

        # Binary search update: if SNR >= target, try lower budget
        good = snr >= target_snr_db
        hi = torch.where(good, mid, hi)
        lo = torch.where(good, lo, mid + 1)

    budgets = hi  # conservative: use the upper bound
    return budgets


def calibrate_channel_budgets(
    model,
    tokenizer,
    calibration_texts,
    target_snr_db=30.0,
    max_length=512,
    batch_size=4,
    online_delay=None,
):
    """
    Run calibration to determine per-channel MSD cycle budgets.

    Runs forward passes over calibration_texts with exact MX mode,
    hooks into each MXFP linear layer to collect block-level statistics,
    then finds the minimum budget per channel meeting the target SNR.

    Args:
        model: Qwen3ForCausalLM (or similar) with MXFP layers
        tokenizer: associated tokenizer
        calibration_texts: list of strings for calibration
        target_snr_db: target signal-to-noise ratio in dB (default: 30.0)
        max_length: max token length for calibration inputs
        batch_size: batch size for calibration passes
        online_delay: MSD online delay (default: from config)

    Side Effects:
        Sets model.config.msd_calibration_data with per-layer per-channel budgets.
    """
    # Avoid circular import at module level
    from .modular_qwen3 import _MXFPLinearBase

    config = model.config
    if online_delay is None:
        online_delay = getattr(config, "msd_online_delay", 2)

    device = next(model.parameters()).device
    model.eval()

    # Collect MXFP layers
    mxfp_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, _MXFPLinearBase):
            mxfp_layers[name] = module

    if not mxfp_layers:
        logger.warning("No MXFP linear layers found. Calibration skipped.")
        return

    logger.info(f"Calibrating {len(mxfp_layers)} MXFP layers over {len(calibration_texts)} samples")

    # Hook to capture inputs and quantized blocks
    layer_data = defaultdict(list)
    hooks = []

    def make_hook(layer_name, layer_module):
        def hook_fn(module, args, output):
            x = args[0]  # input tensor
            with torch.no_grad():
                batch_shape = x.shape[:-1]
                N = math.prod(batch_shape) if batch_shape else 1
                x_2d = x.float().reshape(N, layer_module.in_features)
                w_2d = layer_module.weight.float()
                x_q, x_scales, _ = layer_module._prepare_blocks(x_2d, N)
                w_q, w_scales, _ = layer_module._prepare_blocks(w_2d, layer_module.out_features)
                layer_data[layer_name].append(
                    (x_q.cpu(), x_scales.cpu(), w_q.cpu(), w_scales.cpu())
                )
        return hook_fn

    for name, module in mxfp_layers.items():
        h = module.register_forward_hook(make_hook(name, module))
        hooks.append(h)

    # Run forward passes
    # Temporarily disable MSD truncation for exact computation
    old_msd = getattr(config, "use_msd_truncation", False)
    config.use_msd_truncation = False

    try:
        with torch.no_grad():
            for i in range(0, len(calibration_texts), batch_size):
                batch_texts = calibration_texts[i : i + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                model(**inputs)
    finally:
        config.use_msd_truncation = old_msd
        for h in hooks:
            h.remove()

    # Process collected data to find per-channel budgets
    calibration_result = {}

    for layer_name, data_list in layer_data.items():
        logger.info(f"Computing budgets for {layer_name} ({len(data_list)} batches)")
        # Concatenate all batches
        all_x_q = torch.cat([d[0] for d in data_list], dim=0)
        all_x_scales = torch.cat([d[1] for d in data_list], dim=0)
        # Weights are the same across batches; take from first
        w_q = data_list[0][2]
        w_scales = data_list[0][3]

        # Compute delay statistics
        stats = _compute_block_delay_stats(
            all_x_q, all_x_scales, w_q, w_scales,
            mxfp_layers[layer_name].block_size,
            mxfp_layers[layer_name].FORMAT_MAX,
        )

        # Find budget per channel
        budgets = _find_budget_for_snr(
            all_x_q, all_x_scales, w_q, w_scales,
            stats, online_delay, target_snr_db,
        )

        calibration_result[layer_name] = budgets.tolist()
        logger.info(
            f"  {layer_name}: budget range [{min(budgets.tolist()):.0f}, {max(budgets.tolist()):.0f}], "
            f"mean={budgets.mean():.1f}"
        )

    # Store in config
    config.msd_calibration_data = calibration_result
    logger.info("Calibration complete. Results stored in config.msd_calibration_data")
    return calibration_result


def apply_calibration_to_config(config, calibration_data):
    """
    Load calibration data dict into a Qwen3Config.

    Args:
        config: Qwen3Config instance
        calibration_data: dict mapping layer names to lists of per-channel budgets
    """
    config.msd_calibration_data = calibration_data
