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

All heavy computation runs on GPU with output-chunked loops to cap peak
memory at ~2 GiB of intermediate 4D tensors, mirroring the chunking
strategy used in `_forward_msd_truncated`.

Usage (CLI — recommended):
    cd /home/xzj/coding/onlinearith
    python calibrate.py --list                              # list setups
    python calibrate.py --setup 1                           # single format
    torchrun --nproc_per_node=4 calibrate.py                # all 4 formats

Usage (library):
    from transformers.models.qwen3.calibration_msd import calibrate_channel_budgets

    calibrate_channel_budgets(model, tokenizer, texts, target_snr_db=30.0)
"""

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from ...utils import logging


logger = logging.get_logger(__name__)

# Peak 4D intermediate tensor budget (bytes).  Matches _MSD_CHUNK_TARGET_BYTES
# in modeling_qwen3.py.  With float32 elements of size 4, a chunk of `c`
# output channels produces tensors of shape (N, c, nb, bs) = c * N * nb * bs * 4.
_CAL_CHUNK_TARGET_BYTES: int = 256 * 1024**2  # 256 MiB — keeps _msd_truncate peak within ~1 GiB
_BUDGET_MIN: int = 4
_BUDGET_MAX: int = 48


def _safe_log2_cal(x):
    """log2 that maps 0 and negative values to a large negative number."""
    return torch.where(x > 0, torch.log2(x), torch.tensor(-60.0, dtype=x.dtype, device=x.device))


def _compute_intra_delays(x_q):
    """
    Compute intra-block delays from quantized activation blocks.

    Args:
        x_q: (N, nb, bs) quantized activation blocks

    Returns:
        intra_delays: (N, nb, bs) — per-element delay within each block
    """
    abs_vals = x_q.abs()
    elem_log2 = torch.where(
        abs_vals > 0,
        torch.floor(torch.log2(abs_vals)),
        torch.tensor(-60.0, dtype=x_q.dtype, device=x_q.device),
    )
    e_max_block = elem_log2.amax(dim=-1, keepdim=True)
    return e_max_block - elem_log2  # (N, nb, bs)


# ── Staged calibration API ──────────────────────────────────────────────────


@dataclass
class LayerBlockCache:
    """
    Immutable cache of per-layer block data for budget optimization.

    Stores concatenated activations and a single copy of weights (fixing
    the duplication bug in the original hook where w_q/w_scales were stored
    per batch).  All tensors live on GPU.
    """
    layer_name: str
    x_q: torch.Tensor          # (N, nb, bs) concatenated activations
    x_scales: torch.Tensor     # (N, nb)
    w_q: torch.Tensor          # (out, nb, bs) single weight copy
    w_scales: torch.Tensor     # (out, nb)
    out_features: int
    online_delay: int
    device: torch.device
    # Pre-computed intermediates (computed once in from_layer_data)
    intra_delays: torch.Tensor   # (N, nb, bs)
    log2_x: torch.Tensor        # (N, 1, nb)
    log2_w_full: torch.Tensor   # (out, nb)
    x_q_exp: torch.Tensor       # (N, 1, nb, bs)
    intra_exp: torch.Tensor     # (N, 1, nb, bs)
    x_scales_exp: torch.Tensor  # (N, 1, nb)
    chunk_size: int

    @classmethod
    def from_layer_data(
        cls,
        layer_name: str,
        x_q_list: list,
        x_scales_list: list,
        w_q: torch.Tensor,
        w_scales: torch.Tensor,
        online_delay: int,
    ) -> "LayerBlockCache":
        """Construct cache from per-batch activation lists and single weight tensors."""
        device = x_q_list[0].device
        x_q = torch.cat(x_q_list, dim=0)
        x_scales = torch.cat(x_scales_list, dim=0)
        N, nb, bs = x_q.shape
        out = w_q.shape[0]

        intra_delays = _compute_intra_delays(x_q)
        log2_x = _safe_log2_cal(x_scales).unsqueeze(1)
        log2_w_full = _safe_log2_cal(w_scales)
        x_q_exp = x_q.unsqueeze(1)
        intra_exp = intra_delays.unsqueeze(1)
        x_scales_exp = x_scales.unsqueeze(1)

        elem_per_slice = N * nb * bs
        chunk_size = max(1, _CAL_CHUNK_TARGET_BYTES // (4 * elem_per_slice))
        chunk_size = min(chunk_size, out)

        return cls(
            layer_name=layer_name,
            x_q=x_q, x_scales=x_scales,
            w_q=w_q, w_scales=w_scales,
            out_features=out, online_delay=online_delay, device=device,
            intra_delays=intra_delays,
            log2_x=log2_x, log2_w_full=log2_w_full,
            x_q_exp=x_q_exp, intra_exp=intra_exp, x_scales_exp=x_scales_exp,
            chunk_size=chunk_size,
        )


def collect_layer_block_cache(
    model,
    tokenizer,
    calibration_texts,
    max_length: int = 512,
    batch_size: int = 4,
    online_delay: Optional[int] = None,
    show_progress: bool = False,
    progress_prefix: str = "",
) -> dict:
    """
    Stage 1: Capture block-level data from forward passes.

    Runs forward passes with MSD disabled, hooks into each MXFP linear layer,
    and collects (x_q, x_scales) per batch plus (w_q, w_scales) once per layer.

    Returns:
        dict mapping layer_name -> LayerBlockCache
    """
    from .modeling_qwen3 import _MXFPLinearBase

    config = model.config
    if online_delay is None:
        online_delay = getattr(config, "msd_online_delay", 2)
    assert online_delay is not None  # for type checker

    device = next(model.parameters()).device
    model.eval()

    mxfp_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, _MXFPLinearBase):
            mxfp_layers[name] = module

    if not mxfp_layers:
        logger.warning("No MXFP linear layers found.")
        return {}

    # Separate storage: activations per batch, weights once
    layer_x_q: dict = defaultdict(list)
    layer_x_scales: dict = defaultdict(list)
    layer_w_q: dict = {}
    layer_w_scales: dict = {}

    hooks = []

    def make_hook(layer_name, layer_module):
        def hook_fn(module, args, output):
            x = args[0]
            with torch.no_grad():
                batch_shape = x.shape[:-1]
                N = math.prod(batch_shape) if batch_shape else 1
                x_2d = x.float().reshape(N, layer_module.in_features)

                x_q, x_scales, _ = layer_module._prepare_blocks(x_2d, N)
                layer_x_q[layer_name].append(x_q)
                layer_x_scales[layer_name].append(x_scales)

                if layer_name not in layer_w_q:
                    w_2d = layer_module.weight.float()
                    w_q, w_scales, _ = layer_module._prepare_blocks(
                        w_2d, layer_module.out_features
                    )
                    layer_w_q[layer_name] = w_q
                    layer_w_scales[layer_name] = w_scales
        return hook_fn

    for name, module in mxfp_layers.items():
        h = module.register_forward_hook(make_hook(name, module))
        hooks.append(h)

    old_msd = getattr(config, "use_msd_truncation", False)
    config.use_msd_truncation = False

    try:
        with torch.no_grad():
            num_batches = (len(calibration_texts) + batch_size - 1) // batch_size
            batch_iter = range(0, len(calibration_texts), batch_size)
            batch_iter = tqdm(
                batch_iter, total=num_batches,
                desc=f"{progress_prefix}capture forward" if progress_prefix else "capture forward",
                disable=not show_progress,
            )
            for i in batch_iter:
                batch_texts = calibration_texts[i:i + batch_size]
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

    caches = {}
    for layer_name in mxfp_layers:
        if layer_name not in layer_x_q:
            continue
        caches[layer_name] = LayerBlockCache.from_layer_data(
            layer_name=layer_name,
            x_q_list=layer_x_q[layer_name],
            x_scales_list=layer_x_scales[layer_name],
            w_q=layer_w_q[layer_name],
            w_scales=layer_w_scales[layer_name],
            online_delay=online_delay,
        )

    logger.info(f"Captured block data for {len(caches)} layers")
    return caches


# ── Reusable computation helpers ─────────────────────────────────────────────


def _compute_exact_result(cache: LayerBlockCache) -> torch.Tensor:
    """Compute exact (non-truncated) MX dot-product result.  Returns (N, out)."""
    out = cache.out_features
    N = cache.x_q.shape[0]

    exact_result = torch.zeros(N, out, dtype=torch.float32, device=cache.device)
    for j0 in range(0, out, cache.chunk_size):
        j1 = min(j0 + cache.chunk_size, out)
        w_q_c = cache.w_q[j0:j1]
        w_scales_c = cache.w_scales[j0:j1]
        prods = cache.x_q_exp * w_q_c.unsqueeze(0)
        block_dots = prods.sum(dim=-1)
        combined_scales = cache.x_scales_exp * w_scales_c.unsqueeze(0)
        exact_result[:, j0:j1] = (block_dots * combined_scales).sum(dim=-1)
        del prods, block_dots, combined_scales
    return exact_result


def _compute_truncated_result(
    cache: LayerBlockCache,
    budgets: torch.Tensor,
    exact_result: Optional[torch.Tensor] = None,
) -> tuple:
    """
    Compute truncated result for a given budget vector.

    Args:
        cache: LayerBlockCache with precomputed intermediates
        budgets: (out,) per-channel budget values
        exact_result: if provided, also returns per-channel squared error

    Returns:
        (result, channel_errors) where channel_errors is (out,) or None
    """
    from .modeling_qwen3 import _msd_truncate

    out = cache.out_features
    N = cache.x_q.shape[0]
    device = cache.device

    result = torch.zeros(N, out, dtype=torch.float32, device=device)
    for j0 in range(0, out, cache.chunk_size):
        j1 = min(j0 + cache.chunk_size, out)
        w_q_c = cache.w_q[j0:j1]
        w_scales_c = cache.w_scales[j0:j1]
        b_c = budgets[j0:j1]

        log2_w_c = cache.log2_w_full[j0:j1].unsqueeze(0)
        combined_e = torch.floor(cache.log2_x + log2_w_c)
        e_max_c = combined_e.amax(dim=-1)
        inter_delays_c = e_max_c.unsqueeze(-1) - combined_e

        prods = cache.x_q_exp * w_q_c.unsqueeze(0)
        total_delay = inter_delays_c.unsqueeze(-1) + cache.intra_exp + cache.online_delay
        b_exp = b_c.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        p_eff = torch.clamp(b_exp - total_delay, min=0.0)
        del total_delay

        prods_trunc = _msd_truncate(prods, p_eff)
        del prods, p_eff
        block_dots = prods_trunc.sum(dim=-1)
        del prods_trunc
        combined_scales = cache.x_scales_exp * w_scales_c.unsqueeze(0)
        result[:, j0:j1] = (block_dots * combined_scales).sum(dim=-1)
        del block_dots, combined_scales, inter_delays_c

    channel_errors = None
    if exact_result is not None:
        channel_errors = ((exact_result - result) ** 2).sum(dim=0)
    return result, channel_errors


# ── Staged solvers ───────────────────────────────────────────────────────────


def solve_min_snr_budgets_from_cache(
    cache: LayerBlockCache,
    target_snr_db: float = 30.0,
    budget_range: tuple = (_BUDGET_MIN, _BUDGET_MAX),
    collect_channel_detail: bool = False,
) -> tuple:
    """
    Stage 2a: Binary search for per-channel SNR-min budgets using cached data.

    Equivalent to ``_find_budget_for_snr()`` but operates on a
    ``LayerBlockCache`` instead of raw tensors.

    Returns:
        (budgets, layer_summary, channel_detail)
    """
    device = cache.device
    out = cache.out_features
    N = cache.x_q.shape[0]

    exact_result = _compute_exact_result(cache)
    signal_power = (exact_result ** 2).mean(dim=0)

    lo = torch.full((out,), float(budget_range[0]), dtype=torch.float32, device=device)
    hi = torch.full((out,), float(budget_range[1]), dtype=torch.float32, device=device)

    for _ in range(12):
        mid = torch.floor((lo + hi) / 2.0)
        result_trunc, _ = _compute_truncated_result(cache, mid)
        noise_power = ((exact_result - result_trunc) ** 2).mean(dim=0) + 1e-30
        snr = 10.0 * torch.log10(signal_power / noise_power)
        del result_trunc

        good = snr >= target_snr_db
        hi = torch.where(good, mid, hi)
        lo = torch.where(good, lo, mid + 1)

    budgets = hi

    # Final SNR computation
    result_final, _ = _compute_truncated_result(cache, budgets)
    noise_power_final = ((exact_result - result_final) ** 2).mean(dim=0) + 1e-30
    snr_at_budget = 10.0 * torch.log10(signal_power / noise_power_final)
    signal_power_db = 10.0 * torch.log10(signal_power.clamp(min=1e-30))
    del result_final, noise_power_final

    layer_summary, channel_detail = _collect_channel_stats(
        budgets, cache.x_q_exp, cache.x_scales_exp, cache.intra_exp,
        cache.log2_x, cache.log2_w_full, cache.online_delay,
        cache.chunk_size, out, N,
        snr_at_budget=snr_at_budget,
        signal_power_db=signal_power_db,
        collect_channel_detail=collect_channel_detail,
        budget_range=budget_range,
    )
    return budgets, layer_summary, channel_detail


@dataclass
class LayerErrorCurves:
    """Error curves E_j(b) for all channels in a single layer."""
    layer_name: str
    num_channels: int
    budget_snr_min: np.ndarray       # (num_channels,) int budgets from snr_min
    budget_range: tuple               # global (min, max)
    budget_values: np.ndarray         # (num_budget_points,) sorted unique values
    errors: np.ndarray                # (num_channels, num_budget_points) squared error

    def error_at(self, channel: int, budget: int) -> float:
        """Look up E_j(b) from the pre-computed table."""
        idx = np.searchsorted(self.budget_values, budget)
        if idx >= len(self.budget_values) or self.budget_values[idx] != budget:
            return float("inf")
        return float(self.errors[channel, idx])


def build_error_curves_from_cache(
    cache: LayerBlockCache,
    budgets_snr_min: torch.Tensor,
    window: int = 3,
    budget_range: tuple = (_BUDGET_MIN, _BUDGET_MAX),
) -> LayerErrorCurves:
    """
    Build per-channel error curves E_j(b) around the SNR-min budget.

    For each channel j, evaluates E_j(b) = sum_n (y_exact[n,j] - y_hat[n,j;b])^2
    for b in [B_j* - window, B_j* + window], clamped to budget_range.

    Args:
        cache: LayerBlockCache
        budgets_snr_min: (out,) SNR-min budgets
        window: how many budget points on each side
        budget_range: global limits

    Returns:
        LayerErrorCurves
    """
    device = cache.device
    out = cache.out_features

    exact_result = _compute_exact_result(cache)

    budgets_cpu = budgets_snr_min.cpu().numpy().astype(np.int32)
    all_bvals = set()
    for b_star in budgets_cpu:
        lo = max(budget_range[0], int(b_star) - window)
        hi = min(budget_range[1], int(b_star) + window)
        for b in range(lo, hi + 1):
            all_bvals.add(b)

    budget_values = np.array(sorted(all_bvals), dtype=np.int32)
    num_points = len(budget_values)
    errors = np.zeros((out, num_points), dtype=np.float64)

    for i, b in enumerate(budget_values):
        b_tensor = torch.full((out,), float(b), dtype=torch.float32, device=device)
        _, ch_err = _compute_truncated_result(cache, b_tensor, exact_result)
        errors[:, i] = ch_err.cpu().numpy()

    return LayerErrorCurves(
        layer_name=cache.layer_name,
        num_channels=out,
        budget_snr_min=budgets_cpu,
        budget_range=budget_range,
        budget_values=budget_values,
        errors=errors,
    )


def solve_fixed_sum_from_error_curves(
    curves: LayerErrorCurves,
    target_sum: Optional[int] = None,
    max_iterations: int = 100000,
) -> tuple:
    """
    Redistribute cycles to minimise total error while preserving budget sum.

    Greedy donor/receiver swap using heaps:
      - donor loss:    L_j = E_j(B_j - 1) - E_j(B_j)
      - receiver gain: G_j = E_j(B_j) - E_j(B_j + 1)
      - swap one cycle from smallest-loss donor to largest-gain receiver
        whenever G > L.

    Constraints: sum(B) == target_sum, budget_range[0] <= B_j <= budget_range[1].

    Returns:
        (budgets, stats)  where budgets is np.ndarray (int) and stats is dict
    """
    bmin, bmax = curves.budget_range
    budgets = curves.budget_snr_min.copy()
    n_ch = curves.num_channels
    bvals = curves.budget_values
    bval_set = set(bvals.tolist())

    if target_sum is None:
        target_sum = int(budgets.sum())

    # Build index map for fast lookup
    bval_to_idx = {int(b): i for i, b in enumerate(bvals)}

    def _delta_minus(j):
        """Loss from donating one cycle: E_j(B_j-1) - E_j(B_j)."""
        b = int(budgets[j])
        if b <= bmin:
            return float("inf")
        b_minus = b - 1
        if b_minus not in bval_set or b not in bval_set:
            return float("inf")
        return curves.errors[j, bval_to_idx[b_minus]] - curves.errors[j, bval_to_idx[b]]

    def _delta_plus(j):
        """Gain from receiving one cycle: E_j(B_j) - E_j(B_j+1)."""
        b = int(budgets[j])
        if b >= bmax:
            return float("inf")
        b_plus = b + 1
        if b_plus not in bval_set or b not in bval_set:
            return float("inf")
        return curves.errors[j, bval_to_idx[b]] - curves.errors[j, bval_to_idx[b_plus]]

    # Heaps: (value, channel_idx)
    # donor_heap: min-heap on delta_minus (pop smallest loss)
    # receiver_heap: min-heap on -delta_plus (pop largest gain)
    donor_heap = []
    receiver_heap = []

    for j in range(n_ch):
        dm = _delta_minus(j)
        if dm < float("inf"):
            heapq.heappush(donor_heap, (dm, j))
        dp = _delta_plus(j)
        if dp < float("inf"):
            heapq.heappush(receiver_heap, (-dp, j))

    swaps = 0
    total_improvement = 0.0

    for _ in range(max_iterations):
        # Drain stale entries from donor heap
        while donor_heap:
            dm_top, j_d = donor_heap[0]
            dm_actual = _delta_minus(j_d)
            if abs(dm_top - dm_actual) < 1e-12:
                break
            heapq.heapreplace(donor_heap, (dm_actual, j_d))
        # Drain stale entries from receiver heap
        while receiver_heap:
            neg_dp_top, j_r = receiver_heap[0]
            dp_actual = _delta_plus(j_r)
            if abs(-neg_dp_top - dp_actual) < 1e-12:
                break
            heapq.heapreplace(receiver_heap, (-dp_actual, j_r))

        if not donor_heap or not receiver_heap:
            break

        dm_best, j_d = donor_heap[0]
        dp_best = -receiver_heap[0][0]
        j_r = receiver_heap[0][1]

        if dp_best <= dm_best or j_d == j_r:
            break

        # Perform swap
        heapq.heappop(donor_heap)
        heapq.heappop(receiver_heap)

        budgets[j_d] -= 1
        budgets[j_r] += 1
        swaps += 1
        total_improvement += dp_best - dm_best

        # Re-insert both channels into both heaps
        for j in (j_d, j_r):
            dm = _delta_minus(j)
            if dm < float("inf"):
                heapq.heappush(donor_heap, (dm, j))
            dp = _delta_plus(j)
            if dp < float("inf"):
                heapq.heappush(receiver_heap, (-dp, j))

    stats = {
        "swaps_performed": swaps,
        "total_improvement": total_improvement,
        "final_sum": int(budgets.sum()),
        "target_sum": target_sum,
        "sum_preserved": int(budgets.sum()) == target_sum,
    }
    return budgets, stats


def evaluate_budget_vector_from_cache(
    cache: LayerBlockCache,
    budgets: torch.Tensor,
) -> dict:
    """
    Evaluate a budget vector against cached data.

    Returns dict with per-channel SNR, error, and aggregate metrics.
    """
    exact_result = _compute_exact_result(cache)
    signal_power = (exact_result ** 2).mean(dim=0)
    N = cache.x_q.shape[0]

    _, channel_errors = _compute_truncated_result(cache, budgets, exact_result)
    noise_power = channel_errors / N + 1e-30
    snr_per_channel = 10.0 * torch.log10(signal_power / noise_power)

    return {
        "layer_snr_mean": round(float(snr_per_channel.mean()), 2),
        "layer_snr_min": round(float(snr_per_channel.min()), 2),
        "total_error": float(channel_errors.sum()),
        "budget_sum": float(budgets.sum()),
        "budget_mean": round(float(budgets.mean()), 2),
    }


def _find_budget_for_snr(
    x_q,
    x_scales,
    w_q,
    w_scales,
    online_delay,
    target_snr_db=30.0,
    budget_range=(4, 48),
    collect_channel_detail=False,
):
    """
    Output-chunked binary search for per-channel budget B_base meeting target SNR.

    All computation runs on the same device as the input tensors (GPU).
    The output dimension is processed in chunks so that the peak 4D
    intermediate tensor ``(N, chunk, nb, bs)`` stays under
    ``_CAL_CHUNK_TARGET_BYTES``.

    Args:
        collect_channel_detail: if True, return full per-channel detail arrays
            in addition to the layer summary.  When False (default), only
            the compact layer summary is returned (much smaller JSON).

    Returns:
        (budgets, layer_summary, channel_detail) where:
        - budgets: tensor of shape (out_features,) on the input device
        - layer_summary: dict of scalar summary statistics for this layer
        - channel_detail: dict of per-channel lists (only populated when
          collect_channel_detail=True, else empty dict)
    """
    from .modeling_qwen3 import _msd_truncate

    device = x_q.device
    N = x_q.shape[0]
    out = w_q.shape[0]
    nb = x_q.shape[1]
    bs = x_q.shape[2]

    # ── Pre-compute small tensors ──
    intra_delays = _compute_intra_delays(x_q)             # (N, nb, bs)
    log2_x = _safe_log2_cal(x_scales).unsqueeze(1)        # (N, 1, nb)
    log2_w_full = _safe_log2_cal(w_scales)                 # (out, nb)
    intra_exp = intra_delays.unsqueeze(1)                  # (N, 1, nb, bs)
    x_q_exp = x_q.unsqueeze(1)                             # (N, 1, nb, bs)
    x_scales_exp = x_scales.unsqueeze(1)                   # (N, 1, nb)

    # ── Determine output chunk size ──
    elem_per_slice = N * nb * bs  # elements in one (N, 1, nb, bs) slice
    chunk_size = max(1, _CAL_CHUNK_TARGET_BYTES // (4 * elem_per_slice))
    chunk_size = min(chunk_size, out)

    # ── Phase 1: Compute exact result (N, out) via chunked loop ──
    exact_result = torch.zeros(N, out, dtype=torch.float32, device=device)
    for j0 in range(0, out, chunk_size):
        j1 = min(j0 + chunk_size, out)
        w_q_c = w_q[j0:j1]                                    # (c, nb, bs)
        w_scales_c = w_scales[j0:j1]                           # (c, nb)
        prods = x_q_exp * w_q_c.unsqueeze(0)                   # (N, c, nb, bs)
        block_dots = prods.sum(dim=-1)                         # (N, c, nb)
        combined_scales = x_scales_exp * w_scales_c.unsqueeze(0)  # (N, c, nb)
        exact_result[:, j0:j1] = (block_dots * combined_scales).sum(dim=-1)
        del prods, block_dots, combined_scales

    # ── Phase 2: Per-channel binary search with chunking ──
    lo = torch.full((out,), float(budget_range[0]), dtype=torch.float32, device=device)
    hi = torch.full((out,), float(budget_range[1]), dtype=torch.float32, device=device)

    signal_power = (exact_result**2).mean(dim=0)  # (out,) — constant across iterations

    for _iter in range(12):  # ~12 iterations for budget_range [4, 48]
        mid = torch.floor((lo + hi) / 2.0)

        # Accumulate truncated result for all output channels via chunks
        result_trunc = torch.zeros(N, out, dtype=torch.float32, device=device)

        for j0 in range(0, out, chunk_size):
            j1 = min(j0 + chunk_size, out)
            c = j1 - j0
            w_q_c = w_q[j0:j1]
            w_scales_c = w_scales[j0:j1]
            mid_c = mid[j0:j1]  # (c,)

            # 1. Inter-block delays for this chunk
            log2_w_c = log2_w_full[j0:j1].unsqueeze(0)          # (1, c, nb)
            combined_e = torch.floor(log2_x + log2_w_c)          # (N, c, nb)
            e_max_c = combined_e.amax(dim=-1)                     # (N, c)
            inter_delays_c = e_max_c.unsqueeze(-1) - combined_e   # (N, c, nb)

            # 2. Element-wise products
            prods = x_q_exp * w_q_c.unsqueeze(0)                 # (N, c, nb, bs)

            # 3. Total delay & effective precision
            total_delay = inter_delays_c.unsqueeze(-1) + intra_exp + online_delay  # (N, c, nb, bs)
            b_exp = mid_c.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, c, 1, 1)
            p_eff = torch.clamp(b_exp - total_delay, min=0.0)
            del total_delay

            # 4. Truncate & accumulate
            prods_trunc = _msd_truncate(prods, p_eff)
            del prods, p_eff
            block_dots = prods_trunc.sum(dim=-1)                 # (N, c, nb)
            del prods_trunc
            combined_scales = x_scales_exp * w_scales_c.unsqueeze(0)
            result_trunc[:, j0:j1] = (block_dots * combined_scales).sum(dim=-1)
            del block_dots, combined_scales, inter_delays_c

        # SNR per channel
        noise_power = ((exact_result - result_trunc) ** 2).mean(dim=0) + 1e-30
        snr = 10.0 * torch.log10(signal_power / noise_power)
        del result_trunc

        good = snr >= target_snr_db
        hi = torch.where(good, mid, hi)
        lo = torch.where(good, lo, mid + 1)

    # ── Phase 3: Compute SNR & signal power at the converged budget ──
    budgets = hi

    # One final truncated forward pass at the converged budget to get actual SNR
    result_trunc_final = torch.zeros(N, out, dtype=torch.float32, device=device)
    for j0 in range(0, out, chunk_size):
        j1 = min(j0 + chunk_size, out)
        w_q_c = w_q[j0:j1]
        w_scales_c = w_scales[j0:j1]
        b_c = budgets[j0:j1]  # (c,)

        log2_w_c = log2_w_full[j0:j1].unsqueeze(0)
        combined_e = torch.floor(log2_x + log2_w_c)
        e_max_c = combined_e.amax(dim=-1)
        inter_delays_c = e_max_c.unsqueeze(-1) - combined_e

        prods = x_q_exp * w_q_c.unsqueeze(0)
        total_delay = inter_delays_c.unsqueeze(-1) + intra_exp + online_delay
        b_exp = b_c.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        p_eff = torch.clamp(b_exp - total_delay, min=0.0)
        del total_delay

        prods_trunc = _msd_truncate(prods, p_eff)
        del prods, p_eff
        block_dots = prods_trunc.sum(dim=-1)
        del prods_trunc
        combined_scales = x_scales_exp * w_scales_c.unsqueeze(0)
        result_trunc_final[:, j0:j1] = (block_dots * combined_scales).sum(dim=-1)
        del block_dots, combined_scales, inter_delays_c

    noise_power_final = ((exact_result - result_trunc_final) ** 2).mean(dim=0) + 1e-30
    snr_at_budget = 10.0 * torch.log10(signal_power / noise_power_final)  # (out,)
    signal_power_db = 10.0 * torch.log10(signal_power.clamp(min=1e-30))   # (out,)
    del result_trunc_final, noise_power_final

    # ── Phase 4: Collect statistics at the converged budget ──
    layer_summary, channel_detail = _collect_channel_stats(
        budgets, x_q_exp, x_scales_exp, intra_exp,
        log2_x, log2_w_full, online_delay, chunk_size, out, N,
        snr_at_budget=snr_at_budget,
        signal_power_db=signal_power_db,
        collect_channel_detail=collect_channel_detail,
        budget_range=budget_range,
    )
    return budgets, layer_summary, channel_detail


def _collect_channel_stats(
    budgets, x_q_exp, x_scales_exp, intra_exp,
    log2_x, log2_w_full, online_delay, chunk_size, out, N,
    *,
    snr_at_budget=None,
    signal_power_db=None,
    collect_channel_detail=False,
    budget_range=(4, 48),
):
    """
    Compute layer-wise summary statistics and optionally per-channel detail.

    Runs one chunked pass over the output dimension to collect delay and
    effective-precision statistics without materialising the full (N,out,nb,bs)
    tensor.

    Returns:
        (layer_summary, channel_detail) where:
        - layer_summary: dict of scalar/small-structure summary statistics
        - channel_detail: dict of per-channel lists (empty dict if
          collect_channel_detail is False)
    """
    device = budgets.device

    # Accumulators (on GPU, shape (out,))
    e_combined_sum = torch.zeros(out, dtype=torch.float64, device=device)
    e_combined_sq  = torch.zeros(out, dtype=torch.float64, device=device)
    e_combined_max = torch.full((out,), -1e30, dtype=torch.float32, device=device)
    e_combined_min = torch.full((out,),  1e30, dtype=torch.float32, device=device)
    inter_delay_sum = torch.zeros(out, dtype=torch.float64, device=device)
    intra_delay_mean_global = intra_exp.squeeze(1).mean().item()  # scalar, same for all channels
    eff_prec_sum   = torch.zeros(out, dtype=torch.float64, device=device)
    eff_prec_min   = torch.full((out,),  1e30, dtype=torch.float32, device=device)

    # intra_exp shape: (N, 1, nb, bs)  — nb and bs from its shape
    nb = intra_exp.shape[2]
    bs = intra_exp.shape[3]

    for j0 in range(0, out, chunk_size):
        j1 = min(j0 + chunk_size, out)
        b_c = budgets[j0:j1]  # (c,)

        # Combined log2 scale: (N, c, nb)
        log2_w_c = log2_w_full[j0:j1].unsqueeze(0)       # (1, c, nb)
        combined_e = torch.floor(log2_x + log2_w_c)       # (N, c, nb)
        e_max_c = combined_e.amax(dim=-1)                  # (N, c) — per-(sample, channel)

        # e_combined stats per channel (reduce over N)
        e_combined_sum[j0:j1] = e_max_c.double().sum(dim=0)
        e_combined_sq[j0:j1]  = (e_max_c.double() ** 2).sum(dim=0)
        e_combined_max[j0:j1] = torch.maximum(e_combined_max[j0:j1], e_max_c.amax(dim=0))
        e_combined_min[j0:j1] = torch.minimum(e_combined_min[j0:j1], e_max_c.amin(dim=0))

        # Inter-block delays: (N, c, nb)
        inter_delays_c = e_max_c.unsqueeze(-1) - combined_e
        inter_delay_sum[j0:j1] = inter_delays_c.double().mean(dim=(0, 2))  # mean over N, nb

        # Total delay -> effective precision: (N, c, nb, bs)
        total_delay = inter_delays_c.unsqueeze(-1) + intra_exp + online_delay
        b_exp = b_c.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, c, 1, 1)
        p_eff = torch.clamp(b_exp - total_delay, min=0.0)

        # Effective precision stats per channel (reduce over N, nb, bs)
        eff_prec_sum[j0:j1] = p_eff.double().mean(dim=(0, 2, 3))  # mean over N, nb, bs
        eff_prec_min[j0:j1] = torch.minimum(
            eff_prec_min[j0:j1],
            p_eff.amin(dim=(0, 2, 3)),
        )

        del combined_e, e_max_c, inter_delays_c, total_delay, p_eff

    # ── Finalize per-channel tensors (on GPU) ──
    e_combined_mean = (e_combined_sum / N).float()
    e_combined_var  = (e_combined_sq / N - (e_combined_sum / N) ** 2).clamp(min=0.0)
    e_combined_std  = e_combined_var.sqrt().float()
    inter_delay_mean = inter_delay_sum.float()
    eff_precision_mean = eff_prec_sum.float()

    # Move to CPU for aggregation
    budgets_cpu = budgets.cpu()
    e_combined_mean_cpu = e_combined_mean.cpu()
    e_combined_std_cpu = e_combined_std.cpu()
    e_combined_max_cpu = e_combined_max.cpu()
    e_combined_min_cpu = e_combined_min.cpu()
    inter_delay_mean_cpu = inter_delay_mean.cpu()
    eff_precision_mean_cpu = eff_precision_mean.cpu()
    eff_prec_min_cpu = eff_prec_min.cpu()
    snr_cpu = snr_at_budget.cpu() if snr_at_budget is not None else None
    signal_power_cpu = signal_power_db.cpu() if signal_power_db is not None else None

    # ── Build compact layer summary (always) ──
    b_min, b_max = float(budgets_cpu.min()), float(budgets_cpu.max())
    budget_hist = {}
    for val in budgets_cpu.tolist():
        key = str(int(val))
        budget_hist[key] = budget_hist.get(key, 0) + 1

    sorted_b = budgets_cpu.sort().values
    n_ch = len(sorted_b)

    layer_summary = {
        # Budget distribution
        "budget_mean": round(float(budgets_cpu.mean()), 2),
        "budget_min": b_min,
        "budget_max": b_max,
        "budget_std": round(float(budgets_cpu.std()), 2) if n_ch > 1 else 0.0,
        "budget_p25": float(sorted_b[n_ch // 4]),
        "budget_p50": float(sorted_b[n_ch // 2]),
        "budget_p75": float(sorted_b[3 * n_ch // 4]),
        "budget_histogram": budget_hist,
        "frac_at_min_budget": round(float((budgets_cpu == budget_range[0]).sum()) / n_ch, 4),
        "frac_at_max_budget": round(float((budgets_cpu == budget_range[1]).sum()) / n_ch, 4),
        # SNR validation
        "snr_mean": round(float(snr_cpu.mean()), 2) if snr_cpu is not None else None,
        "snr_min": round(float(snr_cpu.min()), 2) if snr_cpu is not None else None,
        # Combined exponent
        "e_combined_mean": round(float(e_combined_mean_cpu.mean()), 2),
        "e_combined_std": round(float(e_combined_std_cpu.mean()), 2),
        "e_combined_range": [round(float(e_combined_min_cpu.min()), 2),
                             round(float(e_combined_max_cpu.max()), 2)],
        # Delays
        "inter_delay_mean": round(float(inter_delay_mean_cpu.mean()), 4),
        "intra_delay_mean": round(intra_delay_mean_global, 4),
        # Effective precision
        "eff_precision_mean": round(float(eff_precision_mean_cpu.mean()), 2),
        "eff_precision_min": round(float(eff_prec_min_cpu.min()), 2),
        # Signal power
        "signal_power_db_mean": round(float(signal_power_cpu.mean()), 2) if signal_power_cpu is not None else None,
        "signal_power_db_range": [round(float(signal_power_cpu.min()), 2),
                                  round(float(signal_power_cpu.max()), 2)] if signal_power_cpu is not None else None,
    }

    # ── Build per-channel detail (only when requested) ──
    channel_detail = {}
    if collect_channel_detail:
        channel_detail = {
            "budget": budgets_cpu.tolist(),
            "snr_at_budget": snr_cpu.tolist() if snr_cpu is not None else [],
            "e_combined_mean": e_combined_mean_cpu.tolist(),
            "e_combined_std": e_combined_std_cpu.tolist(),
            "inter_delay_mean": inter_delay_mean_cpu.tolist(),
            "eff_precision_mean": eff_precision_mean_cpu.tolist(),
            "signal_power_db": signal_power_cpu.tolist() if signal_power_cpu is not None else [],
        }

    return layer_summary, channel_detail


def calibrate_channel_budgets(
    model,
    tokenizer,
    calibration_texts,
    target_snr_db=30.0,
    max_length=512,
    batch_size=4,
    online_delay=None,
    show_progress=False,
    progress_prefix="",
    detail_layer=2,
):
    """
    Run calibration to determine per-channel MSD cycle budgets.

    Runs forward passes over calibration_texts with exact MX mode,
    hooks into each MXFP linear layer to collect block-level statistics,
    then finds the minimum budget per channel meeting the target SNR.

    All block data is kept on GPU and the budget search is output-chunked
    to cap peak GPU memory at ~2 GiB of intermediates.

    Args:
        model: Qwen3ForCausalLM (or similar) with MXFP layers
        tokenizer: associated tokenizer
        calibration_texts: list of strings for calibration
        target_snr_db: target signal-to-noise ratio in dB (default: 30.0)
        max_length: max token length for calibration inputs
        batch_size: batch size for calibration passes
        online_delay: MSD online delay (default: from config)
        show_progress: whether to show tqdm progress bars
        progress_prefix: prefix string for tqdm descriptions (e.g. rank tag)
        detail_layer: transformer layer index for which full per-channel
            statistics are collected (default: 2).  Layers matching
            ``model.layers.<detail_layer>.`` get channel-wise detail;
            all other layers only get compact layer summaries.

    Side Effects:
        Sets model.config.msd_calibration_data with per-layer per-channel budgets.

    Returns:
        (calibration_result, layer_summaries, channel_details) where:
        - calibration_result: dict[layer_name -> list[float]] per-channel budgets
        - layer_summaries: dict[layer_name -> dict] compact summary per layer
        - channel_details: dict[layer_name -> dict] per-channel detail (only
          for layers matching detail_layer)
    """
    # Avoid circular import at module level
    from .modeling_qwen3 import _MXFPLinearBase

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
        return {}

    logger.info(f"Calibrating {len(mxfp_layers)} MXFP layers over {len(calibration_texts)} samples")

    # Hook to capture inputs and quantized blocks — kept on GPU
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
                # Keep on GPU — the budget search is GPU-accelerated
                layer_data[layer_name].append((x_q, x_scales, w_q, w_scales))
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
            num_batches = (len(calibration_texts) + batch_size - 1) // batch_size
            batch_iter = range(0, len(calibration_texts), batch_size)
            batch_iter = tqdm(
                batch_iter,
                total=num_batches,
                desc=f"{progress_prefix}calib forward" if progress_prefix else "calib forward",
                disable=not show_progress,
            )
            for i in batch_iter:
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

    # Determine which layers get full channel-level detail
    detail_prefix = f"model.layers.{detail_layer}."

    # Process collected data to find per-channel budgets
    calibration_result = {}
    layer_summaries = {}
    channel_details = {}

    layer_iter = list(layer_data.items())
    layer_iter = tqdm(
        layer_iter,
        total=len(layer_data),
        desc=f"{progress_prefix}calib layers" if progress_prefix else "calib layers",
        disable=not show_progress,
    )

    for layer_name, data_list in layer_iter:
        logger.info(f"Computing budgets for {layer_name} ({len(data_list)} batches)")
        # Concatenate all batches (still on GPU)
        all_x_q = torch.cat([d[0] for d in data_list], dim=0)
        all_x_scales = torch.cat([d[1] for d in data_list], dim=0)
        # Weights are the same across batches; take from first
        w_q = data_list[0][2]
        w_scales = data_list[0][3]

        # Free per-batch copies to reclaim GPU memory before budget search
        data_list.clear()

        # Decide whether this layer gets channel-level detail
        want_detail = detail_prefix in layer_name

        # Find budget per channel + stats (GPU, output-chunked)
        budgets, layer_summary, channel_detail = _find_budget_for_snr(
            all_x_q, all_x_scales, w_q, w_scales,
            online_delay, target_snr_db,
            collect_channel_detail=want_detail,
        )

        budget_list = budgets.cpu().tolist()
        calibration_result[layer_name] = budget_list
        layer_summaries[layer_name] = layer_summary
        if channel_detail:
            channel_details[layer_name] = channel_detail

        logger.info(
            f"  {layer_name}: budget range [{min(budget_list):.0f}, {max(budget_list):.0f}], "
            f"mean={budgets.mean():.1f}"
        )

        # Free this layer's block data to keep GPU memory bounded
        del all_x_q, all_x_scales, w_q, w_scales, budgets

    # Store in config
    config.msd_calibration_data = calibration_result
    logger.info("Calibration complete. Results stored in config.msd_calibration_data")
    return calibration_result, layer_summaries, channel_details


def apply_calibration_to_config(config, calibration_data):
    """
    Load calibration data dict into a Qwen3Config.

    Args:
        config: Qwen3Config instance
        calibration_data: dict mapping layer names to lists of per-channel budgets
    """
    config.msd_calibration_data = calibration_data
