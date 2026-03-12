# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen3 model.

This is the **reference / modular** implementation.  MSDComputeContext,
_resolve_channel_budgets, and all MSD/MXFP helpers are kept in sync with
modeling_qwen3.py.  The key difference is that _forward_msd_truncated here
uses a simple non-chunked (N, out, nb, bs) allocation, whereas modeling_qwen3.py
has the production output-chunked version (2 GiB cap) for large models.

Calibration code lives in calibration_msd.py and imports from modeling_qwen3.py.
"""

import math
from collections.abc import Callable

import torch

# need a highly editable version of mxfp8, so we use nn to implement instead of blackbox import
import torch.nn as nn
import torch.nn.functional as F


# Per-format max representable magnitude (OCP MX spec)
_FP8_E4M3_MAX  = 448.0   # FP8  E4M3FN
_FP4_E2M1_MAX  =   6.0   # FP4  E2M1
_FP6_E2M3_MAX  =   7.5   # FP6  E2M3
_FP6_E3M2_MAX  =  28.0   # FP6  E3M2

_HAS_FP8 = hasattr(torch, "float8_e4m3fn")


# ── OCP-spec representable-value grids ───────────────────────────────────────

def _build_fp4_e2m1_grid() -> list[float]:
    """All 16 FP4 E2M1 values (OCP MX spec, bias=1, no ±inf, no NaN)."""
    vals: set[float] = set()
    for bits in range(16):          # 4 bits
        s = (bits >> 3) & 1
        e = (bits >> 1) & 0b11
        m = bits & 1
        if e == 0:                  # subnormal: 0.m × 2^(1-bias)=2^0
            v = m * 0.5
        else:                       # normal: (1+m×0.5) × 2^(e-bias)
            v = (1.0 + m * 0.5) * (2.0 ** (e - 1))
        vals.add(-v if s else v)
    return sorted(vals)


def _build_fp6_e2m3_grid() -> list[float]:
    """All 64 FP6 E2M3 values (OCP MX spec, bias=1, no ±inf, no NaN)."""
    vals: set[float] = set()
    for bits in range(64):          # 6 bits
        s = (bits >> 5) & 1
        e = (bits >> 3) & 0b11
        m = bits & 0b111
        if e == 0:                  # subnormal: 0.mantissa × 2^(1-1)
            v = m / 8.0
        else:                       # normal: (1+m/8) × 2^(e-1)
            v = (1.0 + m / 8.0) * (2.0 ** (e - 1))
        vals.add(-v if s else v)
    return sorted(vals)


def _build_fp6_e3m2_grid() -> list[float]:
    """All 64 FP6 E3M2 values (OCP MX spec, bias=3, no ±inf, no NaN).

    Bit layout: [s | e2 e1 e0 | m1 m0]  (bit-5 = sign, bits 4-2 = exp, bits 1-0 = mant)
    """
    vals: set[float] = set()
    for bits in range(64):          # 6 bits
        s = (bits >> 5) & 1
        e = (bits >> 2) & 0b111    # 3-bit exponent (bits 4:2)
        m = bits & 0b11             # 2-bit mantissa  (bits 1:0)
        if e == 0:                  # subnormal: 0.mm × 2^(1-3)=2^(-2)
            v = (m / 4.0) * 0.25
        else:                       # normal: (1+m/4) × 2^(e-3)
            v = (1.0 + m / 4.0) * (2.0 ** (e - 3))
        vals.add(-v if s else v)
    return sorted(vals)


# Precomputed Python lists (converted to tensors per-device lazily)
# NOTE: must be plain Assign (not AnnAssign) — the modular converter only tracks plain assignments
_FP4_E2M1_GRID_LIST = _build_fp4_e2m1_grid()
_FP6_E2M3_GRID_LIST = _build_fp6_e2m3_grid()
_FP6_E3M2_GRID_LIST = _build_fp6_e3m2_grid()


def _get_grid(grid_list: list[float], device: torch.device) -> torch.Tensor:
    """Return a sorted float32 tensor of representable values on the target device."""
    return torch.tensor(grid_list, dtype=torch.float32, device=device)


def _nearest_on_grid(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Round each element of x to the nearest value in sorted 1-D `grid`.
    Both x and grid must be float32 and on the same device.
    """
    # searchsorted returns the index at which x would be inserted to keep grid sorted
    idx = torch.searchsorted(grid.contiguous(), x.contiguous())
    idx = idx.clamp(0, len(grid) - 1)
    idx_lo = (idx - 1).clamp(0)
    v_hi = grid[idx]
    v_lo = grid[idx_lo]
    # Select whichever representable value is closer
    return torch.where((x - v_lo).abs() <= (x - v_hi).abs(), v_lo, v_hi)

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ..llama.modeling_llama import (
    LlamaAttention,
)
from ..qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2ForQuestionAnswering,
    Qwen2ForSequenceClassification,
    Qwen2ForTokenClassification,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_qwen3 import Qwen3Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen3-8B"


# ── MSD Compute Context ──────────────────────────────────────────────────────

class MSDComputeContext:
    """
    Runtime context for MSD-first time-domain truncated dot-product simulation.

    Carries per-channel cycle budgets, pipeline precision state, and
    activation scale caches through the model forward pass.

    Uses class-level _active to hold the context during a forward pass
    (set by ForCausalLM, read by DecoderLayer).
    """

    _active = None

    def __init__(self, channel_budgets, default_budget, config):
        self.channel_budgets = channel_budgets
        self.default_budget = default_budget
        self.pipeline_precision_remaining = None  # DEPRECATED: kept for backward compat
        self.activation_scales = {}
        self.budget_dynamic_scale = config.msd_budget_dynamic_scale
        self.budget_dynamic_threshold = config.msd_budget_dynamic_threshold
        self.budget_dynamic_mode = config.msd_budget_dynamic_mode
        self.online_delay = config.msd_online_delay
        self.pipeline_precision_loss = config.msd_pipeline_precision_loss
        self.pipeline_budget = getattr(config, "msd_pipeline_budget", 24)
        self.bsd_penetration = getattr(config, "msd_bsd_penetration", False)
        self.silu_pwl_segments = getattr(config, "msd_silu_pwl_segments", 8)

    @staticmethod
    def activate(ctx):
        """Set ctx as the active MSD context for the current forward pass."""
        MSDComputeContext._active = ctx

    @staticmethod
    def deactivate():
        """Clear the active MSD context after the forward pass."""
        MSDComputeContext._active = None

    @staticmethod
    def get_active():
        """Return the currently active MSD context, or None."""
        return MSDComputeContext._active

    @staticmethod
    def create_from_config(config, model):
        """
        Factory: create an MSDComputeContext from config and model.
        Loads per-channel calibration data if available, else uses uniform default.
        Also registers layer_name on each _MXFPLinearBase module.
        """
        channel_budgets = {}
        calibration_data = config.msd_calibration_data
        for name, module in model.named_modules():
            if isinstance(module, _MXFPLinearBase):
                module.layer_name = name
                module._msd_config = config
                if calibration_data and name in calibration_data:
                    budgets_list = calibration_data[name]
                    channel_budgets[name] = torch.tensor(
                        budgets_list, dtype=torch.float32
                    )
                else:
                    channel_budgets[name] = torch.full(
                        (module.out_features,), float(config.msd_cycle_budget),
                        dtype=torch.float32
                    )
        return MSDComputeContext(channel_budgets, config.msd_cycle_budget, config)


# ── Two-level delay computation ──────────────────────────────────────────────

def _safe_log2(x):
    """log2 that maps 0 and negative values to a large negative number."""
    return torch.where(x > 0, torch.log2(x), torch.tensor(-60.0, dtype=x.dtype, device=x.device))


def _compute_inter_block_delays(w_scales, x_scales):
    """
    Compute inter-block delays from shared MX block scales.

    The combined log2 scale for block i in output (n, j) is:
        E_i = floor(log2(w_scales[j, i] * x_scales[n, i]))
    Delay = E_max - E_i for each output element.

    Args:
        w_scales: (out, nb) fp32 shared scales for weight blocks
        x_scales: (N, nb) fp32 shared scales for activation blocks
    Returns:
        inter_block_delays: (N, out, nb) int-valued fp32 tensor
        e_max: (N, out) maximum combined log2 scale
    """
    # Combined log2 scale: (N, 1, nb) + (1, out, nb) -> (N, out, nb)
    log2_x = _safe_log2(x_scales).unsqueeze(1)          # (N, 1, nb)
    log2_w = _safe_log2(w_scales).unsqueeze(0)           # (1, out, nb)
    combined_log2 = log2_x + log2_w                      # (N, out, nb)
    combined_e = torch.floor(combined_log2)              # integer exponents

    e_max = combined_e.amax(dim=-1)                      # (N, out)
    inter_block_delays = e_max.unsqueeze(-1) - combined_e  # (N, out, nb)
    return inter_block_delays, e_max


def _compute_intra_block_delays(x_q_blocks):
    """
    Compute intra-block delays from individual activation element exponents.

    Within each block, elements with smaller exponents start later.
    Weights are pre-aligned to fixed-point offline (no intra-block delay).

    Args:
        x_q_blocks: (N, nb, bs) quantized activation elements (float32)
    Returns:
        intra_block_delays: (N, nb, bs) int-valued fp32 tensor
    """
    abs_vals = x_q_blocks.abs()
    # log2 of absolute value; zeros get large delay
    elem_log2 = torch.where(
        abs_vals > 0,
        torch.floor(torch.log2(abs_vals)),
        torch.tensor(-60.0, dtype=x_q_blocks.dtype, device=x_q_blocks.device)
    )
    # Per-block max exponent
    e_max_block = elem_log2.amax(dim=-1, keepdim=True)   # (N, nb, 1)
    intra_block_delays = e_max_block - elem_log2          # (N, nb, bs)
    return intra_block_delays


# ── MSD truncation primitive (BSD / Non-Adjacent Form) ──────────────────────


def _to_naf_components(x_int):
    """
    Convert integer tensor to Non-Adjacent Form (NAF) positive/negative masks.

    NAF is the unique BSD representation with no adjacent non-zero digits
    and minimum Hamming weight (fewest non-zero digits). It is computed via
    the vectorised identity:
        x_h  = x >> 1
        s    = x + x_h
        naf_pos = s & ~x_h   (positions where digit = +1)
        naf_neg = x_h & ~s   (positions where digit = -1)
    so that  x  =  naf_pos - naf_neg.

    Args:
        x_int: int32 tensor of non-negative integers
    Returns:
        naf_pos: int32 tensor, bitmask of +1 digit positions
        naf_neg: int32 tensor, bitmask of -1 digit positions
    """
    x_h = x_int >> 1
    s = x_int + x_h
    naf_pos = s & (~x_h)
    naf_neg = x_h & (~s)
    return naf_pos, naf_neg


def _naf_digit_width(naf_pos, naf_neg):
    """
    Return the number of digit positions used by an NAF representation.

    The width is 1 + floor(log2(highest set bit in naf_pos | naf_neg)).
    For zero inputs the width is 0.

    Args:
        naf_pos, naf_neg: int32 bitmasks from `_to_naf_components`
    Returns:
        width: int32 tensor
    """
    combined = naf_pos | naf_neg
    # bit_length via float log2 (works up to 2^23 with float32)
    comb_f = combined.float()
    # floor(log2(x)) + 1  gives bit-length;  0 maps to 0
    width = torch.where(
        combined > 0,
        torch.floor(torch.log2(comb_f)).int() + 1,
        torch.zeros_like(combined),
    )
    return width


def _msd_truncate(value, num_digits):
    """
    Truncate each element to its `num_digits` most significant BSD (NAF) digits.

    Models MSD-first computation under a cycle budget: the hardware streams
    digits from MSD to LSD in Binary Signed-Digit (BSD) representation.
    After `num_digits` cycles only those digits survive.

    We use the Non-Adjacent Form (NAF) — the canonical, minimum-weight BSD
    encoding — as the simulation reference. NAF can shift the MSD position
    by +1 compared to plain binary (e.g. 7 = 0b111 in binary but 100(-1) in
    NAF), which makes its truncation behaviour distinct from binary truncation.

    Memory note: NAF conversion creates several same-shape intermediates.  We
    inline the helper calls and `del` every tensor as soon as it is no longer
    needed to minimise peak GPU allocation.  The whole function runs under
    ``torch.no_grad()`` to prevent autograd from retaining the graph.

    Args:
        value: float32 tensor (any shape)
        num_digits: float32 tensor (same shape or broadcastable), effective precision
    Returns:
        truncated: float32 tensor, same shape as value
    """
    with torch.no_grad():
        abs_v = value.abs()
        sign = value.sign()

        # Compute the zero-output mask early; abs_v still needed for scaling below.
        mask = (num_digits > 0) & (abs_v > 0)

        # Scale to integer mantissa: shift so that MSB sits near bit-22/23.
        # We use 2^(23 - msb_pos) so the integer has its MSB at bit 23.
        msb_pos = torch.floor(torch.log2(abs_v.clamp(min=1e-45)))  # float position of MSB
        scale_up = torch.pow(2.0, 23.0 - msb_pos)  # shift to [2^23, 2^24)
        del msb_pos
        x_scaled = torch.round(abs_v * scale_up).to(torch.int32)  # int mantissa
        del abs_v

        # ── Inline _to_naf_components with early frees ──────────────────────
        # Identity: x_h = x >> 1;  s = x + x_h
        #   naf_pos = s & ~x_h   (+1 digit positions)
        #   naf_neg = x_h & ~s   (-1 digit positions)
        x_h = x_scaled >> 1
        s = x_scaled + x_h          # x_scaled still needed for s
        del x_scaled
        naf_pos = s & (~x_h)
        naf_neg = x_h & (~s)
        del x_h, s
        # ────────────────────────────────────────────────────────────────────

        # ── Inline _naf_digit_width with early frees ─────────────────────
        combined = naf_pos | naf_neg
        comb_f = combined.float()
        naf_width = torch.where(
            combined > 0,
            torch.floor(torch.log2(comb_f)).int() + 1,
            torch.zeros_like(combined),
        )
        del combined, comb_f
        # ────────────────────────────────────────────────────────────────────

        # Number of digit positions to ZERO-out from the bottom
        num_digits_i = num_digits.to(torch.int32) if num_digits.is_floating_point() else num_digits
        drop = (naf_width - num_digits_i).clamp(min=0)  # int32
        del naf_width, num_digits_i

        # Build a bit-mask that keeps only the top `num_digits` positions:
        #   mask_out = (1 << drop) - 1   (bits to clear)
        #   keep_mask = ~mask_out
        keep_mask = ~((1 << drop) - 1)  # int32
        del drop

        naf_pos_trunc = naf_pos & keep_mask
        naf_neg_trunc = naf_neg & keep_mask
        del naf_pos, naf_neg, keep_mask

        # Reconstruct float value:  result = sign * (pos - neg) / scale_up
        reconstructed = (naf_pos_trunc.float() - naf_neg_trunc.float()) / scale_up
        del naf_pos_trunc, naf_neg_trunc, scale_up
        result = sign * reconstructed
        del sign, reconstructed

        # Zero out elements with num_digits <= 0 or value == 0
        return torch.where(mask, result, torch.zeros_like(result))


# ── BSD metadata & PWL SiLU for BSD-penetration FFN ─────────────────────────

# SiLU cycle cost (PWL approximation, MSD-first with exponent-based segment detection):
#   Segment detection (0 cycles from BSD stream): uses BSDMetadata.exponent directly,
#     which is available as parallel metadata before the BSD mantissa stream begins.
#   PWL sigmoid eval (3 cycles): online multiply δ=2 (slope × x) + online add δ=1 (+intercept)
#   SiLU multiply x × σ_PWL(x) (2 cycles): online multiplier delay δ=2
_SILU_PWL_LATENCY_CYCLES = 5


class BSDMetadata:
    """
    Per-element metadata for values in BSD (NAF) representation.

    Carries the exponent (MSD position) and mantissa precision separately
    through the FFN pipeline.  This separation is a key design principle:
    exponents propagate as parallel metadata available immediately, while
    BSD mantissa digits flow serially in the MSD-first stream.  Avoiding
    shift-and-align hardware, necessary alignment is offloaded to the
    time domain via inter-block delays in MSD-first online arithmetic.

    For GEMM outputs:
      - exponent = E_max (maximum combined block-scale exponent), known
        from MXFP block scales before any BSD mantissa digit arrives.
      - precision = B_final - δ  (cycle budget minus online startup delay).
        Inter-block delays are NOT subtracted here because they already
        govern per-element product truncation inside the GEMM.

    For element-wise ops (SiLU, gating multiply):
      - exponent is recomputed from the truncated result value.
      - precision is input precision minus the operation's cycle cost.

    When entering a downstream GEMM (down_proj), elements are grouped into
    blocks and the metadata maps to the delay model:
      - block-level max exponent → inter-block alignment delay
      - per-element precision → caps product effective precision

    Attributes:
        exponent: (N, dim) float32 — MSD position per element
        precision: (N, dim) float32 — valid BSD mantissa digits remaining
    """

    __slots__ = ("exponent", "precision")

    def __init__(self, exponent: torch.Tensor, precision: torch.Tensor):
        self.exponent = exponent
        self.precision = precision


def _extract_bsd_metadata(output, b_final, e_max, online_delay):
    """
    Extract per-output-element BSD metadata from an MSD truncated dot-product.

    The exponent is E_max — the maximum combined block-scale exponent per
    (sample, output-channel) pair, known from MXFP block scales before any
    BSD mantissa digit arrives.  This avoids deriving exponent from the
    computed output value, keeping exponent and mantissa metadata separate.

    The precision is B_final - δ (cycle budget minus online startup delay).
    Inter-block delays are NOT subtracted: they already govern per-element
    product truncation inside the GEMM.  Once the first MSD of the
    accumulator output arrives, every subsequent cycle produces one more
    valid digit, up to B_final - δ total.

    Args:
        output:   (N, out) float32 — dot-product result (used for zero-masking)
        b_final:  (N, out) float32 — per-channel cycle budget used
        e_max:    (N, out) float32 — maximum combined block-scale exponent
        online_delay: int — MSD online multiplier delay δ
    Returns:
        BSDMetadata with exponent (N, out) and precision (N, out)
    """
    # Exponent: E_max from block scales (hardware-assigned MSD position).
    # For zero outputs, set exponent to -60 (effectively no signal).
    exponent = torch.where(
        output.abs() > 0,
        e_max,
        torch.tensor(-60.0, dtype=output.dtype, device=output.device),
    )
    # Precision: budget minus online startup delay.
    # Inter-block delays are already accounted for in per-element product
    # truncation (step 6 of the GEMM), so they do not reduce the output
    # stream's digit count.
    precision = torch.clamp(b_final - online_delay, min=0.0)
    return BSDMetadata(exponent, precision)


def _build_pwl_sigmoid_lut(n_segments, device):
    """
    Build a piecewise-linear (PWL) approximation of sigmoid(x) in [-6, 6].

    The approximation uses n_segments linear pieces in [-6, 6] plus two
    saturation tails (σ(x)≈0 for x<-6, σ(x)≈1 for x>6).

    For each segment i covering [x_i, x_{i+1}]:
        σ_approx(x) = slopes[i] * x + intercepts[i]

    The slopes and intercepts are computed by fitting to the exact sigmoid
    at the segment boundaries (linear interpolation).

    Args:
        n_segments: int — number of linear pieces in the transition region
        device: torch.device
    Returns:
        boundaries: (n_segments + 1,) float32 — segment boundary x-values
        slopes:     (n_segments,) float32
        intercepts: (n_segments,) float32
    """
    import math as _math
    # Segment boundaries: uniformly spaced in [-6, 6]
    boundaries = torch.linspace(-6.0, 6.0, n_segments + 1, dtype=torch.float32, device=device)
    # Exact sigmoid at boundaries
    sig_vals = 1.0 / (1.0 + torch.exp(-boundaries))
    # Slopes and intercepts for each segment
    slopes = (sig_vals[1:] - sig_vals[:-1]) / (boundaries[1:] - boundaries[:-1])
    intercepts = sig_vals[:-1] - slopes * boundaries[:-1]
    return boundaries, slopes, intercepts


def _pwl_sigmoid(x, n_segments=8, _cache={}):
    """
    Piecewise-linear sigmoid approximation.

    Models the hardware PWL evaluation:
      1. Segment detection (from BSDMetadata.exponent — available as parallel
         metadata before the BSD mantissa stream begins, 0 cycles cost)
      2. Linear evaluation: slope[i] * x + intercept[i]
         (online multiply δ=2 for slope*x, online add δ=1 for +intercept)

    Saturates to 0 for x < -6, to 1 for x > 6.

    Args:
        x: float32 tensor (any shape)
        n_segments: int — number of PWL segments (default 8)
    Returns:
        float32 tensor, same shape as x
    """
    cache_key = (n_segments, x.device)
    if cache_key not in _cache:
        _cache[cache_key] = _build_pwl_sigmoid_lut(n_segments, x.device)
    boundaries, slopes, intercepts = _cache[cache_key]

    # Determine segment index for each element via searchsorted
    # searchsorted returns position where x would be inserted in boundaries
    # We want the segment index = clamp(searchsorted - 1, 0, n_segments - 1)
    idx = torch.searchsorted(boundaries.contiguous(), x.contiguous().reshape(-1))
    idx = (idx - 1).clamp(0, n_segments - 1)  # segment index

    # Evaluate PWL: slope[idx] * x + intercept[idx]
    x_flat = x.reshape(-1)
    result = slopes[idx] * x_flat + intercepts[idx]

    # Saturate tails
    result = result.clamp(0.0, 1.0)
    return result.view(x.shape)


def _msd_silu_pwl(x, input_bsd, n_segments=8, online_delay=2):
    """
    MSD-first SiLU using PWL sigmoid approximation with BSD metadata propagation.

    Computes SiLU(x) = x * sigmoid_pwl(x), then truncates the result to
    the effective output precision, accounting for the SiLU hardware latency.

    Segment detection uses input_bsd.exponent (available immediately as
    parallel metadata from the upstream GEMM block scales) to identify the
    PWL region.  This costs 0 cycles from the BSD mantissa stream.

    Total SiLU latency = 5 cycles:
      - PWL sigmoid eval: 3 cycles (online multiply δ=2 + online add δ=1)
      - SiLU multiply x × σ(x): 2 cycles (online multiplier δ=2)

    Output precision per element:
        p_out[n,j] = max(0, p_in[n,j] - 5)

    Args:
        x:           (N, dim) float32 — input values (gate_proj output)
        input_bsd:   BSDMetadata — BSD metadata from gate_proj
        n_segments:  int — PWL segment count
        online_delay: int — base online delay δ (used in latency model)
    Returns:
        result:    (N, dim) float32 — SiLU output
        out_bsd:   BSDMetadata — updated BSD metadata
    """
    silu_latency = _SILU_PWL_LATENCY_CYCLES

    # Compute PWL sigmoid * x
    sigmoid_approx = _pwl_sigmoid(x, n_segments=n_segments)
    silu_out = x * sigmoid_approx

    # Output precision: input precision minus SiLU hardware cycles
    p_out = torch.clamp(input_bsd.precision - silu_latency, min=0.0)

    # Truncate to output precision (models digit-limited hardware output)
    result = _msd_truncate(silu_out, p_out)

    # Output exponent: recompute from actual result values
    abs_r = result.abs()
    out_exp = torch.where(
        abs_r > 0,
        torch.floor(torch.log2(abs_r.clamp(min=1e-45))),
        torch.tensor(-60.0, dtype=result.dtype, device=result.device),
    )

    return result, BSDMetadata(out_exp, p_out)


def _msd_gating_mul(silu_val, silu_bsd, up_val, up_bsd, online_delay=2):
    """
    MSD-first element-wise gating multiply: SiLU(gate) * up.

    Both inputs arrive as MSD-first BSD streams. The gating multiply waits
    for both streams, then performs an online multiplication with delay δ.

    The SiLU path is slower by _SILU_PWL_LATENCY_CYCLES cycles (the up_proj
    path's digits have been buffered during this time — no precision loss
    from waiting). The precision loss comes only from the online multiply delay.

    Output precision per element:
        p_gating[n,j] = max(0, min(silu_bsd.precision, up_bsd.precision) - online_delay)

    Args:
        silu_val: (N, dim) float32 — SiLU output values
        silu_bsd: BSDMetadata — SiLU output BSD metadata
        up_val:   (N, dim) float32 — up_proj output values
        up_bsd:   BSDMetadata — up_proj BSD metadata
        online_delay: int — online multiplier delay δ
    Returns:
        result:   (N, dim) float32 — gating product
        out_bsd:  BSDMetadata — output BSD metadata
    """
    # Precision limited by the less precise input, minus online multiply delay
    p_gating = torch.clamp(
        torch.minimum(silu_bsd.precision, up_bsd.precision) - online_delay,
        min=0.0,
    )

    # Compute product and truncate to output precision
    product = silu_val * up_val
    result = _msd_truncate(product, p_gating)

    # Output exponent from actual result
    abs_r = result.abs()
    out_exp = torch.where(
        abs_r > 0,
        torch.floor(torch.log2(abs_r.clamp(min=1e-45))),
        torch.tensor(-60.0, dtype=result.dtype, device=result.device),
    )

    return result, BSDMetadata(out_exp, p_gating)


# ── Deep pipeline helpers (DEPRECATED — kept for backward compat) ────────────

def _msd_silu(x, precision_digits):
    """
    [DEPRECATED] Apply SiLU activation in MSD-simulated mode.

    Superseded by _msd_silu_pwl() which uses PWL sigmoid approximation and
    tracks BSD metadata. Kept for backward compatibility with legacy deep
    pipeline mode (msd_deep_pipeline=True without msd_bsd_penetration=True).
    """
    result = F.silu(x)
    return _msd_truncate(result, precision_digits)


def _msd_elementwise_mul(a, b, precision_digits):
    """
    [DEPRECATED] MSD-simulated element-wise multiply with BSD (NAF) truncation.

    Superseded by _msd_gating_mul() which tracks BSD metadata.
    Kept for backward compatibility.
    """
    result = a * b
    return _msd_truncate(result, precision_digits)

# ── Shared base class ────────────────────────────────────────────────────────

class _MXFPLinearBase(nn.Module):
    """
    Base class for MX-format linear layers (OCP MX spec).

    Block-wise forward pass (common to all formats):
        out[n, j] = sum_b  scale_x[n,b] * scale_w[j,b] * dot(x_q[n,b,:], w_q[j,b,:])

    Subclasses implement `FORMAT_MAX` and `_quantize_elements`, which converts
    a float32 tensor of values normalised into [-FORMAT_MAX, FORMAT_MAX] to the
    target low-precision format (returned as float32).
    """

    FORMAT_MAX: float = 1.0  # override in subclasses

    def __init__(self, in_features: int, out_features: int, bias: bool = True, config=None):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.block_size   = self._get_block_size(config)
        self._msd_config  = config   # stored for MSD truncation parameters
        self.layer_name   = None     # set by _create_msd_context after model init
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_param", None)

    def _get_block_size(self, config) -> int:
        """Resolved by subclasses; fall back to 32."""
        return 32

    def _quantize_elements(self, x_normalized: torch.Tensor) -> torch.Tensor:
        """
        Quantize float32 values already normalised into [-FORMAT_MAX, FORMAT_MAX]
        to the target low-precision format.  Return float32.
        """
        raise NotImplementedError

    def _quantize_to_blocks(
        self, x_blocks: torch.Tensor, eps: float = 1e-30
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-block fp32 shared scales and quantize elements.

        Args:
            x_blocks: float32  (..., num_blocks, block_size)
        Returns:
            x_q:    same shape, float32 values rounded to low-precision format
            scales: (..., num_blocks) fp32 shared scales
        """
        # 1. fp32 shared scale = max_abs / FORMAT_MAX  (one per block)
        max_abs = x_blocks.abs().amax(dim=-1)                            # (..., nb)
        scales  = torch.clamp(max_abs / self.FORMAT_MAX, min=eps)        # fp32 scale
        # 2. Normalise, quantize elements, return
        x_normalized = x_blocks / scales.unsqueeze(-1)                   # (..., nb, bs)
        x_q = self._quantize_elements(x_normalized)                      # (..., nb, bs)
        return x_q, scales

    def _prepare_blocks(
        self, tensor: torch.Tensor, rows: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Pad `tensor` to a multiple of block_size in the last dimension,
        reshape into blocks, and quantize.
            tensor : (rows, in_like)  float32
            returns: q_blocks  (rows, num_blocks, block_size)  float32
                     scales    (rows, num_blocks)               fp32
                     pad_len   int
        """
        feat = tensor.shape[-1]
        pad_len = (-feat) % self.block_size
        if pad_len:
            tensor = F.pad(tensor, (0, pad_len))
        num_blocks = tensor.shape[-1] // self.block_size
        blocks = tensor.view(rows, num_blocks, self.block_size)
        q, scales = self._quantize_to_blocks(blocks)
        return q, scales, pad_len

    def _resolve_channel_budgets(self, compute_context, x_scales, w_scales, N):
        """
        Resolve per-output-channel cycle budgets using hybrid Glocal budgeting
        with combined activation + weight scales.

        The dynamic budget adjustment uses the maximum combined log2 scale
        per (sample, output-channel) pair, which better predicts the magnitude
        of the channel's dot-product result than activation scale alone.

        Args:
            compute_context: MSDComputeContext or None
            x_scales: (N, nb) activation shared scales
            w_scales: (out, nb) weight shared scales
            N: batch size
        Returns:
            b_final: (N, out) per-sample, per-output-channel final budget
        """
        cfg = self._msd_config
        # Base budget per output channel
        if compute_context is not None and self.layer_name in compute_context.channel_budgets:
            b_base = compute_context.channel_budgets[self.layer_name].to(x_scales.device)  # (out,)
        else:
            b_base = torch.full(
                (self.out_features,), float(cfg.msd_cycle_budget),
                dtype=torch.float32, device=x_scales.device
            )
        # Combined scale exponent: max across blocks of floor(log2(x_scale * w_scale))
        # This gives a per-(sample, output-channel) measure of the channel result magnitude.
        log2_x = _safe_log2(x_scales)   # (N, nb)
        log2_w = _safe_log2(w_scales)   # (out, nb)
        combined_log2 = log2_x.unsqueeze(1) + log2_w.unsqueeze(0)  # (N, out, nb)
        e_combined = torch.floor(combined_log2).amax(dim=-1)  # (N, out)
        del combined_log2  # free (N, out, nb) immediately

        threshold = cfg.msd_budget_dynamic_threshold
        alpha = cfg.msd_budget_dynamic_scale
        mode = cfg.msd_budget_dynamic_mode

        if mode == "step":
            delta_b = torch.where(
                e_combined > threshold,
                torch.tensor(alpha, dtype=torch.float32, device=x_scales.device),
                torch.tensor(0.0, dtype=torch.float32, device=x_scales.device),
            )  # (N, out)
        else:  # "linear" (default)
            delta_b = alpha * torch.clamp(e_combined - threshold, min=0.0)  # (N, out)

        # b_final: (N, out) = (1, out) + (N, out)
        b_final = b_base.unsqueeze(0) + delta_b
        return b_final

    def _forward_msd_truncated(
        self, x_q, x_scales, w_q, w_scales, N, compute_context,
        return_bsd_metadata=False,
    ):
        """
        MSD-first truncated dot-product simulation.

        Computes the dot-product with two-level time-domain delays
        (inter-block from shared scales, intra-block from element exponents)
        and cycle-budget-based early termination.

        Args:
            x_q:      (N, nb, bs) quantized activation blocks (float32)
            x_scales: (N, nb) activation block shared scales
            w_q:      (out, nb, bs) quantized weight blocks (float32)
            w_scales: (out, nb) weight block shared scales
            N:        batch dimension
            compute_context: MSDComputeContext or None
            return_bsd_metadata: bool — if True, also return BSDMetadata
        Returns:
            result: (N, out) float32
            [bsd_meta]: BSDMetadata (only if return_bsd_metadata=True)
        """
        cfg = self._msd_config
        nb = x_q.shape[1]
        bs = x_q.shape[2]
        out = w_q.shape[0]
        online_delay = cfg.msd_online_delay

        # 1. Inter-block delays from shared scales: (N, out, nb)
        # e_max: (N, out) maximum combined block-scale exponent per channel
        inter_delays, e_max = _compute_inter_block_delays(w_scales, x_scales)

        # 2. Intra-block delays from activation element exponents: (N, nb, bs)
        intra_delays = _compute_intra_block_delays(x_q)

        # 3. Per-channel budget: (N, out) — uses combined activation + weight scales
        b_final = self._resolve_channel_budgets(compute_context, x_scales, w_scales, N)

        # 4. Element-wise products: (N, out, nb, bs)
        # x_q: (N, nb, bs) -> (N, 1, nb, bs);  w_q: (out, nb, bs) -> (1, out, nb, bs)
        prods = x_q.unsqueeze(1) * w_q.unsqueeze(0)  # (N, out, nb, bs)

        # 5. Total delay per element: (N, out, nb, bs)
        # inter_delays: (N, out, nb) -> (N, out, nb, 1)
        # intra_delays: (N, nb, bs) -> (N, 1, nb, bs)
        total_delay = (
            inter_delays.unsqueeze(-1)
            + intra_delays.unsqueeze(1)
            + online_delay
        )  # (N, out, nb, bs)

        # 6. Effective precision: (N, out, nb, bs)
        # b_final: (N, out) -> (N, out, 1, 1)
        p_eff = b_final.unsqueeze(-1).unsqueeze(-1) - total_delay
        p_eff = torch.clamp(p_eff, min=0.0)

        # 7. Truncate each element-wise product to its effective precision
        prods_trunc = _msd_truncate(prods, p_eff)

        # 8. Sum within blocks: (N, out, nb)
        block_dots = prods_trunc.sum(dim=-1)

        # 9. Apply shared scales: combined_scales (N, out, nb)
        # x_scales: (N, nb) -> (N, 1, nb);  w_scales: (out, nb) -> (1, out, nb)
        combined_scales = x_scales.unsqueeze(1) * w_scales.unsqueeze(0)  # (N, out, nb)

        # 10. Final summation across blocks
        result = (block_dots * combined_scales).sum(dim=-1)  # (N, out)

        if return_bsd_metadata:
            bsd_meta = _extract_bsd_metadata(result, b_final, e_max, online_delay)
            return result, bsd_meta
        return result

    def _forward_msd_truncated_bsd_input(
        self, x, x_bsd, w_q, w_scales, N, compute_context,
    ):
        """
        MSD-first truncated dot-product with BSD-precision-limited input.

        Variant of _forward_msd_truncated for down_proj in BSD-penetration mode.
        The input `x` retains its BSD-limited precision from pipeline stages
        (SiLU, gating multiply) — it is NOT re-quantized to the MXFP format.
        Instead, block scales are computed from actual values but elements keep
        their precision-limited representation.

        The effective precision of each product element is capped by both the
        cycle budget and the input element's remaining BSD precision:
            p_eff[n,j,b,k] = min(
                B[n,j] - inter_delay[n,j,b] - intra_delay[n,b,k] - δ,
                bsd_precision[n, b*bs+k]
            )

        Args:
            x:      (N, dim) float32 — input values (gating output, BSD-limited)
            x_bsd:  BSDMetadata — per-element BSD metadata from gating
            w_q:    (out, nb, bs) quantized weight blocks (float32)
            w_scales: (out, nb) weight block shared scales
            N:      batch dimension
            compute_context: MSDComputeContext or None
        Returns:
            result: (N, out) float32
        """
        cfg = self._msd_config
        bs = self.block_size
        online_delay = cfg.msd_online_delay

        # Reshape input into blocks WITHOUT MXFP re-quantization.
        # We compute block scales from actual values (needed for inter-block delays)
        # but keep the original values (they already carry BSD precision limits).
        feat = x.shape[-1]
        pad_len = (-feat) % bs
        x_padded = F.pad(x, (0, pad_len)) if pad_len else x
        nb = x_padded.shape[-1] // bs
        x_blocks = x_padded.view(N, nb, bs)  # (N, nb, bs)

        # Compute block scales from actual input magnitudes (for inter-block alignment)
        x_scales = x_blocks.abs().amax(dim=-1).clamp(min=1e-30)  # (N, nb)

        # Pad precision tensor to match
        if pad_len:
            precision_padded = F.pad(x_bsd.precision, (0, pad_len))
        else:
            precision_padded = x_bsd.precision
        precision_blocks = precision_padded.view(N, nb, bs)  # (N, nb, bs)

        out = w_q.shape[0]

        # 1. Inter-block delays: (N, out, nb)
        inter_delays, _ = _compute_inter_block_delays(w_scales, x_scales)

        # 2. Intra-block delays from input element exponents: (N, nb, bs)
        intra_delays = _compute_intra_block_delays(x_blocks)

        # 3. Per-channel budget: (N, out)
        b_final = self._resolve_channel_budgets(compute_context, x_scales, w_scales, N)

        # 4. Element-wise products: (N, out, nb, bs)
        prods = x_blocks.unsqueeze(1) * w_q.unsqueeze(0)

        # 5. Total delay: (N, out, nb, bs)
        total_delay = (
            inter_delays.unsqueeze(-1)
            + intra_delays.unsqueeze(1)
            + online_delay
        )

        # 6. Effective precision, capped by input BSD precision
        # Standard budget-limited precision
        p_eff_budget = b_final.unsqueeze(-1).unsqueeze(-1) - total_delay
        p_eff_budget = torch.clamp(p_eff_budget, min=0.0)

        # Input precision cap: (N, nb, bs) -> (N, 1, nb, bs)
        p_eff_input = precision_blocks.unsqueeze(1)

        # Final effective precision: minimum of budget and input precision
        p_eff = torch.minimum(p_eff_budget, p_eff_input)

        # 7. Truncate products
        prods_trunc = _msd_truncate(prods, p_eff)

        # 8. Sum within blocks: (N, out, nb)
        block_dots = prods_trunc.sum(dim=-1)

        # 9. Apply shared scales
        combined_scales = x_scales.unsqueeze(1) * w_scales.unsqueeze(0)

        # 10. Final summation
        result = (block_dots * combined_scales).sum(dim=-1)  # (N, out)
        return result

    def forward(self, x: torch.Tensor, compute_context=None,
                return_bsd_metadata=False, input_bsd=None) -> torch.Tensor:
        """
        MX-format block-wise forward pass.

        For each block b along in_features:
            out[n, j] += scale_x[n, b] * scale_w[j, b] * dot(x_q[n, b, :], w_q[j, b, :])

        When use_msd_truncation is enabled, uses time-domain truncated dot-product
        with two-level delays and cycle budgeting.

        Args:
            x: input tensor
            compute_context: MSDComputeContext or None
            return_bsd_metadata: bool — if True, also return BSDMetadata for output
            input_bsd: BSDMetadata or None — if provided, use BSD-input path
                       (skip MXFP re-quantization of input, cap precision by input BSD)
        """
        orig_dtype  = x.dtype
        batch_shape = x.shape[:-1]
        N = math.prod(batch_shape) if batch_shape else 1

        x_2d = x.float().reshape(N, self.in_features)   # (N, in)
        w_2d = self.weight.float()                       # (out, in)

        # Always prepare weight blocks (needed in all paths)
        w_q, w_scales, _ = self._prepare_blocks(w_2d, self.out_features)  # (out, nb, bs), (out, nb)

        use_msd = getattr(self._msd_config, "use_msd_truncation", False) if self._msd_config else False

        if use_msd and input_bsd is not None:
            # BSD-input path: skip MXFP re-quantization of input
            result = self._forward_msd_truncated_bsd_input(
                x_2d, input_bsd, w_q, w_scales, N, compute_context,
            )
        elif use_msd:
            # Standard MSD-first truncated dot-product path
            x_q, x_scales, _ = self._prepare_blocks(x_2d, N)
            result = self._forward_msd_truncated(
                x_q, x_scales, w_q, w_scales, N, compute_context,
                return_bsd_metadata=return_bsd_metadata,
            )
            if return_bsd_metadata:
                result, bsd_meta = result
        else:
            # Standard exact MX block-wise matmul path
            x_q, x_scales, _ = self._prepare_blocks(x_2d, N)
            elem_dots = torch.bmm(
                x_q.permute(1, 0, 2),   # (nb, N,  bs)
                w_q.permute(1, 2, 0),   # (nb, bs, out)
            )                           # (nb, N, out)

            combined_scales = x_scales.t().unsqueeze(-1) * w_scales.t().unsqueeze(1)

            result = (elem_dots * combined_scales).sum(dim=0)  # (N, out)

        if self.bias_param is not None:
            result = result + self.bias_param

        out = result.view(*batch_shape, self.out_features).to(orig_dtype)

        if return_bsd_metadata and use_msd and input_bsd is None:
            return out, bsd_meta
        return out


# ── Format-specific subclasses ────────────────────────────────────────────────

class MXFP8Linear(_MXFPLinearBase):
    """
    MX FP8 linear layer (FP8 E4M3FN, OCP MX spec).
    Uses native torch.float8_e4m3fn for element quantization.
    """

    FORMAT_MAX = _FP8_E4M3_MAX  # 448.0

    def __init__(self, in_features: int, out_features: int, bias: bool = True, config=None):
        if not _HAS_FP8:
            raise RuntimeError("MXFP8Linear requires PyTorch >= 2.1 with float8_e4m3fn support.")
        super().__init__(in_features, out_features, bias, config)

    def _get_block_size(self, config) -> int:
        return config.mxfp8_block_size if config else 32



    def _quantize_elements(self, x_normalized: torch.Tensor) -> torch.Tensor:
        # Native fp8 cast performs the rounding; cast back to float32 for arithmetic
        return x_normalized.to(torch.float8_e4m3fn).float()


class MXFP4Linear(_MXFPLinearBase):
    """
    MX FP4 linear layer (FP4 E2M1, OCP MX spec).

    All 16 FP4 E2M1 representable values (bias=1, no ±inf/NaN):
        positive: 0, 0.5, 1, 1.5, 2, 3, 4, 6
        max = 6.0

    Element quantization via nearest-neighbor lookup on the representable grid.
    """

    FORMAT_MAX = _FP4_E2M1_MAX  # 6.0

    def _get_block_size(self, config) -> int:
        return config.mxfp4_block_size if config else 32

    def _quantize_elements(self, x_normalized: torch.Tensor) -> torch.Tensor:
        grid = _get_grid(_FP4_E2M1_GRID_LIST, x_normalized.device)
        flat = x_normalized.reshape(-1)
        return _nearest_on_grid(flat, grid).view(x_normalized.shape)


class MXFP6Linear(_MXFPLinearBase):
    """
    MX FP6 linear layer, supporting two OCP MX FP6 variants:

    * ``"e2m3"``  (default)  1-sign + 2-exp + 3-mant, bias=1, max=7.5
    * ``"e3m2"``             1-sign + 3-exp + 2-mant, bias=3, max=28.0

    Element quantization via nearest-neighbor lookup on the representable grid.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config=None,
        fp6_format: str = "e2m3",
    ):
        self.fp6_format = (getattr(config, "mxfp6_format", None) or fp6_format).lower()
        if self.fp6_format not in ("e2m3", "e3m2"):
            raise ValueError(f"fp6_format must be 'e2m3' or 'e3m2', got '{self.fp6_format}'")
        super().__init__(in_features, out_features, bias, config)

    @property
    def FORMAT_MAX(self) -> float:  # type: ignore[override]
        return _FP6_E2M3_MAX if self.fp6_format == "e2m3" else _FP6_E3M2_MAX

    def _get_block_size(self, config) -> int:
        return config.mxfp6_block_size if config else 32

    def _quantize_elements(self, x_normalized: torch.Tensor) -> torch.Tensor:
        grid_list = _FP6_E2M3_GRID_LIST if self.fp6_format == "e2m3" else _FP6_E3M2_GRID_LIST
        grid = _get_grid(grid_list, x_normalized.device)
        flat = x_normalized.reshape(-1)
        return _nearest_on_grid(flat, grid).view(x_normalized.shape)

class Qwen3RMSNorm(Qwen2RMSNorm):
    pass

# Mapping from config flag to linear class (evaluated lazily so subclasses are always defined)
_MXFP_LINEAR_REGISTRY = {
    "use_mxfp8": MXFP8Linear,
    "use_mxfp4": MXFP4Linear,
    "use_mxfp6": MXFP6Linear,
}


def _make_linear(in_f: int, out_f: int, config) -> nn.Module:
    """
    Return the appropriate linear layer class based on config flags.
    Priority: mxfp8 > mxfp6 > mxfp4 > standard.
    At most one format flag should be True; the first truthy one wins.
    """
    for flag, cls in _MXFP_LINEAR_REGISTRY.items():
        if getattr(config, flag, False):
            return cls(in_f, out_f, bias=False, config=config)
    return nn.Linear(in_f, out_f, bias=False)


# explicitly define MLP to implement mxfp quantization and future MSB-first bit-serial
# computation, instead of using GemmaMLP which is a blackbox import
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = _make_linear(self.hidden_size, self.intermediate_size, config)
        self.up_proj   = _make_linear(self.hidden_size, self.intermediate_size, config)
        self.down_proj = _make_linear(self.intermediate_size, self.hidden_size, config)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, compute_context=None):
        use_msd = getattr(self.config, "use_msd_truncation", False)
        use_bsd_pen = getattr(self.config, "msd_bsd_penetration", False)
        use_pipeline = getattr(self.config, "msd_deep_pipeline", False)

        if use_msd and (use_bsd_pen or use_pipeline) and compute_context is not None:
            cfg = self.config
            online_delay = cfg.msd_online_delay
            n_segments = getattr(cfg, "msd_silu_pwl_segments", 8)

            if use_pipeline:
                # ── Mode 2: Deep pipeline ────────────────────────────────
                # Unified pipeline budget governs gate_proj→SiLU→gating as
                # a single time-domain unit. down_proj gets independent budget.
                #
                # Cycle allocation:
                #   T_gemm = B_pipe - silu_latency - online_delay
                #   gate_proj and up_proj each receive T_gemm as cycle budget
                #   SiLU consumes silu_latency from gate_proj output precision
                #   Gating multiply consumes online_delay
                silu_latency = _SILU_PWL_LATENCY_CYCLES
                pipeline_budget = getattr(cfg, "msd_pipeline_budget", 24)

                # Effective GEMM budget: pipeline budget minus SiLU and gating overhead
                t_gemm = max(1, pipeline_budget - silu_latency - online_delay)

                # Temporarily override cycle budget for gate/up projections
                saved_budget = cfg.msd_cycle_budget
                cfg.msd_cycle_budget = t_gemm
                # Invalidate cached channel budgets for this layer's gate/up projections
                if hasattr(compute_context, '_pipeline_budget_override'):
                    compute_context._pipeline_budget_override = t_gemm

                try:
                    # Stage 1: gate_proj and up_proj with pipeline-derived budget
                    gate_out, gate_bsd = self.gate_proj(
                        x, compute_context=compute_context, return_bsd_metadata=True,
                    )
                    up_out, up_bsd = self.up_proj(
                        x, compute_context=compute_context, return_bsd_metadata=True,
                    )
                finally:
                    cfg.msd_cycle_budget = saved_budget

                # Stage 2: PWL SiLU on gate output
                silu_out, silu_bsd = _msd_silu_pwl(
                    gate_out.float().reshape(-1, gate_out.shape[-1]),
                    gate_bsd, n_segments=n_segments, online_delay=online_delay,
                )

                # Stage 3: Gating multiply
                up_flat = up_out.float().reshape(-1, up_out.shape[-1])
                gating_out, gating_bsd = _msd_gating_mul(
                    silu_out, silu_bsd, up_flat, up_bsd,
                    online_delay=online_delay,
                )

                # Stage 4: down_proj with independent budget and BSD-precision input
                result = self.down_proj(
                    gating_out.view(x.shape[:-1] + (gating_out.shape[-1],)).to(x.dtype),
                    compute_context=compute_context,
                    input_bsd=gating_bsd,
                )
                return result

            else:
                # ── Mode 1: BSD penetration (no deep pipeline) ───────────
                # Each step has its own independent budget, but data stays in
                # BSD representation (no MXFP re-quantization between steps).
                # SiLU uses PWL approximation with proper cycle cost modeling.

                # Stage 1: gate_proj and up_proj with standard MSD budgets
                gate_out, gate_bsd = self.gate_proj(
                    x, compute_context=compute_context, return_bsd_metadata=True,
                )
                up_out, up_bsd = self.up_proj(
                    x, compute_context=compute_context, return_bsd_metadata=True,
                )

                # Stage 2: PWL SiLU on gate output
                silu_out, silu_bsd = _msd_silu_pwl(
                    gate_out.float().reshape(-1, gate_out.shape[-1]),
                    gate_bsd, n_segments=n_segments, online_delay=online_delay,
                )

                # Stage 3: Gating multiply
                up_flat = up_out.float().reshape(-1, up_out.shape[-1])
                gating_out, gating_bsd = _msd_gating_mul(
                    silu_out, silu_bsd, up_flat, up_bsd,
                    online_delay=online_delay,
                )

                # Stage 4: down_proj with BSD-precision input
                result = self.down_proj(
                    gating_out.view(x.shape[:-1] + (gating_out.shape[-1],)).to(x.dtype),
                    compute_context=compute_context,
                    input_bsd=gating_bsd,
                )
                return result

        elif use_msd:
            # MSD truncation without BSD penetration: standard structure, pass context
            # (per-GEMM MSD truncation, exact fp16 SiLU and gating between GEMMs)
            gate_out = self.gate_proj(x, compute_context=compute_context)
            up_out = self.up_proj(x, compute_context=compute_context)
            intermediate = self.act_fn(gate_out) * up_out
            return self.down_proj(intermediate, compute_context=compute_context)
        else:
            # Standard forward pass
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            return down_proj


class Qwen3RotaryEmbedding(Qwen2RotaryEmbedding):
    pass

class Qwen3Attention(LlamaAttention):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        super().__init__(config, layer_idx)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(Qwen2DecoderLayer):
    """
    Qwen3 decoder layer. Overrides Qwen2DecoderLayer to thread compute_context
    through to the MLP for MSD-first simulation.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Get compute_context from module-level holder (set by ForCausalLM)
        compute_context = MSDComputeContext.get_active()

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention (no MSD context needed for attention projections)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected (thread compute_context to MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, compute_context=compute_context)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3ForCausalLM(Qwen2ForCausalLM):
    _msd_context = None
    _msd_context_config_hash = None

    def _get_msd_context(self):
        """Return cached MSDComputeContext, creating or refreshing only when config changes."""
        cfg = self.config
        # Simple hash of MSD-relevant config fields to detect changes
        cfg_hash = (
            getattr(cfg, "use_msd_truncation", False),
            getattr(cfg, "use_mxfp8", False),
            getattr(cfg, "use_mxfp6", False),
            getattr(cfg, "use_mxfp4", False),
            getattr(cfg, "msd_cycle_budget", 16),
            getattr(cfg, "msd_online_delay", 2),
            getattr(cfg, "msd_budget_dynamic_scale", 1.0),
            getattr(cfg, "msd_budget_dynamic_threshold", 0.0),
            getattr(cfg, "msd_budget_dynamic_mode", "linear"),
            getattr(cfg, "msd_deep_pipeline", False),
            getattr(cfg, "msd_bsd_penetration", False),
            getattr(cfg, "msd_pipeline_budget", 24),
            getattr(cfg, "msd_silu_pwl_segments", 8),
            getattr(cfg, "msd_pipeline_precision_loss", 2),
            id(getattr(cfg, "msd_calibration_data", None)),
        )
        if self._msd_context is None or self._msd_context_config_hash != cfg_hash:
            self._msd_context = MSDComputeContext.create_from_config(cfg, self.model)
            self._msd_context_config_hash = cfg_hash
        return self._msd_context

    def forward(
        self,
        **super_kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        # Activate cached MSD compute context if MSD truncation is active
        use_msd = getattr(self.config, "use_msd_truncation", False)
        has_mxfp = getattr(self.config, "use_mxfp8", False) or getattr(self.config, "use_mxfp6", False) or getattr(self.config, "use_mxfp4", False)
        if use_msd and has_mxfp:
            MSDComputeContext.activate(self._get_msd_context())
        try:
            return super().forward(**super_kwargs)
        finally:
            MSDComputeContext.deactivate()


class Qwen3ForSequenceClassification(Qwen2ForSequenceClassification):
    pass


class Qwen3ForTokenClassification(Qwen2ForTokenClassification):
    pass


class Qwen3ForQuestionAnswering(Qwen2ForQuestionAnswering):
    pass


__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering",
    "Qwen3PreTrainedModel",  # noqa: F822
    "Qwen3Model",  # noqa: F822
    "Qwen3DecoderLayer",  # noqa: F822
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
]
