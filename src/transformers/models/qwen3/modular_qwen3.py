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
"""PyTorch Qwen3 model."""

from collections.abc import Callable

import math

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

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ..gemma.modeling_gemma import GemmaMLP
from ..llama.modeling_llama import (
    LlamaAttention,
)
from ..qwen2.modeling_qwen2 import (
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

from ...activations import ACT2FN


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen3-8B"

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

    def forward(self, x: torch.Tensor, compute_context=None) -> torch.Tensor:
        """
        MX-format block-wise forward pass.

        For each block b along in_features:
            out[n, j] += scale_x[n, b] * scale_w[j, b] * dot(x_q[n, b, :], w_q[j, b, :])

        Scales are fp32; element dot-products use low-precision-simulated float32.
        """
        orig_dtype  = x.dtype
        batch_shape = x.shape[:-1]
        N = math.prod(batch_shape) if batch_shape else 1

        x_2d = x.float().reshape(N, self.in_features)   # (N, in)
        w_2d = self.weight.float()                       # (out, in)

        x_q, x_scales, _ = self._prepare_blocks(x_2d, N)                # (N,   nb, bs), (N,   nb)
        w_q, w_scales, _ = self._prepare_blocks(w_2d, self.out_features) # (out, nb, bs), (out, nb)

        # Batched block-wise matmul:
        #   x_q: (N,   nb, bs) -> (nb, N,  bs)
        #   w_q: (out, nb, bs) -> (nb, bs, out)
        #   bmm -> (nb, N, out)
        elem_dots = torch.bmm(
            x_q.permute(1, 0, 2),   # (nb, N,  bs)
            w_q.permute(1, 2, 0),   # (nb, bs, out)
        )                           # (nb, N, out)

        # Combined fp32 scale product: (nb, N, 1) * (nb, 1, out) -> (nb, N, out)
        combined_scales = x_scales.t().unsqueeze(-1) * w_scales.t().unsqueeze(1)

        # Placeholder for future MSB-first bit-serial computation implementation

        result = (elem_dots * combined_scales).sum(dim=0)  # (N, out)

        if self.bias_param is not None:
            result = result + self.bias_param

        return result.view(*batch_shape, self.out_features).to(orig_dtype)


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

    def forward(self, x):
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


class Qwen3ForCausalLM(Qwen2ForCausalLM):
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
        return super().forward(**super_kwargs)


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
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
]
