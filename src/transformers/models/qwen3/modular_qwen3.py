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

# FP8 E4M3FN: max representable magnitude
_FP8_E4M3_MAX = 448.0
_HAS_FP8 = hasattr(torch, "float8_e4m3fn")

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

# Create a new class to replace nn.Linear which simulates mxfp8 and exposes bit-level computation.
class MXFP8Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear that simulates MX FP8 (OCP MX spec) computation:
      - Each contiguous block of `block_size` elements along the in_features dimension
        shares a single fp32 scale (derived from the block max-abs).
      - Elements within each block are quantized to FP8 E4M3FN.
      - The matmul is computed block-wise:
            out = sum_b  scale_x[b] * scale_w[b]  *  dot(x_fp8[b], w_fp8[b])
        so the shared scales and element values are kept separate throughout.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, config=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = config.mxfp8_block_size if config else 32
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_param", None)
        if not _HAS_FP8:
            raise RuntimeError("MXFP8Linear requires PyTorch >= 2.1 with float8_e4m3fn support.")

    @staticmethod
    def _quantize_fp8(
        x_blocks: torch.Tensor, eps: float = 1e-30
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a block-partitioned tensor to FP8 E4M3FN.

        Args:
            x_blocks: float32 tensor of shape (..., num_blocks, block_size)
            eps: minimum scale to avoid division by zero

        Returns:
            x_fp8:  same shape as x_blocks, float32 values rounded to fp8 precision
            scales: float32 tensor of shape (..., num_blocks) — one shared scale per block
        """
        # 1. Compute per-block max-abs -> fp32 shared scale
        max_abs = x_blocks.abs().amax(dim=-1)  # (..., num_blocks)
        scales = torch.clamp(max_abs / _FP8_E4M3_MAX, min=eps)  # fp32 scale

        # 2. Normalise into fp8 representable range, then cast to fp8 and back
        x_scaled = x_blocks / scales.unsqueeze(-1)  # (..., num_blocks, block_size)
        x_fp8 = x_scaled.to(torch.float8_e4m3fn).float()  # simulate fp8 rounding

        return x_fp8, scales

    def _prepare_blocks(
        self, tensor: torch.Tensor, rows: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Pad tensor's last dimension to a multiple of block_size, reshape into blocks,
        and return (fp8_blocks, scales, pad_len).

            tensor: (rows, in_like)   — must be contiguous float32
            returns fp8: (rows, num_blocks, block_size)
                    scales: (rows, num_blocks)        fp32
        """
        feat = tensor.shape[-1]
        pad_len = (-feat) % self.block_size
        if pad_len:
            tensor = F.pad(tensor, (0, pad_len))
        num_blocks = tensor.shape[-1] // self.block_size
        blocks = tensor.view(rows, num_blocks, self.block_size)
        fp8, scales = self._quantize_fp8(blocks)
        return fp8, scales, pad_len

    def forward(self, x: torch.Tensor, compute_context=None) -> torch.Tensor:
        """
        MX FP8 forward pass.

        For each block b along in_features:
            out[n, j] += scale_x[n, b] * scale_w[j, b] * dot(x_fp8[n, b, :], w_fp8[j, b, :])

        Scales are fp32; element products are performed in fp8-simulated float32.
        """
        orig_dtype = x.dtype
        batch_shape = x.shape[:-1]
        N = math.prod(batch_shape) if batch_shape else 1

        x_2d = x.float().reshape(N, self.in_features)    # (N, in_features)
        w_2d = self.weight.float()                        # (out_features, in_features)

        # Quantise: fp8 elements + fp32 shared scales, one scale per block
        x_fp8, x_scales, _ = self._prepare_blocks(x_2d, N)               # (N,   nb, bs), (N,   nb)
        w_fp8, w_scales, _ = self._prepare_blocks(w_2d, self.out_features) # (out, nb, bs), (out, nb)

        num_blocks = x_fp8.shape[1]

        # Block-wise accumulation:
        #   elem_dot  = x_fp8[:, b, :] @ w_fp8[:, b, :].T   shape (N, out)
        #   blk_scale = x_scales[:, b].unsqueeze(1)          shape (N, 1)
        #             * w_scales[:, b].unsqueeze(0)           shape (1, out)
        # -> out += blk_scale * elem_dot
        #
        # Use batched matmul to avoid a Python loop overhead:
        #   x_fp8: (N, nb, bs) -> permute -> (nb, N, bs)
        #   w_fp8: (out, nb, bs) -> permute -> (nb, bs, out)
        #   bmm   -> (nb, N, out)
        elem_dots = torch.bmm(
            x_fp8.permute(1, 0, 2),            # (nb, N, bs)
            w_fp8.permute(1, 2, 0),            # (nb, bs, out)
        )                                       # (nb, N, out)

        # Combined fp32 scales: (nb, N, 1) * (nb, 1, out) -> (nb, N, out)
        combined_scales = x_scales.t().unsqueeze(-1) * w_scales.t().unsqueeze(1)

        # 2. Placeholder for future MSB-first bit-serial computation implementation

        # Weighted sum over blocks -> (N, out)
        result = (elem_dots * combined_scales).sum(dim=0)

        if self.bias_param is not None:
            result = result + self.bias_param

        return result.view(*batch_shape, self.out_features).to(orig_dtype)

class Qwen3RMSNorm(Qwen2RMSNorm):
    pass

# explicitly define MLP to implement mxfp8 quantization and future MSB-first bit-serial computation, instead of using GemmaMLP which is a blackbox import
class Qwen3MLP(nn.Module):
    _mxfp8_logged: bool = False  # class-level flag: log status only once across all layers

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        if not Qwen3MLP._mxfp8_logged:
            if config.use_mxfp8:
                logger.info(
                    f"Qwen3MLP: MX FP8 quantization ENABLED "
                    f"(block_size={config.mxfp8_block_size}, format=E4M3FN, max={_FP8_E4M3_MAX})"
                )
            else:
                logger.info("Qwen3MLP: MX FP8 quantization DISABLED (using standard fp32 nn.Linear)")
            Qwen3MLP._mxfp8_logged = True

        if config.use_mxfp8:
            self.gate_proj = MXFP8Linear(self.hidden_size, self.intermediate_size, bias=False, config=config)
            self.up_proj = MXFP8Linear(self.hidden_size, self.intermediate_size, bias=False, config=config)
            self.down_proj = MXFP8Linear(self.intermediate_size, self.hidden_size, bias=False, config=config)
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
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
