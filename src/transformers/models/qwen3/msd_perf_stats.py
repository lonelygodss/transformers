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
Hierarchical performance statistics for MSD-first online arithmetic inference.

Collects fine-grained statistics during MSD truncated dot-product inference,
covering four hierarchy levels:

    Bit level    — effective precision (p_eff) distribution per element
    Block level  — zero / partial / full block activation counts
    Channel level — total cycles consumed vs budget, utilization ratio
    MAC level    — total / active / skipped multiply-accumulate operations

These statistics reflect *actual runtime behaviour* during inference (not
calibration-time properties).  They are designed to feed future hardware
simulation: latency = f(cycle counts), energy = f(sparsity, utilization).

Output structure (from ``finalize``):

    {
        "global": { ... model-wide aggregates ... },
        "per_layer": {
            "<layer_name>": {
                "summary": { bit_level, block_level, channel_level, mac_level },
                "channel_detail": { ... }   # only for detail_layer layers
            }
        }
    }

Layer summaries are compact scalar/small-structure dicts produced for every
layer.  Per-channel detail arrays (one entry per output channel) are only
emitted for layers belonging to the transformer layer index specified by
``detail_layer`` (default 2), matching the pattern used in calibration.

Usage:
    Automatically active when MSD truncation is enabled.  The accumulator
    lives inside MSDComputeContext and is fed by _forward_msd_truncated
    on every forward call.  Retrieve results via model.get_perf_stats().
"""

import torch

from ...utils import logging


logger = logging.get_logger(__name__)

# Histogram bin edges for effective precision distribution.
# Bins: [0], [1,4], [5,8], [9,12], [13,16], [17,24], [25,32], [33,+inf)
_P_EFF_BIN_EDGES = [0.5, 4.5, 8.5, 12.5, 16.5, 24.5, 32.5]
_P_EFF_BIN_LABELS = ["0", "1-4", "5-8", "9-12", "13-16", "17-24", "25-32", "33+"]
_NUM_BINS = len(_P_EFF_BIN_LABELS)


class _LayerAccumulator:
    """
    Per-layer accumulator for hierarchical MSD performance statistics.

    All tensors are 1D of shape (out_features,) or (out_features, num_bins),
    stored in float64/int64 for numerical stability across many forward passes.
    """

    def __init__(self, out_features: int, device: torch.device):
        self.out = out_features
        self.device = device

        # ── Bit level: effective precision distribution ──
        self.p_eff_sum = torch.zeros(out_features, dtype=torch.float64, device=device)
        self.p_eff_sq_sum = torch.zeros(out_features, dtype=torch.float64, device=device)
        self.p_eff_hist = torch.zeros(out_features, _NUM_BINS, dtype=torch.int64, device=device)

        # ── Bit level (active-only): conditional p_eff for elements where p_eff > 0 ──
        self.active_p_eff_sum = torch.zeros(out_features, dtype=torch.float64, device=device)
        self.active_element_count = torch.zeros(out_features, dtype=torch.int64, device=device)

        # ── Block level: activation categories ──
        # At block granularity:
        #   "zero"    — ALL bs elements have p_eff == 0  (no work done)
        #   "full"    — ALL bs elements completed: p_eff >= naf_width (every MAC finished)
        #   "partial" — neither zero nor full (some elements truncated by budget)
        self.block_zero_count = torch.zeros(out_features, dtype=torch.int64, device=device)
        self.block_partial_count = torch.zeros(out_features, dtype=torch.int64, device=device)
        self.block_full_count = torch.zeros(out_features, dtype=torch.int64, device=device)

        # ── Block level (partial-block detail): active element count within partial blocks ──
        # Used to compute mean active fraction within partial blocks.
        self.partial_block_active_elem_sum = torch.zeros(out_features, dtype=torch.int64, device=device)

        # ── Channel level: cycle accounting ──
        # total_budget_cycles = sum of b_final[n,j] across all N samples
        # effective_cycles = sum of p_eff across all elements (total "work done")
        self.total_budget_sum = torch.zeros(out_features, dtype=torch.float64, device=device)
        self.effective_cycles_sum = torch.zeros(out_features, dtype=torch.float64, device=device)

        # ── Channel level: max latency indicators ──
        # max_budget = running max of b_final[n,j] per channel across all samples
        # max_total_delay = running max of total_delay per channel across all (n,b,k)
        self.max_budget = torch.zeros(out_features, dtype=torch.float32, device=device)
        self.max_total_delay = torch.zeros(out_features, dtype=torch.float32, device=device)

        # ── MAC level: element-wise sparsity & completion ──
        # zero_element_count = number of elements with p_eff == 0 (fully skipped MACs)
        # completed_element_count = number of elements with p_eff >= naf_width (fully finished MACs)
        self.zero_element_count = torch.zeros(out_features, dtype=torch.int64, device=device)
        self.completed_element_count = torch.zeros(out_features, dtype=torch.int64, device=device)

        # ── Counting ──
        self.total_elements = torch.zeros(out_features, dtype=torch.int64, device=device)
        self.total_blocks = torch.zeros(out_features, dtype=torch.int64, device=device)
        self.num_samples = 0  # total N across all forward calls
        self.in_features = 0


class MSDPerfAccumulator:
    """
    Collects per-layer, per-channel hierarchical performance statistics
    during MSD inference.

    Hierarchy:
        Bit level    — p_eff distribution (mean, std, histogram)
        Block level  — zero / partial / full block activation counts
        Channel level — total cycles consumed, effective cycles, utilization
        MAC level    — total / active / skipped MACs, sparsity ratio
        Global       — model-wide aggregates across all layers

    Thread-safety: NOT thread-safe (same as MSDComputeContext — one per process).
    """

    def __init__(self):
        self._layers: dict[str, _LayerAccumulator] = {}
        self._bin_edges_cache: dict[torch.device, torch.Tensor] = {}

    def _get_or_create_layer(self, layer_name: str, out_features: int, device: torch.device) -> _LayerAccumulator:
        """Get existing accumulator or create a new one for this layer."""
        if layer_name not in self._layers:
            self._layers[layer_name] = _LayerAccumulator(out_features, device)
        return self._layers[layer_name]

    def _get_bin_edges(self, device: torch.device) -> torch.Tensor:
        """Cached bin edges tensor for the given device."""
        if device not in self._bin_edges_cache:
            self._bin_edges_cache[device] = torch.tensor(
                _P_EFF_BIN_EDGES, dtype=torch.float32, device=device
            )
        return self._bin_edges_cache[device]

    def record_chunk(
        self,
        layer_name: str,
        p_eff: torch.Tensor,
        b_final_c: torch.Tensor,
        j0: int,
        j1: int,
        N: int,
        nb: int,
        bs: int,
        *,
        max_delay_chunk: torch.Tensor | None = None,
        max_budget_chunk: torch.Tensor | None = None,
        naf_width: torch.Tensor | None = None,
    ):
        """
        Record statistics for one output chunk from _forward_msd_truncated.

        Called inside the per-chunk loop with tensors that already exist
        (no new large allocations).

        Args:
            layer_name: e.g. "layers.0.mlp.gate_proj"
            p_eff: (N, c, nb, bs) effective precision tensor (float32)
            b_final_c: (N, c) per-sample per-channel budget for this chunk
            j0, j1: output channel slice [j0, j1)
            N: batch dimension
            nb: number of blocks
            bs: block size
            max_delay_chunk: (c,) max total_delay per channel for this chunk (optional)
            max_budget_chunk: (c,) max b_final per channel for this chunk (optional)
            naf_width: (N, c, nb, bs) NAF digit width of each product element (int32, optional).
                       When provided, enables correct full/partial block classification
                       (full = every element completed, not just active).
        """
        c = j1 - j0
        device = p_eff.device

        # Lazily determine full output size from the first chunk.
        acc = self._layers.get(layer_name)
        if acc is None:
            acc = _LayerAccumulator(j1, device)
            self._layers[layer_name] = acc
        elif j1 > acc.out:
            acc = self._grow_accumulator(layer_name, j1, device)

        # ── Bit level ──
        p_flat = p_eff.reshape(N, c, nb * bs)  # (N, c, nb*bs)
        p_flat_d = p_flat.double()
        channel_p_sum = p_flat_d.sum(dim=(0, 2))  # (c,)
        channel_p_sq = (p_flat_d ** 2).sum(dim=(0, 2))  # (c,)
        acc.p_eff_sum[j0:j1] += channel_p_sum
        acc.p_eff_sq_sum[j0:j1] += channel_p_sq

        # Histogram: bucketize p_eff into bins, count per channel
        bin_edges = self._get_bin_edges(device)
        bin_idx = torch.bucketize(p_eff.float(), bin_edges)  # (N, c, nb, bs)
        bin_idx_flat = bin_idx.reshape(N * c, nb * bs)
        for bi in range(_NUM_BINS):
            counts = (bin_idx_flat == bi).sum(dim=-1).reshape(N, c).sum(dim=0)  # (c,)
            acc.p_eff_hist[j0:j1, bi] += counts.long()
        del bin_idx, bin_idx_flat

        # ── Bit level (active-only) ──
        active_mask = p_flat > 0  # (N, c, nb*bs)
        active_count_per_ch = active_mask.sum(dim=(0, 2))  # (c,)
        acc.active_element_count[j0:j1] += active_count_per_ch.long()
        acc.active_p_eff_sum[j0:j1] += (p_flat_d * active_mask).sum(dim=(0, 2))
        del active_mask

        # ── MAC level (element sparsity & completion) ──
        zero_count = (p_flat == 0).sum(dim=(0, 2))  # (c,)
        acc.zero_element_count[j0:j1] += zero_count.long()

        # ── Block level ──
        p_block = p_eff.reshape(N, c, nb, bs)
        block_active_count = (p_block > 0).sum(dim=-1)  # (N, c, nb)
        block_is_zero = (block_active_count == 0)

        if naf_width is not None:
            # Correct definitions using NAF digit widths:
            #   "full"  = every element in the block completed (p_eff >= naf_width)
            #   "partial" = not zero AND not full (some elements truncated by budget)
            naf_block = naf_width.reshape(N, c, nb, bs)  # (N, c, nb, bs)
            # An element is "completed" when it has no unprocessed digits
            elem_completed = (p_eff.reshape(N, c, nb, bs) >= naf_block.float())
            block_completed_count = elem_completed.sum(dim=-1)  # (N, c, nb)
            block_is_full = (block_completed_count == bs)
            del naf_block, elem_completed, block_completed_count

            # Also count completed elements model-wide for MAC stats
            completed_flat = (p_flat >= naf_width.reshape(N, c, nb * bs).float())
            completed_count = completed_flat.sum(dim=(0, 2))  # (c,)
            acc.completed_element_count[j0:j1] += completed_count.long()
            del completed_flat, completed_count
        else:
            # Fallback: without naf_width, use the old heuristic (all elements active)
            block_is_full = (block_active_count == bs)

        block_is_partial = ~block_is_zero & ~block_is_full

        acc.block_zero_count[j0:j1] += block_is_zero.sum(dim=(0, 2)).long()
        acc.block_full_count[j0:j1] += block_is_full.sum(dim=(0, 2)).long()
        acc.block_partial_count[j0:j1] += block_is_partial.sum(dim=(0, 2)).long()

        # Active elements within partial blocks (for mean active fraction)
        partial_active = block_active_count * block_is_partial  # zero out non-partial
        acc.partial_block_active_elem_sum[j0:j1] += partial_active.sum(dim=(0, 2)).long()
        del block_active_count, block_is_zero, block_is_full, block_is_partial, partial_active

        # ── Channel level ──
        acc.total_budget_sum[j0:j1] += b_final_c.double().sum(dim=0)
        acc.effective_cycles_sum[j0:j1] += p_flat_d.sum(dim=(0, 2))

        # ── Channel level: running max latency indicators ──
        if max_budget_chunk is not None:
            torch.maximum(acc.max_budget[j0:j1], max_budget_chunk, out=acc.max_budget[j0:j1])
        if max_delay_chunk is not None:
            torch.maximum(acc.max_total_delay[j0:j1], max_delay_chunk, out=acc.max_total_delay[j0:j1])

        # ── Counting ──
        n_elements = N * nb * bs
        acc.total_elements[j0:j1] += n_elements
        acc.total_blocks[j0:j1] += N * nb
        acc.num_samples += N
        acc.in_features = nb * bs

        del p_flat_d

    def _grow_accumulator(self, layer_name: str, new_out: int, device: torch.device) -> _LayerAccumulator:
        """Grow an existing accumulator to accommodate more output channels."""
        old = self._layers[layer_name]
        new_acc = _LayerAccumulator(new_out, device)
        o = old.out
        # Copy existing data
        new_acc.p_eff_sum[:o] = old.p_eff_sum
        new_acc.p_eff_sq_sum[:o] = old.p_eff_sq_sum
        new_acc.p_eff_hist[:o] = old.p_eff_hist
        new_acc.active_p_eff_sum[:o] = old.active_p_eff_sum
        new_acc.active_element_count[:o] = old.active_element_count
        new_acc.block_zero_count[:o] = old.block_zero_count
        new_acc.block_partial_count[:o] = old.block_partial_count
        new_acc.block_full_count[:o] = old.block_full_count
        new_acc.partial_block_active_elem_sum[:o] = old.partial_block_active_elem_sum
        new_acc.total_budget_sum[:o] = old.total_budget_sum
        new_acc.effective_cycles_sum[:o] = old.effective_cycles_sum
        new_acc.max_budget[:o] = old.max_budget
        new_acc.max_total_delay[:o] = old.max_total_delay
        new_acc.zero_element_count[:o] = old.zero_element_count
        new_acc.completed_element_count[:o] = old.completed_element_count
        new_acc.total_elements[:o] = old.total_elements
        new_acc.total_blocks[:o] = old.total_blocks
        new_acc.num_samples = old.num_samples
        new_acc.in_features = old.in_features
        self._layers[layer_name] = new_acc
        return new_acc

    # ──────────────────────────────────────────────────────────────────────
    #  finalize — reduce accumulated stats into JSON-serializable output
    # ──────────────────────────────────────────────────────────────────────

    def finalize(self, detail_layer: int = 2) -> dict:
        """
        Reduce all accumulated statistics to a JSON-serializable dict.

        Produces **compact layer summaries** (scalars / small structures) for
        every layer, and **per-channel detail arrays** only for layers whose
        name contains ``model.layers.<detail_layer>.``.

        Args:
            detail_layer: transformer layer index whose sub-modules receive
                full per-channel detail (default: 2).

        Returns:
            {
                "global": { ... model-wide summary ... },
                "per_layer": {
                    "<layer_name>": {
                        "summary": { bit_level, block_level, channel_level,
                                     mac_level },
                        "channel_detail": { ... }   # only for detail_layer
                    }
                }
            }
        """
        detail_prefix = f"layers.{detail_layer}."
        per_layer = {}

        # Global accumulators
        g_total_elements = 0
        g_zero_elements = 0
        g_active_elements = 0
        g_p_eff_sum = 0.0
        g_active_p_eff_sum = 0.0
        g_total_budget = 0.0
        g_budget_mac_capacity = 0.0
        g_effective_cycles = 0.0
        g_total_blocks = 0
        g_zero_blocks = 0
        g_partial_blocks = 0
        g_full_blocks = 0
        g_completed_elements = 0
        g_partial_active_elem_sum = 0
        g_max_budget = 0.0
        g_max_total_delay = 0.0

        for layer_name, acc in self._layers.items():
            want_detail = detail_prefix in layer_name
            layer_entry = self._finalize_layer(acc, want_detail)
            per_layer[layer_name] = layer_entry

            # ── Accumulate global stats ──
            g_total_elements += acc.total_elements.sum().item()
            g_zero_elements += acc.zero_element_count.sum().item()
            g_active_elements += acc.active_element_count.sum().item()
            g_p_eff_sum += acc.p_eff_sum.sum().item()
            g_active_p_eff_sum += acc.active_p_eff_sum.sum().item()
            g_total_budget += acc.total_budget_sum.sum().item()
            g_budget_mac_capacity += acc.total_budget_sum.sum().item() * acc.in_features
            g_effective_cycles += acc.effective_cycles_sum.sum().item()
            g_total_blocks += acc.total_blocks.sum().item()
            g_zero_blocks += acc.block_zero_count.sum().item()
            g_partial_blocks += acc.block_partial_count.sum().item()
            g_full_blocks += acc.block_full_count.sum().item()
            g_completed_elements += acc.completed_element_count.sum().item()
            g_partial_active_elem_sum += acc.partial_block_active_elem_sum.sum().item()
            g_max_budget = max(g_max_budget, acc.max_budget.max().item())
            g_max_total_delay = max(g_max_total_delay, acc.max_total_delay.max().item())

        g_total_safe = max(g_total_elements, 1)
        g_active_safe = max(g_active_elements, 1)
        g_blocks_safe = max(g_total_blocks, 1)
        g_active_macs = g_total_elements - g_zero_elements
        g_partial_safe = max(g_partial_blocks, 1)

        global_stats = {
            "num_layers": len(self._layers),
            # MAC level (model-wide)
            "total_macs": g_total_elements,
            "active_macs": g_active_macs,
            "completed_macs": g_completed_elements,
            "mac_sparsity": round(g_zero_elements / g_total_safe, 6),
            "mac_completion_ratio": round(g_completed_elements / g_total_safe, 6),
            # Bit level (model-wide)
            "mean_effective_precision": round(g_p_eff_sum / g_total_safe, 4),
            "active_p_eff_mean": round(g_active_p_eff_sum / g_active_safe, 4),
            # Channel level (model-wide)
            "total_budget_cycles": round(g_total_budget, 1),
            "effective_cycles": round(g_effective_cycles, 1),
            "global_utilization": round(g_effective_cycles / max(g_budget_mac_capacity, 1e-30), 6),
            # Block level (model-wide)
            "total_blocks": g_total_blocks,
            "zero_blocks": g_zero_blocks,
            "zero_block_ratio": round(g_zero_blocks / g_blocks_safe, 6),
            "partial_blocks": g_partial_blocks,
            "partial_block_ratio": round(g_partial_blocks / g_blocks_safe, 6),
            "full_blocks": g_full_blocks,
            "full_block_ratio": round(g_full_blocks / g_blocks_safe, 6),
            # Latency indicators (model-wide worst case)
            "max_budget": round(g_max_budget, 2),
            "max_total_delay": round(g_max_total_delay, 2),
        }

        return {
            "global": global_stats,
            "per_layer": per_layer,
        }

    def _finalize_layer(self, acc: _LayerAccumulator, want_detail: bool) -> dict:
        """
        Produce summary (always) and channel_detail (only if want_detail)
        for a single layer's accumulator.
        """
        total_elem = acc.total_elements.float()  # (out,)
        total_elem_safe = total_elem.clamp(min=1)

        # ── Per-channel intermediate tensors (on GPU) ──
        p_mean_ch = (acc.p_eff_sum / total_elem_safe.double()).float()  # (out,)
        p_var_ch = (acc.p_eff_sq_sum / total_elem_safe.double()
                    - (acc.p_eff_sum / total_elem_safe.double()) ** 2).clamp(min=0)
        p_std_ch = p_var_ch.sqrt().float()  # (out,)

        active_safe = acc.active_element_count.clamp(min=1).double()
        active_p_mean_ch = (acc.active_p_eff_sum / active_safe).float()  # (out,)

        total_blk = acc.total_blocks.float().clamp(min=1)
        total_blk_sum = acc.total_blocks.sum().float().clamp(min=1)
        total_instances = max(acc.num_samples * acc.out, 1)
        partial_safe = acc.block_partial_count.clamp(min=1).float()

        budget_mac_capacity = acc.total_budget_sum * acc.in_features
        budget_safe = budget_mac_capacity.clamp(min=1e-30)
        utilization_ch = (acc.effective_cycles_sum / budget_safe).float()  # (out,)

        mac_sparsity_ch = (acc.zero_element_count.float() / total_elem_safe)  # (out,)

        # Aggregate histogram across channels: (num_bins,)
        hist_agg = acc.p_eff_hist.sum(dim=0)  # (num_bins,)

        # Partial block mean active fraction (per-channel, then average)
        partial_active_frac_ch = (
            acc.partial_block_active_elem_sum.float() / partial_safe
        )
        # Normalize by bs: total_elements[0] / total_blocks[0] = bs
        bs_est = (total_elem[0] / acc.total_blocks[0].float()).item() if acc.total_blocks[0] > 0 else 1.0
        partial_active_frac_ch = partial_active_frac_ch / bs_est  # fraction in [0, 1]

        # ── Build compact SUMMARY (always) ──────────────────────────
        summary = {
            "bit_level": {
                "p_eff_mean": round(float(p_mean_ch.mean()), 4),
                "p_eff_std": round(float(p_std_ch.mean()), 4),
                "active_p_eff_mean": round(float(active_p_mean_ch.mean()), 4),
                "p_eff_histogram": {
                    "bin_labels": _P_EFF_BIN_LABELS,
                    "counts": hist_agg.cpu().tolist(),
                },
            },
            "block_level": {
                "total_blocks": int(acc.total_blocks.sum().item()),
                "zero_block_ratio": round(float(
                    acc.block_zero_count.sum().float() / total_blk_sum
                ), 6),
                "partial_block_ratio": round(float(
                    acc.block_partial_count.sum().float() / total_blk_sum
                ), 6),
                "full_block_ratio": round(float(
                    acc.block_full_count.sum().float() / total_blk_sum
                ), 6),
                "partial_block_mean_active_frac": round(float(partial_active_frac_ch.mean()), 6),
            },
            "channel_level": {
                "budget_mean": round(float(
                    acc.total_budget_sum.sum() / total_instances
                ), 2),
                "effective_cycles_total": round(float(acc.effective_cycles_sum.sum()), 1),
                "utilization_mean": round(float(utilization_ch.mean()), 6),
                "max_budget": round(float(acc.max_budget.max()), 2),
                "max_total_delay": round(float(acc.max_total_delay.max()), 2),
            },
            "mac_level": {
                "total_macs": int(acc.total_elements.sum().item()),
                "active_macs": int(acc.active_element_count.sum().item()),
                "completed_macs": int(acc.completed_element_count.sum().item()),
                "mac_sparsity": round(float(
                    acc.zero_element_count.sum().float()
                    / acc.total_elements.sum().float().clamp(min=1)
                ), 6),
                "mac_completion_ratio": round(float(
                    acc.completed_element_count.sum().float()
                    / acc.total_elements.sum().float().clamp(min=1)
                ), 6),
            },
        }

        layer_entry = {"summary": summary}

        # ── Build per-channel DETAIL (only for detail_layer layers) ──
        if want_detail:
            channel_detail = {
                "bit_level": {
                    "p_eff_mean": p_mean_ch.cpu().tolist(),
                    "p_eff_std": p_std_ch.cpu().tolist(),
                    "active_p_eff_mean": active_p_mean_ch.cpu().tolist(),
                    "p_eff_histogram": {
                        "bin_labels": _P_EFF_BIN_LABELS,
                        "counts": acc.p_eff_hist.cpu().tolist(),  # (out, num_bins)
                    },
                },
                "block_level": {
                    "zero_block_count": acc.block_zero_count.cpu().tolist(),
                    "partial_block_count": acc.block_partial_count.cpu().tolist(),
                    "full_block_count": acc.block_full_count.cpu().tolist(),
                    "zero_block_ratio": (acc.block_zero_count.float() / total_blk).cpu().tolist(),
                    "partial_block_ratio": (acc.block_partial_count.float() / total_blk).cpu().tolist(),
                    "full_block_ratio": (acc.block_full_count.float() / total_blk).cpu().tolist(),
                    "partial_block_active_frac": partial_active_frac_ch.cpu().tolist(),
                },
                "channel_level": {
                    "total_budget_cycles": acc.total_budget_sum.float().cpu().tolist(),
                    "effective_cycles": acc.effective_cycles_sum.float().cpu().tolist(),
                    "skipped_cycles": (
                        budget_mac_capacity - acc.effective_cycles_sum
                    ).clamp(min=0).float().cpu().tolist(),
                    "utilization": utilization_ch.cpu().tolist(),
                    "max_budget": acc.max_budget.cpu().tolist(),
                    "max_total_delay": acc.max_total_delay.cpu().tolist(),
                },
                "mac_level": {
                    "total_elements": acc.total_elements.cpu().tolist(),
                    "zero_elements": acc.zero_element_count.cpu().tolist(),
                    "completed_elements": acc.completed_element_count.cpu().tolist(),
                    "mac_sparsity": mac_sparsity_ch.cpu().tolist(),
                    "mac_completion_ratio": (acc.completed_element_count.float() / total_elem_safe).cpu().tolist(),
                },
            }
            layer_entry["channel_detail"] = channel_detail

        return layer_entry

    def reset(self):
        """Clear all accumulated statistics."""
        self._layers.clear()
        self._bin_edges_cache.clear()

    @property
    def has_data(self) -> bool:
        """True if any statistics have been recorded."""
        return len(self._layers) > 0
