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
    Global       — model-wide sparsity ratios, total vs effective MACs

These statistics reflect *actual runtime behaviour* during inference (not
calibration-time properties).  They are designed to feed future hardware
simulation: latency = f(cycle counts), energy = f(sparsity, utilization).

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
    stored in float64 for numerical stability across many forward passes.
    """

    def __init__(self, out_features: int, device: torch.device):
        self.out = out_features
        self.device = device

        # ── Bit level: effective precision distribution ──
        self.p_eff_sum = torch.zeros(out_features, dtype=torch.float64, device=device)
        self.p_eff_sq_sum = torch.zeros(out_features, dtype=torch.float64, device=device)
        self.p_eff_hist = torch.zeros(out_features, _NUM_BINS, dtype=torch.int64, device=device)

        # ── Block level: activation categories ──
        # An "element" here is one (n, j, b, k) product entry.
        # zero:    p_eff == 0 (fully skipped, no computation)
        # partial: 0 < p_eff < naf_width of the product (truncated)
        # full:    p_eff >= naf_width (all BSD digits computed)
        # For simplicity, we use p_eff == 0 as zero, p_eff > 0 as "active".
        # At block granularity: a block (n,j,b) is "zero" if ALL bs elements
        # in it have p_eff == 0; "full" if ALL have p_eff > 0; else "partial".
        self.block_zero_count = torch.zeros(out_features, dtype=torch.int64, device=device)
        self.block_partial_count = torch.zeros(out_features, dtype=torch.int64, device=device)
        self.block_full_count = torch.zeros(out_features, dtype=torch.int64, device=device)

        # ── Channel level: cycle accounting ──
        # total_budget_cycles = sum of b_final[n,j] across all N samples
        # effective_cycles = sum of p_eff across all elements (total "work done")
        # skipped_cycles = total_budget_cycles * nb * bs - effective_cycles
        self.total_budget_sum = torch.zeros(out_features, dtype=torch.float64, device=device)
        self.effective_cycles_sum = torch.zeros(out_features, dtype=torch.float64, device=device)

        # ── Counting ──
        self.total_elements = torch.zeros(out_features, dtype=torch.int64, device=device)
        self.total_blocks = torch.zeros(out_features, dtype=torch.int64, device=device)
        self.num_samples = 0  # total N across all forward calls


class MSDPerfAccumulator:
    """
    Collects per-layer, per-channel hierarchical performance statistics
    during MSD inference.

    Hierarchy:
        Bit level    — p_eff distribution (mean, std, histogram)
        Block level  — zero / partial / full block activation counts
        Channel level — total cycles consumed, effective cycles, utilization
        Global       — model-wide aggregates (sparsity, mean precision)

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
    ):
        """
        Record statistics for one output chunk from _forward_msd_truncated.

        Called inside the per-chunk loop with tensors that already exist
        (no new large allocations).

        Args:
            layer_name: e.g. "model.layers.0.mlp.gate_proj"
            p_eff: (N, c, nb, bs) effective precision tensor (float32)
            b_final_c: (N, c) per-sample per-channel budget for this chunk
            j0, j1: output channel slice [j0, j1)
            N: batch dimension
            nb: number of blocks
            bs: block size
        """
        c = j1 - j0
        out_total = j1 if j0 == 0 else None  # we'll need the full out from the accumulator
        device = p_eff.device

        # Lazily determine full output size from the first chunk that creates the accumulator.
        # Subsequent chunks contribute to the same accumulator via slice indexing.
        # We defer full-out discovery: the accumulator is created with max(j1) seen.
        acc = self._layers.get(layer_name)
        if acc is None:
            # We don't know full out_features yet from just a chunk.
            # Use a generous initial size; _ensure_size will grow if needed.
            # Actually, we'll set it properly when finalize is called.
            # For now, we pre-allocate for the maximum j1 we've seen.
            acc = _LayerAccumulator(j1, device)
            self._layers[layer_name] = acc
        elif j1 > acc.out:
            # Grow accumulator (rare — only if chunks arrive out of order or
            # first chunk didn't cover full output)
            acc = self._grow_accumulator(layer_name, j1, device)

        # ── Bit level ──
        # p_eff: (N, c, nb, bs) — reduce to per-channel stats
        p_flat = p_eff.reshape(N, c, nb * bs)  # (N, c, nb*bs)
        channel_p_mean = p_flat.double().sum(dim=(0, 2))  # (c,) sum over N and elements
        channel_p_sq = (p_flat.double() ** 2).sum(dim=(0, 2))  # (c,)
        acc.p_eff_sum[j0:j1] += channel_p_mean
        acc.p_eff_sq_sum[j0:j1] += channel_p_sq

        # Histogram: bucketize p_eff into bins, count per channel
        bin_edges = self._get_bin_edges(device)
        bin_idx = torch.bucketize(p_eff.float(), bin_edges)  # (N, c, nb, bs), values in [0, _NUM_BINS-1]
        bin_idx_flat = bin_idx.reshape(N * c, nb * bs)
        # Count per (channel_within_chunk) per bin
        for bi in range(_NUM_BINS):
            counts = (bin_idx_flat == bi).sum(dim=-1).reshape(N, c).sum(dim=0)  # (c,)
            acc.p_eff_hist[j0:j1, bi] += counts.long()
        del bin_idx, bin_idx_flat

        # ── Block level ──
        # Per block (n, j, b): check if all bs elements are zero or all active
        p_block = p_eff.reshape(N, c, nb, bs)
        block_active_count = (p_block > 0).sum(dim=-1)  # (N, c, nb) — num active elements per block
        block_is_zero = (block_active_count == 0)        # all elements skipped
        block_is_full = (block_active_count == bs)        # all elements computed
        block_is_partial = ~block_is_zero & ~block_is_full

        acc.block_zero_count[j0:j1] += block_is_zero.sum(dim=(0, 2)).long()     # (c,)
        acc.block_full_count[j0:j1] += block_is_full.sum(dim=(0, 2)).long()     # (c,)
        acc.block_partial_count[j0:j1] += block_is_partial.sum(dim=(0, 2)).long()  # (c,)
        del block_active_count, block_is_zero, block_is_full, block_is_partial

        # ── Channel level ──
        # b_final_c: (N, c) — total budget cycles allocated to each channel
        acc.total_budget_sum[j0:j1] += b_final_c.double().sum(dim=0)
        # effective_cycles: sum of p_eff across all nb*bs elements per channel
        acc.effective_cycles_sum[j0:j1] += p_flat.double().sum(dim=(0, 2))

        # ── Counting ──
        n_elements = N * nb * bs
        acc.total_elements[j0:j1] += n_elements
        acc.total_blocks[j0:j1] += N * nb
        acc.num_samples += N  # Note: accumulated per-chunk, will divide later

    def _grow_accumulator(self, layer_name: str, new_out: int, device: torch.device) -> _LayerAccumulator:
        """Grow an existing accumulator to accommodate more output channels."""
        old = self._layers[layer_name]
        new_acc = _LayerAccumulator(new_out, device)
        o = old.out
        # Copy existing data
        new_acc.p_eff_sum[:o] = old.p_eff_sum
        new_acc.p_eff_sq_sum[:o] = old.p_eff_sq_sum
        new_acc.p_eff_hist[:o] = old.p_eff_hist
        new_acc.block_zero_count[:o] = old.block_zero_count
        new_acc.block_partial_count[:o] = old.block_partial_count
        new_acc.block_full_count[:o] = old.block_full_count
        new_acc.total_budget_sum[:o] = old.total_budget_sum
        new_acc.effective_cycles_sum[:o] = old.effective_cycles_sum
        new_acc.total_elements[:o] = old.total_elements
        new_acc.total_blocks[:o] = old.total_blocks
        new_acc.num_samples = old.num_samples
        self._layers[layer_name] = new_acc
        return new_acc

    def finalize(self) -> dict:
        """
        Reduce all accumulated statistics to a JSON-serializable dict.

        Returns:
            {
                "global": { ... model-wide summary ... },
                "per_layer": {
                    "layer_name": {
                        "bit_level": { ... },
                        "block_level": { ... },
                        "channel_level": { ... },
                    },
                    ...
                }
            }
        """
        per_layer = {}

        # Global accumulators
        g_total_elements = 0
        g_zero_elements = 0
        g_p_eff_sum = 0.0
        g_total_budget = 0.0
        g_effective_cycles = 0.0
        g_total_blocks = 0
        g_zero_blocks = 0
        g_partial_blocks = 0
        g_full_blocks = 0

        for layer_name, acc in self._layers.items():
            total_elem = acc.total_elements.float()  # (out,)
            total_elem_safe = total_elem.clamp(min=1)  # avoid div-by-zero

            # ── Bit level ──
            p_mean = (acc.p_eff_sum / total_elem_safe.double()).float()
            p_var = (acc.p_eff_sq_sum / total_elem_safe.double()
                     - (acc.p_eff_sum / total_elem_safe.double()) ** 2).clamp(min=0)
            p_std = p_var.sqrt().float()

            bit_level = {
                "p_eff_mean": p_mean.cpu().tolist(),
                "p_eff_std": p_std.cpu().tolist(),
                "p_eff_histogram": {
                    "bin_labels": _P_EFF_BIN_LABELS,
                    "counts": acc.p_eff_hist.cpu().tolist(),  # (out, num_bins)
                },
            }

            # ── Block level ──
            total_blk = acc.total_blocks.float().clamp(min=1)
            block_level = {
                "zero_block_count": acc.block_zero_count.cpu().tolist(),
                "partial_block_count": acc.block_partial_count.cpu().tolist(),
                "full_block_count": acc.block_full_count.cpu().tolist(),
                "zero_block_ratio": (acc.block_zero_count.float() / total_blk).cpu().tolist(),
                "partial_block_ratio": (acc.block_partial_count.float() / total_blk).cpu().tolist(),
                "full_block_ratio": (acc.block_full_count.float() / total_blk).cpu().tolist(),
            }

            # ── Channel level ──
            budget_safe = acc.total_budget_sum.clamp(min=1e-30)
            # utilization = effective_cycles / (total_budget * nb * bs)
            # But total_budget_sum is already per-sample budget sum.
            # effective_cycles_sum is sum of p_eff across all elements.
            # For utilization: ratio of actual work to maximum possible work.
            # Max possible = total_budget_sum * nb * bs / nb (budget is per-channel,
            #   each element in nb*bs can use up to budget cycles)
            # Actually: total possible cycles for channel j across all samples =
            #   sum_n(b_final[n,j]) * nb * bs  (each of the nb*bs elements could
            #   use up to b_final cycles).
            # But effective_cycles = sum of p_eff which is already the actual
            #   cycle count per element.
            # So: total_possible = total_budget_sum * nb * bs (when b_final was
            #   the budget PER dot-product, and we have nb*bs partial products).
            # Wait — p_eff is per-element, and there are nb*bs elements per (n,j).
            # total_budget_sum[j] = sum_n b_final[n,j]
            # total_possible[j] = total_budget_sum[j] * nb * bs
            # utilization[j] = effective_cycles_sum[j] / total_possible[j]
            #
            # But we don't have nb/bs at finalize time. Compute total_possible
            # from total_elements and total_budget:
            # total_possible = total_budget_sum * (total_elements / num_samples_seen)
            # where total_elements / num_samples_seen = nb * bs per sample.
            #
            # Actually total_elements[j] = N_total * nb * bs (same for all j per layer)
            # So nb*bs = total_elements[j] / N_total ... but N_total was accumulated
            # redundantly per chunk. Let's use the ratio directly:
            # utilization = effective_cycles / (total_budget * nb_bs_per_sample)
            # where nb_bs_per_sample = total_elements / total_budget_sum * b_mean
            # This is getting circular. Let's just store both raw numbers and let
            # the user compute ratios.

            channel_level = {
                "total_budget_cycles": acc.total_budget_sum.float().cpu().tolist(),
                "effective_cycles": acc.effective_cycles_sum.float().cpu().tolist(),
                "skipped_cycles": (
                    acc.total_budget_sum - acc.effective_cycles_sum
                ).clamp(min=0).float().cpu().tolist(),
                "utilization": (
                    acc.effective_cycles_sum / budget_safe
                ).float().cpu().tolist(),
            }

            per_layer[layer_name] = {
                "bit_level": bit_level,
                "block_level": block_level,
                "channel_level": channel_level,
            }

            # ── Accumulate global stats ──
            g_total_elements += acc.total_elements.sum().item()
            g_zero_elements += acc.p_eff_hist[:, 0].sum().item()  # bin 0 = p_eff in [0, 0.5)
            g_p_eff_sum += acc.p_eff_sum.sum().item()
            g_total_budget += acc.total_budget_sum.sum().item()
            g_effective_cycles += acc.effective_cycles_sum.sum().item()
            g_total_blocks += acc.total_blocks.sum().item()
            g_zero_blocks += acc.block_zero_count.sum().item()
            g_partial_blocks += acc.block_partial_count.sum().item()
            g_full_blocks += acc.block_full_count.sum().item()

        g_total_safe = max(g_total_elements, 1)
        g_blocks_safe = max(g_total_blocks, 1)

        global_stats = {
            "num_layers": len(self._layers),
            "total_elements": g_total_elements,
            "zero_elements": g_zero_elements,
            "zero_element_ratio": round(g_zero_elements / g_total_safe, 6),
            "mean_effective_precision": round(g_p_eff_sum / g_total_safe, 4),
            "total_budget_cycles": round(g_total_budget, 1),
            "effective_cycles": round(g_effective_cycles, 1),
            "global_utilization": round(g_effective_cycles / max(g_total_budget, 1e-30), 6),
            "total_blocks": g_total_blocks,
            "zero_blocks": g_zero_blocks,
            "zero_block_ratio": round(g_zero_blocks / g_blocks_safe, 6),
            "partial_blocks": g_partial_blocks,
            "partial_block_ratio": round(g_partial_blocks / g_blocks_safe, 6),
            "full_blocks": g_full_blocks,
            "full_block_ratio": round(g_full_blocks / g_blocks_safe, 6),
        }

        return {
            "global": global_stats,
            "per_layer": per_layer,
        }

    def reset(self):
        """Clear all accumulated statistics."""
        self._layers.clear()
        self._bin_edges_cache.clear()

    @property
    def has_data(self) -> bool:
        """True if any statistics have been recorded."""
        return len(self._layers) > 0
