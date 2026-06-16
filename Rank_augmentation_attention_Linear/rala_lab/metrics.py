"""Metrics and diagnostics for RALA experiments."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch


@dataclass(init=False)
class AttentionStats:
    layer: int | None = None
    memory_rank: float | None = None
    memory_rank_ratio: float | None = None
    global_output_rank: float | None = None
    global_output_rank_ratio: float | None = None
    output_rank: float | None = None
    output_rank_ratio: float | None = None
    alpha_sum_mean: float | None = None
    min_denominator: float | None = None
    warnings: list[str] = field(default_factory=list)
    # Raw tensors stashed during forward pass for deferred SVD computation.
    # These are NOT serialised — they exist only in memory between the forward
    # pass and the post-timer stats computation step.
    _raw_tensors: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    def __init__(
        self,
        layer: int | None = None,
        memory_rank: float | None = None,
        memory_rank_ratio: float | None = None,
        global_output_rank: float | None = None,
        global_output_rank_ratio: float | None = None,
        output_rank: float | None = None,
        output_rank_ratio: float | None = None,
        alpha_sum_mean: float | None = None,
        min_denominator: float | None = None,
        warnings: list[str] | None = None,
        kv_rank: float | None = None,
        kv_rank_ratio: float | None = None,
        _raw_tensors: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.layer = layer
        self.memory_rank = memory_rank if memory_rank is not None else kv_rank
        self.memory_rank_ratio = memory_rank_ratio if memory_rank_ratio is not None else kv_rank_ratio
        self.global_output_rank = global_output_rank
        self.global_output_rank_ratio = global_output_rank_ratio
        self.output_rank = output_rank
        self.output_rank_ratio = output_rank_ratio
        self.alpha_sum_mean = alpha_sum_mean
        self.min_denominator = min_denominator
        self.warnings = warnings or []
        self._raw_tensors = _raw_tensors or {}

    @property
    def kv_rank(self) -> float | None:
        """Backward-compatible alias for memory_rank."""
        return self.memory_rank

    @kv_rank.setter
    def kv_rank(self, value: float | None) -> None:
        self.memory_rank = value

    @property
    def kv_rank_ratio(self) -> float | None:
        """Backward-compatible alias for memory_rank_ratio."""
        return self.memory_rank_ratio

    @kv_rank_ratio.setter
    def kv_rank_ratio(self, value: float | None) -> None:
        self.memory_rank_ratio = value


@dataclass
class Timer:
    elapsed_ms: float = 0.0


@contextmanager
def timed() -> Iterator[Timer]:
    timer = Timer()
    start = time.perf_counter()
    yield timer
    timer.elapsed_ms = (time.perf_counter() - start) * 1000.0


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


DEFAULT_RANK_TOL = 1e-5


def average_matrix_rank(x: torch.Tensor, tol: float = DEFAULT_RANK_TOL) -> float:
    """Average rank across all leading dimensions of a matrix tensor."""

    if x.ndim < 2:
        return 0.0
    matrices = x.reshape(-1, x.shape[-2], x.shape[-1])
    ranks = torch.linalg.matrix_rank(matrices.float(), tol=tol)
    return ranks.float().mean().item()


def rank_ratio(x: torch.Tensor, tol: float = DEFAULT_RANK_TOL) -> tuple[float, float]:
    rank = average_matrix_rank(x, tol=tol)
    full_rank = min(x.shape[-2], x.shape[-1]) if x.ndim >= 2 else 1
    return rank, rank / max(float(full_rank), 1.0)


def compute_deferred_stats(
    stats: AttentionStats,
    rank_tol: float = DEFAULT_RANK_TOL,
) -> None:
    """Compute SVD rank metrics from raw tensors stashed during forward().

    This function is designed to be called AFTER the inference timer has
    stopped, so that the expensive SVD computation does not pollute the
    reported inference_ms.

    After computation, _raw_tensors is cleared to free GPU memory.
    """
    raw = stats._raw_tensors
    if not raw:
        return

    with torch.no_grad():
        if "memory" in raw:
            memory_rank, memory_ratio = rank_ratio(raw["memory"], tol=rank_tol)
            stats.memory_rank = memory_rank
            stats.memory_rank_ratio = memory_ratio

        if "o_global" in raw:
            global_rank, global_ratio = rank_ratio(raw["o_global"], tol=rank_tol)
            stats.global_output_rank = global_rank
            stats.global_output_rank_ratio = global_ratio

        if "output" in raw:
            output_rank, output_ratio = rank_ratio(raw["output"], tol=rank_tol)
            stats.output_rank = output_rank
            stats.output_rank_ratio = output_ratio

        if "stability_tensors" in raw:
            stats.warnings = stability_warnings(raw["stability_tensors"])

    # Free GPU memory — raw tensors are no longer needed.
    stats._raw_tensors = {}


def merge_layer_stats(stats_list: list[AttentionStats]) -> list[AttentionStats]:
    """Tag each stat with its layer index."""
    for i, s in enumerate(stats_list):
        s.layer = i
    return stats_list


def aggregate_stats_batches(batches: list[list[AttentionStats]]) -> list[AttentionStats]:
    """Average per-layer stats across multiple batches."""
    if not batches:
        return []
    n_layers = len(batches[0])
    aggregated: list[AttentionStats] = []
    for layer_idx in range(n_layers):
        layer_batch = [b[layer_idx] for b in batches if layer_idx < len(b)]
        n = len(layer_batch)
        if n == 0:
            aggregated.append(AttentionStats(layer=layer_idx))
            continue
        def _avg(attr: str) -> float | None:
            vals = [getattr(s, attr) for s in layer_batch if getattr(s, attr) is not None]
            return sum(vals) / len(vals) if vals else None
        all_warnings: list[str] = []
        for s in layer_batch:
            all_warnings.extend(s.warnings)
        # deduplicate warnings
        seen: set[str] = set()
        unique_warnings: list[str] = []
        for w in all_warnings:
            if w not in seen:
                seen.add(w)
                unique_warnings.append(w)
        aggregated.append(AttentionStats(
            layer=layer_idx,
            memory_rank=_avg("memory_rank"),
            memory_rank_ratio=_avg("memory_rank_ratio"),
            global_output_rank=_avg("global_output_rank"),
            global_output_rank_ratio=_avg("global_output_rank_ratio"),
            output_rank=_avg("output_rank"),
            output_rank_ratio=_avg("output_rank_ratio"),
            alpha_sum_mean=_avg("alpha_sum_mean"),
            min_denominator=_avg("min_denominator"),
            warnings=unique_warnings,
        ))
    return aggregated


def stability_warnings(
    tensors: dict[str, torch.Tensor | None],
    min_denominator: float | None = None,
) -> list[str]:
    warnings: list[str] = []
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        if torch.isnan(tensor).any().item():
            warnings.append(f"{name} contains NaN values.")
        if torch.isinf(tensor).any().item():
            warnings.append(f"{name} contains Inf values.")
        max_abs = tensor.detach().abs().max().item()
        if math.isfinite(max_abs) and max_abs > 1e4:
            warnings.append(f"{name} has large magnitude values: max |x| = {max_abs:.2e}.")
    if min_denominator is not None and min_denominator < 1e-7:
        warnings.append(f"Linear-attention denominator is near zero: {min_denominator:.2e}.")
    return warnings


METRIC_HELP = {
    "Accuracy": "Fraction of correct predictions. Higher is better, but compare repeated seeds before making claims.",
    "Loss": "Cross-entropy training or validation objective. Lower is better.",
    "Memory rank": "Average rank of the true global memory matrix. For hybrid attention this is S = K^T(g ⊙ V), or final M_T in recurrent mode.",
    "Global output rank": "Average rank of the retrieved global branch output before local attention and output gating.",
    "Output rank": "Average rank of the final attention output feature map. This is a diagnostic, not proof of higher accuracy by itself.",
    "Rank ratio": "Rank divided by the maximum possible rank. Values closer to 1.0 are closer to full-rank.",
    "Inference time": "Forward-pass wall-clock time for one validation batch. Lower is faster.",
    "Stability": "Warnings for NaN, Inf, large activations, or near-zero denominators.",
}
