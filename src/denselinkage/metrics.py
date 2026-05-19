"""Evaluation — pure functions over already-computed outputs."""

from collections.abc import Sequence

from denselinkage.core.models import CandidatePair
from denselinkage.core.results import (
    BlockingMetrics,
    LabeledPairs,
    LinkageMetrics,
    LinkageResult,
)


def linkage_metrics(result: LinkageResult, *, gold: LabeledPairs) -> LinkageMetrics: ...


def blocking_metrics(
    candidates: Sequence[CandidatePair],
    *,
    gold: LabeledPairs,
    ks: Sequence[int],
) -> BlockingMetrics: ...


def pair_completeness_at_k(
    candidates: Sequence[CandidatePair], gold: LabeledPairs, *, k: int
) -> float: ...


__all__ = ["blocking_metrics", "linkage_metrics", "pair_completeness_at_k"]
