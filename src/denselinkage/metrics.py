"""Evaluation — pure functions over already-computed outputs."""

from collections.abc import Sequence

from denselinkage.core.models import CandidatePair
from denselinkage.core.results import (
    BlockingMetrics,
    LabeledPairs,
    LinkageMetrics,
    LinkageResult,
)


def linkage_metrics(result: LinkageResult, *, gold: LabeledPairs) -> LinkageMetrics:
    """Precision/recall/F1 over all candidate pairs against ``gold``.

    Errored pairs (a ``MatchError`` in ``result.errors``) are excluded from
    tp/fp/fn and surfaced as ``LinkageMetrics.n_errors``. Every ``gold`` pair
    not predicted a match counts as a false negative — including gold pairs
    the blocker never surfaced — so recall is honest end-to-end.

    Pair identity: a ``link`` result is compared to ``gold`` using the
    pair order as given (``(left_id, right_id)``); a ``dedupe`` result is
    compared after canonicalizing both sides to an unordered key
    (``frozenset({a, b})``), since left/right is arbitrary within one source.
    """
    ...


def blocking_metrics(
    candidates: Sequence[CandidatePair],
    *,
    gold: LabeledPairs,
    ks: Sequence[int],
) -> BlockingMetrics:
    """Pair-completeness@k for each k in ``ks``.

    Pair identity: same comparison rule as ``linkage_metrics`` — ordered for ``link``
    candidates, canonicalized to an unordered key for ``dedupe`` candidates.
    """
    ...


def pair_completeness_at_k(
    candidates: Sequence[CandidatePair], *, gold: LabeledPairs, k: int
) -> float:
    """Single-k pair-completeness (kwarg ``gold``, consistent with the other
    metrics). Pair identity: same comparison rule as ``linkage_metrics``."""

    ...


__all__ = ["blocking_metrics", "linkage_metrics", "pair_completeness_at_k"]
