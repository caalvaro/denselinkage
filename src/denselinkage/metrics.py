"""Evaluation — pure functions over already-computed outputs."""

from collections.abc import Sequence

from denselinkage.core.models import CandidatePair, RecordId
from denselinkage.core.results import (
    BlockingMetrics,
    Clustering,
    ClusteringMetrics,
    LabeledPairs,
    LinkageMetrics,
    LinkageResult,
)


def _pair_key(
    left: RecordId, right: RecordId, *, directed: bool
) -> tuple[RecordId, RecordId] | frozenset[RecordId]:
    """D1 comparison key: ordered for ``link`` (directed), unordered for
    ``dedupe`` (undirected, ``frozenset``)."""
    return (left, right) if directed else frozenset((left, right))


def linkage_metrics(
    result: LinkageResult, *, gold: LabeledPairs, directed: bool = True
) -> LinkageMetrics:
    """Precision/recall/F1 over all candidate pairs against ``gold``.

    Errored pairs (a ``MatchError`` in ``result.errors``) are excluded from
    tp/fp/fn and surfaced as ``LinkageMetrics.n_errors``. Every ``gold`` pair
    not predicted a match counts as a false negative — including gold pairs the
    blocker never surfaced — so recall is honest end-to-end. (A gold pair that
    *errored* is excluded rather than charged as a false negative.)

    Pair identity (D1): with ``directed=True`` (the default, for ``link``)
    pairs compare by order as ``(left_id, right_id)``; with ``directed=False``
    (for ``dedupe``) both sides canonicalize to an unordered key
    (``frozenset({a, b})``), since left/right is arbitrary within one source.
    The verb is not recoverable from ``result`` alone, so the caller supplies
    it — ``dedupe`` callers pass ``directed=False``.
    """
    gold_keys = {_pair_key(a, b, directed=directed) for a, b in gold.pairs}
    predicted = {
        _pair_key(pair.record_a.id, pair.record_b.id, directed=directed)
        for pair, decision in result.decisions
        if decision.is_match
    }
    errored = {
        _pair_key(pair.record_a.id, pair.record_b.id, directed=directed)
        for pair, _ in result.errors
    }
    return LinkageMetrics(
        true_positive=len(gold_keys & predicted),
        false_positive=len(predicted - gold_keys),
        false_negative=len(gold_keys - predicted - errored),
        n_gold=len(gold.pairs),
        n_errors=len(result.errors),
    )


def blocking_metrics(
    candidates: Sequence[CandidatePair],
    *,
    gold: LabeledPairs,
    ks: Sequence[int],
    directed: bool = True,
) -> BlockingMetrics:
    """Pair-completeness@k for each k in ``ks``.

    Pair identity (D1): same rule as ``linkage_metrics`` — ``directed=True``
    (``link``) compares ordered; ``directed=False`` (``dedupe``) canonicalizes
    to an unordered key.
    """
    ...


def pair_completeness_at_k(
    candidates: Sequence[CandidatePair],
    *,
    gold: LabeledPairs,
    k: int,
    directed: bool = True,
) -> float:
    """Single-k pair-completeness (kwarg ``gold``, consistent with the other
    metrics). Pair identity (D1): same rule as ``linkage_metrics``; pass
    ``directed=False`` for ``dedupe`` candidates."""

    ...


def clustering_metrics(
    clustering: Clustering, *, gold: LabeledPairs
) -> ClusteringMetrics:
    """B³ clustering quality of ``clustering`` against ``gold``.

    ``gold`` pairs are treated as edges and closed transitively into gold
    clusters, so the same ``LabeledPairs`` used for ``linkage_metrics`` scores
    clustering too — one gold type everywhere (kwarg ``gold``, consistent with
    the other metrics). Records present in ``clustering`` but in no gold pair
    are singleton gold clusters. Returns B³ precision/recall/F1; see
    :class:`denselinkage.core.results.ClusteringMetrics`.
    """
    ...


__all__ = [
    "blocking_metrics",
    "clustering_metrics",
    "linkage_metrics",
    "pair_completeness_at_k",
]
