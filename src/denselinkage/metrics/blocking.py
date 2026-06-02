"""Blocking-quality metrics — pair-completeness@k over candidate pairs."""

from collections.abc import Sequence

from denselinkage.core.models import CandidatePair
from denselinkage.core.results import BlockingMetrics, LabeledPairs


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
