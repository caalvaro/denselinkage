"""Blocking-quality metrics — the ``BlockingMetrics`` report and the
``blocking_metrics`` / ``pair_completeness_at_k`` functions over candidate
pairs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from denselinkage.core.models import CandidatePair
from denselinkage.core.results import LabeledPairs


@dataclass(frozen=True, slots=True)
class BlockingMetrics:
    """Pair-completeness@k. ``pc_at(k)`` is the sole supported accessor;
    construct via :meth:`from_pc_map` (no leading-underscore public
    constructor param)."""

    pc: Mapping[int, float]
    n_gold: int

    @classmethod
    def from_pc_map(
        cls, pc: Mapping[int, float], *, n_gold: int
    ) -> "BlockingMetrics": ...

    def pc_at(self, k: int) -> float:
        """PC@k. Raises ``KeyError`` if ``k`` was not among the ``ks`` passed
        to ``blocking_metrics`` (no silent 0.0 — an uncomputed k is a usage
        error, not a zero result)."""
        ...


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
