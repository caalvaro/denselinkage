"""Blocking-adjusted metrics — the ``AdjustedMetrics`` report and its
``adjusted_metrics`` producer."""

from collections.abc import Sequence
from dataclasses import dataclass

from denselinkage.core.models import CandidatePair
from denselinkage.core.results import LabeledPairs, LinkageResult
from denselinkage.metrics._pairing import pair_key
from denselinkage.metrics.blocking import pair_completeness_at_k
from denselinkage.metrics.linkage import LinkageMetrics, linkage_metrics


@dataclass(frozen=True, slots=True)
class AdjustedMetrics:
    """End-to-end honest number: matcher metrics adjusted by blocker
    pair-completeness@k (``recall_adjusted = matcher.recall * pc@k``)."""

    matcher: LinkageMetrics
    blocking_recall_at_k: float
    k: int

    @property
    def recall_adjusted(self) -> float:
        """Matcher recall scaled by the blocker's pair-completeness@k — the
        recall ceiling the blocker imposes on the matcher."""
        return self.matcher.recall * self.blocking_recall_at_k

    @property
    def f1_adjusted(self) -> float:
        """Harmonic mean of matcher precision and :attr:`recall_adjusted`."""
        precision = self.matcher.precision
        denom = precision + self.recall_adjusted
        return 2 * precision * self.recall_adjusted / denom if denom else 0.0


def adjusted_metrics(
    result: LinkageResult,
    candidates: Sequence[CandidatePair],
    *,
    gold: LabeledPairs,
    k: int,
    directed: bool = True,
) -> AdjustedMetrics:
    """Decompose end-to-end recall into matcher x blocker components.

    ``blocking_recall_at_k`` is the blocker's pair-completeness@k over
    ``candidates``. ``matcher`` is the matcher's metrics measured *conditionally
    on blocking* — recall over only the gold pairs the candidate set surfaced —
    so ``recall_adjusted = matcher.recall * pc@k`` is the honest end-to-end
    recall, not a double-counting of the blocking loss (measuring the matcher
    against the full gold would already fold blocking misses in as false
    negatives). Precision is unaffected by the conditioning. When the matched
    candidate set is the top-k set used for ``pc@k``, ``recall_adjusted`` equals
    ``linkage_metrics(result, gold=gold).recall`` exactly. ``directed`` follows
    ``linkage_metrics`` / ``pair_completeness_at_k``.
    """
    pc = pair_completeness_at_k(candidates, gold=gold, k=k, directed=directed)
    surfaced = {
        pair_key(pair.record_a.id, pair.record_b.id, directed=directed)
        for pair, _ in result.decisions
    }
    surfaced |= {
        pair_key(pair.record_a.id, pair.record_b.id, directed=directed)
        for pair, _ in result.errors
    }
    surfaced_gold = LabeledPairs.from_pairs(
        (left, right)
        for left, right in gold.pairs
        if pair_key(left, right, directed=directed) in surfaced
    )
    matcher = linkage_metrics(result, gold=surfaced_gold, directed=directed)
    return AdjustedMetrics(matcher=matcher, blocking_recall_at_k=pc, k=k)
