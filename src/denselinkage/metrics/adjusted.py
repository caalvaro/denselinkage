"""Blocking-adjusted metrics — the ``AdjustedMetrics`` report. Its producer
(``adjusted_metrics``) lands here in Phase B."""

from dataclasses import dataclass

from denselinkage.metrics.linkage import LinkageMetrics


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
