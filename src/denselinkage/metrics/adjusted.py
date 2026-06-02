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
    def recall_adjusted(self) -> float: ...

    @property
    def f1_adjusted(self) -> float: ...
