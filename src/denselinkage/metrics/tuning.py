"""Threshold tuning — the ``ThresholdSweep`` report. Its producer
(``tune_threshold``) lands here in Phase B."""

from dataclasses import dataclass

from denselinkage.metrics.linkage import LinkageMetrics


@dataclass(frozen=True, slots=True)
class ThresholdSweep:
    """Full precision/recall/F1 curve over a threshold grid. Typed accessor,
    never a bare ``{threshold: f1}`` dict."""

    rows: tuple[tuple[float, LinkageMetrics], ...]

    def best_f1(self) -> tuple[float, LinkageMetrics]: ...

    def at_recall(self, target: float) -> tuple[float, LinkageMetrics]: ...
