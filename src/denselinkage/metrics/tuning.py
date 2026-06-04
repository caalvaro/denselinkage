"""Threshold tuning — the ``ThresholdSweep`` report. Its producer
(``tune_threshold``) lands here in Phase B."""

from dataclasses import dataclass

from denselinkage.metrics.linkage import LinkageMetrics


@dataclass(frozen=True, slots=True)
class ThresholdSweep:
    """Full precision/recall/F1 curve over a threshold grid. Typed accessor,
    never a bare ``{threshold: f1}`` dict."""

    rows: tuple[tuple[float, LinkageMetrics], ...]

    def best_f1(self) -> tuple[float, LinkageMetrics]:
        """The ``(threshold, metrics)`` row with the highest F1 (ties broken by
        the higher threshold). Raises ``ValueError`` on an empty sweep."""
        if not self.rows:
            raise ValueError("empty threshold sweep")
        return max(self.rows, key=lambda row: (row[1].f1, row[0]))

    def at_recall(self, target: float) -> tuple[float, LinkageMetrics]:
        """Among rows reaching ``recall >= target``, the highest-F1 row (ties
        broken by the higher threshold). Raises ``ValueError`` if no threshold
        reaches the target recall."""
        eligible = [row for row in self.rows if row[1].recall >= target]
        if not eligible:
            raise ValueError(f"no threshold reaches recall >= {target}")
        return max(eligible, key=lambda row: (row[1].f1, row[0]))
