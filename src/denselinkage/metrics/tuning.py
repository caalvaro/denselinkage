"""Threshold tuning — the ``ThresholdSweep`` report and its ``tune_threshold``
producer."""

from collections.abc import Sequence
from dataclasses import dataclass

from denselinkage.core.models import CandidatePair
from denselinkage.core.results import LabeledPairs
from denselinkage.metrics._pairing import PairKey, pair_key
from denselinkage.metrics.linkage import LinkageMetrics, _metrics_from_keys


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


def tune_threshold(
    candidates: Sequence[CandidatePair],
    *,
    gold: LabeledPairs,
    thresholds: Sequence[float] | None = None,
    directed: bool = True,
) -> ThresholdSweep:
    """Sweep a similarity threshold over scored ``candidates`` and return the
    full precision/recall/F1 curve as a ``ThresholdSweep``.

    Each candidate is classified a match iff ``similarity_score >= t``
    (inclusive, matching ``ThresholdMatcher``). A candidate with
    ``similarity_score is None`` is undecidable and counts as an error at every
    threshold — excluded from false negatives and surfaced as ``n_errors`` —
    the same accounting ``linkage_metrics`` uses. ``directed`` follows
    ``linkage_metrics`` (pass ``directed=False`` for ``dedupe`` candidates).

    ``thresholds`` defaults to the sorted distinct candidate scores — the only
    cut points that change the prediction, so the curve is exact and
    ``ThresholdSweep.best_f1`` finds the true optimum; pass an explicit grid to
    override. An empty candidate set (or empty grid) yields an empty sweep.
    """
    gold_keys = {pair_key(left, right, directed=directed) for left, right in gold.pairs}
    scored: list[tuple[PairKey, float]] = []
    errored: set[PairKey] = set()
    n_errors = 0
    for pair in candidates:
        key = pair_key(pair.record_a.id, pair.record_b.id, directed=directed)
        if pair.similarity_score is None:
            errored.add(key)
            n_errors += 1
        else:
            scored.append((key, pair.similarity_score))
    grid = (
        sorted({score for _, score in scored})
        if thresholds is None
        else sorted(thresholds)
    )
    rows = tuple(
        (
            float(threshold),
            _metrics_from_keys(
                gold_keys=gold_keys,
                predicted={key for key, score in scored if score >= threshold},
                errored=errored,
                n_gold=len(gold.pairs),
                n_errors=n_errors,
            ),
        )
        for threshold in grid
    )
    return ThresholdSweep(rows=rows)
