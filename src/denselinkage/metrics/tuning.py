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


def _unorderable(value: float) -> bool:
    """True for a value no ordering can place: a NaN, which compares unequal to
    itself and satisfies no ``>=``. Such a score is never predicted and such a
    cut point predicts nothing, so both are held out of the descending sweep;
    admitting one would stall the cursor and starve every lower cut."""
    return value != value


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

    Pair identity is by comparison key, not by candidate: a key carried by
    several candidates is predicted from the highest score among them and
    counts once in tp/fp. Cost is ``O((n + g) log (n + g))`` for ``n``
    candidates and ``g`` cut points: one descending pass admits each key once,
    rather than rescanning every candidate per cut point.
    """
    gold_keys = {pair_key(left, right, directed=directed) for left, right in gold.pairs}
    scores: set[float] = set()
    best: dict[PairKey, float] = {}
    errored: set[PairKey] = set()
    n_errors = 0
    for pair in candidates:
        key = pair_key(pair.record_a.id, pair.record_b.id, directed=directed)
        score = pair.similarity_score
        if score is None:
            errored.add(key)
            n_errors += 1
            continue
        # Recorded before the NaN filter: a NaN score still contributes its own
        # default-grid cut point; it just never clears one.
        scores.add(score)
        if _unorderable(score):
            continue
        previous = best.get(key)
        if previous is None or score > previous:
            best[key] = score
    grid = sorted(scores) if thresholds is None else sorted(thresholds)
    # The row for a cut nothing clears. It fixes every row's ``n_gold`` and
    # ``n_errors``, and its ``false_negative`` is the base |gold - errored| the
    # sweep subtracts recovered gold keys from, so the shared key-set formula
    # still determines all three.
    unpredicted = _metrics_from_keys(
        gold_keys=gold_keys,
        predicted=set(),
        errored=errored,
        n_gold=len(gold.pairs),
        n_errors=n_errors,
    )
    metrics = [unpredicted] * len(grid)
    # Sorted on the score alone: the default tuple order would fall through to
    # comparing ``PairKey``s on a tie, and a ``frozenset`` key
    # (``directed=False``) is only partially ordered.
    admissions = sorted(best.items(), key=lambda entry: entry[1], reverse=True)
    cursor = 0
    true_positive = 0
    false_positive = 0
    recovered = 0
    # Cut points are visited high to low and their rows written back by
    # position, so the emitted order stays whatever ``grid`` says, a NaN cut
    # included: it keeps the ``unpredicted`` row it is entitled to.
    for position in sorted(
        (index for index, cut in enumerate(grid) if not _unorderable(cut)),
        key=lambda index: grid[index],
        reverse=True,
    ):
        threshold = grid[position]
        # Drain the whole tie group before emitting, so every key sharing a
        # score crosses together; the comparison uses the raw cut point, only
        # the emitted label is float-coerced.
        while cursor < len(admissions) and admissions[cursor][1] >= threshold:
            key = admissions[cursor][0]
            cursor += 1
            if key in gold_keys:
                true_positive += 1
                if key not in errored:
                    recovered += 1
            else:
                false_positive += 1
        metrics[position] = LinkageMetrics(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=unpredicted.false_negative - recovered,
            n_gold=unpredicted.n_gold,
            n_errors=unpredicted.n_errors,
        )
    return ThresholdSweep(
        rows=tuple((float(cut), row) for cut, row in zip(grid, metrics, strict=True))
    )
