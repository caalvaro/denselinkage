"""Pairwise linkage metrics — the ``LinkageMetrics`` report and the
``linkage_metrics`` function that produces it."""

from dataclasses import dataclass

from denselinkage.core.results import LabeledPairs, LinkageResult
from denselinkage.metrics._pairing import PairKey, pair_key


@dataclass(frozen=True, slots=True)
class LinkageMetrics:
    """Contract: pairs that errored (a ``MatchError`` in
    ``LinkageResult.errors``) are excluded from tp/fp/fn and reported as
    ``n_errors``. ``false_negative`` counts every gold pair not predicted a
    match — including gold pairs the blocker never surfaced — so recall is
    honest end-to-end, not conditional on blocking."""

    true_positive: int
    false_positive: int
    false_negative: int
    n_gold: int
    n_errors: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0


def _metrics_from_keys(
    *,
    gold_keys: set[PairKey],
    predicted: set[PairKey],
    errored: set[PairKey],
    n_gold: int,
    n_errors: int,
) -> LinkageMetrics:
    """Assemble ``LinkageMetrics`` from comparison-key sets — the shared tp/fp/fn
    formula behind both ``linkage_metrics`` and ``tune_threshold``. ``errored``
    keys are excluded from false negatives (recall ignores pairs the matcher
    could not decide); ``n_errors`` is the raw errored-pair count."""
    return LinkageMetrics(
        true_positive=len(gold_keys & predicted),
        false_positive=len(predicted - gold_keys),
        false_negative=len(gold_keys - predicted - errored),
        n_gold=n_gold,
        n_errors=n_errors,
    )


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
    gold_keys = {pair_key(a, b, directed=directed) for a, b in gold.pairs}
    predicted = {
        pair_key(pair.record_a.id, pair.record_b.id, directed=directed)
        for pair, decision in result.decisions
        if decision.is_match
    }
    errored = {
        pair_key(pair.record_a.id, pair.record_b.id, directed=directed)
        for pair, _ in result.errors
    }
    return _metrics_from_keys(
        gold_keys=gold_keys,
        predicted=predicted,
        errored=errored,
        n_gold=len(gold.pairs),
        n_errors=len(result.errors),
    )
