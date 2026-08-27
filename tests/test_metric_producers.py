"""Behavioural tests for the B2 metric producers: ``tune_threshold`` (the
P/R/F1 curve over a similarity grid) and ``adjusted_metrics`` (the matcher x
blocker recall decomposition)."""

import math
import random
from collections.abc import Sequence

import pytest

from denselinkage import LabeledPairs, LinkageResult
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.metrics import adjusted_metrics, linkage_metrics, tune_threshold
from denselinkage.metrics._pairing import PairKey, pair_key
from denselinkage.metrics.linkage import LinkageMetrics, _metrics_from_keys


def _pair(a: str, b: str, score: float | None) -> CandidatePair:
    return CandidatePair(Record(a, ""), Record(b, ""), similarity_score=score)


def _rescan_rows(
    candidates: Sequence[CandidatePair],
    *,
    gold: LabeledPairs,
    thresholds: Sequence[float] | None = None,
    directed: bool = True,
) -> tuple[tuple[float, LinkageMetrics], ...]:
    """The pre-sweep ``tune_threshold`` body: one full rescan of the candidates
    per cut point. It calls ``_metrics_from_keys`` instead of restating the
    tp/fp/fn arithmetic, which is what keeps that helper authoritative now that
    the producer maintains those counts incrementally. Inline the formula here
    and an edit to the helper stops being checked against the producer."""
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
    return tuple(
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


def _seeded_case(seed: int) -> tuple[list[CandidatePair], LabeledPairs]:
    """400 candidates over a 30x30 id space, so pair keys repeat at different
    scores; scores rounded to two places, so exact ties are frequent; 8 percent
    of them ``None``, so every case carries errored keys and some share a key
    with a scored candidate."""
    rng = random.Random(seed)
    candidates = [
        _pair(
            f"a{rng.randrange(30)}",
            f"b{rng.randrange(30)}",
            None if rng.random() < 0.08 else round(rng.random(), 2),
        )
        for _ in range(400)
    ]
    gold = LabeledPairs.from_pairs(
        [(f"a{rng.randrange(30)}", f"b{rng.randrange(30)}") for _ in range(80)]
    )
    return candidates, gold


_COMPARISONS = [0]


class _CountingScore(float):
    """A score that tallies every ordering comparison it takes part in, so a
    test can bound the work ``tune_threshold`` does rather than time it.
    ``CandidatePair`` stores ``similarity_score`` without coercion, so the
    subclass survives into the producer; a body that called ``float()`` on it
    would silently zero the tally, which is what the lower bound catches."""

    __slots__ = ()

    def __ge__(self, other: float) -> bool:
        _COMPARISONS[0] += 1
        return float.__ge__(self, other)

    def __gt__(self, other: float) -> bool:
        _COMPARISONS[0] += 1
        return float.__gt__(self, other)

    def __le__(self, other: float) -> bool:
        _COMPARISONS[0] += 1
        return float.__le__(self, other)

    def __lt__(self, other: float) -> bool:
        _COMPARISONS[0] += 1
        return float.__lt__(self, other)


def _counted_case(n: int) -> tuple[list[CandidatePair], LabeledPairs]:
    """``n`` candidates whose scores tally their own comparisons, over an
    ``n/2`` id pool so keys repeat."""
    rng = random.Random(7)
    pool = n // 2
    candidates = [
        CandidatePair(
            Record(f"a{rng.randrange(pool)}", ""),
            Record(f"b{rng.randrange(pool)}", ""),
            similarity_score=_CountingScore(rng.random()),
        )
        for _ in range(n)
    ]
    gold = LabeledPairs.from_pairs(
        [(f"a{rng.randrange(pool)}", f"b{rng.randrange(pool)}") for _ in range(n // 5)]
    )
    return candidates, gold


# --- tune_threshold ---------------------------------------------------------


def test_tune_threshold_sweeps_distinct_scores_by_default() -> None:
    candidates = [
        _pair("A1", "B1", 0.9),  # gold
        _pair("A2", "B2", 0.6),  # gold
        _pair("A3", "B3", 0.3),  # non-gold
    ]
    gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2")])
    sweep = tune_threshold(candidates, gold=gold)

    assert [t for t, _ in sweep.rows] == [0.3, 0.6, 0.9]
    m03, m06, m09 = (m for _, m in sweep.rows)
    assert (m03.true_positive, m03.false_positive, m03.false_negative) == (2, 1, 0)
    assert (m06.true_positive, m06.false_positive, m06.false_negative) == (2, 0, 0)
    assert (m09.true_positive, m09.false_positive, m09.false_negative) == (1, 0, 1)


def test_tune_threshold_best_f1_and_at_recall() -> None:
    candidates = [
        _pair("A1", "B1", 0.9),
        _pair("A2", "B2", 0.6),
        _pair("A3", "B3", 0.3),
    ]
    gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2")])
    sweep = tune_threshold(candidates, gold=gold)

    t_best, m_best = sweep.best_f1()
    assert t_best == 0.6
    assert (m_best.precision, m_best.recall, m_best.f1) == (1.0, 1.0, 1.0)
    # At full recall, the 0.6 cut (P=R=1) beats the 0.3 cut (P=2/3).
    assert sweep.at_recall(1.0)[0] == 0.6


def test_tune_threshold_honours_explicit_grid() -> None:
    candidates = [_pair("A1", "B1", 0.9), _pair("A2", "B2", 0.6)]
    gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2")])
    sweep = tune_threshold(candidates, gold=gold, thresholds=[0.95, 0.5])
    assert [t for t, _ in sweep.rows] == [0.5, 0.95]  # sorted ascending
    rows = dict(sweep.rows)
    assert rows[0.5].true_positive == 2  # both clear 0.5
    assert rows[0.95].true_positive == 0  # neither clears 0.95


def test_tune_threshold_directed_false_canonicalizes() -> None:
    candidates = [_pair("1", "2", 0.9)]
    gold = LabeledPairs.from_pairs([("2", "1")])  # reversed order
    _, m = tune_threshold(candidates, gold=gold, directed=False).best_f1()
    assert m.true_positive == 1


def test_tune_threshold_unscored_pair_is_an_error_at_every_threshold() -> None:
    candidates = [_pair("A1", "B1", 0.9), _pair("A2", "B2", None)]
    gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2")])
    [(threshold, m)] = tune_threshold(candidates, gold=gold).rows
    assert threshold == 0.9
    assert m.n_errors == 1
    assert m.true_positive == 1
    assert m.false_negative == 0  # the errored gold pair is excluded, not charged


def test_tune_threshold_empty_candidates_gives_empty_sweep() -> None:
    sweep = tune_threshold([], gold=LabeledPairs.from_pairs([("A1", "B1")]))
    assert sweep.rows == ()
    with pytest.raises(ValueError, match="empty threshold sweep"):
        sweep.best_f1()


def test_tune_threshold_empty_explicit_grid_gives_empty_sweep() -> None:
    candidates = [_pair("A1", "B1", 0.9)]
    sweep = tune_threshold(
        candidates, gold=LabeledPairs.from_pairs([("A1", "B1")]), thresholds=[]
    )
    assert sweep.rows == ()


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize(
    "thresholds", [None, [0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 2.0, -1.0]]
)
def test_tune_threshold_matches_a_full_rescan(
    seed: int, directed: bool, thresholds: list[float] | None
) -> None:
    candidates, gold = _seeded_case(seed)
    expected = _rescan_rows(
        candidates, gold=gold, thresholds=thresholds, directed=directed
    )
    actual = tune_threshold(
        candidates, gold=gold, thresholds=thresholds, directed=directed
    ).rows
    assert actual == expected


def test_tune_threshold_ties_cross_together_at_the_shared_score() -> None:
    candidates = [
        _pair("A1", "B1", 0.5),  # gold
        _pair("A2", "B2", 0.5),  # gold
        _pair("A3", "B3", 0.5),  # non-gold
        _pair("A4", "B4", 0.2),  # non-gold
    ]
    gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2")])
    # One cut per distinct score: the three at 0.5 collapse to a single row.
    assert [t for t, _ in tune_threshold(candidates, gold=gold).rows] == [0.2, 0.5]

    # `>=` is inclusive, so a cut equal to a score admits every candidate at it,
    # and the next representable double above admits none.
    rows = dict(
        tune_threshold(
            candidates,
            gold=gold,
            thresholds=[0.5, math.nextafter(0.5, 1.0), math.nextafter(0.5, 0.0)],
        ).rows
    )
    assert (rows[0.5].true_positive, rows[0.5].false_positive) == (2, 1)
    assert rows[math.nextafter(0.5, 0.0)].true_positive == 2
    above = rows[math.nextafter(0.5, 1.0)]
    assert (above.true_positive, above.false_positive) == (0, 0)
    assert above.false_negative == 2


def test_tune_threshold_counts_a_repeated_key_once_at_its_best_score() -> None:
    candidates = [
        _pair("A1", "B1", 0.3),  # gold, lower score arrives first
        _pair("A1", "B1", 0.9),
        _pair("A2", "B2", 0.9),  # non-gold, higher score arrives first
        _pair("A2", "B2", 0.3),
    ]
    gold = LabeledPairs.from_pairs([("A1", "B1")])
    rows = dict(tune_threshold(candidates, gold=gold, thresholds=[0.3, 0.9]).rows)
    for threshold in (0.3, 0.9):
        m = rows[threshold]
        assert (m.true_positive, m.false_positive) == (1, 1)
    assert rows[0.9].false_negative == 0


def test_tune_threshold_errored_and_scored_key_is_never_a_false_negative() -> None:
    candidates = [_pair("A1", "B1", None), _pair("A1", "B1", 0.9)]
    gold = LabeledPairs.from_pairs([("A1", "B1")])
    rows = dict(tune_threshold(candidates, gold=gold, thresholds=[0.5, 0.95]).rows)
    assert (rows[0.5].true_positive, rows[0.5].false_negative) == (1, 0)
    # Above its score the key stops being a true positive but is still errored,
    # so it is excluded from false negatives rather than charged.
    assert (rows[0.95].true_positive, rows[0.95].false_negative) == (0, 0)
    assert rows[0.5].n_errors == rows[0.95].n_errors == 1


def test_tune_threshold_nan_score_clears_no_threshold() -> None:
    candidates = [_pair("A1", "B1", 0.9), _pair("A2", "B2", float("nan"))]
    gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2")])
    rows = dict(tune_threshold(candidates, gold=gold, thresholds=[-1.0, 0.9]).rows)
    # A NaN score is unmatchable, not undecidable: it clears no cut even below
    # every finite score, and it is charged a false negative rather than counted
    # in n_errors, which only `similarity_score is None` reaches.
    assert (rows[-1.0].true_positive, rows[-1.0].false_negative) == (1, 1)
    assert rows[-1.0].n_errors == 0


def test_tune_threshold_nan_cut_point_predicts_nothing() -> None:
    candidates = [_pair("A1", "B1", 0.5), _pair("A2", "B2", 0.9)]
    gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2")])
    # An explicit grid keeps `sorted()`'s placement of the NaN, which is a pure
    # function of the caller's input order. The default grid's placement depends
    # on set iteration order and is not asserted anywhere.
    rows = tune_threshold(
        candidates, gold=gold, thresholds=[0.1, float("nan"), 0.9]
    ).rows
    assert [math.isnan(t) for t, _ in rows] == [False, True, False]
    nan_metrics = rows[1][1]
    assert (
        nan_metrics.true_positive,
        nan_metrics.false_positive,
        nan_metrics.false_negative,
    ) == (0, 0, 2)


def test_tune_threshold_nan_score_does_not_starve_the_keys_behind_it() -> None:
    # A NaN sorts nowhere: `sorted` leaves it where the candidate order put it.
    # Admitted to the descending sweep it would sit at the head of the queue and
    # stall the cursor, and every key behind it would go unpredicted at every
    # cut. Holding it out is what keeps A2/B2 a true positive here.
    candidates = [_pair("A1", "B1", float("nan")), _pair("A2", "B2", 0.9)]
    gold = LabeledPairs.from_pairs([("A2", "B2")])
    [(_, m)] = tune_threshold(candidates, gold=gold, thresholds=[0.5]).rows
    assert (m.true_positive, m.false_negative) == (1, 0)


def test_tune_threshold_nan_score_still_contributes_a_default_grid_cut() -> None:
    # The default grid is every distinct score, and a NaN is one of them: it
    # clears no cut but it is still a cut, so holding it out of the sweep must
    # not drop its row. Its position depends on set iteration order, so only the
    # multiset of rows is asserted.
    candidates = [_pair("A1", "B1", 0.9), _pair("A2", "B2", float("nan"))]
    gold = LabeledPairs.from_pairs([("A1", "B1")])
    rows = tune_threshold(candidates, gold=gold).rows
    assert sorted(math.isnan(t) for t, _ in rows) == [False, True]


def test_tune_threshold_nan_cut_does_not_disorder_the_finite_cuts() -> None:
    # `sorted` cannot order a grid holding a NaN: [0.9, nan, 0.1] comes back
    # unchanged, so grid position stops tracking cut value. The sweep visits the
    # finite cuts by value; visiting them by position would run 0.1 first and
    # leave the 0.9 row showing the admissions the 0.1 cut made.
    candidates = [_pair("A1", "B1", 0.9), _pair("A2", "B2", 0.2)]
    gold = LabeledPairs.from_pairs([("A1", "B1")])
    rows = tune_threshold(
        candidates, gold=gold, thresholds=[0.9, float("nan"), 0.1]
    ).rows
    assert [rows[0][0], rows[2][0]] == [0.9, 0.1]
    assert (rows[0][1].true_positive, rows[0][1].false_positive) == (1, 0)
    assert (rows[2][1].true_positive, rows[2][1].false_positive) == (1, 1)


def test_tune_threshold_explicit_grid_keeps_duplicates_and_out_of_range_cuts() -> None:
    candidates = [_pair("A1", "B1", 0.9), _pair("A2", "B2", 0.6)]
    gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2")])
    rows = tune_threshold(candidates, gold=gold, thresholds=[0.7, 0.7, 2.0, -1.0]).rows
    assert [t for t, _ in rows] == [-1.0, 0.7, 0.7, 2.0]  # sorted, not deduplicated
    assert rows[1] == rows[2]  # a repeated cut re-emits the same row
    assert rows[0][1].true_positive == 2  # below every score: all predicted
    assert (rows[3][1].true_positive, rows[3][1].false_negative) == (0, 2)


def test_tune_threshold_undirected_self_pair_uses_a_one_element_key() -> None:
    candidates = [_pair("a", "a", 0.9), _pair("a", "a", 0.2)]
    gold = LabeledPairs.from_pairs([("a", "a")])
    [(_, m)] = tune_threshold(
        candidates, gold=gold, thresholds=[0.5], directed=False
    ).rows
    assert (m.true_positive, m.false_positive) == (1, 0)


def test_tune_threshold_compares_log_linearly_in_the_candidate_count() -> None:
    n = 800
    candidates, gold = _counted_case(n)
    _COMPARISONS[0] = 0
    tune_threshold(candidates, gold=gold)
    # Upper bound: the producer sorts and sweeps rather than rescanning, so its
    # comparison count is O(n log n). Lower bound: it must actually compare the
    # scores it was given, so a body that coerced them to plain floats (losing
    # the tally) fails here instead of passing the upper bound vacuously.
    assert n <= _COMPARISONS[0] <= 10 * n * math.log2(n)


def test_rescan_reference_exceeds_the_log_linear_comparison_bound() -> None:
    n = 800
    candidates, gold = _counted_case(n)
    _COMPARISONS[0] = 0
    _rescan_rows(candidates, gold=gold)
    # Keeps the bound above honest: it has to separate the two implementations,
    # not merely be loose enough for both.
    assert _COMPARISONS[0] > 10 * n * math.log2(n)


# --- adjusted_metrics -------------------------------------------------------


def test_adjusted_metrics_decomposes_recall_conditionally() -> None:
    # Blocker surfaced only one of two gold pairs; matcher is perfect on it.
    gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2")])
    candidates = [_pair("A1", "B1", 0.9)]
    result = LinkageResult(decisions=((candidates[0], MatchDecision(is_match=True)),))
    adj = adjusted_metrics(result, candidates, gold=gold, k=1)

    assert adj.k == 1
    assert adj.blocking_recall_at_k == 0.5  # blocker recalled 1/2 gold
    assert adj.matcher.recall == 1.0  # conditional: perfect on what was surfaced
    assert adj.recall_adjusted == 0.5  # 1.0 * 0.5 = honest end-to-end
    # Without conditioning, the matcher's recall over the full gold would be 0.5;
    # multiplying that by pc@k would double-count the blocking loss.
    assert linkage_metrics(result, gold=gold).recall == 0.5


def test_adjusted_metrics_precision_unaffected_and_f1() -> None:
    gold = LabeledPairs.from_pairs([("A1", "B1")])
    candidates = [_pair("A1", "B1", 0.9), _pair("A3", "B1", 0.8)]
    result = LinkageResult(
        decisions=(
            (candidates[0], MatchDecision(is_match=True)),
            (candidates[1], MatchDecision(is_match=True)),  # false positive
        )
    )
    adj = adjusted_metrics(result, candidates, gold=gold, k=2)
    assert adj.matcher.precision == 0.5
    assert adj.blocking_recall_at_k == 1.0
    assert adj.recall_adjusted == 1.0
    assert adj.f1_adjusted == pytest.approx(2 / 3)


def test_adjusted_recall_equals_honest_when_topk_is_matched_set() -> None:
    # Both gold pairs surfaced within top-k; the matcher misses one.
    gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2")])
    candidates = [_pair("A1", "B1", 0.9), _pair("A2", "B2", 0.7)]
    result = LinkageResult(
        decisions=(
            (candidates[0], MatchDecision(is_match=True)),
            (candidates[1], MatchDecision(is_match=False)),
        )
    )
    adj = adjusted_metrics(result, candidates, gold=gold, k=2)
    assert adj.recall_adjusted == linkage_metrics(result, gold=gold).recall == 0.5


def test_adjusted_metrics_directed_false() -> None:
    gold = LabeledPairs.from_pairs([("2", "1")])
    candidates = [_pair("1", "2", 0.9)]
    result = LinkageResult(decisions=((candidates[0], MatchDecision(is_match=True)),))
    adj = adjusted_metrics(result, candidates, gold=gold, k=1, directed=False)
    assert adj.matcher.recall == 1.0
    assert adj.recall_adjusted == 1.0


def test_adjusted_metrics_counts_errored_surface() -> None:
    gold = LabeledPairs.from_pairs([("A1", "B1")])
    candidates = [_pair("A1", "B1", None)]
    result = LinkageResult(
        decisions=(),
        errors=((candidates[0], MatchError(reason="undecidable")),),
    )
    adj = adjusted_metrics(result, candidates, gold=gold, k=1)
    assert adj.matcher.n_errors == 1
    assert adj.matcher.recall == 0.0
    assert adj.recall_adjusted == 0.0
