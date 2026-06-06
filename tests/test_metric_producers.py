"""Behavioural tests for the B2 metric producers: ``tune_threshold`` (the
P/R/F1 curve over a similarity grid) and ``adjusted_metrics`` (the matcher x
blocker recall decomposition)."""

import pytest

from denselinkage import LabeledPairs, LinkageResult
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.metrics import adjusted_metrics, linkage_metrics, tune_threshold


def _pair(a: str, b: str, score: float | None) -> CandidatePair:
    return CandidatePair(Record(a, ""), Record(b, ""), similarity_score=score)


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
