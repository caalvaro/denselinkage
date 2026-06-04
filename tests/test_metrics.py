"""Unit tests for ``linkage_metrics`` — the honest, provenance-preserving
counting the contract promises (the ``directed`` flag is covered in
``test_quickstart_end_to_end``)."""

import pandas as pd
import pytest

from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.core.results import ClusteringResult, LabeledPairs, LinkageResult
from denselinkage.metrics import (
    AdjustedMetrics,
    LinkageMetrics,
    ThresholdSweep,
    blocking_metrics,
    clustering_metrics,
    linkage_metrics,
    pair_completeness_at_k,
)


def _pair(left: str, right: str) -> CandidatePair:
    return CandidatePair(Record(left, ""), Record(right, ""), similarity_score=1.0)


def _match(
    left: str, right: str, *, is_match: bool = True
) -> tuple[CandidatePair, MatchDecision]:
    return (_pair(left, right), MatchDecision(is_match=is_match))


def test_missed_gold_pair_counts_as_a_false_negative() -> None:
    # End-to-end recall is honest: a gold pair the blocker never surfaced is a FN.
    gold = LabeledPairs.from_pairs([("A", "B"), ("C", "D")])
    result = LinkageResult(decisions=(_match("A", "B"),))  # C-D never appears
    m = linkage_metrics(result, gold=gold)
    assert (m.true_positive, m.false_positive, m.false_negative) == (1, 0, 1)
    assert m.recall == 0.5


def test_errored_gold_pair_is_excluded_not_charged_as_false_negative() -> None:
    gold = LabeledPairs.from_pairs([("A", "B")])
    result = LinkageResult(
        decisions=(), errors=((_pair("A", "B"), MatchError(reason="boom")),)
    )
    m = linkage_metrics(result, gold=gold)
    assert (m.true_positive, m.false_positive, m.false_negative) == (0, 0, 0)
    assert m.n_errors == 1


def test_non_match_decision_is_not_a_prediction() -> None:
    gold = LabeledPairs.from_pairs([("A", "B")])
    result = LinkageResult(decisions=(_match("A", "B", is_match=False),))
    m = linkage_metrics(result, gold=gold)
    assert (m.true_positive, m.false_negative) == (0, 1)


def test_spurious_match_is_a_false_positive() -> None:
    result = LinkageResult(decisions=(_match("A", "B"),))
    m = linkage_metrics(result, gold=LabeledPairs.from_pairs([]))
    assert (m.true_positive, m.false_positive) == (0, 1)


def test_empty_result_and_gold_give_zero_not_division_error() -> None:
    m = linkage_metrics(LinkageResult(decisions=()), gold=LabeledPairs.from_pairs([]))
    assert (m.precision, m.recall, m.f1) == (0.0, 0.0, 0.0)


def test_labeled_pairs_from_frame() -> None:
    df = pd.DataFrame({"l": ["A", "C"], "r": ["B", "D"]})
    gold = LabeledPairs.from_frame(df, left_id="l", right_id="r")
    assert gold.pairs == frozenset({("A", "B"), ("C", "D")})


# --- blocking_metrics / pair_completeness_at_k ------------------------------


def _cand(left: str, right: str, score: float) -> CandidatePair:
    # Orientation matches the dense blocker: record_a = indexed, record_b = query.
    return CandidatePair(Record(left, ""), Record(right, ""), similarity_score=score)


def test_blocking_metrics_pc_at_k_increases_with_k() -> None:
    # R1's true match L1 ranks 2nd (a distractor outranks it); R2's ranks 1st.
    candidates = [
        _cand("X", "R1", 0.9),  # distractor, not gold
        _cand("L1", "R1", 0.5),  # gold, rank 2 for R1
        _cand("L2", "R2", 0.8),  # gold, rank 1 for R2
    ]
    gold = LabeledPairs.from_pairs([("L1", "R1"), ("L2", "R2")])
    bm = blocking_metrics(candidates, gold=gold, ks=[1, 2])
    assert bm.pc_at(1) == 0.5
    assert bm.pc_at(2) == 1.0
    assert bm.n_gold == 2


def test_blocking_metrics_directed_vs_undirected() -> None:
    candidates = [_cand("A", "B", 0.9)]  # oriented (A, B); gold is reversed
    gold = LabeledPairs.from_pairs([("B", "A")])
    assert blocking_metrics(candidates, gold=gold, ks=[1]).pc_at(1) == 0.0
    undirected = blocking_metrics(candidates, gold=gold, ks=[1], directed=False)
    assert undirected.pc_at(1) == 1.0


def test_blocking_metrics_empty_gold_is_zero() -> None:
    bm = blocking_metrics(
        [_cand("A", "B", 0.9)], gold=LabeledPairs.from_pairs([]), ks=[1]
    )
    assert bm.pc_at(1) == 0.0


def test_blocking_metrics_pc_at_unswept_k_raises() -> None:
    gold = LabeledPairs.from_pairs([("A", "B")])
    bm = blocking_metrics([_cand("A", "B", 0.9)], gold=gold, ks=[1])
    with pytest.raises(KeyError):
        bm.pc_at(5)


def test_pair_completeness_at_k_matches_blocking_metrics() -> None:
    candidates = [_cand("L1", "R1", 0.9), _cand("L2", "R2", 0.8)]
    gold = LabeledPairs.from_pairs([("L1", "R1"), ("L2", "R2")])
    assert pair_completeness_at_k(candidates, gold=gold, k=1) == 1.0


def test_blocking_metrics_ks_order_independent() -> None:
    # ks passed out of order must give the same PC@k (the sweep sorts internally).
    candidates = [_cand("X", "R1", 0.9), _cand("L1", "R1", 0.5), _cand("L2", "R2", 0.8)]
    gold = LabeledPairs.from_pairs([("L1", "R1"), ("L2", "R2")])
    bm = blocking_metrics(candidates, gold=gold, ks=[2, 1])
    assert bm.pc_at(1) == 0.5
    assert bm.pc_at(2) == 1.0


# --- clustering_metrics (B3) ------------------------------------------------


def _clustering(labels: dict[str, int]) -> ClusteringResult:
    return ClusteringResult(labels=labels)


def test_clustering_metrics_perfect() -> None:
    clustering = _clustering({"A": 0, "B": 0, "C": 1, "D": 1})
    gold = LabeledPairs.from_pairs([("A", "B"), ("C", "D")])
    cm = clustering_metrics(clustering, gold=gold)
    assert (cm.b3_precision, cm.b3_recall, cm.b3_f1) == (1.0, 1.0, 1.0)
    assert (cm.n_clusters, cm.n_gold_clusters) == (2, 2)


def test_clustering_metrics_over_merge_lowers_precision() -> None:
    clustering = _clustering({"A": 0, "B": 0, "C": 0})  # C wrongly merged in
    gold = LabeledPairs.from_pairs([("A", "B")])  # C is a gold singleton
    cm = clustering_metrics(clustering, gold=gold)
    assert cm.b3_recall == 1.0
    # A,B: overlap 2/3 each; C (wrongly merged): 1/3 -> (2/3 + 2/3 + 1/3) / 3.
    assert cm.b3_precision == pytest.approx(5 / 9)


def test_clustering_metrics_split_lowers_recall() -> None:
    clustering = _clustering({"A": 0, "B": 1})  # a true cluster split apart
    gold = LabeledPairs.from_pairs([("A", "B")])
    cm = clustering_metrics(clustering, gold=gold)
    assert cm.b3_precision == 1.0
    # A,B each recall 1 of 2 gold-cluster members -> (1/2 + 1/2) / 2.
    assert cm.b3_recall == pytest.approx(0.5)


def test_clustering_metrics_empty() -> None:
    empty = ClusteringResult(labels={})
    cm = clustering_metrics(empty, gold=LabeledPairs.from_pairs([]))
    assert (cm.b3_precision, cm.b3_recall, cm.b3_f1) == (0.0, 0.0, 0.0)
    assert (cm.n_clusters, cm.n_gold_clusters) == (0, 0)


def test_clustering_metrics_ignores_gold_outside_the_universe() -> None:
    # Gold links A to Z, but Z is not in the clustering -> that edge is dropped,
    # so A is a gold singleton and scores perfectly on its own.
    clustering = _clustering({"A": 0, "B": 1})
    gold = LabeledPairs.from_pairs([("A", "Z")])  # Z absent from clustering
    cm = clustering_metrics(clustering, gold=gold)
    assert (cm.b3_precision, cm.b3_recall, cm.b3_f1) == (1.0, 1.0, 1.0)
    assert cm.n_gold_clusters == 2  # {A}, {B} — both singletons


def test_clustering_metrics_mixed_precision_and_recall() -> None:
    # Predicted {A,B}{C}; gold {A,C}{B} — a wrong merge AND a wrong split, so
    # both B3 terms drop. Per record: P = (1/2 + 1/2 + 1)/3, R = (1/2 + 1 + 1/2)/3.
    clustering = _clustering({"A": 0, "B": 0, "C": 1})
    gold = LabeledPairs.from_pairs([("A", "C")])
    cm = clustering_metrics(clustering, gold=gold)
    assert cm.b3_precision == pytest.approx(2 / 3)
    assert cm.b3_recall == pytest.approx(2 / 3)
    assert cm.b3_f1 == pytest.approx(2 / 3)
    assert (cm.n_clusters, cm.n_gold_clusters) == (2, 2)


# --- ThresholdSweep / AdjustedMetrics accessors -----------------------------


def _lm(tp: int, fp: int, fn: int) -> LinkageMetrics:
    return LinkageMetrics(
        true_positive=tp, false_positive=fp, false_negative=fn, n_gold=tp + fn
    )


def test_threshold_sweep_best_f1() -> None:
    sweep = ThresholdSweep(rows=((0.3, _lm(5, 5, 0)), (0.6, _lm(5, 0, 1))))
    threshold, _ = sweep.best_f1()
    assert threshold == 0.6


def test_threshold_sweep_best_f1_ties_break_to_higher_threshold() -> None:
    rows = ((0.3, _lm(5, 5, 5)), (0.7, _lm(5, 5, 5)))  # identical F1
    threshold, _ = ThresholdSweep(rows=rows).best_f1()
    assert threshold == 0.7


def test_threshold_sweep_best_f1_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        ThresholdSweep(rows=()).best_f1()


def test_threshold_sweep_at_recall() -> None:
    sweep = ThresholdSweep(rows=((0.3, _lm(10, 5, 0)), (0.6, _lm(7, 0, 3))))
    threshold, _ = sweep.at_recall(0.9)  # only the 0.3 row reaches recall >= 0.9
    assert threshold == 0.3


def test_threshold_sweep_at_recall_unreachable_raises() -> None:
    sweep = ThresholdSweep(rows=((0.6, _lm(7, 0, 3)),))  # max recall 0.7
    with pytest.raises(ValueError, match="recall"):
        sweep.at_recall(0.9)


def test_threshold_sweep_at_recall_picks_best_f1_among_eligible() -> None:
    # Both rows clear recall 0.8; the higher-F1 (more precise) one is returned.
    rows = ((0.3, _lm(8, 8, 2)), (0.6, _lm(8, 2, 2)))  # recall 0.8 each; F1 differs
    threshold, _ = ThresholdSweep(rows=rows).at_recall(0.8)
    assert threshold == 0.6


def test_adjusted_metrics() -> None:
    matcher = _lm(8, 2, 0)  # precision 0.8, recall 1.0
    adjusted = AdjustedMetrics(matcher=matcher, blocking_recall_at_k=0.5, k=5)
    assert adjusted.recall_adjusted == 0.5
    expected_f1 = (2 * 0.8 * 0.5) / (0.8 + 0.5)
    assert abs(adjusted.f1_adjusted - expected_f1) < 1e-9


def test_adjusted_metrics_zero_denominator_f1_is_zero() -> None:
    # No predictions (precision 0) and zero blocking recall -> f1_adjusted is a
    # clean 0.0, not a ZeroDivisionError.
    matcher = _lm(0, 0, 5)  # precision 0, recall 0
    adjusted = AdjustedMetrics(matcher=matcher, blocking_recall_at_k=0.0, k=5)
    assert adjusted.recall_adjusted == 0.0
    assert adjusted.f1_adjusted == 0.0
