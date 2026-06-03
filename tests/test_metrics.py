"""Unit tests for ``linkage_metrics`` — the honest, provenance-preserving
counting the contract promises (the ``directed`` flag is covered in
``test_quickstart_end_to_end``)."""

import pandas as pd

from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.core.results import LabeledPairs, LinkageResult
from denselinkage.metrics import linkage_metrics


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
