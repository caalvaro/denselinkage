"""Behavioural tests for the B3 gold/clustering utilities: ``LabeledPairs.split``
and the full-record-universe ``connected_components(all_record_ids=...)`` path."""

import pytest

from denselinkage import LabeledPairs, connected_components
from denselinkage.core.models import CandidatePair, MatchDecision, Record
from denselinkage.core.results import LinkageResult
from denselinkage.metrics import clustering_metrics

# --- LabeledPairs.split -----------------------------------------------------


def _gold(n: int) -> LabeledPairs:
    return LabeledPairs.from_pairs([(str(i), str(i + 100)) for i in range(n)])


def test_split_sizes_disjoint_and_cover() -> None:
    train, test = _gold(10).split(test_size=0.3, seed=0)
    assert len(test.pairs) == 3
    assert len(train.pairs) == 7
    assert train.pairs.isdisjoint(test.pairs)
    assert train.pairs | test.pairs == _gold(10).pairs


def test_split_is_deterministic_with_seed() -> None:
    a_train, a_test = _gold(10).split(test_size=0.4, seed=42)
    b_train, b_test = _gold(10).split(test_size=0.4, seed=42)
    assert a_train.pairs == b_train.pairs
    assert a_test.pairs == b_test.pairs


def test_split_extremes_route_all_to_one_side() -> None:
    gold = _gold(4)
    train, test = gold.split(test_size=0.0)
    assert test.pairs == frozenset()
    assert train.pairs == gold.pairs
    train, test = gold.split(test_size=1.0)
    assert train.pairs == frozenset()
    assert test.pairs == gold.pairs


def test_split_rejects_out_of_range() -> None:
    gold = _gold(2)
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        gold.split(test_size=1.5)
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        gold.split(test_size=-0.1)


def test_split_empty_gold() -> None:
    train, test = LabeledPairs.from_pairs([]).split(test_size=0.5)
    assert train.pairs == frozenset()
    assert test.pairs == frozenset()


# --- connected_components(all_record_ids=...) -------------------------------


def _match(a: str, b: str) -> tuple[CandidatePair, MatchDecision]:
    return (CandidatePair(Record(a, ""), Record(b, "")), MatchDecision(is_match=True))


def test_all_record_ids_adds_unmatched_records_as_singletons() -> None:
    result = LinkageResult(decisions=(_match("1", "2"),))
    clusters = connected_components(result, all_record_ids=["1", "2", "3"])
    assert clusters.labels["1"] == clusters.labels["2"]
    assert clusters.labels["3"] != clusters.labels["1"]  # 3 is its own singleton
    assert clusters.n_clusters == 2


def test_default_universe_is_pairs_only() -> None:
    result = LinkageResult(decisions=(_match("1", "2"),))
    clusters = connected_components(result)  # no all_record_ids
    assert set(clusters.labels) == {"1", "2"}


def test_all_record_ids_are_stringified() -> None:
    result = LinkageResult(decisions=(_match("1", "2"),))
    clusters = connected_components(result, all_record_ids=[1, 2, 3])  # int ids
    assert "3" in clusters.labels
    assert clusters.labels["1"] == clusters.labels["2"]


def test_full_universe_yields_complete_b3() -> None:
    # Gold says {1,2,3} are one entity; the matcher only linked (1,2) and never
    # surfaced 3. Without the universe, B³ recall is falsely perfect; with it, 3
    # is a singleton and recall honestly drops.
    gold = LabeledPairs.from_pairs([("1", "2"), ("1", "3")])
    result = LinkageResult(decisions=(_match("1", "2"),))

    partial = clustering_metrics(connected_components(result), gold=gold)
    assert partial.b3_recall == 1.0  # 3 invisible -> looks perfect

    full = clustering_metrics(
        connected_components(result, all_record_ids=["1", "2", "3"]), gold=gold
    )
    assert full.b3_recall == pytest.approx(5 / 9)  # 3 unmerged -> honest
    assert full.b3_precision == 1.0
