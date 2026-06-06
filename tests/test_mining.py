"""Behavioural tests for B5 ``mine_hard_negatives`` — the highest-similarity
non-matches, as training material."""

import pytest

from denselinkage import LabeledPairs
from denselinkage.core.models import CandidatePair, Record
from denselinkage.mining import mine_hard_negatives


def _pair(a: str, b: str, score: float | None) -> CandidatePair:
    return CandidatePair(Record(a, ""), Record(b, ""), similarity_score=score)


def _keys(pairs: list[CandidatePair]) -> list[tuple[str, str]]:
    return [(p.record_a.id, p.record_b.id) for p in pairs]


def test_mines_hardest_non_gold_in_descending_order() -> None:
    candidates = [
        _pair("A1", "B1", 0.9),  # gold -> excluded
        _pair("A1", "B2", 0.8),  # non-gold, hardest
        _pair("A2", "B1", 0.4),  # non-gold
    ]
    gold = LabeledPairs.from_pairs([("A1", "B1")])
    assert _keys(mine_hard_negatives(candidates, gold=gold)) == [
        ("A1", "B2"),
        ("A2", "B1"),
    ]


def test_n_caps_to_the_hardest() -> None:
    candidates = [_pair("A1", "B2", 0.8), _pair("A2", "B1", 0.4)]
    gold = LabeledPairs.from_pairs([])
    assert _keys(mine_hard_negatives(candidates, gold=gold, n=1)) == [("A1", "B2")]


def test_unscored_pairs_are_excluded() -> None:
    candidates = [_pair("A1", "B2", None), _pair("A2", "B1", 0.4)]
    gold = LabeledPairs.from_pairs([])
    assert _keys(mine_hard_negatives(candidates, gold=gold)) == [("A2", "B1")]


def test_all_gold_yields_empty() -> None:
    candidates = [_pair("A1", "B1", 0.9)]
    gold = LabeledPairs.from_pairs([("A1", "B1")])
    assert mine_hard_negatives(candidates, gold=gold) == []


def test_directed_false_excludes_reversed_gold() -> None:
    candidates = [_pair("2", "1", 0.9)]
    gold = LabeledPairs.from_pairs([("1", "2")])
    assert mine_hard_negatives(candidates, gold=gold, directed=False) == []
    # directed=True: ("2","1") != gold ("1","2"), so it is a hard negative.
    assert len(mine_hard_negatives(candidates, gold=gold, directed=True)) == 1


def test_ties_broken_deterministically_by_id() -> None:
    candidates = [_pair("A2", "B1", 0.5), _pair("A1", "B2", 0.5)]  # equal scores
    gold = LabeledPairs.from_pairs([])
    assert _keys(mine_hard_negatives(candidates, gold=gold)) == [
        ("A1", "B2"),
        ("A2", "B1"),
    ]


def test_negative_n_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mine_hard_negatives([], gold=LabeledPairs.from_pairs([]), n=-1)
