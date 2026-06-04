"""Unit tests for ``SimilarityThresholdFilter`` (the dependency-free Filter)."""

from denselinkage.core.models import CandidatePair, Record
from denselinkage.filtering import SimilarityThresholdFilter


def _pair(score: float | None) -> CandidatePair:
    return CandidatePair(Record("a", ""), Record("b", ""), similarity_score=score)


def test_keeps_pairs_at_or_above_threshold() -> None:
    kept = SimilarityThresholdFilter(threshold=0.5).filter(
        [_pair(0.9), _pair(0.5), _pair(0.4)]
    )
    assert [p.similarity_score for p in kept] == [0.9, 0.5]


def test_keeps_unscored_pairs_by_default() -> None:
    kept = SimilarityThresholdFilter(threshold=0.5).filter([_pair(None), _pair(0.9)])
    assert [p.similarity_score for p in kept] == [None, 0.9]


def test_drops_unscored_pairs_when_configured() -> None:
    kept = SimilarityThresholdFilter(threshold=0.5, drop_unscored=True).filter(
        [_pair(None), _pair(0.9)]
    )
    assert [p.similarity_score for p in kept] == [0.9]


def test_empty_input() -> None:
    assert SimilarityThresholdFilter().filter([]) == []
