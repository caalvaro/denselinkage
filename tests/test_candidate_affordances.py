"""Behavioural tests for the B1 candidate-pair affordances: ``DenseLinker.block``
/ ``LinkageIndex.candidates`` (blocking-only output) and
``candidate_pairs_from_frame`` (frame -> ``CandidatePair`` for ``match_pairs``)."""

import pandas as pd
import pytest

from denselinkage import (
    DenseLinker,
    LabeledPairs,
    Source,
    TemplateSerializer,
    candidate_pairs_from_frame,
)
from denselinkage.core.errors import InvalidTopK
from denselinkage.core.models import CandidatePair
from denselinkage.matching import ThresholdMatcher
from denselinkage.metrics import blocking_metrics, pair_completeness_at_k


def _left() -> Source:
    df = pd.DataFrame({"id": ["A1", "A2"], "name": ["Apple Inc", "Microsoft Corp"]})
    return Source(df, id_column="id", serializer=TemplateSerializer("{name}"))


def _right() -> Source:
    df = pd.DataFrame({"id": ["B1", "B2"], "name": ["Apple Incorporated", "Microsoft"]})
    return Source(df, id_column="id", serializer=TemplateSerializer("{name}"))


# --- block / candidates -----------------------------------------------------


def test_block_returns_oriented_candidate_pairs_without_matching() -> None:
    pairs = DenseLinker.with_defaults().block(_left(), _right())
    assert pairs and all(isinstance(p, CandidatePair) for p in pairs)
    # record_a = indexed (left), record_b = query (right); blocking sets a score.
    assert all(p.record_a.id in {"A1", "A2"} for p in pairs)
    assert all(p.record_b.id in {"B1", "B2"} for p in pairs)
    assert all(p.similarity_score is not None for p in pairs)


def test_block_equals_index_then_candidates() -> None:
    linker = DenseLinker.with_defaults()
    left, right = _left(), _right()

    def key(ps: list[CandidatePair]) -> list[tuple[str, str]]:
        return sorted((p.record_a.id, p.record_b.id) for p in ps)

    assert key(linker.block(left, right)) == key(linker.index(left).candidates(right))


def test_block_requires_a_blocker() -> None:
    linker = DenseLinker(matcher=ThresholdMatcher())  # no blocker
    with pytest.raises(ValueError, match="requires a blocker"):
        linker.block(_left(), _right())


def test_candidates_top_k_override_limits_neighbours() -> None:
    # top_k=1 -> exactly one nearest indexed record per query record.
    index = DenseLinker.with_defaults().index(_left())
    assert len(index.candidates(_right(), top_k=1)) == 2


def test_candidates_similarity_threshold_override_prunes() -> None:
    # No cosine similarity exceeds 1.0, so a 1.1 floor yields nothing.
    index = DenseLinker.with_defaults().index(_left())
    assert index.candidates(_right(), similarity_threshold=1.1) == []


def test_candidates_rejects_non_positive_top_k() -> None:
    index = DenseLinker.with_defaults().index(_left())
    with pytest.raises(InvalidTopK):
        index.candidates(_right(), top_k=0)


def test_block_output_feeds_blocking_metrics() -> None:
    # The ergonomic PC@k path: block(...) -> blocking_metrics / pc@k. With
    # top_k=2 every indexed record is retrieved per query, so both gold pairs
    # are covered -> pair-completeness is 1.0 (robust, stack-independent).
    candidates = DenseLinker.with_defaults().block(_left(), _right(), top_k=2)
    gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2")])
    bm = blocking_metrics(candidates, gold=gold, ks=[1, 2])
    assert bm.pc_at(2) == 1.0
    assert pair_completeness_at_k(candidates, gold=gold, k=2) == 1.0


# --- candidate_pairs_from_frame ---------------------------------------------


def test_from_frame_builds_pairs_and_feeds_match_pairs() -> None:
    frame = pd.DataFrame({"l": ["A1", "A2"], "r": ["B1", "B2"], "sim": [0.9, 0.2]})
    pairs = candidate_pairs_from_frame(
        frame, left=_left(), right=_right(), left_id="l", right_id="r", similarity="sim"
    )
    assert [(p.record_a.id, p.record_b.id, p.similarity_score) for p in pairs] == [
        ("A1", "B1", 0.9),
        ("A2", "B2", 0.2),
    ]
    # Text materialized from the sources via their serializers.
    assert (pairs[0].record_a.text, pairs[0].record_b.text) == (
        "Apple Inc",
        "Apple Incorporated",
    )
    # End-to-end through the matcher-only path.
    result = DenseLinker(matcher=ThresholdMatcher(threshold=0.5)).match_pairs(pairs)
    decided = {(p.record_a.id, p.record_b.id): d.is_match for p, d in result.decisions}
    assert decided == {("A1", "B1"): True, ("A2", "B2"): False}


def test_from_frame_similarity_column_is_optional() -> None:
    frame = pd.DataFrame({"l": ["A1"], "r": ["B1"]})
    [pair] = candidate_pairs_from_frame(
        frame, left=_left(), right=_right(), left_id="l", right_id="r"
    )
    assert pair.similarity_score is None


def test_from_frame_nan_similarity_becomes_none() -> None:
    frame = pd.DataFrame({"l": ["A1"], "r": ["B1"], "sim": [float("nan")]})
    [pair] = candidate_pairs_from_frame(
        frame, left=_left(), right=_right(), left_id="l", right_id="r", similarity="sim"
    )
    assert pair.similarity_score is None


def test_from_frame_missing_id_column_raises() -> None:
    frame = pd.DataFrame({"l": ["A1"], "r": ["B1"]})
    with pytest.raises(ValueError, match="right_id column 'nope'"):
        candidate_pairs_from_frame(
            frame, left=_left(), right=_right(), left_id="l", right_id="nope"
        )


def test_from_frame_missing_similarity_column_raises() -> None:
    frame = pd.DataFrame({"l": ["A1"], "r": ["B1"]})
    with pytest.raises(ValueError, match="similarity column 'nope'"):
        candidate_pairs_from_frame(
            frame,
            left=_left(),
            right=_right(),
            left_id="l",
            right_id="r",
            similarity="nope",
        )


def test_from_frame_unknown_left_id_raises() -> None:
    frame = pd.DataFrame({"l": ["ZZ"], "r": ["B1"]})
    with pytest.raises(ValueError, match="left id 'ZZ'"):
        candidate_pairs_from_frame(
            frame, left=_left(), right=_right(), left_id="l", right_id="r"
        )


def test_from_frame_unknown_right_id_raises() -> None:
    frame = pd.DataFrame({"l": ["A1"], "r": ["ZZ"]})
    with pytest.raises(ValueError, match="right id 'ZZ'"):
        candidate_pairs_from_frame(
            frame, left=_left(), right=_right(), left_id="l", right_id="r"
        )
