"""Behavioural tests for the dependency-free vertical slice (``with_defaults()``
→ ``link()`` → ``to_frame()`` / ``linkage_metrics``).

This is the reference-implementation oracle of the A0.5 freeze gate: it proves
the contract is implementable end to end on the light stack, pins the calibrated
defaults, exercises the hard-failure taxonomy and edge cases, and locks the two
contract decisions the slice surfaced — the ``directed`` pair-identity flag (D1)
and the spec→artifact reuse guarantee (D6).
"""

import numpy as np
import pandas as pd
import pytest

from denselinkage import DenseLinker, Source
from denselinkage.blocking import DenseBlocker
from denselinkage.core import errors
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.core.results import LabeledPairs, LinkageResult
from denselinkage.embedding import HashedNGramEmbedder
from denselinkage.indexing import NumpyFlatIndex
from denselinkage.matching import ThresholdMatcher
from denselinkage.metrics import linkage_metrics

GOLD = [("A1", "B1"), ("A2", "B2"), ("A3", "B3")]
SCHEMA = ["left_id", "right_id", "similarity", "match", "confidence", "reason"]


def _sources() -> tuple[Source, Source]:
    df_a = pd.DataFrame(
        {
            "id": ["A1", "A2", "A3"],
            "name": ["Apple Inc", "Microsoft Corp", "Google LLC"],
            "city": ["Cupertino", "Redmond", "Mountain View"],
        }
    )
    df_b = pd.DataFrame(
        {
            "id": ["B1", "B2", "B3"],
            "name": ["Apple Incorporated", "Microsoft", "Google"],
            "city": ["Cupertino", "Redmond", "Mountain View"],
        }
    )
    return Source(df_a, id_column="id"), Source(df_b, id_column="id")


def test_quickstart_link_is_perfect() -> None:
    left, right = _sources()
    result = DenseLinker.with_defaults().link(left, right)
    metrics = linkage_metrics(result, gold=LabeledPairs.from_pairs(GOLD))
    assert (metrics.precision, metrics.recall, metrics.f1) == (1.0, 1.0, 1.0)
    assert metrics.n_errors == 0


def test_to_frame_schema_and_matches() -> None:
    left, right = _sources()
    frame = DenseLinker.with_defaults().link(left, right).to_frame()
    assert list(frame.columns) == SCHEMA
    assert frame["similarity"].notna().all()  # a dense blocker always sets it
    matched = {(row.left_id, row.right_id) for row in frame[frame.match].itertuples()}
    assert matched == set(GOLD)


def test_default_stack_separates_true_from_cross_pairs() -> None:
    # The calibrated default threshold (0.5) must sit in the gap between true and
    # cross-pair similarity; this guards the defaults against silent drift.
    left, right = _sources()
    frame = DenseLinker.with_defaults().link(left, right).to_frame()
    true_sim = frame[frame.left_id.str[1] == frame.right_id.str[1]]["similarity"]
    cross_sim = frame[frame.left_id.str[1] != frame.right_id.str[1]]["similarity"]
    assert cross_sim.max() < 0.5 < true_sim.min()


def test_config_is_reusable_across_calls() -> None:
    # D6 spec→artifact payoff: a frozen linker reused across datasets is not
    # corrupted (no deepcopy). link == index().query(), and is idempotent.
    linker = DenseLinker.with_defaults()
    left, right = _sources()
    first = linker.link(left, right).to_frame()
    second = linker.link(left, right).to_frame()
    assert first.equals(second)
    assert linker.index(left).query(right).to_frame().equals(first)


def _matched(left_id: str, right_id: str) -> tuple[CandidatePair, MatchDecision]:
    pair = CandidatePair(
        Record(left_id, ""), Record(right_id, ""), similarity_score=1.0
    )
    return (pair, MatchDecision(is_match=True))


def test_directed_flag_controls_pair_identity() -> None:
    # D1: the verb is not recoverable from the result, so linkage_metrics takes
    # `directed`. A reversed pair is a miss under link (directed) but a hit under
    # dedupe (undirected). This is the contract gap the slice surfaced.
    gold = LabeledPairs.from_pairs([("1", "2")])
    reversed_result = LinkageResult(decisions=(_matched("2", "1"),))

    directed = linkage_metrics(reversed_result, gold=gold, directed=True)
    assert (directed.true_positive, directed.false_positive) == (0, 1)
    assert directed.false_negative == 1

    undirected = linkage_metrics(reversed_result, gold=gold, directed=False)
    assert (undirected.true_positive, undirected.false_positive) == (1, 0)
    assert undirected.false_negative == 0


# --- hard-failure taxonomy and edge cases the slice now implements ---


def test_unknown_id_column_raises() -> None:
    linker = DenseLinker.with_defaults()
    good = Source(pd.DataFrame({"id": ["B1"]}), id_column="id")
    bad = Source(pd.DataFrame({"name": ["A1"]}), id_column="id")
    with pytest.raises(errors.UnknownIdColumn):
        linker.link(bad, good)


def test_empty_source_raises() -> None:
    linker = DenseLinker.with_defaults()
    good = Source(pd.DataFrame({"id": ["B1"]}), id_column="id")
    empty = Source(pd.DataFrame({"id": []}), id_column="id")
    with pytest.raises(errors.EmptySource):
        linker.link(empty, good)


def test_duplicate_record_id_raises() -> None:
    linker = DenseLinker.with_defaults()
    good = Source(pd.DataFrame({"id": ["B1"]}), id_column="id")
    dupes = Source(pd.DataFrame({"id": ["A1", "A1"]}), id_column="id")
    with pytest.raises(errors.DuplicateRecordId):
        linker.link(dupes, good)


def test_invalid_top_k_raises_at_build_and_at_query() -> None:
    embedder = HashedNGramEmbedder()
    with pytest.raises(errors.InvalidTopK):
        DenseBlocker(embedder=embedder, vector_index=NumpyFlatIndex(), top_k=0).build(
            [Record("a", "alpha")]
        )
    built = DenseBlocker(
        embedder=embedder, vector_index=NumpyFlatIndex(), top_k=5
    ).build([Record("a", "alpha")])
    with pytest.raises(errors.InvalidTopK):
        built.query([Record("q", "alpha")], top_k=0)


def test_dimension_mismatch_raises() -> None:
    index = NumpyFlatIndex().build(np.zeros((1, 4), dtype=np.float32), ["a"])
    with pytest.raises(errors.DimensionMismatch):
        index.search(np.zeros((1, 8), dtype=np.float32), top_k=1)


def test_threshold_matcher_unscored_pair_is_match_error() -> None:
    pair = CandidatePair(Record("a", ""), Record("b", ""), similarity_score=None)
    [outcome] = ThresholdMatcher().match([pair])
    assert isinstance(outcome, MatchError)


def test_extended_is_not_implemented_in_v1() -> None:
    index = NumpyFlatIndex().build(np.zeros((1, 4), dtype=np.float32), ["a"])
    with pytest.raises(NotImplementedError):
        index.extended(np.zeros((1, 4), dtype=np.float32), ["b"])


def test_empty_text_embeds_to_zero_vector() -> None:
    # Empty / whitespace-only serializations must not collide at cosine 1.0.
    vectors = HashedNGramEmbedder().encode(["", "   ", "real text"])
    assert not vectors[0].any()
    assert not vectors[1].any()
    assert vectors[2].any()
