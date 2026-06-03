"""Unit tests for ``DenseBlockingIndex`` query behaviour: pair orientation, the
query-time threshold override, and artifact reuse. InvalidTopK is covered in
``test_quickstart_end_to_end``."""

from denselinkage.blocking import DenseBlocker
from denselinkage.core.models import Record
from denselinkage.embedding import HashedNGramEmbedder
from denselinkage.indexing import NumpyFlatIndex


def _blocker(*, top_k: int = 5, similarity_threshold: float = 0.0) -> DenseBlocker:
    return DenseBlocker(
        embedder=HashedNGramEmbedder(),
        vector_index=NumpyFlatIndex(),
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )


def test_pair_orientation_is_indexed_a_query_b() -> None:
    index = _blocker().build([Record("L1", "acme inc"), Record("L2", "globex corp")])
    pairs = index.query([Record("R1", "acme incorporated")])
    assert pairs  # the lexical stack surfaces at least one candidate
    assert all(pair.record_a.id in {"L1", "L2"} for pair in pairs)  # indexed/reference
    assert all(pair.record_b.id == "R1" for pair in pairs)  # query side


def test_query_threshold_override_gates_pairs() -> None:
    index = _blocker(similarity_threshold=0.0).build([Record("L1", "acme inc")])
    # Identical text -> cosine 1.0, so it survives a 0.99 gate but not a 1.01 gate.
    kept = index.query([Record("R1", "acme inc")], similarity_threshold=0.99)
    assert [pair.record_a.id for pair in kept] == ["L1"]
    assert kept[0].similarity_score >= 0.99
    assert index.query([Record("R1", "acme inc")], similarity_threshold=1.01) == []


def test_built_index_is_reusable_across_query_sets() -> None:
    index = _blocker().build([Record("L1", "acme inc")])
    first = index.query([Record("R1", "acme inc")])
    second = index.query([Record("R2", "acme inc")])
    assert (first[0].record_a.id, first[0].record_b.id) == ("L1", "R1")
    assert (second[0].record_a.id, second[0].record_b.id) == ("L1", "R2")
    assert first[0].similarity_score == second[0].similarity_score
