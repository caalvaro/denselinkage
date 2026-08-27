"""Unit tests for ``DenseBlockingIndex`` query behaviour: pair orientation, the
query-time threshold override, artifact reuse, and the ``batch_size`` the spec
fixes reaching the embedder on the build *and* the query path (#50). InvalidTopK
is covered in ``test_quickstart_end_to_end``."""

from collections.abc import Sequence

import pytest

from denselinkage.blocking import DenseBlocker
from denselinkage.core.models import Record
from denselinkage.core.ports import Vectors
from denselinkage.embedding import HashedNGramEmbedder
from denselinkage.indexing import NumpyFlatIndex


class _RecordingEmbedder(HashedNGramEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int | None] = []

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> Vectors:
        self.batch_sizes.append(batch_size)
        return super().encode(texts, batch_size=batch_size, show_progress=show_progress)


def _blocker(*, top_k: int = 5, similarity_threshold: float = 0.0) -> DenseBlocker:
    return DenseBlocker(
        embedder=HashedNGramEmbedder(),
        vector_index=NumpyFlatIndex(),
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )


def test_explicit_batch_size_reaches_the_embedder_on_both_paths() -> None:
    """The whole recorded sequence, so a regression on either call fails.

    ``batch_size`` is chosen to fit a corpus in memory; a query set embedded at
    the embedder's own default exhausts it on a call the caller never tuned.
    """
    embedder = _RecordingEmbedder()
    blocker = DenseBlocker(
        embedder=embedder,
        vector_index=NumpyFlatIndex(),
        batch_size=64,
    )

    index = blocker.build([Record("L1", "acme inc")])
    index.query([Record("R1", "acme incorporated")])

    assert embedder.batch_sizes == [64, 64]  # build, then query


def test_batch_size_defaults_to_none_on_both_paths() -> None:
    embedder = _RecordingEmbedder()
    blocker = DenseBlocker(embedder=embedder, vector_index=NumpyFlatIndex())

    index = blocker.build([Record("L1", "acme inc")])
    index.query([Record("R1", "acme incorporated")])

    # None, not a number: the embedder's own default still applies.
    assert embedder.batch_sizes == [None, None]


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


def test_records_accessor_is_a_read_only_view() -> None:
    # The built artifact is immutable; its record map cannot be mutated.
    index = _blocker().build([Record("L1", "acme inc")])
    assert index.records["L1"].text == "acme inc"
    with pytest.raises(TypeError):  # MappingProxyType forbids item assignment
        index.records["L2"] = Record("L2", "globex")
