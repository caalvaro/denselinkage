"""``FaissFlatIndex`` / ``FaissSearchableIndex`` behaviour (extra: ``[faiss]``).

Adapter-marked: needs the ``faiss`` backend, so excluded from the dependency-free
coverage run and exercised by the dedicated ``adapter-tests`` CI job. The headline
assertion is the **differential vs the numpy reference** — FAISS must return the
same neighbours as ``NumpyFlatIndex`` for the same vectors (both are exact, flat,
inner-product search), so the two backends are interchangeable behind the port.
"""

from collections.abc import Sequence

import numpy as np
import pytest

from denselinkage.core.errors import DimensionMismatch
from denselinkage.core.models import RecordId
from denselinkage.indexing import FaissFlatIndex, FaissSearchableIndex, NumpyFlatIndex

pytestmark = pytest.mark.adapter


def _index(vectors: list[list[float]], ids: Sequence[RecordId]) -> FaissSearchableIndex:
    return FaissFlatIndex().build(np.asarray(vectors, dtype=np.float32), ids)


def _query(vector: list[float]) -> np.ndarray:
    return np.asarray([vector], dtype=np.float32)


def test_search_orders_hits_by_descending_score() -> None:
    index = _index([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
    [hits] = index.search(_query([1.0, 0.0]), top_k=2)
    assert [record_id for record_id, _ in hits] == ["a", "b"]
    assert hits[0][1] == pytest.approx(1.0)
    assert hits[1][1] == pytest.approx(0.0)


def test_search_top_k_is_clamped_to_index_size() -> None:
    index = _index([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
    [hits] = index.search(_query([1.0, 0.0]), top_k=5)
    assert len(hits) == 2


def test_search_is_deterministic_across_calls() -> None:
    index = _index([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], ["a", "b", "c"])
    query = _query([1.0, 0.0])
    assert index.search(query, top_k=2) == index.search(query, top_k=2)


def test_search_on_empty_index_returns_empty_per_query() -> None:
    index = FaissFlatIndex().build(np.zeros((0, 2), dtype=np.float32), [])
    assert index.search(_query([1.0, 0.0]), top_k=3) == [[]]


def test_search_raises_on_dimension_mismatch() -> None:
    index = _index([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
    with pytest.raises(DimensionMismatch):
        index.search(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), top_k=1)


def test_vectors_accessor_is_a_read_only_view() -> None:
    index = _index([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
    vectors = index.vectors
    assert vectors.shape == (2, 2)
    with pytest.raises(ValueError):  # assignment to a read-only array
        vectors[0, 0] = 9.0


def test_ids_accessor_is_immutable() -> None:
    index = _index([[1.0, 0.0]], ["a"])
    assert index.ids == ("a",)
    assert not isinstance(index.ids, list)  # a tuple, not the internal list


def test_vectors_on_empty_index_is_empty() -> None:
    index = FaissFlatIndex().build(np.zeros((0, 4), dtype=np.float32), [])
    vectors = index.vectors
    assert vectors.shape == (0, 4)
    assert vectors.dtype == np.float32


def test_extended_is_not_implemented_in_v1() -> None:
    index = _index([[1.0, 0.0]], ["a"])
    with pytest.raises(NotImplementedError, match="out of scope"):
        index.extended(np.asarray([[0.0, 1.0]], dtype=np.float32), ["b"])


def test_faiss_matches_numpy_neighbours() -> None:
    # The two exact-flat backends must agree. Random unit vectors make ties
    # measure-zero, so neighbour ids AND scores match position-for-position.
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((20, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"r{i}" for i in range(20)]
    queries = vectors[:5]

    faiss_hits = FaissFlatIndex().build(vectors, ids).search(queries, top_k=5)
    numpy_hits = NumpyFlatIndex().build(vectors, ids).search(queries, top_k=5)

    faiss_ids = [[record_id for record_id, _ in row] for row in faiss_hits]
    numpy_ids = [[record_id for record_id, _ in row] for row in numpy_hits]
    assert faiss_ids == numpy_ids

    faiss_scores = [[score for _, score in row] for row in faiss_hits]
    numpy_scores = [[score for _, score in row] for row in numpy_hits]
    assert np.allclose(faiss_scores, numpy_scores, atol=1e-5)
