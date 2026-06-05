"""Unit tests for ``NumpySearchableIndex`` search behaviour (inner-product
nearest-neighbour). DimensionMismatch is covered in
``test_quickstart_end_to_end``."""

from collections.abc import Sequence

import numpy as np

from denselinkage.core.models import RecordId
from denselinkage.indexing import NumpyFlatIndex
from denselinkage.indexing.numpy_searchable_index import NumpySearchableIndex


def _index(vectors: list[list[float]], ids: Sequence[RecordId]) -> NumpySearchableIndex:
    return NumpyFlatIndex().build(np.asarray(vectors, dtype=np.float32), ids)


def _query(vector: list[float]) -> np.ndarray:
    return np.asarray([vector], dtype=np.float32)


def test_search_orders_hits_by_descending_score() -> None:
    index = _index([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
    [hits] = index.search(_query([1.0, 0.0]), top_k=2)
    assert [record_id for record_id, _ in hits] == ["a", "b"]
    assert hits[0][1] == 1.0
    assert hits[1][1] == 0.0


def test_search_top_k_is_clamped_to_index_size() -> None:
    index = _index([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
    [hits] = index.search(_query([1.0, 0.0]), top_k=5)
    assert len(hits) == 2


def test_search_is_deterministic_across_calls() -> None:
    # Two identical vectors tie at score 1.0; the result must be reproducible.
    index = _index([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], ["a", "b", "c"])
    query = _query([1.0, 0.0])
    assert index.search(query, top_k=2) == index.search(query, top_k=2)


def test_search_on_empty_index_returns_empty_per_query() -> None:
    index = _index([], [])
    assert index.search(_query([1.0, 0.0]), top_k=3) == [[]]
