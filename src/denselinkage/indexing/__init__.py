"""Vector indexes (own port/module). ``FaissFlatIndex`` is the heavy adapter
(extra: ``[faiss]``).

Each backend is a *spec* (``VectorIndex``): ``build`` produces a fresh,
immutable ``SearchableIndex`` artifact populated with the given vectors. The
spec is stateless and reusable; state lives only in the artifact.
"""

from collections.abc import Sequence

import numpy as np

from denselinkage.core.errors import DimensionMismatch
from denselinkage.core.models import RecordId
from denselinkage.core.ports import SearchableIndex, VectorIndex, Vectors


class NumpyFlatIndex(VectorIndex):
    """Dependency-free reference index spec (brute-force / flat search)."""

    def build(
        self, vectors: Vectors, ids: Sequence[RecordId]
    ) -> "NumpySearchableIndex":
        return NumpySearchableIndex(vectors, ids)


class NumpySearchableIndex(SearchableIndex):
    """Immutable artifact built by ``NumpyFlatIndex`` — exhaustive (flat)
    nearest-neighbour search by inner product (which equals cosine for the
    L2-normalized vectors the reference embedder produces)."""

    def __init__(self, vectors: Vectors, ids: Sequence[RecordId]) -> None:
        self._vectors = np.asarray(vectors, dtype=np.float32)
        self._ids: list[RecordId] = list(ids)

    def search(
        self, queries: Vectors, *, top_k: int
    ) -> list[list[tuple[RecordId, float]]]:
        query_matrix = np.asarray(queries, dtype=np.float32)
        n_indexed = len(self._ids)
        if n_indexed == 0:
            return [[] for _ in range(query_matrix.shape[0])]
        if query_matrix.shape[1] != self._vectors.shape[1]:
            raise DimensionMismatch(
                f"query width {query_matrix.shape[1]} does not match indexed "
                f"width {self._vectors.shape[1]}"
            )
        k = min(top_k, n_indexed)
        scores = query_matrix @ self._vectors.T
        results: list[list[tuple[RecordId, float]]] = []
        for row in scores:
            top = np.argpartition(-row, k - 1)[:k]
            top = top[np.argsort(-row[top], kind="stable")]  # deterministic order
            results.append([(self._ids[int(j)], float(row[int(j)])) for j in top])
        return results

    def extended(
        self, vectors: Vectors, ids: Sequence[RecordId]
    ) -> "NumpySearchableIndex":
        raise NotImplementedError(
            "incremental indexing is out of scope for v1 (see ADR-0001)"
        )


class FaissFlatIndex(VectorIndex):
    """FAISS-backed index spec (extra: ``[faiss]``)."""

    def build(
        self, vectors: Vectors, ids: Sequence[RecordId]
    ) -> "FaissSearchableIndex": ...


class FaissSearchableIndex(SearchableIndex):
    """Immutable artifact built by ``FaissFlatIndex``."""

    def search(
        self, queries: Vectors, *, top_k: int
    ) -> list[list[tuple[RecordId, float]]]: ...

    def extended(
        self, vectors: Vectors, ids: Sequence[RecordId]
    ) -> "FaissSearchableIndex": ...


__all__ = [
    "FaissFlatIndex",
    "FaissSearchableIndex",
    "NumpyFlatIndex",
    "NumpySearchableIndex",
]
