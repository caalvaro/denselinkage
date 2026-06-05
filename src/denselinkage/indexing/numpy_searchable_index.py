"""``NumpySearchableIndex`` — artifact built by ``NumpyFlatIndex``."""

from collections.abc import Sequence

import numpy as np

from denselinkage.core.errors import DimensionMismatch
from denselinkage.core.models import RecordId
from denselinkage.core.ports import SearchableIndex, Vectors


class NumpySearchableIndex(SearchableIndex):
    """Immutable artifact built by ``NumpyFlatIndex`` — exhaustive (flat)
    nearest-neighbour search by inner product (which equals cosine for the
    L2-normalized vectors the reference embedder produces)."""

    def __init__(self, vectors: Vectors, ids: Sequence[RecordId]) -> None:
        self._vectors = np.asarray(vectors, dtype=np.float32)
        self._ids: list[RecordId] = list(ids)

    @property
    def vectors(self) -> Vectors:
        """The indexed vectors (float32, ``n_records x embedding_dim``)."""
        return self._vectors

    @property
    def ids(self) -> list[RecordId]:
        """Record ids aligned positionally with :attr:`vectors`."""
        return self._ids

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
