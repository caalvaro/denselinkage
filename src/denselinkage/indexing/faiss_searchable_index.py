"""``FaissSearchableIndex`` — artifact built by ``FaissFlatIndex``."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from denselinkage.core.errors import DimensionMismatch
from denselinkage.core.models import RecordId
from denselinkage.core.ports import SearchableIndex, Vectors


class FaissSearchableIndex(SearchableIndex):
    """Immutable artifact built by ``FaissFlatIndex`` — exact (flat) inner-product
    nearest-neighbour search over a ``faiss.IndexFlatIP``.

    Inner product equals cosine for the L2-normalized vectors the embedders
    produce, so this artifact answers the *same* similarity as
    ``NumpySearchableIndex`` and the ``similarity_threshold`` keeps its meaning
    across the numpy and FAISS backends. The vectors are fixed at build time
    (no ``add``); :meth:`extended` is the not-yet-implemented escape hatch.
    """

    def __init__(self, index: Any, ids: Sequence[RecordId]) -> None:
        # ``index`` is a populated ``faiss.Index`` (kept untyped — faiss ships no
        # stubs). ``index.d`` is the embedding width.
        self._index = index
        self._ids: list[RecordId] = list(ids)
        # Enforce the build-time invariant: a mismatch would let faiss return a
        # ``-1`` sentinel that ``self._ids[-1]`` resolves to the *wrong* record.
        assert index.ntotal == len(self._ids)  # pragma: no branch

    @property
    def vectors(self) -> Vectors:
        """The indexed vectors (float32, ``n_records x embedding_dim``), as a
        read-only array reconstructed from the FAISS index — this artifact is
        immutable."""
        n = len(self._ids)
        if n == 0:
            reconstructed = np.zeros((0, int(self._index.d)), dtype=np.float32)
        else:
            reconstructed = np.asarray(
                self._index.reconstruct_n(0, n), dtype=np.float32
            )
        reconstructed.flags.writeable = False
        return reconstructed

    @property
    def ids(self) -> Sequence[RecordId]:
        """Record ids aligned positionally with :attr:`vectors` (read-only)."""
        return tuple(self._ids)

    def search(
        self, queries: Vectors, *, top_k: int
    ) -> list[list[tuple[RecordId, float]]]:
        query_matrix = np.ascontiguousarray(queries, dtype=np.float32)
        n_indexed = len(self._ids)
        if n_indexed == 0:
            return [[] for _ in range(query_matrix.shape[0])]
        if query_matrix.shape[1] != self._index.d:
            raise DimensionMismatch(
                f"query width {query_matrix.shape[1]} does not match indexed "
                f"width {self._index.d}"
            )
        k = min(top_k, n_indexed)  # clamp so faiss returns k valid ids (no -1 pad)
        scores, indices = self._index.search(query_matrix, k)
        results: list[list[tuple[RecordId, float]]] = []
        for row_scores, row_indices in zip(scores, indices, strict=True):
            hits = [
                (self._ids[int(j)], float(score))
                for score, j in zip(row_scores, row_indices, strict=True)
            ]
            # faiss returns descending-by-score already; a stable re-sort keeps
            # the ordering deterministic (parity with NumpySearchableIndex).
            hits.sort(key=lambda hit: -hit[1])
            results.append(hits)
        return results

    def extended(
        self, vectors: Vectors, ids: Sequence[RecordId]
    ) -> "FaissSearchableIndex":
        raise NotImplementedError(
            "incremental indexing is out of scope for v1 (see ADR-0001)"
        )
