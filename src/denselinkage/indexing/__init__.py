"""Vector indexes (own port/module). ``FaissFlatIndex`` is the heavy adapter
(extra: ``[faiss]``).

Each backend is a *spec* (``VectorIndex``): ``build`` produces a fresh,
immutable ``SearchableIndex`` artifact populated with the given vectors. The
spec is stateless and reusable; state lives only in the artifact.
"""

from collections.abc import Sequence

from denselinkage.core.models import RecordId
from denselinkage.core.ports import SearchableIndex, VectorIndex, Vectors


class NumpyFlatIndex(VectorIndex):
    """Dependency-free reference index spec (brute-force / flat search)."""

    def build(
        self, vectors: Vectors, ids: Sequence[RecordId]
    ) -> "NumpySearchableIndex": ...


class NumpySearchableIndex(SearchableIndex):
    """Immutable artifact built by ``NumpyFlatIndex`` — exhaustive (flat)
    nearest-neighbour search over the indexed vectors."""

    def search(
        self, queries: Vectors, *, top_k: int
    ) -> list[list[tuple[RecordId, float]]]: ...

    def extended(
        self, vectors: Vectors, ids: Sequence[RecordId]
    ) -> "NumpySearchableIndex": ...


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
