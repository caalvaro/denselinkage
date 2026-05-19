"""Vector indexes (own port/module). ``FaissFlatIndex`` is the heavy adapter
(extra: ``[faiss]``)."""

from collections.abc import Sequence

from denselinkage.core.ports import VectorIndex, Vectors


class NumpyFlatIndex(VectorIndex):
    """Dependency-free reference index."""

    def add(self, vectors: Vectors, ids: Sequence[str]) -> None: ...

    def search(
        self, queries: Vectors, *, top_k: int
    ) -> list[list[tuple[str, float]]]: ...


class FaissFlatIndex(VectorIndex):
    def add(self, vectors: Vectors, ids: Sequence[str]) -> None: ...

    def search(
        self, queries: Vectors, *, top_k: int
    ) -> list[list[tuple[str, float]]]: ...


__all__ = ["FaissFlatIndex", "NumpyFlatIndex"]
