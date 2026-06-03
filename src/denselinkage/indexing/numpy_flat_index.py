"""``NumpyFlatIndex`` — the dependency-free reference index spec."""

from collections.abc import Sequence

from denselinkage.core.models import RecordId
from denselinkage.core.ports import VectorIndex, Vectors
from denselinkage.indexing.numpy_searchable_index import NumpySearchableIndex


class NumpyFlatIndex(VectorIndex):
    """Dependency-free reference index spec (brute-force / flat search)."""

    def build(self, vectors: Vectors, ids: Sequence[RecordId]) -> NumpySearchableIndex:
        return NumpySearchableIndex(vectors, ids)
