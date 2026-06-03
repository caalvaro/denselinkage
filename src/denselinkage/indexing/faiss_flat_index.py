"""``FaissFlatIndex`` — FAISS-backed index spec (extra: ``[faiss]``)."""

from collections.abc import Sequence

from denselinkage.core.models import RecordId
from denselinkage.core.ports import VectorIndex, Vectors
from denselinkage.indexing.faiss_searchable_index import FaissSearchableIndex


class FaissFlatIndex(VectorIndex):
    """FAISS-backed index spec (extra: ``[faiss]``)."""

    def build(
        self, vectors: Vectors, ids: Sequence[RecordId]
    ) -> FaissSearchableIndex: ...
