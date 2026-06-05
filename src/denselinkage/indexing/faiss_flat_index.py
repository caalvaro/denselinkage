"""``FaissFlatIndex`` — FAISS-backed index spec (extra: ``[faiss]``)."""

from collections.abc import Sequence

from denselinkage.core.models import RecordId
from denselinkage.core.ports import VectorIndex, Vectors
from denselinkage.indexing.faiss_searchable_index import FaissSearchableIndex


class FaissFlatIndex(VectorIndex):
    """FAISS-backed index spec (extra: ``[faiss]``).

    Planned for a future release; ``build`` raises ``NotImplementedError``. Use
    ``NumpyFlatIndex`` on the dependency-free stack until then.
    """

    def build(self, vectors: Vectors, ids: Sequence[RecordId]) -> FaissSearchableIndex:
        raise NotImplementedError(
            "FaissFlatIndex is planned for a future release; "
            "use NumpyFlatIndex on the dependency-free stack"
        )
