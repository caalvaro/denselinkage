"""``FaissFlatIndex`` — FAISS-backed index spec (extra: ``[faiss]``)."""

from collections.abc import Sequence

import numpy as np

from denselinkage._optional import require
from denselinkage.core.models import RecordId
from denselinkage.core.ports import VectorIndex, Vectors
from denselinkage.indexing.faiss_searchable_index import FaissSearchableIndex


class FaissFlatIndex(VectorIndex):
    """FAISS-backed reference index spec — exact (flat) inner-product search
    (extra: ``[faiss]``).

    ``build`` is a factory: it returns a fresh populated ``FaissSearchableIndex``
    and mutates neither ``self`` nor its arguments, so one spec safely builds many
    artifacts. Uses ``faiss.IndexFlatIP`` (inner product, **not** L2) so that for
    the L2-normalized vectors the embedders produce the score equals cosine —
    keeping ``similarity_threshold`` semantics identical to ``NumpyFlatIndex``.
    """

    def build(self, vectors: Vectors, ids: Sequence[RecordId]) -> FaissSearchableIndex:
        # ``require`` raises with an install hint if the extra is missing; the
        # plain ``import`` that follows is what mypy sees (typed ``Any`` via the
        # tool.mypy override) so faiss attribute access type-checks.
        require("faiss")
        import faiss

        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return FaissSearchableIndex(index, ids)
