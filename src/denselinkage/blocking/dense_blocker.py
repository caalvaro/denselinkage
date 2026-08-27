"""``DenseBlocker`` — the dense-blocking spec."""

from collections.abc import Sequence

from denselinkage.blocking.dense_blocking_index import DenseBlockingIndex
from denselinkage.core.errors import InvalidTopK
from denselinkage.core.models import Record
from denselinkage.core.ports import Blocker, Embedder, VectorIndex


class DenseBlocker(Blocker):
    """Dense-blocking spec. ``embedder`` and ``vector_index`` are injected
    independently (the embedder is a pure strategy; the vector index is a spec
    whose ``build`` mints the artifact). ``similarity_threshold`` / ``top_k``
    are defaults that ``DenseBlockingIndex.query`` may override per call.
    ``batch_size`` is forwarded to the embedder on both paths: encoding the
    record corpus here, and encoding every query set the built index is given."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_index: VectorIndex,
        similarity_threshold: float = 0.0,
        top_k: int = 10,
        batch_size: int | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_index = vector_index
        self._similarity_threshold = similarity_threshold
        self._top_k = top_k
        self._batch_size = batch_size

    def build(self, records: Sequence[Record]) -> DenseBlockingIndex:
        if self._top_k <= 0:
            raise InvalidTopK(f"top_k must be a positive integer, got {self._top_k}")
        vectors = self._embedder.encode(
            [record.text for record in records], batch_size=self._batch_size
        )
        searchable = self._vector_index.build(
            vectors, [record.id for record in records]
        )
        return DenseBlockingIndex(
            searchable=searchable,
            embedder=self._embedder,
            records_by_id={record.id: record for record in records},
            top_k=self._top_k,
            similarity_threshold=self._similarity_threshold,
            batch_size=self._batch_size,
        )
