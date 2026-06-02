"""Dense blocking — composes an ``Embedder`` and a ``VectorIndex`` spec.

``DenseBlocker`` is a *spec* (``Blocker``): ``build`` indexes the records into a
fresh, immutable ``DenseBlockingIndex`` artifact. The spec is stateless and
reusable; the populated index lives only in the artifact.
"""

from collections.abc import Sequence

from denselinkage.core.errors import InvalidTopK
from denselinkage.core.models import CandidatePair, Record, RecordId
from denselinkage.core.ports import (
    Blocker,
    BlockingIndex,
    Embedder,
    SearchableIndex,
    VectorIndex,
)


class DenseBlocker(Blocker):
    """Dense-blocking spec. ``embedder`` and ``vector_index`` are injected
    independently (the embedder is a pure strategy; the vector index is a spec
    whose ``build`` mints the artifact). ``similarity_threshold`` / ``top_k``
    are defaults that ``DenseBlockingIndex.query`` may override per call."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_index: VectorIndex,
        similarity_threshold: float = 0.0,
        top_k: int = 10,
    ) -> None:
        self._embedder = embedder
        self._vector_index = vector_index
        self._similarity_threshold = similarity_threshold
        self._top_k = top_k

    def build(self, records: Sequence[Record]) -> "DenseBlockingIndex":
        if self._top_k <= 0:
            raise InvalidTopK(f"top_k must be a positive integer, got {self._top_k}")
        vectors = self._embedder.encode([record.text for record in records])
        searchable = self._vector_index.build(
            vectors, [record.id for record in records]
        )
        return DenseBlockingIndex(
            searchable=searchable,
            embedder=self._embedder,
            records_by_id={record.id: record for record in records},
            top_k=self._top_k,
            similarity_threshold=self._similarity_threshold,
        )


class DenseBlockingIndex(BlockingIndex):
    """Immutable artifact built by ``DenseBlocker``: owns the reference
    records' ``SearchableIndex`` and embedder, and generates ``CandidatePair``s
    for a query record set. Each pair is oriented ``record_a`` = indexed
    (left/reference) record, ``record_b`` = query record. ``top_k`` /
    ``similarity_threshold`` default to the originating spec's values and may be
    overridden per query."""

    def __init__(
        self,
        *,
        searchable: SearchableIndex,
        embedder: Embedder,
        records_by_id: dict[RecordId, Record],
        top_k: int,
        similarity_threshold: float,
    ) -> None:
        self._searchable = searchable
        self._embedder = embedder
        self._records_by_id = records_by_id
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    def query(
        self,
        records: Sequence[Record],
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[CandidatePair]:
        k = self._top_k if top_k is None else top_k
        if k <= 0:
            raise InvalidTopK(f"top_k must be a positive integer, got {k}")
        threshold = (
            self._similarity_threshold
            if similarity_threshold is None
            else similarity_threshold
        )
        query_vectors = self._embedder.encode([record.text for record in records])
        neighbours = self._searchable.search(query_vectors, top_k=k)
        pairs: list[CandidatePair] = []
        for query_record, hits in zip(records, neighbours, strict=True):
            for indexed_id, score in hits:
                if score >= threshold:
                    pairs.append(
                        CandidatePair(
                            record_a=self._records_by_id[indexed_id],
                            record_b=query_record,
                            similarity_score=score,
                        )
                    )
        return pairs


__all__ = ["DenseBlocker", "DenseBlockingIndex"]
