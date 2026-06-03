"""``DenseBlockingIndex`` — the artifact built by ``DenseBlocker``."""

from collections.abc import Sequence

from denselinkage.core.errors import InvalidTopK
from denselinkage.core.models import CandidatePair, Record, RecordId
from denselinkage.core.ports import BlockingIndex, Embedder, SearchableIndex


class DenseBlockingIndex(BlockingIndex):
    """Immutable artifact built by ``DenseBlocker`` for one reference set.

    Owns the reference records' ``SearchableIndex`` and embedder, and generates
    ``CandidatePair`` objects for a query record set. Each pair is oriented
    ``record_a`` = indexed (left/reference) record, ``record_b`` = query record.
    ``top_k`` / ``similarity_threshold`` default to the originating spec's values
    and may be overridden per query."""

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
