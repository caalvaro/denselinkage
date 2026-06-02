"""``FaissSearchableIndex`` — artifact built by ``FaissFlatIndex``."""

from collections.abc import Sequence

from denselinkage.core.models import RecordId
from denselinkage.core.ports import SearchableIndex, Vectors


class FaissSearchableIndex(SearchableIndex):
    """Immutable artifact built by ``FaissFlatIndex``."""

    def search(
        self, queries: Vectors, *, top_k: int
    ) -> list[list[tuple[RecordId, float]]]: ...

    def extended(
        self, vectors: Vectors, ids: Sequence[RecordId]
    ) -> "FaissSearchableIndex": ...
