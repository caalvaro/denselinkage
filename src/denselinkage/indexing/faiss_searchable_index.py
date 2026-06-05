"""``FaissSearchableIndex`` — artifact built by ``FaissFlatIndex``."""

from collections.abc import Sequence

from denselinkage.core.models import RecordId
from denselinkage.core.ports import SearchableIndex, Vectors


class FaissSearchableIndex(SearchableIndex):
    """Immutable artifact built by ``FaissFlatIndex``.

    Planned for a future release; ``search`` / ``extended`` raise
    ``NotImplementedError``. Use ``NumpyFlatIndex`` on the dependency-free stack
    until then.
    """

    def search(
        self, queries: Vectors, *, top_k: int
    ) -> list[list[tuple[RecordId, float]]]:
        raise NotImplementedError(
            "FaissSearchableIndex is planned for a future release; "
            "use NumpyFlatIndex on the dependency-free stack"
        )

    def extended(
        self, vectors: Vectors, ids: Sequence[RecordId]
    ) -> "FaissSearchableIndex":
        raise NotImplementedError(
            "FaissSearchableIndex is planned for a future release; "
            "use NumpyFlatIndex on the dependency-free stack"
        )
