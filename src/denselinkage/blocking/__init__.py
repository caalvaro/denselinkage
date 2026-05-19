"""Dense blocking — composes an Embedder and a VectorIndex."""

from collections.abc import Sequence

from denselinkage.core.models import CandidatePair, Record
from denselinkage.core.ports import Blocker, Embedder, VectorIndex


class DenseBlocker(Blocker):
    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_index: VectorIndex,
        similarity_threshold: float = 0.0,
        top_k: int = 10,
    ) -> None: ...

    def index(self, records: Sequence[Record]) -> None: ...

    def query(self, records: Sequence[Record]) -> list[CandidatePair]: ...


__all__ = ["DenseBlocker"]
