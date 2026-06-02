"""Dense blocking — composes an ``Embedder`` and a ``VectorIndex`` spec.

``DenseBlocker`` is a *spec* (``Blocker``): ``build`` indexes the records into a
fresh, immutable ``DenseBlockingIndex`` artifact. The spec is stateless and
reusable; the populated index lives only in the artifact.
"""

from collections.abc import Sequence

from denselinkage.core.models import CandidatePair, Record
from denselinkage.core.ports import Blocker, BlockingIndex, Embedder, VectorIndex


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
    ) -> None: ...

    def build(self, records: Sequence[Record]) -> "DenseBlockingIndex": ...


class DenseBlockingIndex(BlockingIndex):
    """Immutable artifact built by ``DenseBlocker``: owns the reference
    records' ``SearchableIndex`` and embedder, and generates ``CandidatePair``s
    for a query record set. ``top_k`` / ``similarity_threshold`` default to the
    originating spec's values and may be overridden per query."""

    def query(
        self,
        records: Sequence[Record],
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[CandidatePair]: ...


__all__ = ["DenseBlocker", "DenseBlockingIndex"]
