"""``LinkageIndex`` — prepared linkage state built by ``DenseLinker.index``."""

from pathlib import Path

from denselinkage._reader import RecordReader
from denselinkage.core.models import CandidatePair, Source
from denselinkage.core.ports import BlockingIndex, Embedder, Matcher
from denselinkage.core.results import LinkageResult
from denselinkage.linkage._assembly import assemble_linkage_result


class LinkageIndex:
    """Prepared linkage state: a built ``BlockingIndex`` fused with a
    ``Matcher``. Constructed by ``DenseLinker.index``; not typically built
    directly. Its ``__init__`` is keyword-only, for consistency with
    ``DenseLinker``. It is a plain class rather than a dataclass: the private
    attribute names and ``save``/``load`` depend on that."""

    def __init__(self, *, blocking_index: BlockingIndex, matcher: Matcher) -> None:
        self._blocking_index = blocking_index
        self._matcher = matcher

    def candidates(
        self,
        source: Source,
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[CandidatePair]:
        """Blocking-only counterpart to :meth:`query`: read ``source`` and return
        the blocker's ``CandidatePair`` objects (``record_a`` = indexed,
        ``record_b`` = from ``source``) *without* running the matcher — the
        ergonomic input to ``blocking_metrics`` / ``pair_completeness_at_k``.

        ``top_k`` / ``similarity_threshold`` override the built blocker's spec for
        this call only (e.g. a large ``top_k`` to sweep pair-completeness over
        several k), reusing this prepared index instead of rebuilding it.
        ``InvalidTopK`` if an override ``top_k <= 0``; otherwise the same
        ``RecordReader`` failure modes as :meth:`query`.
        """
        records = RecordReader().read(source)
        return self._blocking_index.query(
            records, top_k=top_k, similarity_threshold=similarity_threshold
        )

    def query(self, source: Source) -> LinkageResult:
        """Query the prepared index with ``source``.

        Raises (from the RecordReader seam): ``UnknownIdColumn`` if
        ``source.id_column`` is absent, ``EmptySource`` if the frame is empty,
        ``DuplicateRecordId`` on duplicate ids, ``DimensionMismatch`` if the
        query embedding width differs from the indexed vectors. All subclass
        ``denselinkage.core.errors.DenseLinkageError``.
        """
        return assemble_linkage_result(self.candidates(source), self._matcher)

    def save(self, path: str | Path) -> None:
        """Persist this prepared index to ``path`` (a directory) so the reference
        set can be reused later without re-embedding. Supported for the
        dependency-free reference stack (``DenseBlocker`` over ``NumpyFlatIndex``);
        other backends raise ``NotImplementedError``. The matcher is **not**
        persisted — supply one (and the matching embedder) to :meth:`load`.
        """
        from denselinkage._store import save_reference_index

        save_reference_index(self._blocking_index, path)

    @classmethod
    def load(
        cls, path: str | Path, *, embedder: Embedder, matcher: Matcher
    ) -> "LinkageIndex":
        """Reload an index saved by :meth:`save`, re-supplying the live
        ``embedder`` and ``matcher``. The ``embedder`` must match the stored
        provenance (``model_id`` / ``embedding_dim``) or
        ``denselinkage.core.errors.IncompatibleStore`` is raised — a persisted
        index cannot be queried with a different embedding model.
        """
        from denselinkage._store import load_reference_index

        blocking_index = load_reference_index(path, embedder=embedder)
        return cls(blocking_index=blocking_index, matcher=matcher)
