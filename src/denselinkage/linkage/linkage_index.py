"""``LinkageIndex`` — prepared linkage state built by ``DenseLinker.index``."""

from denselinkage._reader import RecordReader
from denselinkage.core.models import CandidatePair, Source
from denselinkage.core.ports import BlockingIndex, Matcher
from denselinkage.core.results import LinkageResult
from denselinkage.linkage._assembly import assemble_linkage_result


class LinkageIndex:
    """Prepared linkage state: a built ``BlockingIndex`` fused with a
    ``Matcher``. Constructed by ``DenseLinker.index``; not typically built
    directly. ``kw_only`` for consistency with ``DenseLinker``."""

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
