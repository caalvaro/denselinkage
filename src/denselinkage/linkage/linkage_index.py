"""``LinkageIndex`` — prepared linkage state built by ``DenseLinker.index``."""

from denselinkage._reader import RecordReader
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Source
from denselinkage.core.ports import BlockingIndex, Matcher
from denselinkage.core.results import LinkageResult


class LinkageIndex:
    """Prepared linkage state: a built ``BlockingIndex`` fused with a
    ``Matcher``. Constructed by ``DenseLinker.index``; not typically built
    directly. ``kw_only`` for consistency with ``DenseLinker``."""

    def __init__(self, *, blocking_index: BlockingIndex, matcher: Matcher) -> None:
        self._blocking_index = blocking_index
        self._matcher = matcher

    def query(self, source: Source) -> LinkageResult:
        """Query the prepared index with ``source``.

        Raises (from the RecordReader seam): ``UnknownIdColumn`` if
        ``source.id_column`` is absent, ``EmptySource`` if the frame is empty,
        ``DuplicateRecordId`` on duplicate ids, ``DimensionMismatch`` if the
        query embedding width differs from the indexed vectors. All subclass
        ``denselinkage.core.errors.DenseLinkageError``.
        """
        records = RecordReader().read(source)
        pairs = self._blocking_index.query(records)
        outcomes = self._matcher.match(pairs)
        if len(outcomes) != len(pairs):
            raise ValueError(
                f"matcher returned {len(outcomes)} outcomes for {len(pairs)} "
                "pairs; Matcher.match must return exactly one outcome per input "
                "pair, aligned by position"
            )
        decisions: list[tuple[CandidatePair, MatchDecision]] = []
        errors: list[tuple[CandidatePair, MatchError]] = []
        for pair, outcome in zip(pairs, outcomes, strict=True):
            if isinstance(outcome, MatchError):
                errors.append((pair, outcome))
            else:
                decisions.append((pair, outcome))
        return LinkageResult(decisions=tuple(decisions), errors=tuple(errors))
