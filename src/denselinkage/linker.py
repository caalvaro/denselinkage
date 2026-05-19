"""Orchestration — config in ``DenseLinker``, prepared state in
``LinkageIndex``.

Source -> Record materialization (resolving ``serializer=None`` and validating
the frame) is performed by the internal ``denselinkage._reader.RecordReader``
seam; the hard failures below come from there. Soft per-pair matcher
failures are ``MatchError`` in ``LinkageResult.errors``, never exceptions.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from denselinkage.core.models import CandidatePair, Source
from denselinkage.core.ports import Blocker, Matcher
from denselinkage.core.results import LinkageResult


class LinkageIndex:
    """Prepared (indexed) state. Constructed by ``DenseLinker.index``; not
    typically built directly. ``kw_only`` for consistency with
    ``DenseLinker``."""

    def __init__(self, *, blocker: Blocker, matcher: Matcher) -> None: ...

    def query(self, source: Source) -> LinkageResult:
        """Query the prepared index with ``source``.

        Raises (from the RecordReader seam): ``UnknownIdColumn`` if
        ``source.id_column`` is absent, ``EmptySource`` if the frame is empty,
        ``DuplicateRecordId`` on duplicate ids, ``DimensionMismatch`` if the
        query embedding width differs from the indexed vectors. All subclass
        ``denselinkage.core.errors.DenseLinkageError``.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class DenseLinker:
    """Pure config. ``link(a, b) == index(a).query(b)``.

    ``kw_only`` so ``blocker`` is optional while ``matcher`` stays required
    (and to forbid positional ambiguity — callers always write
    ``DenseLinker(blocker=..., matcher=...)``).

    ``blocker`` is optional: developers who already have candidate pairs from
    rule-based / external blocking have no blocker. ``link``/``dedupe``/
    ``index`` require one and raise ``ValueError`` if it is ``None``;
    ``match_pairs`` does not (it is inference with no blocking step — no
    learning on the linker, so the immutable-config contract holds).
    ``with_defaults`` always yields a linker with a blocker.
    """

    blocker: Blocker | None = None
    matcher: Matcher

    @classmethod
    def with_defaults(
        cls, *, blocker: Blocker | None = None, matcher: Matcher | None = None
    ) -> "DenseLinker": ...

    def index(self, source: Source) -> LinkageIndex:
        """Build the searchable index.

        Raises ``ValueError`` if ``blocker`` is ``None``. From the RecordReader
        seam: ``UnknownIdColumn``, ``EmptySource``, ``DuplicateRecordId``;
        ``InvalidTopK`` if the blocker's ``top_k <= 0``; ``DimensionMismatch``
        if the embedder width differs from the index. All
        ``denselinkage.core.errors`` subclasses.
        """
        ...

    def link(self, left: Source, right: Source) -> LinkageResult:
        """Two-table linkage.

        Raises ``ValueError`` if ``blocker`` is ``None``; otherwise the same
        ``denselinkage.core.errors`` taxonomy as ``index`` (``UnknownIdColumn``,
        ``EmptySource``, ``DuplicateRecordId``, ``InvalidTopK``,
        ``DimensionMismatch``), evaluated for each of ``left``/``right``.
        """
        ...

    def dedupe(self, source: Source) -> LinkageResult:
        """Single-table dedupe (self-pairs suppressed).

        Raises ``ValueError`` if ``blocker`` is ``None``; otherwise the same
        ``denselinkage.core.errors`` taxonomy as ``index``.
        """
        ...

    def match_pairs(self, candidates: Sequence[CandidatePair]) -> LinkageResult:
        """Matcher-only path: score externally supplied candidate pairs (e.g.
        rule-based / pre-blocked) with ``self.matcher``, skipping blocking.
        Does not require ``blocker``. Result flows through the same
        ``LinkageResult`` / metrics path as ``link``.

        The ergonomic ``DataFrame -> CandidatePair`` constructor
        (``LinkageResult.from_candidate_frame``) is a **Phase-B** addition, so
        this path is contract-complete now but only end-to-end usable once B
        lands. Raises no Source-validation errors (it takes pre-built
        ``CandidatePair``s, whose ``similarity_score`` may be ``None``);
        backend matcher failures surface per-pair as ``MatchError``, never as
        exceptions.
        """
        ...


__all__ = ["DenseLinker", "LinkageIndex"]
