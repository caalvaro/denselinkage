"""``DenseLinker`` — the immutable orchestration config."""

from collections.abc import Sequence
from dataclasses import dataclass

from denselinkage._reader import RecordReader
from denselinkage.core.models import CandidatePair, Source
from denselinkage.core.ports import Blocker, Matcher
from denselinkage.core.results import LinkageResult
from denselinkage.linkage.linkage_index import LinkageIndex


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
    ) -> "DenseLinker":
        """Low-floor entry point: wire the dependency-free reference stack
        (``HashedNGramEmbedder`` + ``NumpyFlatIndex`` behind ``DenseBlocker``,
        plus ``ThresholdMatcher``). Pass ``blocker=`` / ``matcher=`` to override
        either half. The default stack is lexical (character n-gram hashing) —
        it recovers abbreviations/punctuation/typos, not semantic renames.
        Imports are local so ``import denselinkage`` stays light.
        """
        if blocker is None:
            from denselinkage.blocking import DenseBlocker
            from denselinkage.embedding import HashedNGramEmbedder
            from denselinkage.indexing import NumpyFlatIndex

            blocker = DenseBlocker(
                embedder=HashedNGramEmbedder(n_features=1024, ngram=3),
                vector_index=NumpyFlatIndex(),
                similarity_threshold=0.0,
                top_k=5,
            )
        if matcher is None:
            from denselinkage.matching import ThresholdMatcher

            matcher = ThresholdMatcher(threshold=0.5)
        return cls(blocker=blocker, matcher=matcher)

    def index(self, source: Source) -> LinkageIndex:
        """Build the prepared linkage state by delegating indexing to
        ``self.blocker.build`` (which returns a fresh ``BlockingIndex`` — this
        frozen config is never mutated) and composing it with ``self.matcher``.

        Raises ``ValueError`` if ``blocker`` is ``None``. From the RecordReader
        seam: ``UnknownIdColumn``, ``EmptySource``, ``DuplicateRecordId``;
        ``InvalidTopK`` if the blocker's ``top_k <= 0``; ``DimensionMismatch``
        if the embedder width differs from the index. All
        ``denselinkage.core.errors`` subclasses.
        """
        if self.blocker is None:
            raise ValueError(
                "index() requires a blocker; construct the linker with one "
                "(DenseLinker(blocker=..., matcher=...) or "
                "DenseLinker.with_defaults())"
            )
        records = RecordReader().read(source)
        blocking_index = self.blocker.build(records)
        return LinkageIndex(blocking_index=blocking_index, matcher=self.matcher)

    def link(self, left: Source, right: Source) -> LinkageResult:
        """Two-table linkage.

        Raises ``ValueError`` if ``blocker`` is ``None``; otherwise the same
        ``denselinkage.core.errors`` taxonomy as ``index`` (``UnknownIdColumn``,
        ``EmptySource``, ``DuplicateRecordId``, ``InvalidTopK``,
        ``DimensionMismatch``), evaluated for each of ``left``/``right``.
        """
        return self.index(left).query(right)

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
