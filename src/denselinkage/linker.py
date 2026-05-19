"""Orchestration — config in ``DenseLinker``, prepared state in
``LinkageIndex``."""

from collections.abc import Sequence
from dataclasses import dataclass

from denselinkage.core.models import CandidatePair, Source
from denselinkage.core.ports import Blocker, Matcher
from denselinkage.core.results import LinkageResult


class LinkageIndex:
    def __init__(self, blocker: Blocker, matcher: Matcher) -> None: ...

    def query(self, source: Source) -> LinkageResult: ...


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
        """Build the searchable index. Raises ``ValueError`` if ``blocker``
        is ``None``."""
        ...

    def link(self, left: Source, right: Source) -> LinkageResult:
        """Two-table linkage. Raises ``ValueError`` if ``blocker`` is
        ``None``."""
        ...

    def dedupe(self, source: Source) -> LinkageResult:
        """Single-table dedupe (self-pairs suppressed). Raises ``ValueError``
        if ``blocker`` is ``None``."""
        ...

    def match_pairs(self, candidates: Sequence[CandidatePair]) -> LinkageResult:
        """Matcher-only path: score externally supplied candidate pairs (e.g.
        rule-based / pre-blocked) with ``self.matcher``, skipping blocking.
        Does not require ``blocker``. Result flows through the same
        ``LinkageResult`` / metrics path as ``link``."""
        ...


__all__ = ["DenseLinker", "LinkageIndex"]
