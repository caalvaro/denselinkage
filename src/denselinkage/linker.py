"""Orchestration — config in ``DenseLinker``, prepared state in
``LinkageIndex``."""

from dataclasses import dataclass

from denselinkage.core.models import Source
from denselinkage.core.ports import Blocker, Matcher
from denselinkage.core.results import LinkageResult


class LinkageIndex:
    def __init__(self, blocker: Blocker, matcher: Matcher) -> None: ...

    def query(self, source: Source) -> LinkageResult: ...


@dataclass(frozen=True, slots=True)
class DenseLinker:
    """Pure config. ``link(a, b) == index(a).query(b)``."""

    blocker: Blocker
    matcher: Matcher

    @classmethod
    def with_defaults(
        cls, *, blocker: Blocker | None = None, matcher: Matcher | None = None
    ) -> "DenseLinker": ...

    def index(self, source: Source) -> LinkageIndex: ...

    def link(self, left: Source, right: Source) -> LinkageResult: ...

    def dedupe(self, source: Source) -> LinkageResult: ...


__all__ = ["DenseLinker", "LinkageIndex"]
