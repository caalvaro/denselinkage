"""Matchers. ``LangChainMatcher`` is the heavy adapter (extra:
``[langchain]``). The user prompt is only the question; the matcher owns
output and returns typed ``MatchDecision``s."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from denselinkage.core.models import CandidatePair, MatchDecision
from denselinkage.core.ports import Matcher


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 3
    backoff_seconds: float = 0.0


class ThresholdMatcher(Matcher):
    """Dependency-free reference matcher; gates on the carried similarity."""

    def __init__(self, *, threshold: float = 0.5) -> None: ...

    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision]: ...


class LangChainMatcher(Matcher):
    def __init__(
        self,
        *,
        llm: Any,
        prompt: str,
        retry: RetryPolicy | None = None,
        max_concurrency: int = 1,
    ) -> None: ...

    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision]: ...


__all__ = ["LangChainMatcher", "RetryPolicy", "ThresholdMatcher"]
