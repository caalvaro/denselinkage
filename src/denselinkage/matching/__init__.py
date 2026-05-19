"""Matchers. ``LangChainMatcher`` is the heavy adapter (extra:
``[langchain]``). The user prompt is only the question; the matcher owns
output and returns typed ``MatchDecision``s."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from denselinkage.core.models import CandidatePair, MatchDecision, MatchError
from denselinkage.core.ports import Matcher


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 3
    backoff_seconds: float = 0.0


class ThresholdMatcher(Matcher):
    """Dependency-free reference matcher; gates on the carried similarity."""

    def __init__(self, *, threshold: float = 0.5) -> None: ...

    def match(
        self, pairs: Sequence[CandidatePair]
    ) -> list[MatchDecision | MatchError]: ...


class LangChainMatcher(Matcher):
    """LLM matcher (extra: ``[langchain]``).

    Prompt/output contract (pinned; A7 implements against this): ``prompt``
    carries ONLY the semantic question and may reference the pair fields. The
    response *structure* is framework-owned — the matcher binds structured
    output and returns typed ``MatchDecision``s; callers never parse text and
    the prompt never asks for a format. On exhausted ``retry`` the matcher
    yields a ``MatchError(reason=...)`` for that pair (aligned by position),
    never raising into the batch.
    """

    def __init__(
        self,
        *,
        llm: Any,
        prompt: str,
        retry: RetryPolicy | None = None,
        max_concurrency: int = 1,
    ) -> None: ...

    def match(
        self, pairs: Sequence[CandidatePair]
    ) -> list[MatchDecision | MatchError]: ...


__all__ = ["LangChainMatcher", "RetryPolicy", "ThresholdMatcher"]
