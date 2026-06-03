"""``LangChainMatcher`` — LLM matcher (extra: ``[langchain]``)."""

from collections.abc import Sequence
from typing import Any

from denselinkage.core.models import CandidatePair, MatchDecision, MatchError
from denselinkage.core.ports import Matcher
from denselinkage.matching.retry_policy import RetryPolicy


class LangChainMatcher(Matcher):
    """LLM matcher (extra: ``[langchain]``).

    Prompt/output contract: ``prompt``
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
