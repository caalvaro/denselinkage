"""``LangChainMatcher`` — LLM matcher (extra: ``[langchain]``)."""

import time
from collections.abc import Sequence
from typing import Any, TypedDict

from denselinkage._optional import require
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError
from denselinkage.core.ports import Matcher
from denselinkage.matching.retry_policy import RetryPolicy


class _Decision(TypedDict):
    """Structured verdict the LLM returns for one candidate pair. Bound via
    ``with_structured_output`` so the framework owns the response shape — the
    prompt never asks for a format and callers never parse text."""

    is_match: bool
    confidence: float | None
    rationale: str | None


def _field(result: Any, name: str) -> Any:
    """Read ``name`` from a structured-output result that may be a dict (TypedDict
    schema) or an object (pydantic schema)."""
    if isinstance(result, dict):
        return result.get(name)
    return getattr(result, name, None)


class LangChainMatcher(Matcher):
    """LLM matcher (extra: ``[langchain]``).

    The ``llm`` is injected (any LangChain chat model); model / operational /
    domain config stay separate. Prompt/output contract: ``prompt`` carries ONLY
    the semantic question and may reference the pair fields (``{record_a}`` /
    ``{record_b}``). The response *structure* is framework-owned — the matcher
    binds structured output and returns typed ``MatchDecision``s; callers never
    parse text and the prompt never asks for a format. On exhausted ``retry`` the
    matcher yields a ``MatchError(reason=...)`` for that pair (aligned by
    position), never raising into the batch.
    """

    def __init__(
        self,
        *,
        llm: Any,
        prompt: str,
        retry: RetryPolicy | None = None,
        max_concurrency: int = 1,
    ) -> None:
        require("langchain_core")
        from langchain_core.prompts import ChatPromptTemplate

        self._retry = retry or RetryPolicy()
        self._max_concurrency = max_concurrency
        structured_llm = llm.with_structured_output(_Decision)
        self._chain = ChatPromptTemplate.from_template(prompt) | structured_llm

    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision | MatchError]:
        inputs = [
            {"record_a": pair.record_a.text, "record_b": pair.record_b.text}
            for pair in pairs
        ]
        decisions: dict[int, MatchDecision] = {}
        last_error: dict[int, str] = {}
        pending = list(range(len(pairs)))
        for attempt in range(self._retry.max_retries + 1):
            if not pending:
                break
            if attempt and self._retry.backoff_seconds:
                time.sleep(self._retry.backoff_seconds)
            batch_results = self._chain.batch(
                [inputs[i] for i in pending],
                config={"max_concurrency": self._max_concurrency},
                return_exceptions=True,
            )
            retry_next: list[int] = []
            for i, result in zip(pending, batch_results, strict=True):
                if isinstance(result, Exception):
                    last_error[i] = repr(result)  # always non-empty + typed
                    retry_next.append(i)
                else:
                    decisions[i] = MatchDecision(
                        is_match=bool(_field(result, "is_match")),
                        confidence=_field(result, "confidence"),
                        rationale=_field(result, "rationale"),
                    )
            pending = retry_next

        outcomes: list[MatchDecision | MatchError] = []
        for i in range(len(pairs)):
            decision = decisions.get(i)
            if decision is not None:
                outcomes.append(decision)
            else:
                outcomes.append(
                    MatchError(
                        reason="LLM matching failed after "
                        f"{self._retry.max_retries + 1} attempt(s): "
                        f"{last_error.get(i, 'unknown error')}"
                    )
                )
        return outcomes
