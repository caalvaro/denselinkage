"""``LangChainMatcher`` — LLM matcher (extra: ``[langchain]``)."""

import time
from collections.abc import Callable, Sequence
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


# The single definition of what a prompt may reference: the keys are the names
# `__init__` validates a template against, and the values are what `match` binds
# for each pair. Two hand-written copies would drift, and drift in either
# direction re-creates issue #51 at the site meant to prevent it.
_PROMPT_FIELDS: dict[str, Callable[[CandidatePair], str]] = {
    "record_a": lambda pair: pair.record_a.text,
    "record_b": lambda pair: pair.record_b.text,
}


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

    ``__init__`` raises plain ``ValueError`` (API misuse, tier 3) if ``prompt``
    references any placeholder other than ``{record_a}`` / ``{record_b}``. Such a
    template cannot render for a single pair. Left to fail at call time it would
    spend the whole retry budget per pair and surface as a ``MatchError``, which
    is reserved for a pair the matcher could not decide. Using only one of the
    two, or neither, renders and is accepted.
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

        template = ChatPromptTemplate.from_template(prompt)
        # Checked here, not in match(): the template is fixed at construction, so
        # a bad one is a caller bug rather than a per-pair outcome. Ordered before
        # the with_structured_output bind, so a rejected prompt leaves `llm` alone.
        unknown = sorted(set(template.input_variables) - _PROMPT_FIELDS.keys())
        if unknown:
            # Rendered as written ("{record_left}", not "record_left") so a
            # positional "{}" reads as itself and stray whitespace is visible.
            offenders = ", ".join(f"{{{name}}}" for name in unknown)
            available = ", ".join(f"{{{name}}}" for name in sorted(_PROMPT_FIELDS))
            raise ValueError(
                f"prompt references {offenders}, which match() cannot supply for "
                f"a pair; it may only reference {available}. Escape a literal "
                "brace as {{ }} if that text was not meant as a placeholder."
            )
        self._retry = retry or RetryPolicy()
        self._max_concurrency = max_concurrency
        structured_llm = llm.with_structured_output(_Decision)
        self._chain = template | structured_llm

    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision | MatchError]:
        inputs = [
            {name: read(pair) for name, read in _PROMPT_FIELDS.items()}
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
