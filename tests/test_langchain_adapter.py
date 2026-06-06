"""``LangChainMatcher`` behaviour (extra: ``[langchain]``).

Adapter-marked: needs ``langchain_core`` installed, but **no** API key — a fake
chat model stands in for the LLM so the match / retry / error-handling logic is
exercised hermetically. The fake's ``with_structured_output`` returns a
``RunnableLambda`` that maps each formatted prompt to a canned verdict (or
raises), mirroring how a real structured-output chain behaves.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.matching import LangChainMatcher, RetryPolicy

pytestmark = pytest.mark.adapter

_PROMPT = "Same entity?\nA: {record_a}\nB: {record_b}"


class _FakeChatModel:
    """Minimal stand-in for a LangChain chat model. ``with_structured_output``
    ignores the schema and returns a ``RunnableLambda`` over ``responder``, which
    receives the formatted prompt value and returns a verdict dict (or raises)."""

    def __init__(self, responder: Callable[[Any], Any]) -> None:
        self._responder = responder

    def with_structured_output(self, schema: Any) -> Any:
        # Imported lazily (not at module top) so collecting this file pulls no
        # heavy backend into sys.modules — the dependency-cut test runs in-process.
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(self._responder)


def _pair(a_text: str, b_text: str) -> CandidatePair:
    return CandidatePair(
        record_a=Record(id="a", text=a_text),
        record_b=Record(id="b", text=b_text),
    )


def test_match_returns_position_aligned_decisions() -> None:
    def responder(prompt_value: Any) -> dict[str, Any]:
        same = "yes" in prompt_value.to_string()
        return {"is_match": same, "confidence": None, "rationale": None}

    matcher = LangChainMatcher(llm=_FakeChatModel(responder), prompt=_PROMPT)
    outcomes = matcher.match([_pair("yes", "yes"), _pair("no", "different")])

    assert all(isinstance(outcome, MatchDecision) for outcome in outcomes)
    assert [outcome.is_match for outcome in outcomes] == [True, False]  # type: ignore[union-attr]


def test_match_passes_through_confidence_and_rationale() -> None:
    def responder(prompt_value: Any) -> dict[str, Any]:
        return {"is_match": True, "confidence": 0.87, "rationale": "same address"}

    matcher = LangChainMatcher(llm=_FakeChatModel(responder), prompt=_PROMPT)
    [outcome] = matcher.match([_pair("x", "y")])

    assert isinstance(outcome, MatchDecision)
    assert outcome.confidence == 0.87
    assert outcome.rationale == "same address"


def test_match_converts_exhausted_retries_to_match_error() -> None:
    def responder(prompt_value: Any) -> dict[str, Any]:
        if "boom" in prompt_value.to_string():
            raise RuntimeError("kaboom")
        return {"is_match": True, "confidence": None, "rationale": None}

    matcher = LangChainMatcher(
        llm=_FakeChatModel(responder), prompt=_PROMPT, retry=RetryPolicy(max_retries=2)
    )
    outcomes = matcher.match([_pair("ok-a", "ok-b"), _pair("boom", "x")])

    assert isinstance(outcomes[0], MatchDecision)  # one bad pair does not abort
    assert isinstance(outcomes[1], MatchError)
    assert "attempt" in outcomes[1].reason
    assert "kaboom" in outcomes[1].reason


def test_match_fail_fast_with_zero_retries() -> None:
    calls = {"n": 0}

    def responder(prompt_value: Any) -> dict[str, Any]:
        calls["n"] += 1
        raise RuntimeError("nope")

    matcher = LangChainMatcher(
        llm=_FakeChatModel(responder), prompt=_PROMPT, retry=RetryPolicy(max_retries=0)
    )
    [outcome] = matcher.match([_pair("a", "b")])

    assert isinstance(outcome, MatchError)
    assert calls["n"] == 1  # fail fast: a single attempt, no retry
    assert "1 attempt" in outcome.reason


def test_match_retries_transient_failure_then_succeeds() -> None:
    calls = {"n": 0}

    def responder(prompt_value: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return {"is_match": True, "confidence": None, "rationale": None}

    # backoff_seconds > 0 exercises the inter-attempt sleep on the retry path.
    matcher = LangChainMatcher(
        llm=_FakeChatModel(responder),
        prompt=_PROMPT,
        retry=RetryPolicy(max_retries=3, backoff_seconds=0.01),
    )
    [outcome] = matcher.match([_pair("a", "b")])

    assert isinstance(outcome, MatchDecision)
    assert outcome.is_match is True
    assert calls["n"] == 2  # failed once, retried, succeeded


def test_match_reads_object_schema_results() -> None:
    # A pydantic-style schema yields attribute objects (not dicts); the matcher
    # reads either shape.
    def responder(prompt_value: Any) -> SimpleNamespace:
        return SimpleNamespace(is_match=True, confidence=0.5, rationale="obj")

    matcher = LangChainMatcher(llm=_FakeChatModel(responder), prompt=_PROMPT)
    [outcome] = matcher.match([_pair("x", "y")])

    assert isinstance(outcome, MatchDecision)
    assert outcome.is_match is True
    assert outcome.confidence == 0.5
    assert outcome.rationale == "obj"


def test_match_empty_pairs_returns_empty() -> None:
    matcher = LangChainMatcher(
        llm=_FakeChatModel(lambda _pv: {"is_match": True}), prompt=_PROMPT
    )
    assert matcher.match([]) == []
