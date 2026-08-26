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

from denselinkage.core.errors import DenseLinkageError
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.matching import LangChainMatcher, RetryPolicy

pytestmark = pytest.mark.adapter

_PROMPT = "Same entity?\nA: {record_a}\nB: {record_b}"


class _FakeChatModel:
    """Minimal stand-in for a LangChain chat model. ``with_structured_output``
    ignores the schema and returns a ``RunnableLambda`` over ``responder``, which
    receives the formatted prompt value and returns a verdict dict (or raises).

    ``bind_calls`` counts ``with_structured_output`` invocations, the matcher's
    only touch of the model outside ``match``. The prompt-validation tests read
    it to show that a rejected template set up no LLM work at all."""

    def __init__(self, responder: Callable[[Any], Any]) -> None:
        self._responder = responder
        self.bind_calls = 0

    def with_structured_output(self, schema: Any) -> Any:
        # Imported lazily (not at module top) so collecting this file pulls no
        # heavy backend into sys.modules — the dependency-cut test runs in-process.
        from langchain_core.runnables import RunnableLambda

        self.bind_calls += 1
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


def test_unknown_prompt_placeholder_raises_value_error() -> None:
    """Tier 3: a placeholder ``match`` cannot supply is the caller's bug.

    ``match`` renders with exactly ``record_a`` / ``record_b``, so ``{record_left}``
    can never bind. That is API misuse, and API misuse is a plain ``ValueError``
    sitting outside ``DenseLinkageError`` on purpose (AGENTS.md, "Failure tiers,
    disjoint on purpose"), so a caller's ``except DenseLinkageError`` around data
    handling cannot swallow it.
    """
    with pytest.raises(ValueError) as excinfo:
        LangChainMatcher(
            llm=_FakeChatModel(lambda _pv: {"is_match": True}),
            prompt="Same entity? A: {record_a} B: {record_left}",
        )

    assert not isinstance(excinfo.value, DenseLinkageError)
    message = str(excinfo.value)
    assert "{record_left}" in message  # the offending placeholder
    assert "{record_a}" in message and "{record_b}" in message  # the two available


def test_unknown_prompt_placeholder_message_names_the_escape() -> None:
    """The likeliest cause of an unexpected placeholder is a literal brace.

    A prompt carrying a JSON or code example trips this check, and the remedy is
    to double the braces rather than to rename anything. The ``KeyError`` this
    replaces said so; the ``ValueError`` has to say so too, or it is the less
    useful of the two on the case that fires most.
    """
    with pytest.raises(ValueError) as excinfo:
        LangChainMatcher(
            llm=_FakeChatModel(lambda _pv: {"is_match": True}),
            prompt='Compare {record_a} and {record_b}. Reply like {"is_match": true}',
        )

    assert "{{ }}" in str(excinfo.value)


def test_positional_placeholder_is_named_in_the_message() -> None:
    # `{}` parses to an empty variable name; reporting the bare name would print
    # `''` and point at nothing, so offenders are rendered as written.
    with pytest.raises(ValueError) as excinfo:
        LangChainMatcher(
            llm=_FakeChatModel(lambda _pv: {"is_match": True}), prompt="{}"
        )

    assert "prompt references {}," in str(excinfo.value)


def test_unknown_prompt_placeholder_reports_every_offender() -> None:
    with pytest.raises(ValueError) as excinfo:
        LangChainMatcher(
            llm=_FakeChatModel(lambda _pv: {"is_match": True}),
            prompt="{record_left} {record_right} {{record_middle}}",
        )

    message = str(excinfo.value)
    assert "{record_left}" in message and "{record_right}" in message
    # Escaped braces are literal text, so the name inside them is not an offender.
    # It has to be a name outside _PROMPT_FIELDS for this to test anything: an
    # escaped `{{record_a}}` would be filtered out either way.
    assert "record_middle" not in message


def test_unknown_prompt_placeholder_never_reaches_the_model() -> None:
    """Rejecting at construction leaves no ``match`` call to produce a
    ``MatchError``.

    Before this check the render failure was caught per pair by
    ``batch(..., return_exceptions=True)``, retried ``max_retries + 1`` times
    with backoff, and returned as a ``MatchError``, so a caller's typo was
    counted in ``LinkageMetrics.n_errors`` as model unreliability. The template
    renders upstream of the model, so those attempts cost wall-clock rather than
    tokens; the defect is the tier violation and the wasted attempts come with
    it. The ``retry`` below is inert on purpose: construction raises before it is
    ever stored, which is the point.
    """
    calls = {"n": 0}

    def responder(prompt_value: Any) -> dict[str, Any]:
        calls["n"] += 1
        return {"is_match": True, "confidence": None, "rationale": None}

    llm = _FakeChatModel(responder)
    with pytest.raises(ValueError):
        LangChainMatcher(
            llm=llm, prompt="A: {record_left}", retry=RetryPolicy(max_retries=5)
        )

    assert llm.bind_calls == 0  # with_structured_output never ran
    assert calls["n"] == 0  # no attempt, so no retry budget consumed


def test_valid_prompt_binds_structured_output_once() -> None:
    # Positive control for the `bind_calls == 0` assertions above: without it
    # they would also pass if the matcher stopped binding the model at all.
    llm = _FakeChatModel(lambda _pv: {"is_match": True})
    LangChainMatcher(llm=llm, prompt=_PROMPT)

    assert llm.bind_calls == 1


@pytest.mark.parametrize(
    ("prompt", "expected_tail"),
    [("only {record_b}", "only b"), ("Same entity?", "Same entity?")],
)
def test_prompt_using_fewer_pair_fields_is_accepted(
    prompt: str, expected_tail: str
) -> None:
    """One of the two fields, or neither, renders and is accepted (issue #51).

    An unused input key is ignored at render, so such a template is not the
    laundered-render-failure defect and stays accepted. Both halves of that
    documented sentence are pinned here, so narrowing to strict equality later
    has to be a deliberate edit.
    """
    rendered: list[str] = []

    def responder(prompt_value: Any) -> dict[str, Any]:
        # Recorded, not asserted, here: an assertion inside this callback would
        # be caught by batch(return_exceptions=True), retried, and reported as a
        # MatchError instead of as the failure it is.
        rendered.append(prompt_value.to_string())
        return {"is_match": True, "confidence": None, "rationale": None}

    matcher = LangChainMatcher(llm=_FakeChatModel(responder), prompt=prompt)
    [outcome] = matcher.match([_pair("a-text", "b")])

    assert isinstance(outcome, MatchDecision)
    # `endswith` rather than equality: the role prefix is LangChain's rendering
    # of a HumanMessage, not behaviour this adapter owns.
    assert len(rendered) == 1 and rendered[0].endswith(expected_tail)
    assert "a-text" not in rendered[0]  # record_a really was not substituted
