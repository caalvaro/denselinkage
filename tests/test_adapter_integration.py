"""Integration: the heavy index + heavy matcher composing in the real
orchestration path (``DenseLinker.link``).

The per-adapter suites test each adapter in isolation; this proves the
``FaissFlatIndex`` blocker and the ``LangChainMatcher`` compose end to end through
``Source -> serialize -> embed -> FAISS build/query -> candidate pairs -> LLM
match -> LinkageResult``. It runs **fast** — a dependency-free ``HashedNGramEmbedder``
(the ST side is covered by ``test_sentence_transformer_adapter``) and a fake LLM
(no API key) — so it is ``adapter``-marked (needs faiss + langchain) but not ``slow``.
"""

from typing import Any

import pandas as pd
import pytest

from denselinkage import DenseLinker, Source, TemplateSerializer
from denselinkage.blocking import DenseBlocker
from denselinkage.embedding import HashedNGramEmbedder
from denselinkage.indexing import FaissFlatIndex
from denselinkage.matching import LangChainMatcher

pytestmark = pytest.mark.adapter


class _FakeChatModel:
    def __init__(self, responder: Any) -> None:
        self._responder = responder

    def with_structured_output(self, schema: Any) -> Any:
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(self._responder)


def _decide(prompt_value: Any) -> dict[str, Any]:
    # A true pair mentions the same company token in BOTH records, so the token
    # appears (at least) twice across the formatted prompt; a cross pair does not.
    text = prompt_value.to_string().lower()
    is_match = text.count("apple") >= 2 or text.count("microsoft") >= 2
    return {"is_match": is_match, "confidence": None, "rationale": None}


def test_faiss_and_langchain_compose_in_full_linkage() -> None:
    left = pd.DataFrame({"id": ["A1", "A2"], "name": ["Apple Inc", "Microsoft Corp"]})
    right = pd.DataFrame(
        {"id": ["B1", "B2"], "name": ["Apple Incorporated", "Microsoft Corporation"]}
    )

    blocker = DenseBlocker(
        embedder=HashedNGramEmbedder(), vector_index=FaissFlatIndex(), top_k=2
    )
    matcher = LangChainMatcher(
        llm=_FakeChatModel(_decide), prompt="A: {record_a}\nB: {record_b}"
    )
    linker = DenseLinker(blocker=blocker, matcher=matcher)

    serializer = TemplateSerializer("{name}")
    result = linker.link(
        Source(left, id_column="id", serializer=serializer),
        Source(right, id_column="id", serializer=serializer),
    )

    frame = result.to_frame()
    matched = {
        frozenset((row.left_id, row.right_id))
        for row in frame.itertuples()
        if row.match
    }
    # FAISS blocking surfaced the candidates; the LLM matcher matched the true
    # pairs (Apple<->Apple, Microsoft<->Microsoft) and rejected the cross pairs.
    assert frozenset(("A1", "B1")) in matched
    assert frozenset(("A2", "B2")) in matched
    assert frozenset(("A1", "B2")) not in matched
    assert frozenset(("A2", "B1")) not in matched
