"""Example 01 — End-to-End Dense Linkage (full control).

Explicitly assembled components: a dense blocker (SentenceTransformer
embeddings + a FAISS index) then an LLM matcher. Vector indexes live in
``denselinkage.indexing`` (their own port, parallel to embedders — M1).

The prompt carries ONLY the semantic question (H2): the matcher owns output
structure and returns typed ``MatchDecision``s, so a brittle "Answer YES or
NO" instruction is neither needed nor wanted.

NOTE: Design mock — heavy adapters are deferred (and live behind optional
extras), so this type-checks against the real core but does not run yet.
"""

import pandas as pd
from langchain_openai import ChatOpenAI

from denselinkage import DenseLinker, Source, TemplateSerializer
from denselinkage.blocking import DenseBlocker
from denselinkage.core.results import LabeledPairs
from denselinkage.embedding import SentenceTransformerEmbedder
from denselinkage.indexing import FaissFlatIndex
from denselinkage.matching import LangChainMatcher, RetryPolicy
from denselinkage.metrics import linkage_metrics


def main() -> None:
    df_a = pd.DataFrame(
        {
            "id_a": ["A1", "A2", "A3"],
            "name": ["Apple Inc", "Microsoft Corp", "Google LLC"],
            "city": ["Cupertino", "Redmond", "Mountain View"],
        }
    )
    df_b = pd.DataFrame(
        {
            "id_b": ["B1", "B2", "B3"],
            "company_name": ["Apple Incorporated", "Microsoft", "Alphabet"],
            "headquarters": ["Cupertino, CA", "Redmond, WA", "Mountain View, CA"],
        }
    )
    gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2"), ("A3", "B3")])

    # Blocker: embedder and vector index injected independently.
    blocker = DenseBlocker(
        embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
        vector_index=FaissFlatIndex(),
        similarity_threshold=0.80,  # retrieve top_k, then keep >= this (L4)
        top_k=5,
    )

    # Matcher: the LLM is injected; model / operational / domain config stay
    # separate. The prompt is just the question — no format instruction.
    matcher = LangChainMatcher(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0, seed=42),
        prompt=(
            "Are these two records the same real-world entity?\n"
            "Record A: {record_a}\n"
            "Record B: {record_b}"
        ),
        retry=RetryPolicy(max_retries=3),
        max_concurrency=8,
    )

    # The linker is pure config — no data, no schema, nothing fitted.
    linker = DenseLinker(blocker=blocker, matcher=matcher)

    # Schema travels with each frame. df_b maps its columns onto the shared
    # template; the linker never learns either schema.
    template = "Name: {name}, City: {city}"
    left = Source(df_a, id_column="id_a", serializer=TemplateSerializer(template))
    right = Source(
        df_b,
        id_column="id_b",
        serializer=TemplateSerializer(
            template,
            column_mapping={"company_name": "name", "headquarters": "city"},
        ),
    )

    result = linker.link(left, right)  # one call, no mutation

    print("\n--- Match Results ---")
    # Fixed schema, independent of input column names (H3):
    # left_id, right_id, match (bool|None), confidence (float|None),
    # reason (str|None), similarity (float). Contains ALL candidate pairs.
    print(result.to_frame())

    print("\n--- Evaluation Metrics ---")
    metrics = linkage_metrics(result, gold=gold)  # -> LinkageMetrics
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall:    {metrics.recall:.4f}")
    print(f"F1 Score:  {metrics.f1:.4f}")
    if metrics.n_errors:
        print(f"(errored pairs excluded from P/R: {metrics.n_errors})")


if __name__ == "__main__":
    main()
