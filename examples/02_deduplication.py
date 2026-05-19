"""Example 02 — Deduplication (finding duplicates within one dataset).

Dedup is its own verb. ``linker.dedupe(source)`` links a dataset against
itself and suppresses self-pairs internally — there is no
``suppress_self_pairs`` knob on the public surface. The same ``DenseLinker``
config works for both ``link`` and ``dedupe``; the task is the method name.

``connected_components`` returns a typed ``Clustering``, not a loose dict.

NOTE: Design mock — heavy adapters are deferred, so this type-checks against
the real core but does not run end to end yet.
"""

import logging

import pandas as pd
from langchain_openai import ChatOpenAI

from denselinkage import DenseLinker, Source, TemplateSerializer, connected_components
from denselinkage.blocking import DenseBlocker
from denselinkage.embedding import SentenceTransformerEmbedder
from denselinkage.indexing import FaissFlatIndex
from denselinkage.matching import LangChainMatcher

# Components log through the standard logging system; there is no verbose= flag.
logging.basicConfig(level=logging.INFO)


def main() -> None:
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5", "6"],
            "name": [
                "Apple Inc",
                "Apple Incorporated",
                "Microsoft Corp",
                "Microsoft Corporation",
                "Alphabet",
                "Google LLC",
            ],
            "city": [
                "Cupertino",
                "Cupertino, CA",
                "Redmond",
                "Redmond, Washington",
                "Mountain View",
                "Mountain View, CA",
            ],
        }
    )

    linker = DenseLinker(
        blocker=DenseBlocker(
            embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
            vector_index=FaissFlatIndex(),
            similarity_threshold=0.80,
            top_k=3,
        ),
        matcher=LangChainMatcher(
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0, seed=42),
            prompt=(
                "Are these two records the same real-world entity?\n"
                "Record A: {record_a}\n"
                "Record B: {record_b}"
            ),
        ),
    )

    src = Source(df, id_column="id", serializer=TemplateSerializer("{name} — {city}"))

    result = linker.dedupe(src)  # -> LinkageResult; self-pairs suppressed internally
    print("\n--- Pairwise Matches ---")
    print(result.to_frame())

    clusters = connected_components(result)  # -> Clustering
    print(f"\n--- {clusters.n_clusters} Clusters ---")
    print(clusters.to_frame())


if __name__ == "__main__":
    main()
