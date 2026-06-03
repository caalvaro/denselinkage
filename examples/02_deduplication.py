"""Example 02 — Deduplication (finding duplicates within one dataset).

Dedup is its own verb. ``linker.dedupe(source)`` links a dataset against
itself and suppresses self-pairs internally — there is no
``suppress_self_pairs`` knob on the public surface. The same ``DenseLinker``
config works for both ``link`` and ``dedupe``; the task is the method name.

``connected_components`` returns a typed ``ClusteringResult``, not a loose dict.
This
example also shows the honest dedup tail: B3 cluster quality via
``clustering_metrics`` against the same ``LabeledPairs`` gold, and triaging the
matcher's per-pair ``MatchError``s from ``result.errors``.

NOTE: Design mock — heavy adapters are deferred, so this type-checks against
the real core but does not run end to end yet.
"""

import logging

import pandas as pd
from langchain_openai import ChatOpenAI

from denselinkage import (
    DenseLinker,
    LabeledPairs,
    Source,
    TemplateSerializer,
    connected_components,
)
from denselinkage.blocking import DenseBlocker
from denselinkage.embedding import SentenceTransformerEmbedder
from denselinkage.indexing import FaissFlatIndex
from denselinkage.matching import LangChainMatcher
from denselinkage.metrics import clustering_metrics

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

    # Gold for dedup is order-insensitive: metrics canonicalize each pair
    # to an unordered key, so ("1","2") and ("2","1") are the same gold pair —
    # unlike `link`, where left/right is meaningful. Alphabet(5)/Google LLC(6)
    # are intentionally NOT gold-linked: parent holding-co vs subsidiary is a
    # resolution-policy call (see 00's note). If your policy says they are the
    # same entity, add ("5", "6").
    gold = LabeledPairs.from_pairs([("1", "2"), ("3", "4")])

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

    # LLM matchers fail some pairs (rate limits, refusals, unparseable replies).
    # Those are MatchErrors in result.errors — excluded from metrics, never
    # silently dropped. Surface them so they can be retried / triaged.
    if result.errors:
        print(f"\n--- {len(result.errors)} errored pairs (excluded from metrics) ---")
        for pair, err in result.errors:
            print(f"  {pair.record_a.id} vs {pair.record_b.id}: {err.reason}")

    # connected_components is TRANSITIVE: if A matches B and B matches C, all
    # three land in one cluster even when A and C were never matched. With a
    # noisy / LLM matcher this can snowball into runaway mega-clusters — watch
    # for B3 recall >> precision and oversized clusters.
    clusters = connected_components(result)  # -> ClusteringResult
    print(f"\n--- {clusters.n_clusters} Clusters ---")
    print(clusters.to_frame())

    # B3 (Bagga-Baldwin) cluster quality against the same gold — the metric that
    # tells you whether resolution over-merged. One gold type scores both.
    cm = clustering_metrics(clusters, gold=gold)
    print(f"B3 P/R/F1: {cm.b3_precision:.3f} {cm.b3_recall:.3f} {cm.b3_f1:.3f}")


if __name__ == "__main__":
    main()
