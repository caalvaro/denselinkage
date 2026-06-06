"""Example 04 — Deduplication on the dependency-free stack (runnable).

The same dedup workflow as ``02``, but wired on the lexical reference stack
(``HashedNGramEmbedder`` + ``NumpyFlatIndex`` + ``ThresholdMatcher`` via
``DenseLinker.with_defaults``), so it runs with no extras: dedupe -> cluster ->
B3 quality, all on numpy + pandas. ``02`` shows the production semantic + LLM
shape (needs the ``[all]`` extras + a live LLM); this is the runnable counterpart.
"""

import pandas as pd

from denselinkage import (
    DenseLinker,
    LabeledPairs,
    Source,
    TemplateSerializer,
    connected_components,
)
from denselinkage.metrics import clustering_metrics


def main() -> None:
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5"],
            "name": [
                "Apple Inc",
                "Apple Incorporated",
                "Microsoft Corp",
                "Microsoft Corporation",
                "Google LLC",
            ],
            "city": ["Cupertino", "Cupertino", "Redmond", "Redmond", "Mountain View"],
        }
    )
    # Gold dedup pairs are order-insensitive (see 02). Google (5) is a singleton.
    gold = LabeledPairs.from_pairs([("1", "2"), ("3", "4")])

    linker = DenseLinker.with_defaults()  # lexical reference stack
    src = Source(df, id_column="id", serializer=TemplateSerializer("{name} — {city}"))

    result = linker.dedupe(src)  # -> LinkageResult; self-pairs suppressed internally
    print("--- Pairwise matches ---")
    print(result.to_frame())
    # result.errors is empty here: the threshold matcher decides every scored
    # pair; an LLM matcher (see 02) surfaces undecided pairs as MatchErrors.

    # connected_components is TRANSITIVE (A~B, B~C -> one cluster); with a noisy
    # matcher this can over-merge. Watch for B3 recall >> precision + big clusters.
    clusters = connected_components(result)  # -> ClusteringResult
    print(f"\n--- {clusters.n_clusters} clusters ---")
    print(clusters.to_frame())

    cm = clustering_metrics(clusters, gold=gold)  # B3 (Bagga-Baldwin) quality
    print(f"\nB3 P/R/F1: {cm.b3_precision:.3f} {cm.b3_recall:.3f} {cm.b3_f1:.3f}")


if __name__ == "__main__":
    main()
