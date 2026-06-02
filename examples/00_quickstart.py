"""Example 00 — Quickstart (the shortest real path).

Schema-aligned data + ``Source`` defaulting to the whole-row serializer means
this is genuinely minimal: no template, no column mapping, one call.

``DenseLinker.with_defaults()`` is the low-floor entry point: it wires the
dependency-free reference stack (``HashedNGramEmbedder`` + ``NumpyFlatIndex``
behind a ``DenseBlocker``, plus ``ThresholdMatcher``). Pass ``blocker=`` /
``matcher=`` to override either half (see ``01``/``03`` for full control).

The default stack is **lexical** (``HashedNGramEmbedder`` is character n-gram
feature hashing): it recovers abbreviations, punctuation and typos
(``Apple Inc`` / ``Apple Incorporated``; ``Microsoft Corp`` / ``Microsoft``;
``Google LLC`` / ``Google``) but not semantic renames such as
``Google`` -> ``Alphabet``. The gold below is lexically recoverable on purpose;
for semantic matches reach for ``SentenceTransformerEmbedder`` (see ``01``).
"""

import pandas as pd

from denselinkage import DenseLinker, Source
from denselinkage.core.results import LabeledPairs
from denselinkage.metrics import linkage_metrics


def main() -> None:
    df_a = pd.DataFrame(
        {
            "id": ["A1", "A2", "A3"],
            "name": ["Apple Inc", "Microsoft Corp", "Google LLC"],
            "city": ["Cupertino", "Redmond", "Mountain View"],
        }
    )
    df_b = pd.DataFrame(
        {
            "id": ["B1", "B2", "B3"],
            "name": ["Apple Incorporated", "Microsoft", "Google"],
            "city": ["Cupertino", "Redmond", "Mountain View"],
        }
    )
    gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2"), ("A3", "B3")])

    linker = DenseLinker.with_defaults()  # picks a sensible embedder/index/matcher
    left = Source(df_a, id_column="id")  # serializer=None -> whole-row default
    right = Source(df_b, id_column="id")

    result = linker.link(left, right)  # -> LinkageResult

    # columns: left_id, right_id, similarity, match, confidence, reason
    print(result.to_frame())
    metrics = linkage_metrics(result, gold=gold)  # -> LinkageMetrics
    print(f"P/R/F1: {metrics.precision:.3f} {metrics.recall:.3f} {metrics.f1:.3f}")


if __name__ == "__main__":
    main()
