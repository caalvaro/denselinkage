"""Example 05 — Failure accounting (a typed ``MatchError`` vs a silent no).

A real LLM matcher fails on some pairs: it refuses, times out, or returns
something unparseable. What a tool *does* with that failure changes the number
it reports. The ``Matcher`` port here returns one outcome per pair, and that
outcome is a ``MatchDecision`` **or** a ``MatchError`` — so an adapter can say
"I could not decide this pair" instead of guessing. The linker routes a
``MatchError`` into ``LinkageResult.errors``; ``linkage_metrics`` then excludes
it from precision and recall and reports it as ``n_errors``. Most existing
tools take the other path: a failure is recorded as a non-match, which turns
every failure on a true pair into a false negative and pulls F1 down with the
failure rate.

This example makes the difference concrete. We wrap the dependency-free
``ThresholdMatcher`` in one adapter, ``FlakyMatcher``, that fails on a fixed
fraction ``f`` of pairs and reports the failure two ways: as a ``MatchError``
(this library's accounting) or as a ``MatchDecision(is_match=False)`` (the
silent convention). The *same* pairs fail in both modes, so the runs differ
only in how a failure is counted. As ``f`` grows the excluded F1 barely moves
while the silent F1 falls — the effect is arithmetic, not a property of this
data. It is the same comparison the companion paper applies to the DBLP-ACM
benchmark (see ``benchmarks/failure_accounting_experiment.py``).

Runs on the dependency-free stack (numpy + pandas), no API key, deterministic.
"""

import zlib
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from denselinkage import DenseLinker, LabeledPairs, Source
from denselinkage.blocking import DenseBlocker
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError
from denselinkage.core.ports import Matcher, Serializer
from denselinkage.embedding import HashedNGramEmbedder
from denselinkage.indexing import NumpyFlatIndex
from denselinkage.matching import ThresholdMatcher
from denselinkage.metrics import LinkageMetrics, linkage_metrics


class FlakyMatcher(Matcher):
    """Wraps a base matcher and fails on a deterministic fraction of pairs.

    It asks the base matcher for every outcome, then overrides a fraction
    ``fail_rate`` of pairs with a failure. ``on_failure="error"`` reports it the
    way this library recommends — a typed ``MatchError`` the linker routes into
    ``LinkageResult.errors`` and the metrics exclude. ``on_failure="silent"``
    reports it the way most existing tools do — a plain non-match. The pair that
    fails is chosen by a stable hash of its record ids, so both modes fail on
    exactly the same pairs and differ only in how the failure is accounted.
    """

    def __init__(self, base: Matcher, *, fail_rate: float, on_failure: str) -> None:
        self._base = base
        self._fail_rate = fail_rate
        self._on_failure = on_failure  # "error" (typed) or "silent" (non-match)

    def _fails(self, pair: CandidatePair) -> bool:
        # crc32 is a stable hash (unlike builtin hash() on str, which is salted):
        # the same pair maps to the same point in [0, 1) across processes.
        key = f"{pair.record_a.id}|{pair.record_b.id}".encode()
        return zlib.crc32(key) / 2**32 < self._fail_rate

    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision | MatchError]:
        outcomes = self._base.match(pairs)
        result: list[MatchDecision | MatchError] = []
        for pair, outcome in zip(pairs, outcomes, strict=True):
            if self._fails(pair):
                if self._on_failure == "error":
                    result.append(MatchError(reason="simulated refusal/timeout"))
                else:  # the failure disappears into a non-match
                    result.append(MatchDecision(is_match=False))
            else:
                result.append(outcome)
        return result


class _OrgField(Serializer):
    """Serializer: render a record from its ``org`` field only."""

    def serialize(self, record: Mapping[str, Any]) -> str:
        return str(record["org"])


def _toy_sources() -> tuple[Source, Source, LabeledPairs]:
    """A deterministic two-table linkage problem on the lexical stack: each of
    ``N`` entities appears in both tables under a noisy spelling, with a unique
    numeric token so blocking can recover the true pair."""
    n = 150
    suffix_a, suffix_b = "Inc", "Incorporated"
    left_rows, right_rows, gold = [], [], []
    for i in range(n):
        stem = f"Northwind {i} Logistics"
        left_rows.append({"id": f"A{i}", "org": f"{stem} {suffix_a}"})
        # right side: lowercase, longer suffix, a dropped space — lexically
        # recoverable by char n-grams but not identical.
        right_rows.append({"id": f"B{i}", "org": f"{stem.lower()} {suffix_b}"})
        gold.append((f"A{i}", f"B{i}"))

    serializer = _OrgField()
    left = Source(pd.DataFrame(left_rows), id_column="id", serializer=serializer)
    right = Source(pd.DataFrame(right_rows), id_column="id", serializer=serializer)
    return left, right, LabeledPairs.from_pairs(gold)


def main() -> None:
    left, right, gold = _toy_sources()

    # One fixed, dependency-free pipeline; only the matcher's failure handling
    # changes between the two columns below.
    blocker = DenseBlocker(
        embedder=HashedNGramEmbedder(n_features=1024, ngram=3),
        vector_index=NumpyFlatIndex(),
        top_k=1,
    )
    base_matcher = ThresholdMatcher(threshold=0.5)

    def score(fail_rate: float, on_failure: str) -> LinkageMetrics:
        matcher = FlakyMatcher(base_matcher, fail_rate=fail_rate, on_failure=on_failure)
        linker = DenseLinker(blocker=blocker, matcher=matcher)
        return linkage_metrics(linker.link(left, right), gold=gold, directed=True)

    header = (
        f"{'f':>4} {'n_errors':>9} {'F1 (excl)':>10} {'F1 (silent)':>12} "
        f"{'recall (excl)':>14} {'recall (silent)':>16}"
    )
    print(header)
    print("-" * len(header))
    for f in (0.0, 0.05, 0.10, 0.20):
        excl = score(f, "error")  # typed MatchError -> excluded, counted
        silent = score(f, "silent")  # failure -> non-match, folded into F1
        print(
            f"{f:>4.0%} {excl.n_errors:>9} {excl.f1:>10.3f} {silent.f1:>12.3f} "
            f"{excl.recall:>14.3f} {silent.recall:>16.3f}"
        )

    print(
        "\nThe excluded F1 barely moves; the silent F1 falls as failures become "
        "false\nnegatives. Same pipeline, same gold; only the accounting of a "
        "failed pair differs."
    )


if __name__ == "__main__":
    main()
