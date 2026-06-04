"""Blocking-quality metrics — the ``BlockingMetrics`` report and the
``blocking_metrics`` / ``pair_completeness_at_k`` functions over candidate
pairs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from denselinkage.core.models import CandidatePair, RecordId
from denselinkage.core.results import LabeledPairs
from denselinkage.metrics._pairing import pair_key


@dataclass(frozen=True, slots=True)
class BlockingMetrics:
    """Pair-completeness@k. ``pc_at(k)`` is the sole supported accessor;
    construct via :meth:`from_pc_map` (no leading-underscore public
    constructor param)."""

    pc: Mapping[int, float]
    n_gold: int

    @classmethod
    def from_pc_map(cls, pc: Mapping[int, float], *, n_gold: int) -> "BlockingMetrics":
        return cls(pc=dict(pc), n_gold=n_gold)

    def pc_at(self, k: int) -> float:
        """PC@k. Raises ``KeyError`` if ``k`` was not among the ``ks`` passed
        to ``blocking_metrics`` (no silent 0.0 — an uncomputed k is a usage
        error, not a zero result)."""
        return self.pc[k]


def _score(pair: CandidatePair) -> float:
    return pair.similarity_score if pair.similarity_score is not None else float("-inf")


def blocking_metrics(
    candidates: Sequence[CandidatePair],
    *,
    gold: LabeledPairs,
    ks: Sequence[int],
    directed: bool = True,
) -> BlockingMetrics:
    """Pair-completeness@k for each k in ``ks``.

    Candidates are grouped by query record (``record_b``) and ranked by
    similarity; PC@k is the fraction of ``gold`` pairs recalled when each query
    keeps its top-k candidates. Candidates are expected to be blocker-oriented
    (``record_a`` indexed/reference, ``record_b`` query), as produced by
    ``BlockingIndex.query``. To sweep ``ks`` meaningfully, pass the blocker's
    full ranked retrieval (a large ``top_k``), not an already-truncated set.

    Pair identity (D1): same rule as ``linkage_metrics`` — ``directed=True``
    (``link``) compares ordered; ``directed=False`` (``dedupe``) canonicalizes
    to an unordered key.
    """
    gold_keys = {pair_key(left, right, directed=directed) for left, right in gold.pairs}
    by_query: dict[RecordId, list[CandidatePair]] = {}
    for pair in candidates:
        by_query.setdefault(pair.record_b.id, []).append(pair)
    for group in by_query.values():
        group.sort(key=_score, reverse=True)
    # covered@k is monotone in k, so accumulate as k grows (over sorted, unique
    # ks): each candidate's key is computed once across the whole sweep.
    covered: set[tuple[RecordId, RecordId] | frozenset[RecordId]] = set()
    pc: dict[int, float] = {}
    prev_k = 0
    for k in sorted(set(ks)):
        for group in by_query.values():
            for pair in group[prev_k:k]:
                covered.add(
                    pair_key(pair.record_a.id, pair.record_b.id, directed=directed)
                )
        pc[k] = len(gold_keys & covered) / len(gold_keys) if gold_keys else 0.0
        prev_k = k
    return BlockingMetrics.from_pc_map(pc, n_gold=len(gold.pairs))


def pair_completeness_at_k(
    candidates: Sequence[CandidatePair],
    *,
    gold: LabeledPairs,
    k: int,
    directed: bool = True,
) -> float:
    """Single-k pair-completeness (kwarg ``gold``, consistent with the other
    metrics). Pair identity (D1): same rule as ``linkage_metrics``; pass
    ``directed=False`` for ``dedupe`` candidates."""
    return blocking_metrics(candidates, gold=gold, ks=[k], directed=directed).pc_at(k)
