"""``mine_hard_negatives`` — the highest-similarity non-matches."""

from collections.abc import Sequence

from denselinkage.core.models import CandidatePair
from denselinkage.core.results import LabeledPairs
from denselinkage.metrics._pairing import pair_key


def mine_hard_negatives(
    candidates: Sequence[CandidatePair],
    *,
    gold: LabeledPairs,
    n: int | None = None,
    directed: bool = True,
) -> list[CandidatePair]:
    """The hardest negatives among ``candidates``: scored pairs *not* in ``gold``,
    ordered by descending similarity (ties broken by id for determinism).

    These are the pairs the blocker ranked as similar but gold says are
    non-matches — contrastive material a Phase-C trainer turns into the
    ``negatives`` of a ``TrainingPairs``. Unscored pairs
    (``similarity_score is None``) carry no hardness signal and are excluded.
    ``n`` caps the result to the ``n`` hardest (``None`` = all); ``directed``
    follows ``linkage_metrics`` (pass ``directed=False`` for ``dedupe``
    candidates). Raises ``ValueError`` if ``n`` is negative.
    """
    if n is not None and n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    gold_keys = {pair_key(left, right, directed=directed) for left, right in gold.pairs}
    scored_negatives: list[tuple[float, CandidatePair]] = []
    for pair in candidates:
        score = pair.similarity_score
        if score is None:
            continue
        if pair_key(pair.record_a.id, pair.record_b.id, directed=directed) in gold_keys:
            continue
        scored_negatives.append((score, pair))
    scored_negatives.sort(
        key=lambda item: (-item[0], item[1].record_a.id, item[1].record_b.id)
    )
    chosen = scored_negatives if n is None else scored_negatives[:n]
    return [pair for _, pair in chosen]
