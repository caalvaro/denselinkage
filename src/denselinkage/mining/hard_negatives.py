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
    negatives = [
        pair
        for pair in candidates
        if pair.similarity_score is not None
        and pair_key(pair.record_a.id, pair.record_b.id, directed=directed)
        not in gold_keys
    ]
    negatives.sort(
        key=lambda pair: (
            -(pair.similarity_score if pair.similarity_score is not None else 0.0),
            pair.record_a.id,
            pair.record_b.id,
        )
    )
    return negatives if n is None else negatives[:n]
