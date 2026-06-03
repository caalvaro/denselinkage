"""Typed pipeline outputs and gold / training value objects the API returns and
consumes.

Per ADR-0002, evaluation *report* types (``LinkageMetrics``, ``BlockingMetrics``,
``ClusteringMetrics``, ``ThresholdSweep``, ``AdjustedMetrics``) live in
``denselinkage.metrics``, co-located with the functions that produce them.
``core`` keeps only the outputs a port references (``LinkageResult``,
``ClusteringResult``) plus the domain gold / training value objects
(``LabeledPairs``,
``TrainingPairs``).
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from denselinkage.core.models import (
    CandidatePair,
    MatchDecision,
    MatchError,
    Record,
    RecordId,
)

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class LabeledPairs:
    """The gold set of true matches — one type everywhere.

    Pairs are stored
    **exactly as given (ordered)**; no symmetrization happens at construction.
    Each tuple is ``(left_id, right_id)``. Evaluation comparison depends on the
    verb:

    - ``link`` (two sources): the order is meaningful — a gold ``(a, b)``
      matches a result pair whose left id is ``a`` and right id is ``b``.
    - ``dedupe`` (one source): left/right is arbitrary, so metrics canonicalize
      *both* gold and result pairs to an unordered key (``frozenset({a, b})``)
      before comparing. This removes the silent recall/precision fork.

    See the matching docstrings of ``linkage_metrics`` /
    ``pair_completeness_at_k`` for which comparison each applies.
    """

    pairs: frozenset[tuple[RecordId, RecordId]]

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]]) -> "LabeledPairs":
        return cls(pairs=frozenset(pairs))

    @classmethod
    def from_frame(
        cls, frame: "pd.DataFrame", *, left_id: str, right_id: str
    ) -> "LabeledPairs":
        return cls(
            pairs=frozenset(
                (str(left), str(right))
                for left, right in zip(frame[left_id], frame[right_id], strict=True)
            )
        )


@dataclass(frozen=True, slots=True)
class LinkageResult:
    """All candidate pairs with their match decisions."""

    decisions: tuple[tuple[CandidatePair, MatchDecision], ...]
    errors: tuple[tuple[CandidatePair, MatchError], ...] = ()

    def to_frame(self) -> "pd.DataFrame":
        """One row per *decided* candidate pair (matches AND non-matches,
        required for honest precision). Errored pairs are NOT rows here — they
        live in :attr:`errors` and are counted as ``LinkageMetrics.n_errors``.

        Fixed column schema, independent of input id-column names (echoing
        them would collide on dedupe):

        - ``left_id``, ``right_id``, ``similarity`` — always populated.
        - ``match`` (``bool``) — always populated (errors are not rows).
        - ``confidence`` (``float | None``), ``reason`` (``str | None``) —
          nullable; ``None`` when the matcher does not produce them
          (e.g. ``ThresholdMatcher``).
        """
        import pandas as pd

        columns = ["left_id", "right_id", "similarity", "match", "confidence", "reason"]
        rows = [
            {
                "left_id": pair.record_a.id,
                "right_id": pair.record_b.id,
                "similarity": pair.similarity_score,
                "match": decision.is_match,
                "confidence": decision.confidence,
                "reason": decision.rationale,
            }
            for pair, decision in self.decisions
        ]
        return pd.DataFrame(rows, columns=columns)


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    """Entity clusters as a record-id -> cluster-id map (sklearn-style labels).

    Cluster ids are contiguous ``0..n_clusters-1``, assigned deterministically by
    the producer (``connected_components``) so the labelling is reproducible.
    """

    labels: Mapping[RecordId, int]

    @property
    def n_clusters(self) -> int:
        """Number of distinct clusters."""
        return len(set(self.labels.values()))

    def to_frame(self) -> "pd.DataFrame":
        """One row per record — ``record_id``, ``cluster_id`` — sorted by
        ``(cluster_id, record_id)`` for a stable, readable ordering."""
        import pandas as pd

        columns = ["record_id", "cluster_id"]
        rows: list[dict[str, str | int]] = [
            {"record_id": record_id, "cluster_id": cluster_id}
            for record_id, cluster_id in sorted(
                self.labels.items(), key=lambda item: (item[1], item[0])
            )
        ]
        return pd.DataFrame(rows, columns=columns)


@dataclass(frozen=True, slots=True)
class TrainingPairs:
    """Supervised material for a Trainer (v2). A distinct sibling of
    ``LabeledPairs`` — never an overload: ``LabeledPairs`` is positives-only
    gold for *evaluation*; ``TrainingPairs`` carries negatives and is consumed
    only by training. ``batches`` preserves in-batch contrastive grouping
    (the paper's loss is batch-structure dependent)."""

    positives: tuple[tuple[Record, Record], ...]
    negatives: tuple[tuple[Record, Record], ...]
    batches: tuple[tuple[int, ...], ...] | None = None
