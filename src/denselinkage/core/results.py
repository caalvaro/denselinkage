"""Typed results / metrics / labels the API returns."""

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
class LinkageMetrics:
    """Contract: pairs that errored (a ``MatchError`` in
    ``LinkageResult.errors``) are excluded from tp/fp/fn and reported as
    ``n_errors``. ``false_negative`` counts every gold pair not predicted a
    match — including gold pairs the blocker never surfaced — so recall is
    honest end-to-end, not conditional on blocking."""

    true_positive: int
    false_positive: int
    false_negative: int
    n_gold: int
    n_errors: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0


@dataclass(frozen=True, slots=True)
class BlockingMetrics:
    """Pair-completeness@k. ``pc_at(k)`` is the sole supported accessor;
    construct via :meth:`from_pc_map` (no leading-underscore public
    constructor param)."""

    pc: Mapping[int, float]
    n_gold: int

    @classmethod
    def from_pc_map(
        cls, pc: Mapping[int, float], *, n_gold: int
    ) -> "BlockingMetrics": ...

    def pc_at(self, k: int) -> float:
        """PC@k. Raises ``KeyError`` if ``k`` was not among the ``ks`` passed
        to ``blocking_metrics`` (no silent 0.0 — an uncomputed k is a usage
        error, not a zero result)."""
        ...


@dataclass(frozen=True, slots=True)
class Clustering:
    labels: Mapping[RecordId, int]

    @property
    def n_clusters(self) -> int: ...

    def to_frame(self) -> "pd.DataFrame": ...


@dataclass(frozen=True, slots=True)
class ClusteringMetrics:
    """B³ (Bagga-Baldwin) clustering quality.

    Gold clusters are the transitive closure of ``LabeledPairs`` (gold pairs ->
    connected components), so the same gold that scores pairwise
    ``linkage_metrics`` also scores clustering — one gold type everywhere.
    ``b3_precision`` / ``b3_recall`` are per-record B³ averages; ``b3_f1`` is
    their harmonic mean. ``n_clusters`` / ``n_gold_clusters`` are reported for
    context. Construct via :meth:`from_b3` — the two adjacent ratios are easy to
    transpose positionally, so the keyword constructor is the supported path
    (mirrors ``BlockingMetrics.from_pc_map``).
    """

    b3_precision: float
    b3_recall: float
    n_clusters: int
    n_gold_clusters: int

    @classmethod
    def from_b3(
        cls,
        *,
        b3_precision: float,
        b3_recall: float,
        n_clusters: int,
        n_gold_clusters: int,
    ) -> "ClusteringMetrics": ...

    @property
    def b3_f1(self) -> float: ...


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


@dataclass(frozen=True, slots=True)
class ThresholdSweep:
    """Full precision/recall/F1 curve over a threshold grid. Typed accessor,
    never a bare ``{threshold: f1}`` dict."""

    rows: tuple[tuple[float, LinkageMetrics], ...]

    def best_f1(self) -> tuple[float, LinkageMetrics]: ...

    def at_recall(self, target: float) -> tuple[float, LinkageMetrics]: ...


@dataclass(frozen=True, slots=True)
class AdjustedMetrics:
    """End-to-end honest number: matcher metrics adjusted by blocker
    pair-completeness@k (``recall_adjusted = matcher.recall * pc@k``)."""

    matcher: LinkageMetrics
    blocking_recall_at_k: float
    k: int

    @property
    def recall_adjusted(self) -> float: ...

    @property
    def f1_adjusted(self) -> float: ...
