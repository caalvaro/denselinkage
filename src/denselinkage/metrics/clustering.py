"""B³ clustering metrics — the ``ClusteringMetrics`` report and the
``clustering_metrics`` function that produces it."""

from dataclasses import dataclass

from denselinkage.core.results import ClusteringResult, LabeledPairs


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


def clustering_metrics(
    clustering: ClusteringResult, *, gold: LabeledPairs
) -> ClusteringMetrics:
    """B³ clustering quality of ``clustering`` against ``gold``.

    ``gold`` pairs are treated as edges and closed transitively into gold
    clusters, so the same ``LabeledPairs`` used for ``linkage_metrics`` scores
    clustering too — one gold type everywhere (kwarg ``gold``, consistent with
    the other metrics). Records present in ``clustering`` but in no gold pair
    are singleton gold clusters. Returns B³ precision/recall/F1; see
    :class:`ClusteringMetrics`.
    """
    ...
