"""B³ clustering metrics — the ``ClusteringMetrics`` report and the
``clustering_metrics`` function that produces it."""

from collections.abc import Mapping
from dataclasses import dataclass

from denselinkage.clustering._union_find import label_components
from denselinkage.core.models import RecordId
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
    ) -> "ClusteringMetrics":
        return cls(
            b3_precision=b3_precision,
            b3_recall=b3_recall,
            n_clusters=n_clusters,
            n_gold_clusters=n_gold_clusters,
        )

    @property
    def b3_f1(self) -> float:
        denom = self.b3_precision + self.b3_recall
        return 2 * self.b3_precision * self.b3_recall / denom if denom else 0.0


def _groups_by_label(labels: Mapping[RecordId, int]) -> dict[int, set[RecordId]]:
    groups: dict[int, set[RecordId]] = {}
    for record_id, label in labels.items():
        groups.setdefault(label, set()).add(record_id)
    return groups


def clustering_metrics(
    clustering: ClusteringResult, *, gold: LabeledPairs
) -> ClusteringMetrics:
    """B³ clustering quality of ``clustering`` against ``gold``.

    ``gold`` pairs are treated as edges and closed transitively into gold
    clusters, so the same ``LabeledPairs`` used for ``linkage_metrics`` scores
    clustering too — one gold type everywhere (kwarg ``gold``, consistent with
    the other metrics). Records present in ``clustering`` but in no gold pair are
    singleton gold clusters. Returns B³ precision/recall/F1; see
    :class:`ClusteringMetrics`.
    """
    records = set(clustering.labels)
    if not records:
        return ClusteringMetrics.from_b3(
            b3_precision=0.0, b3_recall=0.0, n_clusters=0, n_gold_clusters=0
        )
    predicted = _groups_by_label(clustering.labels)
    gold_edges = [
        (left, right)
        for left, right in gold.pairs
        if left in records and right in records
    ]
    gold_labels = label_components(gold_edges, records)
    gold_clusters = _groups_by_label(gold_labels)

    precision_sum = 0.0
    recall_sum = 0.0
    # Records sharing a (predicted, gold) label pair share their overlap, so
    # compute each intersection once — keeps a single mega-cluster from being
    # O(R^2).
    overlap_cache: dict[tuple[int, int], int] = {}
    for record_id in records:
        pred_label = clustering.labels[record_id]
        gold_label = gold_labels[record_id]
        cache_key = (pred_label, gold_label)
        if cache_key not in overlap_cache:
            overlap_cache[cache_key] = len(
                predicted[pred_label] & gold_clusters[gold_label]
            )
        overlap = overlap_cache[cache_key]
        precision_sum += overlap / len(predicted[pred_label])
        recall_sum += overlap / len(gold_clusters[gold_label])
    n = len(records)
    return ClusteringMetrics.from_b3(
        b3_precision=precision_sum / n,
        b3_recall=recall_sum / n,
        n_clusters=clustering.n_clusters,
        n_gold_clusters=len(gold_clusters),
    )
