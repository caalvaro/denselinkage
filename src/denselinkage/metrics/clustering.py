"""``clustering_metrics`` — B³ clustering quality against gold."""

from denselinkage.core.results import Clustering, ClusteringMetrics, LabeledPairs


def clustering_metrics(
    clustering: Clustering, *, gold: LabeledPairs
) -> ClusteringMetrics:
    """B³ clustering quality of ``clustering`` against ``gold``.

    ``gold`` pairs are treated as edges and closed transitively into gold
    clusters, so the same ``LabeledPairs`` used for ``linkage_metrics`` scores
    clustering too — one gold type everywhere (kwarg ``gold``, consistent with
    the other metrics). Records present in ``clustering`` but in no gold pair
    are singleton gold clusters. Returns B³ precision/recall/F1; see
    :class:`denselinkage.core.results.ClusteringMetrics`.
    """
    ...
