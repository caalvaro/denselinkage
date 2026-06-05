"""``ConnectedComponentsClusterer`` — the reference ``Clusterer`` adapter."""

from denselinkage.clustering.connected_components import connected_components
from denselinkage.core.ports import Clusterer
from denselinkage.core.results import ClusteringResult, LinkageResult


class ConnectedComponentsClusterer(Clusterer):
    """Reference ``Clusterer`` adapter; delegates to
    :func:`denselinkage.clustering.connected_components`."""

    def cluster(self, result: LinkageResult) -> ClusteringResult:
        return connected_components(result)
