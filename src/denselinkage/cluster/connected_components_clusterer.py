"""``ConnectedComponentsClusterer`` — the reference ``Clusterer`` adapter."""

from denselinkage.core.ports import Clusterer
from denselinkage.core.results import Clustering, LinkageResult


class ConnectedComponentsClusterer(Clusterer):
    """Reference ``Clusterer`` adapter; delegates to
    :func:`denselinkage.cluster.connected_components`."""

    def cluster(self, result: LinkageResult) -> Clustering: ...
