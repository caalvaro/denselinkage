"""``connected_components`` — the prelude convenience clustering function."""

from denselinkage.core.results import ClusteringResult, LinkageResult


def connected_components(result: LinkageResult) -> ClusteringResult:
    """Connected-components clustering: transitively close the matched pairs in
    ``result`` and label each record with its component id. Convenience form of
    :class:`ConnectedComponentsClusterer`; kept in the prelude."""
    ...
