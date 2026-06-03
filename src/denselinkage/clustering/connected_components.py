"""``connected_components`` — the prelude convenience clustering function."""

from denselinkage.core.models import RecordId
from denselinkage.core.results import ClusteringResult, LinkageResult


def connected_components(result: LinkageResult) -> ClusteringResult:
    """Connected-components clustering: transitively close the matched pairs in
    ``result`` and label each record with its component id. Convenience form of
    :class:`ConnectedComponentsClusterer`; kept in the prelude.

    Nodes are every record the pipeline compared — those in ``result.decisions``
    (matched or not) or ``result.errors`` — so a record that was never matched
    (including pairs the matcher could not decide) becomes its own singleton
    cluster; edges are the matched pairs only. Clustering is **transitive**: if A
    matches B and B matches C, all three share a cluster even if A and C were
    never matched directly. Cluster ids are ``0..n_clusters-1``, assigned
    deterministically by each component's smallest record id.

    A record that produced *no* candidate pair at all (e.g. ``dedupe`` with a
    small ``top_k`` whose only neighbour was itself) does not appear in ``result``
    and so cannot be clustered here; carrying the full record universe into
    clustering is a Phase-B addition.
    """
    parent: dict[RecordId, RecordId] = {}

    def find(node: RecordId) -> RecordId:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    def add(node: RecordId) -> None:
        parent.setdefault(node, node)

    def union(left: RecordId, right: RecordId) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            hi, lo = sorted((left_root, right_root), reverse=True)
            parent[hi] = lo

    for pair, decision in result.decisions:
        add(pair.record_a.id)
        add(pair.record_b.id)
        if decision.is_match:
            union(pair.record_a.id, pair.record_b.id)
    for pair, _ in result.errors:  # undecided records still get a singleton
        add(pair.record_a.id)
        add(pair.record_b.id)

    # Group records by representative root. find() compresses paths as it runs,
    # mutating parent's *values* (never its keys), so iterating it here is safe.
    components: dict[RecordId, list[RecordId]] = {}
    for node in parent:
        components.setdefault(find(node), []).append(node)

    ordered_roots = sorted(components, key=lambda root: min(components[root]))
    labels: dict[RecordId, int] = {
        record_id: cluster_id
        for cluster_id, root in enumerate(ordered_roots)
        for record_id in components[root]
    }
    return ClusteringResult(labels=labels)
