"""Shared union-find: transitive closure of pair edges into deterministic
cluster labels. Used by ``connected_components`` (matched decisions) and
``clustering_metrics`` (gold pairs)."""

from collections.abc import Iterable

from denselinkage.core.models import RecordId


def label_components(
    edges: Iterable[tuple[RecordId, RecordId]], nodes: Iterable[RecordId]
) -> dict[RecordId, int]:
    """Label each node with its connected-component id over ``edges``.

    Every id in ``nodes`` (and every endpoint of ``edges``) is labelled; a node
    with no edge is its own singleton. Cluster ids are ``0..n-1``, assigned
    deterministically by each component's smallest id, so the labelling is
    reproducible regardless of input order.
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

    for node in nodes:
        add(node)
    for left, right in edges:
        add(left)
        add(right)
        union(left, right)

    # Group records by representative root. find() compresses paths as it runs,
    # mutating parent's *values* (never its keys), so iterating it here is safe.
    groups: dict[RecordId, list[RecordId]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)

    ordered_roots = sorted(groups, key=lambda root: min(groups[root]))
    return {
        record_id: cluster_id
        for cluster_id, root in enumerate(ordered_roots)
        for record_id in groups[root]
    }
