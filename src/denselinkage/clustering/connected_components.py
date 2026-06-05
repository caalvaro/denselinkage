"""``connected_components`` — the prelude convenience clustering function."""

from collections.abc import Iterable

from denselinkage.clustering._union_find import label_components
from denselinkage.core.models import RecordId
from denselinkage.core.results import ClusteringResult, LinkageResult


def connected_components(
    result: LinkageResult, *, all_record_ids: Iterable[RecordId] | None = None
) -> ClusteringResult:
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
    small ``top_k`` whose only neighbour was itself) does not appear in
    ``result``. Pass ``all_record_ids`` to seed the clustering universe with the
    full id set: every listed record is labelled — an unmatched one becomes its
    own singleton — so ``clustering_metrics`` reports a **complete** B³ over all
    records instead of one inflated by the dropped records. Ids are stringified
    to match record ids; ``None`` (the default) keeps the universe to the records
    seen in ``result``.
    """
    nodes: set[RecordId] = (
        {str(record_id) for record_id in all_record_ids}
        if all_record_ids is not None
        else set()
    )
    edges: list[tuple[RecordId, RecordId]] = []
    for pair, decision in result.decisions:
        nodes.add(pair.record_a.id)
        nodes.add(pair.record_b.id)
        if decision.is_match:
            edges.append((pair.record_a.id, pair.record_b.id))
    for pair, _ in result.errors:  # undecided records still get a singleton
        nodes.add(pair.record_a.id)
        nodes.add(pair.record_b.id)
    return ClusteringResult(labels=label_components(edges, nodes))
