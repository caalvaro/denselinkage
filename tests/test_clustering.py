"""Behavioural tests for connected-components clustering and ``ClusteringResult``
(Batch 1). Pure over a ``LinkageResult`` — no blocking/matching needed.
"""

from denselinkage.clustering import ConnectedComponentsClusterer, connected_components
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.core.results import LinkageResult


def _dec(a: str, b: str, *, match: bool) -> tuple[CandidatePair, MatchDecision]:
    pair = CandidatePair(Record(a, ""), Record(b, ""), similarity_score=1.0)
    return (pair, MatchDecision(is_match=match))


def test_transitive_closure_merges_chain() -> None:
    result = LinkageResult(
        decisions=(_dec("A", "B", match=True), _dec("B", "C", match=True))
    )
    clusters = connected_components(result)
    assert clusters.labels["A"] == clusters.labels["B"] == clusters.labels["C"]
    assert clusters.n_clusters == 1


def test_deep_chain_collapses_to_one_cluster() -> None:
    # A long chain D-C-B-A forces multi-level union-find paths (path
    # compression); all four must still land in a single cluster.
    result = LinkageResult(
        decisions=(
            _dec("D", "C", match=True),
            _dec("C", "B", match=True),
            _dec("B", "A", match=True),
        )
    )
    clusters = connected_components(result)
    assert set(clusters.labels) == {"A", "B", "C", "D"}
    assert clusters.n_clusters == 1


def test_disjoint_matches_form_separate_clusters() -> None:
    result = LinkageResult(
        decisions=(_dec("A", "B", match=True), _dec("C", "D", match=True))
    )
    clusters = connected_components(result)
    assert clusters.labels["A"] == clusters.labels["B"]
    assert clusters.labels["C"] == clusters.labels["D"]
    assert clusters.labels["A"] != clusters.labels["C"]
    assert clusters.n_clusters == 2


def test_decided_but_unmatched_record_is_a_singleton() -> None:
    # Nodes come from all decisions; a non-match still contributes its records.
    result = LinkageResult(decisions=(_dec("A", "B", match=False),))
    clusters = connected_components(result)
    assert set(clusters.labels) == {"A", "B"}
    assert clusters.labels["A"] != clusters.labels["B"]
    assert clusters.n_clusters == 2


def test_errored_pair_records_become_singletons() -> None:
    # Records the matcher could not decide still appear in the clustering.
    pair = CandidatePair(Record("A", ""), Record("B", ""), similarity_score=0.5)
    result = LinkageResult(
        decisions=(), errors=((pair, MatchError(reason="undecided")),)
    )
    clusters = connected_components(result)
    assert set(clusters.labels) == {"A", "B"}
    assert clusters.labels["A"] != clusters.labels["B"]
    assert clusters.n_clusters == 2


def test_non_match_edges_do_not_merge() -> None:
    result = LinkageResult(
        decisions=(_dec("A", "B", match=True), _dec("A", "C", match=False))
    )
    clusters = connected_components(result)
    assert clusters.labels["A"] == clusters.labels["B"]
    assert clusters.labels["C"] != clusters.labels["A"]
    assert clusters.n_clusters == 2


def test_empty_result_is_empty_clustering() -> None:
    clusters = connected_components(LinkageResult(decisions=()))
    assert dict(clusters.labels) == {}
    assert clusters.n_clusters == 0
    frame = clusters.to_frame()
    assert frame.empty
    assert list(frame.columns) == ["record_id", "cluster_id"]


def test_cluster_ids_are_deterministic_by_min_record_id() -> None:
    # {C,D} is added before {A,B}, but labels are assigned by each component's
    # smallest id, so {A,B}->0 and {C,D}->1 regardless of input order.
    result = LinkageResult(
        decisions=(_dec("C", "D", match=True), _dec("A", "B", match=True))
    )
    clusters = connected_components(result)
    assert (clusters.labels["A"], clusters.labels["B"]) == (0, 0)
    assert (clusters.labels["C"], clusters.labels["D"]) == (1, 1)


def test_to_frame_schema_and_order() -> None:
    result = LinkageResult(
        decisions=(_dec("A", "B", match=True), _dec("C", "D", match=True))
    )
    frame = connected_components(result).to_frame()
    assert list(frame.columns) == ["record_id", "cluster_id"]
    assert list(frame["record_id"]) == ["A", "B", "C", "D"]
    assert list(frame["cluster_id"]) == [0, 0, 1, 1]


def test_clusterer_delegates_to_function() -> None:
    result = LinkageResult(decisions=(_dec("A", "B", match=True),))
    via_class = ConnectedComponentsClusterer().cluster(result)
    via_func = connected_components(result)
    assert dict(via_class.labels) == dict(via_func.labels)
