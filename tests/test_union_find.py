"""Unit tests for the shared union-find core (``label_components``), used by both
``connected_components`` and ``clustering_metrics``."""

from denselinkage.clustering._union_find import label_components


def test_isolated_nodes_are_singletons() -> None:
    labels = label_components(edges=[], nodes=["A", "B", "C"])
    assert set(labels) == {"A", "B", "C"}
    assert len(set(labels.values())) == 3  # three singleton clusters


def test_edges_merge_transitively() -> None:
    labels = label_components(edges=[("A", "B"), ("B", "C")], nodes=["A", "B", "C"])
    assert labels["A"] == labels["B"] == labels["C"]


def test_redundant_edge_within_a_component_is_a_no_op() -> None:
    # The (A, C) edge closes a triangle already connected via B — union must
    # short-circuit when both endpoints already share a root.
    labels = label_components(
        edges=[("A", "B"), ("B", "C"), ("A", "C")], nodes=["A", "B", "C"]
    )
    assert len(set(labels.values())) == 1  # still one cluster


def test_endpoint_only_in_edges_is_labelled() -> None:
    # A node appearing only as an edge endpoint (not in `nodes`) is still added.
    labels = label_components(edges=[("A", "Z")], nodes=["A"])
    assert labels["A"] == labels["Z"]


def test_labels_are_contiguous_and_deterministic_by_min_id() -> None:
    # {C,D} added before {A,B}; labels assigned by component-min-id, so {A,B}=0.
    labels = label_components(
        edges=[("D", "C"), ("B", "A")], nodes=["A", "B", "C", "D"]
    )
    assert (labels["A"], labels["B"]) == (0, 0)
    assert (labels["C"], labels["D"]) == (1, 1)
    assert set(labels.values()) == {0, 1}  # contiguous 0..n-1


def test_empty() -> None:
    assert label_components(edges=[], nodes=[]) == {}
