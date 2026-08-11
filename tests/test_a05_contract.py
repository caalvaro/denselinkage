"""Contract-hardening tests.

Structure-stage: assert the *shape* the brief pins (bodies are still ``...``).
Covers the exception taxonomy, ordering semantics, the
``pair_completeness_at_k`` kwarg, the ``BlockingMetrics`` constructor
shape, the ``CandidatePair`` default, and the named reader seam,
plus the Resolvi-conformance additions: the ``Clusterer`` port,
``clustering_metrics`` / ``ClusteringMetrics``, and the ``Filter`` port.
"""

import dataclasses
import inspect

from denselinkage import core, metrics
from denselinkage._reader import RecordReader
from denselinkage.clustering import ConnectedComponentsClusterer, connected_components
from denselinkage.core import errors, ports
from denselinkage.core.errors import DenseLinkageError
from denselinkage.core.models import CandidatePair
from denselinkage.core.results import LabeledPairs
from denselinkage.filtering import SimilarityThresholdFilter
from denselinkage.metrics import BlockingMetrics, ClusteringMetrics

TAXONOMY = (
    "DenseLinkageError",
    "UnknownIdColumn",
    "EmptySource",
    "DuplicateRecordId",
    "DimensionMismatch",
    "InvalidTopK",
    "IncompatibleStore",
)


def test_exception_taxonomy_exists_rooted_and_exported() -> None:
    for name in TAXONOMY:
        assert name in core.__all__, f"{name} not exported from denselinkage.core"
        exc = getattr(errors, name)
        assert isinstance(exc, type) and issubclass(exc, Exception)
        assert issubclass(exc, DenseLinkageError)
        assert getattr(core, name) is exc


def test_labeledpairs_is_ordered_and_not_symmetrized() -> None:
    lp = LabeledPairs(pairs=frozenset({("A1", "B1")}))
    assert ("A1", "B1") in lp.pairs
    # Order is meaningful for `link`; construction must NOT symmetrize.
    assert ("B1", "A1") not in lp.pairs


def test_pair_completeness_at_k_gold_is_keyword_only() -> None:
    sig = inspect.signature(metrics.pair_completeness_at_k)
    assert sig.parameters["gold"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["k"].kind is inspect.Parameter.KEYWORD_ONLY


def test_blocking_metrics_constructor_shape() -> None:
    field_names = {f.name for f in dataclasses.fields(BlockingMetrics)}
    assert "pc" in field_names and "_pc" not in field_names
    assert hasattr(BlockingMetrics, "from_pc_map")


def test_candidate_pair_similarity_optional_with_default() -> None:
    fields = {f.name: f for f in dataclasses.fields(CandidatePair)}
    assert fields["similarity_score"].default is None


def test_record_reader_seam_is_named_and_documented() -> None:
    import denselinkage._reader as reader_mod

    assert hasattr(RecordReader, "read")
    assert reader_mod.__doc__ is not None
    # The seam's documented responsibility includes serializer=None resolution.
    assert "serializer" in reader_mod.__doc__.lower()


def test_clusterer_port_and_reference_adapter() -> None:
    # (Resolvi: clustering is a swappable Strategy expressed as a port).
    assert getattr(ports.Clusterer, "_is_runtime_protocol", False)
    assert hasattr(ports.Clusterer, "cluster")
    # Reference adapter declares the port (MRO check, like every other adapter).
    assert ports.Clusterer in ConnectedComponentsClusterer.__mro__
    # The prelude convenience function is preserved (extend, never modify).
    assert callable(connected_components)


def test_clustering_metrics_signature_and_result() -> None:
    # Resolvi explicitly recommends clustering-quality metrics; kwarg `gold`
    # stays consistent with the other metrics.
    sig = inspect.signature(metrics.clustering_metrics)
    assert sig.parameters["gold"].kind is inspect.Parameter.KEYWORD_ONLY
    assert dataclasses.is_dataclass(ClusteringMetrics)
    field_names = {f.name for f in dataclasses.fields(ClusteringMetrics)}
    assert field_names == {
        "b3_precision",
        "b3_recall",
        "n_clusters",
        "n_gold_clusters",
    }
    assert hasattr(ClusteringMetrics, "from_b3")
    assert isinstance(ClusteringMetrics.b3_f1, property)


def test_filter_port_and_reference_adapter() -> None:
    # Resolvi's filtering stage as its own port, distinct from blocking.
    assert getattr(ports.Filter, "_is_runtime_protocol", False)
    assert hasattr(ports.Filter, "filter")
    assert ports.Filter in SimilarityThresholdFilter.__mro__
