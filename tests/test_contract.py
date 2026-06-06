"""Structure-stage contract tests.

The package is declarations only (signatures with ``...`` bodies), so these
assert the *shape* of the public API — that it imports with zero heavy deps
and the types/ports exist as expected. Behavioural tests come with the
implementation.
"""

import dataclasses
import inspect

import pytest

import denselinkage as dl
from denselinkage import metrics
from denselinkage.blocking import DenseBlocker, DenseBlockingIndex
from denselinkage.clustering import ConnectedComponentsClusterer
from denselinkage.core import models, ports, results
from denselinkage.embedding import HashedNGramEmbedder, SentenceTransformerEmbedder
from denselinkage.filtering import SimilarityThresholdFilter
from denselinkage.indexing import (
    FaissFlatIndex,
    FaissSearchableIndex,
    NumpyFlatIndex,
    NumpySearchableIndex,
)
from denselinkage.matching import LangChainMatcher, RetryPolicy, ThresholdMatcher
from denselinkage.serializing import (
    FieldwiseSerializer,
    TemplateSerializer,
    WholeRowSerializer,
)

EXPECTED_PRELUDE = {
    "BlockingMetrics",
    "ClusteringMetrics",
    "ClusteringResult",
    "DenseLinker",
    "FieldwiseSerializer",
    "LabeledPairs",
    "LinkageIndex",
    "LinkageMetrics",
    "LinkageResult",
    "Source",
    "TemplateSerializer",
    "WholeRowSerializer",
    "candidate_pairs_from_frame",
    "connected_components",
}


def test_prelude_surface() -> None:
    assert set(dl.__all__) == EXPECTED_PRELUDE
    for name in dl.__all__:
        assert hasattr(dl, name), name


@pytest.mark.parametrize(
    "cls",
    [
        models.Record,
        models.CandidatePair,
        models.MatchDecision,
        models.MatchError,
        models.Source,
        results.LabeledPairs,
        results.LinkageResult,
        results.ClusteringResult,
        metrics.LinkageMetrics,
        metrics.BlockingMetrics,
        metrics.ClusteringMetrics,
        RetryPolicy,
    ],
)
def test_value_objects_are_dataclasses(cls: type) -> None:
    assert dataclasses.is_dataclass(cls)


def test_record_fields() -> None:
    names = {f.name for f in dataclasses.fields(models.Record)}
    assert names == {"id", "text", "fields"}


def test_match_decision_is_bool_only_no_error_field() -> None:
    # Pinned contract: is_match is a real bool; failures are modelled by the
    # sibling MatchError + LinkageResult.errors, never inside MatchDecision.
    names = {f.name for f in dataclasses.fields(models.MatchDecision)}
    assert names == {"is_match", "confidence", "rationale"}
    assert models.MatchDecision.__annotations__["is_match"] in (bool, "bool")
    assert {f.name for f in dataclasses.fields(models.MatchError)} == {"reason"}


def test_linkage_result_has_separate_errors_channel() -> None:
    fields = {f.name for f in dataclasses.fields(results.LinkageResult)}
    assert fields == {"decisions", "errors"}


@pytest.mark.parametrize(
    "port",
    [
        ports.Serializer,
        ports.Embedder,
        ports.VectorIndex,
        ports.SearchableIndex,
        ports.Blocker,
        ports.BlockingIndex,
        ports.Filter,
        ports.Matcher,
        ports.Clusterer,
    ],
)
def test_ports_are_runtime_checkable_protocols(port: type) -> None:
    # @runtime_checkable protocols expose _is_runtime_protocol.
    assert getattr(port, "_is_runtime_protocol", False)


@pytest.mark.parametrize(
    ("adapter", "port"),
    [
        (DenseBlocker, ports.Blocker),
        (DenseBlockingIndex, ports.BlockingIndex),
        (HashedNGramEmbedder, ports.Embedder),
        (SentenceTransformerEmbedder, ports.Embedder),
        (NumpyFlatIndex, ports.VectorIndex),
        (NumpySearchableIndex, ports.SearchableIndex),
        (FaissFlatIndex, ports.VectorIndex),
        (FaissSearchableIndex, ports.SearchableIndex),
        (ThresholdMatcher, ports.Matcher),
        (LangChainMatcher, ports.Matcher),
        (TemplateSerializer, ports.Serializer),
        (FieldwiseSerializer, ports.Serializer),
        (WholeRowSerializer, ports.Serializer),
        (ConnectedComponentsClusterer, ports.Clusterer),
        (SimilarityThresholdFilter, ports.Filter),
    ],
)
def test_adapters_declare_their_port(adapter: type, port: type) -> None:
    # issubclass() is rejected for runtime_checkable Protocols with non-method
    # members (e.g. Embedder's properties), so check explicit subclassing via
    # the MRO instead.
    assert port in adapter.__mro__


def test_spec_ports_build_artifacts_not_legacy_methods() -> None:
    # D6: specs expose a single `build` factory; the pre-refactor mutating
    # methods (`add`/`index`) and the artifacts' read methods are gone from them.
    assert hasattr(ports.VectorIndex, "build")
    assert not hasattr(ports.VectorIndex, "add")
    assert not hasattr(ports.VectorIndex, "search")
    assert hasattr(ports.Blocker, "build")
    assert not hasattr(ports.Blocker, "index")
    assert not hasattr(ports.Blocker, "query")


def test_artifact_ports_expose_read_methods_not_build() -> None:
    # D6: artifacts are produced by `build`; they never expose it.
    assert hasattr(ports.SearchableIndex, "search")
    assert hasattr(ports.SearchableIndex, "extended")  # incremental escape hatch
    assert not hasattr(ports.SearchableIndex, "build")
    assert not hasattr(ports.SearchableIndex, "add")
    assert hasattr(ports.BlockingIndex, "query")
    assert not hasattr(ports.BlockingIndex, "build")


@pytest.mark.parametrize("owner", [ports.BlockingIndex, DenseBlockingIndex])
def test_blocking_index_query_overrides_are_keyword_only(owner: type) -> None:
    # top_k / similarity_threshold are query-time overrides
    # so a ThresholdSweep reuses one built index. Pin them keyword-only with a
    # None default, on both the port and the reference adapter.
    sig = inspect.signature(owner.query)
    for name in ("top_k", "similarity_threshold"):
        param = sig.parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None
