"""Structure-stage contract tests.

The package is declarations only (signatures with ``...`` bodies), so these
assert the *shape* of the public API — that it imports with zero heavy deps
and the types/ports exist as expected. Behavioural tests come with the
implementation.
"""

import dataclasses

import pytest

import denselinkage as dl
from denselinkage.blocking import DenseBlocker
from denselinkage.core import models, ports, results
from denselinkage.embedding import HashedNGramEmbedder, SentenceTransformerEmbedder
from denselinkage.indexing import FaissFlatIndex, NumpyFlatIndex
from denselinkage.matching import LangChainMatcher, RetryPolicy, ThresholdMatcher
from denselinkage.serialize import (
    FieldwiseSerializer,
    TemplateSerializer,
    WholeRowSerializer,
)

EXPECTED_PRELUDE = {
    "BlockingMetrics",
    "Clustering",
    "DenseLinker",
    "FieldwiseSerializer",
    "LabeledPairs",
    "LinkageIndex",
    "LinkageMetrics",
    "LinkageResult",
    "Source",
    "TemplateSerializer",
    "WholeRowSerializer",
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
        results.LinkageMetrics,
        results.BlockingMetrics,
        results.Clustering,
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
    [ports.Serializer, ports.Embedder, ports.VectorIndex, ports.Blocker, ports.Matcher],
)
def test_ports_are_runtime_checkable_protocols(port: type) -> None:
    # @runtime_checkable protocols expose _is_runtime_protocol.
    assert getattr(port, "_is_runtime_protocol", False)


@pytest.mark.parametrize(
    ("adapter", "port"),
    [
        (DenseBlocker, ports.Blocker),
        (HashedNGramEmbedder, ports.Embedder),
        (SentenceTransformerEmbedder, ports.Embedder),
        (NumpyFlatIndex, ports.VectorIndex),
        (FaissFlatIndex, ports.VectorIndex),
        (ThresholdMatcher, ports.Matcher),
        (LangChainMatcher, ports.Matcher),
        (TemplateSerializer, ports.Serializer),
        (FieldwiseSerializer, ports.Serializer),
        (WholeRowSerializer, ports.Serializer),
    ],
)
def test_adapters_declare_their_port(adapter: type, port: type) -> None:
    # issubclass() is rejected for runtime_checkable Protocols with non-method
    # members (e.g. Embedder's properties), so check explicit subclassing via
    # the MRO instead.
    assert port in adapter.__mro__
