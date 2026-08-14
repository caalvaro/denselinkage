"""Contract tests: the *shape* of the frozen public API.

These assert shape by reflection — ``dataclasses.fields``, ``inspect.signature``,
``__mro__``, ``__all__``, ``_is_runtime_protocol`` — and never call the code under
test: that the package imports with zero heavy dependencies, that the ports and
types exist as expected, and that each first-party adapter subclasses its port.

Behaviour is asserted elsewhere, by the value-based tests in the sibling files
(``test_metrics.py``, ``test_blocking.py``, ``test_linker_verbs.py`` and the rest).
Keep the two kinds separate: a contract test that computes a value, or a behaviour
test that inspects a signature, belongs in the other file.
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
    assert set(dl.__all__) == EXPECTED_PRELUDE, (
        "denselinkage.__all__ changed.\n"
        f"  missing: {sorted(EXPECTED_PRELUDE - set(dl.__all__))}\n"
        f"  added:   {sorted(set(dl.__all__) - EXPECTED_PRELUDE)}\n"
        "The top-level prelude is the first thing a user imports, so removing a "
        "name breaks working code. Adding one is additive and ships in a MINOR "
        "release (ADR-0003); update this list in the same commit and say so in "
        "the PR. Note this is the PRELUDE, not the frozen contract: "
        "denselinkage.core.__all__ is gated separately by "
        "tests/test_frozen_surface.py."
    )
    for name in dl.__all__:
        assert hasattr(dl, name), (
            f"denselinkage.__all__ lists {name!r} but the module has no such "
            "attribute, so `from denselinkage import " + name + "` raises "
            "ImportError. Either export it or drop it from __all__."
        )


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
    assert getattr(port, "_is_runtime_protocol", False), (
        f"{port.__name__} lost its @runtime_checkable decorator.\n"
        "Every port carries it so third parties can assert conformance at "
        "runtime; removing it breaks their isinstance checks while this "
        "repository stays green, because mypy cannot see them (AGENTS.md).\n"
        "Note this is separate from the ban on runtime isinstance dispatch "
        "AGAINST a port inside denselinkage (core/ports.py:30-34), which is "
        "settled and stays."
    )


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
    assert port in adapter.__mro__, (
        f"{adapter.__name__} does not explicitly subclass {port.__name__}.\n"
        "Structural typing would still accept it, so this is a deliberate "
        "belt-and-braces rule: explicit subclassing is what makes mypy report a "
        "signature drift as an [override] error on the adapter, and what makes "
        "the port discoverable from the adapter's MRO.\n"
        f"Fix: `class {adapter.__name__}({port.__name__}):`. Do not delete the "
        "row from this table; a new adapter must be added to it, and deriving "
        "this list rather than hand-maintaining it is issue #31."
    )


_SPEC_ARTIFACT_RULE = (
    "ADR-0001 splits every stateful component into a SPEC that builds and an "
    "ARTIFACT that is read. `build()` returns a new artifact and never mutates "
    "self, which is what makes `link(a, b) == index(a).query(b)` hold. Putting a "
    "mutating `add`/`index` back on a spec, or `build` on an artifact, collapses "
    "that split and is settled: it needs an ADR to reopen (AGENTS.md)."
)


def _assert_members(
    owner: type, present: tuple[str, ...], absent: tuple[str, ...]
) -> None:
    for name in present:
        assert hasattr(owner, name), (
            f"{owner.__name__}.{name} is gone.\n"
            f"Removing a port member is 'relaxing' under ADR-0003's asymmetry "
            f"table, but it still deletes something third parties call.\n"
            f"{_SPEC_ARTIFACT_RULE}"
        )
    for name in absent:
        assert not hasattr(owner, name), (
            f"{owner.__name__} gained a `{name}` member.\n"
            f"Adding a member to an EXISTING Protocol is BREAKING FOR "
            f"IMPLEMENTERS and forces a MAJOR version (ADR-0003): every "
            f"third-party implementer must now provide it, and mypy cannot see "
            f"them, so this repository stays green while their code breaks.\n"
            f"{_SPEC_ARTIFACT_RULE}"
        )


def test_spec_ports_build_artifacts_not_legacy_methods() -> None:
    # D6: specs expose a single `build` factory; the pre-refactor mutating
    # methods (`add`/`index`) and the artifacts' read methods are gone from them.
    _assert_members(ports.VectorIndex, ("build",), ("add", "search"))
    _assert_members(ports.Blocker, ("build",), ("index", "query"))


def test_artifact_ports_expose_read_methods_not_build() -> None:
    # D6: artifacts are produced by `build`; they never expose it.
    # `extended` is the incremental escape hatch.
    _assert_members(ports.SearchableIndex, ("search", "extended"), ("build", "add"))
    _assert_members(ports.BlockingIndex, ("query",), ("build",))


@pytest.mark.parametrize("owner", [ports.BlockingIndex, DenseBlockingIndex])
def test_blocking_index_query_overrides_are_keyword_only(owner: type) -> None:
    # top_k / similarity_threshold are query-time overrides
    # so a ThresholdSweep reuses one built index. Pin them keyword-only with a
    # None default, on both the port and the reference adapter.
    sig = inspect.signature(owner.query)
    for name in ("top_k", "similarity_threshold"):
        param = sig.parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{owner.__name__}.query({name}=...) is now "
            f"{param.kind.description}, not keyword-only.\n"
            "Parameter kind is part of the signature, so this is BREAKING and "
            "forces a MAJOR version (ADR-0003). These two are query-time "
            "overrides deliberately: keeping them keyword-only is what lets a "
            "ThresholdSweep reuse one built index instead of rebuilding per "
            "threshold."
        )
        assert param.default is None, (
            f"{owner.__name__}.query({name}=...) now defaults to "
            f"{param.default!r}, not None.\n"
            "Changing a default is BREAKING for every caller relying on the old "
            "one (ADR-0003). `None` means 'use the value the index was built "
            "with'; any other default silently overrides it."
        )
