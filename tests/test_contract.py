"""Contract tests: the *shape* of the public API, derived rather than listed.

These assert shape by reflection and never call the code under test. Behaviour
lives in the sibling value-based files (``test_metrics.py``, ``test_blocking.py``,
``test_linker_verbs.py``). A contract test that computes a value, or a behaviour
test that inspects a signature, belongs in the other file.

Every subject here is **derived from source** by ``tests/_contract_scan.py``.
Three hand-written lists used to live in this file and in two others: the
exception taxonomy, the ports, and the ``(adapter, port)`` pairs. A literal list
is fail-open. It cannot see anything added after it was written, and that is not
hypothetical: ``IncompatibleStore`` reached ``errors.__all__`` while missing from
``denselinkage.core`` entirely, and the taxonomy test passed throughout
(issue #31, PR #29). Adding a row is now impossible to forget, because there is
no row to add.

What is asserted where:

- The *frozen* surface, and whether it moved since the last release, is
  ``tests/test_frozen_surface.py`` against ``tests/api_snapshot.json``. That is a
  baseline diff, and its repair is a deliberate ``--regenerate --authority``.
- This file asserts invariants over the code as it stands now, whose repair is
  always to fix the source. Several hold at any point in history and so cannot
  be expressed as a diff against a baseline: a new ``Protocol`` missing
  ``@runtime_checkable`` is merely an addition to the snapshot, and a name in
  ``core.__all__`` that resolves to nothing is invisible to a check that reads
  ``__all__`` by AST and never imports.
- Adapters are outside the freeze by ADR-0007 and appear in no snapshot, so the
  conformance rules below are the only thing holding them.

ADR-0001 splits every stateful component into a SPEC that builds and an ARTIFACT
that is read: ``build()`` returns a new artifact and never mutates ``self``,
which is what makes ``link(a, b) == index(a).query(b)`` hold. The port member
sets encode that split, and the frozen-surface gate is what now holds them; this
note is here because a snapshot diff reports that a member moved and says
nothing about why ``VectorIndex`` has ``build`` and no ``add``.
"""

import importlib
import inspect
import pkgutil

import pytest

import denselinkage as dl
from denselinkage import core, metrics
from denselinkage.blocking import DenseBlockingIndex
from denselinkage.core import ports
from denselinkage.core.errors import DenseLinkageError
from denselinkage.matching import RetryPolicy
from tests import _contract_scan as scan

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

METRICS_REPORT_SYMBOLS = (
    "LinkageMetrics",
    "BlockingMetrics",
    "ClusteringMetrics",
    "ThresholdSweep",
    "AdjustedMetrics",
)

_PORTS = scan.ports()
_PORT_MEMBERS = scan.port_members()
_ADAPTERS = scan.adapters()
_TAXONOMY = scan.taxonomy()

_ADAPTER_IDS = [entry["record"]["name"] for entry in _ADAPTERS]
_ERROR_NAMES = [c["name"] for c in _TAXONOMY if c["name"] != scan.ROOT_ERROR]


def test_the_derivation_is_not_blind() -> None:
    """Guard the guards.

    Every parametrized test below is a set operation over these inventories. If
    a derivation silently returned nothing, each would pass vacuously and the
    suite would report a green contract over an empty world.
    """
    assert len(_PORTS) >= 10, f"only {len(_PORTS)} ports derived from core/ports.py"
    assert len(_ADAPTERS) >= 15, f"only {len(_ADAPTERS)} adapters derived"
    assert len(_TAXONOMY) >= 7, f"only {len(_TAXONOMY)} exceptions derived"


def test_prelude_surface() -> None:
    assert set(dl.__all__) == EXPECTED_PRELUDE, (
        "denselinkage.__all__ changed.\n"
        f"  missing: {sorted(EXPECTED_PRELUDE - set(dl.__all__))}\n"
        f"  added:   {sorted(set(dl.__all__) - EXPECTED_PRELUDE)}\n"
        "The top-level prelude is the first thing a user imports, so removing a "
        "name breaks working code. Adding one is additive and ships in a MINOR "
        "release (ADR-0003); update this list in the same commit. This is the "
        "PRELUDE, not the frozen contract: denselinkage.core.__all__ is gated "
        "by tests/test_frozen_surface.py."
    )
    for name in dl.__all__:
        assert hasattr(dl, name), (
            f"denselinkage.__all__ lists {name!r} but the module has no such "
            f"attribute, so `from denselinkage import {name}` raises ImportError."
        )


@pytest.mark.parametrize("name", sorted(_PORTS))
def test_every_port_is_runtime_checkable(name: str) -> None:
    """Derived from core/ports.py, so a new port is covered on the day it lands.

    The snapshot records the decorator, but a *new* Protocol is only an addition
    to it, so an undecorated one would be absorbed by a regeneration. This is
    the invariant that outlives any baseline.
    """
    port = getattr(ports, name)
    assert getattr(port, "_is_runtime_protocol", False), (
        f"{name} is a Protocol in core/ports.py without @runtime_checkable.\n"
        "Every port carries it so third parties can assert conformance at "
        "runtime; omitting it breaks their isinstance checks while this "
        "repository stays green, because mypy cannot see them (AGENTS.md).\n"
        "This is separate from the ban on runtime isinstance dispatch AGAINST a "
        "port inside denselinkage, which is settled and stays."
    )


@pytest.mark.parametrize("entry", _ADAPTERS, ids=_ADAPTER_IDS)
def test_every_adapter_declares_a_port(entry: dict) -> None:
    """A public class with a public callable outside the frozen surface names a port.

    Structural typing would accept an adapter that never says so, which is
    exactly why this is checked: explicit subclassing is what makes mypy report
    a signature drift as an [override] error, and what makes the port
    discoverable from the adapter's MRO.
    """
    record = entry["record"]
    names = set(_PORTS)
    assert scan.declared(record, names), (
        f"{record['name']} ({entry['module']}) declares no port.\n"
        f"It is public and has public callables, so it is treated as an "
        f"adapter. Ports available: {sorted(names)}.\n"
        "Three resolutions: subclass the port it implements; rename the member "
        "that made it look like an adapter; or make the class private with a "
        "leading underscore, which ADR-0006 puts outside the contract. If none "
        "applies, it is an architecture decision and needs an ADR.\n"
        "There is deliberately no exemption list: one would rebuild, slowly, "
        "the hand-written table issue #31 deleted."
    )


@pytest.mark.parametrize("entry", _ADAPTERS, ids=_ADAPTER_IDS)
def test_every_adapter_satisfies_the_port_it_declares(entry: dict) -> None:
    """Declaring a port is a promise about the calling convention.

    The port's parameters must be a prefix of the adapter's, and anything extra
    must be keyword-only with a default. That is mypy's compatible-override
    rule: a caller holding the port must be able to call the adapter.
    """
    record = entry["record"]
    for port_name in scan.declared(record, set(_PORTS)):
        required = _PORT_MEMBERS[port_name]
        missing = scan.missing_members(record, required)
        assert not missing, (
            f"{record['name']} declares {port_name} but does not provide "
            f"{missing}.\n"
            f"Defined at {entry['module']}. Every port member is required: a "
            "caller typed against the port will call it."
        )
        bad = scan.incompatible_members(record, required)
        assert not bad, (
            f"{record['name']}.{bad} does not satisfy {port_name}'s calling "
            "convention.\n"
            "The port's parameters must be a prefix of the adapter's, and any "
            "extra parameter must be keyword-only with a default, so that a "
            "caller holding the port can still call it (mypy's override rule)."
        )


@pytest.mark.parametrize("name", _ERROR_NAMES)
def test_every_error_is_rooted_and_exported(name: str) -> None:
    """The taxonomy is closed-world, derived from core/errors.py.

    The rule is that ``except DenseLinkageError`` catches the whole documented
    family and that every member is reachable from ``denselinkage.core``. The
    hand-written tuple this replaces missed ``IncompatibleStore`` on both counts
    and passed anyway (issue #31, PR #29).
    """
    exc = getattr(core, name, None)
    assert exc is not None, (
        f"{name} is declared in core/errors.py but is not importable from "
        "denselinkage.core.\n"
        "docs/development/contract.md states the taxonomy is all exported from "
        "denselinkage.core, so a caller can catch the family without reaching "
        "into a submodule. Add it to the imports and to __all__ in "
        "src/denselinkage/core/__init__.py."
    )
    assert issubclass(exc, DenseLinkageError), (
        f"{name} is in core/errors.py but does not subclass DenseLinkageError.\n"
        "Tier 2 of three is the hard-failure family, and its whole value is "
        "that one `except DenseLinkageError` catches all of it (AGENTS.md)."
    )
    errors_all = scan.module_all(scan.ERRORS_MODULE) or []
    assert name in errors_all, f"{name} is missing from errors.__all__."
    core_all = scan.module_all(scan.CORE_INIT) or []
    assert name in core_all, (
        f"{name} is missing from denselinkage.core.__all__.\n"
        "This is the exact defect that shipped: IncompatibleStore was in "
        "errors.__all__ and absent from core, and no test noticed."
    )


def test_no_error_subclass_is_declared_outside_the_taxonomy_module() -> None:
    """The family is closed. Every member lives in core/errors.py.

    A subclass declared elsewhere still gets caught by
    ``except DenseLinkageError`` while being invisible to the taxonomy, so the
    documented family and the real one drift apart.
    """
    stray = scan.error_family_outside()
    assert not stray, (
        "DenseLinkageError subclasses are declared outside core/errors.py:\n"
        + "\n".join(f"  {d.name} at {d.where}" for d in stray)
        + "\nMove them into src/denselinkage/core/errors.py and export them "
        "from denselinkage.core. The taxonomy is a published contract: a "
        "caller writing `except DenseLinkageError` expects the set of things "
        "it catches to be the documented one."
    )


def test_no_error_subclass_is_created_at_runtime() -> None:
    """The AST cannot see a class built by ``type()`` or a base bound by name.

    Importing every module and walking ``__subclasses__`` closes that gap. It
    costs well under a second and pulls in no heavy backend, because the
    dependency cut keeps those behind method-local imports.
    """
    for info in pkgutil.walk_packages(dl.__path__, prefix="denselinkage."):
        importlib.import_module(info.name)

    declared_here = {c["name"] for c in _TAXONOMY}
    seen: set[type] = set()
    stack = [DenseLinkageError]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    unexpected = sorted(c.__name__ for c in seen if c.__name__ not in declared_here)
    assert not unexpected, (
        f"DenseLinkageError subclasses exist at runtime that core/errors.py "
        f"does not declare: {unexpected}.\n"
        "They are caught by `except DenseLinkageError` while being absent from "
        "the documented taxonomy."
    )


@pytest.mark.parametrize("name", METRICS_REPORT_SYMBOLS)
def test_no_metrics_report_type_leaks_into_core(name: str) -> None:
    """ADR-0002: evaluation report types live in denselinkage.metrics, never core.

    A per-module snapshot records what each module contains and cannot express
    this negative, so it needs its own assertion.
    """
    assert name in metrics.__all__, f"{name} is not exported from denselinkage.metrics."
    assert hasattr(metrics, name), f"denselinkage.metrics.{name} does not resolve."
    assert not hasattr(core, name), (
        f"{name} is reachable from denselinkage.core.\n"
        "ADR-0002 keeps the evaluation report types out of core so that nothing "
        "in core depends on them. Moving one in inverts that dependency."
    )


@pytest.mark.parametrize("name", sorted(scan.module_all(scan.CORE_INIT) or []))
def test_every_core_export_resolves(name: str) -> None:
    """``core.__all__`` is read by AST elsewhere, so nothing checks it imports.

    A name listed but not imported raises ImportError for a caller doing
    ``from denselinkage.core import X`` while every static check stays green.
    """
    assert hasattr(core, name), (
        f"denselinkage.core.__all__ lists {name!r} but it does not resolve, so "
        f"`from denselinkage.core import {name}` raises ImportError."
    )


def test_retry_policy_is_a_value_object_not_an_adapter() -> None:
    """The one public class outside the surface that implements no port.

    Recorded because the adapter derivation excludes it by predicate rather than
    by name: it has no public callables. If it ever grows one it becomes an
    adapter candidate and must declare a port, which is the intended behaviour
    and not a bug in the sweep.
    """
    import dataclasses

    assert dataclasses.is_dataclass(RetryPolicy)
    assert RetryPolicy.__name__ not in _ADAPTER_IDS


def test_blocking_index_query_overrides_are_keyword_only() -> None:
    """The reference adapter's half. The port's half is in the snapshot.

    ``top_k`` and ``similarity_threshold`` are query-time overrides so that a
    ThresholdSweep reuses one built index instead of rebuilding per threshold.
    """
    sig = inspect.signature(DenseBlockingIndex.query)
    for name in ("top_k", "similarity_threshold"):
        param = sig.parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"DenseBlockingIndex.query({name}=...) is now "
            f"{param.kind.description}, not keyword-only."
        )
        assert param.default is None, (
            f"DenseBlockingIndex.query({name}=...) now defaults to "
            f"{param.default!r}. None means 'use the value the index was built "
            "with'; any other default silently overrides it."
        )
