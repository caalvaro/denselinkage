"""Negative controls: the derived contract checks fail when they should.

``tests/test_contract.py`` proves the derivations agree with the package as it
stands. That is the happy path, and on its own it cannot distinguish a working
gate from one that returns an empty inventory and passes vacuously.

These plant a violation in a scratch copy and assert the derived fact the
corresponding test asserts over. They assert the fact rather than running the
test, because each test is a set operation over that fact, and because the
runtime half of some checks cannot be pointed at a copy on disk.

Every case is also a claim about the contract. The three "must not fire" cases
matter as much as the rest: without them the first contributor to add a
legitimate adapter concludes the gate is broken, and the cheapest apparent
repair for a false positive is to exclude a symbol from the derivation, which
restores the fail-open whitelist issue #31 deleted.
"""

from pathlib import Path

import pytest

from tests import _contract_scan as scan
from tests._mutation import Edit, plant

_ERRORS = "core/errors.py"
_CORE_INIT = "core/__init__.py"
_PORTS = "core/ports.py"

_PORT_NAMES = set(scan.ports())


# --- the taxonomy is closed and fully exported -------------------------------


def test_an_exception_missing_from_both_alls_is_visible(tmp_path: Path) -> None:
    tree = plant(
        tmp_path,
        [
            (
                _ERRORS,
                "__all__ = [",
                "class PlantedError(DenseLinkageError):\n"
                '    """Planted."""\n\n\n__all__ = [',
            )
        ],
    )
    names = {c["name"] for c in scan.taxonomy(tree)}
    assert "PlantedError" in names
    assert "PlantedError" not in (scan.module_all(_ERRORS, tree) or [])
    assert "PlantedError" not in (scan.module_all(_CORE_INIT, tree) or [])


def test_an_exception_missing_from_core_all_is_visible(tmp_path: Path) -> None:
    """The exact defect that shipped: IncompatibleStore (issue #31, PR #29)."""
    tree = plant(tmp_path, [(_CORE_INIT, '    "IncompatibleStore",\n', "")])
    names = {c["name"] for c in scan.taxonomy(tree)}
    assert "IncompatibleStore" in names
    assert "IncompatibleStore" not in (scan.module_all(_CORE_INIT, tree) or [])


def test_an_exception_missing_from_errors_all_is_visible(tmp_path: Path) -> None:
    tree = plant(tmp_path, [(_ERRORS, '    "IncompatibleStore",\n', "")])
    names = {c["name"] for c in scan.taxonomy(tree) if c["name"] != scan.ROOT_ERROR}
    assert "IncompatibleStore" in names
    assert "IncompatibleStore" not in (scan.module_all(_ERRORS, tree) or [])


def test_an_unrooted_exception_is_visible(tmp_path: Path) -> None:
    tree = plant(
        tmp_path,
        [
            (
                _ERRORS,
                "__all__ = [",
                'class WeirdError(Exception):\n    """Planted."""\n\n\n__all__ = [',
            )
        ],
    )
    planted = next(c for c in scan.taxonomy(tree) if c["name"] == "WeirdError")
    assert planted["bases"] == ["Exception"]


# --- the family is closed: no member declared outside core/errors.py ---------


@pytest.mark.parametrize(
    ("case", "edits", "expected"),
    [
        pytest.param(
            "plain",
            [
                (
                    "_store/reference_store.py",
                    "_FORMAT = 1",
                    "class SneakyError(IncompatibleStore):\n"
                    '    """Planted."""\n\n\n_FORMAT = 1',
                )
            ],
            ["SneakyError"],
            id="plain-subclass",
        ),
        pytest.param(
            "aliased",
            [
                (
                    "blocking/dense_blocker.py",
                    "class DenseBlocker(",
                    "from denselinkage.core.errors import "
                    "DenseLinkageError as _DLE\n\n\n"
                    "class AliasedError(_DLE):\n"
                    '    """Planted."""\n\n\n'
                    "class DenseBlocker(",
                )
            ],
            ["AliasedError"],
            id="aliased-import",
        ),
        pytest.param(
            "transitive",
            [
                (
                    "_store/reference_store.py",
                    "_FORMAT = 1",
                    "class MidError(IncompatibleStore):\n"
                    '    """Planted."""\n\n\n_FORMAT = 1',
                ),
                (
                    "mining/hard_negatives.py",
                    "def mine_hard_negatives(",
                    "from denselinkage._store.reference_store import MidError\n\n\n"
                    "class LeafError(MidError):\n"
                    '    """Planted."""\n\n\n'
                    "def mine_hard_negatives(",
                ),
            ],
            ["LeafError", "MidError"],
            id="transitive-two-hops",
        ),
    ],
)
def test_an_error_subclass_declared_outside_is_visible(
    tmp_path: Path, case: str, edits: list[Edit], expected: list[str]
) -> None:
    """Including through an import alias and through two hops of inheritance.

    A fixed point rather than one pass, because a subclass of a subclass is
    still caught by ``except DenseLinkageError`` and still absent from the
    documented taxonomy.
    """
    tree = plant(tmp_path, edits)
    found = sorted(d.name for d in scan.error_family_outside(tree))
    assert found == sorted(expected), f"{case}: got {found}"


# --- adapters declare and satisfy a port -------------------------------------


def test_an_adapter_that_drops_its_port_is_visible(tmp_path: Path) -> None:
    tree = plant(
        tmp_path,
        [
            (
                "matching/threshold_matcher.py",
                "class ThresholdMatcher(Matcher):",
                "class ThresholdMatcher:",
            )
        ],
    )
    record = next(
        e["record"]
        for e in scan.adapters(tree)
        if e["record"]["name"] == "ThresholdMatcher"
    )
    assert scan.declared(record, _PORT_NAMES) == ()


def test_a_brand_new_adapter_is_collected(tmp_path: Path) -> None:
    """The point of the change: a new adapter needs no row added anywhere."""
    tree = plant(tmp_path, [])
    (tree / "serializing" / "rogue_serializer.py").write_text(
        '"""Planted."""\n\nfrom collections.abc import Mapping\n'
        "from typing import Any\n\n\n"
        "class RogueSerializer:\n"
        "    def serialize(self, record: Mapping[str, Any]) -> str:\n"
        '        return ""\n',
        encoding="utf-8",
    )
    names = [e["record"]["name"] for e in scan.adapters(tree)]
    assert "RogueSerializer" in names
    record = next(
        e["record"]
        for e in scan.adapters(tree)
        if e["record"]["name"] == "RogueSerializer"
    )
    assert scan.declared(record, _PORT_NAMES) == ()


def test_an_adapter_that_declares_the_wrong_port_is_visible(tmp_path: Path) -> None:
    tree = plant(
        tmp_path,
        [
            (
                "indexing/numpy_flat_index.py",
                "class NumpyFlatIndex(VectorIndex):",
                "class NumpyFlatIndex(SearchableIndex):",
            ),
            (
                "indexing/numpy_flat_index.py",
                "from denselinkage.core.ports import VectorIndex",
                "from denselinkage.core.ports import SearchableIndex",
            ),
        ],
    )
    record = next(
        e["record"]
        for e in scan.adapters(tree)
        if e["record"]["name"] == "NumpyFlatIndex"
    )
    assert scan.declared(record, _PORT_NAMES) == ("SearchableIndex",)
    missing = scan.missing_members(record, scan.port_members(tree)["SearchableIndex"])
    assert missing, "declaring the wrong port must surface as missing members"


# --- ports carry @runtime_checkable ------------------------------------------


def test_a_new_protocol_without_runtime_checkable_is_visible(tmp_path: Path) -> None:
    """The snapshot records this only as an addition, so it needs its own rule."""
    tree = plant(
        tmp_path,
        [
            (
                _PORTS,
                "@runtime_checkable\nclass Filter(Protocol):",
                "class Reranker(Protocol):\n"
                "    def rerank(self, n: int) -> int: ...\n\n\n"
                "@runtime_checkable\nclass Filter(Protocol):",
            )
        ],
    )
    derived = scan.ports(tree)
    assert "Reranker" in derived
    decorators = {d["name"] for d in derived["Reranker"]["decorators"]}
    assert "runtime_checkable" not in decorators


# --- must NOT fire ------------------------------------------------------------


def test_a_fully_wired_new_exception_is_accepted(tmp_path: Path) -> None:
    """The false-positive guard. Four edits across two files.

    Without this the suite would only prove the gates are loud, never that a
    legitimate addition passes.
    """
    tree = plant(
        tmp_path,
        [
            (
                _ERRORS,
                "__all__ = [",
                "class PlantedError(DenseLinkageError):\n"
                '    """Planted."""\n\n\n__all__ = [',
            ),
            (_ERRORS, '    "InvalidTopK",', '    "InvalidTopK",\n    "PlantedError",'),
            (_CORE_INIT, "    InvalidTopK,", "    InvalidTopK,\n    PlantedError,"),
            (
                _CORE_INIT,
                '    "InvalidTopK",',
                '    "InvalidTopK",\n    "PlantedError",',
            ),
        ],
    )
    names = {c["name"] for c in scan.taxonomy(tree) if c["name"] != scan.ROOT_ERROR}
    assert names <= set(scan.module_all(_ERRORS, tree) or [])
    assert names <= set(scan.module_all(_CORE_INIT, tree) or [])
    assert scan.error_family_outside(tree) == []


def test_a_declared_new_adapter_is_accepted(tmp_path: Path) -> None:
    tree = plant(tmp_path, [])
    (tree / "serializing" / "good_serializer.py").write_text(
        '"""Planted."""\n\nfrom collections.abc import Mapping\n'
        "from typing import Any\n\n"
        "from denselinkage.core.ports import Serializer\n\n\n"
        "class GoodSerializer(Serializer):\n"
        "    def serialize(self, record: Mapping[str, Any]) -> str:\n"
        '        return ""\n',
        encoding="utf-8",
    )
    record = next(
        e["record"]
        for e in scan.adapters(tree)
        if e["record"]["name"] == "GoodSerializer"
    )
    assert scan.declared(record, _PORT_NAMES) == ("Serializer",)
    assert scan.missing_members(record, scan.port_members(tree)["Serializer"]) == []


def test_a_private_conforming_class_is_not_an_adapter(tmp_path: Path) -> None:
    """ADR-0006 puts private names outside the contract."""
    tree = plant(tmp_path, [])
    (tree / "serializing" / "_private_helper.py").write_text(
        '"""Planted."""\n\nfrom collections.abc import Mapping\n'
        "from typing import Any\n\n\n"
        "class _Hidden:\n"
        "    def serialize(self, record: Mapping[str, Any]) -> str:\n"
        '        return ""\n',
        encoding="utf-8",
    )
    assert "_Hidden" not in [e["record"]["name"] for e in scan.adapters(tree)]


def test_a_docstring_edit_changes_no_derived_fact(tmp_path: Path) -> None:
    """ADR-0006: formatting and docstrings are not the contract."""
    baseline_errors = {c["name"] for c in scan.taxonomy()}
    baseline_adapters = {e["record"]["name"] for e in scan.adapters()}
    tree = plant(
        tmp_path,
        [
            (
                _ERRORS,
                '"""Root of every hard failure raised by denselinkage."""',
                '"""COMPLETELY REWRITTEN. Root of every hard failure."""',
            )
        ],
    )
    assert {c["name"] for c in scan.taxonomy(tree)} == baseline_errors
    assert {e["record"]["name"] for e in scan.adapters(tree)} == baseline_adapters


# --- vacuity ------------------------------------------------------------------


def test_the_guard_sees_a_blinded_port_derivation(tmp_path: Path) -> None:
    """test_the_derivation_is_not_blind must be able to fail."""
    tree = plant(tmp_path, [(_PORTS, "(Protocol)", "(object)")])
    assert len(scan.ports(tree)) < len(_PORT_NAMES)


def test_the_guard_sees_a_blinded_adapter_derivation(tmp_path: Path) -> None:
    tree = plant(tmp_path, [])
    for name in ("serializing", "indexing"):
        for path in (tree / name).glob("*.py"):
            path.unlink()
    assert len(scan.adapters(tree)) < len(scan.adapters())
