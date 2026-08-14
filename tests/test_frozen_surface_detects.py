"""Negative controls: the frozen-surface gate fails when it should.

``tests/test_frozen_surface.py`` proves the gate passes on unmodified source.
That is the happy path, and on its own it exercises none of the detection or the
message: with no changes, ``diff_surface`` returns an empty list and
``format_violation`` never runs. Coverage of ``tests/_api_snapshot.py`` was 68%
before this module existed, and every uncovered line was on the failure path.

A gate whose failure path never executes is a gate nobody has seen work. These
cases mutate a scratch copy of the package, one edit at a time, and assert both
that the change is caught and that it is classified the way ADR-0003's
add/remove asymmetry says it should be.

Each case is also a claim about the contract. ``NARROW a port`` is here because
mypy fails open on it (issue #30); ``drop frozen=`` is here because it breaks the
purity invariant while leaving every field byte-identical; the three docstring
and comment cases are here because ADR-0006 says formatting is not the contract.
"""

import json
from pathlib import Path

import pytest

from tests import _api_snapshot
from tests._api_snapshot import (
    ADDITIVE,
    BREAKING,
    RELAXING,
    Change,
    diff_surface,
    extract_surface,
    format_violation,
    load_snapshot,
)
from tests._api_snapshot import main as _snapshot_main
from tests._mutation import plant

# (id, module, find, replace, expected top severity or None for "must not fire")
_MUTATIONS: list[tuple[str, str, str, str, str | None]] = [
    (
        "narrow-a-port",
        "core/ports.py",
        "        batch_size: int | None = None,\n"
        "        show_progress: bool = False,\n",
        "        batch_size: int | None = None,\n",
        BREAKING,
    ),
    (
        "widen-a-port",
        "core/ports.py",
        "    def match(self, pairs: Sequence[CandidatePair])",
        "    def match(self, pairs: Sequence[CandidatePair], *, timeout: float = 0.0)",
        BREAKING,
    ),
    (
        "add-member-to-existing-protocol",
        "core/ports.py",
        "@runtime_checkable\nclass Filter(Protocol):",
        "@runtime_checkable\nclass Filter(Protocol):\n"
        "    def reset(self) -> None: ...\n",
        BREAKING,
    ),
    (
        "add-a-new-protocol",
        "core/ports.py",
        "@runtime_checkable\nclass Filter(Protocol):",
        "@runtime_checkable\nclass Reranker(Protocol):\n"
        "    def rerank(self, n: int) -> int: ...\n\n\n"
        "@runtime_checkable\nclass Filter(Protocol):",
        ADDITIVE,
    ),
    (
        "remove-a-port-member",
        "core/ports.py",
        "    def extended(self, vectors: Vectors, ids: Sequence[RecordId])"
        ' -> "SearchableIndex":',
        "    def _extended(self, vectors: Vectors, ids: Sequence[RecordId])"
        ' -> "SearchableIndex":',
        RELAXING,
    ),
    (
        "reorder-dataclass-fields",
        "core/models.py",
        "    is_match: bool\n    confidence: float | None = None\n"
        "    rationale: str | None = None",
        "    is_match: bool\n    rationale: str | None = None\n"
        "    confidence: float | None = None",
        BREAKING,
    ),
    (
        "drop-frozen-from-a-model",
        "core/models.py",
        "@dataclass(frozen=True, slots=True)\nclass Record:",
        "@dataclass(slots=True)\nclass Record:",
        BREAKING,
    ),
    (
        "drop-kw-only-from-denselinker",
        "linkage/dense_linker.py",
        "frozen=True, slots=True, kw_only=True",
        "frozen=True, slots=True",
        BREAKING,
    ),
    (
        "change-the-vectors-dtype",
        "core/ports.py",
        "Vectors = npt.NDArray[np.float32]",
        "Vectors = npt.NDArray[np.float64]",
        BREAKING,
    ),
    (
        "change-a-field-type",
        "core/models.py",
        "    similarity_score: float | None = None",
        "    similarity_score: float = 0.0",
        BREAKING,
    ),
    (
        "add-a-required-field",
        "core/models.py",
        "    reason: str",
        "    reason: str\n    code: int",
        BREAKING,
    ),
    (
        "add-an-optional-field",
        "core/models.py",
        "    rationale: str | None = None",
        "    rationale: str | None = None\n    provenance: str | None = None",
        ADDITIVE,
    ),
    (
        "remove-a-name-from-core-all",
        "core/__init__.py",
        '    "IncompatibleStore",\n',
        "",
        BREAKING,
    ),
    (
        "change-a-to-frame-column-schema",
        "core/results.py",
        '"similarity"',
        '"score"',
        BREAKING,
    ),
    # ADR-0006: docstrings, comments and formatting are outside the freeze.
    (
        "edit-a-class-docstring",
        "core/ports.py",
        '"""Maps text to dense vectors.',
        '"""REWRITTEN. Maps text to dense vectors.',
        None,
    ),
    (
        "edit-the-module-docstring",
        "core/models.py",
        '"""Core domain value objects."""',
        '"""Core domain value objects.\n\nAn added paragraph.\n"""',
        None,
    ),
    (
        "add-a-comment",
        "core/errors.py",
        "class DenseLinkageError(Exception):",
        "# an added comment\nclass DenseLinkageError(Exception):",
        None,
    ),
]


def _mutated_surface(tmp_path: Path, module: str, find: str, replace: str) -> dict:
    """Copy the package, apply one edit, and derive the surface from the copy.

    The planting itself lives in ``tests/_mutation.py``, shared with
    ``tests/test_contract_detects.py``. Both suites rest on a pattern that no
    longer matches being an error rather than a silent no-op, and that rule
    belongs in one place.
    """
    return extract_surface(plant(tmp_path, [(module, find, replace)]))


def _committed() -> dict:
    snapshot = load_snapshot()
    snapshot.pop("authority", None)
    return snapshot


@pytest.mark.parametrize(
    ("module", "find", "replace", "expected"),
    [pytest.param(*case[1:], id=case[0]) for case in _MUTATIONS],
)
def test_gate_classifies_each_contract_change(
    tmp_path: Path, module: str, find: str, replace: str, expected: str | None
) -> None:
    changes = diff_surface(
        _committed(), _mutated_surface(tmp_path, module, find, replace)
    )

    if expected is None:
        assert not changes, (
            "The gate fired on a change ADR-0006 puts outside the freeze "
            "(docstrings, comments, formatting). A false positive here makes "
            "the contract documentation unmaintainable, which is the whole "
            "reason the snapshot is AST-derived rather than a blob hash.\n\n"
            + format_violation(changes)
        )
        return

    assert changes, (
        "The gate did NOT fire on a real contract change. This is a false "
        "negative in the mechanism enforcing the project's central published "
        "claim, so the derivation in tests/_api_snapshot.py is not recording "
        "whatever this mutation moved."
    )
    assert changes[0].severity == expected, (
        f"Expected top severity {expected}, got {changes[0].severity}. "
        "Severity follows ADR-0003's add/remove asymmetry table and decides "
        "whether the change needs a major version, so a misclassification is "
        "as wrong as a miss.\n\n" + format_violation(changes)
    )


def test_regeneration_refuses_without_the_explicit_flag() -> None:
    """A bare invocation must not rewrite the contract.

    ``--update`` deliberately does not exist. Regenerating is the signal that
    the public API moved, so it has to be something a contributor types on
    purpose rather than reaches for when a test is red.
    """
    with pytest.raises(SystemExit) as excinfo:
        _snapshot_main([])
    assert excinfo.value.code != 0


def test_regeneration_refuses_without_a_cited_authority() -> None:
    """A contract change needs a decision behind it.

    If no ADR and no issue authorises the change, the change is not ready. The
    authority is recorded in the snapshot so the regeneration commit is
    traceable to what permitted it.
    """
    with pytest.raises(SystemExit) as excinfo:
        _snapshot_main(["--regenerate"])
    assert excinfo.value.code != 0


def test_regeneration_writes_a_snapshot_the_gate_then_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regenerate, then verify. The two halves must agree.

    If the writer and the reader ever disagree, a contributor who follows the
    documented recovery lands in a loop where the gate rejects the file the
    tool just produced.
    """
    target = tmp_path / "api_snapshot.json"
    monkeypatch.setattr(_api_snapshot, "SNAPSHOT_PATH", target)

    assert _snapshot_main(["--regenerate", "--authority", "#30"]) == 0
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["authority"] == "#30"

    stored = dict(written)
    stored.pop("authority")
    assert not diff_surface(stored, extract_surface())


def test_failure_message_states_rule_authority_and_escalation() -> None:
    """The message is the gate's whole user interface (issue #32).

    A reader who does not know the architecture learns from it that an
    expression was false, and concludes the cheapest repair is to weaken the
    test. These four elements are what prevent that.
    """
    message = format_violation(
        [
            Change(
                severity=BREAKING,
                module="core/ports.py",
                symbol="Matcher.match",
                verb="changed",
                was="match(self, pairs)",
                now="match(self, pairs, *, timeout: float = 0.0)",
                note="every third-party implementer must now provide this member",
            )
        ]
    )
    assert "FROZEN CONTRACT VIOLATION" in message
    assert "BREAKING (1):" in message
    assert "Matcher.match" in message
    assert "EXTEND, NEVER MODIFY" in message
    assert "0003-pre-freeze-contract-ratification.md" in message
    assert "releasing.md" in message
    assert "DO NOT edit tests/api_snapshot.json by hand" in message
    assert "--regenerate --authority" in message
