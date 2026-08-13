"""The frozen public API is what the committed snapshot says it is.

This is the mechanical half of *extend, never modify*. Before it existed the
rule was enforced by nothing: ``core/ports.py`` was byte-identical between
``v1.0.0-freeze`` and ``v1.0.0``, and no test captured a port signature.

mypy is only a partial backstop and it fails open. Widening a port while leaving
an adapter alone is caught (``Signature of "match" incompatible with supertype``)
but only until someone applies the obvious repair of updating the first-party
adapter in the same commit. *Narrowing* a port is not caught at all, and
narrowing breaks every caller typed against it. Third-party implementers are
invisible to mypy entirely, so this repository stays green while their code
breaks. See issue #30 for the two experiments.

The snapshot is derived from the AST, so it is invariant to docstrings, comments
and formatting (ADR-0006) and covers the enumerated surface rather than
``core/`` alone (ADR-0007).
"""

import json

import pytest

from denselinkage.core import errors
from tests._api_snapshot import (
    SNAPSHOT_PATH,
    diff_surface,
    extract_surface,
    format_violation,
    load_snapshot,
    serialize,
)


def _committed() -> dict:
    """The snapshot as committed, without the regeneration provenance."""
    snapshot = load_snapshot()
    snapshot.pop("authority", None)
    return snapshot


def _current() -> dict:
    return extract_surface()


def test_frozen_surface_matches_the_snapshot() -> None:
    """Every parsed public API detail is unchanged since the last regeneration."""
    changes = diff_surface(_committed(), _current())
    assert not changes, format_violation(changes)


def test_snapshot_is_canonical_on_disk() -> None:
    """The committed file is exactly what the generator writes.

    A hand-edit that happens to be semantically equivalent (reordered keys,
    different indentation) still fails here, because the regeneration script is
    the only supported way to move the contract.
    """
    stored = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    authority = stored.get("authority")
    assert authority, (
        "tests/api_snapshot.json has no `authority` key.\n"
        "Every regeneration records the ADR or issue that authorises the "
        "contract change. Regenerate with:\n"
        "    python -m tests._api_snapshot --regenerate --authority <ADR-#### "
        "or #issue>"
    )
    expected = _current()
    expected["authority"] = authority
    assert SNAPSHOT_PATH.read_text(encoding="utf-8") == serialize(expected), (
        "tests/api_snapshot.json is not in canonical form. Do not hand-edit it; "
        "regenerate with `python -m tests._api_snapshot --regenerate "
        "--authority <ref>`."
    )


def test_port_count_is_the_number_the_paper_reports() -> None:
    """Ten ports, asserted against the snapshot rather than a literal.

    The paper's ten-ports claim previously rested on two hand-written lists in
    two files, neither of which was checked against the module. Carrying the
    count in the snapshot means it moves only on deliberate regeneration, while
    a literal here would turn every additive port into a test edit.
    """
    committed = _committed()
    current = _current()
    assert current["protocol_count"] == committed["protocol_count"], (
        f"core/ports.py now defines {current['protocol_count']} Protocols; the "
        f"snapshot records {committed['protocol_count']}.\n"
        "A NEW Protocol is additive and ships in a MINOR release (ADR-0003), "
        "but it is still an architecture decision: propose it, then regenerate "
        "the snapshot citing the ADR or issue that accepted it."
    )


def test_match_decision_has_no_error_field() -> None:
    """The three failure tiers stay disjoint (AGENTS.md).

    A soft, per-pair failure is the sibling ``MatchError`` value in
    ``LinkageResult.errors``. Adding an error field to ``MatchDecision`` would
    merge the tiers and let a caller treat an undecided pair as a decision.
    """
    models = next(m for m in _current()["modules"] if m["path"] == "core/models.py")
    decision = next(c for c in models["classes"] if c["name"] == "MatchDecision")
    fields = [m["name"] for m in decision["members"] if m["kind"] == "field"]
    assert fields == ["is_match", "confidence", "rationale"], (
        f"MatchDecision fields are now {fields}.\n"
        "Tier 1 of three: a Matcher returns one outcome per input pair, and a "
        "pair it cannot decide yields a MatchError VALUE in that list, never an "
        "error field on the decision. Merging the tiers is forbidden without an "
        "ADR (AGENTS.md, 'Failure tiers, disjoint on purpose')."
    )


@pytest.mark.parametrize("wrong_parent", [ValueError, TypeError])
def test_hard_failures_stay_outside_the_api_misuse_tier(
    wrong_parent: type[Exception],
) -> None:
    """Tier 2 and tier 3 never merge.

    API misuse raises a plain ``ValueError`` and sits *outside*
    ``DenseLinkageError`` on purpose, so ``except DenseLinkageError`` guarding
    data handling cannot swallow a caller's own bug. Reparenting the root would
    leave every class name and the whole taxonomy intact, so the snapshot's
    ``bases`` record catches the source edit and this catches the runtime fact.
    """
    assert not issubclass(errors.DenseLinkageError, wrong_parent), (
        f"DenseLinkageError now subclasses {wrong_parent.__name__}.\n"
        "The three failure tiers are disjoint on purpose (AGENTS.md): tier 2 is "
        "DenseLinkageError for hard data failures, tier 3 is plain ValueError "
        "for API misuse. Merging them makes `except DenseLinkageError` swallow "
        "programmer errors it was written to let through."
    )
