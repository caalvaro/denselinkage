"""The two coverage configurations must name the same adapter modules.

A heavy adapter is registered twice, at opposite polarity: `omit` in
`pyproject.toml` keeps it out of the dependency-free gate, and `include` in
`.coveragerc.adapter` scopes the adapter job's 100% gate to exactly those
modules (ADR-0005). AGENTS.md states the rule and nothing enforced it.

The asymmetry is why this matters. Forgetting the `omit` entry fails the matrix
loudly, because the module's backend-touching body is unreachable without the
extras. Forgetting the `include` entry **silently ungates the module**: the
adapter job stops measuring it and its coverage can fall to zero unnoticed. A
prose rule guards the quiet direction.

This is the fourth hand-maintained list issue #31 removes, and the one the issue
did not name. It is a narrower check than its siblings: it enforces that the two
configurations *agree*, not that either is complete. The heavy set is not
derivable from source, measured on
`src/denselinkage/indexing/faiss_searchable_index.py`, which is registered in
both configurations while calling neither `require()` nor any heavy import.
"""

import configparser
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_ADAPTER_RC = _REPO_ROOT / ".coveragerc.adapter"


def _omitted_modules() -> set[str]:
    """`[tool.coverage.report] omit` from pyproject.toml.

    `tomllib` is 3.11+. The 3.10 matrix leg skips rather than parsing TOML with
    a regex, which is the kind of thing this project rejects elsewhere. Three
    legs of four is enough to catch a mismatch on the PR that introduces it.
    """
    tomllib = pytest.importorskip(
        "tomllib", reason="tomllib is 3.11+; this check runs on the other legs"
    )
    config = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    omit = config["tool"]["coverage"]["report"]["omit"]
    return {entry for entry in omit if entry.endswith(".py")}


def _included_modules() -> set[str]:
    parser = configparser.ConfigParser()
    parser.read(_ADAPTER_RC, encoding="utf-8")
    raw = parser["report"]["include"]
    return {line.strip() for line in raw.splitlines() if line.strip()}


def test_the_two_coverage_configs_register_the_same_modules() -> None:
    omitted = _omitted_modules()
    included = _included_modules()
    assert omitted == included, (
        "The two coverage configurations disagree about the heavy adapters.\n"
        f"  omitted in pyproject.toml only: {sorted(omitted - included)}\n"
        f"  included in .coveragerc.adapter only: {sorted(included - omitted)}\n"
        "A new adapter module must be registered in BOTH, at opposite polarity "
        "(AGENTS.md). Missing from `omit` fails the matrix loudly; missing from "
        "`include` SILENTLY ungates the module, so the adapter job stops "
        "measuring it and nothing goes red.\n"
        "This check enforces agreement, not completeness: the heavy set cannot "
        "be derived from source, so adding an adapter still means editing both "
        "files by hand."
    )


@pytest.mark.parametrize("pattern", sorted(_included_modules()))
def test_every_registered_adapter_module_exists(pattern: str) -> None:
    """A registration that names no file gates nothing.

    `include` and `omit` take glob patterns, so a typo or a renamed module is
    not an error to coverage. It simply matches nothing, and the module it was
    meant to gate goes unmeasured.
    """
    relative = re.sub(r"^\*/denselinkage/", "", pattern)
    assert (_REPO_ROOT / "src" / "denselinkage" / relative).is_file(), (
        f"{pattern} is registered for coverage but matches no file.\n"
        "Coverage treats an unmatched pattern as empty rather than as an error, "
        "so the module it names is silently unmeasured. Fix the path or drop "
        "the entry from both configurations."
    )
