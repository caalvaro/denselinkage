"""Plant a violation in a scratch copy of the package.

Shared by the two negative-control suites. Both prove a gate fails when it
should, which is the only way to know a gate works: a check that has never been
seen to fail has not been seen to do anything.

Multi-edit by design. The most valuable control is the false-positive guard,
"a fully wired new exception is accepted", and wiring one takes four edits
across two files. A single-edit harness cannot express it, and without it the
suites would only prove the gates are loud, never that they are right.
"""

import shutil
from pathlib import Path

from tests._api_snapshot import PACKAGE_ROOT

#: One edit: the module, the text to find, and what to put in its place.
Edit = tuple[str, str, str]


def plant(tmp_path: Path, edits: list[Edit], package_root: Path = PACKAGE_ROOT) -> Path:
    """Copy the package into ``tmp_path`` and apply every edit in order.

    Returns the copy's root, ready to hand to any derivation.

    A pattern that no longer matches raises rather than silently applying
    nothing, because a control that quietly stops mutating still passes and
    reports that the gate works.
    """
    tree = tmp_path / "denselinkage"
    shutil.copytree(package_root, tree)
    for module, find, replace in edits:
        target = tree / module
        source = target.read_text(encoding="utf-8")
        if find not in source:
            raise AssertionError(
                f"The mutation pattern for {module} no longer matches:\n"
                f"  {find!r}\n"
                "The source moved under the control. Update the pattern so the "
                "case still plants what it claims to; do not delete the case, "
                "and do not let it pass by mutating nothing."
            )
        target.write_text(source.replace(find, replace, 1), encoding="utf-8")
    return tree
