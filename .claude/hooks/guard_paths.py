"""PreToolUse guard for the two paths where an edit is expensive to get wrong.

Reads the hook payload on stdin and writes a permission decision on stdout.

- ``paper/`` is the frozen evidence behind the published ISE 2026 paper.
  Editing it invalidates reported results, so edits are denied outright.
- ``src/denselinkage/core/ports.py`` carries the frozen contract. What the
  freeze binds is the public surface enumerated in
  ``docs/development/freeze-gate.md`` — ports, models, results, errors,
  orchestration, metrics — so a docstring edit is fine and a signature edit is a
  major-version event. The two are hard to tell apart from the tool input alone,
  so this asks rather than guesses.

Both paths are compared against the repository root derived from ``__file__``
(see ``_payload``), so the guard fires on *this* repository's ``paper/`` and not
on any other directory that happens to be called ``paper``.

Invoked as ``uv run --no-sync python .claude/hooks/guard_paths.py || exit 0``.
``uv run`` rather than a bare ``python`` so the interpreter does not depend on
PATH, and ``--no-sync`` so a hook never mutates the project environment as a
side effect of an edit. The ``|| exit 0`` is load-bearing: a PreToolUse hook
that exits 2 DENIES the tool call, and the interpreter exits 2 when it cannot
open the script — which is what happens whenever the session cwd is not the
repository root, because the command path is relative. Without the tail, a
session started one directory down blocks every edit it attempts.

Fails open on every other error path too: unparseable input, an unrecognised
payload shape, or a path outside the repository all leave the decision to the
normal permission flow, because a guard that blocks every edit when it misparses
is a guard people switch off.

This is a fast local check. It is not a second line of defence, because there is
no first one: no CI workflow inspects ``paper/`` or ``ports.py`` (verify with
``grep -rn 'paper\\|ports\\|freeze' .github/workflows/``), and the matcher covers
only the file-editing tools, so a write issued through ``Bash`` bypasses it.
Human review is the enforcement mechanism.
"""

import json
import sys

from _payload import REPO_ROOT, read_stdin_target

PAPER = REPO_ROOT / "paper"
PORTS = REPO_ROOT / "src" / "denselinkage" / "core" / "ports.py"

DENY_PAPER = """\
`paper/` is the frozen evidence behind the published ISE 2026 paper, and this \
edit would change it. Rerunning, tidying or regenerating anything under \
`paper/probe/` invalidates results reported in a published venue.

Read it freely. `paper/probe/verify_probe.py` may be RUN unchanged against a new \
single-file Matcher adapter as a conformance self-check.

If this edit is genuinely intended, make it yourself outside the agent session."""

ASK_PORTS = """\
This edits `src/denselinkage/core/ports.py`, which carries the frozen contract.

ALLOWED (the freeze binds the public surface enumerated in
docs/development/freeze-gate.md, not the file bytes):
  docstrings, comments, formatting.

BREAKING, and a MAJOR version event (ADR-0003):
  adding or removing a member on an existing Protocol; changing any signature,
  parameter name, default, or annotation; changing a field type.

Adding a whole new Protocol is additive and ships in a minor release, but it is
an architecture decision: propose it rather than merging it in passing.

Confirm which of these the edit is."""


def main() -> int:
    target = read_stdin_target()
    if target is None:
        return 0

    decision = reason = None
    if target.is_relative_to(PAPER):
        decision, reason = "deny", DENY_PAPER
    elif target == PORTS:
        decision, reason = "ask", ASK_PORTS

    if decision is None:
        return 0

    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision,
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
