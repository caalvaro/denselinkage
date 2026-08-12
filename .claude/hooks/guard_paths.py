"""PreToolUse guard for the two paths where an edit is expensive to get wrong.

Reads the hook payload on stdin and writes a permission decision on stdout.

- ``paper/`` is the frozen evidence behind the published ISE 2026 paper.
  Editing it invalidates reported results, so edits are denied outright.
- ``src/denselinkage/core/ports.py`` carries the frozen contract. The freeze
  binds the parsed public API rather than the file bytes (ADR-0006), so a
  docstring edit is fine and a signature edit is a major-version event. The two
  are hard to tell apart from the tool input alone, so this asks rather than
  guesses.

Invoked as ``uv run --no-sync python .claude/hooks/guard_paths.py``. ``uv run``
rather than a bare ``python`` so the interpreter does not depend on PATH, and
``--no-sync`` so a hook never mutates the project environment as a side effect of
an edit. The script itself is stdlib-only and runs under any Python 3.10+.

Fails open: any unexpected input leaves the decision to the normal permission
flow, because a guard that blocks every edit when it misparses is a guard people
switch off. The corollary is worth knowing: if ``uv`` or the project environment
is missing, this guard does not fire at all. It is a fast local check, not the
enforcement mechanism. CI is the enforcement mechanism.
"""

import json
import sys

PAPER = "paper/"
PORTS = "src/denselinkage/core/ports.py"

DENY_PAPER = """\
`paper/` is the frozen evidence behind the published ISE 2026 paper, and this \
edit would change it. Rerunning, tidying or regenerating anything under \
`paper/probe/` invalidates results reported in a published venue.

Read it freely. `paper/probe/verify_probe.py` may be RUN unchanged against a new \
single-file Matcher adapter as a conformance self-check.

If this edit is genuinely intended, make it yourself outside the agent session."""

ASK_PORTS = """\
This edits `src/denselinkage/core/ports.py`, which carries the frozen contract.

ALLOWED (ADR-0006 scopes the freeze to the parsed public API, not the bytes):
  docstrings, comments, formatting.

BREAKING, and a MAJOR version event (ADR-0003):
  adding or removing a member on an existing Protocol; changing any signature,
  parameter name, default, or annotation; changing a field type.

Adding a whole new Protocol is additive and ships in a minor release, but it is
an architecture decision: propose it rather than merging it in passing.

Confirm which of these the edit is."""


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        tool_input = payload.get("tool_input") or {}
        path = str(tool_input.get("file_path") or "").replace("\\", "/")
    except Exception:
        return 0

    if not path:
        return 0

    decision = reason = None
    if f"/{PAPER}" in f"/{path}":
        decision, reason = "deny", DENY_PAPER
    elif path.endswith(PORTS):
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
