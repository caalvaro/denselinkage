"""PostToolUse hook: format the file that was just edited.

Runs ``ruff format`` on a single Python file under ``src/``, ``tests/``,
``examples/`` or ``.claude/hooks/``, so an edit arrives already matching what
``ruff format --check`` asserts in CI.

The four roots are compared against the repository root derived from ``__file__``
(see ``_payload``), not matched as substrings, so a file in another checkout that
happens to live under a directory called ``src`` is left alone.

Formatting only. ``ruff check --fix`` is deliberately NOT run: it deletes unused
imports, and a hook that fires on every edit would strip an import written one
edit before the line that uses it. Lint violations are caught by ``ruff check``
in CI and by the pre-commit hook, where a whole-file view makes the fix correct.

Invoked as ``uv run --no-sync python .claude/hooks/format_edited.py || exit 0``.
``uv run`` rather than a bare ``python`` so the interpreter does not depend on
PATH, ``--no-sync`` so a hook never mutates the project environment as a side
effect of an edit, and ``|| exit 0`` so a session started outside the repository
root (where the relative command path does not resolve, and the interpreter
exits 2) gets a no-op rather than an error fed back into the transcript.

Fails open on every error path, including ruff being absent: a formatter that
blocks the session when the toolchain is missing is worse than an unformatted
file, which CI catches anyway.
"""

import subprocess
import sys
from pathlib import Path

from _payload import REPO_ROOT, read_stdin_target

ROOTS = tuple(
    REPO_ROOT / name for name in ("src", "tests", "examples", ".claude/hooks")
)
VENV_RUFF = (REPO_ROOT / ".venv/Scripts/ruff.exe", REPO_ROOT / ".venv/bin/ruff")


def _ruff_command() -> list[str]:
    """Prefer the project venv's ruff, then the running interpreter's."""
    for candidate in VENV_RUFF:
        if candidate.is_file():
            return [str(candidate)]
    return [sys.executable, "-m", "ruff"]


def _in_scope(path: Path) -> bool:
    return path.suffix == ".py" and any(path.is_relative_to(root) for root in ROOTS)


def main() -> int:
    target = read_stdin_target()
    if target is None or not _in_scope(target) or not target.is_file():
        return 0

    try:
        subprocess.run(
            [*_ruff_command(), "format", str(target)],
            capture_output=True,
            timeout=45,
            check=False,
        )
    except Exception:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
