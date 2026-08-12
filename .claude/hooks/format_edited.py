"""PostToolUse hook: format the file that was just edited.

Runs ``ruff format`` on a single Python file under ``src/``, ``tests/`` or
``examples/``, so an edit arrives already matching what ``ruff format --check``
asserts in CI.

Formatting only. ``ruff check --fix`` is deliberately NOT run: it deletes unused
imports, and a hook that fires on every edit would strip an import written one
edit before the line that uses it. Lint violations are caught by ``ruff check``
in CI and by the pre-commit hook, where a whole-file view makes the fix correct.

Invoked as ``uv run --no-sync python .claude/hooks/format_edited.py``. ``uv run``
rather than a bare ``python`` so the interpreter does not depend on PATH, and
``--no-sync`` so a hook never mutates the project environment as a side effect of
an edit.

Fails open on every error path, including ruff being absent: a formatter that
blocks the session when the toolchain is missing is worse than an unformatted
file, which CI catches anyway.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOTS = ("src/", "tests/", "examples/")
VENV_RUFF = (Path(".venv/Scripts/ruff.exe"), Path(".venv/bin/ruff"))


def _ruff_command() -> list[str]:
    """Prefer the project venv's ruff, then the running interpreter's."""
    for candidate in VENV_RUFF:
        if candidate.is_file():
            return [str(candidate)]
    return [sys.executable, "-m", "ruff"]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        raw = str((payload.get("tool_input") or {}).get("file_path") or "")
    except Exception:
        return 0

    if not raw.endswith(".py"):
        return 0

    path = Path(raw)
    if not path.is_file():
        return 0

    posix = raw.replace("\\", "/")
    if not any(f"/{r}" in f"/{posix}" for r in ROOTS):
        return 0

    try:
        subprocess.run(
            [*_ruff_command(), "format", str(path)],
            capture_output=True,
            timeout=45,
            check=False,
        )
    except Exception:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
