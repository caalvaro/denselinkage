"""PostToolUse hook: format the file that was just edited.

Runs ``ruff format`` then ``ruff check --fix`` on a single Python file under
``src/``, ``tests/`` or ``examples/``, so an agent's edit arrives already
matching what ``ruff format --check`` asserts in CI.

Fails open on every error path, including ruff being absent: a formatter that
blocks the session when the toolchain is missing is worse than an unformatted
file, which CI catches anyway.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOTS = ("src/", "tests/", "examples/")


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

    for args in (["format"], ["check", "--fix"]):
        try:
            subprocess.run(
                [sys.executable, "-m", "ruff", *args, str(path)],
                capture_output=True,
                timeout=45,
                check=False,
            )
        except Exception:
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
