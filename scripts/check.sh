#!/usr/bin/env bash
# Local CI parity — runs the same checks GitHub Actions runs.
# See scripts/check.ps1 for the rationale; this is the bash twin used by
# the pre-push hook (Git for Windows runs hooks under Git Bash).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Windows venvs put binaries in Scripts/; POSIX venvs put them in bin/.
if [ -d ".venv-py312/Scripts" ]; then
    PY312_BIN=".venv-py312/Scripts"
    PY310_BIN=".venv/Scripts"
    EXT=".exe"
else
    PY312_BIN=".venv-py312/bin"
    PY310_BIN=".venv/bin"
    EXT=""
fi

if [ ! -x "$PY312_BIN/mypy$EXT" ]; then
    echo "Missing $PY312_BIN — run:" >&2
    echo "  uv venv --python 3.12 .venv-py312" >&2
    echo "  uv pip install --python $PY312_BIN/python$EXT -e '.[dev]'" >&2
    exit 1
fi
if [ ! -x "$PY310_BIN/pytest$EXT" ]; then
    echo "Missing $PY310_BIN — run:" >&2
    echo "  uv venv --python 3.10 .venv" >&2
    echo "  uv pip install --python $PY310_BIN/python$EXT -e '.[dev,faiss]'" >&2
    exit 1
fi

echo "== ruff (py3.12) =="
"$PY312_BIN/ruff$EXT" check src/ tests/

echo "== mypy (py3.12) =="
"$PY312_BIN/mypy$EXT" src/

echo "== pytest (py3.10, CI markers) =="
"$PY310_BIN/pytest$EXT" -m "not adapter and not slow" -q

echo "All checks passed."
