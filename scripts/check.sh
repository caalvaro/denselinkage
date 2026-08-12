#!/usr/bin/env bash
# Local CI parity — runs the same checks GitHub Actions runs, against the
# repo's single ``.venv`` (created from ``.python-version``). The full Python
# matrix (3.10–3.13) lives in CI; this reproduces lint + format + type-check +
# compile + tests once locally. Bash twin of scripts/check.ps1; used by the
# pre-push hook (Git for Windows runs hooks under Git Bash).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Windows venvs put binaries in Scripts/; POSIX venvs put them in bin/.
if [ -d ".venv/Scripts" ]; then
    BIN=".venv/Scripts"
    EXT=".exe"
else
    BIN=".venv/bin"
    EXT=""
fi

if [ ! -x "$BIN/ruff$EXT" ] || [ ! -x "$BIN/mypy$EXT" ] || [ ! -x "$BIN/pytest$EXT" ]; then
    echo "Missing dev tools in .venv — run:" >&2
    echo "  uv venv .venv" >&2
    echo "  uv pip install --python $BIN/python$EXT -e '.[dev]'" >&2
    exit 1
fi

echo "== ruff check (src, tests, examples) =="
"$BIN/ruff$EXT" check src/ tests/ examples/

echo "== ruff format --check =="
"$BIN/ruff$EXT" format --check src/ tests/ examples/

echo "== mypy (src + examples) =="
"$BIN/mypy$EXT" src/ examples/

echo "== compileall examples =="
"$BIN/python$EXT" -m compileall -q examples

# --cov is what makes this parity: pytest alone never exercises the
# `fail_under = 100` gate that CI enforces.
echo "== pytest (CI markers, with the 100% coverage gate) =="
"$BIN/pytest$EXT" -m "not adapter and not slow" -q --cov=denselinkage --cov-report=term

echo "All checks passed."
