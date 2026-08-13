---
description: Run the exact checks CI runs, and report the real output
---

Run these from the repository root, in order, and paste the actual output of each. Do not
summarise, and do not report a step as passing without its output.

```bash
uv sync --locked --extra dev
uv run ruff check src/ tests/ examples/ .claude/hooks/
uv run ruff format --check src/ tests/ examples/ .claude/hooks/
uv run mypy src/ examples/ .claude/hooks/ tests/_api_snapshot.py
uv run python -m compileall examples
uv run python examples/00_quickstart.py
uv run python examples/03_custom_embedder.py
uv run python examples/04_dedupe.py
uv run python examples/05_failure_accounting.py
uv run pytest -m "not adapter and not slow" --cov=denselinkage --cov-report=term
```

The `test` job additionally re-runs mypy against its oldest leg, which is the only
environment installing the lock's numpy 2.2.6 fork. Reproduce it with a 3.10 interpreter:

```bash
UV_PROJECT_ENVIRONMENT=.venv-310 uv sync --locked --extra dev --extra faiss --python 3.10
UV_PROJECT_ENVIRONMENT=.venv-310 uv run --no-sync mypy --python-version=3.10 src/ examples/
```

Five of these differ from the short form people run by habit, and the difference is the
whole point:

- `--locked` is what CI uses. Without it a stale lock is silently rewritten here and the
  divergence only surfaces on the PR, where every job asserts the lock is unchanged.
- `uv sync --extra dev` installs the toolchain; `dev` is an entry in
  `[project.optional-dependencies]`, not a PEP 735 dependency group, so `uv sync --dev`
  is a no-op flag that then *uninstalls* ruff, mypy, pytest and pytest-cov.
- Bare `uv run mypy` checks `src/` only, because `[tool.mypy] files = ["src"]` silently
  skips `examples/`.
- Bare `uv run pytest` never exercises the `fail_under = 100` gate.
- Compiling the examples is not running them. CI executes the four dependency-free ones,
  so an example that imports cleanly and raises at runtime fails on the PR, not here.

If the extras are installed, also run the adapter gate:

```bash
uv sync --locked --extra dev --extra faiss --extra sentence-transformers --extra langchain
uv run pytest -m adapter --cov=denselinkage --cov-config=.coveragerc.adapter --cov-report=term-missing
```

One CI job has no local equivalent: `core-only` installs the package with no extras and
asserts that no heavy backend is importable and that `import denselinkage` pulls none into
`sys.modules`. It needs a clean environment, so reproducing it locally means a throwaway
venv (`UV_PROJECT_ENVIRONMENT=.venv-core uv sync --locked --extra dev`).
A module-scope `import faiss` passes every check above and fails there.

If anything fails, assume your change caused it. `main` is green. Never attribute a
coverage drop to an unrelated change: the gate is exact.
