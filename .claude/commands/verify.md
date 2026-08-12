---
description: Run the exact checks CI runs, and report the real output
---

Run these from the repository root, in order, and paste the actual output of each. Do not
summarise, and do not report a step as passing without its output.

```bash
uv sync --extra dev
uv run ruff check src/ tests/ examples/ .claude/hooks/
uv run ruff format --check src/ tests/ examples/ .claude/hooks/
uv run mypy src/ examples/ .claude/hooks/
uv run python -m compileall examples
uv run python examples/00_quickstart.py
uv run python examples/03_custom_embedder.py
uv run python examples/04_dedupe.py
uv run python examples/05_failure_accounting.py
uv run pytest -m "not adapter and not slow" --cov=denselinkage --cov-report=term
```

Four of these differ from the short form people run by habit, and the difference is the
whole point:

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
uv sync --extra dev --extra all
uv run pytest -m adapter --cov=denselinkage --cov-config=.coveragerc.adapter --cov-report=term-missing
```

One CI job has no local equivalent: `core-only` installs the package with no extras and
asserts that no heavy backend is importable and that `import denselinkage` pulls none into
`sys.modules`. It needs a clean environment, so reproducing it locally means a throwaway
venv (`uv venv /tmp/core-only && uv pip install --python /tmp/core-only -e '.[dev]'`).
A module-scope `import faiss` passes every check above and fails there.

If anything fails, assume your change caused it. `main` is green. Never attribute a
coverage drop to an unrelated change: the gate is exact.
