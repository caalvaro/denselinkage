---
description: Run the exact checks CI runs, and report the real output
---

Run these from the repository root, in order, and paste the actual output of each. Do not
summarise, and do not report a step as passing without its output.

```bash
uv sync --dev
uv run ruff check src/ tests/ examples/
uv run ruff format --check src/ tests/ examples/
uv run mypy src/ examples/
uv run python -m compileall examples
uv run pytest -m "not adapter and not slow" --cov=denselinkage --cov-report=term
```

Two of these differ from the short form people run by habit, and the difference is the
whole point: bare `uv run mypy` checks `src/` only because `[tool.mypy] files = ["src"]`,
silently skipping `examples/`, and bare `uv run pytest` never exercises the
`fail_under = 100` gate.

If the extras are installed, also run the adapter gate:

```bash
uv sync --dev --extra all
uv run pytest -m adapter --cov=denselinkage --cov-config=.coveragerc.adapter --cov-report=term-missing
```

If anything fails, assume your change caused it. `main` is green. Never attribute a
coverage drop to an unrelated change: the gate is exact.
