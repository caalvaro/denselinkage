# Contributing

Thanks for your interest in `denselinkage`.

## Project stage

The package is at the **structure stage**: `src/denselinkage/` defines the
public types, ports (`typing.Protocol`s) and signatures; method/function
bodies are `...` placeholders. The `examples/` are the design spec for the
intended API. Implementation lands incrementally against this frozen contract.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev          # core deps (numpy, pandas) + dev tools
uv run pre-commit install
```

## Checks (must pass before a PR)

```bash
uv run ruff check .            # lint
uv run ruff format .           # format (use --check in CI)
uv run mypy                    # strict type check (src/)
uv run pytest                  # contract tests
uv run python -m compileall examples
```

All of the above run in CI (`.github/workflows/ci.yml`) on Python
3.10–3.13. `pre-commit` runs ruff + mypy locally on each commit.

## Conventions

- Branch off `main`; open a PR; keep it focused.
- Keep the core dependency-free of heavy ML backends — FAISS / LangChain /
  sentence-transformers stay behind optional extras (`[faiss]`,
  `[langchain]`, `[sentence-transformers]`).
- First-party adapters subclass their port explicitly (completeness-checked
  by mypy); third-party code may conform structurally.
- Public types are typed and immutable where reasonable; the package ships a
  `py.typed` marker.

## Optional extras (for implementing/running adapters)

```bash
uv sync --dev --extra all      # faiss-cpu, sentence-transformers, langchain
```
