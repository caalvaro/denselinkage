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
- Stateful components follow **spec→artifact** (D6): a stateless *spec*
  (`VectorIndex`, `Blocker`) exposes `build(...)` returning an immutable
  *artifact* (`SearchableIndex`, `BlockingIndex`). Per-dataset state lives only
  in artifacts, never on injected configuration — do not give a spec a mutating
  method or hand it a populated collaborator.
- Similarity-cutoff naming: the blocking stage names its retrieval cutoff
  `similarity_threshold` (it sits beside `top_k` as a retrieval knob on
  `DenseBlocker` / `BlockingIndex.query`); single-purpose decision/filter
  components whose class name already carries the qualifier use `threshold`
  (`ThresholdMatcher`, `SimilarityThresholdFilter`).

## Optional extras (for implementing/running adapters)

```bash
uv sync --dev --extra all      # faiss-cpu, sentence-transformers, langchain
```
