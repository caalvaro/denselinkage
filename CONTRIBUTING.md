# Contributing

Thanks for your interest in `denselinkage`.

## Project stage

`denselinkage` is in **beta** against a **frozen** public contract (the A0.5
freeze gate — see [`docs/development/freeze-gate.md`](docs/development/freeze-gate.md)).
The dependency-free core (`link` / `dedupe` / `match_pairs`, connected-components
clustering, the linkage / blocking / clustering metrics, and filtering) **and**
the four heavy adapters (`SentenceTransformerEmbedder`,
`FaissFlatIndex` / `FaissSearchableIndex`, `LangChainMatcher`) are implemented and
tested at 100% branch coverage. Evolution is **extend, never modify**: add an
optional field with a default, a sibling type, or a new classmethod — never change
a frozen signature. See [`docs/development/decisions.md`](docs/development/decisions.md)
and the [ADRs](docs/ADRs/).

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

## Optional extras (for running the heavy adapters)

```bash
uv sync --dev --extra all      # faiss-cpu, sentence-transformers, langchain
```
