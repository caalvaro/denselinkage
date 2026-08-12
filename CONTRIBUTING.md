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
uv sync --extra dev    # core deps (numpy, pandas) + dev tools
uv run pre-commit install
```

`dev` is an entry in `[project.optional-dependencies]`, not a PEP 735 dependency group, so
it needs `--extra dev`. `uv sync --dev` is a no-op flag that installs the runtime
dependencies and *uninstalls* ruff, mypy, pytest and pytest-cov.

## Checks (must pass before a PR)

```bash
uv run ruff check src/ tests/ examples/ .claude/hooks/
uv run ruff format --check src/ tests/ examples/ .claude/hooks/
uv run mypy src/ examples/ .claude/hooks/   # strict; CI checks all three too
uv run python -m compileall examples
uv run python examples/00_quickstart.py     # and 03, 04, 05
uv run pytest -m "not adapter and not slow" --cov=denselinkage --cov-report=term
```

These are what the `lint-and-type` and `test` CI jobs run
(`.github/workflows/ci.yml`) on Python 3.10–3.13, and `scripts/check.sh` /
`scripts/check.ps1` run the same sequence against `.venv`. Three of the shorter forms are
traps: bare `uv run mypy` checks `src/` only, because `[tool.mypy] files = ["src"]`
silently skips `examples/`; bare `uv run pytest` never exercises the `fail_under = 100`
gate; and compiling the examples is not running them, which CI also does.
`pre-commit` runs ruff + mypy on each commit, and shares the mypy gap.

A third CI job, `core-only`, has no local equivalent: it installs with no extras and
asserts that no heavy backend is importable and that `import denselinkage` pulls none into
`sys.modules`. Reproducing it needs a throwaway venv, so a module-scope `import faiss`
passes every check above and fails only on the PR.

With the extras installed, the adapter modules are gated separately:

```bash
uv run pytest -m adapter --cov=denselinkage --cov-config=.coveragerc.adapter
```

Open pull requests against `main`; there is no separate development or release
branch. `main` requires a pull request and passing status checks, with no bypass
actors. Releasing, version numbering, and the trunk-based branching model are
documented in [docs/development/releasing.md](docs/development/releasing.md).

## Where the conventions are written down

[docs/development/conventions.md](docs/development/conventions.md) records the coding,
style and testing conventions derived from the codebase: the design patterns in use and
their anti-patterns, what the linter does not enforce, how tests are written here, and the
checklist for adding an adapter. Read it before a first contribution.

[AGENTS.md](AGENTS.md) is the short form of the invariants, written for AI coding
assistants, and is a first-class contributor resource: it is the fastest statement of what
must not break. `CLAUDE.md` is a one-line pointer to it. The reviewed assistant
configuration lives in `.claude/` and is committed; `.claude/settings.local.json` is yours
and stays untracked.

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
