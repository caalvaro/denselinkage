# ADR-0005: Heavy adapters — implementation & adapter-coverage policy

**Status:** Accepted (2026-06-05)
**Date:** 2026-06-05
**Deciders:** Alvaro (author); thesis advisor sign-off

## Context

ADR-0004 shipped the dependency-free beta and deferred the four heavy adapters
(`FaissFlatIndex`, `FaissSearchableIndex`, `SentenceTransformerEmbedder`,
`LangChainMatcher`), which raised `NotImplementedError`. This track fills those
bodies so the **headline method runs** — dense *semantic* blocking
(sentence-transformer embeddings → FAISS ANN) + LLM matching — alongside the
lexical reference stack. The work is purely **additive** (extend-never-modify):
no port or public signature changes, only `NotImplementedError` bodies replaced.

ADR-0004 **D-4.7** explicitly deferred one decision to this point: the
`fail_under = 100` coverage job runs `-m "not adapter and not slow"`, so once the
adapter bodies are real they go unexercised by that job and the gate would break.
This ADR ratifies how coverage adapts, and records the implementation invariants.

## Decision

- **D-5.1 — Adapter-coverage policy (the D-4.7 revisit).** Split coverage by
  surface. The dependency-free gate keeps `fail_under = 100`, with the four
  adapter modules added to `[tool.coverage.report] omit` so the no-extras surface
  stays airtight. A dedicated **`adapter-tests`** CI job installs the extras,
  runs `-m "adapter"`, and enforces its own coverage gate over the adapter
  modules. Net: every line is gated, on the job that can actually import its
  backend. Supersedes D-4.7's "revisit".
- **D-5.2 — Lazy-import discipline preserved.** Each adapter imports its backend
  *inside* the method via `denselinkage._optional.require` (install hint), then a
  plain `import` (typed `Any` through the existing mypy overrides). `import
  denselinkage` still pulls in no heavy backend; the `core-only` dependency-cut
  job is unchanged and still green. (The old "raise before any import" guard is
  replaced by "require inside the method".)
- **D-5.3 — Cosine via L2-normalized inner product (cross-stack parity).**
  `SentenceTransformerEmbedder.encode` sets `normalize_embeddings=True`;
  `FaissFlatIndex` builds `faiss.IndexFlatIP` (inner product, not L2). The score
  is therefore cosine on both backends, so `similarity_threshold` keeps its
  meaning when a user swaps hashed↔ST or numpy↔FAISS. A differential test pins
  FAISS neighbours to the numpy reference.
- **D-5.4 — `extended` stays deferred.** Implementing FAISS does **not** mean
  implementing incremental indexing: `FaissSearchableIndex.extended` raises the
  same `NotImplementedError("… out of scope for v1 (see ADR-0001)")` as the numpy
  artifact. The escape hatch's signature stays frozen for a future, non-breaking
  landing.
- **D-5.5 — LangChain: structured output, soft failures, hermetic tests.** The
  matcher binds `llm.with_structured_output(...)` (a stdlib `TypedDict` schema, so
  the matcher module needs no pydantic at import time); the prompt carries only
  the semantic question. Per-pair failures are retried per `RetryPolicy` and, on
  exhaustion, become a position-aligned `MatchError` — never an exception into the
  batch (ADR-0003). Tests inject a fake chat model (no API key), so the
  match/retry/error logic runs in CI.

## Consequences

**Easier**
- The headline dense-semantic + LLM stack runs and is testable end to end; the
  FAISS backend is a drop-in for numpy behind the `VectorIndex` port.
- Coverage stays meaningful on every line without weakening the dependency-free
  gate.

**Harder**
- CI now has a heavier `adapter-tests` job (faiss wheel + a cached ST model);
  mitigated by the existing cache steps.

**Unchanged / still deferred**
- The dependency cut, the frozen contract, and the `extend-never-modify` corridor.
- FAISS persistence in the Reference Store (the store still rejects non-numpy
  stacks with `NotImplementedError`); incremental `extended`; v2 `[train]`
  trainers. The dense-vs-lexical **benchmark** (Abt-Buy / DBLP-ACM) is a separate
  evaluation workstream, not library code.

## Release note

The adapters are additive, so the version/release vehicle is the open Track-D
scope call (continue the `1.0.0bN` beta line vs. cut `1.0.0` final then a `1.1.0`
feature release). The change is documented under `CHANGELOG.md → [Unreleased]`;
the number is chosen at release time.

## References

- ADR-0001 (spec→artifact; `extended` escape hatch), ADR-0003 (failure taxonomy /
  freeze gate), ADR-0004 (dependency-free beta; D-4.7).
- `docs/development/roadmap.md`, `CHANGELOG.md`.
