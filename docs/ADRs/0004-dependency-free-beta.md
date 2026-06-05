# ADR-0004: Dependency-free beta — A1 implementation decisions

**Status:** Accepted (2026-06-05)
**Date:** 2026-06-05
**Deciders:** Alvaro (author); thesis advisor sign-off

## Context

ADR-0003 froze the v1 contract at the A0.5 gate. A1 fills the method bodies. The
maintainer chose a **dependency-free beta** — implement the no-extras surface
completely and ship it as `1.0.0b1`, deferring the heavy adapters. This ADR
records the implementation-phase decisions that shape that beta, so the design
record stays in-tree (ADRs are not published; see `architecture.md`).

## Decision

A1 implements the frozen contract for the **dependency-free** surface
(numpy + pandas) and ratifies:

- **D-4.1 — Beta scope: dependency-free only.** Complete `link` / `dedupe` /
  `match_pairs`, connected-components clustering, the linkage / blocking /
  clustering metrics, and `SimilarityThresholdFilter`, on the lexical reference
  stack. Cut as `1.0.0b1`.
- **D-4.2 — Heavy-adapter guards.** The four deferred adapters
  (`FaissFlatIndex`, `FaissSearchableIndex`, `SentenceTransformerEmbedder`,
  `LangChainMatcher`) raise `NotImplementedError("… planned for a future
  release")` rather than silently returning `None`. Guarded at `__init__` where
  one exists (so downstream methods stay coverage-excluded); the raise precedes
  any backend import, so the dependency cut holds and the guard tests need no
  extras.
- **D-4.3 — Clustering node universe.** `connected_components` clusters every
  record in `decisions ∪ errors`; a record that produced *no* candidate pair is
  absent from the result and cannot be clustered here. The full-record-universe
  path (an additive `connected_components(result, *, all_record_ids=…)`) is
  deferred to Phase B.
- **D-4.4 — B³ universe (Option A).** `clustering_metrics` scores B³ over the
  records present in the `ClusteringResult` (matches its frozen docstring); the
  full-universe variant rides on the D-4.3 escape hatch.
- **D-4.5 — Shared internals.** `metrics/_pairing.py` (the D1 directed/undirected
  pair key) and `clustering/_union_find.py` (transitive closure) are private
  modules shared across the metrics and clustering layers — one implementation
  each.
- **D-4.6 — Tuning/adjusted: accessors now, producers later.** The report
  accessors (`ThresholdSweep.best_f1` / `at_recall`,
  `AdjustedMetrics.recall_adjusted` / `f1_adjusted`) ship; their producers
  (`tune_threshold`, `adjusted_metrics`) stay Phase B.
- **D-4.7 — Coverage policy.** Branch coverage on, `fail_under = 100` enforced.
  The one unreachable defensive branch is `# pragma: no branch`. Revisit the
  threshold when adapter-only bodies (whose tests are `adapter`-marked and
  excluded from the coverage run) land.
- **D-4.8 — Examples.** `02_deduplication.py` stays the semantic + LLM design
  mock; `04_dedupe.py` is the runnable dependency-free dedup. CI smoke-runs the
  dependency-free examples (`00`, `03`, `04`).

## Options Considered

The pivotal fork was **D-4.4** (B³ record universe):

| Option | Complexity | Trade-off |
|---|---|---|
| **A — score over records-in-clustering (chosen)** | Low | Matches the frozen `clustering_metrics` docstring; minimal surface; the zero-candidate edge is rare at the default `top_k` |
| B — additive `all_record_ids=` for full-universe B³ | Low–Med | Airtight full-universe B³, but adds public surface now; the user passes ids from the frame |

A was chosen for the beta; B remains a clean, non-breaking Phase-B follow-up.

## Consequences

**Easier**
- A clean, fully-tested, installable beta: everything a plain
  `pip install denselinkage` (no extras) does **runs**.
- Predictable evolution — a wide additive corridor, a narrow known breaking one.

**Harder**
- A few declared-but-deferred adapters ship raising `NotImplementedError`
  (documented as experimental).

**To revisit (Phase B, all additive)**
- The heavy adapters; the full-record-universe clustering path (D-4.3/D-4.4);
  `tune_threshold` / `adjusted_metrics` producers; the ergonomic candidate-pair
  affordances (`DenseLinker.block`, `LinkageResult.from_candidate_frame`).

## Action Items

1. [x] Batch 1 — verbs + clustering bodies.
2. [x] Batch 2 — metrics + filtering bodies; shared `_pairing` / `_union_find`.
3. [x] Heavy-adapter guards + runnable `examples/04`; CI runs the dep-free examples.
4. [x] Branch coverage + `fail_under = 100`.
5. [ ] **Packaging**: version → `1.0.0b1`, `Development Status :: 4 - Beta`,
   `CHANGELOG`, `uv build` → `twine check` → clean-venv smoke test → TestPyPI →
   Trusted Publishing → tag `v1.0.0b1`.

## References

- ADR-0001 (spec→artifact), ADR-0003 (pre-freeze freeze gate).
- `docs/development/roadmap.md` (phase boundaries), `CHANGELOG.md`.
