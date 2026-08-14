# Recorded decisions

> Part of the [denselinkage development docs](./README.md). This is the decision
> log for D1, D2, D3 and D6. The Resolvi-surfaced forks **D4** (matching-optional)
> and **D5** (`Clusterer` port) are recorded in
> [resolvi-conformance.md](./resolvi-conformance.md).

## D1 — `LabeledPairs` pair identity
**Ruling (recommended default, no override).** Pairs are stored **exactly as
given (ordered)** `(left_id, right_id)`; construction never symmetrizes.
Evaluation comparison is verb-dependent: `link` compares by order;
`dedupe` canonicalizes both gold and result pairs to an unordered key
(`frozenset({a, b})`) before comparing. Removes the silent recall/precision
fork. Recorded in: `LabeledPairs` docstring + `linkage_metrics` /
`blocking_metrics` / `pair_completeness_at_k` docstrings. Test:
`tests/test_gold_clustering_utils.py::test_labeledpairs_is_ordered_and_not_symmetrized`
(rehomed from the deleted `test_a05_contract.py` in issue #31; it asserts a computed
value, so it belongs with the behaviour tests).

**Amendment (A0.5, surfaced by oracle 4).** "Verb-dependent comparison" is not
recoverable from a `LinkageResult` / candidate list alone — the result does not
carry which verb produced it, and `LinkageResult`'s fields are frozen to
`{decisions, errors}`. The reference-slice implementation surfaced this gap: the
comparison verb is now an explicit `directed: bool = True` parameter on
`linkage_metrics` / `blocking_metrics` / `pair_completeness_at_k` (default
`True` = `link`/ordered; `dedupe` callers pass `directed=False`). Contract-shape,
landed pre-freeze. Locked by
`tests/test_quickstart_end_to_end.py::test_directed_flag_controls_pair_identity`.

## D2 — `match_pairs` input contract
**Ruling (recommended default, no override).** `CandidatePair.similarity_score`
is now `float | None = None`. A dense `Blocker` always sets it; externally
/ rule-blocked pairs passed to `DenseLinker.match_pairs` use `None`. Done
pre-freeze precisely because `CandidatePair` is `frozen` (can't tighten
later). The ergonomic `LinkageResult.from_candidate_frame` constructor is a
**Phase-B** addition, so `match_pairs` is contract-complete now but only
end-to-end usable in B. Recorded in: `CandidatePair` docstring +
`DenseLinker.match_pairs` docstring.

## D3 — `@runtime_checkable` on ports
**Ruling (recommended default, no override): keep it.** Retained ONLY so the
structure-stage contract test can assert `_is_runtime_protocol`; no runtime
`isinstance` dispatch against ports exists or is intended (first-party
adapters subclass their port explicitly; mypy completeness-checks them).
Recorded in: rationale comment in `src/denselinkage/core/ports.py` + here.

## D6 — Stateless-spec / immutable-artifact ports (spec→artifact)
**Ruling (recommended default, no override; confirmed).** Of the ports exactly
one — `VectorIndex` — is inherently stateful (`add()` accumulates); `Blocker`
is stateful only by containing one. Stateless **specs** are injected;
per-dataset state lives only in **artifacts** produced by a `build` step,
applying the `DenseLinker → index() → LinkageIndex` seam (already present)
uniformly one level down: `VectorIndex.build(vectors, ids) -> SearchableIndex`
and `Blocker.build(records) -> BlockingIndex`. This makes the `frozen`
immutability of `DenseLinker` honest — the `deepcopy` that would otherwise guard
it disappears — and makes amortized index reuse safe (the artifact *is* the
build-once / query-many unit).

**Classification — contract-shape (CS), the most severe breaking-if-late case.**
Adding the two artifact ports and reshaping the central blocking/indexing
calling convention *after* the freeze would break every index/blocker adapter
and the orchestrator; *before* it the change is signatures + `...` only and the
construction API is byte-identical (`examples/01`–`03` unchanged, still
type-check). Triage row: `paper_evaluation_data.md` §3.

**Sub-decisions (confirmed).** (1) `top_k` / `similarity_threshold` are
*query-time* parameters, overridable on `BlockingIndex.query(records, *,
top_k=None, similarity_threshold=None)`, so a `ThresholdSweep` reuses one built
index instead of rebuilding it. (2) Incremental indexing is out of scope for v1;
`SearchableIndex.extended(vectors, ids) -> SearchableIndex` is the designed,
not-yet-implemented escape hatch (returns a NEW artifact) — it also
pre-positions Resolvi's Reference-Store / incremental-ER pillar without
committing to it.

**Provenance (method note).** Unlike D1–D5 (surfaced by the Resolvi conformance
walk), D6 was surfaced by *attempting the first reference implementation*
(`with_defaults()` + the `link` vertical slice). Strict typing and
`compileall examples` both passed over the original mutable `add()/search()`
ports, because the examples *construct* a blocker but never *exercise*
`index()/query()`; the defect appears only when the contract is implemented.
This is the evidence motivating the freeze gate's fourth oracle —
reference-implementation attemptability (see [freeze-gate.md](./freeze-gate.md),
and the paper's freeze-gate refinement).

Recorded in: `core/ports.py` (four ports + rationale), `core/__init__.py`
(`__all__`), `indexing/__init__.py`, `blocking/__init__.py`, `linker.py`,
`tests/test_contract.py` (the two artifact ports joined the
adapter-declares-its-port and runtime-checkable fitness functions). Full record:
[`docs/ADRs/0001-statefull-components-as-artifacts.md`](../ADRs/0001-statefull-components-as-artifacts.md)
and the ADR-0001 summary below.

## ADR-0001 — Stateful components modeled as artifacts (spec→artifact)

Full record: [`docs/ADRs/0001-statefull-components-as-artifacts.md`](../ADRs/0001-statefull-components-as-artifacts.md).
A pre-freeze, contract-shape change (signatures + `...`), landed this pass.

**Principle.** Of the eight ports, exactly one (`VectorIndex`) is inherently
stateful; `Blocker` is stateful only by containment. Stateless *specs* are
injected; per-dataset state lives only in *artifacts* produced by a `build`
step — the `DenseLinker → index() → LinkageIndex` seam (already present),
applied uniformly. This makes the `frozen` immutability honest (no `deepcopy`)
and makes amortized index reuse safe (the artifact *is* the build-once /
query-many unit).

**Reshape (one law at three nested levels):**
- `VectorIndex.build(vectors, ids) -> SearchableIndex` — artifact exposes
  `search`, plus `extended(vectors, ids) -> SearchableIndex` (the incremental
  escape hatch; out of scope for v1, signature only).
- `Blocker.build(records) -> BlockingIndex` — artifact exposes `query`.
- `DenseLinker.index(source) -> LinkageIndex` — holds a `BlockingIndex` +
  `Matcher`; `index()` is delegation + composition (no `deepcopy`).

**Sub-decisions (confirmed).** (1) `top_k` / `similarity_threshold` are
query-time parameters, overridable on `BlockingIndex.query(records, *,
top_k=None, similarity_threshold=None)` so a `ThresholdSweep` reuses one built
index instead of rebuilding. (2) Incremental indexing is out of scope for v1;
`SearchableIndex.extended` is the designed, not-yet-implemented escape hatch.

**Surface impact.** New artifact ports `SearchableIndex` / `BlockingIndex` +
adapters `NumpySearchableIndex` / `FaissSearchableIndex` / `DenseBlockingIndex`,
exported from `denselinkage.core`. Construction API byte-identical
(`examples/01`–`03` unchanged, still type-check). Recorded in: `core/ports.py` +
`core/__init__.py` + `indexing` + `blocking` + `linker.py` +
`tests/test_contract.py` (the two new ports joined the adapter-declares-port and
runtime-checkable fitness functions). Contract-shape: adding these ports
post-freeze would be breaking, so they land now.

## D7 — Evaluation report types live in the metrics layer, not `core`
**Ruling (recommended default, no override; ADR-0002; implemented).** `core`
keeps only the contract/domain types — the outputs a port references
(`LinkageResult`, `ClusteringResult`), the gold / training value objects
(`LabeledPairs`, `TrainingPairs`), plus models, ports and errors. The evaluation
**report** types — `LinkageMetrics`, `BlockingMetrics`, `ClusteringMetrics`,
`ThresholdSweep`, `AdjustedMetrics` — move to `denselinkage.metrics`,
co-located with the functions that produce them, because **nothing in `core`
depends on them**. Decisive test: `core/ports.py` references only
`LinkageResult` / `ClusteringResult` / `TrainingPairs` from results, never the five
reports.

**Classification — contract-shape (CS).** Five public types change module; the
public prelude re-exports them from the new home, so `from denselinkage import
LinkageMetrics` is unchanged. Landed pre-freeze (2026-06-02). `LabeledPairs`
stays in `core` as domain ground-truth (not port-referenced, but a value object
like `Source`, and read beyond evaluation by Phase-B mining).

Full record: [`docs/ADRs/0002-evaluation-types-out-of-core.md`](../ADRs/0002-evaluation-types-out-of-core.md).
Recorded in: `core/__init__.py` (`__all__` + docstring), `denselinkage/__init__.py`
(prelude re-export), the `metrics` modules (`linkage`/`blocking`/`clustering`/
`tuning`/`adjusted`), and the fitness function `tests/test_contract.py`, whose
`test_no_metrics_report_type_leaks_into_core` asserts membership in `metrics` and
absence from `core` (issue #31 folded the former `test_a05_contract.py` and
`test_phase_a_additions.py` into it and derived their subjects from source). The `core/results.py` →
`core/outputs.py` rename is deferred (cosmetic; would churn every import site).

## D8 — Pre-freeze contract ratification (ADR-0003)
**Ruling (ratified; ADR-0003).** Freeze readiness is governed by the
**add/remove asymmetry**: post-freeze, new ports/types/methods and new optional
fields are additive (safe), but adding a member to an existing port or changing a
field type / signature is breaking — while removing an *unused* port member later
is cheap. So only existing-port signatures, field types, and calling conventions
must be correct *now*. Ratified calls:

- **D4 declined** — `matcher` stays required (`ThresholdMatcher` covers the
  degenerate "no real matcher" case); recorded as a decision, not a default.
- **Two-tier errors** — `ValueError` = API misuse / programmer error
  (`blocker=None`, wrong-length matcher return); `DenseLinkageError` =
  data/runtime hard failure. Documented on `core/errors.py`.
- **Keep + document speculative port surface** — `Embedder.model_id` /
  `embedding_dim`, the conformance ports (`Filter`, `Clusterer`, `Trainer`), and
  `SearchableIndex.extended` stay (the asymmetry punishes trimming); intended
  consumers documented rather than removed.
- **Frozen-object hashability** — standard Python behaviour; nothing hashes
  them; documented-or-ignored, out of freeze scope.
- **Freeze checklist** — the one remaining high-value task is an adversarial
  signatures-and-field-types pass; then flip the gate.

Full record: [`docs/ADRs/0003-pre-freeze-contract-ratification.md`](../ADRs/0003-pre-freeze-contract-ratification.md).

## D9 — Dependency-free beta: A1 implementation (ADR-0004)
**Ruling (ratified; ADR-0004).** A1 fills the frozen contract's bodies for the
**dependency-free beta** (`1.0.0b1`). Key calls:

- **Beta scope** — dependency-free only (numpy + pandas); the heavy extras
  (faiss / sentence-transformers / langchain) are deferred.
- **Heavy-adapter guards** — the four deferred adapters raise
  `NotImplementedError("… planned for a future release")` rather than returning
  `None`; guarded at `__init__` where present (the raise precedes any backend
  import, so the dependency cut holds).
- **Clustering universe** — `connected_components` clusters records in
  `decisions ∪ errors`; `clustering_metrics` scores B³ over the records present
  (Option A). The full-record-universe path is an additive Phase-B escape hatch.
- **Shared internals** — `metrics/_pairing.py` (D1 key) and
  `clustering/_union_find.py` (transitive closure) are DRY'd across the metrics
  and clustering layers.
- **Tuning/adjusted** — report accessors ship; their producers
  (`tune_threshold`, `adjusted_metrics`) stay Phase B.
- **Coverage** — branch coverage + `fail_under = 100` enforced.
- **Examples** — `02` stays the semantic + LLM design mock; `04_dedupe.py` is the
  runnable dependency-free dedup; CI smoke-runs `00` / `03` / `04`.

Full record: [`docs/ADRs/0004-dependency-free-beta.md`](../ADRs/0004-dependency-free-beta.md).

## D10 — Heavy adapters: A2 implementation (ADR-0005)
**Ruling (ratified; ADR-0005).** A2 fills the four deferred adapter bodies so the
headline dense-semantic + LLM stack runs; additive to the frozen contract. Key
calls:

- **Replace the guards** — `SentenceTransformerEmbedder`, `FaissFlatIndex` /
  `FaissSearchableIndex`, and `LangChainMatcher` are implemented behind their
  optional extras; the `NotImplementedError` guards (D9) are gone.
- **Lazy-import discipline** — each backend is imported *inside* the method via
  `_optional.require`; `import denselinkage` still pulls in no backend (the
  `core-only` dependency-cut job is unchanged).
- **Cosine parity** — ST encodes L2-normalized; FAISS uses `IndexFlatIP`, so the
  score is cosine on both backends and `similarity_threshold` keeps its meaning
  across hashed↔ST and numpy↔FAISS (pinned by a FAISS↔numpy differential test).
- **`extended` stays deferred** — implementing FAISS is not implementing
  incremental indexing; `FaissSearchableIndex.extended` raises like the numpy
  artifact.
- **Coverage** — the dependency-free gate keeps `fail_under = 100` with the
  adapter modules `omit`ted; a dedicated `adapter-tests` CI job gates the adapter
  modules at 100% (LangChain via a fake LLM — no API key). The D9 "revisit"
  (D-4.7) is closed.
- **Examples** — `01` / `02` are no longer design mocks: they run with the heavy
  extras + an `OPENAI_API_KEY` (type-checked + compiled in CI, not executed there).

Full record: [`docs/ADRs/0005-heavy-adapters.md`](../ADRs/0005-heavy-adapters.md).
