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
`tests/test_a05_contract.py::test_d1_labeledpairs_is_ordered_and_not_symmetrized`.

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
