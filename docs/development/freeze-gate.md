# Freeze semantics & the A0.5 gate (what "frozen" means)

> Part of the [denselinkage development docs](./README.md).

"Freeze" in this plan is a specific event, not a vague aspiration: the **A0.5
re-freeze gate** — the point at which the public surface is locked and body
implementation (A1+) may begin. It is the project's **v1 / 1.0 contract
commitment**. The mapping to the usual version vocabulary:

- **Structure stage (A0, A0.5) = the 0.x hardening window.** The contract is
  deliberately malleable; *contract-shape* changes (a new port, a widened frozen
  field, a reshaped calling convention) are **cheap and expected** here.
- **The A0.5 gate = the freeze = the 1.0 commitment.** After it, the surface is
  fixed and evolution is constrained to **extend-never-modify**.

Therefore **extend-never-modify is a *post*-freeze rule.** D1–D6,
`clustering_metrics`, and the `Filter` port are all contract-shape and all land
*in A0.5* — that is the gate doing its job (absorbing breaking-if-late changes
before the lock), not a violation of it. The decision-log triage
(`paper_evaluation_data.md` §3) measures exactly this: the bulk of
architecture-bearing decisions are contract-shape, which is *why* the freeze
must happen after they are resolved, not before.

**The freeze gate is an ordered battery of oracles**, escalating from cheap and
broad to expensive and decisive:

1. **Reference-architecture conformance** (the Resolvi walk) — drives *which*
   variation points must exist (surfaced D4, D5, `Filter`, `clustering_metrics`,
   and the declines). See [resolvi-conformance.md](./resolvi-conformance.md).
2. **Strict static typing** (`mypy --strict` on `src` + `examples`) — surface
   self-consistency.
3. **Examples-as-specification** (`compileall` + lint + type-check the
   `examples/` against the real core in CI) — documented usage cannot drift from
   the contract.
4. **Reference-implementation attemptability** — a dependency-free vertical
   slice (`with_defaults()` → `link()`) must be *implementable* against the
   stubs with no contract-shape surprise.

Oracles 1–3 are necessary but **not sufficient**: D6 (ADR-0001) passed all three
and was caught only by oracle 4. Implementing the slice surfaced a **second**
oracle-4 catch — D1's verb-dependent comparison was unimplementable as recorded
(the verb is not recoverable from the result), resolved by an explicit
`directed=` metric parameter (see the D1 amendment in
[decisions.md](./decisions.md)). A0.5 therefore carries an explicit exit
criterion — *the `with_defaults()` / `00_quickstart` vertical slice is shown
implementable against the frozen stubs* — now **met**: the slice is implemented
on the dependency-free stack and green (`tests/test_quickstart_end_to_end.py`,
`00_quickstart.py` runs at P/R/F1 = 1.0). The paper's C-B contribution is the
completed four-oracle gate, with D6 and the D1 amendment as its two worked
motivations.

## Frozen contract (re-declared after A0.5 verification)

**Status: FROZEN — A0.5 gate passed 2026-06-03.** The public surface is locked;
evolution is now constrained to **extend-never-modify**. A1 (the dependency-free
beta, `1.0.0b1`) may begin.

### Gate evidence
All four oracles plus the A0.5 DoD checks are green:

- ruff check + `ruff format --check` (src / tests / examples) ✓
- `mypy --strict` (src + examples, 48 files) ✓
- `compileall examples` ✓
- pytest (`not adapter and not slow`) ✓
- import-isolation / dependency-cut (`import denselinkage` pulls in no heavy
  backend) ✓
- oracle 4: the `with_defaults()` → `00_quickstart` vertical slice is implemented
  and runs at P/R/F1 = 1.0.

### Signatures-and-field-types pass (ADR-0003 freeze checklist)
An adversarial read of every port signature, every frozen field type, and the
public calling conventions found **no modify-if-late changes required**. Items
confirmed *intentional* (not defects):

- `clustering_metrics` takes no `directed=` (clusters are inherently unordered),
  while `linkage_metrics` / `blocking_metrics` / `pair_completeness_at_k` do.
- `DenseLinker` carries only `blocker` + `matcher`; filtering is the separate
  `Filter` port and clustering is the post-hoc `connected_components` step, not
  linker fields (wiring them in later is additive).
- `Embedder.model_id` / `embedding_dim` retained (ADR-0003).

### The frozen surface
- **Ports** (`core.ports`): `Serializer`, `Embedder`, `VectorIndex`,
  `SearchableIndex`, `Blocker`, `BlockingIndex`, `Filter`, `Matcher`,
  `Clusterer`, `Trainer`.
- **Models** (`core.models`): `Record`, `CandidatePair`, `MatchDecision`,
  `MatchError`, `Source`, `RecordId`.
- **Results** (`core.results`): `LabeledPairs`, `LinkageResult`,
  `ClusteringResult`, `TrainingPairs` (field types + the `to_frame` column
  schema).
- **Errors** (`core.errors`): the `DenseLinkageError` taxonomy + the
  `ValueError` (API-misuse) tier.
- **Orchestration**: `DenseLinker`
  (`with_defaults` / `index` / `link` / `dedupe` / `match_pairs`), `LinkageIndex`
  (`query`).
- **Metrics** public functions + report types: `linkage_metrics`,
  `blocking_metrics`, `pair_completeness_at_k`, `clustering_metrics`;
  `LinkageMetrics`, `BlockingMetrics`, `ClusteringMetrics`, `ThresholdSweep`,
  `AdjustedMetrics`.

### Open in the additive corridor (non-breaking, post-freeze)
- `LinkageResult.from_candidate_frame` (Phase-B ergonomic `DataFrame ->
  CandidatePair` constructor).
- An optional `LinkageMetrics.from_counts` classmethod, for constructor parity
  with `BlockingMetrics.from_pc_map` / `ClusteringMetrics.from_b3`.
- `DenseLinker` gaining optional `filter=` / `clusterer=` fields if/when wired.
- New ports, new optional fields (with defaults), new sibling types.

Freezing the **contract** is distinct from implementing the **bodies**: many
bodies are still `...` stubs; filling the dependency-free ones is A1 and does not
touch the frozen shape.
