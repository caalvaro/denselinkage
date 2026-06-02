# denselinkage — Implementation Plan & Frozen Contract

This is the in-tree, diffable record of the contract, the phase boundaries,
and every recorded design decision. It is authoritative; chat threads are not.

## Project stage

**Structure stage.** `src/denselinkage/` defines public types, ports
(`typing.Protocol`s) and signatures; bodies are `...` placeholders. The
`examples/` are the design spec for the intended API and are type-checked,
lint-checked and compiled in CI against the real core. Implementation lands
incrementally against the frozen contract below (do not change a frozen
surface — *extend, never modify*: add an optional field with a default, a
sibling type, or a new classmethod).

## Phases & exit criteria

- **A0** *(done)* — typed core skeleton: models, ports, results, linker,
  metrics, serializers, reference adapters; dependency-cut CI; `[train]`
  namespace reserved.
- **A0.5** *(this pass — contract hardening before freeze)* — close the gaps
  that would become breaking post-freeze: D1/D2/D3, the hard-failure
  exception taxonomy, the named Source→Record seam, `BlockingMetrics`
  constructor shape, kwarg-`gold` consistency, examples wired into CI
  (lint+format+type+compile), one Python-version story, conventions aligned.
  **Exit:** every DoD check green and pasted verbatim; contract re-frozen.
- **A1+** — fill method bodies behind the frozen contract. **Gated:** do not
  start A1 until A0.5 is complete and the contract is re-declared frozen.
- **Phase B** *(v1, dependency-light)* — `mine_hard_negatives`,
  `tune_threshold`, `adjusted_metrics`, `LabeledPairs.split`,
  `LinkageResult.from_candidate_frame` (the ergonomic frame→pairs constructor
  that makes `match_pairs` end-to-end usable — see D2).
- **Phase C** *(v2, behind `[train]`)* — `EmbedderTrainer` /
  `CrossEncoderTrainer` (implement the `Trainer` protocol) + few-shot
  selection. LLM fine-tuning is out of scope. The first/second-pass
  hard-negative *simulation* example (former `examples/04`) is a Phase-C
  deliverable — restored then as a **real** (non-faked) example once mining +
  trainers exist (see "examples/04" below).

## Freeze semantics & the A0.5 gate (what "frozen" means)

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
   and the declines).
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
`directed=` metric parameter (see the D1 amendment above). A0.5 therefore carries
an explicit exit criterion — *the `with_defaults()` / `00_quickstart` vertical
slice is shown implementable against the frozen stubs* — now **met**: the slice
is implemented on the dependency-free stack and green
(`tests/test_quickstart_end_to_end.py`, `00_quickstart.py` runs at P/R/F1 = 1.0).
The paper's C-B contribution is the completed four-oracle gate, with D6 and the
D1 amendment as its two worked motivations.

## Reference component map (resolved deltas)

- `to_frame()` schema is fixed and **`match` is non-null `bool`** (not
  `bool | None`): one row per *decided* pair; pairs the matcher could not
  decide are in `LinkageResult.errors`, not rows. Columns: `left_id`,
  `right_id`, `similarity`, `match`, `confidence` (`float|None`), `reason`
  (`str|None`).
- The dependency-free reference vector index is **`NumpyFlatIndex`** in
  `denselinkage.indexing` (not `FlatIndex`; indexes have their own
  module/port parallel to embedders).
- A3 reference adapters include `WholeRowSerializer` and
  `default_serializer()` (the `Source(serializer=None)` resolution target),
  alongside `HashedNGramEmbedder`, `NumpyFlatIndex`, `ThresholdMatcher`.
- Soft vs hard failure split: soft per-pair = `MatchError` in
  `LinkageResult.errors` (never exceptions); hard = the exception taxonomy
  below.

## Recorded decisions

### D1 — `LabeledPairs` pair identity
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

### D2 — `match_pairs` input contract
**Ruling (recommended default, no override).** `CandidatePair.similarity_score`
is now `float | None = None`. A dense `Blocker` always sets it; externally
/ rule-blocked pairs passed to `DenseLinker.match_pairs` use `None`. Done
pre-freeze precisely because `CandidatePair` is `frozen` (can't tighten
later). The ergonomic `LinkageResult.from_candidate_frame` constructor is a
**Phase-B** addition, so `match_pairs` is contract-complete now but only
end-to-end usable in B. Recorded in: `CandidatePair` docstring +
`DenseLinker.match_pairs` docstring.

### D3 — `@runtime_checkable` on ports
**Ruling (recommended default, no override): keep it.** Retained ONLY so the
structure-stage contract test can assert `_is_runtime_protocol`; no runtime
`isinstance` dispatch against ports exists or is intended (first-party
adapters subclass their port explicitly; mypy completeness-checks them).
Recorded in: rationale comment in `src/denselinkage/core/ports.py` + here.

### D6 — Stateless-spec / immutable-artifact ports (spec→artifact)
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
reference-implementation attemptability (see `## Freeze semantics & the A0.5
gate` below, and the paper's freeze-gate refinement).

Recorded in: `core/ports.py` (four ports + rationale), `core/__init__.py`
(`__all__`), `indexing/__init__.py`, `blocking/__init__.py`, `linker.py`,
`tests/test_contract.py` (the two artifact ports joined the
adapter-declares-its-port and runtime-checkable fitness functions). Full record:
[`docs/adr/0001-stateful-components-as-artifacts.md`](../adr/0001-stateful-components-as-artifacts.md)
and the `## ADR-0001` summary below.

## Hard-failure exception taxonomy (2.1)

`denselinkage.core.errors`, dependency-free, all rooted at
`DenseLinkageError`, all exported from `denselinkage.core`:
`UnknownIdColumn`, `EmptySource`, `DuplicateRecordId`, `DimensionMismatch`,
`InvalidTopK`. Verb→exception mapping is documented on
`DenseLinker.index/link/dedupe/match_pairs` and `LinkageIndex.query`
(plus the existing `ValueError` when `blocker is None`). Bodies land in A1.

## Source → Record seam (2.3)

Named: internal `denselinkage._reader.RecordReader` (underscore-private; not
a public port, not in the prelude). Documented responsibility: resolve
`Source.serializer is None` → `denselinkage.serialize.default_serializer`
(`WholeRowSerializer`), apply the serializer (incl. `column_mapping`),
validate the frame raising the taxonomy above. Referenced from the `Source`
and `linker` docstrings so the contract has no unspecified boundary.

## examples/04

`examples/04_second_pass_blocking.py` was deleted (orphan `.pyc` removed in
this pass). **Decision:** *removed from Phase-A exit criteria.* A
hard-negative second-pass example that fakes blockers in v1 would ship a
mock of the package's own motivating method against a "frozen" contract;
instead it is a **Phase-C** deliverable, restored as a real (non-faked)
example once `mine_hard_negatives` + trainers exist. The plan no longer
treats it as a Phase-A labelled-simulation deliverable.

## Part 5 polish — outcomes (all done, none deferred)

- **RecordId** threaded through identity-bearing types (`LabeledPairs.pairs`,
  `VectorIndex.build` / `SearchableIndex.search` ids+returns across the
  `Numpy*` / `Faiss*` adapters, `Clustering.labels`). `Record.id` already used
  it. Alias kept (not dropped).
- **`from __future__ import annotations`** removed from `_optional.py` — the
  package convention is uniformly quoted forward refs / `TYPE_CHECKING`, no
  `__future__` import.
- **`LinkageIndex.__init__`** aligned to `kw_only` for consistency with
  `DenseLinker`. *Brief-vs-code note:* the brief said `LinkageIndex` is
  "absent from the prelude" — it is in fact already exported in
  `denselinkage.__all__`. Decision: **keep it in the prelude** (it is the
  documented power-path return type referenced by `examples/01`).
- **Leaked decision codes** (`L1/H2/M1/H3/L4/M4/M3/M5`) stripped from all
  shipped example docstrings.

## Resolvi reference-architecture conformance & deltas (A0.5 addendum)

External yardstick: Olar, *"Resolvi: A Reference Architecture for Extensible,
Scalable and Interoperable Entity Resolution"* (arXiv:2503.08087v3) — a
**Type-3** reference architecture synthesized from nine ER systems. A Type-3
reference describes a *design space*, not a feature checklist. denselinkage is
a **library** that implements the *engine subset* of Resolvi's
*system/platform* model; its platform components are out-of-scope-**by-design**
(declines below), not gaps. This section records the conformance verdict and
the deltas it implies; D5/`clustering_metrics`/`Filter` join D1–D3 as
ratify-before-freeze items.

### Conformance — strong (evidence-cited)

- **Extensibility pillar:** `core.ports` Protocols + constructor DI;
  first-party adapters subclass their port (mypy-completeness-checked);
  `Trainer` is an *orthogonal* port, not a method on `Embedder`/`Matcher`.
- **Design-time vs runtime separation:** `DenseLinker` (frozen config) vs
  `LinkageIndex` (prepared state) is exactly Resolvi's split.
- **Provenance-preserving deconstructible output** — Resolvi's *preferred*
  form for regulated domains: `LinkageResult.errors` is a separate channel
  from `decisions`; `to_frame` is decided-pairs-only; `n_errors` is excluded
  from tp/fp/fn; there is **no eager merge**.
- **Extraction Engine + Trait:** `denselinkage._reader.RecordReader` *is*
  Resolvi's Extraction Engine (already named, §2.3); `Serializer` is its
  Extraction Trait (Adapter). Composite/trait-family enrichment = roadmap.
- **RL-centric taxonomy:** `link`/`dedupe`/`match_pairs` cover record linkage,
  deduplication, and entity-matching (RL − blocking); optional `blocker` +
  `match_pairs` already realize Resolvi's "matching-without-blocking" variant.
- **Configuration / human-in-the-loop pillar (partial):** `Trainer` port +
  `TrainingPairs` + Phase-B `mine_hard_negatives`/`tune_threshold`/
  `ThresholdSweep`/`AdjustedMetrics` are design-time configuration and
  active-learning material. Remainder is Phase B/C.

### D4 — matching-optional ER variant

**Fork.** Resolvi: matching is optional ("at least one of matching/
clustering"); d-blink skips matching entirely. `DenseLinker.matcher` is
required.
**Ruling (recommended): conscious decline — not a gap.** The package identity
is dense blocking **plus matching**; the degenerate "no real matcher" case is
already served by `ThresholdMatcher` (dependency-free, gates on the carried
similarity) without weakening the `matcher`-always-present invariant that keeps
the `LinkageResult`/metrics path uniform. Resolvi's "at least one of matching/
clustering" is satisfied — matching is always present. Recorded in the Declines
ledger.
**Alternative (only if the maintainer wants true d-blink-style matching-skip):**
ratify `matcher: Matcher | None = None` **now** — `DenseLinker` is `frozen`, so
it cannot be relaxed post-freeze (contract-shape, same reasoning as D2's
blocker-optional). Decision must be explicit either way.

### D5 — `Clusterer` port

**Fork.** Clustering is "a separate, pluggable function"
(`cluster.connected_components`) but **not a typed port**, so an alternative
algorithm cannot be injected through the contract. Resolvi: clustering is a
swappable **Strategy** (agglomerative is incremental).
**Ruling (recommended, ratify pre-freeze).** Add a `Clusterer` Protocol to
`denselinkage.core.ports` (`cluster(result: LinkageResult) -> Clustering`); a
`ConnectedComponentsClusterer` reference adapter (in `cluster.py`) declares the
port and joins the adapter-declares-its-port contract test, while the existing
`connected_components` function is kept as its prelude convenience wrapper
(extend, never modify). Bodies stay `...`. Contract-shape: adding the port
post-freeze is breaking. Recorded in: `core/ports.py` + `cluster.py` + here +
`tests/test_contract.py` / `tests/test_a05_contract.py`.

### A0.5 contract-shape additions (signatures / `...` only)

- **`clustering_metrics` + `ClusteringMetrics`** (`metrics.py` / `results.py`):
  B³ precision/recall/F1 (+ cluster count), `*, gold` kwarg like the others.
  Resolvi *explicitly* recommends clustering-quality metrics and notes they are
  rare among ER systems; consistent with the existing typed-metrics pattern
  (`AdjustedMetrics`, `ThresholdSweep`). Pairs with D5. Frozen result +
  shape-only contract test; body A1.
- **`Filter` port** (`core.ports`): a second comparison-space reduction
  distinct from blocking — Resolvi's *filtering* stage. Filtering is currently
  a `DenseBlocker` constructor knob (`similarity_threshold`/`top_k`), not
  injectable. `Filter.filter(pairs: Sequence[CandidatePair]) ->
  list[CandidatePair]` makes the planned Phase-C second-pass-blocking example a
  **real pluggable component** rather than a faked one (resolves the tension
  noted in the "examples/04" section). Declare in A0.5; reference adapter +
  algorithm Phase C.

### Conscious declines (recorded, not built)

| Declined | Resolvi element | Rationale |
|---|---|---|
| Entity alignment (knowledge graphs) | graph-specialized variant | out of tabular-RL scope |
| NERD / unstructured-text extraction | text-specialized variant | the `_reader`/`Serializer` seam *could* host a future NER trait; not core scope |
| Presentation layer, plugin/admin, web services, role model | Resolvi *platform* components | denselinkage is a library; ports **are** the extension mechanism |
| Distributed execution (Spark/Flink analog) | scalability pillar | single-node by design (batched encode + matcher `max_concurrency`); document the envelope |
| Eager merged "golden record" (merge/purge) | merge/purge variant | **declining is conformance** — Resolvi warns merged profiles harm provenance/auditability; the provenant, opt-in alternative is Phase-C survivorship |
| D4 matching-optional | optional matching | see D4 (recommended decline; ratify only on maintainer override) |

### Roadmap deltas (Resolvi-justified; extend existing Phase B/C, do not replace)

- **Phase B +=** Reference Store: `LinkageIndex.save()/load()` + a stable
  serialized result/index format (Resolvi: Reference Store, Memento; also
  serves the interoperability pillar).
- **Phase C +=** incremental indexing via `SearchableIndex.extended(vectors,
  ids) -> SearchableIndex` — returns a NEW artifact (immutable-artifact safe;
  the designed escape hatch, see ADR-0001); continuous input, needs the Phase-B
  store; agglomerative/incremental `Clusterer` adapters on the D5 port;
  composite/trait-family `Serializer`
  enrichment of the Extraction Engine.

These deltas do not alter any already-frozen surface; D4 and the platform
items are recorded declines; D5, `clustering_metrics`, and `Filter` are
contract-shape and must land (signatures + `...` + shape tests) **before** the
re-freeze, exactly like D1–D3.

## Examples-as-spec — review findings & roadmap

A record-linkage review of `examples/` (methodology, not contract fidelity —
the contract *usage* is clean) drove the following. The cheap, spec-safe fixes
are done; the rest are A1/Phase-B example-stage items, one with an affordance
decision attached.

**Done (spec-safe, this pass):**
- `00` quickstart gold made lexically recoverable (`Google LLC`/`Google`, not
  `Google`→`Alphabet`) + a note that the default stack is lexical and semantic
  renames need `SentenceTransformerEmbedder` (`01`). This also retires the
  contestable parent/subsidiary positive from the low-floor example.
- `02` dedup now shows the honest tail: B³ via `clustering_metrics` against the
  same `LabeledPairs` gold; a transitivity-trap warning (connected components
  merges A~B, B~C into one cluster even if A and C never matched → runaway
  mega-clusters); and a D1 note (dedup gold is order-insensitive). Alphabet(5)/
  Google(6) are deliberately excluded from gold as a policy call, not asserted.
- `02` surfaces `result.errors` (per-pair `MatchError` triage) — the LLM-path
  realism the contract already models.
- `01` flags its `Google/Alphabet` gold as a deliberate semantic + boundary
  call, plus a `# TODO(Phase-B): ThresholdSweep.best_f1()` threshold breadcrumb.
- `03.encode` notes it is not vectorized and ignores `batch_size`/`show_progress`
  (clarity over speed; not for a hot path).

**A1 / Phase-B (example-stage):**
- **PC@k blocking-recall example (highest value).** The blocker is the recall
  ceiling; `pair_completeness_at_k` / `BlockingMetrics.pc_at(k)` exist now.
  *Affordance gap:* there is no ergonomic public path to candidate pairs —
  `link()` hides them and `Source→Record` materialization is the private
  `_reader.RecordReader`. **Decision (ratify):** for the A1 example, write the
  lower-level path (`DenseBlocker.build()` → `BlockingIndex.query()` on
  hand-built `Record`s →
  `pair_completeness_at_k`) — works today, slightly verbose; and in **Phase B**
  add `LinkageResult.from_candidate_frame` (already planned) as the ergonomic
  DataFrame→`CandidatePair` constructor. A `DenseLinker.block(source) ->
  list[CandidatePair]` convenience is the larger option and is tied to the
  matching-optional thread (D4) — defer unless D4 is ratified.
- **`index()`/`query()` reuse example.** Show "embed/index the master once,
  query many incoming batches" and the documented `link(a,b) ==
  index(a).query(b)` equivalence. Cheap; add as `examples/05_reuse_index.py` or
  a section in `01` (the `04` slot is reserved for the Phase-C second-pass /
  hard-negative example).

## ADR-0001 — Stateful components modeled as artifacts (spec→artifact)

Full record: [`docs/adr/0001-stateful-components-as-artifacts.md`](../adr/0001-stateful-components-as-artifacts.md).
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

## Frozen contract (re-declared after A0.5 verification)

Once the A0.5 DoD checks are green (ruff incl. format, mypy `src+examples`,
compileall examples, pytest, import-isolation), the public surface above is
frozen and A1 may begin.
