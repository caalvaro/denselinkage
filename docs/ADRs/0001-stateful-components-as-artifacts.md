# ADR-0001: Modeling stateful components as artifacts in an immutable linkage API

**Status:** Accepted; implemented at structure stage (signatures + `...` bodies). Both sub-decisions confirmed.
**Date:** 2026-06-01
**Deciders:** Alvaro (author); thesis advisor sign-off

## Context

`DenseLinker` is an immutable, reusable configuration object
(`@dataclass(frozen=True, slots=True, kw_only=True)`). Its stated contract is
"pure config — no data, no learning": `link(a, b) == index(a).query(b)`. To
perform linkage, however, the system must produce **per-dataset state** — the
vector index built over the left/reference records.

Auditing the eight ports against one question — *does it accumulate per-dataset
state?* — reveals that the trouble is concentrated in a single place:

| Port | Shape | Per-dataset state? | Kind |
|---|---|---|---|
| `Serializer` | `serialize(record) -> str` | no | pure strategy |
| `Embedder` | `encode(texts) -> vectors` | no (weights are fixed params) | pure strategy |
| `Matcher` | `match(pairs) -> decisions` | no | pure strategy |
| `Filter` | `filter(pairs) -> pairs` | no | pure strategy |
| `Clusterer` | `cluster(result) -> clustering` | no | pure strategy |
| `Trainer` | `train(pairs, base) -> component` | no — *"a factory, not a fit"* | pure strategy |
| **`VectorIndex`** | **`add(); search()`** | **YES — `add()` accumulates** | **stateful** |
| `Blocker` | `index(); query()` | yes — *only because it embeds a `VectorIndex`* | derived |

**Exactly one port is inherently stateful: `VectorIndex`.** `Blocker` inherits
statefulness only by holding one; the remaining six are pure strategies that are
safe to inject and reuse indefinitely. The current design injects all components
uniformly *as instances*, so a stateful index instance is placed inside
immutable config and mutated during `index()`. Every fix attempted so far —
`deepcopy`, a clone, an injected factory — is a patch over that one mismatch.

### The forces in tension

The difficulty is that **three desirable properties cannot all hold for an
injected stateful instance**:

| # | Property | Origin |
|---|---|---|
| 1 | The vector index is **stateful** (it holds the indexed dataset) | inherent to ANN search |
| 2 | The index backend is **configurable / injected** (open port) | extensibility — the project's thesis |
| 3 | Config is **immutable and reused** across datasets | the `DenseLinker` contract |

Injecting a stateful *instance* (1) into reusable immutable config (3) through an
open injection point (2) forces one of the three to break — or a runtime tax
(defensive copy / factory) to fake that none did.

### Constraints specific to this project

- **The architecture is the deliverable** (Master's thesis / intended paper);
  conceptual clarity and defensibility outrank line-count.
- The core stays **dependency-light** (numpy + pandas only); heavy backends are
  optional extras. This rules out `attrs`/`pydantic`-style machinery.
- **The examples are the spec** and are type-checked in CI; changes to the
  user-facing construction API are high-cost.
- We are **pre-freeze**: port-shape changes are cheap *now* and expensive later.

### Functional requirement (v1) — confirmed

**Index reuse / amortization is in scope for v1.** Building the index over a
reference dataset is the expensive half (embedding + index construction); a
common entity-resolution workload matches many incoming batches against one
stable reference table. The API must let callers pay that cost once and query
many times:

```python
idx = linker.index(reference)   # pay embedding + index cost ONCE
idx.query(batch_1)              # cheap
idx.query(batch_2)
```

This requirement **rejects Option D** (below) and is a featured capability of
the design.

## Decision

Adopt **Option C — uniform spec→artifact**. Every stateful seam becomes
`Spec.build(data) -> Artifact`, applied recursively. Callers inject **specs**
(stateless, reusable); state lives **only** in **artifacts** produced by a build
step. Concretely:

```
VectorIndex.build(vectors, ids) ─▶ SearchableIndex   (artifact: search only)
Blocker.build(records)          ─▶ BlockingIndex      (artifact: owns a SearchableIndex)
DenseLinker.index(source)       ─▶ LinkageIndex       (artifact: owns a BlockingIndex + matcher)
```

The same shape at three nested levels, with `DenseLinker -> LinkageIndex`
(already present) as the outermost instance.

Two sub-decisions are forced by the analysis and are **recommended pending
confirmation**:

- **Query-time parameters.** `top_k` and `similarity_threshold` are query-time
  knobs; make them overridable at query time
  (`BlockingIndex.query(records, *, top_k=None, similarity_threshold=None)`) so a threshold
  sweep does not trigger an index rebuild.
- **Incremental indexing is out of scope for v1.** Artifacts are immutable;
  updating a built index implies a rebuild. Document the limitation and leave a
  designed escape hatch (`SearchableIndex.extended(vectors, ids) -> SearchableIndex`)
  as future work.

## Options Considered

### Option A — Inject stateful instances + defensive copy (status quo)

| Dimension | Assessment |
|---|---|
| Complexity | Low (no new types) |
| Immutability honesty | **Low — `frozen` is theatrical; a copy hides a mutation** |
| Extensibility | High |
| Amortization | OK |
| Type safety (illegal states) | Low — `query`-before-`index` and `index`-twice are representable |

**Pros:** smallest diff; examples unchanged.
**Cons:** the `deepcopy` is load-bearing and fails silently if forgotten; it
copies a *populated* index; the immutability guarantee is fictional.

### Option B — Inject a factory (`Callable[[], VectorIndex]`)

| Dimension | Assessment |
|---|---|
| Complexity | Medium |
| Immutability honesty | High |
| Extensibility | High |
| Amortization | OK |
| Type safety | Low — same one-type role-mixing as A |

**Pros:** no copy; correct.
**Cons:** leaks an internal concern ("how to mint a fresh index") into the
**public constructor**; `vector_index_factory=...` reads as plumbing. Pays an
ergonomic tax to preserve a flawed representation.

### Option C — Uniform spec→artifact (`build() -> Artifact`) — **chosen**

| Dimension | Assessment |
|---|---|
| Complexity | Medium (+2 small artifact ports, one method each) |
| Immutability honesty | **High — specs hold no state; nothing to copy** |
| Extensibility | High (open ports preserved) |
| Amortization | **High — the artifact *is* the build-once / query-many unit** |
| Type safety | **High — search-before-build / build-twice are unrepresentable** |

**Pros:** one law, zero special cases; **construction API stays byte-identical**
(`vector_index=FaissFlatIndex()` still reads as "use FAISS") because the change
is to the *internal* port contract, not the user-facing wiring; reuses the
project's own blessed idiom (`Trainer` = "a factory, not a fit") and established
prior art (Spark ML / scikit-learn **Estimator → Transformer**); unlocks
thread-safe concurrent queries, on-disk persistence of artifacts, and *trained*
ANN indexes (FAISS IVF/PQ require a train-then-add step that `add()/search()`
has no home for).
**Cons:** +2 small ports; a `build` step exists even for blockers that need no
preparation; immutable artifacts make incremental update a rebuild (see cracks).

### Option D — Fully functional (no exposed `index()` / artifacts) — **rejected**

| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Immutability honesty | High (no stateful object survives a call) |
| Extensibility | Medium — *still needs a spec/factory internally if the backend is configurable* |
| Amortization | **None — rebuilds the reference index on every `link()`** |
| Type safety | High |

**Pros:** simplest possible surface.
**Cons:** forfeits build-once / query-many, which is a **confirmed v1
requirement**; and it does not actually escape the core question — a
configurable stateful backend still needs Option C or B *inside* `link()`. It
only dissolves the problem by *also* dropping backend openness (property 2),
which contradicts the thesis.

## Trade-off Analysis

**Why C dominates.** Options A and B **pay a runtime/ergonomic tax to preserve a
flawed representation**; Option C **fixes the representation so there is no tax**.
Of the three forces, property 1 (statefulness) is inherent to ANN and cannot be
relaxed; property 2 (extensibility) is the thesis and must not be relaxed;
relaxing property 3 is Option D, which forfeits amortization. Option C relaxes
**none** of the three — it changes the *type of the injected thing* from a
stateful instance to a stateless spec, relocating statefulness out of config and
into a produced artifact. That is why it wins on every dimension except raw type
count.

**Why the split and amortization reinforce each other.** The amortization
feature (build once, query many) requires a *reusable, safe-to-share, prepared
index*. That object is exactly the **artifact** the spec→artifact split
introduces. In Option A the "prepared index" is a mutated field of frozen config
(unsafe to reuse — a second `index()` corrupts it); in Option C it is an
immutable `LinkageIndex` that can be queried repeatedly and shared across
threads. The architectural decision and the headline feature are therefore two
views of the same construct — a strong, self-contained section for the paper:
*the discipline that makes immutability honest is the same discipline that makes
amortized index reuse safe.*

**The two cracks (decide now — they move method signatures).**

1. **Index-time vs. query-time parameters.** The split *forces* a question the
   mutable design lets one dodge: `top_k` and `similarity_threshold` are
   query-time knobs (changing them needs no re-embedding), yet they currently
   live on the `Blocker` spec and would bake into the artifact — making a
   `ThresholdSweep` trigger a full rebuild per threshold. Recommendation:
   `BlockingIndex.query(records, *, top_k=None, similarity_threshold=None)` overriding spec
   defaults. Sweep without rebuild, at the cost of a little signature surface.

2. **Incremental / streaming indexing.** An immutable `SearchableIndex` means
   "add 5k new records to a 10M index" implies a rebuild, and a mutable
   C++-backed FAISS index cannot honor immutability cheaply. Recommendation:
   scope incremental *out* of v1 (batch linkage), name the limitation, and leave
   a designed escape hatch (`SearchableIndex.extended(...) -> SearchableIndex`).
   Defensible for a v1 batch ER library; pretending otherwise would be the
   dishonest move.

## Consequences

**Easier**
- Immutability becomes real (and `slots=True`'s "no ad-hoc attribute" guarantee
  stops being theater).
- Concurrent queries against one prepared index are trivially safe.
- Persisting / loading a prebuilt index has a natural unit (the artifact).
- Trained ANN backends (IVF/PQ) fit naturally (`build` = train-then-add).
- The `deepcopy` and the factory both *delete themselves*.

**Harder**
- Two more small types to introduce and teach.
- A `build` ceremony exists even for blockers that need no preparation.
- Incremental updates require a rebuild in v1.

**To revisit**
- The incremental escape hatch (`extended`) if streaming becomes a requirement.
- A persistence port (`save` / `load`) on artifacts.
- Whether `Blocker` remains a distinct port or folds into the linker once it is
  stateless.

**Thesis risk to manage.** A reviewer may misread "+2 ports" as over-engineering.
The narrative must lead with **the law** ("one port is stateful; stateful things
are artifacts, not configuration") and present the ports as its *consequence*,
with the deleted `deepcopy`/factory as evidence the law pays for itself. Framed
as "more classes," it looks like gold-plating; framed as "one principle, applied
without exception," it is a contribution.

## Action Items

1. [x] `core/ports.py`: reshape `VectorIndex -> build() -> SearchableIndex`;
   `Blocker -> build() -> BlockingIndex`; export both artifact ports from
   `core/__init__.py` (`__all__`).
2. [x] Crack #1: query-time `top_k` / `similarity_threshold` overrides on
   `BlockingIndex.query` (named to match the `DenseBlocker` constructor knob).
3. [x] Crack #2: recorded "no incremental indexing in v1" and added the
   `SearchableIndex.extended` escape-hatch stub (signature only).
4. [x] `linker.py`: `index()` delegates to `blocker.build()` (no `deepcopy`);
   `LinkageIndex` holds the `BlockingIndex`; preserve `link == index().query()`.
5. [x] Updated structure-stage tests (`tests/test_contract.py`) to add the two
   artifact ports to the runtime-checkable-protocol and adapter-declares-port
   lists.
6. [x] Examples 01/03: **no construction change**; confirmed they still
   type-check (and `02`, also unaffected).
7. [x] Folded the revised seam into the `with_defaults()` / `00_quickstart`
   implementation plan (`docs/development/implementation_plan.md`).

## References / prior art

- **Estimator → Transformer** separation: scikit-learn (`fit` produces fitted
  state) and Apache Spark ML (`Estimator.fit() -> Transformer`) — an unfitted,
  reusable configuration object produces a separate fitted artifact.
- **Ports & Adapters (Hexagonal Architecture)**, A. Cockburn — structural ports
  with swappable adapters; the basis for keeping backends injected and open.
- **"Make illegal states unrepresentable"** (Y. Minsky) — the type-safety
  rationale for splitting spec from artifact so `search`-before-`build` cannot
  be expressed.
- Internal precedent: `denselinkage.core.ports.Trainer` — *"train is a factory,
  not a fit: it returns a NEW component and mutates neither self nor base."*

## Implementation note (landed 2026-06-02, structure stage)

Implemented exactly as specified; method bodies remain `...` per the structure
stage. Files touched:

- `core/ports.py` — four ports: `VectorIndex` / `Blocker` (specs) +
  `SearchableIndex` / `BlockingIndex` (artifacts), all `@runtime_checkable`.
- `core/__init__.py` — `SearchableIndex` / `BlockingIndex` added to imports and
  `__all__`.
- `indexing/__init__.py` — `NumpyFlatIndex` / `FaissFlatIndex` reshaped to specs
  with `build`; `NumpySearchableIndex` / `FaissSearchableIndex` artifacts added.
- `blocking/__init__.py` — `DenseBlocker` reshaped to a spec with `build`;
  `DenseBlockingIndex` artifact added (the constructor signature is unchanged).
- `linker.py` — `LinkageIndex.__init__(*, blocking_index, matcher)`; `index()`
  docstring states delegation + composition (no `deepcopy`).
- `tests/test_contract.py` — the two artifact ports joined the
  runtime-checkable and adapter-declares-port fitness functions.

The construction API is byte-identical: `examples/01`–`03` are unchanged and
still type-check. The query-time override is named **`similarity_threshold`**
(not the shorthand `threshold` used in the Decision section above) to match the
`DenseBlocker` constructor knob it overrides.

Gates green: `ruff check`, `ruff format --check`, `mypy --strict`
(`src` + `examples`), `pytest` (56 passed), and
the import dependency-cut (no heavy backend pulled on `import denselinkage`).

## Freeze semantics & decision-log status (D6)

This ADR is decision **D6** in the in-tree decision log
([`docs/development/implementation_plan.md`](../development/implementation_plan.md),
`## Recorded decisions`) and is classified **contract-shape (CS)** — the most
severe breaking-if-late row — in the triage analysis
([`docs/development/paper_evaluation_data.md`](../development/paper_evaluation_data.md) §3).

**On the status line "implemented at structure stage."** This means D6 landed in
**A0.5**, the *pre-freeze* hardening window (the 0.x stage), not after the
freeze. A wholesale reshape of the index/blocker ports is legitimate here
precisely because the contract is **not yet frozen**: the project's
`extend-never-modify` rule governs the *post-freeze* (post-1.0) surface, and D6
is a *pre-freeze* change, in the same class as D1–D5, `clustering_metrics`, and
the `Filter` port. See `## Freeze semantics & the A0.5 gate` in the
implementation plan.

**Method note.** D6 was surfaced not by reference-architecture conformance, by
strict typing, or by example compilation — all of which passed over the original
mutable `add()/search()` ports — but by *attempting the reference
implementation* of `with_defaults()`. It is the evidence motivating the freeze
gate's fourth oracle (reference-implementation attemptability); see the paper's
freeze-gate-refinement argument under `docs/paper/`.
