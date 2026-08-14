# Resolvi reference-architecture conformance & deltas (A0.5 addendum)

> Part of the [denselinkage development docs](./README.md). This is the external
> yardstick analysis. The forks it surfaced — **D4** and **D5** — are recorded
> here; the other decisions are in [decisions.md](./decisions.md).

External yardstick: Olar, *"Resolvi: A Reference Architecture for Extensible,
Scalable and Interoperable Entity Resolution"* (arXiv:2503.08087v3) — a
**Type-3** reference architecture synthesized from nine ER systems. A Type-3
reference describes a *design space*, not a feature checklist. denselinkage is
a **library** that implements the *engine subset* of Resolvi's
*system/platform* model; its platform components are out-of-scope-**by-design**
(declines below), not gaps. This section records the conformance verdict and
the deltas it implies; D5/`clustering_metrics`/`Filter` join D1–D3 as
ratify-before-freeze items.

## Conformance — strong (evidence-cited)

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

## D4 — matching-optional ER variant

**Fork.** Resolvi: matching is optional ("at least one of matching/
clustering"); d-blink skips matching entirely. `DenseLinker.matcher` is
required.
**Ruling (ratified — ADR-0003): conscious decline — not a gap.** The package identity
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

## D5 — `Clusterer` port

**Fork.** Clustering is "a separate, pluggable function"
(`cluster.connected_components`) but **not a typed port**, so an alternative
algorithm cannot be injected through the contract. Resolvi: clustering is a
swappable **Strategy** (agglomerative is incremental).
**Ruling (recommended, ratify pre-freeze).** Add a `Clusterer` Protocol to
`denselinkage.core.ports` (`cluster(result: LinkageResult) -> ClusteringResult`); a
`ConnectedComponentsClusterer` reference adapter (in `cluster.py`) declares the
port and joins the adapter-declares-its-port contract test, while the existing
`connected_components` function is kept as its prelude convenience wrapper
(extend, never modify). Bodies stay `...`. Contract-shape: adding the port
post-freeze is breaking. Recorded in: `core/ports.py` + `cluster.py` + here +
`tests/test_contract.py`, whose port and adapter subjects are derived from source
rather than listed (issue #31).

## A0.5 contract-shape additions (signatures / `...` only)

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
  noted in [examples.md](./examples.md)). Declare in A0.5; reference adapter +
  algorithm Phase C.

## Conscious declines (recorded, not built)

| Declined | Resolvi element | Rationale |
|---|---|---|
| Entity alignment (knowledge graphs) | graph-specialized variant | out of tabular-RL scope |
| NERD / unstructured-text extraction | text-specialized variant | the `_reader`/`Serializer` seam *could* host a future NER trait; not core scope |
| Presentation layer, plugin/admin, web services, role model | Resolvi *platform* components | denselinkage is a library; ports **are** the extension mechanism |
| Distributed execution (Spark/Flink analog) | scalability pillar | single-node by design (batched encode + matcher `max_concurrency`); document the envelope |
| Eager merged "golden record" (merge/purge) | merge/purge variant | **declining is conformance** — Resolvi warns merged profiles harm provenance/auditability; the provenant, opt-in alternative is Phase-C survivorship |
| D4 matching-optional | optional matching | see D4 above (recommended decline; ratify only on maintainer override) |

## Roadmap deltas (Resolvi-justified; extend existing Phase B/C, do not replace)

- **Phase B +=** Reference Store: `LinkageIndex.save()/load()` + a stable
  serialized result/index format (Resolvi: Reference Store, Memento; also
  serves the interoperability pillar).
- **Phase C +=** incremental indexing via `SearchableIndex.extended(vectors,
  ids) -> SearchableIndex` — returns a NEW artifact (immutable-artifact safe;
  the designed escape hatch, see
  [`docs/ADRs/0001-statefull-components-as-artifacts.md`](../ADRs/0001-statefull-components-as-artifacts.md));
  continuous input, needs the Phase-B store; agglomerative/incremental
  `Clusterer` adapters on the D5 port; composite/trait-family `Serializer`
  enrichment of the Extraction Engine.

These deltas do not alter any already-frozen surface; D4 and the platform
items are recorded declines; D5, `clustering_metrics`, and `Filter` are
contract-shape and must land (signatures + `...` + shape tests) **before** the
re-freeze, exactly like D1–D3.
