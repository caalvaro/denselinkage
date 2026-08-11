# Changelog

All notable changes to denselinkage are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `DenseBlocker` accepts an optional `batch_size` and forwards it to its
  embedder while building an index.

## [1.0.0] — 2026-06-06

First **stable** release: the heavy adapters land, so the headline dense
*semantic* blocking + LLM matching stack now runs — completing the frozen 1.0
contract. **Additive** to that contract (extend-never-modify); the dependency-free
core is unchanged and still installs on numpy + pandas alone.

### Added
- Heavy adapters (A2) — the four deferred adapters are implemented (replacing
  their `NotImplementedError` guards), each behind its optional extra and
  lazy-imported so `import denselinkage` still pulls in no backend:
  - `SentenceTransformerEmbedder` (`[sentence-transformers]`) — semantic
    embeddings; `encode` returns L2-normalized float32 vectors, so inner product
    equals cosine and the reference stack's `similarity_threshold` keeps its
    meaning.
  - `FaissFlatIndex` / `FaissSearchableIndex` (`[faiss]`) — exact inner-product
    (`IndexFlatIP`) nearest-neighbour search; a drop-in for `NumpyFlatIndex`
    behind the `VectorIndex` port (a differential test pins them to the numpy
    neighbours). Incremental `extended` stays deferred (parity with the numpy
    artifact).
  - `LangChainMatcher` (`[langchain]`) — LLM matching via structured output; the
    prompt carries only the semantic question, and per-pair failures retry per
    `RetryPolicy`, becoming a position-aligned `MatchError` on exhaustion (never
    an exception into the batch).

### Changed
- Coverage policy (ADR-0005, the ADR-0004 D-4.7 revisit): the dependency-free
  surface keeps `fail_under = 100` (the adapter modules are `omit`ted); a
  dedicated `adapter-tests` CI job installs the extras and gates the adapter
  modules at 100% (LangChain tested with a fake LLM — no API key).

### Still deferred
- FAISS persistence in the Reference Store (the store still rejects non-numpy
  stacks); the dense-vs-lexical benchmark (a separate evaluation workstream); the
  v2 `[train]` trainers.

## [1.0.0b2] — 2026-06-05

Second beta — the Phase-B dependency-light features, all **additive** to the
frozen 1.0 contract (extend-never-modify); the dependency-free core remains
numpy + pandas only.

### Added
- Candidate-pair affordances (Phase B, batch B1) — all additive to the frozen
  1.0 contract:
  - `DenseLinker.block(left, right)` / `LinkageIndex.candidates(source)` expose
    the blocker's candidate pairs **without** matching — the ergonomic input to
    `blocking_metrics` / `pair_completeness_at_k`. Both accept `top_k` /
    `similarity_threshold` overrides so a pair-completeness sweep reuses one
    built index instead of rebuilding it.
  - `candidate_pairs_from_frame(frame, *, left, right, left_id, right_id,
    similarity=None)` builds the `match_pairs` input from a DataFrame of
    candidate id-pairs plus the two sources (record text is materialized via the
    sources' serializers). Exported from the package root.
- Metric producers (Phase B, batch B2) — additive:
  - `tune_threshold(candidates, *, gold, thresholds=None, directed=True)` sweeps
    the decision threshold and returns the full P/R/F1 curve as a
    `ThresholdSweep` (default grid = the candidates' distinct scores).
  - `adjusted_metrics(result, candidates, *, gold, k, directed=True)` decomposes
    end-to-end recall into matcher × blocker components as an `AdjustedMetrics`
    (the matcher's recall measured conditionally on what blocking surfaced).
- Gold & clustering utilities (Phase B, batch B3) — additive:
  - `LabeledPairs.split(*, test_size, seed=None)` partitions gold into
    `(train, test)` (pair-level, seeded) for tune-then-evaluate workflows.
  - `connected_components(result, *, all_record_ids=None)` seeds the clustering
    universe with the full id set, so unmatched records become singletons and
    `clustering_metrics` reports a complete B³ instead of one inflated by
    dropped records.
- Reference Store (Phase B, batch B4) — additive:
  - `LinkageIndex.save(path)` / `LinkageIndex.load(path, *, embedder, matcher)`
    persist and reload a built index (the dependency-free reference stack) as a
    portable `vectors.npy` + `meta.json` bundle, so a reference set is embedded
    once and reused. The stored embedder `model_id` / `embedding_dim` are
    validated against the re-supplied embedder (`IncompatibleStore` on mismatch),
    activating the reserved `Embedder.model_id` provenance surface. Adds read
    accessors to `NumpySearchableIndex` (`vectors`, `ids`) and
    `DenseBlockingIndex` (`searchable`, `embedder`, `records`, `top_k`,
    `similarity_threshold`), plus the `IncompatibleStore` error.
- Hard-negative mining (Phase B, batch B5) — additive:
  - `denselinkage.mining.mine_hard_negatives(candidates, *, gold, n=None,
    directed=True)` returns the highest-similarity non-matches — contrastive
    material for the v2 trainers.

## [1.0.0b1] — unreleased

First beta of the frozen 1.0 contract. The **dependency-free core** is
implemented and runs on numpy + pandas; the heavy extras are experimental.

### Added
- Orchestration verbs on `DenseLinker`: `link`, `dedupe`, `match_pairs`,
  `index`, and `with_defaults` (the lexical reference stack).
- Connected-components clustering: `connected_components` /
  `ConnectedComponentsClusterer`, and `ClusteringResult` (`n_clusters`,
  `to_frame`).
- Evaluation metrics: `linkage_metrics`, `blocking_metrics` /
  `pair_completeness_at_k`, `clustering_metrics` (B³), with the report types
  `LinkageMetrics`, `BlockingMetrics`, `ClusteringMetrics`; plus the
  tuning/adjusted accessors `ThresholdSweep.best_f1` / `at_recall` and
  `AdjustedMetrics.recall_adjusted` / `f1_adjusted`.
- `SimilarityThresholdFilter` (the `Filter` reference adapter).
- Dependency-free reference stack: `HashedNGramEmbedder`, `NumpyFlatIndex` /
  `NumpySearchableIndex`, `DenseBlocker` / `DenseBlockingIndex`,
  `ThresholdMatcher`, and the `Template` / `Fieldwise` / `WholeRow` serializers.
- A Sphinx documentation site (GitHub Pages) and a runnable
  `examples/04_dedupe.py`.

### Changed
- The v1 public contract is **frozen** (A0.5 gate, ADR-0003); evolution is now
  extend-never-modify.

### Deferred (experimental this release)
- The heavy adapters `FaissFlatIndex` / `FaissSearchableIndex`,
  `SentenceTransformerEmbedder`, and `LangChainMatcher` are declared (their
  extras and import paths are stable) but raise `NotImplementedError` — use the
  dependency-free reference stack until they land. Their producers
  (`tune_threshold`, `adjusted_metrics`) and the v2 `Trainer` adapters follow in
  later releases; see `docs/development/roadmap.md`.

[1.0.0]: https://github.com/caalvaro/denselinkage/releases/tag/v1.0.0
[1.0.0b2]: https://github.com/caalvaro/denselinkage/releases/tag/v1.0.0b2
[1.0.0b1]: https://github.com/caalvaro/denselinkage/releases/tag/v1.0.0b1
