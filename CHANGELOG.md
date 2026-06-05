# Changelog

All notable changes to denselinkage are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[1.0.0b1]: https://github.com/caalvaro/denselinkage/releases/tag/v1.0.0b1
