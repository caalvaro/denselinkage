# Contract reference

> Part of the [denselinkage development docs](./README.md). The contract surface
> itself (resolved deltas, the failure taxonomy, the Source→Record seam) plus
> the Part-5 polish outcomes. Design rationale lives in
> [decisions.md](./decisions.md) and
> [resolvi-conformance.md](./resolvi-conformance.md).

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
