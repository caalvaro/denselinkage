# Examples as executable spec — status & roadmap

> Part of the [denselinkage development docs](./README.md). The `examples/` are
> the design spec for the intended API (type-/lint-/compile-checked in CI).

## examples/04

`examples/04_second_pass_blocking.py` was deleted (orphan `.pyc` removed in
this pass). **Decision:** *removed from Phase-A exit criteria.* A
hard-negative second-pass example that fakes blockers in v1 would ship a
mock of the package's own motivating method against a "frozen" contract;
instead it is a **Phase-C** deliverable, restored as a real (non-faked)
example once `mine_hard_negatives` + trainers exist. The plan no longer
treats it as a Phase-A labelled-simulation deliverable.

## Review findings & roadmap

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
