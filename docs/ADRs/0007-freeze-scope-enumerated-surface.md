# ADR-0007: The freeze covers the enumerated surface, not `core/` alone

**Status:** Accepted (2026-08-13)
**Date:** 2026-08-13
**Deciders:** Alvaro (author)

## Context

[ADR-0006](./0006-freeze-scope-parsed-api.md) settled *what kind of thing* the
freeze binds: the parsed public API, not the file bytes. Its Decision sentence
also states *where*: "The freeze is scoped to the parsed public API of
`src/denselinkage/core/`."

That location is narrower than the surface the project already publishes as
frozen. [freeze-gate.md](../development/freeze-gate.md) enumerates six groups
under "The frozen surface", and `AGENTS.md` repeats the same six to agents:
ports, models, results, errors, **orchestration**, **metrics**. Only the first
four live in `core/`. The other two resolve to seven modules elsewhere:

| Group | Modules |
| --- | --- |
| Orchestration | `linkage/dense_linker.py`, `linkage/linkage_index.py` |
| Metrics | `metrics/linkage.py`, `metrics/blocking.py`, `metrics/clustering.py`, `metrics/tuning.py`, `metrics/adjusted.py` |

The placement is deliberate, not accidental: [ADR-0002](./0002-evaluation-types-out-of-core.md)
puts the evaluation report types in `denselinkage.metrics` precisely so that
nothing in `core` depends on them. So the two documents cannot be reconciled by
moving code.

Issue #30 forced the question. A fitness function has to be told which modules
it reads, and whichever set it enforces becomes the operative answer regardless
of what either document says. Implementing it under ADR-0006's sentence would
have left `DenseLinker`'s six public methods, `LinkageIndex`, and all five
metrics modules ungated, while `freeze-gate.md` continued to describe them as
frozen.

A third defect surfaced alongside: `AGENTS.md` and
`.claude/agents/contract-reviewer.md` both said the scope ADR "is open in #39;
until it merges, freeze-gate.md is the citable authority". #39 merged on
2026-08-12, so both sentences deferred to a condition that no longer held.

## Decision

**The freeze binds the parsed public API of the surface enumerated in
`docs/development/freeze-gate.md`, currently eleven modules across `core/`,
`linkage/` and `metrics/`.**

ADR-0006's characterisation of *what* is frozen is unchanged and remains the
authority for it: the set of `Protocol`s and each member set; every member's
signature; field names, types, defaults and ordering; the exception taxonomy's
names and inheritance; `denselinkage.core.__all__`. Docstrings, comments,
formatting and private names remain outside the freeze.

Two things this ADR settles that ADR-0006 left open, both of which a mechanical
gate must answer one way or the other:

- **Public methods on frozen dataclasses are in scope.** ADR-0006:48 named
  fields only. `LabeledPairs.split`'s keyword-only `test_size` and `seed`, and
  `BlockingMetrics.from_pc_map`'s keyword-only `n_gold`, are calling conventions
  third parties depend on exactly as much as a field type.
- **Dataclass decorator arguments are in scope.** Dropping `kw_only=True` from
  `DenseLinker`, or `frozen=True` from any model, leaves every field name, type,
  default and ordinal byte-identical while changing the calling convention in
  the first case and breaking the purity invariant in the second.

**Enforcement is closed-world.** The gate records every public class, member,
field, decorator argument, module `__all__` and type alias in those modules,
rather than an allow-list mirroring `freeze-gate.md`. A hand-maintained list is
the defect class issue #31 exists to remove, and a second one inside the gate
would drift from the first. `freeze-gate.md`'s enumeration therefore describes
the surface; `tests/api_snapshot.json` defines it.

## Options Considered

### Option A — Widen the scope to the enumerated surface (chosen)

Matches what `freeze-gate.md` and `AGENTS.md` already promise, and matches what
a third party integrating against `DenseLinker.link` or `LinkageMetrics` would
reasonably assume from reading them.

### Option B — Keep `core/` and narrow the documents to match

Honours ADR-0006's sentence literally. Rejected: it silently withdraws a
published claim. `DenseLinker` is the package's primary entry point and the
metrics report types are its evaluation output; declaring them unfrozen after
1.0.0 would be a larger contract change than the one being avoided.

### Option C — Leave the contradiction and let the gate pick

Rejected as the status quo that produced the problem. Whichever set the gate
reads becomes operative, so leaving it undecided means deciding by accident.

## Trade-off Analysis

Option A widens what a major version protects, which raises the cost of changing
`linkage/` and `metrics/`. That cost is the intended one: those modules were
already described as frozen, and the alternative is that the description was
wrong for two releases.

The closed-world choice has a real cost of its own. Symbols nobody deliberately
froze are now gated, including `DenseLinker.block`, `LinkageIndex.__init__`,
`.candidates`, `.save`, `.load`, `tune_threshold`, `adjusted_metrics`,
`errors.__all__`, `RecordId` and `ComponentT`. Changing any of them now requires
a deliberate snapshot regeneration. That is preferable to the alternative, in
which each is unwatched until someone notices and adds it to a list, which is
how `IncompatibleStore` reached `errors.__all__` while missing from
`denselinkage.core` with no test noticing (issue #31, PR #29).

## Consequences

- `tests/test_frozen_surface.py` compares the eleven modules against
  `tests/api_snapshot.json` on every leg of the 3.10-3.13 matrix. `ast.unparse`
  output is byte-identical across those interpreters, so no new CI job is added.
- `freeze-gate.md`'s "The frozen surface" section is now a description of what
  the gate covers rather than the operative list.
- Regenerating the snapshot requires `--regenerate` **and** an `--authority`
  naming the ADR or issue that permits the change; the authority is recorded in
  the snapshot. Regeneration is the signal that the contract moved, so the
  regeneration commit is where a reviewer looks for the version decision.
- The stale "#39 is open" sentences in `AGENTS.md` and
  `.claude/agents/contract-reviewer.md` are corrected to cite ADR-0006 and this
  ADR.
- The additive corridor in `freeze-gate.md` is unaffected: a new `Protocol`,
  type, function or defaulted field is still additive and still ships in a minor
  release. The gate reports that severity in its failure message rather than
  treating every diff as breaking.

## Action Items

- [x] #30 enforces the enumerated surface, closed-world, and rejects a
      blob-hash design.
- [x] #32's failure messages cite this ADR for scope and ADR-0003 for severity.
- [ ] #31 derives the remaining hand-maintained whitelists (adapter/port pairs)
      from the same AST machinery.

## References / prior art

- [ADR-0003](./0003-pre-freeze-contract-ratification.md) — the add/remove
  asymmetry table that classifies each change the gate reports.
- [ADR-0006](./0006-freeze-scope-parsed-api.md) — parsed API rather than file
  bytes; this ADR widens only its location clause.
- [ADR-0002](./0002-evaluation-types-out-of-core.md) — why the metrics report
  types are outside `core` and cannot be moved into it.
- [freeze-gate.md](../development/freeze-gate.md) — the enumeration this ADR
  makes operative.
- Issues #30 (mechanical enforcement), #32 (failure messages), #31 (the
  remaining whitelists).
