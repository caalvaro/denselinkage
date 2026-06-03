# ADR-0003: Pre-freeze contract ratification

**Status:** Accepted (2026-06-02); one action item (the signature pass) open before the gate flips
**Date:** 2026-06-02
**Deciders:** Alvaro (author); thesis advisor sign-off

## Context

The A0.5 freeze gate is imminent — it locks the v1 public surface and lets body
implementation (A1) begin. A pre-freeze tech-debt scan surfaced ~a dozen
candidate issues spanning unused port surface, error-taxonomy consistency,
value-object hashability, and "should we even freeze this speculative type?".

Before locking, the useful move is not to fix everything but to separate what is
**expensive to change after the freeze** from what is **cheap to reverse**, and
ratify only the former. This ADR records that triage and the resulting freeze
checklist.

## The governing principle — the add/remove asymmetry

Under the project's post-freeze rule (*extend, never modify*), changes are **not**
symmetric:

| Post-freeze change | Class |
|---|---|
| New port / type / module; new method on a class; new **optional** field (with default) on a frozen dataclass | **additive — safe** |
| **Add** a required method/property to an existing port | **breaking** |
| Change a field **type** or a method **signature**; remove a field/type | **breaking** |
| **Remove** an unused method from a port | **relaxing — ~safe** |

Therefore the only surface that must be *correct now* is **existing-port
method/property sets, field types, and calling conventions.** New
ports/types/methods can be added later non-breakingly; unused port members can be
pruned later cheaply. This asymmetry decides every item below — and, notably,
**reverses** the tech-debt scan's "trim the unused `Embedder` properties"
recommendation, because trimming is the one move the asymmetry punishes.

## Decision

Adopt the asymmetry as the freeze rubric, and ratify:

- **D-3.1 — `matcher` stays required (D4 declined).** The package identity is
  dense blocking *plus* matching; the degenerate "no real matcher" case is served
  by `ThresholdMatcher`. `matcher` is a required field on the frozen
  `DenseLinker`, so this is modify-if-late — recorded now as a *decision*, not a
  default.
- **D-3.2 — Two-tier error contract, documented.** `ValueError` = **API misuse /
  programmer error** (`blocker=None` on `link`/`index`/`dedupe`; a `Matcher`
  returning the wrong number of outcomes). `DenseLinkageError` (+ taxonomy) =
  **data/runtime** hard failure. The split is deliberate and now documented on
  `core.errors`, because error types are part of the contract (a post-freeze
  change breaks callers' `except` clauses).
- **D-3.3 — Keep + document speculative port surface; do not trim.**
  `Embedder.model_id` / `embedding_dim`, the Resolvi-conformance ports (`Filter`,
  `Clusterer`, `Trainer`), and `SearchableIndex.extended` stay. Adding a port
  member post-freeze is breaking; removing an unused one later is cheap — so the
  asymmetry favours inclusion. "Mystery surface" is fixed by **documenting the
  intended consumer**, not by removal.
- **D-3.4 — Frozen-object hashability: documented-or-ignored.** A
  `@dataclass(frozen=True)` carrying a `Mapping`/`DataFrame` raises on `hash()` —
  standard Python, not a denselinkage defect, and nothing hashes these objects.
  Out of freeze scope; a one-line "not a hash key" note is optional.
- **D-3.5 — Freeze checklist = one signatures-and-field-types pass.** The single
  remaining high-value pre-freeze task is an adversarial read of every port
  signature and frozen field type (the modify-if-late surface), then flip the
  gate.

## Options Considered

### Option A — Over-freeze: lock the full design space, treat every member as final
| Dimension | Assessment |
|---|---|
| Complexity | Low (no triage) |
| Post-freeze risk | **High — every frozen member is a forever-promise; mistakes are breaking to fix** |
| Thesis fit | Neutral |

### Option B — Minimal surface: defer every speculative port/type
| Dimension | Assessment |
|---|---|
| Complexity | Medium (rip out Filter/Trainer/sweep types) |
| Post-freeze risk | Low (all additive later) |
| Thesis fit | **Negative — loses the Resolvi "full ER design space" conformance story** |

### Option C — Asymmetry-guided (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Low (a short, bounded checklist) |
| Post-freeze risk | **Low — strict only where change is expensive; additive corridor stays open** |
| Thesis fit | **High — keeps the conformance design space; prunes later if unused** |

## Trade-off Analysis

Option B is risk-neutral on the speculative ports (they're additive later) but
spends effort *removing* the very surface that carries the paper's
Resolvi-conformance argument — a net loss for a contract-as-deliverable thesis.
Option A over-commits: it treats reversible decisions as irreversible and invites
breaking "fixes" later. Option C uses the asymmetry as a scalpel — be strict only
on the modify-if-late surface (signatures, field types, calling conventions),
keep additive-safe surface where it earns its place (conformance ports), and lean
on the cheap-to-prune direction for unused members. The scan's "trim
`embedding_dim`" item is reversed under C precisely because *trim-now /
re-add-later* is the asymmetry-punished path, whereas *keep-now / prune-later* is
free.

## Consequences

**Easier**
- The freeze decision is now a short, bounded checklist rather than an open audit.
- Post-freeze evolution is predictable: a wide additive corridor, a narrow and
  known breaking one.

**Harder**
- A few unused port members ship in v1 — adapter authors implement them
  (one-liners) — accepted and documented.

**To revisit (all non-breaking)**
- Prune `Embedder.model_id` / `embedding_dim` / `SearchableIndex.extended` if
  Phase B/C never consumes them.
- The `core/results.py` → `core/outputs.py` rename (ADR-0002, cosmetic).
- Hashability note, only if these objects ever become hash keys.

## Action Items

1. [x] Ratify D4 decline — `matcher` stays required (recorded here +
   `resolvi-conformance.md` D4 marked ratified).
2. [x] Document the two-tier error principle on `core/errors.py`.
3. [x] Document the intended consumers on `Embedder.model_id` / `embedding_dim`
   (+ the asymmetry rationale on the `Embedder` port).
4. [ ] **Run the signatures-and-field-types pass** (the freeze checklist) — the
   last step before flipping the A0.5 gate.
5. [x] Log this as **D8** in `docs/development/decisions.md`.

## References / prior art

- **ADR-0001** (spec→artifact) and **ADR-0002** (evaluation types out of core) —
  this ADR continues the "audit what the contract must contain" discipline and
  adds the *temporal* lens (what is expensive to change *after* the freeze).
- **Semantic Versioning** / **extend-never-modify** — the additive-vs-breaking
  classification the asymmetry rests on.
- **Interface Segregation Principle** — the counter-pressure against unused port
  members; resolved here by documentation + the prune-later escape, not removal.
