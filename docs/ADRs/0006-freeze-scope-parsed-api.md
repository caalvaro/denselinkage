# ADR-0006: The freeze is scoped to the parsed public API, not the file bytes

**Status:** Accepted (2026-08-11)
**Date:** 2026-08-11
**Deciders:** Alvaro (author)

## Context

[ADR-0003](./0003-pre-freeze-contract-ratification.md) established
extend-never-modify as the post-freeze rule and classified changes with the
add/remove asymmetry table: the surface that must be correct now is
"existing-port method/property sets, field types, and calling conventions."

That table is semantic. It never says what a *file* may contain, only what the
*API* may become. The distinction went unstated because it did not arise: from
the `v1.0.0-freeze` tag to `v1.0.0`, `src/denselinkage/core/ports.py` was not
touched at all, and the ISE 2026 paper reports that interval as an empty diff
reproducible from the public tags.

Two events made the ambiguity concrete.

First, the file has since been modified. Issue #17 asked for the `Matcher` port
docstring to cross-reference the `CandidatePair.similarity_score` contract, and
PR #28 implemented it. The blob is `7ad5b0b` at both `v1.0.0-freeze` and
`v1.0.0`, and `327c89e` on `main`. The change adds four docstring lines; no
member, signature or field type moves.

Second, issue #30 proposes a fitness function enforcing extend-never-modify
mechanically, and cannot be designed without an answer. An AST snapshot of the
parsed API and a blob-hash comparison of the file are different checks with
different verdicts, and the second fails on `main` today.

Two port docstrings are also incomplete: `Serializer`
(`core/ports.py:37-39`) carries neither a class nor a method docstring, and
`Matcher` (`:167-177`) has no class docstring, so its contract lives entirely on
`Matcher.match`. Filling those holes is desirable and is blocked on the same
question.

## Decision

**The freeze is scoped to the parsed public API of `src/denselinkage/core/`.**

> **Widened by [ADR-0007](./0007-freeze-scope-enumerated-surface.md)
> (2026-08-13).** The location clause above is superseded: the freeze binds the
> whole surface enumerated in
> [freeze-gate.md](../development/freeze-gate.md), which includes orchestration
> (`linkage/`) and metrics (`metrics/`). Everything below about *what* is frozen
> — the parsed API rather than the file bytes — stands unchanged, and remains
> the authority for it.

Frozen, and changeable only by a major version:

- the set of `Protocol` classes in `core/ports.py` and the member set of each
- every member's signature: parameter names, order, kinds, defaults, and
  annotations, and the return annotation
- field names, types, defaults and ordering on the frozen domain dataclasses
- the exception taxonomy's class names and their inheritance
- the names exported from `denselinkage.core.__all__`

Not frozen, and editable in a patch or minor release:

- docstrings and comments
- formatting, import order, and anything `ruff format` normalises
- private names, and anything under `_store/`, `_reader/`, `_optional/`

Adding a new `Protocol`, a new type, or a new module remains additive and
therefore minor, per ADR-0003. It is nonetheless an architecture decision and is
proposed rather than merged in passing.

## Options Considered

### Option A — Parsed public API (chosen)

Matches what ADR-0003 already says, matches what structural typing actually
constrains, and keeps the file where the contract is documented editable.

### Option B — File bytes

Trivially checkable with a blob hash, and the strongest possible reading. It
forbids fixing a typo or adding a docstring in `core/ports.py`, which is exactly
the file whose docstrings carry the contract. It is also already violated on
`main`, so adopting it would require reverting PR #28 and leaving two ports
without contract docstrings permanently.

## Trade-off Analysis

Option B is cheaper to enforce and easier to state. It was rejected because the
property it protects is not the property that matters: what breaks a third-party
implementer is a member set, a signature or a field type. A docstring cannot
break a structural `Protocol`, and treating it as if it could would make the
contract documentation unmaintainable to protect a guarantee nobody depends on.

The cost of Option A is that enforcement is no longer a one-line hash comparison.
Issue #30 absorbs that cost: an AST-derived snapshot records member names, kinds
and unparsed signatures, and is invariant to docstrings and formatting.

## Consequences

- Issue #30 snapshots the parsed API. `ast.unparse` normalises formatting, so a
  `ruff format` pass cannot produce a spurious diff, and a docstring edit does
  not either.
- The two missing port docstrings may be filled without a major version.
- `AGENTS.md` states the rule in the form a contributor or an agent needs: a
  docstring edit in `core/ports.py` is permitted; adding a member, changing a
  signature, or changing a field type is not.
- PR #28 is retrospectively in conformance. It was already in conformance with
  ADR-0003; this ADR removes the ambiguity that made that unclear.

### What this does not change

The claims published in the ISE 2026 paper are scoped to the interval between
`v1.0.0-freeze` and `v1.0.0`. Both tags carry blob `7ad5b0b`, so
`git diff v1.0.0-freeze v1.0.0 -- src/denselinkage/core/ports.py` is empty and
remains so; the `release tag protection` ruleset prevents either tag from moving.
No edit to `main` can affect a diff between two immutable tags. The paper's
byte-level observation stays true as a historical fact about that interval; this
ADR governs evolution after it, which the paper does not describe.

## Action Items

- [ ] #30 names the parsed API in its acceptance criteria and rejects a
      blob-hash design.
- [ ] `docs/development/freeze-gate.md` states the scope in its frozen-contract
      section.
- [ ] #34 fills the `Serializer` and `Matcher` class docstrings.

## References / prior art

- [ADR-0003](./0003-pre-freeze-contract-ratification.md) — the add/remove
  asymmetry table this ADR scopes.
- [freeze-gate.md](../development/freeze-gate.md) — what "frozen" means and the
  A0.5 gate.
- Issues #38 (this decision), #30 (mechanical enforcement), #17 and PR #28 (the
  docstring change that surfaced the ambiguity), #34 (agent configuration).
