# denselinkage — agent instructions

`denselinkage` is the artifact behind a published ISE 2026 paper: the invariants below are
published claims, so breaking one is a correctness failure, not a style disagreement.
Commands are in [CONTRIBUTING.md](CONTRIBUTING.md); conventions derived from the code are
in [docs/development/conventions.md](docs/development/conventions.md).

## The frozen contract

- NEVER add or remove a member on an existing `Protocol` in `src/denselinkage/core/ports.py`.
  Under structural typing that breaks every third-party implementer and forces a major
  version (ADR-0003). mypy cannot see them: this repo stays green while their code breaks.
- NEVER change a signature, field type, default, or public name on the frozen surface.
- The freeze binds the **parsed public API**, not the file bytes (ADR-0006). A docstring
  edit in `core/ports.py` is allowed; a signature edit is not.
- A new `Protocol` is additive and minor, but it is an architecture decision: propose it.
- ALWAYS check your own diff: `git diff v1.0.0 -- src/denselinkage/core/`

## The dependency cut

- NEVER import `faiss`, `sentence_transformers`, `langchain_*` or `torch` at module scope.
- ALWAYS call `require("<module>")` then issue the real import, both inside the method.
  Do not bind `require`'s return: that hands mypy a `ModuleType` and defeats the stub override.
- NEVER import an adapter or a heavy backend from `src/denselinkage/core/`.

## Failure tiers, disjoint on purpose

- ALWAYS return a `MatchError` from `Matcher.match` for a pair it cannot decide. NEVER raise
  into the batch, and NEVER return a list of a different length than its input.
- ALWAYS raise a `DenseLinkageError` subclass for a hard data failure, exported from
  `denselinkage.core`.
- ALWAYS raise plain `ValueError` for API misuse. It sits outside `DenseLinkageError` on
  purpose, so `except DenseLinkageError` cannot swallow a caller's own bug.
- NEVER merge the tiers, reparent `ValueError`, or add an error field to `MatchDecision`.

## Purity

- NEVER mutate an artifact, or `self`, inside `build()`; return a new artifact. This is what
  makes `link(a, b) == index(a).query(b)` hold. Growing an index returns a new artifact via
  `SearchableIndex.extended`, never mutates in place (ADR-0001).

## Tests and coverage

- ALWAYS meet the 100% branch gate by writing the missing test. NEVER widen `omit`, lower
  `fail_under`, or reach the number with `# pragma: no cover`.
- A new adapter module MUST be registered in BOTH coverage configs: `omit` in
  `pyproject.toml`, `include` in `.coveragerc.adapter`. Opposite polarity, so missing the
  first fails the matrix and missing the second silently ungates the module.
- ALWAYS update `examples/` in the same change as an API change. CI lints, type-checks,
  compiles and runs them: they are the specification of intended use.

## Verifying your own work

- NEVER report a check as passing without its output.
- NEVER assume a failing check is pre-existing. `main` is green; assume your change caused it.
- NEVER attribute a coverage drop to an unrelated change. The gate is exact.

## Reviewing a diff

Check it against the freeze, the dependency cut, the failure tiers and purity. Report with
`path:LINE`, ordered by severity, blockers separated from non-blocking. Do not apply
agent-only workflow rules to human authors, nor flag pre-existing issues.

## Settled. Do not reopen without an ADR

No runtime `isinstance` dispatch against a port (`core/ports.py:30-34`); no `pydantic` or
`attrs` for domain types (ADR-0001); no distributed-execution layer; `Trainer` contract-only
in v1 and `SearchableIndex.extended` raises; evaluation reports live in
`denselinkage.metrics`, never `core` (ADR-0002).

## Do not edit

`paper/` is the frozen evidence behind the published paper: read it, never edit it.
Rerunning or tidying `paper/probe/` invalidates reported results.
`paper/probe/verify_probe.py` may be *run* unchanged against a new single-file `Matcher`.

## Where the answers are

| Question | File |
| --- | --- |
| What "frozen" means, and its scope | `docs/development/freeze-gate.md` |
| Conventions, testing style, adapter checklist | `docs/development/conventions.md` |
| Why a decision was made; v1 versus v2 | `docs/development/decisions.md`, `docs/ADRs/`, `roadmap.md` |
| Releasing and version numbering | `docs/development/releasing.md` |

When the contract is ambiguous, read the port docstring, then the ADR. Do not infer intent
from an adapter.
