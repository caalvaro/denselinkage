# Freeze semantics & the A0.5 gate (what "frozen" means)

> Part of the [denselinkage development docs](./README.md).

"Freeze" in this plan is a specific event, not a vague aspiration: the **A0.5
re-freeze gate** — the point at which the public surface is locked and body
implementation (A1+) may begin. It is the project's **v1 / 1.0 contract
commitment**. The mapping to the usual version vocabulary:

- **Structure stage (A0, A0.5) = the 0.x hardening window.** The contract is
  deliberately malleable; *contract-shape* changes (a new port, a widened frozen
  field, a reshaped calling convention) are **cheap and expected** here.
- **The A0.5 gate = the freeze = the 1.0 commitment.** After it, the surface is
  fixed and evolution is constrained to **extend-never-modify**.

Therefore **extend-never-modify is a *post*-freeze rule.** D1–D6,
`clustering_metrics`, and the `Filter` port are all contract-shape and all land
*in A0.5* — that is the gate doing its job (absorbing breaking-if-late changes
before the lock), not a violation of it. The decision-log triage
(`paper_evaluation_data.md` §3) measures exactly this: the bulk of
architecture-bearing decisions are contract-shape, which is *why* the freeze
must happen after they are resolved, not before.

**The freeze gate is an ordered battery of oracles**, escalating from cheap and
broad to expensive and decisive:

1. **Reference-architecture conformance** (the Resolvi walk) — drives *which*
   variation points must exist (surfaced D4, D5, `Filter`, `clustering_metrics`,
   and the declines). See [resolvi-conformance.md](./resolvi-conformance.md).
2. **Strict static typing** (`mypy --strict` on `src` + `examples`) — surface
   self-consistency.
3. **Examples-as-specification** (`compileall` + lint + type-check the
   `examples/` against the real core in CI) — documented usage cannot drift from
   the contract.
4. **Reference-implementation attemptability** — a dependency-free vertical
   slice (`with_defaults()` → `link()`) must be *implementable* against the
   stubs with no contract-shape surprise.

Oracles 1–3 are necessary but **not sufficient**: D6 (ADR-0001) passed all three
and was caught only by oracle 4. Implementing the slice surfaced a **second**
oracle-4 catch — D1's verb-dependent comparison was unimplementable as recorded
(the verb is not recoverable from the result), resolved by an explicit
`directed=` metric parameter (see the D1 amendment in
[decisions.md](./decisions.md)). A0.5 therefore carries an explicit exit
criterion — *the `with_defaults()` / `00_quickstart` vertical slice is shown
implementable against the frozen stubs* — now **met**: the slice is implemented
on the dependency-free stack and green (`tests/test_quickstart_end_to_end.py`,
`00_quickstart.py` runs at P/R/F1 = 1.0). The paper's C-B contribution is the
completed four-oracle gate, with D6 and the D1 amendment as its two worked
motivations.

## Frozen contract (re-declared after A0.5 verification)

Once the A0.5 DoD checks are green (ruff incl. format, mypy `src+examples`,
compileall examples, pytest, import-isolation), the public surface is frozen and
A1 may begin.
