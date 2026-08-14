# ADR-0008: Published matching methods ship as first-party adapters

**Status:** Proposed (2026-08-14)
**Date:** 2026-08-14
**Deciders:** Alvaro (author)

## Context

Phase C adds training and fine-tuning. Alongside it, the project intends to
carry implementations of published entity-matching methods: Ditto (Li et al.,
VLDB 2020, entity matching as sequence-pair classification over a pre-trained
transformer), the DeepMatcher line, and others.

Those methods are **matchers**, not trainers. Ditto classifies a serialized
record pair, which is exactly what `Matcher.match` already describes. The port
for them exists and is frozen. So the question this ADR answers is not how to
express them. It is **where their code lives**.

The question is forced now rather than later because the answer determines the
shape of Phase C, the dependency surface of the wheel, the size of the
adapter-coverage matrix, and what a reader of the package understands
`denselinkage` to be: a contract other people implement, or a library that
carries the methods.

A second motivation is external. A software paper describing the library as a
research instrument is planned after Phase C (see
[roadmap.md](../development/roadmap.md)). Such a paper is judged on whether the
software is feature-complete and represents substantial effort, weighing signals
including size and demonstrated research use. A package that is mostly protocol
definitions is defensible architecturally and reads thin under that rubric.

## Decision

**Published methods ship inside this repository, each behind its own optional
extra**, following the pattern ADR-0005 established for the heavy adapters.

A method adapter is an ordinary adapter. It subclasses its port, lazy-imports
its backend through `require()`, registers in both coverage configurations, and
is derived automatically by the conformance checks in `tests/test_contract.py`
rather than added to any list. Nothing about the contract changes to
accommodate it, which is the point: if a published method cannot be expressed
against the frozen ports, that is a finding about the ports and is recorded as
one.

The `[train]` extra covers the training machinery. Each method gets its own
extra rather than joining `[train]`, so installing Ditto does not install every
other method's backend, and `[all]` remains the union.

## Options Considered

### Option A — Methods as first-party adapters behind per-method extras (chosen)

One install, one test suite, one coverage gate, one place to look. The
comparison the project exists to enable, running several methods over the same
data through the same ports, works out of the box rather than requiring the
reader to assemble it. It is also the option that most obviously satisfies a
reviewer asking whether the software is feature-complete and substantial.

### Option B — Contract and conformance harness only, methods live outside

`denselinkage` ships ports, the conformance harness, and one or two reference
implementations; methods are separate packages implementing the contract.

This is the option the architecture argues for, and rejecting it is the
substance of this decision. It keeps the wheel small, keeps heavy backends out
entirely, and makes the contract the product. It was rejected because it moves
the burden of assembly onto every user, gives the project no way to guarantee
that a claimed implementation actually conforms, and offers a reviewer a
package that is mostly interface definitions.

### Option C — A flagship method in-repo, the rest documented as a pattern

Rejected as the least decisive of the three: it carries Option A's maintenance
cost for one method while leaving Option B's assembly burden for all the
others, and it makes the boundary between "carried" and "not carried"
arbitrary.

## Trade-off Analysis

Option A's costs are real and are accepted deliberately rather than overlooked.

**Maintenance of code the project did not invent.** A method adapter tracks an
upstream paper and an upstream library. When `transformers` breaks an API, this
repository is what goes red. ADR-0005 already accepted this for four adapters;
this decision multiplies it by the number of methods carried.

**The adapter-coverage matrix grows per method.** Every heavy module is
registered twice at opposite polarity, `omit` in `pyproject.toml` and `include`
in `.coveragerc.adapter`, and gated at 100% by a dedicated CI job. Each method
adds a module to both, and its backend to the adapter job's install. The
`tests/test_coverage_registration.py` gate added in #31 is what keeps the two
lists honest; without it this cost would compound silently.

**CI time and cache pressure.** The adapter job already installs torch and the
CUDA wheels, measured at roughly 2.7 GB, and runs in 56 s cold. Methods that
need a different backend widen that closure.

**A weaker incentive for third-party implementations.** If the project carries
every method, nobody outside needs to implement the contract, and the claim
that the contract is implementable by others loses its evidence. The
conformance harness (#33) is the mitigation and becomes more important under
this decision, not less: it must be a documented, reusable, public thing so an
outside implementer can check their work without vendoring their method here.

**The dependency cut becomes load-bearing for users, not just for tests.** With
several heavy extras, `pip install denselinkage` staying free of all of them is
what keeps the package usable for someone who wants only the dependency-free
stack. The existing `core-only` CI job already proves this and now guards a
larger claim.

## Consequences

- Each method adapter follows the `new-adapter` checklist unchanged: explicit
  port subclassing, `require()` plus a method-local import, registration in both
  coverage configurations, a dedicated test module, and a differential test
  against the dependency-free reference where one applies.
- No contract change is expected. If a method cannot be expressed against the
  frozen ports, that is recorded as a finding and, if it requires a port change,
  it is a major-version event under ADR-0003 and gets its own ADR.
- `[all]` grows to the union of the method extras. Whether `[all]` remains
  practical to install is a question to revisit when the third method lands.
- The conformance harness in #33 is promoted from a test-suite cleanup to a
  public deliverable, because it is the only remaining evidence that the
  contract is implementable by someone outside this repository.
- Method adapters are additive under ADR-0003 and do not by themselves force a
  major version. See [roadmap.md](../development/roadmap.md) on why Phase C
  currently targets 1.2.0.

## Action Items

- [ ] Record the per-method extra naming convention in
      [conventions.md](../development/conventions.md) when the first one lands.
- [ ] #33 makes the conformance harness public and documented, as the mitigation
      named above.
- [ ] Revisit `[all]` when a third method extra exists.

## References / prior art

- [ADR-0005](./0005-heavy-adapters.md) — the heavy-adapter pattern this extends,
  and the source of the per-extra lazy-import discipline.
- [ADR-0003](./0003-pre-freeze-contract-ratification.md) — why a new adapter is
  additive and needs no major version.
- [ADR-0002](./0002-evaluation-types-out-of-core.md) — the precedent for keeping
  a category of code out of `core` while still shipping it in the wheel.
- Li, Li, Suhara, Doan and Tan, "Deep Entity Matching with Pre-Trained Language
  Models" (VLDB 2020) — Ditto. <https://arxiv.org/abs/2004.00584>
- Mudgal et al., "Deep Learning for Entity Matching: A Design Space Exploration"
  (SIGMOD 2018) — DeepMatcher.
