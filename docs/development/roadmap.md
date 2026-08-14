# Roadmap — phases & exit criteria

> Part of the [denselinkage development docs](./README.md). See
> [freeze-gate.md](./freeze-gate.md) for what "frozen" / the A0.5 gate means.

- **A0** *(done)* — typed core skeleton: models, ports, results, linker,
  metrics, serializers, reference adapters; dependency-cut CI; `[train]`
  namespace reserved.
- **A0.5** *(contract hardening before freeze)* — close the gaps
  that would become breaking post-freeze: D1/D2/D3, the hard-failure
  exception taxonomy, the named Source→Record seam, `BlockingMetrics`
  constructor shape, kwarg-`gold` consistency, examples wired into CI
  (lint+format+type+compile), one Python-version story, conventions aligned.
  **Exit:** every DoD check green and pasted verbatim; contract re-frozen.
- **A1 — dependency-free beta (`1.0.0b1`)** *(done)* — fill the frozen
  contract's bodies for the no-extras surface: `link` / `dedupe` / `match_pairs`,
  connected-components clustering, and the linkage / blocking / clustering
  metrics + filtering. The heavy extras (faiss / sentence-transformers /
  langchain) are deferred to A2 (below). Branch coverage + `fail_under = 100`
  enforced. See [ADR-0004](../ADRs/0004-dependency-free-beta.md).
- **A2 — heavy adapters** *(done)* — fill the four deferred adapter bodies
  (`SentenceTransformerEmbedder`, `FaissFlatIndex` / `FaissSearchableIndex`,
  `LangChainMatcher`) behind their optional extras, so the headline dense-semantic
  + LLM stack runs. Additive and lazy-imported (the dependency cut holds);
  incremental `extended` stays deferred. A dedicated adapter-coverage CI job gates
  the adapter modules at 100% (the dependency-free gate `omit`s them). See
  [ADR-0005](../ADRs/0005-heavy-adapters.md).
- **Phase B** *(v1, dependency-light)* — `mine_hard_negatives`,
  `tune_threshold`, `adjusted_metrics`, `LabeledPairs.split`,
  `LinkageResult.from_candidate_frame` (the ergonomic frame→pairs constructor
  that makes `match_pairs` end-to-end usable — see D2 in
  [decisions.md](./decisions.md)).
- **Phase C** *(behind `[train]`; targets 1.2.0, see below)* —
  `EmbedderTrainer` / `CrossEncoderTrainer` (implement the `Trainer` protocol) +
  few-shot selection. LLM fine-tuning is out of scope. The first/second-pass
  hard-negative *simulation* example (former `examples/04`) is a Phase-C
  deliverable — restored then as a **real** (non-faked) example once mining +
  trainers exist (see [examples.md](./examples.md)).
- **Phase D** *(published methods, each behind its own extra)* — implementations
  of published entity-matching methods as first-party adapters: Ditto, the
  DeepMatcher line, and others. They are `Matcher` implementations, so the port
  already covers them and no contract change is expected
  ([ADR-0008](../ADRs/0008-published-methods-as-first-party-adapters.md)). If a
  method cannot be expressed against the frozen ports, that is a finding about
  the ports and is recorded as one.

## Phase C and D: why this is 1.2.0 and not 2.0.0

The `Trainer` protocol, the `[train]` extra and the `denselinkage.training`
namespace were all reserved in Phase A precisely so that filling them later
would add nothing to the frozen surface. Under
[ADR-0003](../ADRs/0003-pre-freeze-contract-ratification.md)'s add/remove
asymmetry, a new adapter, a new module and a filled body are all **additive**,
and additive work ships in a minor release.

So the default target is **1.2.0**, and the decision is deferred to the
evidence rather than taken in advance: `tests/test_frozen_surface.py` compares
the parsed public API against `tests/api_snapshot.json` on every matrix leg, so
it reports whether the release breaks anything. If the snapshot does not move,
the release is minor by the project's own rule.

This is not bookkeeping. The ISE 2026 paper's claim is extend-never-modify, and
delivering training, fine-tuning and several published methods **without
breaking v1's contract** is the strongest available evidence for it. Numbering
that release 2.0.0 when nothing broke would spend the evidence for nothing.

Three things could still force a major version, and each needs its own ADR:

- raising `requires-python`, which drops supported interpreters;
- revising `Trainer`'s signature, which is a live risk because the protocol has
  never had an implementer and Phase C is the first time it meets one;
- reworking `SearchableIndex.extended` beyond what
  [ADR-0001](../ADRs/0001-stateful-components-as-artifacts.md) already settled.

## JOSS submission

A [JOSS](https://joss.theoj.org) submission is planned once Phase C lands. The
relevant constraints, checked against JOSS's own documentation rather than
assumed:

- **Prior publication is not a bar.** JOSS explicitly handles co-publication:
  it "considers submissions for which the implementation of the software itself
  reflects a substantial scientific effort", and asks authors to declare related
  publications. The ISE 2026 paper's contribution is a design method; a JOSS
  paper's contribution is the library as a research instrument. Different
  objects, different claims.
- **The repository must have been public for more than six months with active
  development spanning that period.** This repository has been public since
  2026-05-19, so that window opens around **19 November 2026**.
- **Feature-complete, no half-baked solutions.** This is the binding constraint
  and the reason to submit after Phase C rather than now: `Trainer` is
  contract-only and `SearchableIndex.extended` raises. Issue #19 closes the
  second; Phase C closes the first.
- **Substantial scholarly effort**, judged partly on size, commit history and
  contributor count. External contributors are a signal rather than a
  requirement.

The conformance harness (#33) matters more under
[ADR-0008](../ADRs/0008-published-methods-as-first-party-adapters.md) than it
did before: with methods carried in-repo, it is the only remaining evidence
that the contract is implementable by someone outside it.
