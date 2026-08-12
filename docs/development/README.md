# denselinkage — Development Docs

This folder is the in-tree, diffable, **authoritative** record of the contract,
the phase boundaries, and every recorded design decision. It is authoritative;
chat threads are not. The former monolithic `implementation_plan.md` has been
split into the focused documents below.

## Contents

- [roadmap.md](./roadmap.md) — phases (A0 → C) and their exit criteria.
- [freeze-gate.md](./freeze-gate.md) — what "frozen" means, the four-oracle A0.5
  freeze gate, and the re-declared frozen contract.
- [decisions.md](./decisions.md) — the recorded decision log (D1, D2, D3, D6,
  D7, D8, D9) and the ADR-0001 summary. (D4 and D5 are Resolvi-surfaced forks
  recorded in resolvi-conformance.md.)
- [contract.md](./contract.md) — the contract reference: the reference component
  map, the hard-failure exception taxonomy, the Source→Record seam, and the
  Part-5 polish outcomes.
- [resolvi-conformance.md](./resolvi-conformance.md) — conformance against the
  Resolvi reference architecture, the Resolvi forks (D4, D5), the A0.5
  contract-shape additions, conscious declines, and roadmap deltas.
- [examples.md](./examples.md) — the `examples/` as executable spec: status,
  review findings, and roadmap (including the `examples/04` decision).
- [releasing.md](./releasing.md) — branching model, what decides a version
  number, the release runbook, and what `main` and the `v*` tags are protected
  against.
- [conventions.md](./conventions.md) — the design patterns in use and their
  anti-patterns, style the linter does not enforce, testing style, and the
  checklist for adding an adapter. The short form for AI assistants is
  [`AGENTS.md`](../../AGENTS.md).

Architecture Decision Records live in [`../ADRs/`](../ADRs/): 0001 (spec→artifact),
0002 (evaluation types out of core), 0003 (pre-freeze ratification), 0004
(dependency-free beta / A1 implementation), 0005 (heavy adapters and the
adapter-scoped coverage gate).

## Project stage

**Released — `1.0.0`.** The frozen contract's bodies are
implemented for the no-extras surface, which **runs** on numpy + pandas
(`link` / `dedupe` / `match_pairs`, connected-components clustering, and the
linkage / blocking / clustering metrics + filtering), under branch coverage with
`fail_under = 100`. The heavy extras (faiss / sentence-transformers / langchain)
have since landed behind their optional extras (ADR-0005), gated by a dedicated
adapter-coverage CI job.
Evolution stays *extend, never modify* against the frozen contract (add an
optional field with a default, a sibling type, or a new classmethod). See
[ADR-0004](../ADRs/0004-dependency-free-beta.md).
