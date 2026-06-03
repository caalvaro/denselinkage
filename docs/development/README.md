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
  D7, D8) and the ADR-0001 summary. (D4 and D5 are Resolvi-surfaced forks
  recorded in resolvi-conformance.md.)
- [contract.md](./contract.md) — the contract reference: the reference component
  map, the hard-failure exception taxonomy, the Source→Record seam, and the
  Part-5 polish outcomes.
- [resolvi-conformance.md](./resolvi-conformance.md) — conformance against the
  Resolvi reference architecture, the Resolvi forks (D4, D5), the A0.5
  contract-shape additions, conscious declines, and roadmap deltas.
- [examples.md](./examples.md) — the `examples/` as executable spec: status,
  review findings, and roadmap (including the `examples/04` decision).

Architecture Decision Records live in [`../ADRs/`](../ADRs/).

## Project stage

**Structure stage.** `src/denselinkage/` defines public types, ports
(`typing.Protocol`s) and signatures; bodies are `...` placeholders. The
`examples/` are the design spec for the intended API and are type-checked,
lint-checked and compiled in CI against the real core. Implementation lands
incrementally against the frozen contract (do not change a frozen surface —
*extend, never modify*: add an optional field with a default, a sibling type, or
a new classmethod).
