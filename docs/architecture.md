# Architecture

denselinkage is built **ports-and-adapters** (hexagonal): a small, dependency-
free contract at the centre, with every replaceable part behind a typed port.
The architecture is the project's primary design goal — this page is the
public overview; the full design record (ADRs and decision logs) lives in the
repository, linked at the bottom.

## The contract is the centre

{doc}`denselinkage.core </api/contract>` holds only what the contract needs:
the **ports** ({class}`typing.Protocol`s), the domain **models**, the **results**
ports reference, and the **error** taxonomy. It depends on nothing heavy. Every
other module either implements a port here or orchestrates ports defined here.

First-party adapters **subclass their port explicitly**, so mypy verifies each
implementation is complete; third-party code may conform structurally without
importing anything.

## The dependency rule

The core stays light (NumPy + pandas). Heavy backends — FAISS,
sentence-transformers, LangChain — are optional extras that adapters import
**lazily**, inside the methods that use them. Consequences:

- `import denselinkage` never pulls a heavy backend.
- A missing extra surfaces only when you construct that specific adapter.
- The documentation you are reading builds with no extras installed at all.

This cut is enforced by a test, not just convention.

## Spec → artifact

Stateful work is split into a stateless **spec** and an immutable **artifact**:

- A {class}`~denselinkage.core.ports.Blocker` *spec* builds a
  {class}`~denselinkage.core.ports.BlockingIndex` *artifact*.
- A {class}`~denselinkage.core.ports.VectorIndex` *spec* builds a
  {class}`~denselinkage.core.ports.SearchableIndex` *artifact*.

`build()` is a factory: it returns fresh state and mutates neither `self` nor its
arguments. So one spec safely builds many artifacts, and an artifact is safe to
reuse across queries. The same split appears at the top level —
{class}`~denselinkage.linkage.DenseLinker` (immutable config) versus
{class}`~denselinkage.linkage.LinkageIndex` (prepared state). (ADR-0001.)

## Evaluation lives outside the core

The metric report types live in {doc}`denselinkage.metrics </api/metrics>`, not
in `core`, because nothing in the contract depends on them — the dependency rule
applied to the evaluation layer. (ADR-0002.)

## A curated, two-tier surface

The public API is deliberately shaped (see the {doc}`API reference </api/index>`):
the prelude for everyday use, capability submodules for the adapters, and
`denselinkage.core` for the contract you implement against. The surface is
frozen and evolves by **extend, never modify** — new optional fields, sibling
types, and new ports are additive; existing signatures and field types are
stable. (ADR-0003.)

## The full design record

Architecture Decision Records and the development decision log are kept **in the
repository**, version-controlled alongside the code rather than published here:

- [Architecture Decision Records](https://github.com/caalvaro/denselinkage/tree/main/docs/ADRs)
  — spec→artifact (0001), evaluation-out-of-core (0002), pre-freeze ratification (0003).
- [Development docs](https://github.com/caalvaro/denselinkage/tree/main/docs/development)
  — the roadmap, the freeze gate, the decision log, and the Resolvi
  reference-architecture conformance analysis.
