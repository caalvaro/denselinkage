# Key concepts

The mental model behind the five-line quickstart. Five ideas cover the whole API.

:::{seealso}
Coming from another entity-resolution tool? The
[end-to-end tutorial](tutorial) maps dense blocking onto the classic
blocking → comparison → classification → clustering pipeline, with diagrams.
:::

## 1. `Source` — data bound to its schema

A {class}`~denselinkage.core.models.Source` pairs a DataFrame with the column
that identifies each row and an optional **serializer** that turns a row into the
text the embedder sees:

```python
Source(df, id_column="id", serializer=TemplateSerializer("Name: {name}, City: {city}"))
```

The schema travels *with the frame*, not with the linker — so two sources with
different column names can be linked by giving each its own serializer (and a
`column_mapping` to reconcile names). Materialization validates the frame and
raises the {doc}`error taxonomy </api/contract>`
({class}`~denselinkage.core.errors.UnknownIdColumn`,
{class}`~denselinkage.core.errors.EmptySource`,
{class}`~denselinkage.core.errors.DuplicateRecordId`).

## 2. Three verbs

One {class}`~denselinkage.linkage.DenseLinker` config, three tasks — the task is
the method name:

| Verb | Use it for |
| --- | --- |
| `link(left, right)` | match records *across* two datasets |
| `dedupe(source)` | find duplicates *within* one dataset (self-pairs suppressed) |
| `match_pairs(pairs)` | decide candidate pairs you blocked elsewhere |

## 3. Results carry provenance

{class}`~denselinkage.core.results.LinkageResult` keeps decisions and failures in
**separate channels**. `to_frame()` returns the decided pairs
(`left_id, right_id, similarity, match, confidence, reason`); pairs the matcher
could not decide are {class}`~denselinkage.core.models.MatchError`s in
`result.errors`, counted but excluded from precision/recall — never merged into
the results and never silently dropped.

## 4. Design-time config vs runtime state

`DenseLinker` is an **immutable value object**: pure configuration, no data,
nothing fitted. The prepared, per-dataset state is a separate object:

```python
idx = linker.index(left)     # build once  -> LinkageIndex
idx.query(right_a)           # reuse for many query sets
idx.query(right_b)
```

This mirrors the internal **spec → artifact** law: a
{class}`~denselinkage.core.ports.Blocker` is a stateless *spec* whose `build()`
returns a fresh, immutable {class}`~denselinkage.core.ports.BlockingIndex`
*artifact*. Specs are reusable; state lives only in artifacts. See
{doc}`/architecture`.

## 5. Two tiers of errors

- A plain `ValueError` signals **API misuse** — a bug in the calling code (e.g.
  a linker built without a blocker).
- {class}`~denselinkage.core.errors.DenseLinkageError` and its taxonomy signal
  **data / runtime** failures you might reasonably catch.

They are deliberately separate, so an `except DenseLinkageError` guarding data
handling never swallows a programming mistake.

---

Imports follow the same two-tier shape — prelude, adapters, contract. See the
{doc}`API reference </api/index>`.
