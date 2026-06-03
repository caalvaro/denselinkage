# Quickstart

The shortest real path: two schema-aligned tables, the default stack, one call.
This example **runs today** — we'll build it up a few lines at a time, then show
the complete script.

## Step 1 — Imports

```python
import pandas as pd

from denselinkage import DenseLinker, LabeledPairs, Source
from denselinkage.metrics import linkage_metrics
```

Everything you need for a basic run comes from the package root
(the [prelude](../api/index)) plus the `metrics` module. You only reach into
submodules when you swap components — see [Choosing components](../guide/choosing-components).

## Step 2 — The two tables

```python
df_a = pd.DataFrame({
    "id":   ["A1", "A2", "A3"],
    "name": ["Apple Inc", "Microsoft Corp", "Google LLC"],
    "city": ["Cupertino", "Redmond", "Mountain View"],
})
df_b = pd.DataFrame({
    "id":   ["B1", "B2", "B3"],
    "name": ["Apple Incorporated", "Microsoft", "Google"],
    "city": ["Cupertino", "Redmond", "Mountain View"],
})
```

Two ordinary DataFrames describing the same companies with surface differences
(`Apple Inc` vs `Apple Incorporated`, `Microsoft Corp` vs `Microsoft`). The
columns line up here, so no serializer template is needed — when they don't,
each [`Source`](../api/contract) carries its own serializer and a
`column_mapping` (see [Linking two tables](../guide/linking)).

## Step 3 — Ground truth (optional)

```python
gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2"), ("A3", "B3")])
```

{class}`~denselinkage.core.results.LabeledPairs` is the known-correct matches.
It's only used for *scoring* in Step 7 — linking itself needs no labels.

## Step 4 — Configure the linker and wrap the inputs

```python
linker = DenseLinker.with_defaults()   # picks a sensible embedder + index + matcher
left  = Source(df_a, id_column="id")   # serializer=None -> whole-row default
right = Source(df_b, id_column="id")
```

{meth}`DenseLinker.with_defaults() <denselinkage.linkage.DenseLinker>` wires the
dependency-free reference stack (a {class}`~denselinkage.embedding.HashedNGramEmbedder`
and {class}`~denselinkage.indexing.NumpyFlatIndex` behind a
{class}`~denselinkage.blocking.DenseBlocker`, plus a
{class}`~denselinkage.matching.ThresholdMatcher`). The linker is **immutable
config** — no data, nothing fitted. Each {class}`~denselinkage.core.models.Source`
binds a frame to its id column; the schema travels with the frame, not the
linker (the design-time / runtime split in [Key concepts](concepts)).

## Step 5 — Link

```python
result = linker.link(left, right)   # -> LinkageResult
```

One call — no `fit`, no `predict`, no mutation of the linker or the inputs.
Under the hood this embeds both tables, retrieves nearest neighbours as
candidate pairs, and decides each one. You get back a
{class}`~denselinkage.core.results.LinkageResult` holding decisions and any
failures in separate channels.

## Step 6 — Inspect the results

```python
print(result.to_frame())
```

```text
  left_id right_id  similarity  match confidence reason
0      A1       B1    0.762443   True       None   None
1      A2       B1    0.188329  False       None   None
2      A3       B1    0.151794  False       None   None
3      A2       B2    0.833908   True       None   None
4      A3       B2    0.183309  False       None   None
5      A1       B2    0.160128  False       None   None
6      A3       B3    0.864126   True       None   None
7      A1       B3    0.188713  False       None   None
8      A2       B3    0.178685  False       None   None
```

`to_frame()` is a **fixed schema**, independent of your input column names:
`left_id, right_id, similarity, match, confidence, reason`. Each row is one
*decided* candidate pair — the blocker surfaced nine (every query record against
its nearest neighbours), and the matcher flagged the three real matches on the
diagonal. `confidence` / `reason` are `None` because the threshold matcher
doesn't produce them (an LLM matcher would). Pairs the matcher *couldn't* decide
would be in `result.errors`, never in this frame.

## Step 7 — Score it

```python
metrics = linkage_metrics(result, gold=gold)   # -> LinkageMetrics
print(f"P/R/F1: {metrics.precision:.3f} {metrics.recall:.3f} {metrics.f1:.3f}")
```

```text
P/R/F1: 1.000 1.000 1.000
```

{func}`~denselinkage.metrics.linkage_metrics` scores the run against your gold.
`metrics.n_errors` reports undecided pairs separately (they're excluded from
precision/recall, never silently dropped). For a `dedupe` run, pass
`directed=False` — see [Evaluation](../guide/evaluation).

## The complete script

All seven steps, as the runnable example file:

```{literalinclude} ../../examples/00_quickstart.py
:language: python
:caption: examples/00_quickstart.py
```

## The default stack is lexical

`HashedNGramEmbedder` is character-n-gram feature hashing: it recovers
abbreviations, punctuation, and typos (`Apple Inc` / `Apple Incorporated`) but
not semantic renames such as `Google` → `Alphabet`. For semantic matching, swap
in a {class}`~denselinkage.embedding.SentenceTransformerEmbedder` — see
[Choosing components](../guide/choosing-components).

## Next steps

- [End-to-end tutorial](tutorial) — the full pipeline explained with diagrams,
  for readers who know other entity-resolution methods.
- [Key concepts](concepts) — the mental model behind these seven steps.
- [Linking two tables](../guide/linking) — full component control.
- [Custom components](../guide/custom-components) — implement your own port.
