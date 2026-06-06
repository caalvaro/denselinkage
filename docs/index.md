# denselinkage

**Record linkage and deduplication — dense blocking, optional LLM matching, and
evaluation built in.**

Find the records that refer to the same real-world entity, across **two datasets**
(linkage) or within **one** (deduplication). denselinkage shrinks the all-pairs
comparison with embedding-based **blocking**, decides each candidate with a
pluggable **matcher** — a similarity threshold or an LLM — then clusters and scores
the result. The core runs on a **dependency-free** numpy + pandas stack; FAISS,
sentence-transformers, and LangChain are opt-in extras you add when you need them.

```python
from denselinkage import DenseLinker, LabeledPairs, Source
from denselinkage.metrics import linkage_metrics

linker = DenseLinker.with_defaults()        # sensible embedder + index + matcher
result = linker.link(Source(df_a, id_column="id"), Source(df_b, id_column="id"))
result.to_frame()                           # left_id, right_id, similarity, match, ...
linkage_metrics(result, gold=LabeledPairs.from_pairs([("A1", "B1")]))
```

## Highlights

- **Dependency-free core** — `pip install denselinkage` is just numpy + pandas; the
  heavy backends are opt-in and a CI job proves they never leak into the core.
- **Swap any stage** — the embedder, vector index, and matcher are independent
  ports: lexical → semantic, brute-force → FAISS, threshold → LLM, no pipeline
  rewrite.
- **End to end** — block → match → cluster → evaluate, with linkage, blocking, and
  clustering (B³) **metrics included**.
- **Immutable & typed** — single `link` / `dedupe` / `match_pairs` calls, a frozen
  1.0 contract, strict `mypy`, and 100% branch coverage.

New here? The [tutorial](getting-started/tutorial) links two tables stage by stage;
the [Semantic + LLM guide](guide/semantic-llm) covers the production stack.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Get started
:link: getting-started/quickstart
:link-type: doc

Install, then link two tables in under five minutes.
:::

:::{grid-item-card} {octicon}`book` User guide
:link: guide/linking
:link-type: doc

Task recipes: linking, deduplication, the semantic + LLM stack, and evaluation.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: api/index
:link-type: doc

The curated surface — prelude, adapters, and the `core` contract.
:::

:::{grid-item-card} {octicon}`workflow` Architecture
:link: architecture
:link-type: doc

Ports and adapters, the spec→artifact law, and the design record.
:::

::::

```{toctree}
:hidden:
:caption: Getting started

getting-started/installation
getting-started/quickstart
getting-started/tutorial
getting-started/concepts
```

```{toctree}
:hidden:
:caption: User guide

guide/linking
guide/deduplication
guide/match-pairs
guide/semantic-llm
guide/custom-components
guide/reusing-an-index
guide/evaluation
guide/choosing-components
```

```{toctree}
:hidden:
:caption: Reference

api/index
architecture
```
