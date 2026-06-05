# denselinkage

Record linkage with **dense blocking** — text embeddings to generate candidate
pairs, then a matcher (threshold or LLM) to decide them. One call, no
`fit`/`predict`, no mutation.

```python
from denselinkage import DenseLinker, LabeledPairs, Source
from denselinkage.metrics import linkage_metrics

linker = DenseLinker.with_defaults()        # sensible embedder + index + matcher
result = linker.link(Source(df_a, id_column="id"), Source(df_b, id_column="id"))
result.to_frame()                           # left_id, right_id, similarity, match, ...
linkage_metrics(result, gold=LabeledPairs.from_pairs([("A1", "B1")]))
```

:::{admonition} Beta
:class: note

The dependency-free core is implemented and **runs** — `link` / `dedupe` /
`match_pairs`, connected-components clustering, and the linkage / blocking /
clustering metrics, all on numpy + pandas. The heavy extras (FAISS,
sentence-transformers, LangChain) are **experimental this release**: their
adapters are declared but raise `NotImplementedError`.
:::

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

Task recipes: linking, deduplication, custom components, evaluation.
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
