# Linking two tables

`link(left, right)` matches records *across* two datasets. The
[quickstart](../getting-started/quickstart) uses `with_defaults()`; this page
shows full component control — assembling the blocker and matcher yourself.

## Assemble the components

A {class}`~denselinkage.linkage.DenseLinker` is pure configuration. You inject a
{class}`~denselinkage.core.ports.Blocker` (here a
{class}`~denselinkage.blocking.DenseBlocker` composing an embedder and a vector
index) and a {class}`~denselinkage.core.ports.Matcher`:

```python
from denselinkage import DenseLinker, Source, TemplateSerializer
from denselinkage.blocking import DenseBlocker
from denselinkage.embedding import SentenceTransformerEmbedder
from denselinkage.indexing import FaissFlatIndex
from denselinkage.matching import LangChainMatcher

linker = DenseLinker(
    blocker=DenseBlocker(
        embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
        vector_index=FaissFlatIndex(),
        similarity_threshold=0.80,   # retrieve top_k, then keep >= this
        top_k=5,
    ),
    matcher=LangChainMatcher(llm=...),
)
```

The embedder and the vector index are **independent** ports — swap the index
(NumPy ↔ FAISS) without touching the embedder, and vice versa.

## Reconcile differing schemas

Each `Source` carries its own serializer, so two tables with different column
names share one template via `column_mapping`:

```python
template = "Name: {name}, City: {city}"
left  = Source(df_a, id_column="id_a", serializer=TemplateSerializer(template))
right = Source(df_b, id_column="id_b", serializer=TemplateSerializer(
    template, column_mapping={"company_name": "name", "headquarters": "city"}))

result = linker.link(left, right)   # one call, no mutation
result.to_frame()                   # left_id, right_id, similarity, match, confidence, reason
```

## Full example

```{literalinclude} ../../examples/01_end_to_end_linkage.py
:language: python
:caption: examples/01_end_to_end_linkage.py
```

:::{note}
This example uses the heavy adapters (`SentenceTransformerEmbedder`,
`FaissFlatIndex`, `LangChainMatcher`) behind the `[all]` extra and a live LLM, so
it needs those extras and an `OPENAI_API_KEY` to run. It is type-checked and
compiled in CI but not executed there. See [Semantic + LLM matching](semantic-llm)
for a walk-through of the knobs.
:::

## See also

- [Reusing an index](reusing-an-index) — amortize the build across many queries.
- [Evaluation](evaluation) — score the run with `linkage_metrics`.
- [Choosing components](choosing-components) — lexical vs semantic vs LLM.
