# Deduplication

`dedupe(source)` finds duplicates *within* one dataset. It links the dataset
against itself and suppresses self-pairs internally — there is no
`suppress_self_pairs` knob. The same {class}`~denselinkage.linkage.DenseLinker`
config works for both `link` and `dedupe`; only the method name changes.

```python
src = Source(df, id_column="id", serializer=TemplateSerializer("{name} — {city}"))
result = linker.dedupe(src)         # -> LinkageResult; self-pairs suppressed
```

## From pairs to entities

Pairwise matches are transitive evidence, not final groups. Collapse them into
entity clusters with {func}`~denselinkage.clustering.connected_components`:

```python
from denselinkage import connected_components

clusters = connected_components(result)   # -> ClusteringResult
clusters.to_frame()
```

:::{tip}
A record that produced no candidate pair is absent from `result`, so it is
missing from the clustering — and `clustering_metrics` would then score only the
records that *did* appear, silently inflating B³ recall. Pass the full id set so
unmatched records count as singletons and B³ is complete:

```python
ids = src.frame[src.id_column].astype(str)
clusters = connected_components(result, all_record_ids=ids)
```
:::

:::{warning}
`connected_components` is **transitive**: if A matches B and B matches C, all
three land in one cluster even if A and C were never matched directly. With a
noisy matcher this can snowball into oversized clusters — watch for cluster
quality (recall ≫ precision) using [clustering metrics](evaluation).
:::

## Evaluate against order-insensitive gold

For dedup, a pair is unordered — `("1","2")` and `("2","1")` are the same. The
metrics canonicalize each pair, so pass your gold once and let the evaluator
handle direction (see the `directed` flag in [Evaluation](evaluation)).

## Full example

Runnable on the dependency-free stack (`dedupe → connected_components → B³`):

```{literalinclude} ../../examples/04_dedupe.py
:language: python
:caption: examples/04_dedupe.py
```

:::{note}
For the production semantic + LLM shape, see
[`examples/02_deduplication.py`](https://github.com/caalvaro/denselinkage/tree/main/examples/02_deduplication.py)
— it uses the FAISS / sentence-transformers / LangChain adapters, so it needs
those extras (`pip install "denselinkage[all]"`) and an `OPENAI_API_KEY` to run.
:::
