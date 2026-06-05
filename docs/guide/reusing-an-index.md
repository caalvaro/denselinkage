# Reusing an index

`link` and `dedupe` build everything in one call. When you query the *same*
reference dataset many times — a parameter sweep, streaming queries, or matching
several frames against one master table — build the index **once** and reuse it.

```python
idx = linker.index(left)     # build the blocking index once -> LinkageIndex
idx.query(right_a)           # -> LinkageResult
idx.query(right_b)           # reuses the built index; no re-embedding of `left`
```

{meth}`~denselinkage.linkage.DenseLinker.index` returns a
{class}`~denselinkage.linkage.LinkageIndex` — the prepared, per-dataset state
separated from the linker's configuration.

## Why this is safe to reuse

The index is an **immutable artifact**. Internally a
{class}`~denselinkage.core.ports.Blocker` is a stateless *spec* whose `build()`
produces a fresh {class}`~denselinkage.core.ports.BlockingIndex` holding the
indexed vectors; the spec mutates neither itself nor its inputs. So one built
index answers many queries with no risk of cross-query state leaking, and the
same `DenseLinker` can build many indexes over different datasets.

This is the **spec → artifact** law that runs through the whole library; see
{doc}`/architecture`.

## Sweeping query-time parameters

`top_k` and `similarity_threshold` are query-time parameters with sensible
defaults from the blocker spec, so a threshold/`top_k` sweep reuses one built
index instead of rebuilding it — the expensive embedding work happens once.

## Persisting a built index

Building the index embeds the whole reference set — the expensive step. Save the
built {class}`~denselinkage.linkage.LinkageIndex` and reload it later to query new
data with **no re-embedding**:

```python
idx = linker.index(left)
idx.save("master_index")            # writes a vectors.npy + meta.json bundle

from denselinkage import LinkageIndex
from denselinkage.embedding import HashedNGramEmbedder
from denselinkage.matching import ThresholdMatcher

reloaded = LinkageIndex.load(
    "master_index",
    embedder=HashedNGramEmbedder(n_features=1024, ngram=3),  # must match the saved model
    matcher=ThresholdMatcher(threshold=0.5),
)
reloaded.query(right)               # reuses the stored embeddings of `left`
```

The store records the embedder's `model_id` and `embedding_dim` as **provenance**:
if the embedder you pass to `load` doesn't match, it raises
{class}`~denselinkage.core.errors.IncompatibleStore` — a persisted index cannot be
queried with a different embedding model. The matcher is not persisted (supply any
one at load). Persistence currently covers the dependency-free reference stack;
other backends raise `NotImplementedError`.
