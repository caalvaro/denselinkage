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
