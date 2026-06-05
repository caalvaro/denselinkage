# API reference

denselinkage exposes a **curated, two-tier surface**. The shape of the import
*is* the documentation: the top level is the orchestration entry points, the
submodules are the pluggable adapters, and `denselinkage.core` is the frozen
contract they all share.

## The prelude — `denselinkage`

The symbols a typical script needs, importable straight from the package root:

```python
from denselinkage import DenseLinker, Source, TemplateSerializer
```

| Symbol | Kind | Documented in |
| --- | --- | --- |
| {class}`~denselinkage.linkage.DenseLinker` | orchestration config | [Orchestration](orchestration) |
| {class}`~denselinkage.linkage.LinkageIndex` | prepared state | [Orchestration](orchestration) |
| {class}`~denselinkage.core.models.Source` | input value object | [Contract](contract) |
| {class}`~denselinkage.core.results.LinkageResult` | output | [Contract](contract) |
| {class}`~denselinkage.core.results.ClusteringResult` | output | [Contract](contract) |
| {class}`~denselinkage.core.results.LabeledPairs` | ground truth | [Contract](contract) |
| {class}`~denselinkage.metrics.LinkageMetrics` | report | [Metrics](metrics) |
| {class}`~denselinkage.metrics.BlockingMetrics` | report | [Metrics](metrics) |
| {class}`~denselinkage.metrics.ClusteringMetrics` | report | [Metrics](metrics) |
| {class}`~denselinkage.serializing.TemplateSerializer` | reference serializer | [Components](components) |
| {class}`~denselinkage.serializing.FieldwiseSerializer` | reference serializer | [Components](components) |
| {class}`~denselinkage.serializing.WholeRowSerializer` | reference serializer | [Components](components) |
| {func}`~denselinkage.clustering.connected_components` | convenience function | [Components](components) |
| {func}`~denselinkage.linkage.candidate_pairs_from_frame` | convenience function | [Orchestration](orchestration) |

## Capability submodules

The pluggable adapters live in their own modules, imported only when you reach
past the defaults — one module per pipeline stage:

```python
from denselinkage.embedding import SentenceTransformerEmbedder
from denselinkage.matching import LangChainMatcher
```

`serializing` · `embedding` · `indexing` · `blocking` · `filtering` ·
`matching` · `clustering` · `metrics` — all reference adapters are catalogued
under [Components](components) and [Metrics](metrics).

## The contract — `denselinkage.core`

The ports, models, results, and errors every adapter shares. You import from
here when **writing** a component rather than using one:

```python
from denselinkage.core.ports import Embedder, Serializer
```

See [Contract](contract) for the full port set and the
[custom components guide](../guide/custom-components) for how to implement one.

```{toctree}
:hidden:

orchestration
components
metrics
contract
```
