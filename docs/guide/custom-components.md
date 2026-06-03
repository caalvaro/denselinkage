# Custom components

Every pluggable stage is a {class}`typing.Protocol` port in
{doc}`denselinkage.core.ports </api/contract>`. To add your own embedder,
serializer, index, or matcher, implement the port and inject it — nothing else
in the library needs to change.

## Implement a port

First-party adapters **subclass the port explicitly**. That is ordinary Python,
and it lets the type checker verify your implementation is complete:

```python
from collections.abc import Mapping, Sequence
from typing import Any

from denselinkage.core.ports import Serializer


class FieldJoinSerializer(Serializer):           # explicit: mypy checks completeness
    def __init__(self, fields: Sequence[str], sep: str = " | ") -> None:
        self._fields = list(fields)
        self._sep = sep

    def serialize(self, record: Mapping[str, Any]) -> str:
        return self._sep.join(str(record.get(f, "")) for f in self._fields)
```

Third-party code may also conform **structurally** — matching the method
signatures without importing or subclassing anything — because the ports are
Protocols.

## Inject it

A component is just a constructor argument to
{class}`~denselinkage.linkage.DenseLinker` (or to
{class}`~denselinkage.blocking.DenseBlocker`):

```python
linker = DenseLinker(
    blocker=DenseBlocker(embedder=MyEmbedder(), vector_index=NumpyFlatIndex()),
    matcher=ThresholdMatcher(threshold=0.55),
)
left = Source(df, id_column="id", serializer=FieldJoinSerializer(["name", "city"]))
```

## Full example

A custom char-n-gram embedder and a custom serializer, wired on the
dependency-free stack (`NumpyFlatIndex` + `ThresholdMatcher`):

```{literalinclude} ../../examples/03_custom_embedder.py
:language: python
:caption: examples/03_custom_embedder.py
```

## Which port do I implement?

| Port | Responsibility | Reference adapter |
| --- | --- | --- |
| {class}`~denselinkage.core.ports.Serializer` | row → text | {class}`~denselinkage.serializing.TemplateSerializer` |
| {class}`~denselinkage.core.ports.Embedder` | text → vectors | {class}`~denselinkage.embedding.HashedNGramEmbedder` |
| {class}`~denselinkage.core.ports.VectorIndex` | build a searchable index | {class}`~denselinkage.indexing.NumpyFlatIndex` |
| {class}`~denselinkage.core.ports.Matcher` | candidate pairs → decisions | {class}`~denselinkage.matching.ThresholdMatcher` |
| {class}`~denselinkage.core.ports.Filter` | prune candidates before matching | {class}`~denselinkage.filtering.SimilarityThresholdFilter` |
| {class}`~denselinkage.core.ports.Clusterer` | matches → entity clusters | {class}`~denselinkage.clustering.ConnectedComponentsClusterer` |

See the full port set and method signatures in the
{doc}`contract reference </api/contract>`.
