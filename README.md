# denselinkage

[![CI](https://github.com/caalvaro/denselinkage/actions/workflows/ci.yml/badge.svg)](https://github.com/caalvaro/denselinkage/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Record linkage with dense blocking using text embeddings and LLM matching.

> **Status — structure stage.** The public types, ports and signatures are
> defined; method bodies are `...` placeholders. The snippet below is the
> intended API (the spec the `examples/` demonstrate), not yet runnable.
> Implementation lands incrementally against this frozen contract.

## Intended API

```python
from denselinkage import DenseLinker, Source, TemplateSerializer
from denselinkage.core.results import LabeledPairs
from denselinkage.metrics import linkage_metrics

linker = DenseLinker.with_defaults()  # picks a sensible embedder/index/matcher
left  = Source(df_a, id_column="id_a", serializer=TemplateSerializer("Name: {name}, City: {city}"))
right = Source(df_b, id_column="id_b", serializer=TemplateSerializer(
    "Name: {name}, City: {city}", column_mapping={"company_name": "name", "headquarters": "city"}))

result  = linker.link(left, right)               # one call, no fit/predict, no mutation
metrics = linkage_metrics(result, gold=LabeledPairs.from_pairs([("A1", "B1")]))
result.to_frame()  # left_id, right_id, match, confidence, reason, similarity
```

Deduplicate one dataset with `linker.dedupe(src)`; reuse an index with
`idx = linker.index(left); idx.query(right)`. See [`examples/`](examples/) —
`00_quickstart.py` is the shortest path, `01_end_to_end_linkage.py` shows full
component control.

## Install

```bash
pip install denselinkage                       # core (numpy, pandas)
pip install "denselinkage[faiss]"              # + FAISS vector index
pip install "denselinkage[sentence-transformers]"
pip install "denselinkage[langchain]"          # + LLM matcher
pip install "denselinkage[all]"
```

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details. CI runs lint, format,
strict mypy, and tests on Python 3.10–3.13.

## License

[MIT](LICENSE) © 2026 Alvaro
