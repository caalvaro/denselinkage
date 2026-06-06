# denselinkage

[![CI](https://github.com/caalvaro/denselinkage/actions/workflows/ci.yml/badge.svg)](https://github.com/caalvaro/denselinkage/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/denselinkage.svg)](https://pypi.org/project/denselinkage/)
[![Python versions](https://img.shields.io/pypi/pyversions/denselinkage.svg)](https://pypi.org/project/denselinkage/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://caalvaro.github.io/denselinkage/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Record linkage with dense blocking using text embeddings and LLM matching.

> **Status — stable (1.0).** The dependency-free core is implemented and **runs**:
> `link` / `dedupe` / `match_pairs`, connected-components clustering, and the
> linkage / blocking / clustering metrics — all on numpy + pandas. The heavy
> extras (FAISS, sentence-transformers, LangChain) are **implemented** too —
> install the matching extra to use `FaissFlatIndex`,
> `SentenceTransformerEmbedder`, or `LangChainMatcher`.

## Usage

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

> The `[faiss]`, `[sentence-transformers]`, and `[langchain]` extras enable the
> heavy adapters (`FaissFlatIndex`, `SentenceTransformerEmbedder`,
> `LangChainMatcher`); the dependency-free core runs without them.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details. CI runs lint, format,
strict mypy, and tests on Python 3.10–3.13.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE) © 2026 Alvaro
