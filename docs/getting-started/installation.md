# Installation

denselinkage requires **Python 3.10+**. The core depends only on NumPy and
pandas; every heavy ML backend is an optional extra, so the base install stays
light.

```bash
pip install denselinkage                        # core: numpy + pandas
```

## Optional extras

Install an extra only when you reach past the dependency-free default stack:

| Extra | Adds | Enables |
| --- | --- | --- |
| `faiss` | `faiss-cpu` | {class}`~denselinkage.indexing.FaissFlatIndex` — scalable vector search |
| `sentence-transformers` | `sentence-transformers` | {class}`~denselinkage.embedding.SentenceTransformerEmbedder` — semantic embeddings |
| `langchain` | `langchain-core`, `langchain-openai` | {class}`~denselinkage.matching.LangChainMatcher` — LLM matching |
| `all` | all of the above | — |

```bash
pip install "denselinkage[faiss]"
pip install "denselinkage[sentence-transformers]"
pip install "denselinkage[langchain]"
pip install "denselinkage[all]"
```

The adapters import their backend lazily, so a missing extra surfaces only if you
actually construct that adapter — never at `import denselinkage`.

## Development

The project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

To build these docs locally, install the `docs` extra and run the Sphinx build
(it treats warnings as errors, matching CI):

```bash
pip install -e ".[docs]"
cd docs && make html          # or: make.bat html   (Windows)
```

The rendered site lands in `docs/_build/html/`. See
[CONTRIBUTING.md](https://github.com/caalvaro/denselinkage/blob/main/CONTRIBUTING.md)
for the full contributor guide.
