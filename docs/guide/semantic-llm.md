# Semantic + LLM matching

The dependency-free stack ([Linking](linking)) blocks with **lexical** n-gram
similarity and decides with a similarity **threshold**. The headline stack swaps
in three heavy adapters for what lexical matching misses — semantic renames
(*Google* / *Alphabet*, which share no characters) and pairs that need a judgment
call:

| Stage | Dependency-free | Heavy adapter | Extra |
|---|---|---|---|
| Embed | `HashedNGramEmbedder` | `SentenceTransformerEmbedder` | `[sentence-transformers]` |
| Index | `NumpyFlatIndex` | `FaissFlatIndex` | `[faiss]` |
| Match | `ThresholdMatcher` | `LangChainMatcher` | `[langchain]` |

They are **drop-in** behind the same ports — the linker, serializers, metrics and
clustering are unchanged. Each swap is independent: take semantic embeddings
without an LLM matcher, or an LLM matcher over lexical blocking.

## Install

```bash
pip install "denselinkage[all]"      # all three extras
# or pick what you need:
pip install "denselinkage[sentence-transformers]"
pip install "denselinkage[faiss]"
pip install "denselinkage[langchain]"
```

The core still imports without any of them — `import denselinkage` pulls in no
heavy backend. A missing extra raises a `ModuleNotFoundError` naming the exact
`pip install` to run, and only when you construct the adapter.

## Semantic embeddings — `SentenceTransformerEmbedder`

```python
from denselinkage.embedding import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
```

The argument is any [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html)
checkpoint. `all-MiniLM-L6-v2` (384-dim) is a small, fast default; larger models
(e.g. `all-mpnet-base-v2`, 768-dim) trade speed for accuracy. The model loads
eagerly, so a bad name fails immediately.

`encode` returns **L2-normalized** float32 vectors, so the inner product equals
cosine — identical similarity semantics to `HashedNGramEmbedder`. A
`similarity_threshold` you tuned on the lexical stack therefore keeps its meaning
here.

:::{note}
The first construction downloads the model (cached afterwards under
`~/.cache/huggingface`). Embedding is CPU-bound; pass `batch_size=` and
`show_progress=True` to `encode` for large reference sets.
:::

## Exact vector index — `FaissFlatIndex`

```python
from denselinkage.indexing import FaissFlatIndex

vector_index = FaissFlatIndex()   # exact inner-product (IndexFlatIP) search
```

A drop-in for `NumpyFlatIndex` behind the `VectorIndex` port — same neighbours,
and scores equal to within float32 tolerance (the differential test
`test_faiss_matches_numpy_neighbours` pins them together), because it uses FAISS's
`IndexFlatIP` and so reports cosine for the normalized vectors above.

Both backends are exhaustive, so the swap changes neither the candidate set nor
recall. What it changes is what `search` allocates on the Python side.
`NumpySearchableIndex.search` first materialises the whole query-by-index score
matrix, one float32 array of shape `(n_queries, n_indexed)`, which is
`4 * n_queries * n_indexed` bytes in a single unchunked allocation: at 100,000
queries against 100,000 indexed records, 4.0e10 bytes, or 37.3 GiB.
`FaissSearchableIndex.search` receives back only the per-query top-_k_ scores and
indices, arrays of shape `(n_queries, k)` with `k = min(top_k, n_indexed)`. What
FAISS allocates internally to produce them is not verifiable from this repository,
and neither backend is timed here, so no speed claim is made either way; a
measured envelope is [#56](https://github.com/caalvaro/denselinkage/issues/56).

`DenseBlocker` encodes and searches the whole query table in one call, so
`n_queries` in that formula is the entire query side of a `link` or `dedupe`.

:::{note}
Incremental `extended()` and persistence of a FAISS-backed index are out of scope
for v1 — the [Reference Store](reusing-an-index) persists the numpy stack only.
:::

## LLM matching — `LangChainMatcher`

The matcher replaces the similarity gate with a language-model judgment.

```python
from langchain_openai import ChatOpenAI
from denselinkage.matching import LangChainMatcher, RetryPolicy

matcher = LangChainMatcher(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0, seed=42),
    prompt=(
        "Are these two records the same real-world entity?\n"
        "Record A: {record_a}\n"
        "Record B: {record_b}"
    ),
    retry=RetryPolicy(max_retries=3),
    max_concurrency=8,
)
```

**The `llm` is injected** — any LangChain chat model works (`ChatOpenAI`,
`ChatAnthropic`, …); model and operational config live on the model object, not on
the matcher. Set the provider's credentials in the environment (e.g.
`OPENAI_API_KEY`).

**Prompt contract.** The `prompt` carries *only the semantic question* and **must**
reference `{record_a}` and `{record_b}` — the matcher fills them with the two
records' serialized text. Do **not** ask for an output format: the matcher binds
structured output and returns typed `MatchDecision`s, so you never parse text and
a brittle "answer YES or NO" instruction is neither needed nor wanted. Each
decision carries `is_match` plus optional `confidence` / `rationale` when the model
supplies them.

**Operational knobs.**
- `retry=RetryPolicy(max_retries=…, backoff_seconds=…)` — per-pair retries on a
  transient backend error.
- `max_concurrency=…` — how many pairs are sent to the model in parallel.

**Failures are soft.** A pair the matcher cannot decide after its retries becomes a
`MatchError` in `result.errors` — never an exception into the batch, so one bad
call cannot abort a run. Errored pairs are excluded from precision/recall and
counted as `LinkageMetrics.n_errors`:

```python
result = linker.link(left, right)
for err in result.errors:
    print(err.reason)
```

:::{tip}
**Cost & determinism.** Every surviving candidate pair is one LLM call, so cost
scales with `top_k` × queries — tune `top_k` / `similarity_threshold` on the
blocker (or pre-filter with `SimilarityThresholdFilter`) to keep the matcher's
workload down. Use `temperature=0.0` (and a `seed` where supported) for
reproducible decisions.
:::

## Putting it together

```python
from denselinkage import DenseLinker
from denselinkage.blocking import DenseBlocker

blocker = DenseBlocker(
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
    vector_index=FaissFlatIndex(),
    similarity_threshold=0.80,   # retrieve top_k, then keep >= this
    top_k=5,
)
linker = DenseLinker(blocker=blocker, matcher=matcher)
result = linker.link(left, right)   # the same call as the lexical stack
```

The full runnable assembly — with `Source` / serializers and evaluation — is
[`examples/01_end_to_end_linkage.py`](https://github.com/caalvaro/denselinkage/tree/main/examples/01_end_to_end_linkage.py);
the deduplication shape is
[`examples/02_deduplication.py`](https://github.com/caalvaro/denselinkage/tree/main/examples/02_deduplication.py).
Both type-check and compile in CI but need the extras + an `OPENAI_API_KEY` to run.

## See also

- [Choosing components](choosing-components) — when each fork is worth its cost.
- [Linking](linking) — the orchestration verbs and the dependency-free path.
- [Reusing an index](reusing-an-index) — persist embeddings to skip re-encoding.
