# denselinkage

[![PyPI](https://img.shields.io/pypi/v/denselinkage.svg)](https://pypi.org/project/denselinkage/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://caalvaro.github.io/denselinkage/)
[![CI](https://github.com/caalvaro/denselinkage/actions/workflows/ci.yml/badge.svg)](https://github.com/caalvaro/denselinkage/actions/workflows/ci.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/denselinkage.svg)](https://pypi.org/project/denselinkage/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Record linkage and deduplication for Python — dense blocking, optional LLM matching, and evaluation built in.**

`denselinkage` finds the records that refer to the same real-world entity, whether
they live in **two datasets** (record *linkage*) or **one** (*deduplication*). It
shrinks the impossible all-pairs comparison down to a plausible few with
embedding-based **blocking**, decides each candidate with a pluggable **matcher** —
a fast similarity threshold or a large language model — then clusters and scores
the result.

The core runs on **numpy + pandas alone**. FAISS, sentence-transformers, and
LangChain are optional extras you reach for when you need approximate-nearest-
neighbour search at scale, semantic embeddings, or LLM-based matching — `import
denselinkage` pulls in none of them until you ask.

## Highlights

- 🪶 **Dependency-free core** — `pip install denselinkage` is just numpy + pandas.
  The heavy ML backends are opt-in extras, and the import graph proves it: CI fails
  if a backend ever leaks into the core.
- 🔁 **Swap any stage** — the embedder, vector index, and matcher are independent
  components behind small `Protocol`s. Go from lexical → semantic, brute-force →
  FAISS, threshold → LLM without rewriting your pipeline.
- 📦 **End to end** — block → match → cluster → evaluate, with linkage, blocking,
  and clustering (B³) **metrics included**.
- 🧊 **Immutable by design** — `link` / `dedupe` / `match_pairs` are single calls
  with no hidden `fit`/`predict` state. Build a reference index once and reuse it.
- 🧪 **Typed, tested, stable** — strict `mypy`, a shipped `py.typed` marker,
  **100% branch coverage**, and a frozen 1.0 API (evolution is *extend, never
  modify*).

## Installation

```bash
pip install denselinkage                           # core — numpy + pandas only
```

Add extras when you need them (or `[all]` for everything):

```bash
pip install "denselinkage[sentence-transformers]"  # semantic embeddings
pip install "denselinkage[faiss]"                  # exact (flat) FAISS index
pip install "denselinkage[langchain]"              # LLM matcher
pip install "denselinkage[all]"
```

Requires Python 3.10+.

## Quickstart

Link two tables of companies with messy, inconsistent names — no configuration,
one call:

```python
import pandas as pd
from denselinkage import DenseLinker, LabeledPairs, Source
from denselinkage.metrics import linkage_metrics

left = pd.DataFrame({
    "id":   ["A1", "A2", "A3"],
    "name": ["Apple Inc", "Microsoft Corp", "Google LLC"],
    "city": ["Cupertino", "Redmond", "Mountain View"],
})
right = pd.DataFrame({
    "id":   ["B1", "B2", "B3"],
    "name": ["Apple Incorporated", "Microsoft", "Google"],
    "city": ["Cupertino", "Redmond", "Mountain View"],
})

linker = DenseLinker.with_defaults()         # lexical stack: embed → index → threshold
result = linker.link(                         # one call — no fit/predict, no mutation
    Source(left, id_column="id"),
    Source(right, id_column="id"),
)

print(result.to_frame().query("match"))       # the decided matches, as a DataFrame
gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2"), ("A3", "B3")])
m = linkage_metrics(result, gold=gold)
print(f"precision={m.precision:.2f} recall={m.recall:.2f} f1={m.f1:.2f}")
```

```text
  left_id right_id  similarity  match confidence reason
0      A1       B1    0.762443   True       None   None
3      A2       B2    0.833908   True       None   None
6      A3       B3    0.864126   True       None   None
precision=1.00 recall=1.00 f1=1.00
```

`with_defaults()` wires the dependency-free **lexical** stack — character n-gram
embeddings, brute-force nearest-neighbour search, and a similarity threshold. It
recovers abbreviations, punctuation, and typos (`Apple Inc` ↔ `Apple Incorporated`)
out of the box.

## How it works

denselinkage is a four-stage pipeline, and every stage is a swappable component:

```text
 Sources ──► Block ──────► Match ──────► Cluster ──────► Evaluate
            (embed +      (threshold    (connected      (P/R/F1,
             top-k NN)     or LLM)        components)     B³, …)
```

1. **Block** — embed each record and retrieve its top-k nearest neighbours, turning
   an `N × M` comparison into a handful of candidate pairs.
2. **Match** — decide each candidate. `ThresholdMatcher` gates on similarity;
   `LangChainMatcher` asks an LLM and returns a typed decision.
3. **Cluster** — group the matches into entities with transitive
   `connected_components`.
4. **Evaluate** — score against gold labels with linkage, blocking, or clustering
   (B³) metrics.

Three verbs cover the common shapes — **`link`** (two datasets), **`dedupe`** (one
dataset against itself), and **`match_pairs`** (you already have candidate pairs).
`index()` builds a reusable reference index, so you embed once and query many times.

## Scaling up: semantic + LLM matching

The lexical default is fast and free, but it only sees *characters* — it can't tell
that *Google* and *Alphabet* are the same company. Swap in the heavy adapters for
**meaning** (semantic embeddings), **scale** (FAISS), and **judgment** (an LLM), all
behind the same ports:

| Stage | Lexical (default) | Semantic + LLM |
|------:|-------------------|----------------|
| Embed | `HashedNGramEmbedder` | `SentenceTransformerEmbedder` · `[sentence-transformers]` |
| Index | `NumpyFlatIndex` | `FaissFlatIndex` · `[faiss]` |
| Match | `ThresholdMatcher` | `LangChainMatcher` · `[langchain]` |
| Catches | typos, abbreviations | + semantic renames, + judgment calls |

```python
from denselinkage import DenseLinker
from denselinkage.blocking import DenseBlocker
from denselinkage.embedding import SentenceTransformerEmbedder
from denselinkage.indexing import FaissFlatIndex
from denselinkage.matching import LangChainMatcher
from langchain_openai import ChatOpenAI

linker = DenseLinker(
    blocker=DenseBlocker(
        embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
        vector_index=FaissFlatIndex(),
        top_k=5, similarity_threshold=0.6,
    ),
    matcher=LangChainMatcher(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        prompt="Are these the same entity?\nA: {record_a}\nB: {record_b}",
    ),
)
result = linker.link(left, right)   # the call is unchanged
```

Because the score is cosine on both stacks, a `similarity_threshold` tuned on the
lexical stack keeps its meaning here. See the
[Semantic + LLM guide](https://caalvaro.github.io/denselinkage/guide/semantic-llm.html)
for model selection, the prompt contract, retries, and cost.

## Deduplicate and cluster

```python
from denselinkage import DenseLinker, Source, connected_components

# df: one table that may contain duplicate records, with an "id" column
result   = DenseLinker.with_defaults().dedupe(Source(df, id_column="id"))
clusters = connected_components(result)        # transitive grouping → entities
print(clusters.to_frame())                     # record_id, cluster_id
```

`dedupe` links a dataset against itself and suppresses self-pairs internally.
Clustering is transitive (A~B, B~C ⇒ one cluster), so a noisy matcher can
over-merge — watch for B³ recall ≫ precision.

## Evaluation

Metrics are first-class, split by what they measure:

- **Linkage** — `linkage_metrics` → precision / recall / F1 over matched pairs
  (undecidable pairs are surfaced as errors and counted separately, never mixed in).
- **Blocking** — `blocking_metrics` / `pair_completeness_at_k` → did blocking even
  surface the true pairs?
- **Clustering** — `clustering_metrics` → B³ (Bagga–Baldwin) precision / recall / F1
  over the entity clusters.

Plus `tune_threshold` for a P/R/F1 sweep and `mine_hard_negatives` for contrastive
training material.

## Design

denselinkage is **contract-first** (hexagonal / ports-and-adapters). Domain logic
talks to small `typing.Protocol`s — `Embedder`, `VectorIndex`, `Matcher`, … — and
concrete adapters plug in behind them. Two consequences worth knowing:

- **The dependency cut is structural.** Heavy backends import lazily, inside the
  methods that use them; a CI job asserts `import denselinkage` pulls in no FAISS /
  torch / LangChain.
- **The 1.0 contract is frozen.** Signatures and field types won't change under
  you; the library evolves by *adding* (an optional field, a sibling type, a new
  classmethod), never by modifying. Stateful components follow **spec → artifact**:
  a stateless spec's `build(...)` returns an immutable, fitted artifact.

See the
[architecture overview](https://caalvaro.github.io/denselinkage/architecture.html)
for the full picture.

## Documentation

📖 **[Full documentation →](https://caalvaro.github.io/denselinkage/)**

- [Tutorial](https://caalvaro.github.io/denselinkage/getting-started/tutorial.html)
  — link two tables stage by stage.
- [Semantic + LLM matching](https://caalvaro.github.io/denselinkage/guide/semantic-llm.html)
  and [Choosing components](https://caalvaro.github.io/denselinkage/guide/choosing-components.html).
- [API reference](https://caalvaro.github.io/denselinkage/api/index.html).

Runnable scripts live in [`examples/`](examples/) — `00_quickstart.py` is the
shortest path; `01`/`02` show the full semantic + LLM assembly; and
`05_failure_accounting.py` shows the typed failure contract — why a matcher
failure is excluded from the metric rather than scored as a silent non-match.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

CI runs lint, format, strict mypy, and the test suite on Python 3.10–3.13, with a
separate job for the optional adapters. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Citing

If you use denselinkage in your research, please cite it — see
[`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE) © 2026 Alvaro
