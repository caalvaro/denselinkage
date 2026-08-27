# End-to-end tutorial: record linkage with dense blocking

**Who this is for.** You already know entity resolution — the
blocking → comparison → classification → clustering pipeline, pair completeness
vs. reduction ratio, Fellegi–Sunter, B³. This tutorial maps **dense blocking**
onto that mental model so you can recognise what denselinkage does and drive it
in a few lines. We link two company tables end to end, then deduplicate one.

If you just want to run something first, do the
[5-minute quickstart](quickstart); for the API mental model, see
[Key concepts](concepts). This page is the bridge: the *why* and the *how*,
stage by stage.

## The pipeline you already know

denselinkage does not invent a new pipeline. It is the **same four stages** you
have built before — it only changes how blocking and comparison are done.

```{mermaid}
flowchart LR
    R[Records] --> B["Blocking<br/><i>dense: embed → ANN top-k</i>"]
    B --> C["Comparison<br/><i>cosine similarity from the embeddings</i>"]
    C --> D["Classification<br/><i>ThresholdMatcher or LLM</i>"]
    D --> E["Clustering<br/><i>connected components</i>"]
```

Here is the direct translation from the vocabulary you know to the package:

| ER stage | Classic approach | denselinkage | API surface | Quality metric |
| --- | --- | --- | --- | --- |
| Representation | comparison attributes / blocking-key construction | serialise each row to one text string | {class}`~denselinkage.core.ports.Serializer` | — |
| **Blocking / indexing** | standard blocking, sorted neighbourhood, canopy, q-gram | embed records, retrieve top-_k_ nearest in a vector index | {class}`~denselinkage.core.ports.Embedder` + {class}`~denselinkage.core.ports.VectorIndex` (via {class}`~denselinkage.blocking.DenseBlocker`) | pair completeness@k; reduction ratio is planned in [#53](https://github.com/caalvaro/denselinkage/issues/53) |
| Comparison | per-field similarity → comparison vector | cosine similarity, carried on each pair | {attr}`CandidatePair.similarity_score <denselinkage.core.models.CandidatePair>` | — |
| **Classification** | Fellegi–Sunter, threshold, supervised ML | similarity threshold **or** an LLM | {class}`~denselinkage.core.ports.Matcher` ({class}`~denselinkage.matching.ThresholdMatcher`, {class}`~denselinkage.matching.LangChainMatcher`) | precision / recall / F1 |
| Clustering / merge | transitive closure, hierarchical | connected components (pluggable) | {func}`~denselinkage.clustering.connected_components` / {class}`~denselinkage.core.ports.Clusterer` | B³ precision / recall / F1 |

If you can read that table, you already know the library. The rest of this page
is detail.

## What dense blocking changes (and what it doesn't)

Traditional blocking buckets records by a **discrete key** — Soundex of a
surname, a q-gram, a ZIP prefix — and compares only within a bucket. It is fast
and exact, but the key is hand-crafted, and a record whose key is wrong (a typo
in the surname, a missing ZIP) silently never meets its true match.

Dense blocking replaces the key with a **vector**: serialise the record to text,
embed it, and retrieve the top-_k_ nearest neighbours from an approximate
nearest-neighbour (ANN) index. "Same bucket" becomes "close in vector space."

```{mermaid}
flowchart TB
    subgraph trad["Traditional blocking"]
        direction LR
        t1[Records] --> t2["Blocking key<br/>(Soundex, q-gram, ZIP)"]
        t2 --> t3["Exact-match buckets"]
        t3 --> t4["Pairs within a bucket"]
    end
    subgraph dense["Dense blocking"]
        direction LR
        d1[Records] --> d2["Embedding<br/>(lexical or semantic)"]
        d2 --> d3["Vector index (ANN)"]
        d3 --> d4["Top-k nearest as pairs"]
    end
```

**What carries over** — the recall/cost trade-off is the same one you tune
today. `top_k` and `similarity_threshold` play the role your blocking key's
selectivity played: raise them for higher pair completeness, lower them for a
tighter reduction ratio. You still measure the blocker with pair completeness.

**What changes** — there are no keys to design, and "closeness" is fuzzy by
construction, so abbreviations, punctuation, and (with semantic embeddings)
rewordings cluster together without bespoke rules. The cost is that retrieval is
approximate: a true match can fall outside the top-_k_, exactly as a bad key can
put it in the wrong bucket. **denselinkage counts that honestly** — recall in
{func}`~denselinkage.metrics.linkage_metrics` charges every missed gold pair as a
false negative, including those the blocker never surfaced, so your reported
recall is end-to-end, not conditional on blocking.

One more recognition point: in dense blocking the **comparison vector collapses
to a single number** — the embedding cosine similarity — carried on every
{class}`~denselinkage.core.models.CandidatePair`. The richer per-field
comparison you may be used to moves *into* the classifier when that classifier
is an LLM reading the record text.

## The data flow, end to end

```{mermaid}
flowchart TB
    subgraph build["Index the reference table — built once"]
        direction TB
        FA[DataFrame A] --> SA["Serializer<br/>row → text"]
        SA --> EA["Embedder<br/>text → vector"]
        EA --> VB["VectorIndex.build()"]
        VB --> IDX[("SearchableIndex<br/>immutable artifact")]
    end
    subgraph query["Query — per link() call"]
        direction TB
        FB[DataFrame B] --> SB["Serializer<br/>row → text"]
        SB --> EB["Embedder<br/>text → vector"]
    end
    EB -- "top-k nearest" --> IDX
    IDX --> CP["CandidatePairs<br/>(carry similarity)"]
    CP --> MM["Matcher<br/>threshold / LLM"]
    MM --> RES["LinkageResult<br/>decisions + errors"]
    RES --> CC["connected_components()"]
    CC --> CL[ClusteringResult]
```

The dashed split is deliberate: the reference table is embedded and indexed
**once** into an immutable artifact, which then answers many query batches. That
is the design-time/runtime separation — {class}`~denselinkage.linkage.DenseLinker`
(config) vs {class}`~denselinkage.linkage.LinkageIndex` (prepared state) — covered
in [Key concepts](concepts) and [Reusing an index](../guide/reusing-an-index).

## Build it: link two tables, stage by stage

We will link two tables of companies with **different schemas**. The runnable
path below uses the dependency-free **lexical** stack and is exactly what the
[quickstart](quickstart) runs; afterwards we swap in the **semantic + LLM**
stack for production.

### Stage 0 — the data

```python
import pandas as pd

df_a = pd.DataFrame({
    "id":   ["A1", "A2", "A3"],
    "name": ["Apple Inc", "Microsoft Corp", "Google LLC"],
    "city": ["Cupertino", "Redmond", "Mountain View"],
})
df_b = pd.DataFrame({
    "id":   ["B1", "B2", "B3"],
    "name": ["Apple Incorporated", "Microsoft", "Google"],
    "city": ["Cupertino", "Redmond", "Mountain View"],
})
```

### Stage 1 — representation: rows → text

**ER analog:** choosing comparison attributes / building the blocking
representation. Instead of selecting fields and a key, you serialise each row to
one string. A {class}`~denselinkage.core.models.Source` binds a frame to its id
column and a serializer; with `serializer=None` it uses the whole-row default.

```python
from denselinkage import Source

left  = Source(df_a, id_column="id")   # whole-row serializer by default
right = Source(df_b, id_column="id")
```

When schemas differ, give each source its own serializer and a `column_mapping`
to a shared template — see [Linking two tables](../guide/linking). The linker
never learns either schema; the schema travels with the frame.

### Stage 2 — blocking representation: text → vectors

**ER analog:** this is where a discrete blocking key would be computed. Here an
{class}`~denselinkage.core.ports.Embedder` turns text into a vector.

- **Lexical** ({class}`~denselinkage.embedding.HashedNGramEmbedder`): character
  n-gram feature hashing. Recovers `Apple Inc` ≈ `Apple Incorporated`,
  `Microsoft Corp` ≈ `Microsoft`. Dependency-free. Misses semantic rewrites
  (`Google` → `Alphabet`).
- **Semantic** ({class}`~denselinkage.embedding.SentenceTransformerEmbedder`):
  sentence embeddings that capture meaning, behind the `[sentence-transformers]`
  extra.

### Stage 3 — blocking: index and retrieve top-_k_

**ER analog:** indexing and candidate generation. A
{class}`~denselinkage.core.ports.VectorIndex` is a *spec*; its `build()` produces
an immutable {class}`~denselinkage.core.ports.SearchableIndex` over the reference
vectors. {class}`~denselinkage.blocking.DenseBlocker` composes the embedder and
the index and exposes `top_k` / `similarity_threshold` — your recall knobs. Each
surviving neighbour becomes a {class}`~denselinkage.core.models.CandidatePair`
carrying its cosine similarity.

```python
from denselinkage import DenseLinker
from denselinkage.blocking import DenseBlocker
from denselinkage.embedding import HashedNGramEmbedder
from denselinkage.indexing import NumpyFlatIndex
from denselinkage.matching import ThresholdMatcher

# This is exactly what DenseLinker.with_defaults() assembles for you:
linker = DenseLinker(
    blocker=DenseBlocker(
        embedder=HashedNGramEmbedder(n_features=1024, ngram=3),
        vector_index=NumpyFlatIndex(),
        similarity_threshold=0.0,   # keep all top_k neighbours; let the matcher gate
        top_k=5,
    ),
    matcher=ThresholdMatcher(threshold=0.5),
)
```

:::{tip}
**Measure the blocker, like any ER pipeline.** The blocker is your recall
ceiling. denselinkage exposes {func}`~denselinkage.metrics.pair_completeness_at_k`
and {func}`~denselinkage.metrics.blocking_metrics` (PC@k) over candidate pairs
for exactly this check — sweep `top_k` and watch pair completeness before you
spend anything on the classifier.
:::

### Stage 4 — classification: decide each pair

**ER analog:** the classifier. {class}`~denselinkage.core.ports.Matcher` takes
candidate pairs and returns one outcome each:

- {class}`~denselinkage.matching.ThresholdMatcher` gates on the carried
  similarity — the dense-blocking analog of a single Fellegi–Sunter threshold.
- {class}`~denselinkage.matching.LangChainMatcher` (extra `[langchain]`) asks an
  LLM to read the two records and return a typed decision with a rationale —
  the modern replacement for a hand-built comparison-vector classifier.

A pair the matcher cannot decide becomes a
{class}`~denselinkage.core.models.MatchError`, **not** an exception and **not** a
false match — it is parked in a separate channel (next stage).

### Stage 5 — read the result

```python
result = linker.link(left, right)   # -> LinkageResult; one call, no mutation
print(result.to_frame())            # left_id, right_id, similarity, match, confidence, reason
```

**ER analog:** the linked set with match status — but
{class}`~denselinkage.core.results.LinkageResult` keeps **decisions and failures
in separate channels**. `to_frame()` is decided pairs only; undecided pairs are
`MatchError`s in `result.errors`, counted but never merged in. There is no eager
"golden record" merge: provenance is preserved, which is the audit-friendly
posture you want in regulated linkage.

### Stage 6 — evaluate

```python
from denselinkage import LabeledPairs
from denselinkage.metrics import linkage_metrics

gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2"), ("A3", "B3")])
m = linkage_metrics(result, gold=gold)        # -> LinkageMetrics
print(f"P/R/F1: {m.precision:.3f} {m.recall:.3f} {m.f1:.3f}")
print(f"undecided (errors): {m.n_errors}")
```

On the lexical stack above this prints **P/R/F1 = 1.000** — the gold here is
lexically recoverable on purpose. For `dedupe`, pass `directed=False` so
`("1","2")` and `("2","1")` count as the same pair (see
[Evaluation](../guide/evaluation)).

### Stage 7 — deduplicate and cluster

The within-one-table variant is the same config, a different verb. Pairwise
matches are transitive *evidence*; collapse them into entities with
{func}`~denselinkage.clustering.connected_components`, then score with B³:

```python
from denselinkage import connected_components
from denselinkage.metrics import clustering_metrics

result   = linker.dedupe(src)              # self-pairs suppressed internally
clusters = connected_components(result)    # -> ClusteringResult (transitive closure)
cm = clustering_metrics(clusters, gold=gold)
print(f"B3 P/R/F1: {cm.b3_precision:.3f} {cm.b3_recall:.3f} {cm.b3_f1:.3f}")
```

:::{warning}
Connected components is **transitive**: A~B and B~C put A, B, C in one cluster
even if A and C never matched. With a noisy classifier this snowballs into
mega-clusters — the classic dedup failure mode. Watch B³ recall ≫ precision.
:::

:::{note}
The entire dependency-free evaluation path shown here — `linkage_metrics`,
`blocking_metrics` / `pair_completeness_at_k`, and B³ `clustering_metrics` —
runs today on numpy + pandas. The semantic / FAISS / LLM components (example
`01`) are implemented too, behind their extras (`[all]`) and a live LLM — see
[Semantic + LLM matching](../guide/semantic-llm).
:::

## The runnable example and the production target

The lexical, dependency-free version you just walked through is the quickstart —
**run it today**:

```{literalinclude} ../../examples/00_quickstart.py
:language: python
:caption: examples/00_quickstart.py — runs on numpy + pandas alone
```

The production assembly swaps in semantic embeddings, a FAISS index, and an LLM
matcher (extra `[all]`). Same pipeline, heavier components:

```{literalinclude} ../../examples/01_end_to_end_linkage.py
:language: python
:caption: examples/01_end_to_end_linkage.py — semantic + FAISS + LLM
```

## Tuning and evaluation, per stage

| Stage | Knob | Measure with |
| --- | --- | --- |
| Blocking | `top_k`, `similarity_threshold` on {class}`~denselinkage.blocking.DenseBlocker` | {func}`~denselinkage.metrics.pair_completeness_at_k` (PC@k); reduction ratio is planned in [#53](https://github.com/caalvaro/denselinkage/issues/53) |
| Classification | `ThresholdMatcher(threshold=...)` or the LLM prompt | {func}`~denselinkage.metrics.linkage_metrics` (P/R/F1) |
| Clustering | the {class}`~denselinkage.core.ports.Clusterer` strategy | {func}`~denselinkage.metrics.clustering_metrics` (B³) |

The discipline is the one you already follow: fix blocking recall first (a match
the blocker drops is unrecoverable downstream), then tune the classifier for
precision, then check clustering for over-merging.

## Where denselinkage fits, versus tools you know

| Tool | Blocking | Classifier |
| --- | --- | --- |
| Splink | blocking rules / keys | Fellegi–Sunter (EM-trained) |
| dedupe | learned predicates | logistic regression (active learning) |
| Magellan / recordlinkage | keys + feature engineering | supervised ML on comparison vectors |
| **denselinkage** | **dense retrieval (embeddings + ANN)** | **threshold or LLM, pluggable** |

Bring your ER instincts: swap the hand-tuned blocking key for dense retrieval,
and pick a threshold or an LLM as the classifier. denselinkage is **single-node
by design** (batched encoding; no Spark/Flink layer) and **contract-first** —
every stage is a {class}`typing.Protocol` port you can replace, which is the
subject of [Custom components](../guide/custom-components) and
{doc}`/architecture`.

## Next steps

- [Linking two tables](../guide/linking) — full component control and schema mapping.
- [Deduplication](../guide/deduplication) — the within-one-table variant in depth.
- [Evaluation](../guide/evaluation) — pair completeness, P/R/F1, and B³.
- [Custom components](../guide/custom-components) — implement a port of your own.
