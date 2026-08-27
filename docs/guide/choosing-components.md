# Choosing components

denselinkage ships a dependency-free default stack and heavier adapters for when
you outgrow it. This page is the decision guide.

## The default stack

`DenseLinker.with_defaults()` wires:

| Stage | Default | Extra |
| --- | --- | --- |
| Embedder | {class}`~denselinkage.embedding.HashedNGramEmbedder` (lexical) | none |
| Vector index | {class}`~denselinkage.indexing.NumpyFlatIndex` | none |
| Blocker | {class}`~denselinkage.blocking.DenseBlocker` | none |
| Matcher | {class}`~denselinkage.matching.ThresholdMatcher` | none |

It runs with only NumPy and pandas — good for small-to-medium data and for
matches recoverable from surface text.

## Lexical vs semantic embeddings

- **Lexical** ({class}`~denselinkage.embedding.HashedNGramEmbedder`): character
  n-gram hashing. Recovers abbreviations, punctuation, and typos
  (`Apple Inc` / `Apple Incorporated`). Fast, dependency-free. **Misses**
  semantic renames (`Google` → `Alphabet`).
- **Semantic** ({class}`~denselinkage.embedding.SentenceTransformerEmbedder`,
  extra `[sentence-transformers]`): sentence embeddings that capture meaning.
  Reach for it when matches need world knowledge rather than shared characters.

## Vector index backends

{class}`~denselinkage.indexing.NumpyFlatIndex` is exact, dependency-free and
brute-force. {class}`~denselinkage.indexing.FaissFlatIndex` (extra `[faiss]`) is
brute-force too and returns the same neighbours, behind the same
{class}`~denselinkage.core.ports.VectorIndex` port. The two differ in what
`search` allocates: the numpy artifact materialises the whole
`(n_queries, n_indexed)` score matrix, the FAISS artifact receives back only each
query's top-_k_ scores and indices. See
[Semantic + LLM matching](semantic-llm) for the byte formula.

## Threshold vs LLM matching

- {class}`~denselinkage.matching.ThresholdMatcher` (default): decides on the
  blocker's similarity score. Zero cost, fully deterministic. Use it as the
  second gate above the blocker's retrieval threshold.
- {class}`~denselinkage.matching.LangChainMatcher` (extra `[langchain]`): an LLM
  reads each pair and returns a typed decision with a rationale. Use it for hard
  pairs where similarity alone is ambiguous; pair it with a
  {class}`~denselinkage.matching.RetryPolicy`.

## Rule of thumb

Start with `with_defaults()`. Swap the **embedder** first if you are missing
semantic matches, the **index** if the full numpy score matrix does not fit in
memory, and the **matcher** last if similarity cannot separate true from false
pairs. Every swap is one constructor argument — see
[Custom components](custom-components).
