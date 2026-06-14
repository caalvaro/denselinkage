# Evidence trail for the open-source ER tool landscape (paper Table 1)

Each claim in the paper's landscape table was verified against the named
tool's current release, source code, or issue tracker on 2026-06-12. Versions:
Splink 4.0.16, dedupe 3.0.3, recordlinkage 0.16, pyJedAI 0.3.6,
py_entitymatching 0.4.2, Libem 0.0.26. Ditto and DeepMatcher are research
codebases without packaged releases.

Legend (paper table): Dense = embedding-based blocking; LLM = LLM-based
matching; Fail. = a matcher-failure outcome that is typed and excluded from
quality metrics; Light = dependency-light core with heavy stacks as optional
extras; Typed = a public contract shipped to type checkers (`py.typed`).

## Splink (moj-analytical-services/splink, v4.0.16, MIT)
- Method: Fellegi–Sunter probabilistic linkage, SQL generation over DuckDB
  (default)/Spark/Athena/Postgres.
- Dense blocking: **partial / open request.** GitHub issue #1011 "[FEAT]
  Support for embedding-based similarity functions" still open
  (github.com/moj-analytical-services/splink/issues/1011); workaround only via
  user SQL (discussion #1042).
- LLM matching: none.
- Light: **yes** — core deps are pandas/duckdb/sqlglot/altair/jinja2/numpy/
  igraph; Spark/Athena/Postgres are optional extras (pyproject.toml).
- Typed: **yes** — `py.typed` present in the package tree.

## dedupe (dedupeio/dedupe, v3.0.3, MIT)
- Method: active learning + learned string-comparison functions (README cites
  Bilenko's dissertation); learned blocking rules.
- Dense blocking: none. LLM matching: none. Light: no (C extensions). Typed: no.

## recordlinkage (J535D165/recordlinkage, v0.16)
- Method: blocking + sorted-neighbourhood indexing; string/numeric/date
  comparators; supervised (sklearn) + unsupervised ECM (Fellegi–Sunter).
- Dense blocking: none. LLM matching: none. Light: **yes** (pandas/numpy/
  scipy/sklearn). Typed: no.

## py_entitymatching / Magellan (anhaidgroup, v0.4.2, BSD-3)
- Method: feature-based supervised ML pipeline (sklearn classifiers).
- Dense blocking: none. LLM matching: none (deep learning lived in the
  separate, now-unmaintained deepmatcher). Light: no. Typed: no.

## Ditto (megagonlabs/ditto) and DeepMatcher (anhaidgroup/deepmatcher)
- Research code, not on PyPI / unmaintained. Matching only (both defer
  blocking to Magellan). Pre-generative-LLM (PLM fine-tuning / RNN-attention).

## pyJedAI (AI-team-UoA/pyJedAI, v0.3.6, Apache-2.0)
- Dense blocking: **yes, native** — `EmbeddingsNNBlockBuilding` in
  `vector_based_blocking.py`: gensim / transformer / sentence-transformer
  embeddings, FAISS `IndexFlatL2` nearest-neighbour search.
- LLM matching: **experimental** — `llm_matching.py` (added 2025-08-06,
  commits e3a064c / a1727fc), classes `AbstrackLLMMatching` + `OllamaMatching`.
  Ollama-only; the match decision is `resp['message']['content'] == 'True'`
  (exact string compare), so any other response — refusal, hedge, transport
  error — is scored as a non-match and flows into the library's own
  `evaluation.py` F1. No typed per-pair error outcome; failures are not
  excluded from metrics.
- Fail.: no. Light: **no** — PyPI 0.3.6 `requires_dist` makes `transformers`,
  `sentence-transformers`, `faiss-cpu`, `ray`, `ollama`, `nltk`, `shapely`,
  `networkx`, `valentine`, `py-stringcompare` all mandatory; only `dev` and
  `with-gensim` extras. Typed: **no** (`py.typed` absent from `src/pyjedai/`).

## Libem (abcsys/libem, v0.0.26)
- LLM-EM compound toolchain (GPT-4o/-mini, Llama3, etc.). Has a typed
  `ModelTimedoutException` at the adapter level, but in batch matching missing
  or malformed answers are padded with `Output(answer='no')`
  (`libem/match/function.py`), i.e. failures are coerced to non-match, not kept
  as a per-pair error channel excluded from metrics. Fail.: no.

## Cross-cutting
- A typed per-pair LLM-failure outcome that is excluded from quality metrics
  was found in **none** of the examined systems.
- No examined system combines (typed frozen contract) + (light core, heavy
  optional extras) + (typed LLM failure channel).
