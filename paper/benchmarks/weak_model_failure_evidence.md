# Failure-contract evidence: natural failures, real verdicts, and the arithmetic

This note consolidates the empirical evidence for the failure-contract claims in
§Evaluation. Every number regenerates from the scripts named below. The point of
the new (weak-model) runs is to ground the claim in **failures that occur on
their own** — nothing is injected in R1/R2.

## Setup

- Local runtime: **Ollama 0.30.6**, `langchain-ollama` + `langchain-core 1.4.7`,
  Python 3.14, single **RTX 5060** (models run **100% on GPU**; ~0.39 s/pair at
  concurrency 4 on the structured path).
- Weak models: **`qwen2.5:0.5b`** (481 MB) and **`llama3.2:1b`**, pulled via
  `ollama pull`. Strong reference model: hosted **`gemini-2.5-flash[-lite]`**
  (cached verdicts; no key needed to reproduce the replays).
- Data: standard **DBLP–ACM** (`|A|=2616`, `|B|=2294`, 2 220 gold matches).
- Sampling for R1/R2: dependency-free blocking (`HashedNGramEmbedder`
  n_features=1024 ngram=3 + `NumpyFlatIndex`, top_k=5) → 11 470 candidates;
  seeded representative random sample of **N=1000** (seed 20260613),
  **172 gold pairs in sample**. The structured run and the free-text run use the
  **same seed → same 1000 pairs**, so they are directly comparable.

## R1 — Structured output prevents the format-failure class (NATURAL, no injection)

Script: `local_weak_failure_hunt.py` (`OLLAMA_MODEL=<m> N_BIG=1000`). The real
`LangChainMatcher` path (structured output + retry → typed `MatchError`).

| Matcher (structured output) | pairs | natural `MatchError` | failure rate | decision F1 (P / R) |
|---|---:|---:|---:|---|
| gemini-2.5-flash-lite (cached) | 4 000 | 0 | 0.0% | — |
| llama3.2:1b (local) | 1 000 | 0 | 0.0% | 0.298 (0.208 / 0.529) |
| qwen2.5:0.5b (local) | 1 000 | 0 | 0.0% | 0.292 (0.173 / 0.954) |

Even a 0.5B model never emits a malformed verdict: Ollama's constrained decoding
guarantees schema-valid output. Weakness shows up as bad **decisions**, not
failures. So on the structured path, silent == excluded (nothing to exclude).
Files: `local_weak_failure_hunt_qwen2.5_0.5b.json`,
`local_weak_failure_hunt_llama3.2_1b.json`, `gemini_failure_hunt.json`.

## R2 — The brittle free-text convention collapses on weak models (NATURAL)

Script: `local_competitor_convention.py`. Same pairs, **free-text** output (no
structured output), each raw response scored three ways: exact `== "True"`
(pyJedAI), tolerant parse with unparseable→non-match (silent), and the
denselinkage stance (unparseable excluded + coverage).

| Model | N | non-canonical | unparseable | F1 exact (pyJedAI) | F1 silent-robust | F1 typed-excluded (coverage) |
|---|---:|---:|---:|---:|---:|---:|
| gemini-2.5-flash-lite (cached) | 200 | 36.5% | ~0% | 0.879 | 0.880 | — |
| qwen2.5:0.5b | 1 000 | **100%** | 0.0% | **0.000** | 0.297 | 0.297 (100%) |
| llama3.2:1b | 1 000 | **100%** | **98.7%** | **0.000** | **0.044** | **0.615 (1.3%)** |

Truncation control: rerunning `llama3.2:1b` at **num_predict=256** (4×) over 300
pairs gives unparseable **99.3%** — the high rate is genuine task-failure (the
model echoes record text: `"The correct answer is:\n\nA: <record text>"`), not a
token cutoff. File: `local_competitor_convention_llama3.2_1b_np256.json`.

Reading: on a strong model the exact rule survives by luck (non-canonical answers
fall on true non-matches). On weak models it returns **F1 = 0** (no response is
literally `"True"`). For `llama3.2:1b`, silent-folding the 98.7% unparseable into
non-matches reports **F1 = 0.044** (a false "broken matcher"); the typed contract
reports **coverage = 1.3%, F1 = 0.615 on decided** — the honest "this model can't
do the task here." Same model, same pairs:

| `llama3.2:1b`, same 1000 pairs | reported F1 |
|---|---:|
| structured output (denselinkage) | **0.30** (0 failures) |
| free-text + exact rule (pyJedAI) | 0.00 |
| free-text + silent-robust | 0.044 |
| free-text + typed-excluded (cov 1.3%) | 0.615 |

denselinkage's design choices are what tame the failure: structured output
removes the format-failure class (R1), and typed accounting turns a misleading
number into an honest one (R2).

## R3 — Operational-failure tail on REAL verdicts (injection = mechanism test)

Script: `modern_pipeline_experiment.py sweep` → `modern_results.json`. Real
`gemini-2.5-flash` decisions on a balanced 200-pair sample, failures injected
**through the real retry → `MatchError` path**, uniform vs. concentrated on the
hardest (lowest-similarity) pairs. Blocking PC@{1,5,10,20} =
{0.970, 0.991, 0.994, 0.997} (semantic stack, full benchmark).

| f | regime | coverage | F1 excl. | F1 silent | recall silent |
|---:|---|---:|---:|---:|---:|
| 0.05 | uniform | 0.95 | 0.888 | 0.868 | 0.95 |
| 0.05 | hard | 0.95 | 0.865 | 0.826 | 0.90 |
| 0.10 | uniform | 0.90 | 0.896 | 0.853 | 0.90 |
| 0.10 | hard | 0.90 | 0.851 | 0.769 | 0.80 |
| 0.20 | uniform | 0.80 | 0.904 | 0.812 | 0.80 |
| 0.20 | **hard** | 0.80 | **0.811** | **0.638** | **0.60** |

The "hard" regime also shows the **excluded** F1 degrading (0.904→0.811),
demonstrating that under non-random failures the excluded estimate is itself
biased — which is why coverage must be reported beside it. Injection here is a
mechanism test on **real** decisions, not a prevalence estimate.

## R4 — Closed-form arithmetic at the dependency-free operating point

Script: `failure_accounting_experiment.py` → `results.json`. Operating point
P=0.855, R=0.972, F1=0.910; uniform injection, 20 trials per f.

| f | F1 excl. | F1 silent | ΔF1 | recall silent |
|---:|---:|---:|---:|---:|
| 0.02 | 0.910 | 0.901 | 0.009 | 0.952 |
| 0.05 | 0.910 | 0.888 | 0.022 | 0.923 |
| 0.10 | 0.910 | 0.864 | 0.046 | 0.872 |
| 0.20 | 0.910 | 0.813 | 0.097 | 0.775 |

## Narrative for §Eval

1. Structured output (FR2) prevents format failures by construction — 0 across
   the capability ladder (R1). Natural.
2. The brittle/free-text convention competitors use does not — it collapses on
   weak models (R2). Natural.
3. When residual operational failures occur (refusals/timeouts/rate-limits at
   scale, per the literature), silent scoring confounds F1 while the typed
   contract excludes them and reports coverage (R3 on real verdicts; R4 the
   arithmetic).

This supports **softening** the "matchers fail in routine operation" framing: the
honest, evidence-matched claim is that the failure mode is real and severe for
the brittle convention and for weak/loaded models, and that structured output +
typed accounting is exactly what neutralizes it.

## Reproduce

```
# R4 (no deps beyond numpy/pandas)
python paper/benchmarks/failure_accounting_experiment.py
# R3 (needs langchain-core; replays cached real verdicts, no API key)
python paper/benchmarks/modern_pipeline_experiment.py sweep
# R1 / R2 (need Ollama + langchain-ollama + a pulled weak model)
OLLAMA_MODEL=qwen2.5:0.5b N_BIG=1000 python paper/benchmarks/local_weak_failure_hunt.py
OLLAMA_MODEL=llama3.2:1b   N_BIG=1000 python paper/benchmarks/local_weak_failure_hunt.py
OLLAMA_MODEL=qwen2.5:0.5b N_BIG=1000 python paper/benchmarks/local_competitor_convention.py
OLLAMA_MODEL=llama3.2:1b   N_BIG=1000 python paper/benchmarks/local_competitor_convention.py
```
