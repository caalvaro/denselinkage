# Roadmap — phases & exit criteria

> Part of the [denselinkage development docs](./README.md). See
> [freeze-gate.md](./freeze-gate.md) for what "frozen" / the A0.5 gate means.

- **A0** *(done)* — typed core skeleton: models, ports, results, linker,
  metrics, serializers, reference adapters; dependency-cut CI; `[train]`
  namespace reserved.
- **A0.5** *(contract hardening before freeze)* — close the gaps
  that would become breaking post-freeze: D1/D2/D3, the hard-failure
  exception taxonomy, the named Source→Record seam, `BlockingMetrics`
  constructor shape, kwarg-`gold` consistency, examples wired into CI
  (lint+format+type+compile), one Python-version story, conventions aligned.
  **Exit:** every DoD check green and pasted verbatim; contract re-frozen.
- **A1 — dependency-free beta (`1.0.0b1`)** *(done)* — fill the frozen
  contract's bodies for the no-extras surface: `link` / `dedupe` / `match_pairs`,
  connected-components clustering, and the linkage / blocking / clustering
  metrics + filtering. The heavy extras (faiss / sentence-transformers /
  langchain) are deferred to A2 (below). Branch coverage + `fail_under = 100`
  enforced. See [ADR-0004](../ADRs/0004-dependency-free-beta.md).
- **A2 — heavy adapters** *(done)* — fill the four deferred adapter bodies
  (`SentenceTransformerEmbedder`, `FaissFlatIndex` / `FaissSearchableIndex`,
  `LangChainMatcher`) behind their optional extras, so the headline dense-semantic
  + LLM stack runs. Additive and lazy-imported (the dependency cut holds);
  incremental `extended` stays deferred. A dedicated adapter-coverage CI job gates
  the adapter modules at 100% (the dependency-free gate `omit`s them). See
  [ADR-0005](../ADRs/0005-heavy-adapters.md).
- **Phase B** *(v1, dependency-light)* — `mine_hard_negatives`,
  `tune_threshold`, `adjusted_metrics`, `LabeledPairs.split`,
  `LinkageResult.from_candidate_frame` (the ergonomic frame→pairs constructor
  that makes `match_pairs` end-to-end usable — see D2 in
  [decisions.md](./decisions.md)).
- **Phase C** *(v2, behind `[train]`)* — `EmbedderTrainer` /
  `CrossEncoderTrainer` (implement the `Trainer` protocol) + few-shot
  selection. LLM fine-tuning is out of scope. The first/second-pass
  hard-negative *simulation* example (former `examples/04`) is a Phase-C
  deliverable — restored then as a **real** (non-faked) example once mining +
  trainers exist (see [examples.md](./examples.md)).
