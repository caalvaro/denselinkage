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
- **A1+** — fill method bodies behind the frozen contract. **Gated:** do not
  start A1 until A0.5 is complete and the contract is re-declared frozen.
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
