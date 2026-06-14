# Contract-implementability probe — results

Generator: Claude Opus 4.8 (the orchestrating session's model). Five independent
single-shot attempts per port, each given only the public contract (see
`PROMPT.md`), with every existing adapter withheld. Every attempt was scored
mechanically; all attempts are reported.

| Port | contract surface implemented | attempts | `mypy --strict` | port conformance | behavioural contract | end-to-end run | touched only new file |
|------|------------------------------|---------:|:---------------:|:----------------:|:--------------------:|:--------------:|:---------------------:|
| `Matcher`  | `match()` returning `MatchDecision \| MatchError` | 5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| `Embedder` | `encode()` (numpy `float32`) + `model_id` + `embedding_dim` | 5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

**10 / 10 conforming.** Both the method-only `Matcher` and the harder `Embedder`
(typed numpy return, two property members, L2-normalisation, empty-text edge
case) were implemented correctly from the ports and value types alone.

## Verification (per attempt)
- **C1 type-completeness:** `mypy --strict <module>` reports no issues.
- **C2 conformance:** the class explicitly subclasses its port (checked via the
  MRO; `mypy --strict` independently verifies the implementation is complete
  against the `Protocol`).
- **C3 isolation:** `git status --porcelain src/` is empty — the adapters are
  self-contained new files that modify nothing under `src/`.
- **C4 behaviour + functional:** the port's behavioural contract (for `Matcher`:
  one outcome per pair, a `MatchError` on empty text; for `Embedder`: float32
  array of shape `(n, dim)`, L2-normalised rows, all-zero row for empty text)
  and an end-to-end run inside a real `DenseLinker` pipeline.

Harnesses: `verify_probe.py` (Matcher), `verify_probe_embedder.py` (Embedder).
Generated adapters: `attempt_0k/` (Matcher), `emb_attempt_0k/` (Embedder).

## Scope and honesty
- This is an **extensibility / implementability** probe, **not** a usability
  study. An LLM agent is not an independent human author; it measures whether
  the contract can be implemented from its public surface, not whether a human
  finds the API convenient.
- The tasks **fully specify the algorithm**, so the probe tests whether an
  implementer can map a specified component onto the contract correctly (the
  failure outcome, the numpy width/dtype), not whether it can design a good
  adapter.
- A study with **independent human authors** remains future work.
