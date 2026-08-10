# Contract-implementability probe

Materials and results for the extensibility (QR1) probe reported in the paper.
Start with [`RESULTS.md`](RESULTS.md).

The probe has two arms, with different tasks:

|                | Arm 1: developers                     | Arm 2: LLM agent                             |
| -------------- | ------------------------------------- | -------------------------------------------- |
| Protocol       | [`HANDOUT.md`](HANDOUT.md)            | [`PROMPT.md`](PROMPT.md)                     |
| Task           | `EditDistanceMatcher` (`difflib`)     | `TokenJaccardMatcher`, `HashingEmbedder`     |
| Submissions    | `human_attempt/`                      | `attempt_01..05/`, `emb_attempt_01..05/`     |
| Written report | `human_attempt/REPORT{1,2,3}.md`      | mechanical scoring only                      |
| Harness        | `verify_probe.py`                     | `verify_probe.py`, `verify_probe_embedder.py`|

`HANDOUT.md` is the file the three developers received. It restricts them to the
public contract plus the Python standard library and asks them not to read the
existing adapters or examples. `PROMPT.md` sets a different task, so the two arms
are not directly comparable and neither could reuse the other's solution.

To re-run any single check:

```bash
pip install denselinkage mypy
mypy --strict human_attempt/edit_distance_matcher1.py
python verify_probe.py human_attempt/edit_distance_matcher1.py
```

Expect `Success: no issues found in 1 source file` followed by
`CONFORMANCE+FUNCTIONAL: PASS`.
