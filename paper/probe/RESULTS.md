# Contract-implementability probe: results

Two arms test one question from opposite directions: can a party that did not
design the contract implement a new adapter from the public ports and value types
alone? Both are reported here, including what they do not support.

Neither arm is a controlled experiment. Three participants and five single-shot
agent attempts per port indicate feasibility; they cannot support a general claim
about extensibility or usability.

## Arm 1: three independent developers

Task: a `Matcher` adapter named `EditDistanceMatcher` deciding each pair by
`difflib.SequenceMatcher` string similarity, with the algorithm, the threshold
(0.6) and the empty-text case specified. Protocol and contract: `HANDOUT.md`,
which is the file the participants received. It restricts them to the public
contract plus the Python standard library and asks them not to read the existing
adapters or examples.

Five self-check criteria, all mechanical: `mypy --strict` clean; explicit
subclassing of the port; one outcome per input pair; a `MatchError` on empty
text; an end-to-end run inside a real `DenseLinker` pipeline.

| Submission | `mypy --strict` | five criteria | passed on first run | self-reported time | look-ups beyond the handout |
|---|:--:|:--:|:--:|--:|---|
| `human_attempt/edit_distance_matcher1.py` | pass | 5/5 | yes | ~25 min | none |
| `human_attempt/edit_distance_matcher2.py` | pass | 5/5 | yes | ~5 min | none |
| `human_attempt/edit_distance_matcher3.py` | pass | 5/5 | yes | ~5 min | none |

**3 / 3 conforming.** No submission produced a contract violation. The adapters
are byte-for-byte as received; the short-form answers are quoted verbatim in
`human_attempt/REPORT{1,2,3}.md`.

Two friction points were reported, both in the task specification rather than in
the types:

- whether `CandidatePair.similarity_score` is informational or a hint the matcher
  may rely on (REPORT1);
- whether "empty text" covers whitespace-only text as well as the empty string
  (REPORT2).

Neither is resolved in the released source at `v1.0.0`.

## Arm 2: five single-shot LLM agent attempts per port

Generator: Claude Opus 4.8 (the orchestrating session's model). Protocol and
contract: `PROMPT.md`. Every existing adapter was withheld. The tasks are not the
same as Arm 1's: a `TokenJaccardMatcher` and, for the harder port, a
`HashingEmbedder`.

| Port | contract surface implemented | attempts | `mypy --strict` | port conformance | behavioural contract | end-to-end run | touched only new file |
|------|------------------------------|---------:|:---------------:|:----------------:|:--------------------:|:--------------:|:---------------------:|
| `Matcher`  | `match()` returning `MatchDecision \| MatchError` | 5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| `Embedder` | `encode()` (numpy `float32`) + `model_id` + `embedding_dim` | 5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

**10 / 10 conforming.** All attempts are reported, including any that failed.
Adapters in `attempt_0k/` (Matcher) and `emb_attempt_0k/` (Embedder).

## Verification, per attempt, both arms

- **C1 type-completeness:** `mypy --strict <module>` reports no issues.
- **C2 conformance:** the class explicitly subclasses its port, checked via the
  MRO; `mypy --strict` independently verifies the implementation is complete
  against the `Protocol`.
- **C3 isolation:** `git status --porcelain src/` is empty, so each adapter is a
  self-contained new file that modifies nothing under `src/`.
- **C4 behaviour and function:** the port's behavioural contract (for `Matcher`,
  one outcome per pair and a `MatchError` on empty text; for `Embedder`, a
  float32 array of shape `(n, dim)`, L2-normalised rows, and an all-zero row for
  empty text) plus an end-to-end run inside a real `DenseLinker` pipeline.

Harnesses: `verify_probe.py` (Matcher) and `verify_probe_embedder.py`
(Embedder). Each takes a module path and prints one line per criterion.

## Scope and limits

- This is an extensibility and implementability probe, not a usability study. It
  measures whether the contract can be implemented from its public surface, not
  whether a developer finds the API convenient.
- Both tasks fully specify the algorithm, so the probe tests whether an
  implementer can map a specified component onto the contract correctly, not
  whether they can design a good adapter.
- The agent arm shares a model family with the assistant used during the
  library's development, so it is weaker evidence than the developer arm and the
  paper weights it accordingly.
- n = 3 developers, and n = 5 agent attempts per port. A larger, user-centred
  study remains future work.

## Anonymity and consent

Participation was voluntary and anonymous. No personal information was collected
and no attribution is recorded. `HANDOUT.md` states that the submitted adapter
and the anonymous short-form answers may be included in the paper's public
artifact; they are published here under that consent.
