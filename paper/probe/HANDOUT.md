# Implement one adapter for `denselinkage` (~15–30 min)

Thanks for helping! You're going to add one small component to a Python
entity-resolution library, working **only from its public contract** (below).
The point is to find out whether someone who did **not** design the library can
implement against it from the documented interface alone — so please don't read
the existing adapter source. Your result is checked mechanically; there are no
trick questions.

## Ground rules

- Use **only** the public contract shown below plus the Python standard library.
- **Do not look at** the library's existing adapters or examples — specifically
  anything under `src/denselinkage/matching/`, `.../blocking/`, `.../embedding/`,
  `.../indexing/`, or `examples/03_custom_embedder.py` / `examples/05_*.py`.
  (Reading `core/ports.py` / `core/models.py` docstrings is fine — that *is* the
  contract.)
- Don't use an AI coding assistant for this one — we already measured that; here
  we want a human implementer.
- Single-shot is not required: iterate against the checker as much as you like.

## Setup (2 minutes)

```bash
python -m venv .venv && . .venv/Scripts/activate   # or source .venv/bin/activate
pip install denselinkage          # the dependency-free core is all you need
```

You'll also need the file `verify_probe.py` (provided with this handout) to
self-check.

## The contract (the only API you may use)

```python
# importable from denselinkage.core.models
@dataclass(frozen=True, slots=True)
class Record:
    id: str
    text: str
    fields: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class CandidatePair:
    record_a: Record
    record_b: Record
    similarity_score: float | None = None

@dataclass(frozen=True, slots=True)
class MatchDecision:        # a successful decision; is_match is always a real bool
    is_match: bool
    confidence: float | None = None
    rationale: str | None = None

@dataclass(frozen=True, slots=True)
class MatchError:           # a pair the matcher could not decide
    reason: str

# importable from denselinkage.core.ports
@runtime_checkable
class Matcher(Protocol):
    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision | MatchError]:
        # one outcome per input pair, aligned by position; a pair you cannot
        # decide yields a MatchError (never raise out of the batch).
        ...
```

First-party adapters **subclass the port explicitly** (`class X(Matcher): ...`)
so the type checker can confirm the implementation is complete.

## Your task

Create a single self-contained file `edit_distance_matcher.py` containing a
`Matcher` adapter named **`EditDistanceMatcher`** that decides each pair by the
**string similarity** of the two records' `text`:

- Use `difflib.SequenceMatcher(None, a, b).ratio()` (standard library) on the
  two **lowercased** texts as the similarity `s` (a float in `[0, 1]`).
- Constructor: keyword-only `threshold: float = 0.6`.
- For each pair return `MatchDecision(is_match = (s >= threshold), confidence = s)`.
- If **either** record's text is empty, return a `MatchError` for that pair
  (don't raise, don't guess a decision).
- Return exactly one outcome per input pair, in the same order.

It should:
- import only from `denselinkage.core.models`, `denselinkage.core.ports`, and
  the standard library;
- subclass `Matcher` explicitly;
- type-check under `mypy --strict` (annotate everything).

## Self-check

```bash
pip install mypy
mypy --strict edit_distance_matcher.py          # expect: Success: no issues found
python verify_probe.py edit_distance_matcher.py # expect: CONFORMANCE+FUNCTIONAL: PASS
```

(The checker runs your matcher inside a real linkage pipeline and verifies the
contract — one outcome per pair, a `MatchDecision` for a normal pair, a
`MatchError` for an empty-text pair, and an end-to-end run.)

## What to send back

1. Your `edit_distance_matcher.py`.
2. The output of the two self-check commands.
3. The short form below (be honest — "it was annoying" is useful data):

```
- Roughly how long did this take you (reading + coding + fixing)?  ____ min
- Did mypy + verify pass on your first try? If not, what tripped you up?
- Was anything about the Matcher port or the value types unclear or surprising?
- Did you need to look anything up beyond this handout? What?
- One thing you'd change about this API:
```

## How your contribution is used

This is for a research paper on the library's design. Your submitted adapter and
your (anonymous) answers above may be included in the paper's public artifact.
No personal information is collected and no attribution is recorded — the
submission is anonymous. Participation is voluntary; thank you!
