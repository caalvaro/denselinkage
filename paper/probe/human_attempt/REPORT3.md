# Experiment 3 — EditDistanceMatcher: Implementation Report

## Task

Implement `EditDistanceMatcher`, a `Matcher` adapter for the `denselinkage`
library that decides candidate record pairs using `difflib.SequenceMatcher`
string similarity on the records' `text` field.

---

## Implementation

**File:** `edit_distance_matcher.py`

```python
import difflib
from collections.abc import Sequence

from denselinkage.core.models import CandidatePair, MatchDecision, MatchError
from denselinkage.core.ports import Matcher


class EditDistanceMatcher(Matcher):
    def __init__(self, *, threshold: float = 0.6) -> None:
        self._threshold = threshold

    def match(
        self, pairs: Sequence[CandidatePair]
    ) -> list[MatchDecision | MatchError]:
        results: list[MatchDecision | MatchError] = []
        for pair in pairs:
            a = pair.record_a.text
            b = pair.record_b.text
            if not a or not b:
                results.append(MatchError(reason="empty text"))
                continue
            s = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
            results.append(MatchDecision(is_match=s >= self._threshold, confidence=s))
        return results
```

**Key design decisions:**
- Subclasses `Matcher` explicitly so the type checker can verify completeness.
- Empty-text guard uses `not a or not b` — empty string is falsy in Python.
- Lowercasing is applied before computing the ratio as specified.
- Return type is `list[MatchDecision | MatchError]` with a pre-typed accumulator
  so `mypy --strict` accepts it without casts.

---

## Self-check output

### `mypy --strict edit_distance_matcher.py`

```
Success: no issues found in 1 source file
```

### `python verify_probe.py edit_distance_matcher.py`

```
=== edit_distance_matcher.py ===
  [PASS] subclasses Matcher (explicit)
  [PASS] one outcome per pair
  [PASS] normal pair -> MatchDecision
  [PASS] empty text -> MatchError
  [PASS] end-to-end link() runs
  CONFORMANCE+FUNCTIONAL: PASS
```

---

## Short-form answers

- **Roughly how long did this take you (reading + coding + fixing)?**
  ~5 min (reading the handout + writing the adapter + running the checks).

- **Did mypy + verify pass on your first try? If not, what tripped you up?**
  Yes, both passed on the first attempt. The only typing subtlety was
  pre-declaring the accumulator as `list[MatchDecision | MatchError]` so mypy
  infers the return type correctly under `--strict`.

- **Was anything about the Matcher port or the value types unclear or surprising?**
  The protocol was very clear. The one mildly surprising point is that
  `MatchError` is a value (not an exception) — but the handout states this
  explicitly ("don't raise, return a `MatchError`"), so it was not confusing.
  The separation between `MatchDecision` (always a concrete bool) and
  `MatchError` (undecidable pair) is a clean design choice.

- **Did you need to look anything up beyond this handout? What?**
  No external look-ups were needed. `difflib.SequenceMatcher` signature and
  `collections.abc.Sequence` are standard-library staples.

- **One thing you'd change about this API:**
  The `Matcher.match` return type is `list[MatchDecision | MatchError]`.
  A named union alias — e.g. `MatchOutcome = MatchDecision | MatchError` —
  exported from `core.models` would make implementer signatures more concise
  and the intent of the union more self-documenting.
