# EditDistanceMatcher — Implementation Report

## Self-check output

### mypy --strict
```
Success: no issues found in 1 source file
```

### verify_probe.py
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

## Short-form questions

- **Roughly how long did this take you (reading + coding + fixing)?**
  ~5 min (reading handout + writing the implementation in one pass + running both checks).

- **Did mypy + verify pass on your first try? If not, what tripped you up?**
  Yes, both passed on the first try. The contract specification was precise enough to translate directly into code without any iteration.

- **Was anything about the Matcher port or the value types unclear or surprising?**
  Nothing was surprising. The union return type `list[MatchDecision | MatchError]` is a clean design choice — it forces callers to handle the "could not decide" case explicitly instead of using exceptions or `None`. The only mild ambiguity was whether "empty text" meant `len == 0` only or also whitespace-only strings; I went with the falsy check (`not a or not b`) which covers both.

- **Did you need to look anything up beyond this handout? What?**
  No external lookups were needed. `difflib.SequenceMatcher` is a well-known standard-library class and the rest came directly from the handout's contract.

- **One thing you'd change about this API:**
  The `MatchDecision.is_match` field name implies a binary decision but it carries an optional `confidence` — the types are fine, though a companion `MatchPending` or a dedicated `MatchSkipped` variant might make the three-way outcome (match / no-match / undecidable) even more explicit instead of overloading `MatchError` for the "I chose not to decide" case.

---

## Implementation notes

`EditDistanceMatcher` lives in `edit_distance_matcher.py` and imports only from:
- `denselinkage.core.models` (`CandidatePair`, `MatchDecision`, `MatchError`)
- `denselinkage.core.ports` (`Matcher`)
- Python standard library (`difflib`, `collections.abc`)

It subclasses `Matcher` explicitly and is fully annotated for `mypy --strict`.
