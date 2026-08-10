# EditDistanceMatcher – Submission Report

## Self-check output

### `mypy --strict edit_distance_matcher.py`

```
Success: no issues found in 1 source file
```

### `python verify_probe.py edit_distance_matcher.py`

```
=== edit_distance_matcher.py ===
  [PASS] subclassesAtcher (explicit)
  [PASS] one outcome per pair
  [PASS] normal pair -> MatchDecision
  [PASS] empty text -> MatchError
  [PASS] end-to-end link() runs
  CONFORMANCE+FUNCTIONAL: PASS
```

---

## Short-form answers

- **Roughly how long did this take you (reading + coding + fixing)?**
  ~25 min (reading handout + writing the adapter + running both checks).

- **Did mypy + verify pass on your first try? If not, what tripped you up?**
  Yes, both passed on the first attempt.

- **Was anything about the Matcher port or the value types unclear or surprising?**
  No. The contract is unusually explicit: the alignment guarantee ("one outcome per
  input pair, position-aligned") and the two-variant return type
  (`MatchDecision | MatchError`) tell you exactly what to implement.
  The only mildly surprising detail is that `MatchDecision.is_match` is described
  as "always a real bool" — implying the caller cannot receive a `None` — but this
  is only notable because many Python protocols leave that implicit.

- **Did you need to look anything up beyond this handout? What?**
  No external look-ups were needed. `difflib.SequenceMatcher.ratio()` signature and
  `@runtime_checkable Protocol` semantics are standard-library knowledge.

- **One thing you'd change about this API:**
  The `CandidatePair.similarity_score` field (already computed upstream) is
  passed into `match()` but the contract does not say whether the matcher *should*
  use it or always recompute its own score. A docstring clarifying whether that
  field is informational-only or a hint the matcher may rely on would remove the
  ambiguity.
