<!-- Thanks for contributing to denselinkage! Keep PRs focused. -->

## Summary

<!-- What does this change, and why? Link any related issue (e.g. Closes #123). -->

## Checklist

- [ ] Branched off `main`; the change is focused.
- [ ] `ruff check` and `ruff format --check` pass.
- [ ] `mypy src/ examples/` passes (strict).
- [ ] Tests added/updated; `pytest` is green and the dependency-free gate stays at
      100% branch coverage (`pytest -m "not adapter and not slow" --cov=denselinkage`).
- [ ] The change is **additive** to the frozen contract — no modified public
      signatures (extend, never modify).
- [ ] `CHANGELOG.md` `[Unreleased]` updated if the change is user-facing.
- [ ] Docs updated if behavior or public API changed.
