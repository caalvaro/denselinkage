---
description: Scaffold and check a new adapter behind an existing port
argument-hint: "[port name, e.g. Matcher] [adapter class name]"
---

Add an adapter for $ARGUMENTS. An adapter behind an **existing** port is additive and ships
in a minor release. If the task actually needs a new port, stop: that is an architecture
decision to propose, not to merge in passing.

Work through this checklist and report which items are done:

1. **Module** in the right package, class subclassing its port explicitly. The subclassing
   is what makes `mypy --strict` check the implementation is complete.
2. **Façade**: a line in the package `__init__.py` and in its `__all__`.
3. **Contract row**: add the `(adapter, port)` pair to the table in `tests/test_contract.py`.
4. **Test file** `tests/test_<subject>.py`, module docstring cross-referencing where
   adjacent cases live. Fakes are hand-written; `unittest.mock` only to patch a third-party
   class at its import site.
5. **If it needs a heavy backend**: the extra in `pyproject.toml`; `require("<module>")`
   followed by the real import, both inside the method; a module-scope
   `pytestmark = pytest.mark.adapter`; and the module registered in **both** coverage
   configs — `omit` in `pyproject.toml` and `include` in `.coveragerc.adapter`. They have
   opposite polarity: missing the first fails the matrix at 100%, missing the second
   silently ungates the module.
6. **If it is a second backend behind a port that already has one**: a differential test
   against the dependency-free reference, not a copy of the reference's assertions.
7. **Examples** updated if the adapter changes what intended use looks like.

Conventions with their exemplar files: `docs/development/conventions.md`. Then run
`/verify`.
