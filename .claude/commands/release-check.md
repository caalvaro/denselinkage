---
description: Audit the repository before tagging a release
argument-hint: "[version, e.g. 1.1.0]"
---

Audit readiness to tag $ARGUMENTS. Report each item with its evidence; do not fix anything
without saying so first. The runbook is `docs/development/releasing.md`.

**1. The version agrees in three places.** `pyproject.toml` `[project] version`,
`CITATION.cff` `version` (and `date-released`), and the newest non-`Unreleased` heading in
`CHANGELOG.md`. Nothing enforces this yet.

**2. The bump matches the change.** Diff the frozen surface:

```bash
git diff $(git describe --tags --abbrev=0) -- src/denselinkage/core/
```

A new Protocol, type, module, or optional field with a default is additive: minor. A member
added to an existing Protocol, or any signature or field-type change, is breaking: major.
A docstring or comment is not a contract change at all: the freeze binds the surface
enumerated in `docs/development/freeze-gate.md`, not the bytes. If the diff shows a
breaking change and the planned version is a minor, stop and say so.

**3. Pre-release, if warranted.** A major, or a minor landing a large new surface, ships a
`bN` first so third-party implementers can test against it. PyPI treats `bN` as a
pre-release, so `pip install denselinkage` will not pick it up.

**4. Tags are intact.** `git diff v1.0.0-freeze v1.0.0 -- src/denselinkage/core/ports.py`
must still be empty: the paper's claims are pinned to those two tags, and
`release tag protection` blocks moving them. A botched release is fixed by shipping the
next patch version, never by retagging.

**5. Checks are green.** Run `/verify`, including the adapter gate.

**6. The changelog entry is real.** Added / Changed / Deprecated / Removed / Fixed, written
for a user of the library rather than as a commit log.
