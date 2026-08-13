# Releasing and package evolution

> Part of the [denselinkage development docs](./README.md). The contract rules
> that decide *which* version number a change forces live in
> [freeze-gate.md](./freeze-gate.md) and
> [ADR-0003](../ADRs/0003-pre-freeze-contract-ratification.md); this document is
> the mechanics.

## Branching model: trunk-based

**All PRs target `main`.** There is no permanent development or release branch.

This is deliberate rather than an omission. Release branches exist to stabilise a
release while the trunk races ahead. That problem does not arise here: work is
additive by construction under extend-never-modify, so conflicts are rare, and CI
gates every merge, so `main` is releasable at all times.

### The one case that justifies a release branch

Cut `release/X.Y.x` **from the `vX.Y.0` tag, retroactively, when a patch is needed
on a line `main` has already left behind.** The foreseeable instance: v1.1.0 has
shipped, `main` has started on v2.0.0 (which by definition carries a breaking port
change), and a 1.1.1 fix is required. Then:

```bash
git switch -c release/1.1.x v1.1.0
# fix, PR into release/1.1.x, tag v1.1.1
git cherry-pick <fix>          # forward-port to main
```

Do not create the branch in advance. A release branch with nothing on it is a
merge target people push to by mistake.

## What decides the version number

Semantic versioning, with the asymmetry the frozen contract imposes. The decisive
question is never "how big is the diff" but **"can existing code still compile and
run against this?"**, and for structural `Protocol`s the party that breaks is the
*implementer*, not the caller.

| Change | Version | Why |
|---|---|---|
| New adapter, new sibling type, new optional field with a default, new classmethod | **minor** | Additive. Nothing that satisfied the old surface stops satisfying it. |
| A **new** `Protocol` in `core/ports.py` | **minor** | Additive, but an architecture decision: propose it, do not just add it. |
| Filling a body that previously raised `NotImplementedError` | **minor** | The signature was already frozen; only behaviour arrives. |
| Bug fix with no surface change | **patch** | |
| Adding a member to an **existing** `Protocol` | **major** | Breaking for implementers. Every third-party class satisfying the old port stops conforming. mypy cannot see downstream implementers, so this repository stays green while their code breaks. |
| Changing a signature, field type, or default; renaming; removing | **major** | |
| Raising `requires-python` | **major** | Drops supported interpreters. |

See the add/remove asymmetry table in
[ADR-0003](../ADRs/0003-pre-freeze-contract-ratification.md) for the ruling, and
the additive corridor in [freeze-gate.md](./freeze-gate.md) for the additions
already approved.

### Pre-releases

Ship a `bN` pre-release before any major, and before a minor that lands a large new
surface. PyPI treats `bN` as a pre-release, so `pip install denselinkage` will not
pick it up and only `--pre` or an explicit pin will. That is what gives third-party
implementers a chance to test against a breaking change before it reaches everyone.
`1.0.0b1` and `1.0.0b2` were shipped this way.

## The runbook

### 1. Land the work

PRs to `main`, squash-merged. The `main protection` ruleset requires a pull request
and passing status checks; there are no bypass actors, so this applies to the
maintainer too.

### 2. One release PR

A single commit that touches only release metadata:

- `pyproject.toml` — `version`
- `CITATION.cff` — `version` and `date-released`
- `CHANGELOG.md` — move the `Unreleased` entries under a dated `## [X.Y.Z]` heading
- `uv.lock` — regenerate with `uv lock` after the bump, and commit it

The first three must agree. Nothing currently enforces that; see the
version-drift issue on the v1.1.0 milestone.

`uv.lock` is not optional bookkeeping. The lock records the project's own
version (`name = "denselinkage"`, `version = ...`), so bumping `[project]
version` without re-locking leaves it stale, and every CI job installs with
`uv sync --locked` while `release.yml` runs `uv lock --check`. A release PR that
skips it fails with "The lockfile at `uv.lock` needs to be updated, but
`--locked` was provided" before anything is published.

### 3. Tag and push

```bash
git switch main && git pull
git tag -a v1.1.0 -m "denselinkage 1.1.0"
git push origin v1.1.0
```

`.github/workflows/release.yml` triggers on `v*`. It builds an sdist and wheel with
`uv build`, runs `twine check` on both, then stops at the `pypi` environment, which
requires a reviewer. Approve it and the `pypa/gh-action-pypi-publish` step uploads
over **Trusted Publishing (OIDC)**. There is no API token anywhere in the
repository or its secrets.

`workflow_dispatch` runs the build job only; the publish job is gated on
`startsWith(github.ref, 'refs/tags/v')`, so a manual run can never publish.

### 4. GitHub Release

```bash
gh release create v1.1.0 --title "denselinkage 1.1.0" --notes-file <changelog section>
# add --prerelease for bN
```

## What is protected, and why

| Protection | Covers | Reason |
|---|---|---|
| `main protection` ruleset | `main`: no deletion, no force-push, PR required, 9 status checks required | `main` is releasable at all times, so it must never regress. No bypass actors. |
| `release tag protection` ruleset | `refs/tags/v*`: no deletion, no non-fast-forward | See below. This is the one that matters most. |
| `pypi` environment | Required reviewer before publish | A PyPI upload is irreversible. A version can be yanked but the number can never be reused, so an accidental tag push must be answerable with a decline. |

### Tags are immutable here, more than usual

`v1.0.0-freeze` and `v1.0.0` underpin published claims in the ISE 2026 paper: that
`core/ports.py` is byte-identical between them, that the package grew from 1,703 to
2,732 lines across that interval, and that the core is 586 lines against 911 of
adapters. A reader is invited to re-derive those figures from the public tags.

Because the claims are about *tags* rather than about `main`, they stay true for
every future release without maintenance, on one condition: **the tags never move.**
That is what `release tag protection` guarantees. It is also why a botched release
is fixed by shipping `X.Y.Z+1`, never by retagging.

`CITATION.cff` is the citable record of the artifact and should track the released
version.

## Local verification before a release

The commands in [CONTRIBUTING.md](../../CONTRIBUTING.md) are the short form and run
a weaker `mypy` than CI. Before tagging, run what CI runs:

```bash
uv sync --locked --extra dev
uv run ruff check src/ tests/ examples/ .claude/hooks/
uv run ruff format --check src/ tests/ examples/ .claude/hooks/
uv run mypy src/ examples/ .claude/hooks/   # CI checks all three; bare `uv run mypy` does not
uv run python -m compileall examples
uv run python examples/00_quickstart.py     # and 03, 04, 05: CI executes all four
uv run python examples/03_custom_embedder.py
uv run python examples/04_dedupe.py
uv run python examples/05_failure_accounting.py
uv run pytest -m "not adapter and not slow" --cov=denselinkage --cov-report=term

uv sync --locked --extra dev --extra faiss --extra sentence-transformers --extra langchain
uv run pytest -m "adapter" --cov=denselinkage --cov-config=.coveragerc.adapter --cov-report=term-missing
```

Two of those spellings are load-bearing. `--extra dev` is not `--dev`: `dev` lives in
`[project.optional-dependencies]`, not a PEP 735 group, so `uv sync --dev` *uninstalls*
ruff, mypy, pytest and pytest-cov and the commands below it then run the wrong toolchain
or none at all. `--locked` is what every CI job uses, so without it a local sync can
rewrite `uv.lock` and the divergence surfaces only on the PR.

The `test` job also re-runs mypy against its oldest leg, the only environment installing
the lock's numpy 2.2.6 fork. An error can exist there and not under numpy 2.4.6, so
reproduce it with a 3.10 interpreter before tagging:

```bash
UV_PROJECT_ENVIRONMENT=.venv-310 uv sync --locked --extra dev --extra faiss --python 3.10
UV_PROJECT_ENVIRONMENT=.venv-310 uv run --no-sync mypy --python-version=3.10 src/ examples/
```

One CI job still has no local equivalent: `core-only` syncs with no heavy extras and asserts
that none of them is importable. It needs a clean environment, so reproduce it in a throwaway
one rather than `.venv`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-core uv sync --locked --extra dev
UV_PROJECT_ENVIRONMENT=.venv-core uv run --no-sync python -c "import importlib.util as u; assert all(u.find_spec(m) is not None for m in ('numpy','pandas','denselinkage')); assert all(u.find_spec(m) is None for m in ('faiss','sentence_transformers','langchain_core','langchain_openai','torch'))"
UV_PROJECT_ENVIRONMENT=.venv-core uv run --no-sync python -c "import sys, denselinkage; assert not {'faiss','sentence_transformers','langchain_core'} & sys.modules.keys()"
UV_PROJECT_ENVIRONMENT=.venv-core uv run --no-sync pytest -m "not adapter and not slow"
```

The first assertion checks the core closure is present before checking the heavy backends are
absent. Without that, an empty or half-built environment passes it vacuously.

A new adapter module must be registered in **both** coverage configurations:
`omit` in `pyproject.toml` and `include` in `.coveragerc.adapter`. They use
opposite polarity, so omitting the first fails the matrix at 100% while omitting
the second silently ungates the module.
