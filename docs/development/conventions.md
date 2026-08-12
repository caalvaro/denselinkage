# Coding, style and testing conventions

> Part of the [denselinkage development docs](./README.md). These conventions were derived
> by reading `src/`, `tests/` and `examples/`, not invented: each rule names the file that
> exemplifies it. [AGENTS.md](../../AGENTS.md) carries the short form; this is the depth.

## Design patterns in use

**Strategy via `Protocol`.** All ten ports are `@runtime_checkable class X(Protocol)`
(`core/ports.py`). First-party adapters name their port as a base
(`class HashedNGramEmbedder(Embedder)`), so `mypy --strict` checks the implementation is
complete. Third-party code may conform structurally without importing anything.
*Anti-pattern:* an adapter that does not subclass its port, which passes at runtime and
loses the completeness check.

**Specification builds artifact.** A stateless spec exposes `build()`, a pure factory
returning a fresh immutable artifact; the artifact exposes the read methods and never
`build`. `DenseBlocker.build` writes nothing to `self`. *Anti-pattern:* a spec with an
`add()`, a mutating method, or pre-populated state. This is what makes
`link(a, b) == index(a).query(b)` hold and lets `DenseLinker` be frozen with no defensive
copy (ADR-0001).

**Injection through keyword-only parameters.** `DenseBlocker(*, embedder, vector_index,
similarity_threshold=0.0, top_k=10, batch_size=None)`; `ThresholdMatcher(*, threshold=0.5)`.
All *new* configuration is keyword-only. Two published exceptions predate the rule and are
retained for compatibility: `HashedNGramEmbedder(n_features, ngram)` and
`FieldwiseSerializer(fields, sep)` accept configuration positionally. They cannot be fixed
inside v1, because adding `*` is breaking. **They are not precedents.**

**Frozen value objects.** Every domain type, result type, metric report and `RetryPolicy`
is `@dataclass(frozen=True, slots=True)`, stdlib only.

**Lazy heavy imports.** `require("<module>")` maps an importable module to its pip extra and
re-raises `ModuleNotFoundError` naming the exact install command; the real import follows on
the next line, inside the method. `pandas` is also imported inside the function that uses it
(`LinkageResult.to_frame`), appearing at module scope only under `TYPE_CHECKING`.

**Defensive read-only views.** An artifact accessor never hands out its internal mutable
state: `NumpySearchableIndex.vectors` returns a view with `writeable = False`; `.ids`
returns a `tuple`.

**Façade `__init__`.** Every package `__init__.py` is a docstring, re-exports and `__all__`,
with no implementation.

**Shared rules become named private helpers.** `pair_key` in `metrics/_pairing.py` is the
single comparison key used by five modules; its docstring names them.

## Style beyond ruff

ruff enforces formatting and the `E, F, I, UP, B, SIM, RUF` selection. These it does not
enforce.

- A leading underscore marks *private to users*, not private to `src/`. `_optional/`,
  `_reader/` and `_store/` are packages with their own `__all__`. Never re-export them into
  the public surface, and never import them from `examples/`.
- Spec and artifact are named as a pair, and the module file is the snake_case of the class:
  `NumpyFlatIndex` → `numpy_flat_index.py`, `NumpySearchableIndex` → `numpy_searchable_index.py`.
- The similarity cutoff is named by role: `similarity_threshold` at the blocking stage,
  beside `top_k`; plain `threshold` on a single-purpose component whose class name already
  carries the qualifier (`ThresholdMatcher`, `SimilarityThresholdFilter`).
- `from __future__ import annotations` is **banned**. Forward references are quoted strings
  (`-> "SearchableIndex"`) or `TYPE_CHECKING` imports.
- Two aliases carry meaning throughout: `RecordId = str` and `Vectors`.
- `Any` is the annotation for an injected heavy-backend object, with a comment saying why
  (`index` is a populated `faiss.Index`, kept untyped because faiss ships no stubs).
- Every module has a docstring. Class and method docstrings are selective and carry
  *contract*, not description.
- Error messages quote the offending value with `!r` and name the corrective action.
- Every `zip` over positionally aligned sequences passes `strict=True`. This backs the port
  contract that `Matcher.match` returns one outcome per input pair, aligned by position.
- Each metrics module holds one typed report dataclass plus the function that produces it.
  Never return a bare `dict`.
- `isinstance` is used against concrete adapter classes, never against a port.

## Testing style

- One test file per subject, `test_<subject>.py`, whose module docstring cross-references
  where adjacent cases are covered.
- `adapter` and `slow` are applied as a module-scope `pytestmark`, never per function, and
  the docstring says why.
- Collaborators are faked by hand (`_FakeChatModel`, `_FakeBlockingIndex`).
  `unittest.mock` appears only where a third-party class must be patched at its import site.
- There is no `conftest.py` and almost no fixtures. Test data comes from module-level
  `_helper()` functions returning fresh objects. The single fixture in the suite is
  module-scoped because loading a sentence-transformer model is the expensive part.
- **A contract test asserts shape by reflection** (`dataclasses.fields`,
  `inspect.signature`, `__mro__`, `__all__`) and never calls the code under test.
  **A behaviour test asserts computed values.** They live in different files.
- A second backend behind the same port is validated by a **differential test** against the
  dependency-free reference, not by re-asserting the same expectations.
  `test_faiss_matches_numpy_neighbours` seeds an RNG, runs identical queries through both,
  and asserts equal neighbour ids and scores to `atol=1e-5`. That oracle is specific to the
  exact-flat pair; an approximate backend needs its own recall-bound test, not a relaxation
  of this one.
- The 100% branch gate is met by writing the missing test. The whole of `src/` contains one
  `# pragma: no cover`.

## Adding an adapter: the checklist

1. A module in the right package, with a class subclassing its port.
2. A line in the package `__init__.py` and its `__all__`.
3. A row in the `(adapter, port)` table in `tests/test_contract.py`.
4. A dedicated test file; if it needs an extra, a module-scope `pytestmark`.
5. For a heavy adapter: the extra in `pyproject.toml`, the module registered in **both**
   coverage configs, and `require(...)` plus a method-local import.
6. For a second backend behind an existing port: a differential test against the
   dependency-free reference.
7. `examples/` updated if the adapter changes what intended use looks like.

## Settled, and not to be reopened without an ADR

No runtime `isinstance` dispatch against a port; no `pydantic` or `attrs` for domain types;
no distributed-execution layer; no artifact mutation; `Trainer` intentionally empty in v1;
evaluation report types outside `core` (ADR-0002).
