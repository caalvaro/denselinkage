# ADR-0002: Evaluation result types belong to the metrics layer, not `core`

**Status:** Accepted; implemented at structure stage (2026-06-02). See the Implementation note.
**Date:** 2026-06-02
**Deciders:** Alvaro (author);

## Context

`denselinkage.core` is declared as *"the dependency-free contract … the source
of truth. Everything else in the package either implements a port here or
orchestrates ports defined here."* It is the innermost layer; by the dependency
rule it must not contain names that only the outer layers use.

`core/results.py` currently holds **nine** dataclasses of two ontologically
distinct kinds:

- **Contract / domain data** that flows through the pipeline or is part of a
  port signature: `LinkageResult`, `ClusteringResult`, `TrainingPairs`, `LabeledPairs`.
- **Evaluation reports** computed by the `denselinkage.metrics` layer:
  `LinkageMetrics`, `BlockingMetrics`, `ClusteringMetrics`, `ThresholdSweep`,
  `AdjustedMetrics`.

The single name "results" papers over the split. The decisive test is *what
`core` itself depends on*. `core/ports.py` references, under `TYPE_CHECKING`,
exactly:

```python
from denselinkage.core.results import ClusteringResult, LinkageResult, TrainingPairs
```

Auditing every type in `results.py` against "is it referenced anywhere inside
`core` (a port, a model, or another result a port needs)?":

| Type | Referenced inside `core`? | Nature | Home |
|---|---|---|---|
| `LinkageResult` | yes — `Clusterer.cluster(result)` | pipeline output | **core** |
| `ClusteringResult` | yes — `Clusterer.cluster(...) -> ClusteringResult` | pipeline output | **core** |
| `TrainingPairs` | yes — `Trainer.train(pairs)` | training input | **core** |
| `LabeledPairs` | no (domain ground-truth, like `Source`) | gold input | **core** |
| `LinkageMetrics` | **no** | evaluation report | **metrics** |
| `BlockingMetrics` | **no** | evaluation report | **metrics** |
| `ClusteringMetrics` | **no** | evaluation report | **metrics** |
| `ThresholdSweep` | **no** | evaluation report (Phase B) | **metrics** |
| `AdjustedMetrics` | **no** | evaluation report (Phase B) | **metrics** |

The five evaluation reports are **produced by `denselinkage.metrics` functions
and consumed only by user code** — nothing in `core` depends on them. Housing
them in `core` is a quiet inversion of the dependency rule: the innermost layer
warehouses vocabulary that only an outer layer needs.

### Constraints specific to this project

- **The architecture is the deliverable** (thesis / intended paper); the
  dependency rule and a clean layering story outrank a single import location.
- **Pre-freeze**: contract-shape moves are cheap now, breaking after the A0.5
  gate.
- **Examples-as-spec** and the **public prelude** must keep working — user
  imports like `from denselinkage import LinkageMetrics` cannot break.
- **Dependency-light core** (numpy + pandas) — unaffected either way.

## Decision

Adopt **Option B**: move the five evaluation report types
(`LinkageMetrics`, `BlockingMetrics`, `ClusteringMetrics`, `ThresholdSweep`,
`AdjustedMetrics`) out of `core` into the **`denselinkage.metrics`** package,
co-located with the functions that produce them (e.g. `LinkageMetrics` beside
`linkage_metrics`; `ThresholdSweep` / `AdjustedMetrics` beside their Phase-B
producers). Keep `LinkageResult`, `ClusteringResult`, `TrainingPairs`, and
`LabeledPairs` in `core`. The public prelude re-exports the moved types from
their new home, so `from denselinkage import LinkageMetrics` is unchanged.

Optionally rename `core/results.py` → `core/outputs.py`, since after the move it
holds pipeline outputs and gold/training inputs, not "metrics results."

## Options Considered

### Option A — keep every result type in `core` (status quo)

| Dimension | Assessment |
|---|---|
| Complexity | Low (no change) |
| Layering honesty | **Low — `core` holds types only the metrics layer uses** |
| Cohesion | Low — `metrics` functions return types defined two layers down |
| Public-API churn | None |
| Contract-test churn | None |

**Pros:** one dependency-free home for *all* public types (the "shared type
kernel" the current `core` docstring describes); zero churn.
**Cons:** violates the dependency rule; `core`'s identity ("only what the
contract depends on") is diluted; `metrics` is not self-contained.

### Option B — evaluation reports live in `metrics` — **chosen**

| Dimension | Assessment |
|---|---|
| Complexity | Medium (move 5 types; re-point prelude + 2 tests) |
| Layering honesty | **High — `core` keeps only contract/domain types** |
| Cohesion | **High — `metrics` owns its inputs and outputs** |
| Public-API churn | **None (prelude re-exports unchanged)** |
| Contract-test churn | Low (two structure-stage tests updated) |

**Pros:** restores the dependency rule (no inner-layer warehouse of outer-layer
types); `metrics` becomes self-contained; sharpens `core`'s definition;
construction/usage API unchanged.
**Cons:** five types change module; `core.__all__`, the prelude import source,
the `metrics` function files, and two contract tests must move with them.

### Option C — extract one top-level `results` / `evaluation` package for *all* output types — **rejected**

| Dimension | Assessment |
|---|---|
| Complexity | High |
| Layering honesty | Medium — but creates `core` ↔ `results` coupling |
| Public-API churn | Higher |

**Cons:** `core/ports` references `LinkageResult` / `ClusteringResult` /
`TrainingPairs`, and those depend on `core/models` (`CandidatePair`,
`MatchDecision`, `RecordId`). Pulling them into a separate top-level package
makes `core/ports → results → core/models` — `core` would depend on an outside
package that in turn depends back into `core`. The port-referenced outputs must
live *inside* `core`. For the non-port-referenced subset, Option C collapses
into Option B.

## Trade-off Analysis

The real axis is **the dependency rule vs. a single type kernel**. Option A
optimizes for "one place to find any type"; Option B optimizes for "the inner
layer contains only what the contract depends on." Option C tries to give every
output type one home but breaks on the fact that some outputs are *part of the
contract* (port-referenced) and some are not.

Option B draws the line exactly where the dependency graph already draws it:
the three port-referenced outputs plus the domain gold (`LabeledPairs`) stay in
`core`; the five computed reports move next to their producers. The decisive
evidence is mechanical, not aesthetic — `core/ports.py` names three result types
and never the five reports.

`LabeledPairs` is the one judgment call. It is *not* port-referenced, so a strict
reading would move it too. It stays in `core` because it is **domain
ground-truth** (a value object on the same footing as `Source`, a user-supplied
input), and it is consumed beyond evaluation — Phase-B hard-negative mining and
active learning also read gold. Tying it to `metrics` would under-scope it.
(Revisit if `metrics` remains its only consumer through v1.)

## Consequences

**Easier**
- `core`'s definition becomes enforceable: "models + ports + the outputs a port
  references + errors." A future "does this belong in core?" question has a
  one-line test (is it referenced inside `core`?).
- `metrics/` is self-contained — each function and the report it returns live
  together; the layer can evolve without touching `core`.

**Harder**
- Five types change import location; `core.__all__`, the prelude, the `metrics`
  function modules, and two contract tests update together.
- One more place (the prelude) is the single re-export point users rely on — it
  must keep re-exporting the moved types.

**To revisit**
- `LabeledPairs` placement if `metrics` stays its only consumer.
- Whether to rename `core/results.py` → `core/outputs.py` for accuracy.
- The `core` docstring, which currently frames `core` as the home for *all*
  results — it should be reworded to "the contract: models, ports, the outputs
  ports reference, and errors."

**Contract-test impact (pre-freeze, expected).**
`tests/test_phase_a_additions.py` currently pins `AdjustedMetrics` and
`ThresholdSweep` in `core.__all__`; `tests/test_contract.py` imports
`results.LinkageMetrics` / `results.BlockingMetrics` / `results.ClusteringMetrics`
for the dataclass-shape check. These encode Option A and move with the decision —
the fitness functions are updated to assert the reports live in `metrics`, not
`core`.

## Action Items

1. [x] Move `LinkageMetrics`, `BlockingMetrics`, `ClusteringMetrics`,
   `ThresholdSweep`, `AdjustedMetrics` from `core/results.py` into
   `denselinkage.metrics` (co-located with their producing functions).
2. [x] Update `core/__init__.py` (drop the five from imports + `__all__`); keep
   `LinkageResult`, `ClusteringResult`, `TrainingPairs`, `LabeledPairs`.
3. [x] Re-point `denselinkage/__init__.py` (prelude) to re-export the five report
   types from `denselinkage.metrics`; public import paths stay identical.
4. [x] Update the `metrics` function modules to import their return types locally.
5. [x] Update `tests/test_phase_a_additions.py` and `tests/test_contract.py` to
   assert the reports live in `metrics`.
6. [x] Reword the `core` package docstring; optionally rename
   `core/results.py` → `core/outputs.py`.
7. [x] Record this as a decision in
   [`docs/development/decisions.md`](../development/decisions.md) (D7) and run the
   gate (`ruff`, `mypy --strict`, `pytest`, `compileall examples`).

## References / prior art

- **The Dependency Rule** (R. C. Martin, *Clean Architecture*) — inner layers
  must not depend on, or warehouse, outer-layer concerns.
- **Hexagonal Architecture** (A. Cockburn) — the core holds the domain and the
  ports; adapters and application services (here, `metrics`) sit outside.
- **ADR-0001** ([`0001-stateful-components-as-artifacts.md`](./0001-stateful-components-as-artifacts.md))
  — the spec→artifact decision; this ADR continues the same "audit what `core`
  must contain" discipline.

## Implementation note (landed 2026-06-02, structure stage)

Implemented as specified. The five report types moved to `denselinkage.metrics`,
co-located with their producers:

- `LinkageMetrics` → `metrics/linkage.py`
- `BlockingMetrics` → `metrics/blocking.py`
- `ClusteringMetrics` → `metrics/clustering.py`
- `ThresholdSweep` → `metrics/tuning.py` (Phase-B `tune_threshold` joins it)
- `AdjustedMetrics` → `metrics/adjusted.py` (Phase-B `adjusted_metrics` joins it)

`core/results.py` keeps `LinkageResult`, `ClusteringResult`, `LabeledPairs`,
`TrainingPairs`; its docstring is reworded. The `core/results.py` →
`core/outputs.py` rename (action item 6, "optional") is **deferred** — it would
churn every `from denselinkage.core.results import …` site for a cosmetic gain.
The public prelude re-exports the report types from `denselinkage.metrics`, so
`from denselinkage import LinkageMetrics` is unchanged.

Fitness functions updated to assert the new home: `test_contract.py`
(`metrics.LinkageMetrics` / `BlockingMetrics` / `ClusteringMetrics` are
dataclasses), `test_a05_contract.py` (imports `BlockingMetrics` /
`ClusteringMetrics` from `metrics`), and `test_phase_a_additions.py` (the reports
must be in `metrics.__all__` and **absent** from `core`). Logged as **D7** in
[`docs/development/decisions.md`](../development/decisions.md).

Gate green: `ruff check`, `ruff format --check`, `mypy --strict` (48 files),
`pytest -m "not adapter and not slow"` (74 passed), `00_quickstart.py`
P/R/F1 = 1.0, and the import dependency-cut (no heavy backend on
`import denselinkage`).
