# Failure accounting

An LLM matcher does not always answer. It refuses, times out, or returns text
that will not parse. denselinkage treats that as a first-class outcome: the
{class}`~denselinkage.core.ports.Matcher` port returns a
{class}`~denselinkage.core.models.MatchDecision` **or** a
{class}`~denselinkage.core.models.MatchError` for each pair, the linker collects
the failures into the `errors` channel of
{class}`~denselinkage.core.results.LinkageResult`, and
{func}`~denselinkage.metrics.linkage_metrics` excludes them from precision and
recall and reports them as `n_errors`.

Why it matters: the usual alternative is to score a failure as a non-match. That
folds every failure on a true pair into a false negative, so the reported F1
drops as the matcher's failure rate rises — even though the matcher's decisions
on the pairs it *did* decide have not changed. A metric that moves with the
failure rate is not a property of the matcher alone, so two runs or two models
cannot be compared from F1 unless the failure rate is also held fixed and
reported. Keeping failures in a separate `n_errors` channel is what makes the two
numbers separable.

## The two accountings, side by side

The example below runs one fixed, dependency-free pipeline and changes only how a
failed pair is reported. A small adapter, `FlakyMatcher`, fails on the same
fraction `f` of pairs two ways — as a typed `MatchError` (excluded) and as a
silent non-match — and scores both against the same gold:

| `f` | F1 (excl.) | F1 (silent) | recall (excl.) | recall (silent) |
| --- | ---------- | ----------- | -------------- | --------------- |
| 0%  | 1.000      | 1.000       | 1.000          | 1.000           |
| 5%  | 1.000      | 0.976       | 1.000          | 0.953           |
| 10% | 1.000      | 0.955       | 1.000          | 0.913           |
| 20% | 1.000      | 0.868       | 1.000          | 0.767           |

The excluded F1 stays flat while the silent F1 falls. The effect is arithmetic,
not a property of this toy data: under a uniform failure rate `f`, excluding the
failed pairs leaves precision and recall unchanged, while scoring them as
non-matches sends recall to `(1 - f)` of its no-failure value. The companion
paper applies the same comparison to the DBLP-ACM benchmark; that run is in
[`benchmarks/failure_accounting_experiment.py`](https://github.com/caalvaro/denselinkage/tree/main/benchmarks/failure_accounting_experiment.py).

## Full example

Runnable on the dependency-free stack (no extras, no API key):

```{literalinclude} ../../examples/05_failure_accounting.py
:language: python
:caption: examples/05_failure_accounting.py
```

:::{note}
`FlakyMatcher` is a teaching adapter that *simulates* failures at a controlled
rate. A production matcher returns a `MatchError` when a real backend call
cannot be parsed or retried (see the [semantic + LLM guide](semantic-llm)); the
errors channel, `n_errors`, and the excluded metric all behave the same way.
:::
