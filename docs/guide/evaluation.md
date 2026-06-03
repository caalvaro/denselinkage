# Evaluation & metrics

The evaluators are **pure functions** over already-computed outputs. Each takes
ground truth as a keyword-only `gold` argument and returns a typed report. They
live in {doc}`denselinkage.metrics </api/metrics>`.

## Ground truth

Express gold matches as {class}`~denselinkage.core.results.LabeledPairs`:

```python
from denselinkage import LabeledPairs

gold = LabeledPairs.from_pairs([("A1", "B1"), ("A2", "B2")])
```

## Linkage quality

{func}`~denselinkage.metrics.linkage_metrics` scores decided pairs against gold:

```python
from denselinkage.metrics import linkage_metrics

m = linkage_metrics(result, gold=gold)     # -> LinkageMetrics
print(m.precision, m.recall, m.f1)
print(m.n_errors)                          # undecided pairs, excluded from P/R
```

### Direction matters for dedup

`link` pairs are directed (left vs right is meaningful); dedup pairs are not.
Control this with the `directed` flag:

```python
linkage_metrics(result, gold=gold, directed=False)   # canonicalize unordered pairs
```

Use `directed=False` when scoring a `dedupe` run so `("1","2")` and `("2","1")`
count as the same gold pair.

## Blocking quality

Before matching, check that blocking actually retrieved the true pairs —
otherwise no matcher can recover them:

```python
from denselinkage.metrics import blocking_metrics, pair_completeness_at_k

bm = blocking_metrics(candidates, gold=gold)            # -> BlockingMetrics
pc = pair_completeness_at_k(candidates, gold=gold, k=5) # recall@k of the blocker
```

## Clustering quality

After {func}`~denselinkage.clustering.connected_components`, measure over-/under-
merging with B³ (Bagga–Baldwin):

```python
from denselinkage.metrics import clustering_metrics

cm = clustering_metrics(clusters, gold=gold)            # -> ClusteringMetrics
print(cm.b3_precision, cm.b3_recall, cm.b3_f1)
```

## Threshold tuning

{class}`~denselinkage.metrics.ThresholdSweep` and
{class}`~denselinkage.metrics.AdjustedMetrics` support sweeping the decision
threshold and reporting prevalence-adjusted scores — design-time configuration
material for picking an operating point.
