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
from denselinkage import DenseLinker
from denselinkage.metrics import blocking_metrics, pair_completeness_at_k

# `block` returns the blocker's candidate pairs without matching. Use a large
# `top_k` so pair-completeness can be swept over several k.
candidates = DenseLinker.with_defaults().block(left, right, top_k=10)
bm = blocking_metrics(candidates, gold=gold, ks=[1, 5, 10])  # -> BlockingMetrics
print(bm.pc_at(5))
pc = pair_completeness_at_k(candidates, gold=gold, k=5)       # recall@k of the blocker
```

For a single-table `dedupe` blocker, query the index against its own source —
`DenseLinker.with_defaults().index(source).candidates(source)`.

## Clustering quality

After {func}`~denselinkage.clustering.connected_components`, measure over-/under-
merging with B³ (Bagga–Baldwin):

```python
from denselinkage.metrics import clustering_metrics

cm = clustering_metrics(clusters, gold=gold)            # -> ClusteringMetrics
print(cm.b3_precision, cm.b3_recall, cm.b3_f1)
```

## Threshold tuning

{func}`~denselinkage.metrics.tune_threshold` sweeps the decision threshold over
scored candidates and returns the full P/R/F1 curve as a
{class}`~denselinkage.metrics.ThresholdSweep`; pick an operating point with
`best_f1()` or `at_recall(target)`:

```python
from denselinkage import DenseLinker
from denselinkage.metrics import tune_threshold

candidates = DenseLinker.with_defaults().block(left, right, top_k=10)
sweep = tune_threshold(candidates, gold=gold)    # -> ThresholdSweep
threshold, m = sweep.best_f1()                    # the F1-optimal cut
print(threshold, m.precision, m.recall, m.f1)
```

By default the grid is the candidates' distinct scores — the only cuts that
change the prediction — so `best_f1()` finds the true optimum.

## Separating blocker recall from matcher recall

A low end-to-end recall has two possible culprits: the blocker never surfaced the
true pair, or the matcher saw it and said no.
{func}`~denselinkage.metrics.adjusted_metrics` separates them, reporting
`recall_adjusted = matcher.recall × pc@k` where `matcher.recall` is measured only
over the gold pairs blocking actually surfaced:

```python
from denselinkage.metrics import adjusted_metrics

adj = adjusted_metrics(result, candidates, gold=gold, k=10)  # -> AdjustedMetrics
print(adj.blocking_recall_at_k, adj.recall_adjusted, adj.f1_adjusted)
```
