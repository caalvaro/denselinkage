"""Evaluation — pure functions over already-computed outputs.

This package is a façade: implementations live in sibling modules
(``linkage``, ``blocking``, ``clustering``); import the public names here.
"""

from denselinkage.metrics.blocking import blocking_metrics, pair_completeness_at_k
from denselinkage.metrics.clustering import clustering_metrics
from denselinkage.metrics.linkage import linkage_metrics

__all__ = [
    "blocking_metrics",
    "clustering_metrics",
    "linkage_metrics",
    "pair_completeness_at_k",
]
