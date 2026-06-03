"""Evaluation — the typed report dataclasses and the pure functions that
produce them, over already-computed pipeline outputs.

This package is a façade: implementations live in sibling modules
(``linkage``, ``blocking``, ``clustering``, ``tuning``, ``adjusted``); import
the public names here. Per ADR-0002 these evaluation *report* types live with
the metrics layer that computes them, not in ``denselinkage.core``.
"""

from denselinkage.metrics.adjusted import AdjustedMetrics
from denselinkage.metrics.blocking import (
    BlockingMetrics,
    blocking_metrics,
    pair_completeness_at_k,
)
from denselinkage.metrics.clustering import ClusteringMetrics, clustering_metrics
from denselinkage.metrics.linkage import LinkageMetrics, linkage_metrics
from denselinkage.metrics.tuning import ThresholdSweep

__all__ = [
    "AdjustedMetrics",
    "BlockingMetrics",
    "ClusteringMetrics",
    "LinkageMetrics",
    "ThresholdSweep",
    "blocking_metrics",
    "clustering_metrics",
    "linkage_metrics",
    "pair_completeness_at_k",
]
