Metrics
=======

Pure functions over already-computed pipeline outputs, plus the typed report
dataclasses they return. Each evaluator takes ground truth as a keyword-only
``gold`` argument. Per :doc:`/architecture` (ADR-0002) these live in the metrics
layer, not in ``core`` — nothing in the contract depends on them.

.. currentmodule:: denselinkage.metrics

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   linkage_metrics
   blocking_metrics
   pair_completeness_at_k
   clustering_metrics
   tune_threshold
   adjusted_metrics

Reports
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   LinkageMetrics
   BlockingMetrics
   ClusteringMetrics
   ThresholdSweep
   AdjustedMetrics
