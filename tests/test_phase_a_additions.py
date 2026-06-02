"""Phase A0 / A0.2: the training & evaluation types and the ``Trainer``
protocol are part of the locked contract, and ``DenseLinker``'s
optional-blocker shape is pinned at the signature level.

Per ADR-0002 the evaluation *report* types (``ThresholdSweep``,
``AdjustedMetrics``, and the per-stage ``*Metrics``) live in
``denselinkage.metrics``, not ``core``; only the ``Trainer`` port and its
training-input type (``TrainingPairs``) stay in ``core``.
"""

import dataclasses

from denselinkage import core, metrics
from denselinkage.core import ports, results
from denselinkage.linkage import DenseLinker

# Contract types that stay in core: the Trainer port + its training input.
CORE_CONTRACT_SYMBOLS = {"Trainer", "TrainingPairs"}
# Evaluation report types relocated to the metrics layer (ADR-0002).
METRICS_REPORT_SYMBOLS = {"AdjustedMetrics", "ThresholdSweep"}


def test_core_contract_symbols_exported() -> None:
    assert set(core.__all__) >= CORE_CONTRACT_SYMBOLS
    for name in CORE_CONTRACT_SYMBOLS:
        assert hasattr(core, name), name


def test_evaluation_reports_live_in_metrics_not_core() -> None:
    for name in METRICS_REPORT_SYMBOLS:
        assert name in metrics.__all__, f"{name} not exported from denselinkage.metrics"
        assert hasattr(metrics, name), name
        assert not hasattr(core, name), f"{name} should have moved out of core"


def test_trainer_is_runtime_checkable_protocol() -> None:
    assert getattr(ports.Trainer, "_is_runtime_protocol", False)


def test_training_eval_types_are_dataclasses() -> None:
    assert dataclasses.is_dataclass(results.TrainingPairs)
    assert dataclasses.is_dataclass(metrics.ThresholdSweep)
    assert dataclasses.is_dataclass(metrics.AdjustedMetrics)


def test_denselinker_blocker_optional_matcher_required() -> None:
    fields = {f.name: f for f in dataclasses.fields(DenseLinker)}
    assert fields["blocker"].default is None
    assert fields["matcher"].default is dataclasses.MISSING
    assert hasattr(DenseLinker, "match_pairs")
