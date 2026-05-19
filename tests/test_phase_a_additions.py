"""Phase A0 / A0.2: the training & evaluation types and the ``Trainer``
protocol are part of the locked core contract, and ``DenseLinker``'s
optional-blocker shape is pinned at the signature level.
"""

import dataclasses

from denselinkage import core
from denselinkage.core import ports, results
from denselinkage.linker import DenseLinker

NEW_CORE_SYMBOLS = {
    "AdjustedMetrics",
    "ThresholdSweep",
    "Trainer",
    "TrainingPairs",
}


def test_new_core_symbols_exported() -> None:
    assert set(core.__all__) >= NEW_CORE_SYMBOLS
    for name in NEW_CORE_SYMBOLS:
        assert hasattr(core, name), name


def test_trainer_is_runtime_checkable_protocol() -> None:
    assert getattr(ports.Trainer, "_is_runtime_protocol", False)


def test_training_eval_types_are_dataclasses() -> None:
    for cls in (
        results.TrainingPairs,
        results.ThresholdSweep,
        results.AdjustedMetrics,
    ):
        assert dataclasses.is_dataclass(cls)


def test_denselinker_blocker_optional_matcher_required() -> None:
    fields = {f.name: f for f in dataclasses.fields(DenseLinker)}
    assert fields["blocker"].default is None
    assert fields["matcher"].default is dataclasses.MISSING
    assert hasattr(DenseLinker, "match_pairs")
