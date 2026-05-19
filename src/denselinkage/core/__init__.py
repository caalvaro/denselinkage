"""The dependency-free contract: models, ports, results.

This is the source of truth. Everything else in the package either implements a
port here or orchestrates ports defined here.
"""

from denselinkage.core.models import (
    CandidatePair,
    MatchDecision,
    MatchError,
    Record,
    RecordId,
    Source,
)
from denselinkage.core.ports import (
    Blocker,
    Embedder,
    Matcher,
    Serializer,
    Trainer,
    VectorIndex,
)
from denselinkage.core.results import (
    AdjustedMetrics,
    BlockingMetrics,
    Clustering,
    LabeledPairs,
    LinkageMetrics,
    LinkageResult,
    ThresholdSweep,
    TrainingPairs,
)

__all__ = [
    "AdjustedMetrics",
    "Blocker",
    "BlockingMetrics",
    "CandidatePair",
    "Clustering",
    "Embedder",
    "LabeledPairs",
    "LinkageMetrics",
    "LinkageResult",
    "MatchDecision",
    "MatchError",
    "Matcher",
    "Record",
    "RecordId",
    "Serializer",
    "Source",
    "ThresholdSweep",
    "Trainer",
    "TrainingPairs",
    "VectorIndex",
]
