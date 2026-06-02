"""The dependency-free contract: models, ports, results.

This is the source of truth. Everything else in the package either implements a
port here or orchestrates ports defined here.
"""

from denselinkage.core.errors import (
    DenseLinkageError,
    DimensionMismatch,
    DuplicateRecordId,
    EmptySource,
    InvalidTopK,
    UnknownIdColumn,
)
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
    BlockingIndex,
    Clusterer,
    Embedder,
    Filter,
    Matcher,
    SearchableIndex,
    Serializer,
    Trainer,
    VectorIndex,
)
from denselinkage.core.results import (
    AdjustedMetrics,
    BlockingMetrics,
    Clustering,
    ClusteringMetrics,
    LabeledPairs,
    LinkageMetrics,
    LinkageResult,
    ThresholdSweep,
    TrainingPairs,
)

__all__ = [
    "AdjustedMetrics",
    "Blocker",
    "BlockingIndex",
    "BlockingMetrics",
    "CandidatePair",
    "Clusterer",
    "Clustering",
    "ClusteringMetrics",
    "DenseLinkageError",
    "DimensionMismatch",
    "DuplicateRecordId",
    "Embedder",
    "EmptySource",
    "Filter",
    "InvalidTopK",
    "LabeledPairs",
    "LinkageMetrics",
    "LinkageResult",
    "MatchDecision",
    "MatchError",
    "Matcher",
    "Record",
    "RecordId",
    "SearchableIndex",
    "Serializer",
    "Source",
    "ThresholdSweep",
    "Trainer",
    "TrainingPairs",
    "UnknownIdColumn",
    "VectorIndex",
]
