"""The dependency-free contract: models, ports, the outputs ports reference,
and errors.

This is the source of truth. Everything else in the package either implements a
port here or orchestrates ports defined here. Evaluation *report* types
(``LinkageMetrics``, ``BlockingMetrics``, ``ClusteringMetrics``,
``ThresholdSweep``, ``AdjustedMetrics``) deliberately live in
``denselinkage.metrics``, not here — nothing in ``core`` depends on them
(ADR-0002).
"""

from denselinkage.core.errors import (
    DenseLinkageError,
    DimensionMismatch,
    DuplicateRecordId,
    EmptySource,
    IncompatibleStore,
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
    ClusteringResult,
    LabeledPairs,
    LinkageResult,
    TrainingPairs,
)

__all__ = [
    "Blocker",
    "BlockingIndex",
    "CandidatePair",
    "Clusterer",
    "ClusteringResult",
    "DenseLinkageError",
    "DimensionMismatch",
    "DuplicateRecordId",
    "Embedder",
    "EmptySource",
    "Filter",
    "IncompatibleStore",
    "InvalidTopK",
    "LabeledPairs",
    "LinkageResult",
    "MatchDecision",
    "MatchError",
    "Matcher",
    "Record",
    "RecordId",
    "SearchableIndex",
    "Serializer",
    "Source",
    "Trainer",
    "TrainingPairs",
    "UnknownIdColumn",
    "VectorIndex",
]
