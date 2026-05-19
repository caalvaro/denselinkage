"""The dependency-free contract: models, ports, results.

This is the source of truth. Everything else in the package either implements a
port here or orchestrates ports defined here.
"""

from denselinkage.core.models import (
    CandidatePair,
    MatchDecision,
    Record,
    RecordId,
    Source,
)
from denselinkage.core.ports import (
    Blocker,
    Embedder,
    Matcher,
    Serializer,
    VectorIndex,
)
from denselinkage.core.results import (
    BlockingMetrics,
    Clustering,
    LabeledPairs,
    LinkageMetrics,
    LinkageResult,
)

__all__ = [
    "Blocker",
    "BlockingMetrics",
    "CandidatePair",
    "Clustering",
    "Embedder",
    "LabeledPairs",
    "LinkageMetrics",
    "LinkageResult",
    "MatchDecision",
    "Matcher",
    "Record",
    "RecordId",
    "Serializer",
    "Source",
    "VectorIndex",
]
