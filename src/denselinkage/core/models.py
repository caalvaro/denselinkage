"""Core domain value objects."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from denselinkage.core.ports import Serializer

RecordId = str


@dataclass(frozen=True, slots=True)
class Record:
    id: RecordId
    text: str
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CandidatePair:
    record_a: Record
    record_b: Record
    similarity_score: float


@dataclass(frozen=True, slots=True)
class MatchDecision:
    """``is_match is None`` => undecided/errored."""

    is_match: bool | None
    confidence: float | None = None
    rationale: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Source:
    """Data-bound config travelling with a frame. ``serializer=None`` => the
    package default whole-row serializer."""

    frame: "pd.DataFrame"
    id_column: str
    serializer: "Serializer | None" = None
