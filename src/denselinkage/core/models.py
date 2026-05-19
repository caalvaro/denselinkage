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
    """A successful decision — ``is_match`` is always a real ``bool``.

    Failures are NOT represented here. A matcher that cannot decide a pair
    yields the sibling :class:`MatchError` (see ``Matcher.match`` and
    ``LinkageResult.errors``), keeping this type a pure decision.
    ``confidence`` / ``rationale`` are matcher-dependent — both ``None`` for
    matchers that do not produce them (e.g. ``ThresholdMatcher``).
    """

    is_match: bool
    confidence: float | None = None
    rationale: str | None = None


@dataclass(frozen=True, slots=True)
class MatchError:
    """A pair the matcher could not decide (retries exhausted / backend
    error). Carried in ``LinkageResult.errors`` and counted as
    ``LinkageMetrics.n_errors`` — never mixed into tp/fp/fn."""

    reason: str


@dataclass(frozen=True, slots=True)
class Source:
    """Data-bound config travelling with a frame. ``serializer=None`` => the
    package default whole-row serializer."""

    frame: "pd.DataFrame"
    id_column: str
    serializer: "Serializer | None" = None
