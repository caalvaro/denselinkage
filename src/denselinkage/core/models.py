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
    """A pair to be matched.

    ``similarity_score`` is
    ``float | None``. A dense ``Blocker`` always sets it; pairs supplied to
    ``DenseLinker.match_pairs`` from external / rule-based blocking have no
    similarity and use ``None``. ``CandidatePair`` is ``frozen`` so this shape
    is fixed before freeze — never tightened later (extend-never-modify).
    """

    record_a: Record
    record_b: Record
    similarity_score: float | None = None


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
    """Data-bound config travelling with a frame.

    The Source -> ``Sequence[Record]`` materialization is the one
    orchestration boundary that is otherwise implicit; it is named: the
    internal ``denselinkage._reader.RecordReader`. It resolves ``serializer=None`` to
    :func:`denselinkage.serializing.default_serializer` (a
    ``WholeRowSerializer``), applies the serializer's ``column_mapping`` when
    present, and validates the frame — raising the
    ``denselinkage.core.errors`` taxonomy (``UnknownIdColumn`` if
    ``id_column`` is absent, ``EmptySource`` if no rows, ``DuplicateRecordId``
    on duplicate ids).
    """

    frame: "pd.DataFrame"
    id_column: str
    serializer: "Serializer | None" = None
