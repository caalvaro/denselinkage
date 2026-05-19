"""Typed results / metrics / labels the API returns."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from denselinkage.core.models import CandidatePair, MatchDecision

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class LabeledPairs:
    """The gold set of true matches — one type everywhere."""

    pairs: frozenset[tuple[str, str]]

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]]) -> "LabeledPairs": ...

    @classmethod
    def from_frame(
        cls, frame: "pd.DataFrame", *, left_id: str, right_id: str
    ) -> "LabeledPairs": ...


@dataclass(frozen=True, slots=True)
class LinkageResult:
    """All candidate pairs with their match decisions."""

    decisions: tuple[tuple[CandidatePair, MatchDecision], ...]

    def to_frame(self) -> "pd.DataFrame":
        """Fixed schema: left_id, right_id, match, confidence, reason,
        similarity — independent of input column names."""
        ...


@dataclass(frozen=True, slots=True)
class LinkageMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    n_gold: int
    n_errors: int = 0

    @property
    def precision(self) -> float: ...

    @property
    def recall(self) -> float: ...

    @property
    def f1(self) -> float: ...


@dataclass(frozen=True, slots=True)
class BlockingMetrics:
    """Pair-completeness@k via ``pc_at(k)``."""

    _pc: Mapping[int, float]
    n_gold: int

    def pc_at(self, k: int) -> float: ...


@dataclass(frozen=True, slots=True)
class Clustering:
    labels: Mapping[str, int]

    @property
    def n_clusters(self) -> int: ...

    def to_frame(self) -> "pd.DataFrame": ...
