"""``SimilarityThresholdFilter`` — the dependency-free reference ``Filter``."""

from collections.abc import Sequence

from denselinkage.core.models import CandidatePair
from denselinkage.core.ports import Filter


class SimilarityThresholdFilter(Filter):
    """Keeps pairs whose ``similarity_score >= threshold``.

    Pairs whose ``similarity_score`` is ``None`` (e.g. external / rule-based
    blocking that carries no score) cannot be threshold-judged: kept by default,
    or dropped when ``drop_unscored=True``.
    """

    def __init__(
        self, *, threshold: float = 0.0, drop_unscored: bool = False
    ) -> None: ...

    def filter(self, pairs: Sequence[CandidatePair]) -> list[CandidatePair]: ...
