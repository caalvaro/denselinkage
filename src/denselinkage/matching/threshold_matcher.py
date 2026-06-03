"""``ThresholdMatcher`` — the dependency-free reference matcher."""

from collections.abc import Sequence

from denselinkage.core.models import CandidatePair, MatchDecision, MatchError
from denselinkage.core.ports import Matcher


class ThresholdMatcher(Matcher):
    """Dependency-free reference matcher; gates on the carried similarity."""

    def __init__(self, *, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision | MatchError]:
        outcomes: list[MatchDecision | MatchError] = []
        for pair in pairs:
            if pair.similarity_score is None:
                outcomes.append(
                    MatchError(
                        reason="ThresholdMatcher needs a similarity score; the "
                        "candidate pair carried none"
                    )
                )
            else:
                outcomes.append(
                    MatchDecision(is_match=pair.similarity_score >= self._threshold)
                )
        return outcomes
