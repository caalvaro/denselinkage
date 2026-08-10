"""EditDistanceMatcher: a Matcher adapter for denselinkage using difflib similarity."""

from __future__ import annotations

import difflib
from collections.abc import Sequence

from denselinkage.core.models import CandidatePair, MatchDecision, MatchError
from denselinkage.core.ports import Matcher


class EditDistanceMatcher(Matcher):
    """Decide record pairs by the difflib SequenceMatcher ratio of their texts."""

    def __init__(self, *, threshold: float = 0.6) -> None:
        self._threshold = threshold

    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision | MatchError]:
        results: list[MatchDecision | MatchError] = []
        for pair in pairs:
            a = pair.record_a.text
            b = pair.record_b.text
            if not a or not b:
                results.append(MatchError(reason="one or both records have empty text"))
            else:
                s = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
                results.append(
                    MatchDecision(is_match=s >= self._threshold, confidence=s)
                )
        return results
