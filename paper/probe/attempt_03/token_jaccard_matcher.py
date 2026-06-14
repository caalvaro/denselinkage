"""TokenJaccardMatcher: a Matcher adapter using token-set Jaccard similarity.

This module is a self-contained third-party adapter for the denselinkage
entity-resolution library. It decides each candidate pair by the Jaccard
similarity of the whitespace-delimited token sets of the two records' text,
compared case-insensitively.
"""

from __future__ import annotations

from collections.abc import Sequence

from denselinkage.core.models import (
    CandidatePair,
    MatchDecision,
    MatchError,
)
from denselinkage.core.ports import Matcher


def _tokens(text: str) -> set[str]:
    """Return the set of lowercased whitespace-delimited tokens in *text*."""
    return set(text.lower().split())


class TokenJaccardMatcher(Matcher):
    """Match candidate pairs by Jaccard similarity of their token sets.

    For each pair, the Jaccard similarity ``J`` is computed as
    ``|A & B| / |A | B|`` where ``A`` and ``B`` are the sets of lowercased
    whitespace tokens of the two records' text. A pair is a match when
    ``J >= threshold``. If either record's text has no tokens, the pair
    yields a :class:`MatchError` rather than a guessed decision.
    """

    _threshold: float

    def __init__(self, *, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision | MatchError]:
        """Decide each pair, returning one aligned outcome per input pair."""
        outcomes: list[MatchDecision | MatchError] = []
        for pair in pairs:
            tokens_a = _tokens(pair.record_a.text)
            tokens_b = _tokens(pair.record_b.text)
            if not tokens_a or not tokens_b:
                outcomes.append(MatchError(reason="record text has no tokens"))
                continue
            union = tokens_a | tokens_b
            jaccard = len(tokens_a & tokens_b) / len(union)
            outcomes.append(
                MatchDecision(
                    is_match=jaccard >= self._threshold,
                    confidence=jaccard,
                )
            )
        return outcomes
