"""Token-set Jaccard matcher adapter for denselinkage.

This module provides :class:`TokenJaccardMatcher`, a first-party-style adapter
implementing the :class:`denselinkage.core.ports.Matcher` protocol. It decides
each candidate pair by the Jaccard similarity of the whitespace token sets of
the two records' (lowercased) text.
"""

from __future__ import annotations

from collections.abc import Sequence

from denselinkage.core.models import CandidatePair, MatchDecision, MatchError
from denselinkage.core.ports import Matcher

__all__ = ["TokenJaccardMatcher"]


def _tokens(text: str) -> frozenset[str]:
    """Return the set of lowercased whitespace-delimited tokens of ``text``."""
    return frozenset(text.lower().split())


class TokenJaccardMatcher(Matcher):
    """Decide pairs by Jaccard similarity of their whitespace token sets.

    For each candidate pair, the two records' text values are lowercased and
    split on whitespace into token sets. The Jaccard similarity is

        ``J = |tokens_a & tokens_b| / |tokens_a | tokens_b|``

    and the pair is a match when ``J >= threshold``. If either record's text
    has no tokens (e.g. it is empty or whitespace-only), the union is empty and
    ``J`` is undefined; such a pair yields a :class:`MatchError` rather than a
    guessed decision.
    """

    threshold: float

    def __init__(self, *, threshold: float = 0.5) -> None:
        """Create a matcher.

        Args:
            threshold: Minimum Jaccard similarity (inclusive) for a match.
        """
        self.threshold = threshold

    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision | MatchError]:
        """Decide each pair, returning one outcome per input in order."""
        outcomes: list[MatchDecision | MatchError] = []
        for pair in pairs:
            tokens_a = _tokens(pair.record_a.text)
            tokens_b = _tokens(pair.record_b.text)
            if not tokens_a or not tokens_b:
                outcomes.append(MatchError(reason="record text has no tokens"))
                continue
            union = tokens_a | tokens_b
            similarity = len(tokens_a & tokens_b) / len(union)
            outcomes.append(
                MatchDecision(
                    is_match=similarity >= self.threshold,
                    confidence=similarity,
                )
            )
        return outcomes
