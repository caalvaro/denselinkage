"""Token-set Jaccard matcher for the denselinkage entity-resolution library.

A self-contained :class:`denselinkage.core.ports.Matcher` adapter that decides
each candidate pair by the Jaccard similarity of the whitespace token sets of
the two records' (lowercased) text.
"""

from __future__ import annotations

from collections.abc import Sequence

from denselinkage.core.models import CandidatePair, MatchDecision, MatchError
from denselinkage.core.ports import Matcher


class TokenJaccardMatcher(Matcher):
    """Match candidate pairs by token-set Jaccard similarity.

    For each pair, the two records' ``text`` is lowercased and split on
    whitespace into a set of tokens. The Jaccard similarity is::

        J = |tokens_a & tokens_b| / |tokens_a | tokens_b|

    A pair is a match when ``J >= threshold``. If either record's text yields
    no tokens, the pair cannot be decided and a :class:`MatchError` is produced
    for that position instead of a decision.
    """

    threshold: float

    def __init__(self, *, threshold: float = 0.5) -> None:
        """Create a matcher.

        :param threshold: Minimum Jaccard similarity (inclusive) for a pair to
            be considered a match. Keyword-only; defaults to ``0.5``.
        """
        self.threshold = threshold

    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision | MatchError]:
        """Decide each candidate pair, returning one outcome per input pair.

        Outcomes are aligned by position with ``pairs``. A pair where either
        record has no whitespace tokens yields a :class:`MatchError`; all other
        pairs yield a :class:`MatchDecision`.
        """
        outcomes: list[MatchDecision | MatchError] = []
        for pair in pairs:
            tokens_a: set[str] = set(pair.record_a.text.lower().split())
            tokens_b: set[str] = set(pair.record_b.text.lower().split())
            if not tokens_a or not tokens_b:
                outcomes.append(MatchError(reason="record text has no tokens"))
                continue
            union: set[str] = tokens_a | tokens_b
            intersection: set[str] = tokens_a & tokens_b
            similarity: float = len(intersection) / len(union)
            outcomes.append(
                MatchDecision(
                    is_match=similarity >= self.threshold,
                    confidence=similarity,
                )
            )
        return outcomes
