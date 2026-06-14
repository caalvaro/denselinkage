"""TokenJaccardMatcher: a Matcher adapter using token-set Jaccard similarity.

A self-contained external adapter for the denselinkage entity-resolution
library. It decides each candidate pair by the Jaccard similarity of the
whitespace-delimited, lowercased token sets of the two records' text.
"""

from __future__ import annotations

from denselinkage.core.models import CandidatePair, MatchDecision, MatchError
from denselinkage.core.ports import Matcher


class TokenJaccardMatcher(Matcher):
    """Decide candidate pairs by token-set Jaccard similarity.

    For each pair, the text of both records is lowercased and split on
    whitespace into a set of tokens. The Jaccard similarity is

        J = |tokens_a & tokens_b| / |tokens_a | tokens_b|

    A pair is a match when ``J >= threshold``. If either record's text has
    no tokens, the pair yields a :class:`MatchError` rather than a decision.
    """

    threshold: float

    def __init__(self, *, threshold: float = 0.5) -> None:
        """Create the matcher.

        Args:
            threshold: Minimum Jaccard similarity (inclusive) for a match.
        """
        self.threshold = threshold

    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision | MatchError]:
        """Return one outcome per input pair, aligned by position.

        Args:
            pairs: The candidate pairs to decide.

        Returns:
            A list with exactly one ``MatchDecision`` or ``MatchError`` per
            input pair, in the same order.
        """
        results: list[MatchDecision | MatchError] = []
        for pair in pairs:
            tokens_a: set[str] = set(pair.record_a.text.lower().split())
            tokens_b: set[str] = set(pair.record_b.text.lower().split())
            if not tokens_a or not tokens_b:
                results.append(MatchError(reason="record text has no tokens"))
                continue
            union: set[str] = tokens_a | tokens_b
            intersection: set[str] = tokens_a & tokens_b
            jaccard: float = len(intersection) / len(union)
            results.append(
                MatchDecision(
                    is_match=jaccard >= self.threshold,
                    confidence=jaccard,
                )
            )
        return results


from collections.abc import Sequence  # noqa: E402  (kept local to module tail)
