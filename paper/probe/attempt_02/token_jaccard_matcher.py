"""TokenJaccardMatcher: a Matcher adapter using token-set Jaccard similarity.

This module is self-contained. It imports only from the denselinkage public
contract (denselinkage.core.models, denselinkage.core.ports) and the Python
standard library.
"""

from __future__ import annotations

from collections.abc import Sequence

from denselinkage.core.models import (
    CandidatePair,
    MatchDecision,
    MatchError,
)
from denselinkage.core.ports import Matcher


def _tokenize(text: str) -> set[str]:
    """Return the set of whitespace-separated, lowercased tokens of ``text``."""
    return set(text.lower().split())


class TokenJaccardMatcher(Matcher):
    """Decide each pair by the Jaccard similarity of its records' token sets.

    For a pair, ``J = |A & B| / |A | B|`` where ``A`` and ``B`` are the sets of
    whitespace tokens (lowercased) of each record's ``text``. A pair is a match
    when ``J >= threshold``; the confidence reported is ``J`` itself. If either
    record contributes no tokens, the pair yields a :class:`MatchError`.
    """

    def __init__(self, *, threshold: float = 0.5) -> None:
        self._threshold: float = threshold

    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision | MatchError]:
        outcomes: list[MatchDecision | MatchError] = []
        for pair in pairs:
            tokens_a: set[str] = _tokenize(pair.record_a.text)
            tokens_b: set[str] = _tokenize(pair.record_b.text)
            if not tokens_a or not tokens_b:
                outcomes.append(
                    MatchError(
                        reason="record text has no tokens;"
                        "cannot compute Jaccard similarity"
                    )
                )
                continue
            union: set[str] = tokens_a | tokens_b
            jaccard: float = len(tokens_a & tokens_b) / len(union)
            outcomes.append(
                MatchDecision(
                    is_match=jaccard >= self._threshold,
                    confidence=jaccard,
                )
            )
        return outcomes
