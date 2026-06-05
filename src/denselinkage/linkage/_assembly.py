"""Shared pipeline assembly: candidate pairs + matcher -> ``LinkageResult``.

Used by both ``LinkageIndex.query`` (the blocked path) and
``DenseLinker.match_pairs`` (the matcher-only path) so the matcher contract — one
outcome per pair, soft per-pair ``MatchError``s, the length-alignment check —
lives in exactly one place.
"""

from collections.abc import Sequence

from denselinkage.core.models import CandidatePair, MatchDecision, MatchError
from denselinkage.core.ports import Matcher
from denselinkage.core.results import LinkageResult


def assemble_linkage_result(
    pairs: Sequence[CandidatePair], matcher: Matcher
) -> LinkageResult:
    """Run ``matcher`` over ``pairs`` and split the outcomes into the decisions
    and errors channels of a ``LinkageResult``.

    Raises ``ValueError`` (API misuse) if the matcher does not return exactly one
    outcome per input pair, aligned by position.
    """
    outcomes = matcher.match(pairs)
    if len(outcomes) != len(pairs):
        raise ValueError(
            f"matcher returned {len(outcomes)} outcomes for {len(pairs)} "
            "pairs; Matcher.match must return exactly one outcome per input "
            "pair, aligned by position"
        )
    decisions: list[tuple[CandidatePair, MatchDecision]] = []
    errors: list[tuple[CandidatePair, MatchError]] = []
    for pair, outcome in zip(pairs, outcomes, strict=True):
        if isinstance(outcome, MatchError):
            errors.append((pair, outcome))
        else:
            decisions.append((pair, outcome))
    return LinkageResult(decisions=tuple(decisions), errors=tuple(errors))
