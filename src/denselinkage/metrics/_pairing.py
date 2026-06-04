"""Shared D1 pair-identity key for the metrics layer."""

from denselinkage.core.models import RecordId


def pair_key(
    left: RecordId, right: RecordId, *, directed: bool
) -> tuple[RecordId, RecordId] | frozenset[RecordId]:
    """D1 comparison key: ordered for ``link`` (directed), unordered for
    ``dedupe`` (undirected, ``frozenset``)."""
    return (left, right) if directed else frozenset((left, right))
