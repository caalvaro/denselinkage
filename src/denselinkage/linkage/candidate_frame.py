"""``candidate_pairs_from_frame`` — build ``CandidatePair`` objects from a frame
of candidate id-pairs, for ``DenseLinker.match_pairs``."""

from typing import TYPE_CHECKING

from denselinkage._reader import RecordReader
from denselinkage.core.models import CandidatePair, Source

if TYPE_CHECKING:
    import pandas as pd


def candidate_pairs_from_frame(
    frame: "pd.DataFrame",
    *,
    left: Source,
    right: Source,
    left_id: str,
    right_id: str,
    similarity: str | None = None,
) -> list[CandidatePair]:
    """Build ``CandidatePair`` objects from a frame of candidate id-pairs and the
    two sources they reference — the ergonomic input to
    ``DenseLinker.match_pairs`` when blocking happened elsewhere (rule-based /
    external / a join).

    Each row pairs ``frame[left_id]`` with ``frame[right_id]``; record text is
    materialized from ``left`` / ``right`` via their serializers (the same text
    ``link`` would produce), so content-aware matchers work. ``similarity`` names
    an optional score column — absent, or a ``NaN`` cell, yields
    ``CandidatePair.similarity_score = None``.

    Raises ``ValueError`` if a named column is absent from ``frame`` or a row
    references an id not present in its source. ``left`` / ``right`` are read
    through the same ``RecordReader`` seam as ``link`` (so its
    ``denselinkage.core.errors`` taxonomy applies to the sources themselves).
    """
    import pandas as pd

    for column, role in ((left_id, "left_id"), (right_id, "right_id")):
        if column not in frame.columns:
            raise ValueError(f"{role} column {column!r} is not in the frame")
    if similarity is not None and similarity not in frame.columns:
        raise ValueError(f"similarity column {similarity!r} is not in the frame")

    left_records = {record.id: record for record in RecordReader().read(left)}
    right_records = {record.id: record for record in RecordReader().read(right)}

    pairs: list[CandidatePair] = []
    for row in frame.to_dict(orient="records"):
        left_key = str(row[left_id])
        right_key = str(row[right_id])
        if left_key not in left_records:
            raise ValueError(f"left id {left_key!r} is not in the left source")
        if right_key not in right_records:
            raise ValueError(f"right id {right_key!r} is not in the right source")
        score: float | None = None
        if similarity is not None and not pd.isna(row[similarity]):
            score = float(row[similarity])
        pairs.append(
            CandidatePair(
                record_a=left_records[left_key],
                record_b=right_records[right_key],
                similarity_score=score,
            )
        )
    return pairs
