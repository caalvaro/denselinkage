"""``RecordReader`` — the internal Source -> Record materialization seam."""

from collections.abc import Sequence
from typing import Any

from denselinkage.core.errors import DuplicateRecordId, EmptySource, UnknownIdColumn
from denselinkage.core.models import Record, RecordId, Source
from denselinkage.serializing import default_serializer


class RecordReader:
    """Materializes a ``Source`` into ``Record``s per the package contract
    (see the package docstring)."""

    def read(self, source: Source) -> Sequence[Record]:
        frame = source.frame
        if source.id_column not in frame.columns:
            raise UnknownIdColumn(
                f"{source.id_column!r} is not a column of the source frame"
            )
        if len(frame) == 0:
            raise EmptySource("source frame has no rows")
        serializer = (
            source.serializer if source.serializer is not None else default_serializer()
        )
        records: list[Record] = []
        seen: set[RecordId] = set()
        for raw in frame.to_dict(orient="records"):
            row: dict[str, Any] = {str(key): value for key, value in raw.items()}
            record_id = str(row[source.id_column])
            if record_id in seen:
                raise DuplicateRecordId(f"duplicate record id: {record_id!r}")
            seen.add(record_id)
            records.append(
                Record(id=record_id, text=serializer.serialize(row), fields=row)
            )
        return records
