"""Internal Source -> Record materialization seam.

This names the one orchestration boundary that was otherwise implicit: every
other architectural boundary in the package is an explicit port, but turning a
``Source`` (frame + ``id_column`` + ``Serializer``) into ``Sequence[Record]``
had no named owner. It is **not** a public port — underscore-private, absent
from the prelude — but it is named and its responsibility is documented so the
contract has no unspecified seam.

Responsibilities (the documented contract):

1. Resolve ``Source.serializer is None`` to
   :func:`denselinkage.serialize.default_serializer` (a ``WholeRowSerializer``).
2. Apply the serializer (including its ``column_mapping`` when present) to
   each row to produce ``Record.text``; carry the raw row in ``Record.fields``.
3. Validate, raising the ``denselinkage.core.errors`` taxonomy:
   ``UnknownIdColumn`` (``id_column`` absent), ``EmptySource`` (no rows),
   ``DuplicateRecordId`` (non-unique ids).
"""

from collections.abc import Sequence
from typing import Any

from denselinkage.core.errors import DuplicateRecordId, EmptySource, UnknownIdColumn
from denselinkage.core.models import Record, RecordId, Source
from denselinkage.serialize import default_serializer


class RecordReader:
    """Materializes a ``Source`` into ``Record``s per the module contract."""

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


__all__ = ["RecordReader"]
