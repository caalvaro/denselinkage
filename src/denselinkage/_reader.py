"""Internal Source -> Record materialization seam (2.3).

This names the one orchestration boundary that was otherwise implicit: every
other architectural boundary in the package is an explicit port, but turning a
``Source`` (frame + ``id_column`` + ``Serializer``) into ``Sequence[Record]``
had no named owner. It is **not** a public port — underscore-private, absent
from the prelude — but it is named and its responsibility is documented so the
contract has no unspecified seam.

Responsibilities (the documented contract; body lands in A1):

1. Resolve ``Source.serializer is None`` to
   :func:`denselinkage.serialize.default_serializer` (a ``WholeRowSerializer``).
2. Apply the serializer (including its ``column_mapping`` when present) to
   each row to produce ``Record.text``; carry the raw row in ``Record.fields``.
3. Validate, raising the ``denselinkage.core.errors`` taxonomy:
   ``UnknownIdColumn`` (``id_column`` absent), ``EmptySource`` (no rows),
   ``DuplicateRecordId`` (non-unique ids).
"""

from collections.abc import Sequence

from denselinkage.core.models import Record, Source


class RecordReader:
    """Materializes a ``Source`` into ``Record``s per the module contract."""

    def read(self, source: Source) -> Sequence[Record]: ...


__all__ = ["RecordReader"]
