"""Internal Source -> Record materialization seam.

This names the one orchestration boundary that was otherwise implicit: every
other architectural boundary in the package is an explicit port, but turning a
``Source`` (frame + ``id_column`` + ``Serializer``) into ``Sequence[Record]``
had no named owner. It is **not** a public port — underscore-private, absent
from the prelude — but it is named and its responsibility is documented so the
contract has no unspecified seam.

Responsibilities (the documented contract):

1. Resolve ``Source.serializer is None`` to
   :func:`denselinkage.serializing.default_serializer` (a ``WholeRowSerializer``).
2. Apply the serializer (including its ``column_mapping`` when present) to
   each row to produce ``Record.text``; carry the raw row in ``Record.fields``.
3. Validate, raising the ``denselinkage.core.errors`` taxonomy:
   ``UnknownIdColumn`` (``id_column`` absent), ``EmptySource`` (no rows),
   ``DuplicateRecordId`` (non-unique ids).

This package is a façade: the implementation lives in ``record_reader``.
"""

from denselinkage._reader.record_reader import RecordReader

__all__ = ["RecordReader"]
