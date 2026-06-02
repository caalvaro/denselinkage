"""``WholeRowSerializer`` + ``default_serializer`` — the ``serializer=None``
resolution target."""

from collections.abc import Mapping
from typing import Any

from denselinkage.core.ports import Serializer


class WholeRowSerializer(Serializer):
    """Package default when ``Source(serializer=None)``. Joins the row's values
    (in column order) with `` | `` — a deterministic, lexical-friendly rendering
    suited to the dependency-free default stack."""

    def serialize(self, record: Mapping[str, Any]) -> str:
        return " | ".join(str(value) for value in record.values())


def default_serializer() -> WholeRowSerializer:
    return WholeRowSerializer()
