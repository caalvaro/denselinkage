"""``FieldwiseSerializer`` — joins listed fields with a separator."""

from collections.abc import Mapping, Sequence
from typing import Any

from denselinkage.core.ports import Serializer


class FieldwiseSerializer(Serializer):
    def __init__(self, fields: Sequence[str], sep: str = " | ") -> None:
        self._fields = list(fields)
        self._sep = sep

    def serialize(self, record: Mapping[str, Any]) -> str:
        return self._sep.join(str(record.get(field, "")) for field in self._fields)
