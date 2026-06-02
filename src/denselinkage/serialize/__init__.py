"""Reference serializers.

First-party adapters subclass their port explicitly (CONTRIBUTING convention;
mypy completeness-checks the implementation) — so these declare
``Serializer``, like every other adapter family.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from denselinkage.core.ports import Serializer


class TemplateSerializer(Serializer):
    """``"Name: {name}"``; ``column_mapping`` maps this source's columns onto
    the template variables."""

    def __init__(
        self, template: str, *, column_mapping: Mapping[str, str] | None = None
    ) -> None: ...

    def serialize(self, record: Mapping[str, Any]) -> str: ...


class FieldwiseSerializer(Serializer):
    def __init__(self, fields: Sequence[str], sep: str = " | ") -> None: ...

    def serialize(self, record: Mapping[str, Any]) -> str: ...


class WholeRowSerializer(Serializer):
    """Package default when ``Source(serializer=None)``."""

    def serialize(self, record: Mapping[str, Any]) -> str: ...


def default_serializer() -> WholeRowSerializer: ...


__all__ = [
    "FieldwiseSerializer",
    "TemplateSerializer",
    "WholeRowSerializer",
    "default_serializer",
]
