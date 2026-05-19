"""Reference serializers."""

from collections.abc import Mapping, Sequence
from typing import Any


class TemplateSerializer:
    """``"Name: {name}"``; ``column_mapping`` maps this source's columns onto
    the template variables."""

    def __init__(
        self, template: str, *, column_mapping: Mapping[str, str] | None = None
    ) -> None: ...

    def serialize(self, record: Mapping[str, Any]) -> str: ...


class FieldwiseSerializer:
    def __init__(self, fields: Sequence[str], sep: str = " | ") -> None: ...

    def serialize(self, record: Mapping[str, Any]) -> str: ...


class WholeRowSerializer:
    """Package default when ``Source(serializer=None)``."""

    def serialize(self, record: Mapping[str, Any]) -> str: ...


def default_serializer() -> WholeRowSerializer: ...


__all__ = [
    "FieldwiseSerializer",
    "TemplateSerializer",
    "WholeRowSerializer",
    "default_serializer",
]
