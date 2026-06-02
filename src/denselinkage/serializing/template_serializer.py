"""``TemplateSerializer`` — format-string serializer with column mapping."""

from collections.abc import Mapping
from typing import Any

from denselinkage.core.ports import Serializer


class TemplateSerializer(Serializer):
    """``"Name: {name}"``; ``column_mapping`` maps this source's columns onto
    the template variables.

    Security: ``template`` is trusted, developer-supplied configuration. It is
    rendered with ``str.format_map`` over each row, so do not build it from
    untrusted input — a hostile template could read attributes of the
    substituted values (the standard ``str.format`` injection vector). Field
    *values* are only substituted, never parsed, so untrusted row data is safe.
    """

    def __init__(
        self, template: str, *, column_mapping: Mapping[str, str] | None = None
    ) -> None:
        self._template = template
        self._column_mapping = dict(column_mapping) if column_mapping else {}

    def serialize(self, record: Mapping[str, Any]) -> str:
        values: dict[str, Any] = dict(record)
        for source_column, template_var in self._column_mapping.items():
            if source_column in record:
                values[template_var] = record[source_column]
        return self._template.format_map(values)
