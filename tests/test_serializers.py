"""Unit tests for the reference serializers (row mapping -> text)."""

import pytest

from denselinkage.serializing import (
    FieldwiseSerializer,
    TemplateSerializer,
    WholeRowSerializer,
    default_serializer,
)


def test_template_fills_named_fields() -> None:
    serializer = TemplateSerializer("Name: {name}, City: {city}")
    assert (
        serializer.serialize({"name": "Acme", "city": "NYC"}) == "Name: Acme, City: NYC"
    )


def test_template_column_mapping_reconciles_a_differing_schema() -> None:
    serializer = TemplateSerializer(
        "Name: {name}", column_mapping={"company_name": "name"}
    )
    assert serializer.serialize({"company_name": "Acme"}) == "Name: Acme"


def test_template_column_mapping_overrides_the_template_variable() -> None:
    serializer = TemplateSerializer("{name}", column_mapping={"company_name": "name"})
    record = {"name": "ignored", "company_name": "Acme"}
    assert serializer.serialize(record) == "Acme"


def test_template_column_mapping_skips_absent_source_column() -> None:
    # A mapping entry whose source column is not in the row is silently skipped.
    serializer = TemplateSerializer("Name: {name}", column_mapping={"absent": "x"})
    assert serializer.serialize({"name": "Acme"}) == "Name: Acme"


def test_template_missing_variable_raises_key_error() -> None:
    serializer = TemplateSerializer("{name} {absent}")
    with pytest.raises(KeyError):
        serializer.serialize({"name": "Acme"})


def test_fieldwise_joins_in_listed_order() -> None:
    serializer = FieldwiseSerializer(["name", "city"])
    assert serializer.serialize({"city": "NYC", "name": "Acme"}) == "Acme | NYC"


def test_fieldwise_missing_field_becomes_empty() -> None:
    serializer = FieldwiseSerializer(["name", "phone"])
    assert serializer.serialize({"name": "Acme"}) == "Acme | "


def test_fieldwise_custom_separator() -> None:
    serializer = FieldwiseSerializer(["a", "b"], sep=" / ")
    assert serializer.serialize({"a": "1", "b": "2"}) == "1 / 2"


def test_whole_row_joins_all_values_in_column_order() -> None:
    serializer = WholeRowSerializer()
    assert serializer.serialize({"id": "1", "name": "Acme"}) == "1 | Acme"


def test_default_serializer_is_whole_row() -> None:
    assert isinstance(default_serializer(), WholeRowSerializer)
