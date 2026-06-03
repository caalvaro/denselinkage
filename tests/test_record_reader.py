"""Unit tests for the ``RecordReader`` seam (Source -> Record materialization).

The validation taxonomy (UnknownIdColumn / EmptySource / DuplicateRecordId) is
already exercised end to end through ``DenseLinker.link`` in
``test_quickstart_end_to_end``; here we pin the reader-specific behaviour:
serializer resolution, column mapping, and record materialization.
"""

import pandas as pd

from denselinkage._reader import RecordReader
from denselinkage.core.models import Source
from denselinkage.serializing import TemplateSerializer


def test_serializer_none_resolves_to_whole_row() -> None:
    df = pd.DataFrame({"id": ["1"], "name": ["Acme"], "city": ["NYC"]})
    [record] = RecordReader().read(Source(df, id_column="id"))
    assert record.text == "1 | Acme | NYC"  # whole-row default over all values


def test_column_mapping_is_applied_through_the_serializer() -> None:
    df = pd.DataFrame({"id": ["1"], "company_name": ["Acme"]})
    source = Source(
        df,
        id_column="id",
        serializer=TemplateSerializer(
            "Name: {name}", column_mapping={"company_name": "name"}
        ),
    )
    [record] = RecordReader().read(source)
    assert record.text == "Name: Acme"


def test_record_id_is_stringified_and_fields_preserved() -> None:
    df = pd.DataFrame({"id": [1], "name": ["Acme"]})  # integer id column
    [record] = RecordReader().read(Source(df, id_column="id"))
    assert record.id == "1"  # str(1)
    assert record.fields["name"] == "Acme"  # raw row carried on the record
    assert record.text == "1 | Acme"
