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


def test_the_seam_is_named_and_documents_serializer_resolution() -> None:
    """The Source -> Record materialization is the one named orchestration seam.

    ``Source.serializer=None`` resolving to the default is the reader's job, and
    the module docstring is where that responsibility is recorded. Rehomed from
    the deleted ``test_a05_contract.py`` (issue #31): it pins a private module's
    docstring, which ADR-0006 puts outside the freeze twice over, so it belongs
    with the reader rather than in a contract file.
    """
    import denselinkage._reader as reader_mod

    assert hasattr(RecordReader, "read")
    assert reader_mod.__doc__ is not None
    assert "serializer" in reader_mod.__doc__.lower()
