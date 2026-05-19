"""Phase A0.5 contract-hardening tests.

Structure-stage: assert the *shape* the brief pins (bodies are still ``...``).
Covers the exception taxonomy (2.1), D1 ordering semantics, the
``pair_completeness_at_k`` kwarg (2.2), the ``BlockingMetrics`` constructor
shape (2.4), the D2 ``CandidatePair`` default, and the named reader seam (2.3).
"""

import dataclasses
import inspect

from denselinkage import core, metrics
from denselinkage._reader import RecordReader
from denselinkage.core import errors
from denselinkage.core.errors import DenseLinkageError
from denselinkage.core.models import CandidatePair
from denselinkage.core.results import BlockingMetrics, LabeledPairs

TAXONOMY = (
    "DenseLinkageError",
    "UnknownIdColumn",
    "EmptySource",
    "DuplicateRecordId",
    "DimensionMismatch",
    "InvalidTopK",
)


def test_exception_taxonomy_exists_rooted_and_exported() -> None:
    for name in TAXONOMY:
        assert name in core.__all__, f"{name} not exported from denselinkage.core"
        exc = getattr(errors, name)
        assert isinstance(exc, type) and issubclass(exc, Exception)
        assert issubclass(exc, DenseLinkageError)
        assert getattr(core, name) is exc


def test_d1_labeledpairs_is_ordered_and_not_symmetrized() -> None:
    lp = LabeledPairs(pairs=frozenset({("A1", "B1")}))
    assert ("A1", "B1") in lp.pairs
    # Order is meaningful for `link`; construction must NOT symmetrize.
    assert ("B1", "A1") not in lp.pairs


def test_2_2_pair_completeness_at_k_gold_is_keyword_only() -> None:
    sig = inspect.signature(metrics.pair_completeness_at_k)
    assert sig.parameters["gold"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["k"].kind is inspect.Parameter.KEYWORD_ONLY


def test_2_4_blocking_metrics_constructor_shape() -> None:
    field_names = {f.name for f in dataclasses.fields(BlockingMetrics)}
    assert "pc" in field_names and "_pc" not in field_names
    assert hasattr(BlockingMetrics, "from_pc_map")


def test_d2_candidate_pair_similarity_optional_with_default() -> None:
    fields = {f.name: f for f in dataclasses.fields(CandidatePair)}
    assert fields["similarity_score"].default is None


def test_2_3_record_reader_seam_is_named_and_documented() -> None:
    import denselinkage._reader as reader_mod

    assert hasattr(RecordReader, "read")
    assert reader_mod.__doc__ is not None
    # The seam's documented responsibility includes serializer=None resolution.
    assert "serializer" in reader_mod.__doc__.lower()
