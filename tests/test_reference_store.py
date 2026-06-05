"""Behavioural tests for the B4 Reference Store: ``LinkageIndex.save`` / ``load``
round-trip, provenance validation, and the reference-stack-only scope."""

import json
from pathlib import Path

import pandas as pd
import pytest

from denselinkage import DenseLinker, LinkageIndex, Source, TemplateSerializer
from denselinkage.blocking import DenseBlockingIndex
from denselinkage.core.errors import IncompatibleStore
from denselinkage.core.models import CandidatePair
from denselinkage.core.results import LinkageResult
from denselinkage.embedding import HashedNGramEmbedder
from denselinkage.matching import ThresholdMatcher


def _left() -> Source:
    df = pd.DataFrame({"id": ["A1", "A2"], "name": ["Apple Inc", "Microsoft Corp"]})
    return Source(df, id_column="id", serializer=TemplateSerializer("{name}"))


def _right() -> Source:
    df = pd.DataFrame({"id": ["B1", "B2"], "name": ["Apple Incorporated", "Microsoft"]})
    return Source(df, id_column="id", serializer=TemplateSerializer("{name}"))


def _embedder() -> HashedNGramEmbedder:
    # Matches DenseLinker.with_defaults' blocker embedder (model_id provenance).
    return HashedNGramEmbedder(n_features=1024, ngram=3)


def _decisions(result: LinkageResult) -> dict[tuple[str, str], bool]:
    return {(p.record_a.id, p.record_b.id): d.is_match for p, d in result.decisions}


def test_save_load_round_trip_reproduces_results(tmp_path: Path) -> None:
    idx = DenseLinker.with_defaults().index(_left())
    before = _decisions(idx.query(_right()))

    idx.save(tmp_path / "store")
    reloaded = LinkageIndex.load(
        tmp_path / "store",
        embedder=_embedder(),
        matcher=ThresholdMatcher(threshold=0.5),
    )
    after = _decisions(reloaded.query(_right()))

    assert before == after
    assert before  # non-empty: the reload genuinely queried


def test_store_round_trips_record_text(tmp_path: Path) -> None:
    idx = DenseLinker.with_defaults().index(_left())
    idx.save(tmp_path / "store")
    reloaded = LinkageIndex.load(
        tmp_path / "store", embedder=_embedder(), matcher=ThresholdMatcher()
    )
    texts = {p.record_a.id: p.record_a.text for p in reloaded.candidates(_right())}
    assert texts["A1"] == "Apple Inc"


def test_store_handles_non_json_field_values(tmp_path: Path) -> None:
    # A Timestamp field is not JSON-native; ``default=str`` keeps save from
    # crashing and the index still reloads.
    df = pd.DataFrame(
        {
            "id": ["A1", "A2"],
            "name": ["Acme", "Beta"],
            "when": [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")],
        }
    )
    src = Source(df, id_column="id")
    idx = DenseLinker.with_defaults().index(src)
    idx.save(tmp_path / "store")  # must not raise
    reloaded = LinkageIndex.load(
        tmp_path / "store", embedder=_embedder(), matcher=ThresholdMatcher()
    )
    assert reloaded.candidates(src)  # loaded and queryable


def test_load_rejects_wrong_embedder_model(tmp_path: Path) -> None:
    DenseLinker.with_defaults().index(_left()).save(tmp_path / "store")
    with pytest.raises(IncompatibleStore, match="hashed-ngram"):
        LinkageIndex.load(
            tmp_path / "store",
            embedder=HashedNGramEmbedder(n_features=512, ngram=3),  # different model_id
            matcher=ThresholdMatcher(),
        )


def test_load_rejects_embedding_dim_mismatch(tmp_path: Path) -> None:
    DenseLinker.with_defaults().index(_left()).save(tmp_path / "store")
    meta_file = tmp_path / "store" / "meta.json"
    meta = json.loads(meta_file.read_text())
    meta["embedding_dim"] = 999  # model_id still matches; the dim does not
    meta_file.write_text(json.dumps(meta))
    with pytest.raises(IncompatibleStore, match="embedding_dim"):
        LinkageIndex.load(
            tmp_path / "store", embedder=_embedder(), matcher=ThresholdMatcher()
        )


def test_load_rejects_unsupported_format(tmp_path: Path) -> None:
    DenseLinker.with_defaults().index(_left()).save(tmp_path / "store")
    meta_file = tmp_path / "store" / "meta.json"
    meta = json.loads(meta_file.read_text())
    meta["format"] = 999
    meta_file.write_text(json.dumps(meta))
    with pytest.raises(IncompatibleStore, match="format"):
        LinkageIndex.load(
            tmp_path / "store", embedder=_embedder(), matcher=ThresholdMatcher()
        )


class _FakeBlockingIndex:
    def query(self, records: object, **kwargs: object) -> list[CandidatePair]:
        return []


class _FakeSearchable:
    def search(self, queries: object, *, top_k: int) -> list[list[tuple[str, float]]]:
        return []

    def extended(self, vectors: object, ids: object) -> "_FakeSearchable":
        return self


def test_save_rejects_non_reference_blocking_index(tmp_path: Path) -> None:
    idx = LinkageIndex(blocking_index=_FakeBlockingIndex(), matcher=ThresholdMatcher())
    with pytest.raises(NotImplementedError, match="numpy stack"):
        idx.save(tmp_path / "store")


def test_save_rejects_non_numpy_searchable(tmp_path: Path) -> None:
    blocking = DenseBlockingIndex(
        searchable=_FakeSearchable(),
        embedder=_embedder(),
        records_by_id={},
        top_k=5,
        similarity_threshold=0.0,
    )
    idx = LinkageIndex(blocking_index=blocking, matcher=ThresholdMatcher())
    with pytest.raises(NotImplementedError, match="numpy stack"):
        idx.save(tmp_path / "store")
