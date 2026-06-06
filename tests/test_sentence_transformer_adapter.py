"""``SentenceTransformerEmbedder`` behaviour (extra: ``[sentence-transformers]``).

Marked ``adapter`` **and** ``slow``: the first run downloads the (small) model, so
this is excluded from both the dependency-free coverage run and the default fast
selection, and runs in the dedicated ``adapter-tests`` CI job (which caches the
model). The contract — float32 **unit** vectors, stable ``model_id`` /
``embedding_dim`` — is what the numpy / FAISS indexes and ``similarity_threshold``
rely on; the integration test drives the full headline path (ST -> FAISS blocking).
"""

import numpy as np
import pytest

from denselinkage.blocking import DenseBlocker
from denselinkage.core.models import Record
from denselinkage.embedding import SentenceTransformerEmbedder
from denselinkage.indexing import FaissFlatIndex

pytestmark = [pytest.mark.adapter, pytest.mark.slow]

_MODEL = "all-MiniLM-L6-v2"
_DIM = 384


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(_MODEL)


def test_encode_returns_float32_unit_vectors(
    embedder: SentenceTransformerEmbedder,
) -> None:
    vectors = embedder.encode(["Apple Inc", "Microsoft Corp", "Google LLC"])
    assert vectors.shape == (3, _DIM)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)  # normalize_embeddings -> cosine == IP


def test_model_id_and_embedding_dim(embedder: SentenceTransformerEmbedder) -> None:
    assert embedder.model_id == _MODEL
    assert embedder.embedding_dim == _DIM


def test_encode_captures_meaning_not_characters(
    embedder: SentenceTransformerEmbedder,
) -> None:
    vectors = embedder.encode(["Apple Inc", "Apple Incorporated", "tractor supply"])
    related = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    assert related > unrelated


def test_sentence_transformer_to_faiss_dense_blocking(
    embedder: SentenceTransformerEmbedder,
) -> None:
    reference = [
        Record(id="A1", text="Apple Inc"),
        Record(id="A2", text="Microsoft Corporation"),
    ]
    blocker = DenseBlocker(embedder=embedder, vector_index=FaissFlatIndex(), top_k=1)
    index = blocker.build(reference)

    pairs = index.query([Record(id="B1", text="Apple Incorporated")], top_k=1)

    assert len(pairs) == 1
    assert pairs[0].record_a.id == "A1"  # nearest neighbour is Apple, not Microsoft
    assert pairs[0].record_b.id == "B1"
    assert pairs[0].similarity_score is not None
