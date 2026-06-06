"""Fast, **mocked** unit tests for ``SentenceTransformerEmbedder`` (extra:
``[sentence-transformers]``).

These patch ``sentence_transformers.SentenceTransformer`` so the adapter's *own*
wiring is pinned deterministically and offline — no model download. They assert
the part the real-model integration test (``test_sentence_transformer_adapter``)
cannot easily check: that the adapter actually requests **L2-normalized numpy
float32** output (the cosine-parity invariant) and threads ``batch_size`` through.
Adapter-marked (they need ``sentence_transformers`` importable to patch it) but
**not** ``slow``.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from denselinkage.embedding import SentenceTransformerEmbedder

pytestmark = pytest.mark.adapter

_PATCH_TARGET = "sentence_transformers.SentenceTransformer"


def test_encode_requests_cosine_ready_float32() -> None:
    fake_model = MagicMock()
    fake_model.get_embedding_dimension.return_value = 3
    # float64, non-unit — the adapter must cast to float32 and ask for cosine.
    fake_model.encode.return_value = np.array([[0.0, 3.0, 4.0]], dtype=np.float64)

    with patch(_PATCH_TARGET, return_value=fake_model) as ctor:
        embedder = SentenceTransformerEmbedder("dummy-model")
        out = embedder.encode(["hi"], batch_size=8)

    ctor.assert_called_once_with("dummy-model")
    assert embedder.model_id == "dummy-model"
    assert embedder.embedding_dim == 3

    kwargs = fake_model.encode.call_args.kwargs
    assert kwargs["normalize_embeddings"] is True  # the cosine-parity invariant
    assert kwargs["convert_to_numpy"] is True
    assert kwargs["batch_size"] == 8
    assert out.dtype == np.float32


def test_encode_defaults_batch_size_when_unset() -> None:
    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((1, 2), dtype=np.float32)
    with patch(_PATCH_TARGET, return_value=fake_model):
        SentenceTransformerEmbedder("dummy").encode(["x"])
    assert fake_model.encode.call_args.kwargs["batch_size"] == 32


def test_embedding_dim_falls_back_to_legacy_method() -> None:
    # Older sentence-transformers (<rename) expose only the legacy accessor.
    legacy_model: Any = MagicMock(spec=["encode", "get_sentence_embedding_dimension"])
    legacy_model.get_sentence_embedding_dimension.return_value = 7
    with patch(_PATCH_TARGET, return_value=legacy_model):
        embedder = SentenceTransformerEmbedder("legacy")
    assert embedder.embedding_dim == 7
