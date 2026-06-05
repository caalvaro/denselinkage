"""The heavy adapters (faiss / sentence-transformers / langchain) are declared
but deferred: using them raises ``NotImplementedError`` rather than silently
returning ``None``.

These guards need no extras — the raise precedes any backend import — so the
tests run on the dependency-free stack and are deliberately NOT ``adapter``-marked.
"""

import numpy as np
import pytest

from denselinkage.embedding import SentenceTransformerEmbedder
from denselinkage.indexing import FaissFlatIndex, FaissSearchableIndex
from denselinkage.matching import LangChainMatcher

_VECTORS = np.zeros((1, 2), dtype=np.float32)


def test_sentence_transformer_embedder_is_deferred() -> None:
    with pytest.raises(NotImplementedError, match="future release"):
        SentenceTransformerEmbedder("all-MiniLM-L6-v2")


def test_langchain_matcher_is_deferred() -> None:
    with pytest.raises(NotImplementedError, match="future release"):
        LangChainMatcher(llm=None, prompt="")


def test_faiss_flat_index_build_is_deferred() -> None:
    with pytest.raises(NotImplementedError, match="future release"):
        FaissFlatIndex().build(_VECTORS, ["a"])


def test_faiss_searchable_index_search_is_deferred() -> None:
    with pytest.raises(NotImplementedError, match="future release"):
        FaissSearchableIndex().search(_VECTORS, top_k=1)


def test_faiss_searchable_index_extended_is_deferred() -> None:
    with pytest.raises(NotImplementedError, match="future release"):
        FaissSearchableIndex().extended(_VECTORS, ["a"])
