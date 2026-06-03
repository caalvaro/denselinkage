"""Embedders. ``SentenceTransformerEmbedder`` is the heavy adapter (extra:
``[sentence-transformers]``).

This package is a façade: implementations live in sibling modules; import the
public names here.
"""

from denselinkage.embedding.hashed_ngram_embedder import HashedNGramEmbedder
from denselinkage.embedding.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)

__all__ = ["HashedNGramEmbedder", "SentenceTransformerEmbedder"]
