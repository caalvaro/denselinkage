"""``SentenceTransformerEmbedder`` — heavy adapter (extra:
``[sentence-transformers]``)."""

from collections.abc import Sequence

import numpy as np

from denselinkage._optional import require
from denselinkage.core.ports import Embedder, Vectors


class SentenceTransformerEmbedder(Embedder):
    """Semantic embedder over a ``sentence-transformers`` checkpoint (extra:
    ``[sentence-transformers]``).

    Where the lexical ``HashedNGramEmbedder`` recovers typos and abbreviations,
    this captures *meaning*: it can link semantic renames (e.g. *Google* /
    *Alphabet*) that share no characters. Encodes with
    ``normalize_embeddings=True`` so the unit-vector inner product equals cosine —
    the similarity the numpy / FAISS indexes and ``similarity_threshold`` are
    defined against. The model loads eagerly at construction, so a bad
    ``model_name`` fails fast.
    """

    def __init__(self, model_name: str) -> None:
        require("sentence_transformers")
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        # ``get_sentence_embedding_dimension`` was renamed to
        # ``get_embedding_dimension``; support both across the ``>=3.0`` range
        # (the new name avoids a deprecation warning and survives the old name's
        # eventual removal).
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if get_dim is None:
            get_dim = self._model.get_sentence_embedding_dimension
        return int(get_dim())

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> Vectors:
        embeddings = self._model.encode(
            list(texts),
            batch_size=batch_size or 32,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # unit vectors -> inner product == cosine
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
