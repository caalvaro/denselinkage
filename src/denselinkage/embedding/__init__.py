"""Embedders. ``SentenceTransformerEmbedder`` is the heavy adapter (extra:
``[sentence-transformers]``)."""

from collections.abc import Sequence

from denselinkage.core.ports import Embedder, Vectors


class HashedNGramEmbedder(Embedder):
    """Dependency-free reference embedder."""

    def __init__(self, n_features: int = 256, ngram: int = 3) -> None: ...

    @property
    def model_id(self) -> str: ...

    @property
    def embedding_dim(self) -> int: ...

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> Vectors: ...


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str) -> None: ...

    @property
    def model_id(self) -> str: ...

    @property
    def embedding_dim(self) -> int: ...

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> Vectors: ...


__all__ = ["HashedNGramEmbedder", "SentenceTransformerEmbedder"]
