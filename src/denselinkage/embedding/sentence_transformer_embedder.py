"""``SentenceTransformerEmbedder`` — heavy adapter (extra:
``[sentence-transformers]``)."""

from collections.abc import Sequence

from denselinkage.core.ports import Embedder, Vectors


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
