"""``SentenceTransformerEmbedder`` — heavy adapter (extra:
``[sentence-transformers]``)."""

from collections.abc import Sequence

from denselinkage.core.ports import Embedder, Vectors


class SentenceTransformerEmbedder(Embedder):
    """Semantic embedder (extra: ``[sentence-transformers]``).

    Planned for a future release; constructing it raises
    ``NotImplementedError``. Use ``HashedNGramEmbedder`` on the dependency-free
    stack until then.
    """

    def __init__(self, model_name: str) -> None:
        raise NotImplementedError(
            "SentenceTransformerEmbedder is planned for a future release; "
            "use HashedNGramEmbedder on the dependency-free stack"
        )

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
