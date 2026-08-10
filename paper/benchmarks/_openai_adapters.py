"""Out-of-tree adapters for the OpenAI experiments.

``OpenAIEmbedder`` implements the frozen ``denselinkage.core.ports.Embedder``
protocol (``text-embedding-3-large``) WITHOUT touching the library: a fourth
embedder behind the same port, which is itself QR1 (extensibility) evidence.
Vectors are L2-normalised so the inner-product FAISS/numpy indexes and
``similarity_threshold`` keep their cosine meaning (same contract the
SentenceTransformer adapter relies on).
"""

import os
from collections.abc import Sequence

import numpy as np

from denselinkage.core.ports import Embedder, Vectors

_KNOWN_DIMS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


def load_openai_key() -> None:
    """Set OPENAI_API_KEY from the nearest .env if not already in the env."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        p = os.path.join(d, ".env")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.lower().strip().startswith("openai_api_key"):
                    os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
                    return
        d = os.path.dirname(d)
    raise SystemExit("no openai_api_key in a .env on the path")


class OpenAIEmbedder(Embedder):
    """Semantic embedder over an OpenAI embedding model (out-of-tree adapter)."""

    def __init__(
        self, model_name: str = "text-embedding-3-large", *, batch: int = 256
    ) -> None:
        load_openai_key()
        from langchain_openai import OpenAIEmbeddings

        self._model_name = model_name
        self._batch = batch
        self._client = OpenAIEmbeddings(model=model_name)
        self._dim: int | None = _KNOWN_DIMS.get(model_name)

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._dim = len(self._client.embed_query("dimension probe"))
        return self._dim

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> Vectors:
        items = list(texts)
        bs = batch_size or self._batch
        out: list[list[float]] = []
        for i in range(0, len(items), bs):
            out.extend(self._client.embed_documents(items[i : i + bs]))
            if show_progress:
                print(f"  embedded {min(i + bs, len(items))}/{len(items)}", flush=True)
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).astype(np.float32)
