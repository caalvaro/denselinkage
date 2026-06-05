"""``HashedNGramEmbedder`` — the dependency-free reference embedder."""

import zlib
from collections.abc import Sequence

import numpy as np

from denselinkage.core.ports import Embedder, Vectors


class HashedNGramEmbedder(Embedder):
    """Dependency-free reference embedder.

    Character n-gram feature hashing: count character n-grams into
    ``n_features`` buckets via a stable hash (``zlib.crc32`` — deterministic
    across processes, unlike builtin ``hash()`` which is ``PYTHONHASHSEED``-
    salted), then L2-normalize so inner product equals cosine. Lexical: it
    recovers abbreviations, punctuation and typos, not semantic renames.
    """

    def __init__(self, n_features: int = 256, ngram: int = 3) -> None:
        self._n_features = n_features
        self._ngram = ngram

    @property
    def model_id(self) -> str:
        return f"hashed-ngram-{self._ngram}-{self._n_features}"

    @property
    def embedding_dim(self) -> int:
        return self._n_features

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> Vectors:
        out = np.zeros((len(texts), self._n_features), dtype=np.float32)
        for i, text in enumerate(texts):
            lowered = text.lower()
            if not lowered.strip():
                continue  # empty / whitespace-only -> zero vector (matches nothing)
            for j in range(max(1, len(lowered) - self._ngram + 1)):
                gram = lowered[j : j + self._ngram]
                bucket = zlib.crc32(gram.encode("utf-8")) % self._n_features
                out[i, bucket] += 1.0
            norm = float(np.linalg.norm(out[i]))
            # L2-normalize so inner product == cosine. Whitespace-only text is
            # skipped above, so a counted row always has norm > 0 (defensive guard).
            if norm > 0.0:  # pragma: no branch
                out[i] /= norm
        return out
