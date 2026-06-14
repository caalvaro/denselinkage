"""Deterministic character n-gram feature-hashing Embedder adapter.

A self-contained :class:`HashingEmbedder` that maps texts to fixed-width,
L2-normalised dense float32 vectors using a stable hash (``zlib.crc32``), so
results are reproducible across processes and machines (unlike the builtin
salted ``hash()``).
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from denselinkage.core.ports import Embedder

Vectors = npt.NDArray[np.float32]


class HashingEmbedder(Embedder):
    """Embed texts via deterministic character n-gram feature hashing.

    Each text is lowercased, decomposed into overlapping character n-grams,
    and every n-gram is hashed (with :func:`zlib.crc32`) into one of
    ``n_features`` buckets. Bucket counts form a raw vector that is then
    L2-normalised. A text that yields no n-grams (e.g. one shorter than
    ``ngram``, or empty) maps to an all-zero row.
    """

    def __init__(self, *, n_features: int = 256, ngram: int = 3) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be a positive integer")
        if ngram <= 0:
            raise ValueError("ngram must be a positive integer")
        self._n_features: int = n_features
        self._ngram: int = ngram

    @property
    def model_id(self) -> str:
        """Stable identifier of this embedding configuration."""
        return f"hashing-{self._ngram}gram-{self._n_features}"

    @property
    def embedding_dim(self) -> int:
        """Output width of :meth:`encode` (equal to ``n_features``)."""
        return self._n_features

    def _embed_one(self, text: str) -> Vectors:
        """Hash a single text into an L2-normalised float32 row vector."""
        row: Vectors = np.zeros(self._n_features, dtype=np.float32)
        lowered = text.lower()
        n = self._ngram
        for i in range(len(lowered) - n + 1):
            ngram = lowered[i : i + n]
            digest = zlib.crc32(ngram.encode("utf-8"))
            bucket = digest % self._n_features
            row[bucket] += np.float32(1.0)
        norm = float(np.linalg.norm(row))
        if norm > 0.0:
            row /= np.float32(norm)
        return row

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> Vectors:
        """Encode ``texts`` into a ``(len(texts), n_features)`` float32 array.

        ``batch_size`` and ``show_progress`` are accepted for protocol
        conformance; the computation is independent per row, so they do not
        affect the (deterministic) result.
        """
        out: Vectors = np.zeros((len(texts), self._n_features), dtype=np.float32)
        for idx, text in enumerate(texts):
            out[idx] = self._embed_one(text)
        return out
