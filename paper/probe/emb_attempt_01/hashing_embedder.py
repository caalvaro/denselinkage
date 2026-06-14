"""Deterministic character n-gram feature-hashing Embedder adapter.

This module provides :class:`HashingEmbedder`, an :class:`Embedder`
implementation that maps text to a fixed-width dense vector via stable
character n-gram feature hashing followed by L2 normalisation. It depends
only on :mod:`denselinkage.core.ports`, :mod:`numpy`, and the standard
library.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from denselinkage.core.ports import Embedder

Vectors = npt.NDArray[np.float32]


class HashingEmbedder(Embedder):
    """Embed text via deterministic character n-gram feature hashing.

    Each text is lowercased and split into overlapping character n-grams.
    Every n-gram is hashed with :func:`zlib.crc32` (a stable, unsalted hash)
    into one of ``n_features`` buckets; bucket counts form the raw vector,
    which is then L2-normalised. Texts with no n-grams map to an all-zero row.
    """

    def __init__(self, *, n_features: int = 256, ngram: int = 3) -> None:
        if n_features < 1:
            raise ValueError("n_features must be a positive integer")
        if ngram < 1:
            raise ValueError("ngram must be a positive integer")
        self._n_features: int = n_features
        self._ngram: int = ngram

    @property
    def model_id(self) -> str:
        """Stable identifier of this embedding configuration."""
        return f"hashing-{self._ngram}gram-{self._n_features}"

    @property
    def embedding_dim(self) -> int:
        """Output width of :meth:`encode`."""
        return self._n_features

    def _embed_one(self, text: str) -> Vectors:
        """Embed a single text into an L2-normalised feature-hash row."""
        row: Vectors = np.zeros(self._n_features, dtype=np.float32)
        lowered = text.lower()
        n = self._ngram
        if len(lowered) < n:
            return row
        for i in range(len(lowered) - n + 1):
            gram = lowered[i : i + n]
            bucket = zlib.crc32(gram.encode("utf-8")) % self._n_features
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
        """Map each text to an L2-normalised feature-hash vector.

        Returns an array of shape ``(len(texts), n_features)`` with dtype
        ``float32``. ``batch_size`` and ``show_progress`` are accepted for
        protocol conformance but do not affect the deterministic result.
        """
        out: Vectors = np.zeros((len(texts), self._n_features), dtype=np.float32)
        for idx, text in enumerate(texts):
            out[idx] = self._embed_one(text)
        return out
