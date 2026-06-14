"""Deterministic character n-gram feature-hashing Embedder adapter.

This module provides :class:`HashingEmbedder`, a self-contained implementation
of the :class:`denselinkage.core.ports.Embedder` protocol that maps text to
fixed-width, L2-normalised dense vectors via stable character n-gram hashing.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from denselinkage.core.ports import Embedder

Vectors = npt.NDArray[np.float32]


class HashingEmbedder(Embedder):
    """Embed text by hashing character n-grams into a fixed-width vector.

    Each text is lowercased and decomposed into its character n-grams. Every
    n-gram is hashed with :func:`zlib.crc32` (a stable, unsalted hash) into one
    of ``n_features`` buckets; bucket counts form the raw feature vector, which
    is then L2-normalised. Texts with no n-grams map to an all-zero row.
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

    def _encode_one(self, text: str) -> Vectors:
        """Encode a single text into an L2-normalised feature row."""
        row: Vectors = np.zeros(self._n_features, dtype=np.float32)
        lowered: str = text.lower()
        n: int = self._ngram
        if len(lowered) < n:
            return row
        for i in range(len(lowered) - n + 1):
            ngram: str = lowered[i : i + n]
            bucket: int = zlib.crc32(ngram.encode("utf-8")) % self._n_features
            row[bucket] += np.float32(1.0)
        norm: np.float32 = np.float32(np.linalg.norm(row))
        if norm > np.float32(0.0):
            row /= norm
        return row

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> Vectors:
        """Map each text to a dense L2-normalised float32 vector.

        Parameters are accepted for protocol compatibility; ``batch_size`` and
        ``show_progress`` do not affect the deterministic result.

        Returns an array of shape ``(len(texts), n_features)`` with dtype
        ``float32``.
        """
        out: Vectors = np.zeros((len(texts), self._n_features), dtype=np.float32)
        for idx, text in enumerate(texts):
            out[idx] = self._encode_one(text)
        return out
