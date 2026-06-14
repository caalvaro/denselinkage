"""A deterministic character n-gram feature-hashing :class:`Embedder` adapter.

This module provides :class:`HashingEmbedder`, a dependency-light implementation
of the :class:`denselinkage.core.ports.Embedder` protocol. It maps each input
text to a fixed-width, L2-normalised float32 vector by hashing the text's
character n-grams into buckets with a stable (unsalted) hash.

The adapter is fully deterministic across processes and Python runs because it
uses :func:`zlib.crc32` rather than the builtin :func:`hash`, which is salted
via ``PYTHONHASHSEED`` for strings and bytes.
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

    Each text is lowercased and decomposed into its character n-grams. Every
    n-gram is hashed with :func:`zlib.crc32` into one of ``n_features`` buckets,
    the bucket counts form a raw feature vector, and that vector is L2-normalised
    to unit length. A text with no n-grams (for example, an empty string, or a
    string shorter than ``ngram`` characters) maps to an all-zero row.
    """

    def __init__(self, *, n_features: int = 256, ngram: int = 3) -> None:
        """Create the embedder.

        Args:
            n_features: Width of the output vector (number of hash buckets).
                Must be a positive integer.
            ngram: Length, in characters, of each n-gram. Must be a positive
                integer.

        Raises:
            ValueError: If ``n_features`` or ``ngram`` is not a positive integer.
        """
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features!r}")
        if ngram < 1:
            raise ValueError(f"ngram must be >= 1, got {ngram!r}")
        self._n_features: int = n_features
        self._ngram: int = ngram

    @property
    def model_id(self) -> str:
        """Stable identifier of this embedder's configuration."""
        return f"hashing-{self._ngram}gram-{self._n_features}"

    @property
    def embedding_dim(self) -> int:
        """Output width of :meth:`encode` (equal to ``n_features``)."""
        return self._n_features

    def _embed_one(self, text: str) -> Vectors:
        """Compute the L2-normalised feature-hash vector for a single text.

        Args:
            text: The input string.

        Returns:
            A 1-D float32 array of shape ``(n_features,)``.
        """
        row: Vectors = np.zeros(self._n_features, dtype=np.float32)
        lowered: str = text.lower()
        n: int = self._ngram
        # Iterate over the character n-grams of the lowercased text.
        for start in range(len(lowered) - n + 1):
            ngram: str = lowered[start : start + n]
            bucket: int = zlib.crc32(ngram.encode("utf-8")) % self._n_features
            row[bucket] += np.float32(1.0)
        norm: np.float32 = np.float32(np.sqrt(np.dot(row, row)))
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
        """Encode a sequence of texts into dense float32 vectors.

        Args:
            texts: The texts to embed.
            batch_size: Accepted for protocol compatibility; ignored, as this
                pure-Python implementation processes texts one at a time.
            show_progress: Accepted for protocol compatibility; ignored.

        Returns:
            A float32 array of shape ``(len(texts), n_features)``. Each row is
            L2-normalised, or all zeros if its text has no n-grams.
        """
        del batch_size, show_progress  # Unused; present for protocol parity.
        n_texts: int = len(texts)
        out: Vectors = np.zeros((n_texts, self._n_features), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = self._embed_one(text)
        return out
