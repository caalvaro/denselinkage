"""Deterministic character n-gram feature-hashing Embedder adapter.

This module provides :class:`HashingEmbedder`, an implementation of the
:class:`denselinkage.core.ports.Embedder` protocol that maps each text to a
fixed-width, L2-normalised float32 vector via stable feature hashing of its
character n-grams.

It depends only on :mod:`denselinkage.core.ports`, :mod:`numpy`, and the
Python standard library, and type-checks under ``mypy --strict``.
"""

from __future__ import annotations

import zlib
from collections import Counter
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from denselinkage.core.ports import Embedder

Vectors = npt.NDArray[np.float32]


class HashingEmbedder(Embedder):
    """Hash character n-grams of texts into L2-normalised dense vectors.

    The mapping is fully deterministic: it uses :func:`zlib.crc32` (a stable,
    unsalted hash) so that identical inputs always yield identical vectors
    across processes and runs.
    """

    def __init__(self, *, n_features: int = 256, ngram: int = 3) -> None:
        """Create a hashing embedder.

        Args:
            n_features: Width of the output vectors and number of hash
                buckets. Must be a positive integer.
            ngram: Length (in characters) of each n-gram. Must be a positive
                integer.

        Raises:
            ValueError: If ``n_features`` or ``ngram`` is not positive.
        """
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
        """Output width of :meth:`encode`, equal to ``n_features``."""
        return self._n_features

    def _ngrams(self, text: str) -> list[str]:
        """Return the character n-grams of ``text`` (lowercased upstream).

        A text shorter than ``ngram`` characters yields no n-grams.
        """
        n: int = self._ngram
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    def _encode_one(self, text: str) -> Vectors:
        """Encode a single text into one L2-normalised row vector."""
        row: Vectors = np.zeros(self._n_features, dtype=np.float32)
        grams: list[str] = self._ngrams(text.lower())
        if not grams:
            return row
        counts: Counter[str] = Counter(grams)
        for gram, count in counts.items():
            bucket: int = zlib.crc32(gram.encode("utf-8")) % self._n_features
            row[bucket] += np.float32(count)
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
        """Encode ``texts`` into a ``(len(texts), n_features)`` float32 array.

        Each text is lowercased, decomposed into character n-grams, and each
        n-gram is hashed into one of ``n_features`` buckets with
        :func:`zlib.crc32`. Bucket counts form the row, which is then
        L2-normalised. A text with no n-grams maps to an all-zero row.

        Args:
            texts: The input texts to embed.
            batch_size: Accepted for protocol compatibility; this
                implementation processes texts row by row and ignores it.
            show_progress: Accepted for protocol compatibility; ignored.

        Returns:
            A ``float32`` array of shape ``(len(texts), n_features)``.
        """
        del batch_size, show_progress  # accepted for compatibility; unused
        out: Vectors = np.zeros((len(texts), self._n_features), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = self._encode_one(text)
        return out
