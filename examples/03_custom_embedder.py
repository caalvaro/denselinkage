"""Example 03 — Custom embedder + custom serializer (open ports).

The ports (``Embedder``, ``VectorIndex``, ``Serializer``, ...) are plain
``typing.Protocol``s. First-party adapters are encouraged to **subclass them
explicitly** — that is valid Python and lets the type checker verify the
implementation is complete; third-party code can also conform purely
structurally without importing anything (this resolves the old
docstring-vs-code contradiction).

Wired on the dependency-free stack: ``NumpyFlatIndex`` + ``ThresholdMatcher``,
no GPU / FAISS / API key. The package also ships its own ``HashedNGramEmbedder``
/ ``FieldwiseSerializer``; here we write fresh ones to show the port is open.

Runs on the dependency-free stack (no extras): ``NumpyFlatIndex`` and
``ThresholdMatcher`` are implemented, so this example executes end to end.
"""

import zlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from denselinkage import DenseLinker, Source
from denselinkage.blocking import DenseBlocker
from denselinkage.core.ports import Embedder, Serializer
from denselinkage.indexing import NumpyFlatIndex
from denselinkage.matching import ThresholdMatcher


class CharNGramEmbedder(Embedder):
    """A deterministic char-n-gram feature-hashing embedder — explicitly
    implements the ``Embedder`` port (so the type checker checks it)."""

    def __init__(self, n_features: int = 256, ngram: int = 3) -> None:
        self._n = n_features
        self._k = ngram

    @property
    def model_id(self) -> str:
        return f"char-ngram-{self._k}-{self._n}"

    @property
    def embedding_dim(self) -> int:
        return self._n

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> npt.NDArray[np.float32]:
        out = np.zeros((len(texts), self._n), dtype=np.float32)
        for i, t in enumerate(texts):
            t = t.lower()
            for j in range(max(1, len(t) - self._k + 1)):
                # zlib.crc32 is a stable hash — deterministic across processes,
                # unlike builtin hash() which is PYTHONHASHSEED-salted on str.
                gram = t[j : j + self._k]
                out[i, zlib.crc32(gram.encode("utf-8")) % self._n] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:  # L2-normalise so inner product == cosine
                out[i] /= norm
        return out


class FieldJoinSerializer(Serializer):
    """Joins listed fields with a separator — implements ``Serializer``."""

    def __init__(self, fields: Sequence[str], sep: str = " | ") -> None:
        self._fields = list(fields)
        self._sep = sep

    def serialize(self, record: Mapping[str, Any]) -> str:
        return self._sep.join(str(record.get(f, "")) for f in self._fields)


def main() -> None:
    df_a = pd.DataFrame(
        {
            "id": ["A1", "A2"],
            "name": ["Apple Inc", "Microsoft Corp"],
            "city": ["Cupertino", "Redmond"],
        }
    )
    df_b = pd.DataFrame(
        {
            "id": ["B1", "B2"],
            "name": ["Apple Incorporated", "Microsoft"],
            "city": ["Cupertino, CA", "Redmond, WA"],
        }
    )

    # ThresholdMatcher gates on the similarity the blocker already carried:
    # set it ABOVE the blocker's similarity_threshold so it is a real second
    # gate, not a pass-through.
    linker = DenseLinker(
        blocker=DenseBlocker(
            embedder=CharNGramEmbedder(n_features=512, ngram=3),
            vector_index=NumpyFlatIndex(),
            similarity_threshold=0.30,
            top_k=2,
        ),
        matcher=ThresholdMatcher(threshold=0.55),
    )

    serializer = FieldJoinSerializer(fields=["name", "city"])
    left = Source(df_a, id_column="id", serializer=serializer)
    right = Source(df_b, id_column="id", serializer=serializer)

    result = linker.link(left, right)
    # left_id/right_id are fixed names — no collision though both ids are "id"
    print(result.to_frame())


if __name__ == "__main__":
    main()
