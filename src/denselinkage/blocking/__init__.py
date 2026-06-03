"""Dense blocking — composes an ``Embedder`` and a ``VectorIndex`` spec.

``DenseBlocker`` is a *spec* (``Blocker``): ``build`` indexes the records into a
fresh, immutable ``DenseBlockingIndex`` artifact. The spec is stateless and
reusable; the populated index lives only in the artifact.

This package is a façade: implementations live in sibling modules
(``dense_blocker``, ``dense_blocking_index``); import the public names here.
"""

from denselinkage.blocking.dense_blocker import DenseBlocker
from denselinkage.blocking.dense_blocking_index import DenseBlockingIndex

__all__ = ["DenseBlocker", "DenseBlockingIndex"]
