"""Vector indexes (own port/module). ``FaissFlatIndex`` is the heavy adapter
(extra: ``[faiss]``).

Each backend is a *spec* (``VectorIndex``): ``build`` produces a fresh,
immutable ``SearchableIndex`` artifact populated with the given vectors. The
spec is stateless and reusable; state lives only in the artifact.

This package is a façade: implementations live in sibling modules; import the
public names here.
"""

from denselinkage.indexing.faiss_flat_index import FaissFlatIndex
from denselinkage.indexing.faiss_searchable_index import FaissSearchableIndex
from denselinkage.indexing.numpy_flat_index import NumpyFlatIndex
from denselinkage.indexing.numpy_searchable_index import NumpySearchableIndex

__all__ = [
    "FaissFlatIndex",
    "FaissSearchableIndex",
    "NumpyFlatIndex",
    "NumpySearchableIndex",
]
