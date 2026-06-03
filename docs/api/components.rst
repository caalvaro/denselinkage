Components
==========

The pluggable adapters, grouped by pipeline stage. Each family has a port in
:doc:`contract`; the classes below are the reference adapters that declare it.
Heavy adapters import their backend lazily and require an extra (noted per
class) — installing the core package never pulls them in.

Serializing
-----------

.. currentmodule:: denselinkage.serializing
.. autosummary::
   :toctree: generated
   :nosignatures:

   TemplateSerializer
   FieldwiseSerializer
   WholeRowSerializer
   default_serializer

Embedding
---------

.. currentmodule:: denselinkage.embedding
.. autosummary::
   :toctree: generated
   :nosignatures:

   HashedNGramEmbedder
   SentenceTransformerEmbedder

Indexing
--------

.. currentmodule:: denselinkage.indexing
.. autosummary::
   :toctree: generated
   :nosignatures:

   NumpyFlatIndex
   NumpySearchableIndex
   FaissFlatIndex
   FaissSearchableIndex

Blocking
--------

.. currentmodule:: denselinkage.blocking
.. autosummary::
   :toctree: generated
   :nosignatures:

   DenseBlocker
   DenseBlockingIndex

Filtering
---------

.. currentmodule:: denselinkage.filtering
.. autosummary::
   :toctree: generated
   :nosignatures:

   SimilarityThresholdFilter

Matching
--------

.. currentmodule:: denselinkage.matching
.. autosummary::
   :toctree: generated
   :nosignatures:

   ThresholdMatcher
   LangChainMatcher
   RetryPolicy

Clustering
----------

.. currentmodule:: denselinkage.clustering
.. autosummary::
   :toctree: generated
   :nosignatures:

   ConnectedComponentsClusterer
   connected_components
