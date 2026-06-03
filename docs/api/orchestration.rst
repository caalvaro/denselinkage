Orchestration
=============

Configuration lives in :class:`~denselinkage.linkage.DenseLinker` — an immutable
value object with no data and nothing fitted. Prepared, per-dataset state lives
in :class:`~denselinkage.linkage.LinkageIndex`, returned by
:meth:`~denselinkage.linkage.DenseLinker.index`. See
:doc:`/getting-started/concepts` for the design-time / runtime split.

.. currentmodule:: denselinkage.linkage

.. autosummary::
   :toctree: generated
   :nosignatures:

   DenseLinker
   LinkageIndex
