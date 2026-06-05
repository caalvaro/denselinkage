Orchestration
=============

Configuration lives in :class:`~denselinkage.linkage.DenseLinker` — an immutable
value object with no data and nothing fitted. Prepared, per-dataset state lives
in :class:`~denselinkage.linkage.LinkageIndex`, returned by
:meth:`~denselinkage.linkage.DenseLinker.index`. See
:doc:`/getting-started/concepts` for the design-time / runtime split.

:func:`~denselinkage.linkage.candidate_pairs_from_frame` builds the
:meth:`~denselinkage.linkage.DenseLinker.match_pairs` input — ``CandidatePair``
objects — from a DataFrame of candidate id-pairs and the two sources they
reference.

.. currentmodule:: denselinkage.linkage

.. autosummary::
   :toctree: generated
   :nosignatures:

   DenseLinker
   LinkageIndex
   candidate_pairs_from_frame
