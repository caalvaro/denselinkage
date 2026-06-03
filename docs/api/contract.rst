Contract (``core``)
===================

The dependency-free heart of the library: the **ports** every adapter
implements, the domain **models**, the **results** the ports reference, and the
**error** taxonomy. This is the frozen surface — everything else either
implements a port here or orchestrates ports defined here (see
:doc:`/architecture`).

Ports
-----

Structural contracts (:class:`typing.Protocol`). First-party adapters subclass
their port explicitly so the type checker verifies completeness; third-party
code may conform purely structurally.

.. currentmodule:: denselinkage.core.ports
.. autosummary::
   :toctree: generated
   :nosignatures:

   Serializer
   Embedder
   VectorIndex
   SearchableIndex
   Blocker
   BlockingIndex
   Filter
   Matcher
   Clusterer
   Trainer

Models
------

.. currentmodule:: denselinkage.core.models
.. autosummary::
   :toctree: generated
   :nosignatures:

   Record
   RecordId
   CandidatePair
   MatchDecision
   MatchError
   Source

Results
-------

.. currentmodule:: denselinkage.core.results
.. autosummary::
   :toctree: generated
   :nosignatures:

   LinkageResult
   ClusteringResult
   LabeledPairs
   TrainingPairs

Errors
------

The hard-failure taxonomy. ``DenseLinkageError`` is the catchable root for
data / runtime failures; API misuse raises a plain :class:`ValueError`, kept
deliberately outside this family.

.. currentmodule:: denselinkage.core.errors
.. autosummary::
   :toctree: generated
   :nosignatures:

   DenseLinkageError
   UnknownIdColumn
   EmptySource
   DuplicateRecordId
   DimensionMismatch
   InvalidTopK
