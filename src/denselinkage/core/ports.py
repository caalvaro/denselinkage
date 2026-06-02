"""Ports — structural contracts (``typing.Protocol``)."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from denselinkage.core.models import (
    CandidatePair,
    MatchDecision,
    MatchError,
    Record,
    RecordId,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from denselinkage.core.results import (
        ClusteringResult,
        LinkageResult,
        TrainingPairs,
    )

    Vectors = npt.NDArray[np.float32]
else:
    Vectors = Any

ComponentT = TypeVar("ComponentT")

# ``@runtime_checkable`` is
# retained ONLY so the structure-stage contract test can assert
# ``_is_runtime_protocol`` (tests/test_contract.py). No runtime ``isinstance``
# dispatch against these ports exists or is intended; first-party adapters
# subclass their port explicitly and mypy completeness-checks them.


@runtime_checkable
class Serializer(Protocol):
    def serialize(self, record: Mapping[str, Any]) -> str: ...


@runtime_checkable
class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def embedding_dim(self) -> int: ...

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> Vectors: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Spec for a vector-index backend — stateless and reusable.

    Holds no vectors, only the configuration of *which* index to build.
    ``build`` is a factory (cf. :class:`Trainer`, "a factory, not a fit"): it
    returns a fresh populated :class:`SearchableIndex` and mutates neither
    ``self`` nor its arguments, so one spec safely builds many artifacts over
    different datasets.
    """

    def build(self, vectors: Vectors, ids: Sequence[RecordId]) -> "SearchableIndex": ...


@runtime_checkable
class SearchableIndex(Protocol):
    """Immutable fitted artifact produced by :meth:`VectorIndex.build`.

    Holds the indexed vectors for one dataset and answers nearest-neighbour
    queries; it has no ``add`` — the vectors are fixed at build time.
    ``search`` raises ``DimensionMismatch`` if a query's width differs from the
    indexed vectors.

    v1 is batch-oriented and the artifact is immutable. Incremental update is
    out of scope for v1; :meth:`extended` is the designed (not-yet-implemented)
    escape hatch — it returns a NEW artifact rather than mutating this one.
    """

    def search(
        self, queries: Vectors, *, top_k: int
    ) -> list[list[tuple[RecordId, float]]]: ...

    def extended(self, vectors: Vectors, ids: Sequence[RecordId]) -> "SearchableIndex":
        """Return a NEW artifact holding this index's vectors plus ``vectors``;
        this instance is left unchanged (the immutable-artifact guarantee).
        ``ids`` align positionally with ``vectors`` and must be disjoint from
        the ids already indexed; ``vectors`` must match the indexed width or
        ``DimensionMismatch`` is raised.

        Incremental indexing is out of scope for v1, so the v1 reference
        artifacts raise ``NotImplementedError`` rather than returning ``None``;
        the signature is fixed pre-freeze so the capability can
        land later without a breaking change.
        """
        ...


@runtime_checkable
class Blocker(Protocol):
    """Spec for candidate generation — stateless and reusable.

    ``build`` is a factory: it indexes ``records`` into a fresh
    :class:`BlockingIndex` and mutates nothing, so one ``Blocker`` (and the
    immutable ``DenseLinker`` that holds it) can be reused across datasets.
    """

    def build(self, records: Sequence[Record]) -> "BlockingIndex": ...


@runtime_checkable
class BlockingIndex(Protocol):
    """Immutable fitted artifact produced by :meth:`Blocker.build`.

    Holds the indexed (left/reference) records and generates ``CandidatePair``s
    for a query record set. ``top_k`` and ``similarity_threshold`` are
    query-time parameters: they default to the values fixed on the originating
    ``Blocker`` spec but may be overridden per call, so a ``top_k`` / threshold
    sweep reuses one built index instead of rebuilding it. An override
    ``top_k <= 0`` raises ``InvalidTopK`` (parity with the index-time validation
    documented on :meth:`~denselinkage.linkage.DenseLinker.index`).
    """

    def query(
        self,
        records: Sequence[Record],
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[CandidatePair]: ...


@runtime_checkable
class Filter(Protocol):
    """A second comparison-space reduction, distinct from blocking: prune a
    candidate set before matching. Pure over already-generated pairs (carries
    no indexing state). ``SimilarityThresholdFilter`` is the dependency-free
    reference adapter; multi-pass / rule-based filters conform here.
    """

    def filter(self, pairs: Sequence[CandidatePair]) -> list[CandidatePair]: ...


@runtime_checkable
class Matcher(Protocol):
    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision | MatchError]:
        """One outcome per input pair, aligned by position. A pair the matcher
        cannot decide yields a ``MatchError`` (never raises into the batch, so
        one bad call does not abort the rest)."""
        ...


@runtime_checkable
class Clusterer(Protocol):
    """Groups a ``LinkageResult``'s matches into entity clusters — a swappable
    strategy, not a fixed step. ``ConnectedComponentsClusterer`` (wrapping the
    ``connected_components`` reference function) declares this port; alternative
    algorithms (e.g. agglomerative, incremental) conform here. Pure over an
    already-computed ``LinkageResult``; carries no blocking/matching state.
    """

    def cluster(self, result: "LinkageResult") -> "ClusteringResult": ...


@runtime_checkable
class Trainer(Protocol[ComponentT]):
    """Produces a frozen component from supervised data (v2, ``[train]``).

    ``train`` is a factory, not a fit: it returns a NEW component and mutates
    neither ``self`` nor ``base``. ``base`` is the optional checkpoint to
    continue from (enables the recursive blocker -> mine -> retrain loop).
    The protocol is locked here in Phase A; concrete trainers
    (``EmbedderTrainer -> Embedder``, ``CrossEncoderTrainer -> Matcher``)
    ship in v2 behind the ``[train]`` extra.
    """

    def train(
        self, pairs: "TrainingPairs", *, base: ComponentT | None = None
    ) -> ComponentT: ...
