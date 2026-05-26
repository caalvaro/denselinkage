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

    from denselinkage.core.results import Clustering, LinkageResult, TrainingPairs

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
    def add(self, vectors: Vectors, ids: Sequence[RecordId]) -> None: ...

    def search(
        self, queries: Vectors, *, top_k: int
    ) -> list[list[tuple[RecordId, float]]]: ...


@runtime_checkable
class Blocker(Protocol):
    def index(self, records: Sequence[Record]) -> None: ...

    def query(self, records: Sequence[Record]) -> list[CandidatePair]: ...


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

    def cluster(self, result: "LinkageResult") -> "Clustering": ...


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
