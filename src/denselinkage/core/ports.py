"""Ports — structural contracts (``typing.Protocol``, no ``I`` prefix)."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from denselinkage.core.models import (
    CandidatePair,
    MatchDecision,
    MatchError,
    Record,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from denselinkage.core.results import TrainingPairs

    Vectors = npt.NDArray[np.float32]
else:
    Vectors = Any

# Trainability is an orthogonal capability: it is a separate port, never a
# method on Embedder/Matcher (a HashedNGramEmbedder cannot be trained). A
# Trainer is a *factory* — train() returns a NEW frozen component and never
# mutates self or `base`, so it preserves the package's immutability contract.
ComponentT = TypeVar("ComponentT")


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
    def add(self, vectors: Vectors, ids: Sequence[str]) -> None: ...

    def search(
        self, queries: Vectors, *, top_k: int
    ) -> list[list[tuple[str, float]]]: ...


@runtime_checkable
class Blocker(Protocol):
    def index(self, records: Sequence[Record]) -> None: ...

    def query(self, records: Sequence[Record]) -> list[CandidatePair]: ...


@runtime_checkable
class Matcher(Protocol):
    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision | MatchError]:
        """One outcome per input pair, aligned by position. A pair the matcher
        cannot decide yields a ``MatchError`` (never raises into the batch, so
        one bad call does not abort the rest)."""
        ...


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
