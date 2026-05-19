"""Ports — structural contracts (``typing.Protocol``, no ``I`` prefix)."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from denselinkage.core.models import CandidatePair, MatchDecision, Record

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    Vectors = npt.NDArray[np.float32]
else:
    Vectors = Any


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
    def match(self, pairs: Sequence[CandidatePair]) -> list[MatchDecision]: ...
