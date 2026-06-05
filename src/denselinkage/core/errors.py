"""Hard-failure exception taxonomy.

Three failure models, deliberately separate (ADR-0003):

- **Soft, per-pair** matcher failures use ``MatchError`` in
  ``LinkageResult.errors`` — never exceptions, so one bad LLM call cannot
  abort a batch.
- **Hard, data/runtime** failures (invalid input / incompatible components)
  raise the exceptions below. All subclass :class:`DenseLinkageError`, so
  callers can catch the family with one ``except``. Dependency-free.
- **API misuse / programmer error** (e.g. calling ``link`` / ``index`` /
  ``dedupe`` with ``blocker=None``, or a ``Matcher`` returning the wrong number
  of outcomes) raises a plain ``ValueError`` — deliberately *outside*
  :class:`DenseLinkageError`. It signals a bug in the calling code, not a runtime
  data condition, so an ``except DenseLinkageError`` guarding data handling must
  not swallow it.
"""


class DenseLinkageError(Exception):
    """Root of every hard failure raised by denselinkage."""


class UnknownIdColumn(DenseLinkageError):
    """``Source.id_column`` is not a column of ``Source.frame``."""


class EmptySource(DenseLinkageError):
    """``Source.frame`` has no rows."""


class DuplicateRecordId(DenseLinkageError):
    """``Source.id_column`` contains duplicate ids (record identity must be
    unique within a source)."""


class DimensionMismatch(DenseLinkageError):
    """An embedder's output width does not match the vector index (or a query
    embedding does not match the indexed vectors)."""


class InvalidTopK(DenseLinkageError):
    """A blocker ``top_k`` is not a positive integer."""


class IncompatibleStore(DenseLinkageError):
    """A persisted index cannot be reloaded as requested: the re-supplied
    embedder's ``model_id`` / ``embedding_dim`` does not match the stored
    provenance, or the store's ``format`` version is unsupported."""


__all__ = [
    "DenseLinkageError",
    "DimensionMismatch",
    "DuplicateRecordId",
    "EmptySource",
    "IncompatibleStore",
    "InvalidTopK",
    "UnknownIdColumn",
]
