"""Hard-failure exception taxonomy (2.1).

Two distinct failure models, deliberately separate:

- **Soft, per-pair** matcher failures use ``MatchError`` in
  ``LinkageResult.errors`` — never exceptions, so one bad LLM call cannot
  abort a batch.
- **Hard** failures (invalid input / incompatible components) raise the
  exceptions below. All subclass :class:`DenseLinkageError`, so callers can
  catch the family with one ``except``. Dependency-free.
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


__all__ = [
    "DenseLinkageError",
    "DimensionMismatch",
    "DuplicateRecordId",
    "EmptySource",
    "InvalidTopK",
    "UnknownIdColumn",
]
