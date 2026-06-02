"""Orchestration — config in ``DenseLinker``, prepared state in
``LinkageIndex``.

Source -> Record materialization (resolving ``serializer=None`` and validating
the frame) is performed by the internal ``denselinkage._reader.RecordReader``
seam; the hard failures below come from there. Soft per-pair matcher
failures are ``MatchError`` in ``LinkageResult.errors``, never exceptions.

This package is a façade: implementations live in sibling modules
(``dense_linker``, ``linkage_index``); import the public names here.
"""

from denselinkage.linkage.dense_linker import DenseLinker
from denselinkage.linkage.linkage_index import LinkageIndex

__all__ = ["DenseLinker", "LinkageIndex"]
