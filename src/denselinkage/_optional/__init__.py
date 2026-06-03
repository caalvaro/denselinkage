"""Lazy-import helper for optional heavy adapters.

The core installs with only numpy + pandas. Heavy adapters
(``FaissFlatIndex``, ``SentenceTransformerEmbedder``, ``LangChainMatcher``,
and ``DenseLinker.with_defaults``) MUST import their backend lazily *inside*
methods and go through :func:`require`, so that ``import denselinkage`` never
fails on a missing extra and a missing dependency produces an actionable
message instead of a bare ``ModuleNotFoundError``.

Pattern::

    def search(self, ...):
        faiss = require("faiss")          # raises with install hint if absent
        ...

This package is a façade: the implementation lives in ``require``.
"""

from denselinkage._optional.require import require

__all__ = ["require"]
