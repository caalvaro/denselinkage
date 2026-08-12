"""Lazy-import helper for optional heavy adapters.

The core installs with only numpy + pandas. Heavy adapters
(``FaissFlatIndex``, ``SentenceTransformerEmbedder``, ``LangChainMatcher``,
and ``DenseLinker.with_defaults``) MUST import their backend lazily *inside*
methods and go through :func:`require`, so that ``import denselinkage`` never
fails on a missing extra and a missing dependency produces an actionable
message instead of a bare ``ModuleNotFoundError``.

Pattern::

    def build(self, ...):
        require("faiss")   # raises with the install hint if absent
        import faiss       # the plain import is what mypy resolves

Call :func:`require` for its side effect and then issue the real import on the
next line. Do not bind the return value: that hands mypy a ``ModuleType`` and
defeats the ``ignore_missing_imports`` override in ``pyproject.toml``, so the
backend's own types stop resolving.

This package is a façade: the implementation lives in ``require``.
"""

from denselinkage._optional.require import require

__all__ = ["require"]
