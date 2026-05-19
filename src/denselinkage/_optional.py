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
"""

import importlib
from types import ModuleType

# Maps an importable backend module to the pip extra that provides it.
_EXTRA_FOR_MODULE = {
    "faiss": "faiss",
    "sentence_transformers": "sentence-transformers",
    "langchain_core": "langchain",
    "langchain_openai": "langchain",
}


def require(module: str) -> ModuleType:
    """Import and return ``module``, or raise with an install hint.

    Raises:
        ModuleNotFoundError: if the backend (and thus its extra) is missing,
            with a message naming the exact ``pip install`` to run.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        extra = _EXTRA_FOR_MODULE.get(module, module)
        raise ModuleNotFoundError(
            f"{module!r} is required for this feature. "
            f"Install it with: pip install 'denselinkage[{extra}]'"
        ) from exc


__all__ = ["require"]
