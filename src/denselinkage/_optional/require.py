"""``require`` — import an optional backend or raise with an install hint."""

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
