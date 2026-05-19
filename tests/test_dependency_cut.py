"""Local mirror of the ``core-only`` CI job: importing the package and every
core / reference module must not pull a heavy backend into ``sys.modules``.
This is the structural guarantee of the light core, enforced from Phase A so
later bodies cannot silently break it.
"""

import importlib
import importlib.util
import sys

import pytest

from denselinkage._optional import require

HEAVY = (
    "faiss",
    "sentence_transformers",
    "langchain_core",
    "langchain_openai",
    "torch",
)

CORE_MODULES = (
    "denselinkage",
    "denselinkage.core.models",
    "denselinkage.core.ports",
    "denselinkage.core.results",
    "denselinkage.linker",
    "denselinkage.metrics",
    "denselinkage.cluster",
    "denselinkage.serialize",
    "denselinkage.blocking",
    "denselinkage.embedding",
    "denselinkage.indexing",
    "denselinkage.matching",
    "denselinkage.training",
)


def test_core_imports_pull_no_heavy_backend() -> None:
    for mod in CORE_MODULES:
        importlib.import_module(mod)
    leaked = set(HEAVY) & set(sys.modules)
    assert not leaked, f"heavy backend leaked on import: {sorted(leaked)}"


def test_optional_helper_unknown_module_is_actionable() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"denselinkage\[_dl_absent_\]"):
        require("_dl_absent_")


def test_optional_helper_maps_known_extra_when_absent() -> None:
    if importlib.util.find_spec("faiss") is not None:
        pytest.skip("faiss present; cut covered by the core-only CI job")
    with pytest.raises(ModuleNotFoundError, match=r"denselinkage\[faiss\]"):
        require("faiss")
