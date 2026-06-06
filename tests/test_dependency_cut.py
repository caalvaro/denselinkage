"""Local mirror of the ``core-only`` CI job: importing the package and every
core / reference module must not pull a heavy backend into ``sys.modules``.
This is the structural guarantee of the light core, enforced from Phase A so
later bodies cannot silently break it.
"""

import importlib
import importlib.util
import subprocess
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
    "denselinkage.linkage",
    "denselinkage.metrics",
    "denselinkage.clustering",
    "denselinkage.serializing",
    "denselinkage.blocking",
    "denselinkage.embedding",
    "denselinkage.filtering",
    "denselinkage.indexing",
    "denselinkage.matching",
    "denselinkage.training",
)


def test_core_imports_pull_no_heavy_backend() -> None:
    # Import every core module in-process (keeps them covered) — none may pull a
    # heavy backend. The authoritative assertion then runs in a *fresh*
    # interpreter, so it is immune to backends other tests in this session (e.g.
    # the adapter suite) have since imported into this process.
    for mod in CORE_MODULES:
        importlib.import_module(mod)
    child = (
        "import importlib, sys\n"
        f"for mod in {list(CORE_MODULES)!r}:\n"
        "    importlib.import_module(mod)\n"
        f"leaked = sorted({set(HEAVY)!r} & set(sys.modules))\n"
        "assert not leaked, f'heavy backend leaked on import: {leaked}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_optional_helper_unknown_module_is_actionable() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"denselinkage\[_dl_absent_\]"):
        require("_dl_absent_")


def test_optional_helper_maps_known_extra_when_absent() -> None:
    if importlib.util.find_spec("faiss") is not None:
        pytest.skip("faiss present; cut covered by the core-only CI job")
    with pytest.raises(ModuleNotFoundError, match=r"denselinkage\[faiss\]"):
        require("faiss")
