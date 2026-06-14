"""Mechanical verification for an Embedder-port probe attempt.

Usage: .venv/Scripts/python.exe probe/verify_probe_embedder.py <module.py>
Checks: explicit Embedder subclass; model_id/embedding_dim contract; encode
shape/dtype/L2-normalisation/empty-text handling; and an end-to-end run inside a
real DenseBlocker + ThresholdMatcher pipeline.
"""

import importlib.util
import sys

import numpy as np
import pandas as pd

from denselinkage import DenseLinker, Source
from denselinkage.blocking import DenseBlocker
from denselinkage.core.ports import Embedder
from denselinkage.indexing import NumpyFlatIndex
from denselinkage.matching import ThresholdMatcher


def _load(path):
    spec = importlib.util.spec_from_file_location("candidate_embedder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        obj = getattr(mod, name)
        if (
            isinstance(obj, type)
            and hasattr(obj, "encode")
            and getattr(obj, "__module__", "") == mod.__name__
        ):
            return obj
    raise SystemExit("FAIL: no adapter class with an encode() method found")


def main(path):
    Adapter = _load(path)
    r = {}
    # Embedder is a runtime_checkable Protocol with property members, so
    # issubclass() raises; explicit subclassing is checked via the MRO instead.
    r["subclasses Embedder (explicit)"] = (
        isinstance(Adapter, type) and Embedder in Adapter.__mro__
    )
    e = Adapter()  # default n_features / ngram
    r["model_id is str"] = isinstance(e.model_id, str) and len(e.model_id) > 0
    dim = e.embedding_dim
    r["embedding_dim is int"] = isinstance(dim, int) and dim > 0
    vecs = e.encode(["acme corporation", "acme corp", "globex inc", ""])
    r["encode -> float32 ndarray"] = (
        isinstance(vecs, np.ndarray) and vecs.dtype == np.float32
    )
    r["shape (n_texts, dim)"] = vecs.shape == (4, dim)
    norms = np.linalg.norm(vecs, axis=1)
    r["non-empty rows L2-normalised"] = bool(np.allclose(norms[:3], 1.0, atol=1e-4))
    r["empty text -> zero row"] = bool(norms[3] < 1e-6)

    df_a = pd.DataFrame({"id": ["A1", "A2"], "name": ["Acme Corporation", "Globex"]})
    df_b = pd.DataFrame({"id": ["B1", "B2"], "name": ["Acme Corp", "Globex Inc"]})
    linker = DenseLinker(
        blocker=DenseBlocker(embedder=e, vector_index=NumpyFlatIndex(), top_k=2),
        matcher=ThresholdMatcher(threshold=0.1),
    )
    res = linker.link(Source(df_a, id_column="id"), Source(df_b, id_column="id"))
    r["end-to-end link() runs"] = len(res.decisions) >= 1

    print(f"=== {path} ===")
    ok = True
    for k, v in r.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        ok = ok and v
    print(f"  CONFORMANCE+FUNCTIONAL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
