"""Mechanical verification for one contract-implementability-probe attempt.

Usage (from repo root, project venv):
    .venv/Scripts/python.exe probe/verify_probe.py
    probe/attempt_01/token_jaccard_matcher.py

Checks (all must pass):
  C2  conforms to the Matcher port (explicit subclass)        [informational]
  C4a failure contract: normal pair -> MatchDecision; empty text -> MatchError
  C4b one outcome per pair, position-aligned
  C4c end-to-end: runs inside a real DenseLinker with the shipped blocker
(C1 mypy --strict and C3 git-isolation are run separately by the orchestrator.)
"""

import importlib.util
import sys

import pandas as pd

from denselinkage import DenseLinker, Source
from denselinkage.blocking import DenseBlocker
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.core.ports import Matcher
from denselinkage.embedding import HashedNGramEmbedder
from denselinkage.indexing import NumpyFlatIndex


def _load_adapter(path):
    spec = importlib.util.spec_from_file_location("candidate_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        obj = getattr(mod, name)
        if (
            isinstance(obj, type)
            and hasattr(obj, "match")
            and getattr(obj, "__module__", "") == mod.__name__
        ):
            return obj
    raise SystemExit("FAIL: no adapter class with a match() method found")


def main(path):
    Adapter = _load_adapter(path)
    r = {}
    r["subclasses Matcher (explicit)"] = isinstance(Adapter, type) and issubclass(
        Adapter, Matcher
    )
    m = Adapter()  # the spec mandates a default threshold
    a1, b1 = Record(id="A1", text="acme corporation"), Record(id="B1", text="acme corp")
    a2, b2 = Record(id="A2", text=""), Record(id="B2", text="globex")
    out = m.match(
        [
            CandidatePair(record_a=a1, record_b=b1, similarity_score=0.9),
            CandidatePair(record_a=a2, record_b=b2, similarity_score=0.1),
        ]
    )
    r["one outcome per pair"] = len(out) == 2
    r["normal pair -> MatchDecision"] = bool(out) and isinstance(out[0], MatchDecision)
    r["empty text -> MatchError"] = len(out) > 1 and isinstance(out[1], MatchError)

    df_a = pd.DataFrame(
        {"id": ["A1", "A2", "A3"], "name": ["Acme Corporation", "Globex", "Initech"]}
    )
    df_b = pd.DataFrame(
        {"id": ["B1", "B2", "B3"], "name": ["Acme Corp", "Globex Inc", "Initech LLC"]}
    )
    linker = DenseLinker(
        blocker=DenseBlocker(
            embedder=HashedNGramEmbedder(n_features=1024, ngram=3),
            vector_index=NumpyFlatIndex(),
            top_k=3,
        ),
        matcher=m,
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
