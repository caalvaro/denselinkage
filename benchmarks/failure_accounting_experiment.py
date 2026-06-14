"""End-to-end run of the dependency-free denselinkage pipeline on the standard
DBLP-ACM benchmark, plus a measurement of the failure-accounting distortion.

No API key required: the matcher is the dependency-free ThresholdMatcher, and
matcher failures (refusals/timeouts) are *injected* at a controlled rate f to
isolate the effect of how a tool accounts for them. Run from the repo root with
the project venv. Reproducible (fixed seed)."""

import csv
import json
import os
import random
import statistics

import numpy as np
import pandas as pd

from denselinkage import DenseLinker, LabeledPairs, Source
from denselinkage.blocking import DenseBlocker
from denselinkage.core.models import MatchDecision, MatchError
from denselinkage.core.results import LinkageResult
from denselinkage.embedding import HashedNGramEmbedder
from denselinkage.indexing import NumpyFlatIndex
from denselinkage.matching import ThresholdMatcher
from denselinkage.metrics import linkage_metrics, pair_completeness_at_k

DATA = os.path.join(os.path.dirname(__file__), "dblp_acm")
SEED = 20260613
random.seed(SEED)
np.random.seed(SEED)


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def to_frame(rows, prefix):
    recs = []
    for r in rows:
        text = " ".join(r[c] for c in ("title", "authors", "venue", "year"))
        recs.append({"id": prefix + r["id"], "text": text})
    return pd.DataFrame(recs)


class TextField:
    """Serializer adapter: render a record from its 'text' field only."""

    def serialize(self, record):
        return record["text"]


A = to_frame(load("tableA.csv"), "A")
B = to_frame(load("tableB.csv"), "B")
gold_pairs = []
for f in ("train.csv", "valid.csv", "test.csv"):
    for r in load(f):
        if r["label"] == "1":
            gold_pairs.append(("A" + r["ltable_id"], "B" + r["rtable_id"]))
gold = LabeledPairs.from_pairs(gold_pairs)
print(f"|A|={len(A)}  |B|={len(B)}  gold matches={len(gold_pairs)}")

left = Source(A, id_column="id", serializer=TextField())
right = Source(B, id_column="id", serializer=TextField())

# ---- Blocking quality: pair completeness@k (dense lexical blocking) ----
blocker = DenseBlocker(
    embedder=HashedNGramEmbedder(n_features=1024, ngram=3),
    vector_index=NumpyFlatIndex(),
    top_k=20,
)
linker_block = DenseLinker(blocker=blocker, matcher=ThresholdMatcher(threshold=0.0))
cands = linker_block.block(left, right, top_k=20)
print("\n[Blocking] candidate pairs:", len(cands))
for k in (1, 5, 10, 20):
    pc = pair_completeness_at_k(cands, gold=gold, k=k, directed=True)
    print(f"  pair completeness@{k} = {pc:.4f}")

# ---- End-to-end P/R/F1 at a tuned threshold ----
best = None
for thr in [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]:
    linker = DenseLinker(blocker=blocker, matcher=ThresholdMatcher(threshold=thr))
    res = linker.link(left, right)
    m = linkage_metrics(res, gold=gold, directed=True)
    if best is None or m.f1 > best[1].f1:
        best = (thr, m, res)
thr, m, res = best
print(
    f"\n[End-to-end] best threshold={thr}  P={m.precision:.4f} R={m.recall:.4f} "
    f"F1={m.f1:.4f}  (tp={m.true_positive} fp={m.false_positive} fn={m.false_negative})"
)

# ---- Failure-accounting experiment ----
# Start from the end-to-end decisions at the tuned threshold; inject matcher
# failures on a fraction f of decided pairs and compare the two accountings.
decisions = list(res.decisions)  # all are MatchDecision (no errors at f=0)
print("\n[Failure accounting] decided pairs:", len(decisions))
print(
    f"  {'f':>5} {'F1_excluded':>12} {'F1_silent':>11} "
    f"{'dF1':>7} {'recall_excl':>11} {'recall_silent':>13}"
)
rows_out = []
TRIALS = 20
for f in (0.0, 0.02, 0.05, 0.10, 0.20):
    excl_f1s, sil_f1s, excl_r, sil_r = [], [], [], []
    for t in range(TRIALS if f > 0 else 1):
        rng = random.Random(SEED + t)
        n_err = round(f * len(decisions))
        err_idx = set(rng.sample(range(len(decisions)), n_err))
        kept, errored = [], []
        for i, (pair, dec) in enumerate(decisions):
            if i in err_idx:
                errored.append((pair, MatchError(reason="injected refusal")))
            else:
                kept.append((pair, dec))
        # (a) exclusion accounting (denselinkage): errors in the errors channel
        res_excl = LinkageResult(decisions=tuple(kept), errors=tuple(errored))
        m_excl = linkage_metrics(res_excl, gold=gold, directed=True)
        # (b) silent-non-match convention: errored pairs become is_match=False
        silent = list(kept) + [
            (pair, MatchDecision(is_match=False)) for pair, _ in errored
        ]
        res_sil = LinkageResult(decisions=tuple(silent), errors=())
        m_sil = linkage_metrics(res_sil, gold=gold, directed=True)
        excl_f1s.append(m_excl.f1)
        sil_f1s.append(m_sil.f1)
        excl_r.append(m_excl.recall)
        sil_r.append(m_sil.recall)
    ef, sf = statistics.mean(excl_f1s), statistics.mean(sil_f1s)
    er, sr = statistics.mean(excl_r), statistics.mean(sil_r)
    print(
        f"  {f:>5.2f} {ef:>12.4f} {sf:>11.4f} {ef - sf:>7.4f} {er:>11.4f} {sr:>13.4f}"
    )
    rows_out.append((f, ef, sf, ef - sf, er, sr))

# save machine-readable results
results = {
    "dataset": "DBLP-ACM",
    "seed": SEED,
    "best_threshold": thr,
    "base": {
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "tp": m.true_positive,
        "fp": m.false_positive,
        "fn": m.false_negative,
    },
    "n_decided": len(decisions),
    "trials_per_f": TRIALS,
    "failure_accounting": [
        dict(
            zip(
                [
                    "f",
                    "f1_excluded",
                    "f1_silent",
                    "delta_f1",
                    "recall_excluded",
                    "recall_silent",
                ],
                r,
                strict=True,
            )
        )
        for r in rows_out
    ],
}
with open(os.path.join(DATA, "..", "results.json"), "w") as fh:
    json.dump(results, fh, indent=2)
print("\nsaved benchmarks/results.json")
