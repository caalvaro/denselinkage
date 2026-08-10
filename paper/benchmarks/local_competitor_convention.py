"""The competitor (brittle free-text) convention on a *weak local* model.

``competitor_convention.py`` ran a strong hosted model (gemini) free-form and
found 36% of responses were not the exact token pyJedAI compares against -- but
on a strong model those mostly fell on non-matches, so F1 barely moved. This
script asks what happens on a genuinely *weak* model, where non-canonical and
unparseable answers are common and land on true matches too.

Same DBLP-ACM pairs as ``local_weak_failure_hunt.py`` (same seed -> same sample),
but the model is queried FREE-FORM (no structured output) and each raw response
is scored three ways against gold:
  * exact (pyJedAI)   : ``resp.strip() == "True"`` -- anything else is a silent
                        non-match (the convention this paper criticises).
  * robust-silent     : a tolerant parse (case/punctuation/lead-in); a response
                        that still cannot be classified is folded into non-match.
  * contract-excluded : the denselinkage stance -- an unparseable response is a
                        typed failure, excluded from P/R and reported as coverage.
The point is not the parser but the gap between them: on a weak model a real
fraction of answers are non-canonical or unparseable, and a quality metric that
swallows them as non-matches is reporting the format failure rate, not the
matcher. These failures occur on their own -- nothing is injected.

Config via env: OLLAMA_MODEL (default llama3.2:1b), N_BIG (default 1000),
OLLAMA_CONC (default 4). Reproducible (fixed seed). Saved incrementally.
"""

import collections
import csv
import json
import os
import random

from denselinkage import DenseLinker, Source
from denselinkage.blocking import DenseBlocker
from denselinkage.embedding import HashedNGramEmbedder
from denselinkage.indexing import NumpyFlatIndex
from denselinkage.matching import ThresholdMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dblp_acm")
SEED = 20260613
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
N_BIG = int(os.environ.get("N_BIG", "1000"))
CONC = int(os.environ.get("OLLAMA_CONC", "4"))
CHUNK = 100
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "64"))
_TAG = MODEL.replace(":", "_").replace("/", "_")
OUT = os.path.join(HERE, f"local_competitor_convention_{_TAG}_np{NUM_PREDICT}.json")
PROMPT = (
    "Are record A and record B the same real-world publication? "
    "Answer True or False.\nA: {a}\nB: {b}"
)


class TextField:
    def serialize(self, record):  # type: ignore[no-untyped-def]
        return record["text"]


def _load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _frame(rows, prefix):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "id": prefix + r["id"],
                "text": " ".join(r[c] for c in ("title", "authors", "venue", "year")),
            }
            for r in rows
        ]
    )


def _gold_pairs():
    pairs = []
    for f in ("train.csv", "valid.csv", "test.csv"):
        for r in _load(f):
            if r["label"] == "1":
                pairs.append(("A" + r["ltable_id"], "B" + r["rtable_id"]))
    return pairs


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def classify(raw: str):
    """Tolerant verdict extraction. Returns True / False / None (unparseable)."""
    low = raw.strip().lower().lstrip("*`#>-_ \n\t\"'.(){}[]")
    if low.startswith(("true", "yes")) or low.startswith("same"):
        return True
    if low.startswith(("false", "no", "different", "not")):
        return False
    # last-ditch: a clear phrase near the start
    head = low[:60]
    if "not the same" in head or "are different" in head or "do not" in head:
        return False
    if "the same" in head or "refer to the same" in head:
        return True
    return None


def _f1(preds, gold):
    tp = sum(1 for p, g in zip(preds, gold, strict=False) if p and g)
    fp = sum(1 for p, g in zip(preds, gold, strict=False) if p and not g)
    fn = sum(1 for p, g in zip(preds, gold, strict=False) if not p and g)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def main():
    a, b = _frame(_load("tableA.csv"), "A"), _frame(_load("tableB.csv"), "B")
    left = Source(a, id_column="id", serializer=TextField())
    right = Source(b, id_column="id", serializer=TextField())
    gold_all = set(_gold_pairs())

    blocker = DenseBlocker(
        embedder=HashedNGramEmbedder(n_features=1024, ngram=3),
        vector_index=NumpyFlatIndex(),
        top_k=5,
    )
    cands = DenseLinker(blocker=blocker, matcher=ThresholdMatcher(threshold=0.0)).block(
        left, right, top_k=5
    )
    rng = random.Random(SEED)
    sample = rng.sample(cands, N_BIG) if len(cands) > N_BIG else list(cands)
    rng.shuffle(sample)
    gold = [(c.record_a.id, c.record_b.id) in gold_all for c in sample]
    print(
        f"candidates={len(cands)} sample={len(sample)} gold_in_sample={sum(gold)}",
        flush=True,
    )

    from langchain_ollama import ChatOllama

    # num_predict caps the answer length: a matcher prompt asks for "True/False",
    # so a short budget is the realistic production setting (and keeps runtime
    # near the structured run). A response that needs more is itself unparseable.
    llm = ChatOllama(model=MODEL, temperature=0.0, num_predict=NUM_PREDICT)
    raws: list[str] = []
    print(f"querying {len(sample)} pairs FREE-FORM with local {MODEL} ...", flush=True)
    for i in range(0, len(sample), CHUNK):
        chunk = sample[i : i + CHUNK]
        prompts = [PROMPT.format(a=c.record_a.text, b=c.record_b.text) for c in chunk]
        msgs = llm.batch(prompts, config={"max_concurrency": CONC})
        raws.extend(_text(m.content) for m in msgs)
        json.dump(
            {"model": MODEL, "n": len(raws), "raws": raws}, open(OUT + ".partial", "w")
        )
        print(f"  {len(raws)}/{len(sample)}", flush=True)

    exact = [r.strip() == "True" for r in raws]
    verdict = [classify(r) for r in raws]
    robust_silent = [(v is True) for v in verdict]  # unparseable -> non-match
    n_unparseable = sum(1 for v in verdict if v is None)
    n_noncanonical = sum(1 for r in raws if r.strip() not in ("True", "False"))

    # contract: exclude unparseable from P/R (report as coverage)
    keep = [i for i, v in enumerate(verdict) if v is not None]
    contract_preds = [verdict[i] is True for i in keep]
    contract_gold = [gold[i] for i in keep]

    forms = collections.Counter(
        "exactly 'True'"
        if r.strip() == "True"
        else "exactly 'False'"
        if r.strip() == "False"
        else "other (mis-scored by exact rule)"
        for r in raws
    )
    out = {
        "model": MODEL,
        "n_pairs": len(raws),
        "gold_in_sample": sum(gold),
        "noncanonical_responses": n_noncanonical,
        "noncanonical_rate": round(n_noncanonical / len(raws), 4) if raws else 0.0,
        "unparseable_responses": n_unparseable,
        "unparseable_rate": round(n_unparseable / len(raws), 4) if raws else 0.0,
        "f1_exact_pyjedai": _f1(exact, gold),
        "f1_robust_silent": _f1(robust_silent, gold),
        "f1_contract_excluded": _f1(contract_preds, contract_gold),
        "contract_coverage": round(len(keep) / len(raws), 4) if raws else 0.0,
        "response_forms": dict(forms),
        "sample_other_responses": [
            r[:80] for r in raws if r.strip() not in ("True", "False")
        ][:25],
    }
    json.dump(out | {"raws": raws}, open(OUT, "w"), indent=1)
    print("\n=== brittle convention on a weak model (no injection) ===", flush=True)
    print(
        json.dumps({k: v for k, v in out.items() if k != "raws"}, indent=1), flush=True
    )
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
