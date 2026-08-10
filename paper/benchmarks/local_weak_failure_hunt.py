"""Hunt for NATURAL (non-injected) matcher failures of a *weak* local LLM.

The companion ``gemini_failure_hunt.py`` shows that a strong hosted model
(gemini-2.5-flash-lite) decided all 4000 sampled DBLP-ACM pairs with zero
matcher failures -- the failure contract never fires at that quality/scale.
This script asks the complementary question: does a genuinely *weak* model,
run locally through the SAME ``LangChainMatcher`` real path (structured output
+ retry -> ``MatchError``), produce failures that occur *on their own*, with no
injection? The failure rate here is a property of the model, not of how we
account for a failure -- which is exactly why a reported quality metric must not
absorb it.

Pipeline (all dependency-free except the local model):
  * blocking : HashedNGramEmbedder + NumpyFlatIndex (top_k=5), no API, no GPU.
  * sample   : every gold pair the blocker surfaced + hardest negatives, up to N.
  * matching : ChatOllama(<weak model>) via LangChainMatcher with structured
               output. A refusal / malformed structured answer / transport error
               flows through the matcher's retry and becomes a real MatchError.
  * report   : natural failure rate, reason buckets, and -- on the SAME real
               output, with NO injection -- excluded-vs-silent F1 and coverage.

Config via env: OLLAMA_MODEL (default llama3.2:1b), N_BIG (default 1000),
OLLAMA_CONC (default 4). Reproducible sampling (fixed seed). Saved incrementally.
"""

import collections
import csv
import json
import os
import random

from denselinkage import DenseLinker, LabeledPairs, Source
from denselinkage.blocking import DenseBlocker
from denselinkage.core.models import MatchDecision
from denselinkage.core.results import LinkageResult
from denselinkage.embedding import HashedNGramEmbedder
from denselinkage.indexing import NumpyFlatIndex
from denselinkage.matching import LangChainMatcher, RetryPolicy, ThresholdMatcher
from denselinkage.metrics import linkage_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dblp_acm")
SEED = 20260613
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
_TAG = MODEL.replace(":", "_").replace("/", "_")
OUT = os.path.join(HERE, f"local_weak_failure_hunt_{_TAG}.json")
N_BIG = int(os.environ.get("N_BIG", "1000"))
CONC = int(os.environ.get("OLLAMA_CONC", "4"))
CHUNK = 100
PROMPT = (
    "Do record A and record B refer to the same real-world publication?\n"
    "A: {record_a}\nB: {record_b}"
)


class TextField:
    """Serializer adapter: render a record from its concatenated 'text' field."""

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


def _bucket(reason: str) -> str:
    r = reason.lower()
    if "429" in reason or "resource_exhausted" in r or "rate" in r:
        return "rate_limit"
    if any(
        k in r
        for k in ("parse", "schema", "valid", "format", "json", "output", "coerce")
    ):
        return "format_parse"
    if any(k in r for k in ("refus", "safety", "blocked", "recitation")):
        return "refusal"
    if any(
        k in r for k in ("timeout", "timed out", "deadline", "connection", "transport")
    ):
        return "transport_timeout"
    return "other"


def main():
    a, b = _frame(_load("tableA.csv"), "A"), _frame(_load("tableB.csv"), "B")
    left = Source(a, id_column="id", serializer=TextField())
    right = Source(b, id_column="id", serializer=TextField())
    gold_all = set(_gold_pairs())

    # ---- dependency-free blocking + gold-labelled sample -------------------
    blocker = DenseBlocker(
        embedder=HashedNGramEmbedder(n_features=1024, ngram=3),
        vector_index=NumpyFlatIndex(),
        top_k=5,
    )
    cands = DenseLinker(blocker=blocker, matcher=ThresholdMatcher(threshold=0.0)).block(
        left, right, top_k=5
    )
    # Representative random sample of candidate pairs (keeps both classes; the
    # matcher failure rate should not depend on the pos/neg split).
    rng = random.Random(SEED)
    sample = rng.sample(cands, N_BIG) if len(cands) > N_BIG else list(cands)
    rng.shuffle(sample)
    gold = LabeledPairs.from_pairs(
        [
            (c.record_a.id, c.record_b.id)
            for c in sample
            if (c.record_a.id, c.record_b.id) in gold_all
        ]
    )
    print(
        f"|A|={len(a)} |B|={len(b)} candidates={len(cands)} "
        f"sample={len(sample)} (gold in sample={len(gold.pairs)})",
        flush=True,
    )

    # ---- weak local matcher on the real LangChainMatcher path -------------
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=MODEL, temperature=0.0)
    matcher = LangChainMatcher(
        llm=llm,
        prompt=PROMPT,
        retry=RetryPolicy(max_retries=1, backoff_seconds=0.5),
        max_concurrency=CONC,
    )
    print(
        f"matching {len(sample)} pairs with local {MODEL} (conc {CONC}) ...", flush=True
    )

    decisions, errors = [], []
    buckets: collections.Counter = collections.Counter()
    examples = []
    for i in range(0, len(sample), CHUNK):
        chunk = sample[i : i + CHUNK]
        for pair, o in zip(chunk, matcher.match(chunk), strict=True):
            if isinstance(o, MatchDecision):
                decisions.append((pair, o))
            else:
                errors.append((pair, o))
                buckets[_bucket(o.reason)] += 1
                if len(examples) < 30:
                    examples.append(
                        {
                            "a": pair.record_a.id,
                            "b": pair.record_b.id,
                            "reason": o.reason[:200],
                        }
                    )
        n_done = i + len(chunk)
        rate = len(errors) / n_done if n_done else 0.0
        _save(sample, decisions, errors, buckets, examples, gold, partial=True)
        print(
            f"  {n_done}/{len(sample)}  decided={len(decisions)} "
            f"failures={len(errors)} ({rate:.1%}) buckets={dict(buckets)}",
            flush=True,
        )

    _save(sample, decisions, errors, buckets, examples, gold, partial=False)


def _save(sample, decisions, errors, buckets, examples, gold, *, partial):
    n = len(sample)
    n_err = len(errors)
    # excluded accounting (denselinkage): errors in the errors channel
    m_excl = linkage_metrics(
        LinkageResult(decisions=tuple(decisions), errors=tuple(errors)),
        gold=gold,
        directed=True,
    )
    # silent convention: each failure becomes a non-match decision
    silent = list(decisions) + [(p, MatchDecision(is_match=False)) for p, _ in errors]
    m_sil = linkage_metrics(
        LinkageResult(decisions=tuple(silent), errors=()), gold=gold, directed=True
    )
    out = {
        "model": MODEL,
        "n_sampled": n,
        "decided": len(decisions),
        "match_errors": n_err,
        "failure_rate": round(n_err / n, 4) if n else 0.0,
        "coverage": round(len(decisions) / n, 4) if n else 0.0,
        "reason_buckets": dict(buckets),
        "gold_in_sample": len(gold.pairs),
        "accounting_no_injection": {
            "f1_excluded": round(m_excl.f1, 4),
            "f1_silent": round(m_sil.f1, 4),
            "delta_f1": round(m_excl.f1 - m_sil.f1, 4),
            "recall_excluded": round(m_excl.recall, 4),
            "recall_silent": round(m_sil.recall, 4),
            "precision_excluded": round(m_excl.precision, 4),
            "n_errors": m_excl.n_errors,
        },
        "examples": examples,
    }
    path = OUT + (".partial" if partial else "")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    if not partial:
        print("\n=== NATURAL failure result (no injection) ===", flush=True)
        print(
            json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1),
            flush=True,
        )
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
