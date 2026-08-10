"""denselinkage end-to-end with a LOCAL Ollama model (no API key, no cloud).

Reuses the cached text-embedding-3-large candidates (_openai_candidates.json),
filters to the top-K per right-record so the scale matches a top_k=K blocking,
and matches them with the real denselinkage LangChainMatcher backed by ChatOllama
(structured output + retry->MatchError). Reports end-to-end P/R/F1, coverage,
observed natural failures, latency. Chunked + resumable + checkpointed.

Env: OLLAMA_MODEL (default qwen2.5:7b), TOP_MATCH_K (default 5),
     MATCH_N (0 = all filtered), CONCURRENCY (default 2), CHUNK (default 500),
     RESUME=1.  Run in the denselinkage venv with Ollama up.
"""

import json
import os
import sys
import time
from collections import defaultdict

from openai_pipeline_experiment import (
    CAND_F,
    PROMPT,
    _build_pairs,
    _gold_pairs,
)

from denselinkage import LabeledPairs
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.core.results import LinkageResult
from denselinkage.matching import LangChainMatcher, RetryPolicy
from denselinkage.metrics import linkage_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
TOP_MATCH_K = int(os.environ.get("TOP_MATCH_K", "5"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "2"))
CHUNK = int(os.environ.get("CHUNK", "500"))
SLUG = MODEL.replace(":", "_").replace(".", "_").replace("/", "_")
VERDICT_F = os.path.join(HERE, f"_local_{SLUG}_verdicts.json")
RESULTS_F = os.path.join(HERE, f"local_{SLUG}_results.json")


def main():
    blob = json.load(open(CAND_F, encoding="utf-8"))
    byb = defaultdict(list)
    for r in blob["candidates"]:
        byb[r["b_id"]].append(r)
    rows = []
    for rs in byb.values():
        rs.sort(key=lambda r: r["sim"], reverse=True)
        rows.extend(rs[:TOP_MATCH_K])
    cap = int(os.environ.get("MATCH_N", "0"))
    if cap:
        rows = rows[:cap]
    gold_in = sum(1 for r in rows if r["is_gold"])

    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=MODEL, temperature=0)
    matcher = LangChainMatcher(
        llm=llm,
        prompt=PROMPT,
        retry=RetryPolicy(max_retries=2, backoff_seconds=2.0),
        max_concurrency=CONCURRENCY,
    )

    verdicts = []
    if os.environ.get("RESUME") and os.path.exists(VERDICT_F):
        prev = json.load(open(VERDICT_F, encoding="utf-8"))
        if prev.get("matched_rows_n") == len(rows) and prev.get("model") == MODEL:
            verdicts = prev.get("verdicts", [])
    start = len(verdicts)
    print(
        f"[local {MODEL}] top_k={TOP_MATCH_K} -> {len(rows)} pairs "
        f"({gold_in} gold), conc {CONCURRENCY}, resuming from {start}",
        flush=True,
    )

    t0 = time.time()
    for i in range(start, len(rows), CHUNK):
        sub = rows[i : i + CHUNK]
        for o in matcher.match(_build_pairs(sub)):
            if isinstance(o, MatchDecision):
                verdicts.append({"ok": True, "is_match": bool(o.is_match)})
            else:
                verdicts.append({"ok": False, "reason": o.reason[:160]})
        json.dump(
            {"model": MODEL, "matched_rows_n": len(rows), "verdicts": verdicts},
            open(VERDICT_F, "w"),
        )
        done = len(verdicts)
        el = time.time() - t0
        rate = (done - start) / el if el > 0 else 0.0
        eta = (len(rows) - done) / rate / 60 if rate > 0 else 0.0
        nfail = sum(1 for v in verdicts if not v["ok"])
        print(
            f"  {done}/{len(rows)}  fails={nfail}  {rate:.2f} pair/s  ETA {eta:.1f}m",
            flush=True,
        )
    dt = time.time() - t0

    decisions, errors = [], []
    for r, v in zip(rows, verdicts, strict=True):
        cp = CandidatePair(
            record_a=Record(id=r["a_id"], text=r["a_text"]),
            record_b=Record(id=r["b_id"], text=r["b_text"]),
            similarity_score=r["sim"],
        )
        if v["ok"]:
            decisions.append((cp, MatchDecision(is_match=v["is_match"])))
        else:
            errors.append((cp, MatchError(reason=v.get("reason", "fail"))))

    gold = LabeledPairs.from_pairs(_gold_pairs())
    m = linkage_metrics(
        LinkageResult(decisions=tuple(decisions), errors=tuple(errors)),
        gold=gold,
        directed=True,
    )
    silent = LinkageResult(
        decisions=tuple(decisions)
        + tuple((cp, MatchDecision(is_match=False)) for cp, _ in errors),
        errors=(),
    )
    m_sil = linkage_metrics(silent, gold=gold, directed=True)

    out = {
        "model": MODEL,
        "backend": "ollama (local)",
        "embedder": blob["embedder"],
        "top_match_k": TOP_MATCH_K,
        "n_matched": len(rows),
        "gold_in_candidates_set": gold_in,
        "n_decided": len(decisions),
        "n_natural_failures": len(errors),
        "n_gold": blob["n_gold"],
        "end_to_end_excluded": {
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
            "coverage": round(len(decisions) / len(rows), 4),
        },
        "end_to_end_silent": {
            "precision": round(m_sil.precision, 4),
            "recall": round(m_sil.recall, 4),
            "f1": round(m_sil.f1, 4),
        },
        "wall_seconds": round(dt, 1),
        "sec_per_pair": round(dt / max(len(rows) - start, 1), 3),
    }
    json.dump(out, open(RESULTS_F, "w"), indent=1)
    json.dump({**out, "verdicts": verdicts, "matched_rows": rows}, open(VERDICT_F, "w"))
    print(json.dumps(out, indent=1), flush=True)
    print(f"[local {MODEL}] saved {RESULTS_F}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
