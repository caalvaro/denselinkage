"""Full modern pipeline on DBLP-ACM with OpenAI, end to end (closes Major 3 for
the OpenAI provider and produces identical candidate pairs for the fair pyJedAI
baseline).

  block : OpenAI text-embedding-3-large + FAISS dense blocking over the FULL
          benchmark -> pair completeness@k; caches every candidate pair (with
          gold label and similarity) to _openai_candidates.json.
  match : the cached candidates are matched by gpt-5.4-nano through the real
          denselinkage LangChainMatcher (structured output + retry->MatchError).
          Chunked + resumable + checkpointed. Reports end-to-end P/R/F1 (gold the
          blocker missed is charged as FN, per metrics/linkage.py), coverage,
          observed natural failures, wall clock and a tiktoken cost estimate.

Env: MATCH_N (0 = all candidates; else cap, for a cheap validation run),
     TOP_K (default 20), OPENAI_CHAT (default gpt-5.4-nano),
     OPENAI_EMB (default text-embedding-3-large), CONCURRENCY (default 8),
     CHUNK (default 1000), RESUME (1 = continue from cached verdicts).
The OpenAI key is read from the nearest .env. Fixed seed; reproducible.
"""

import csv
import json
import os
import sys
import time

from _openai_adapters import OpenAIEmbedder, load_openai_key

from denselinkage import DenseLinker, LabeledPairs, Source
from denselinkage.blocking import DenseBlocker
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.core.results import LinkageResult
from denselinkage.indexing import FaissFlatIndex
from denselinkage.matching import LangChainMatcher, RetryPolicy, ThresholdMatcher
from denselinkage.metrics import linkage_metrics, pair_completeness_at_k

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dblp_acm")
CAND_F = os.path.join(HERE, "_openai_candidates.json")
VERDICT_F = os.path.join(HERE, "_openai_verdicts.json")
RESULTS_F = os.path.join(HERE, "openai_results.json")

SEED = 20260613
CHAT = os.environ.get("OPENAI_CHAT", "gpt-5.4-nano")
EMB = os.environ.get("OPENAI_EMB", "text-embedding-3-large")
TOP_K = int(os.environ.get("TOP_K", "20"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
CHUNK = int(os.environ.get("CHUNK", "1000"))
PROMPT = (
    "Do record A and record B refer to the same real-world publication?\n"
    "A: {record_a}\nB: {record_b}"
)

# gpt-5.4-nano published price (USD per 1M tokens); override via env if needed.
PRICE_IN = float(os.environ.get("PRICE_IN", "0.05"))
PRICE_OUT = float(os.environ.get("PRICE_OUT", "0.40"))


class _TextField:
    def serialize(self, record):  # type: ignore[no-untyped-def]
        return record["text"]


def _load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _frame(rows, prefix):
    import pandas as pd

    recs = [
        {
            "id": prefix + r["id"],
            "text": " ".join(r[c] for c in ("title", "authors", "venue", "year")),
        }
        for r in rows
    ]
    return pd.DataFrame(recs)


def _gold_pairs():
    pairs = []
    for f in ("train.csv", "valid.csv", "test.csv"):
        for r in _load(f):
            if r["label"] == "1":
                pairs.append(("A" + r["ltable_id"], "B" + r["rtable_id"]))
    return pairs


def _sources():
    a, b = _frame(_load("tableA.csv"), "A"), _frame(_load("tableB.csv"), "B")
    return (
        Source(a, id_column="id", serializer=_TextField()),
        Source(b, id_column="id", serializer=_TextField()),
        len(a),
        len(b),
    )


def stage_block():
    load_openai_key()
    left, right, na, nb = _sources()
    gold_pairs = _gold_pairs()
    gold_set = set(gold_pairs)
    gold = LabeledPairs.from_pairs(gold_pairs)
    print(
        f"|A|={na} |B|={nb} gold={len(gold_pairs)}  embedding with {EMB} ...",
        flush=True,
    )

    t0 = time.time()
    blocker = DenseBlocker(
        embedder=OpenAIEmbedder(EMB),
        vector_index=FaissFlatIndex(),
        top_k=TOP_K,
    )
    linker = DenseLinker(blocker=blocker, matcher=ThresholdMatcher(threshold=0.0))
    cands = linker.block(left, right, top_k=TOP_K)
    dt = time.time() - t0
    pc = {
        k: round(pair_completeness_at_k(cands, gold=gold, k=k, directed=True), 4)
        for k in (1, 5, 10, 20)
        if k <= TOP_K
    }
    print(
        f"[block] {len(cands)} candidates in {dt:.1f}s  pair_completeness={pc}",
        flush=True,
    )

    rows = [
        {
            "a_id": c.record_a.id,
            "a_text": c.record_a.text,
            "b_id": c.record_b.id,
            "b_text": c.record_b.text,
            "sim": float(c.similarity_score) if c.similarity_score is not None else 0.0,
            "is_gold": (c.record_a.id, c.record_b.id) in gold_set,
        }
        for c in cands
    ]
    payload = {
        "embedder": EMB,
        "top_k": TOP_K,
        "blocking_pc": pc,
        "block_seconds": round(dt, 1),
        "n_candidates": len(cands),
        "n_a": na,
        "n_b": nb,
        "n_gold": len(gold_pairs),
        "gold_in_candidates": sum(r["is_gold"] for r in rows),
        "candidates": rows,
    }
    with open(CAND_F, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(
        f"[block] gold recoverable by blocking: {payload['gold_in_candidates']}"
        f"/{len(gold_pairs)} -> saved {CAND_F}",
        flush=True,
    )


def _build_pairs(rows):
    return [
        CandidatePair(
            record_a=Record(id=r["a_id"], text=r["a_text"]),
            record_b=Record(id=r["b_id"], text=r["b_text"]),
            similarity_score=r["sim"],
        )
        for r in rows
    ]


def _tiktoken_cost(rows):
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(CHAT)
        except Exception:
            enc = tiktoken.get_encoding("o200k_base")
    except Exception:
        return None
    in_tok = 0
    for r in rows:
        text = PROMPT.format(record_a=r["a_text"], record_b=r["b_text"])
        in_tok += len(enc.encode(text)) + 12  # + role/scaffold overhead
    out_tok = len(rows) * 25  # structured verdict, approx
    return {
        "est_prompt_tokens": in_tok,
        "est_completion_tokens": out_tok,
        "est_usd": round(in_tok / 1e6 * PRICE_IN + out_tok / 1e6 * PRICE_OUT, 4),
        "price_in_per_m": PRICE_IN,
        "price_out_per_m": PRICE_OUT,
        "note": "tiktoken input estimate; completion approx 25 tok/pair",
    }


def stage_match():
    load_openai_key()
    blob = json.load(open(CAND_F, encoding="utf-8"))
    rows = blob["candidates"]
    cap = int(os.environ.get("MATCH_N", "0"))
    if cap:
        rows = rows[:cap]
    from langchain_openai import ChatOpenAI

    _eff = os.environ.get("REASONING", "minimal")
    _kw = {"reasoning_effort": _eff} if _eff and _eff != "default" else {}
    llm = ChatOpenAI(model=CHAT, max_retries=2, **_kw)
    matcher = LangChainMatcher(
        llm=llm,
        prompt=PROMPT,
        retry=RetryPolicy(max_retries=3, backoff_seconds=5.0),
        max_concurrency=CONCURRENCY,
    )

    verdicts: list[dict] = []
    if os.environ.get("RESUME") and os.path.exists(VERDICT_F):
        prev = json.load(open(VERDICT_F, encoding="utf-8"))
        if prev.get("matched_rows_n") == len(rows) and prev.get("chat_model") == CHAT:
            verdicts = prev.get("verdicts", [])
    start = len(verdicts)
    print(
        f"[match] {len(rows)} pairs with {CHAT} (conc {CONCURRENCY}, chunk "
        f"{CHUNK}), resuming from {start} ...",
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
            {"chat_model": CHAT, "matched_rows_n": len(rows), "verdicts": verdicts},
            open(VERDICT_F, "w"),
        )
        done = len(verdicts)
        el = time.time() - t0
        rate = (done - start) / el if el > 0 else 0.0
        eta = (len(rows) - done) / rate / 60 if rate > 0 else 0.0
        nfail = sum(1 for v in verdicts if not v["ok"])
        print(
            f"  {done}/{len(rows)}  fails={nfail}  {rate:.1f} pair/s  ETA {eta:.1f}m",
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
    result = LinkageResult(decisions=tuple(decisions), errors=tuple(errors))
    m = linkage_metrics(result, gold=gold, directed=True)
    silent = LinkageResult(
        decisions=tuple(decisions)
        + tuple((cp, MatchDecision(is_match=False)) for cp, _ in errors),
        errors=(),
    )
    m_sil = linkage_metrics(silent, gold=gold, directed=True)

    out = {
        "chat_model": CHAT,
        "embedder": EMB,
        "top_k": blob["top_k"],
        "blocking_pc": blob["blocking_pc"],
        "n_candidates_total": blob["n_candidates"],
        "n_matched": len(rows),
        "n_decided": len(decisions),
        "n_natural_failures": len(errors),
        "gold_in_candidates": blob["gold_in_candidates"],
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
        "sec_per_pair": round(dt / max(len(rows) - start, 1), 4),
        "cost_estimate": _tiktoken_cost(rows),
    }
    json.dump(out, open(RESULTS_F, "w"), indent=1)
    json.dump({**out, "verdicts": verdicts, "matched_rows": rows}, open(VERDICT_F, "w"))
    print(json.dumps(out, indent=1), flush=True)
    print(f"[match] saved {RESULTS_F} and {VERDICT_F}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "block"
    {"block": stage_block, "match": stage_match}[stage]()
