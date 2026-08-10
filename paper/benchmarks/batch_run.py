"""Run both matchers via the OpenAI BATCH API (50% cheaper, server-side), under
the org's 2M enqueued-token cap, on a stratified sample.

Sample = ALL gold-bearing candidates + STRAT_NEG sampled negatives -> exact
end-to-end recall + skew-adjusted precision to the full negative count. struct
(denselinkage) and free (pyJedAI) batches are submitted SEQUENTIALLY so neither
exceeds the 2M cap.

  run   : submit struct -> wait -> submit free -> wait -> fetch+score (one bg job).
  fetch : (re)score from saved batch ids.

gpt-5-nano, reasoning_effort=minimal. Env: STRAT_NEG (default 3000). Run in venv.
"""

import json
import os
import random
import sys
import time

from _openai_adapters import load_openai_key
from openai_pipeline_experiment import CAND_F, _gold_pairs
from pyjedai_baseline import _pyjedai_effective, _robust

from denselinkage import LabeledPairs
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.core.results import LinkageResult
from denselinkage.metrics import linkage_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_F = os.path.join(HERE, "_batch_state_matched.json")
DL_RESULTS = os.path.join(HERE, "openai_results_matched.json")
PJ_RESULTS = os.path.join(HERE, "pyjedai_results_matched.json")
MODEL = os.environ.get("OPENAI_CHAT", "gpt-5-nano")
SEED = 20260613

# Matched base question: byte-identical for both arms (records rendered the same,
# NO lowercasing). Only the trailing FREE_TAIL + the structured response_format
# differ, so the comparison isolates format+accounting, not prompt wording.
Q_BASE = "Are record A and record B the same real-world publication?\nA: {a}\nB: {b}"
FREE_TAIL = "\nAnswer with exactly one word: True or False."

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "decision",
        "strict": True,
        "schema": {
            "type": "object",
            # rationale FIRST so the model reasons in the open before committing
            # the boolean (with minimal reasoning, is_match-first gives it no
            # thinking room -> over-literal/conservative decisions).
            "properties": {
                "rationale": {"type": ["string", "null"]},
                "confidence": {"type": ["number", "null"]},
                "is_match": {"type": "boolean"},
            },
            "required": ["rationale", "confidence", "is_match"],
            "additionalProperties": False,
        },
    },
}
_TERMINAL = {"completed", "failed", "expired", "cancelled"}


def _all():
    return json.load(open(CAND_F, encoding="utf-8"))


def _rows():
    blob = _all()
    rows = blob["candidates"]
    strat = int(os.environ.get("STRAT_NEG", "3000"))
    if not strat:
        return rows
    gold = [r for r in rows if r["is_gold"]]
    neg = [r for r in rows if not r["is_gold"]]
    random.Random(SEED).shuffle(neg)
    sample = gold + neg[:strat]
    random.Random(SEED).shuffle(sample)
    return sample


def _client():
    load_openai_key()
    from openai import OpenAI

    return OpenAI()


def _build(kind, rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            base = Q_BASE.format(a=r["a_text"], b=r["b_text"])
            if kind == "struct":
                body = {
                    "model": MODEL,
                    "reasoning_effort": "minimal",
                    "response_format": _SCHEMA,
                    "messages": [{"role": "user", "content": base}],
                }
            else:
                body = {
                    "model": MODEL,
                    "reasoning_effort": "minimal",
                    "messages": [{"role": "user", "content": base + FREE_TAIL}],
                }
            fh.write(
                json.dumps(
                    {
                        "custom_id": f"{kind[0]}{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                )
                + "\n"
            )


def _submit_kind(c, kind, rows):
    path = os.path.join(HERE, f"_batch_{kind}.jsonl")
    _build(kind, rows, path)
    up = c.files.create(file=open(path, "rb"), purpose="batch")
    b = c.batches.create(
        input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h"
    )
    print(f"[submit] {kind}: batch {b.id} ({len(rows)} reqs)", flush=True)
    return b.id


def _wait_one(c, batch_id):
    while True:
        b = c.batches.retrieve(batch_id)
        rc = b.request_counts
        print(
            f"[wait] {batch_id[:18]} {b.status} {rc.completed}/{rc.total} "
            f"(fail {rc.failed})",
            flush=True,
        )
        if b.status in _TERMINAL:
            return b.status
        time.sleep(60)


def run():
    c = _client()
    rows = _rows()
    gold_in = sum(1 for r in rows if r["is_gold"])
    print(f"[run] stratified sample: {len(rows)} pairs ({gold_in} gold)", flush=True)
    state = {"model": MODEL, "n": len(rows), "strat_neg": len(rows) - gold_in}
    sid = _submit_kind(c, "struct", rows)
    state["struct"] = {"batch_id": sid}
    json.dump(state, open(STATE_F, "w"), indent=1)
    if _wait_one(c, sid) != "completed":
        print("[run] struct batch did not complete; aborting", flush=True)
        return
    fid = _submit_kind(c, "free", rows)
    state["free"] = {"batch_id": fid}
    json.dump(state, open(STATE_F, "w"), indent=1)
    _wait_one(c, fid)
    fetch()


def _download(c, batch_id):
    b = c.batches.retrieve(batch_id)
    out = {}
    if b.output_file_id:
        for line in c.files.content(b.output_file_id).text.splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["custom_id"]] = rec
    return out


def _content(rec):
    if rec is None or rec.get("error"):
        return None
    try:
        return rec["response"]["body"]["choices"][0]["message"]["content"]
    except Exception:
        return None


def fetch():
    c = _client()
    state = json.load(open(STATE_F, encoding="utf-8"))
    rows = _rows()
    blob = _all()
    n_gold_total = blob["n_gold"]
    n_neg_full = blob["n_candidates"] - blob["gold_in_candidates"]
    n_neg_sample = sum(1 for r in rows if not r["is_gold"])
    skew = n_neg_full / max(n_neg_sample, 1)
    gold_set = {(a, b) for a, b in _gold_pairs()}

    # ---- denselinkage structured: exact recall + skew-adjusted precision ----
    so = _download(c, state["struct"]["batch_id"])
    tp = fp = nfail = 0
    pred_match = 0
    for i, r in enumerate(rows):
        ct = _content(so.get(f"s{i}"))
        is_g = (r["a_id"], r["b_id"]) in gold_set
        if ct is None:
            nfail += 1
            continue
        try:
            m = bool(json.loads(ct).get("is_match"))
        except Exception:
            nfail += 1
            continue
        if m:
            pred_match += 1
            if is_g:
                tp += 1
            else:
                fp += 1
    fp_full = fp * skew
    rec = tp / n_gold_total
    prec_s = tp / (tp + fp) if tp + fp else 0.0
    prec_adj = tp / (tp + fp_full) if tp + fp_full else 0.0
    f1_s = 2 * prec_s * rec / (prec_s + rec) if prec_s + rec else 0.0
    f1_adj = 2 * prec_adj * rec / (prec_adj + rec) if prec_adj + rec else 0.0
    dl = {
        "chat_model": MODEL,
        "via": "batch",
        "sample": "all-gold + sampled-neg",
        "n_matched": len(rows),
        "n_gold_total": n_gold_total,
        "n_neg_sample": n_neg_sample,
        "n_neg_full": n_neg_full,
        "skew_factor": round(skew, 2),
        "natural_failures": nfail,
        "tp": tp,
        "fp_sample": fp,
        "fp_full_est": round(fp_full, 1),
        "recall_end_to_end": round(rec, 4),
        "precision_sample": round(prec_s, 4),
        "f1_sample": round(f1_s, 4),
        "precision_full_est": round(prec_adj, 4),
        "f1_full_est": round(f1_adj, 4),
    }
    json.dump(dl, open(DL_RESULTS, "w"), indent=1)
    print(
        "[fetch] denselinkage:",
        json.dumps(
            {
                k: dl[k]
                for k in (
                    "recall_end_to_end",
                    "precision_full_est",
                    "f1_full_est",
                    "natural_failures",
                )
            }
        ),
        flush=True,
    )

    # ---- pyJedAI convention (same sample) ----
    if "free" in state:
        fo = _download(c, state["free"]["batch_id"])
        contents = [_content(fo.get(f"f{i}")) for i in range(len(rows))]
        eff = [_pyjedai_effective(ct) for ct in contents]
        gold = LabeledPairs.from_pairs(_gold_pairs())

        def score(kind):
            dec, err = [], []
            for r, ct, e in zip(rows, contents, eff, strict=True):
                cp = CandidatePair(
                    record_a=Record(id=r["a_id"], text=r["a_text"]),
                    record_b=Record(id=r["b_id"], text=r["b_text"]),
                    similarity_score=r["sim"],
                )
                if kind == "exact":
                    dec.append((cp, MatchDecision(is_match=(e == "True"))))
                elif kind == "silent":
                    dec.append((cp, MatchDecision(is_match=bool(_robust(ct)))))
                else:
                    rb = _robust(ct)
                    if rb is None:
                        err.append((cp, MatchError(reason="unparseable")))
                    else:
                        dec.append((cp, MatchDecision(is_match=rb)))
            mm = linkage_metrics(
                LinkageResult(decisions=tuple(dec), errors=tuple(err)),
                gold=gold,
                directed=True,
            )
            return {
                "precision": round(mm.precision, 4),
                "recall": round(mm.recall, 4),
                "f1": round(mm.f1, 4),
                "coverage": round(len(dec) / len(rows), 4),
            }

        noncanon = sum(1 for e in eff if e not in ("True", "False"))
        pj = {
            "model": MODEL,
            "via": "batch",
            "n_pairs": len(rows),
            "convention_source": "pyjedai==0.3.6 (system prompt via AST)",
            "note": "same stratified sample; recall comparable, precision enriched",
            "response_forms": {
                "non_canonical": noncanon,
                "non_canonical_pct": round(100 * noncanon / len(rows), 1),
                "unparseable": sum(1 for ct in contents if _robust(ct) is None),
            },
            "f1_pyjedai_exact": score("exact"),
            "f1_robust_silent": score("silent"),
            "f1_robust_excluded": score("excluded"),
            "sample_non_canonical": [
                {"effective": e, "raw": (ct[:80] if ct else ct)}
                for ct, e in zip(contents, eff, strict=False)
                if e not in ("True", "False") and ct
            ][:20],
        }
        json.dump(pj, open(PJ_RESULTS, "w"), indent=1)
        print("[fetch] pyjedai exact:", json.dumps(pj["f1_pyjedai_exact"]), flush=True)
    print("[fetch] done", flush=True)


if __name__ == "__main__":
    {"run": run, "fetch": fetch}[sys.argv[1]]()
