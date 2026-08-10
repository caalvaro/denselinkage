"""Fair pyJedAI baseline on the SAME pairs and SAME model as denselinkage.

pyJedAI 0.3.6's LLM matcher (``OllamaMatching.process`` in
``pyjedai/llm_matching.py``) is Ollama-only and cannot target a hosted API, so we
cannot execute its Python object here. Instead we reproduce its decision
convention *faithfully*, extracting the actual artefacts from the installed
source so there is no paraphrase:

  * system prompt  : ``DEFAULT_SYSTEM_PROMPT`` (lines 14-19), read via AST.
  * user query     : ``f"record 1: {r1}, record 2: {r2}. Answer with True. or
                     False."`` (line 199), with r1/r2 lowercased+stripped as in
                     ``_extract_records`` (line 277+).
  * decoding       : ``stop=['\\n','.']`` (line 206).
  * decision rule  : a pair is a match iff the raw content ``== 'True'`` (line
                     211); anything else -- a hedge, a different case, a refusal,
                     a transport error -- is silently a non-match, with no typed
                     error channel and no exclusion from the reported F1.

We run that convention on gpt-5.4-nano over the identical candidate pairs the
denselinkage pipeline produced (``_openai_candidates.json``), so the ONLY
difference vs denselinkage is structured-output + typed accounting, the model
held fixed. We score three ways on the SAME raw outputs:
  (a) pyJedAI exact-string  (== 'True'; failures -> non-match)   [pyJedAI's number]
  (b) robust parse, silent  (parse true/false; unparseable -> non-match)
  (c) robust parse, excluded (unparseable/errored excluded + coverage) [our stance]

Chunked + resumable + checkpointed. Env: MATCH_N, CONCURRENCY (default 12),
CHUNK (default 1000), OPENAI_CHAT (default gpt-5.4-nano), RESUME=1.
"""

import ast
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from _openai_adapters import load_openai_key

from denselinkage import LabeledPairs
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.core.results import LinkageResult
from denselinkage.metrics import linkage_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
CAND_F = os.path.join(HERE, "_openai_candidates.json")
VERDICT_F = os.path.join(HERE, "_pyjedai_verdicts.json")
RESULTS_F = os.path.join(HERE, "pyjedai_results.json")
DATA = os.path.join(HERE, "dblp_acm")

CHAT = os.environ.get("OPENAI_CHAT", "gpt-5.4-nano")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "12"))
CHUNK = int(os.environ.get("CHUNK", "1000"))
STOP = ["\n", "."]

_PYJEDAI_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
    "_pyjedai_env",
    "Lib",
    "site-packages",
    "pyjedai",
    "llm_matching.py",
)


def _real_system_prompt() -> str:
    """Extract pyJedAI's actual DEFAULT_SYSTEM_PROMPT from the installed source."""
    try:
        tree = ast.parse(open(_PYJEDAI_SRC, encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DEFAULT_SYSTEM_PROMPT"
                for t in node.targets
            ):
                return ast.literal_eval(node.value)
    except Exception as e:
        print(
            f"[warn] could not read pyJedAI source ({e}); using vendored copy",
            flush=True,
        )
    return (
        "You are given two record descriptions and your task is to identify\n"
        "if the records refer to the same entity or not.\n\n"
        "You must answer with just one word:\n"
        "True. if the records are referring to the same entity,\n"
        "False. if the records are referring to a different entity."
    )


SYSTEM_PROMPT = _real_system_prompt() + "\n"  # pyJedAI appends a newline


def _query(a_text: str, b_text: str) -> str:
    # pyJedAI _extract_records lowercases+strips the concatenated attributes.
    r1, r2 = a_text.lower().strip(), b_text.lower().strip()
    return f"record 1: {r1}, record 2: {r2}. Answer with True. or False."


def _gold_pairs():
    import csv

    pairs = []
    for f in ("train.csv", "valid.csv", "test.csv"):
        with open(os.path.join(DATA, f), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["label"] == "1":
                    pairs.append(("A" + r["ltable_id"], "B" + r["rtable_id"]))
    return pairs


def _robust(raw):
    """Tolerant parse of a free-text verdict; None if unparseable."""
    if raw is None:
        return None
    s = raw.strip().lower().lstrip("*`# \n\t")
    if s.startswith("true"):
        return True
    if s.startswith("false"):
        return False
    return None


def _pyjedai_effective(content):
    """The string pyJedAI's exact rule actually compares. Ollama applies
    ``stop=['\\n','.']`` server-side, truncating ``"True."`` -> ``"True"`` before
    the ``== 'True'`` test. gpt-5.4-nano rejects the ``stop`` parameter, so we
    reproduce that truncation post-hoc: the response up to the first newline or
    period. This is exactly what pyJedAI's pipeline compares."""
    if content is None:
        return None
    import re

    return re.split(r"[\n.]", content, maxsplit=1)[0]


def main():
    load_openai_key()
    from openai import OpenAI

    client = OpenAI()
    blob = json.load(open(CAND_F, encoding="utf-8"))
    rows = blob["candidates"]
    cap = int(os.environ.get("MATCH_N", "0"))
    if cap:
        rows = rows[:cap]

    raws: list = []
    if os.environ.get("RESUME") and os.path.exists(VERDICT_F):
        prev = json.load(open(VERDICT_F, encoding="utf-8"))
        if prev.get("n") == len(rows) and prev.get("model") == CHAT:
            raws = prev.get("raws", [])
    start = len(raws)
    print(
        f"[pyjedai] {len(rows)} pairs with {CHAT} (conc {CONCURRENCY}, chunk "
        f"{CHUNK}), resuming from {start}",
        flush=True,
    )
    print(
        f"[pyjedai] system prompt (real, {len(SYSTEM_PROMPT)} chars), "
        f"stop={STOP}, rule=(content=='True')",
        flush=True,
    )

    def call(row):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _query(row["a_text"], row["b_text"])},
        ]
        eff = os.environ.get("REASONING", "minimal")
        extra = {"reasoning_effort": eff} if eff and eff != "default" else {}
        for attempt in range(4):
            try:
                r = client.chat.completions.create(model=CHAT, messages=msgs, **extra)
                return r.choices[0].message.content
            except Exception as e:
                if attempt == 3:
                    return {"__error__": f"{type(e).__name__}: {e}"[:160]}
                time.sleep(2.0 * (attempt + 1))

    t0 = time.time()
    for i in range(start, len(rows), CHUNK):
        sub = rows[i : i + CHUNK]
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            raws.extend(ex.map(call, sub))
        json.dump({"model": CHAT, "n": len(rows), "raws": raws}, open(VERDICT_F, "w"))
        done = len(raws)
        el = time.time() - t0
        rate = (done - start) / el if el > 0 else 0.0
        eta = (len(rows) - done) / rate / 60 if rate > 0 else 0.0
        print(f"  {done}/{len(rows)}  {rate:.1f} pair/s  ETA {eta:.1f}m", flush=True)
    dt = time.time() - t0

    gold = LabeledPairs.from_pairs(_gold_pairs())

    def score(kind):
        decisions, errors = [], []
        for r, raw in zip(rows, raws, strict=True):
            cp = CandidatePair(
                record_a=Record(id=r["a_id"], text=r["a_text"]),
                record_b=Record(id=r["b_id"], text=r["b_text"]),
                similarity_score=r["sim"],
            )
            content = None if isinstance(raw, dict) else raw
            if kind == "exact":  # pyJedAI: stop-truncated content == 'True'
                eff = _pyjedai_effective(content)
                decisions.append((cp, MatchDecision(is_match=(eff == "True"))))
            elif kind == "silent":  # robust parse, unparseable -> non-match
                rb = _robust(content)
                decisions.append((cp, MatchDecision(is_match=bool(rb))))
            else:  # excluded: unparseable/errored excluded + coverage
                rb = _robust(content)
                if rb is None:
                    errors.append((cp, MatchError(reason="unparseable/failed")))
                else:
                    decisions.append((cp, MatchDecision(is_match=rb)))
        m = linkage_metrics(
            LinkageResult(decisions=tuple(decisions), errors=tuple(errors)),
            gold=gold,
            directed=True,
        )
        return {
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
            "n_decided": len(decisions),
            "coverage": round(len(decisions) / len(rows), 4),
        }

    contents = [None if isinstance(x, dict) else x for x in raws]
    eff = [_pyjedai_effective(c) for c in contents]
    n_exact_true = sum(1 for e in eff if e == "True")
    n_exact_false = sum(1 for e in eff if e == "False")
    n_canonical = n_exact_true + n_exact_false
    n_noncanonical = sum(1 for e in eff if e not in ("True", "False"))
    n_unparseable = sum(1 for c in contents if _robust(c) is None)
    n_api_errors = sum(1 for x in raws if isinstance(x, dict))

    out = {
        "model": CHAT,
        "n_pairs": len(rows),
        "convention_source": "pyjedai==0.3.6 llm_matching.py (system prompt via AST)",
        "decoding_note": "stop=['\\n','.'] emulated post-hoc (gpt-5.4-nano rejects "
        "the stop param); == Ollama server-side truncation pyJedAI relies on",
        "response_forms": {
            "exact_True": n_exact_true,
            "exact_False": n_exact_false,
            "canonical_total": n_canonical,
            "non_canonical": n_noncanonical,
            "non_canonical_pct": round(100 * n_noncanonical / len(rows), 1),
            "unparseable": n_unparseable,
            "api_errors": n_api_errors,
        },
        "f1_pyjedai_exact": score("exact"),
        "f1_robust_silent": score("silent"),
        "f1_robust_excluded": score("excluded"),
        "wall_seconds": round(dt, 1),
        "sample_non_canonical": [
            {"effective": e, "raw": c[:80]}
            for c, e in zip(contents, eff, strict=False)
            if e not in ("True", "False") and c is not None
        ][:20],
    }
    json.dump(out, open(RESULTS_F, "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print(f"[pyjedai] saved {RESULTS_F}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
