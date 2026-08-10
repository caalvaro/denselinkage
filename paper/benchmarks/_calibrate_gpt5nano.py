"""Measure real gpt-5-nano token usage per pair (structured + free-text) on a tiny
sample, so we can project the full-run cost BEFORE spending. Reads exact usage
from the API (incl. reasoning tokens). ~50 cheap calls total."""

import json
import os
from typing import TypedDict

HERE = os.path.dirname(os.path.abspath(__file__))
CAND_F = os.path.join(HERE, "_openai_candidates.json")
MODEL = os.environ.get("OPENAI_CHAT", "gpt-5-nano")
N = int(os.environ.get("CAL_N", "25"))


def _key():
    if os.environ.get("OPENAI_API_KEY"):
        return
    d = HERE
    for _ in range(6):
        p = os.path.join(d, ".env")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.lower().strip().startswith("openai_api_key"):
                    os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
                    return
        d = os.path.dirname(d)


class _Decision(TypedDict):
    is_match: bool
    confidence: float | None
    rationale: str | None


STRUCT_PROMPT = (
    "Do record A and record B refer to the same real-world publication?\nA: {a}\nB: {b}"
)
FREE_PROMPT = (
    "Are record A and record B the same real-world publication? "
    "Answer True or False.\nA: {a}\nB: {b}"
)


def _usage(msg):
    um = getattr(msg, "usage_metadata", None) or {}
    inp = um.get("input_tokens", 0)
    out = um.get("output_tokens", 0)
    details = um.get("output_token_details", {}) or {}
    reasoning = details.get("reasoning", 0)
    return inp, out, reasoning


def main():
    _key()
    rows = json.load(open(CAND_F, encoding="utf-8"))["candidates"][:N]
    from langchain_openai import ChatOpenAI

    effort = os.environ.get("REASONING", "minimal")
    kw = {"reasoning_effort": effort} if effort and effort != "default" else {}
    print(f"# model={MODEL} reasoning_effort={effort or 'default'}", flush=True)
    # structured path (denselinkage matcher)
    llm = ChatOpenAI(model=MODEL, **kw)
    s = llm.with_structured_output(_Decision, include_raw=True)
    si = so = sr = 0
    for r in rows:
        out = s.invoke(STRUCT_PROMPT.format(a=r["a_text"], b=r["b_text"]))
        i, o, rea = _usage(out["raw"])
        si += i
        so += o
        sr += rea
    # free-text path (pyJedAI convention)
    fi = fo = fr = 0
    for r in rows:
        out = llm.invoke(FREE_PROMPT.format(a=r["a_text"], b=r["b_text"]))
        i, o, rea = _usage(out)
        fi += i
        fo += o
        fr += rea

    def per(tot):
        return round(tot / N, 1)

    res = {
        "model": MODEL,
        "n_calibration": N,
        "structured_per_pair": {
            "input": per(si),
            "output": per(so),
            "reasoning": per(sr),
        },
        "freetext_per_pair": {
            "input": per(fi),
            "output": per(fo),
            "reasoning": per(fr),
        },
        "full_run_45880": {
            "structured_input_M": round(per(si) * 45880 / 1e6, 2),
            "structured_output_M": round(per(so) * 45880 / 1e6, 2),
            "freetext_input_M": round(per(fi) * 45880 / 1e6, 2),
            "freetext_output_M": round(per(fo) * 45880 / 1e6, 2),
        },
    }
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
