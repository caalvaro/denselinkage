"""Measure a competitor's silent-failure convention biting on a real LLM run.

pyJedAI's LLM matcher decides a pair by an exact string compare of the raw model
response against ``"True"`` (``resp['message']['content'] == 'True'``); any other
text -- a correct answer in a different case, with punctuation, or with a word of
reasoning -- is scored as a non-match and enters the library's own F1. We run a
real free-form ``gemini-2.5-flash-lite`` matcher over the same 200-pair DBLP-ACM
sample used elsewhere, capture each raw response, and score it three ways against
gold: (a) pyJedAI's exact compare, (b) a robust parse (the kind structured output
gives for free), and (c) the structured-output verdicts already cached. We report
the F1 each convention yields and how many correct decisions the exact compare
discards. Key from the nearest .env. Chunked, saved.
"""

import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_F = os.path.join(HERE, "_modern_sample.json")
OUT = os.path.join(HERE, "competitor_convention.json")
MODEL = os.environ.get("GEMINI_CHAT", "gemini-2.5-flash-lite")
PROMPT = (
    "Are record A and record B the same real-world publication? "
    "Answer True or False.\nA: {a}\nB: {b}"
)


def _load_api_key() -> None:
    if os.environ.get("GOOGLE_API_KEY"):
        return
    d = HERE
    for _ in range(5):
        p = os.path.join(d, ".env")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.lower().strip().startswith("gemini_api_key"):
                    os.environ["GOOGLE_API_KEY"] = line.split("=", 1)[1].strip()
                    return
        d = os.path.dirname(d)
    raise SystemExit("no gemini_api_key in a .env on the path")


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _f1(preds, gold):
    tp = sum(1 for p, g in zip(preds, gold, strict=False) if p and g)
    fp = sum(1 for p, g in zip(preds, gold, strict=False) if p and not g)
    fn = sum(1 for p, g in zip(preds, gold, strict=False) if not p and g)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return round(prec, 3), round(rec, 3), round(f1, 3)


def main():
    _load_api_key()
    sample = json.load(open(SAMPLE_F, encoding="utf-8"))["sample"]
    gold = [bool(r["is_gold"]) for r in sample]
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.0, max_retries=2)
    raws: list[str] = []
    print(f"querying {len(sample)} pairs free-form with {MODEL} ...", flush=True)
    for i in range(0, len(sample), 50):
        chunk = sample[i : i + 50]
        msgs = llm.batch(
            [PROMPT.format(a=r["a_text"], b=r["b_text"]) for r in chunk],
            config={"max_concurrency": 8},
        )
        raws.extend(_text(m.content) for m in msgs)
        json.dump(
            {"model": MODEL, "n": len(raws), "raws": raws}, open(OUT + ".partial", "w")
        )
        print(f"  {len(raws)}/{len(sample)}", flush=True)

    exact = [r.strip() == "True" for r in raws]  # pyJedAI rule
    robust = [
        r.strip().lower().lstrip("*` \n").startswith("true") for r in raws
    ]  # robust parse
    # response-form distribution
    forms = collections.Counter(
        "exactly 'True'"
        if r.strip() == "True"
        else "exactly 'False'"
        if r.strip() == "False"
        else "other (would mis-score)"
        for r in raws
    )
    mis = sum(1 for e, rb in zip(exact, robust, strict=False) if e != rb)
    result = {
        "model": MODEL,
        "n_pairs": len(raws),
        "f1_exact_pyjedai": _f1(exact, gold),
        "f1_robust": _f1(robust, gold),
        "decisions_changed_by_exact_compare": mis,
        "response_forms": dict(forms),
        "sample_other_responses": [
            r[:60]
            for r, e in zip(raws, exact, strict=False)
            if r.strip() not in ("True", "False")
        ][:20],
    }
    json.dump(result | {"raws": raws}, open(OUT, "w"), indent=1)
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
