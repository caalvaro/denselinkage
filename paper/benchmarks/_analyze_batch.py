"""Diagnose the denselinkage(structured) vs pyJedAI(free-text) recall gap on the
same gpt-5-nano batch: compare per-pair decisions on GOLD pairs and dump the
structured rationale for gold pairs the structured matcher called non-match."""

import json
import os

from batch_run import _client, _content, _download, _rows
from openai_pipeline_experiment import _gold_pairs

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    c = _client()
    st = json.load(open(os.path.join(HERE, "_batch_state.json")))
    rows = _rows()
    gold = {(a, b) for a, b in _gold_pairs()}
    so = _download(c, st["struct"]["batch_id"])
    fo = _download(c, st["free"]["batch_id"])

    g_struct_true = g_free_true = g_n = 0
    neg_struct_true = neg_free_true = neg_n = 0
    disagree = []  # gold, struct=false, free=true
    for i, r in enumerate(rows):
        is_g = (r["a_id"], r["b_id"]) in gold
        sct = _content(so.get(f"s{i}"))
        fct = _content(fo.get(f"f{i}"))
        s_match = None
        if sct:
            try:
                s_match = bool(json.loads(sct).get("is_match"))
            except Exception:
                s_match = None
        f_true = (fct or "").split(".")[0].split("\n")[0].strip() == "True"
        if is_g:
            g_n += 1
            g_struct_true += int(s_match is True)
            g_free_true += int(f_true)
            if s_match is False and f_true and len(disagree) < 6:
                try:
                    rat = json.loads(sct).get("rationale")
                except Exception:
                    rat = sct
                disagree.append(
                    {
                        "a": r["a_text"][:90],
                        "b": r["b_text"][:90],
                        "struct_rationale": (rat or "")[:240],
                    }
                )
        else:
            neg_n += 1
            neg_struct_true += int(s_match is True)
            neg_free_true += int(f_true)

    print(
        json.dumps(
            {
                "gold_pairs": g_n,
                "gold_struct_match_rate": round(g_struct_true / g_n, 3),
                "gold_free_match_rate": round(g_free_true / g_n, 3),
                "neg_pairs": neg_n,
                "neg_struct_match_rate": round(neg_struct_true / neg_n, 3),
                "neg_free_match_rate": round(neg_free_true / neg_n, 3),
            },
            indent=1,
        )
    )
    print("\n--- gold pairs: structured said NON-match, free-text said match ---")
    for d in disagree:
        print(f"\nA: {d['a']}\nB: {d['b']}\nstruct rationale: {d['struct_rationale']}")


if __name__ == "__main__":
    main()
