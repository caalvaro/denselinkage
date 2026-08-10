"""Re-match only the pairs whose cached denselinkage verdict failed (rate-limit
MatchErrors from earlier API contention), contention-free, then recompute the
clean end-to-end metrics. Run after openai_pipeline_experiment.py match finishes.
"""

import json
import os

from _openai_adapters import load_openai_key
from openai_pipeline_experiment import (
    CAND_F,
    CHAT,
    PROMPT,
    RESULTS_F,
    VERDICT_F,
    _build_pairs,
    _gold_pairs,
)

from denselinkage import LabeledPairs
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.core.results import LinkageResult
from denselinkage.matching import LangChainMatcher, RetryPolicy
from denselinkage.metrics import linkage_metrics


def main():
    load_openai_key()
    json.load(open(CAND_F, encoding="utf-8"))
    v = json.load(open(VERDICT_F, encoding="utf-8"))
    rows, verdicts = v["matched_rows"], v["verdicts"]
    failed = [i for i, vv in enumerate(verdicts) if not vv["ok"]]
    print(f"failed verdicts to retry: {len(failed)}", flush=True)

    if failed:
        from langchain_openai import ChatOpenAI

        matcher = LangChainMatcher(
            llm=ChatOpenAI(model=CHAT, max_retries=5),
            prompt=PROMPT,
            retry=RetryPolicy(max_retries=5, backoff_seconds=8.0),
            max_concurrency=6,  # low: contention-free, prioritise success
        )
        sub = [rows[i] for i in failed]
        out = []
        for j in range(0, len(sub), 500):
            out.extend(matcher.match(_build_pairs(sub[j : j + 500])))
            print(f"  retried {min(j + 500, len(sub))}/{len(sub)}", flush=True)
        for i, o in zip(failed, out, strict=True):
            if isinstance(o, MatchDecision):
                verdicts[i] = {"ok": True, "is_match": bool(o.is_match)}
            else:
                verdicts[i] = {"ok": False, "reason": o.reason[:160]}

    decisions, errors = [], []
    for r, vv in zip(rows, verdicts, strict=True):
        cp = CandidatePair(
            record_a=Record(id=r["a_id"], text=r["a_text"]),
            record_b=Record(id=r["b_id"], text=r["b_text"]),
            similarity_score=r["sim"],
        )
        if vv["ok"]:
            decisions.append((cp, MatchDecision(is_match=vv["is_match"])))
        else:
            errors.append((cp, MatchError(reason=vv.get("reason", "fail"))))

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

    res = (
        json.load(open(RESULTS_F, encoding="utf-8"))
        if os.path.exists(RESULTS_F)
        else {}
    )
    res.update(
        {
            "n_matched": len(rows),
            "n_decided": len(decisions),
            "n_natural_failures": len(errors),
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
            "retried_failures": len(failed),
            "residual_failures_after_retry": len(errors),
        }
    )
    json.dump(res, open(RESULTS_F, "w"), indent=1)
    json.dump({**res, "verdicts": verdicts, "matched_rows": rows}, open(VERDICT_F, "w"))
    print(
        json.dumps(
            {
                k: res[k]
                for k in (
                    "n_decided",
                    "n_natural_failures",
                    "residual_failures_after_retry",
                    "end_to_end_excluded",
                    "end_to_end_silent",
                )
            },
            indent=1,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
