"""Modern-pipeline experiment for the denselinkage paper (SBCARS revision).

Runs the *modern* ER stack end to end on DBLP-ACM and MEASURES the failure
contract on real LLM verdicts (not the arithmetic projection of Table 4):

  1A  semantic blocking : SentenceTransformerEmbedder(all-MiniLM-L6-v2) + FAISS
                          -> pair completeness@k on the full benchmark (no key).
  1D  real LLM matching : a stratified sample of candidate pairs is matched by
                          a real Gemini model through LangChainMatcher, with
                          structured output. Verdicts are cached.
  1B/1C failure sweep   : the cached real verdicts are replayed through the SAME
                          LangChainMatcher while a controlled fraction f of pairs
                          is made to fail (refusal/timeout/malformed) -> real
                          MatchError via the retry path. We measure silent-vs-
                          excluded F1 and coverage, with failures spread UNIFORMLY
                          and CONCENTRATED on the hardest (lowest-similarity)
                          pairs, to test the uniform-failure assumption.

Stages (argv[1]): "block" | "match" | "sweep". "match" needs GOOGLE_API_KEY in
the environment; the other two do not. Reproducible (fixed seed).
"""

import csv
import json
import os
import random
import re
import sys

from denselinkage import DenseLinker, LabeledPairs, Source
from denselinkage.blocking import DenseBlocker
from denselinkage.core.models import CandidatePair, MatchDecision, MatchError, Record
from denselinkage.core.results import LinkageResult
from denselinkage.embedding import SentenceTransformerEmbedder
from denselinkage.indexing import FaissFlatIndex
from denselinkage.matching import LangChainMatcher, RetryPolicy, ThresholdMatcher
from denselinkage.metrics import linkage_metrics, pair_completeness_at_k

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "dblp_acm")
SAMPLE_F = os.path.join(HERE, "_modern_sample.json")
VERDICT_F = os.path.join(HERE, "_modern_verdicts.json")
RESULTS_F = os.path.join(HERE, "modern_results.json")

SEED = 20260613
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
N_POS, N_NEG = 100, 100  # stratified matching sample (balanced)
N_MATCH = 200  # how many of the sample to send to the real LLM
CONCURRENCY = 4  # parallel real calls (billed account -> higher rate limit)
PROMPT = (
    "Do record A and record B refer to the same real-world publication?\n"
    "A: {record_a}\nB: {record_b}"
)


# --------------------------------------------------------------------------- data
class _TextField:
    """Serializer: render a record from its concatenated 'text' field."""

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


def _gold_set():
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


# ------------------------------------------------------------------------ metrics
def _metrics(sample, decided_idx, match_idx, errored_idx):
    """Build a LinkageResult from a partition of the sample and score it.

    decided_idx -> got a verdict; match_idx subset of decided -> predicted match;
    errored_idx -> excluded (counted as n_errors). Returns (excluded, silent)
    LinkageMetrics, where 'silent' folds the errored pairs into non-matches.
    """

    def cp(row):
        return CandidatePair(
            record_a=Record(id=row["a_id"], text=row["a_text"]),
            record_b=Record(id=row["b_id"], text=row["b_text"]),
            similarity_score=row["sim"],
        )

    gold = LabeledPairs.from_pairs(
        [(r["a_id"], r["b_id"]) for r in sample if r["is_gold"]]
    )
    decisions, errors, silent = [], [], []
    for i, row in enumerate(sample):
        if i in errored_idx:
            errors.append((cp(row), MatchError(reason="injected failure")))
            silent.append((cp(row), MatchDecision(is_match=False)))
        elif i in decided_idx:
            dec = MatchDecision(is_match=(i in match_idx))
            decisions.append((cp(row), dec))
            silent.append((cp(row), dec))
    m_excl = linkage_metrics(
        LinkageResult(decisions=tuple(decisions), errors=tuple(errors)),
        gold=gold,
        directed=True,
    )
    m_sil = linkage_metrics(
        LinkageResult(decisions=tuple(silent), errors=()), gold=gold, directed=True
    )
    return m_excl, m_sil


# -------------------------------------------------------------------------- stages
def stage_block():
    left, right, na, nb = _sources()
    gold_pairs = _gold_set()
    gold_set = set(gold_pairs)
    gold = LabeledPairs.from_pairs(gold_pairs)
    print(f"|A|={na} |B|={nb} gold={len(gold_pairs)}")

    blocker = DenseBlocker(
        embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
        vector_index=FaissFlatIndex(),
        top_k=20,
    )
    linker = DenseLinker(blocker=blocker, matcher=ThresholdMatcher(threshold=0.0))
    cands = linker.block(left, right, top_k=20)
    pc = {
        k: round(pair_completeness_at_k(cands, gold=gold, k=k, directed=True), 4)
        for k in (1, 5, 10, 20)
    }
    print("[1A semantic blocking] candidates:", len(cands), "pair_completeness:", pc)

    pos, neg = [], []
    for c in cands:
        row = {
            "a_id": c.record_a.id,
            "a_text": c.record_a.text,
            "b_id": c.record_b.id,
            "b_text": c.record_b.text,
            "sim": float(c.similarity_score) if c.similarity_score is not None else 0.0,
            "is_gold": (c.record_a.id, c.record_b.id) in gold_set,
        }
        (pos if row["is_gold"] else neg).append(row)
    rng = random.Random(SEED)
    rng.shuffle(pos)
    neg.sort(key=lambda r: r["sim"], reverse=True)  # hardest negatives first
    sample = pos[:N_POS] + neg[:N_NEG]
    rng.shuffle(sample)
    payload = {
        "blocking_pc": pc,
        "n_candidates": len(cands),
        "n_a": na,
        "n_b": nb,
        "n_gold": len(gold_pairs),
        "sample": sample,
    }
    with open(SAMPLE_F, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(
        f"saved sample: {len(sample)} pairs "
        f"({sum(r['is_gold'] for r in sample)} gold) -> {SAMPLE_F}"
    )


def stage_match():
    with open(SAMPLE_F, encoding="utf-8") as fh:
        blob = json.load(fh)
    sample = blob["sample"][:N_MATCH]
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.0, max_retries=2)
    matcher = LangChainMatcher(
        llm=llm,
        prompt=PROMPT,
        retry=RetryPolicy(max_retries=3, backoff_seconds=5.0),
        max_concurrency=CONCURRENCY,
    )
    pairs = [
        CandidatePair(
            record_a=Record(id=r["a_id"], text=r["a_text"]),
            record_b=Record(id=r["b_id"], text=r["b_text"]),
            similarity_score=r["sim"],
        )
        for r in sample
    ]
    print(
        f"[1D real LLM] matching {len(pairs)} pairs with {MODEL} "
        f"(concurrency {CONCURRENCY}) ...",
        flush=True,
    )
    outcomes = matcher.match(pairs)
    verdicts = [
        {"ok": True, "is_match": bool(o.is_match)}
        if isinstance(o, MatchDecision)
        else {"ok": False, "reason": o.reason[:120]}
        for o in outcomes
    ]
    with open(VERDICT_F, "w") as fh:
        json.dump(
            {"model": MODEL, "verdicts": verdicts, "sample_used": sample}, fh, indent=1
        )
    decided = {i for i, v in enumerate(verdicts) if v["ok"]}
    match_idx = {i for i in decided if verdicts[i].get("is_match")}
    m_excl, _ = _metrics(sample, decided, match_idx, set())
    n_err = len(sample) - len(decided)
    print(
        f"[1D] decided={len(decided)} real_failures={n_err}  base P/R/F1="
        f"{m_excl.precision:.3f} {m_excl.recall:.3f} {m_excl.f1:.3f}",
        flush=True,
    )
    print(f"saved {VERDICT_F}", flush=True)


class _ReplayModel:
    """Replays cached real verdicts through the real LangChainMatcher, raising on
    a chosen set of pairs so the matcher's retry->MatchError path fires for real.
    Pairs are keyed by a sentinel ``<<Pi>>`` injected into record_a text."""

    def __init__(self, verdicts, fail_idx):
        self._v = verdicts
        self._fail = fail_idx

    def with_structured_output(self, schema):
        from langchain_core.runnables import RunnableLambda

        def responder(prompt_value):
            m = re.search(r"<<P(\d+)>>", prompt_value.to_string())
            i = int(m.group(1))
            if i in self._fail:
                raise RuntimeError("injected: refusal/timeout/malformed output")
            return {"is_match": self._v[i], "confidence": None, "rationale": None}

        return RunnableLambda(responder)


def stage_sweep():
    with open(SAMPLE_F, encoding="utf-8") as fh:
        blob = json.load(fh)
    with open(VERDICT_F, encoding="utf-8") as fh:
        vblob = json.load(fh)
    sample = vblob["sample_used"]  # the exact subset sent to the real LLM
    verdicts = vblob["verdicts"]
    model = vblob["model"]

    # restrict the sweep to pairs the real model actually decided
    base = [i for i, v in enumerate(verdicts) if v["ok"]]
    sub = [sample[i] for i in base]
    sub_match = [bool(verdicts[i]["is_match"]) for i in base]
    n = len(sub)
    # rebuild pairs with a sentinel so the replay is order-independent
    pairs = [
        CandidatePair(
            record_a=Record(id=r["a_id"], text=f"<<P{j}>> {r['a_text']}"),
            record_b=Record(id=r["b_id"], text=r["b_text"]),
            similarity_score=r["sim"],
        )
        for j, r in enumerate(sub)
    ]
    hard_order = sorted(
        range(n), key=lambda j: sub[j]["sim"]
    )  # hardest (low sim) first

    def run(fail_idx):
        matcher = LangChainMatcher(
            llm=_ReplayModel(sub_match, fail_idx),
            prompt=PROMPT,
            retry=RetryPolicy(max_retries=0),
            max_concurrency=1,
        )
        outcomes = matcher.match(pairs)
        decided = {j for j, o in enumerate(outcomes) if isinstance(o, MatchDecision)}
        errored = {j for j, o in enumerate(outcomes) if isinstance(o, MatchError)}
        match_idx = {j for j in decided if sub_match[j]}
        return _metrics(sub, decided, match_idx, errored)

    rows = []
    for f in (0.0, 0.05, 0.10, 0.20):
        k = round(f * n)
        for regime, sel in (
            ("uniform", set(random.Random(SEED).sample(range(n), k))),
            ("hard", set(hard_order[:k])),
        ):
            if f == 0.0 and regime == "hard":
                continue
            m_excl, m_sil = run(sel)
            cov = (n - len(sel)) / n
            rows.append(
                {
                    "f": f,
                    "regime": regime,
                    "coverage": round(cov, 3),
                    "f1_excl": round(m_excl.f1, 3),
                    "f1_silent": round(m_sil.f1, 3),
                    "recall_excl": round(m_excl.recall, 3),
                    "recall_silent": round(m_sil.recall, 3),
                }
            )

    out = {
        "model": model,
        "blocking_pc": blob["blocking_pc"],
        "n_candidates": blob["n_candidates"],
        "n_a": blob["n_a"],
        "n_b": blob["n_b"],
        "n_gold": blob["n_gold"],
        "sample_size": len(sample),
        "decided_in_sweep": n,
        "rows": rows,
    }
    with open(RESULTS_F, "w") as fh:
        json.dump(out, fh, indent=1)
    base0 = next(r for r in rows if r["f"] == 0.0)
    print(f"model={model}  decided={n}  base F1={base0['f1_excl']}")
    hdr = (
        f"{'f':>5} {'regime':>8} {'coverage':>9} {'F1_excl':>8} "
        f"{'F1_silent':>10} {'rec_excl':>9} {'rec_silent':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['f']:>5.2f} {r['regime']:>8} {r['coverage']:>9.2f} "
            f"{r['f1_excl']:>8.3f} {r['f1_silent']:>10.3f} "
            f"{r['recall_excl']:>9.3f} {r['recall_silent']:>11.3f}"
        )
    print(f"saved {RESULTS_F}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "block"
    {"block": stage_block, "match": stage_match, "sweep": stage_sweep}[stage]()
