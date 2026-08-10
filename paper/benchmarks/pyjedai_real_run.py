"""Run pyJedAI 0.3.6's ACTUAL LLM-matching workflow end to end on DBLP-ACM with a
local Ollama model. This executes pyJedAI's real objects -- Data,
EmbeddingsNNBlockBuilding, OllamaMatching.process(), Evaluation -- not a
reproduction. Must run in the pyJedAI venv (_pyjedai_env) and needs Ollama up.

Env: OLLAMA_MODEL (default qwen2.5:7b), PYJ_TOPK (blocking top_k, default 1),
     PYJ_MAXPAIRS (cap candidate pairs for a quick validation; 0 = all).

We log every difficulty encountered to stderr/console as it happens.
"""

import contextlib
import csv
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# pyJedAI's report() prints Unicode box-drawing chars; Windows cp1252 stdout
# crashes on them (a real difficulty). Force UTF-8 so we can capture the score.
for _s in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _s.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dblp_acm")
OUT = os.path.join(HERE, "pyjedai_real_results.json")

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
TOPK = int(os.environ.get("PYJ_TOPK", "1"))
MAXPAIRS = int(os.environ.get("PYJ_MAXPAIRS", "0"))


def _read_csv(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    import pandas as pd

    log = []

    def note(msg):
        log.append(msg)
        print("[pyjedai-real]", msg, flush=True)

    # --- datasets as pyJedAI expects: plain DataFrames, string ids ---
    a = pd.DataFrame(_read_csv("tableA.csv")).astype(str)
    b = pd.DataFrame(_read_csv("tableB.csv")).astype(str)
    gold_rows = []
    for f in ("train.csv", "valid.csv", "test.csv"):
        for r in _read_csv(f):
            if r["label"] == "1":
                gold_rows.append((str(r["ltable_id"]), str(r["rtable_id"])))
    gold = pd.DataFrame(gold_rows, columns=["id1", "id2"])
    note(f"|A|={len(a)} |B|={len(b)} gold={len(gold)} model={MODEL} topk={TOPK}")

    from pyjedai.datamodel import Data
    from pyjedai.llm_matching import OllamaMatching
    from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding

    data = Data(
        dataset_1=a,
        id_column_name_1="id",
        dataset_2=b,
        id_column_name_2="id",
        ground_truth=gold,
    )
    note("Data object built")

    # --- pyJedAI's own dense blocking (sentence-transformer + FAISS) ---
    t0 = time.time()
    emb = EmbeddingsNNBlockBuilding(
        vectorizer="sdistilroberta", similarity_search="faiss"
    )
    _blocks, g = emb.build_blocks(
        data,
        top_k=TOPK,
        similarity_distance="cosine",
        with_entity_matching=True,
        load_embeddings_if_exist=True,
    )
    note(f"blocking done in {time.time() - t0:.1f}s; graph edges={g.number_of_edges()}")

    if MAXPAIRS and g.number_of_edges() > MAXPAIRS:
        import networkx as nx

        keep = list(g.edges())[:MAXPAIRS]
        g = nx.Graph()
        g.add_edges_from(keep)
        note(f"capped graph to {g.number_of_edges()} edges for validation")

    # --- pyJedAI's REAL Ollama LLM matcher ---
    matcher = OllamaMatching(llm_model=MODEL)
    note(
        "OllamaMatching constructed; running process() "
        "(creates a server-side model, matches pairs SEQUENTIALLY, deletes it)"
    )
    t1 = time.time()
    try:
        pairs = matcher.process(prediction=g, data=data, tqdm_disable=False)
    except Exception as e:
        note(f"DIFFICULTY: process() raised {type(e).__name__}: {e}")
        raise
    dt = time.time() - t1
    note(
        f"matched {len(pairs)} predicted-positive pairs in {dt:.1f}s "
        f"({dt / max(g.number_of_edges(), 1):.2f}s/pair)"
    )

    # our own metric, independent of pyJedAI's crash-prone report()
    gi = set()
    miss = 0
    for o1, o2 in gold_rows:
        try:
            gi.add((data._ids_mapping_1[o1], data._ids_mapping_2[o2]))
        except KeyError:
            miss += 1
    pred = {(p[0], p[1]) for p in pairs}
    tp = sum(1 for gp in gi if gp in pred or (gp[1], gp[0]) in pred)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gi) if gi else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    own = {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "predicted_positive": len(pred),
        "gold_total": len(gi),
        "gold_unmapped": miss,
    }
    note(f"independent score (end-to-end vs all gold): {own}")

    scores = None
    try:
        scores = matcher.evaluate(pairs, export_to_dict=True, verbose=True)
        note(f"pyJedAI evaluation: {scores}")
    except Exception as e:
        note(f"DIFFICULTY: pyJedAI evaluate() crashed: {type(e).__name__}: {e}")

    json.dump(
        {
            "model": MODEL,
            "blocking_top_k": TOPK,
            "candidate_pairs": g.number_of_edges(),
            "predicted_positive": len(pairs),
            "match_seconds": round(dt, 1),
            "sec_per_pair": round(dt / max(g.number_of_edges(), 1), 3),
            "independent_score": own,
            "pyjedai_scores": scores,
            "difficulty_log": log,
        },
        open(OUT, "w"),
        indent=1,
        default=str,
    )
    print(f"[pyjedai-real] saved {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
