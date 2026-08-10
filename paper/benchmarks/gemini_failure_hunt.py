"""Hunt for NATURAL failures of the modern pipeline on DBLP-ACM, fully on Gemini.

Blocking uses a real Gemini embedding model (models/gemini-embedding-001) behind
a small Embedder adapter; matching uses gemini-2.5-flash-lite with structured
output. We run a large sample of candidate pairs and count how many the matcher
could not decide -- failures that flow through its real retry -> MatchError path
-- classifying each as a rate-limit artifact vs a genuine format/parse/refusal
failure. Embedding failures are counted too. The API key is read from the
nearest .env (key=gemini_api_key). Chunked and saved incrementally.
"""

import collections
import csv
import json
import os
import random

import numpy as np
import numpy.typing as npt

from denselinkage import DenseLinker, Source
from denselinkage.blocking import DenseBlocker
from denselinkage.core.models import MatchDecision
from denselinkage.core.ports import Embedder
from denselinkage.indexing import FaissFlatIndex
from denselinkage.matching import LangChainMatcher, RetryPolicy, ThresholdMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dblp_acm")
OUT = os.path.join(HERE, "gemini_failure_hunt.json")
SEED = 20260613
CHAT_MODEL = os.environ.get("GEMINI_CHAT", "gemini-2.5-flash-lite")
EMB_MODEL = os.environ.get("GEMINI_EMB", "models/gemini-embedding-001")
N_BIG = int(os.environ.get("N_BIG", "4000"))
CHUNK = 200
CONC = 8
PROMPT = (
    "Do record A and record B refer to the same real-world publication?\n"
    "A: {record_a}\nB: {record_b}"
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
    raise SystemExit("no gemini_api_key found in a .env on the path")


class GeminiEmbedder(Embedder):
    """Embedder adapter over a Gemini embedding model (L2-normalised rows)."""

    def __init__(self, model: str = EMB_MODEL, batch: int = 50) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._client = GoogleGenerativeAIEmbeddings(model=model)
        self._model = model
        self._batch = batch
        self._dim = 0
        self.embed_failures = 0

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def embedding_dim(self) -> int:
        if not self._dim:
            self._dim = len(self._client.embed_query("dimension probe"))
        return self._dim

    def encode(
        self, texts, *, batch_size=None, show_progress=False
    ) -> npt.NDArray[np.float32]:
        bs = batch_size or self._batch
        rows: list[list[float]] = []
        for i in range(0, len(texts), bs):
            chunk = list(texts[i : i + bs])
            try:
                rows.extend(self._client.embed_documents(chunk))
            except Exception as exc:  # an embedding-API failure
                self.embed_failures += 1
                print(f"  [EMB FAILURE] batch {i}: {exc!r}"[:140], flush=True)
                rows.extend([[0.0] * self.embedding_dim] * len(chunk))
            if show_progress:
                print(f"  embedded {min(i + bs, len(texts))}/{len(texts)}", flush=True)
        arr = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class _TextField:
    def serialize(self, record):  # type: ignore[no-untyped-def]
        return record["text"]


def _load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _frame(rows, prefix):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "id": prefix + r["id"],
                "text": " ".join(r[c] for c in ("title", "authors", "venue", "year")),
            }
            for r in rows
        ]
    )


def _bucket(reason: str) -> str:
    r = reason.lower()
    if "429" in reason or "resource_exhausted" in r:
        return "rate_limit"
    if any(k in r for k in ("parse", "schema", "valid", "format", "json", "output")):
        return "format_parse"
    if any(k in r for k in ("refus", "safety", "blocked", "recitation")):
        return "refusal"
    return "other"


def main():
    _load_api_key()
    a, b = _frame(_load("tableA.csv"), "A"), _frame(_load("tableB.csv"), "B")
    left = Source(a, id_column="id", serializer=_TextField())
    right = Source(b, id_column="id", serializer=_TextField())

    embedder = GeminiEmbedder()
    print(
        f"embedding+blocking with {EMB_MODEL} (dim {embedder.embedding_dim}) ...",
        flush=True,
    )
    blocker = DenseBlocker(embedder=embedder, vector_index=FaissFlatIndex(), top_k=5)
    cands = DenseLinker(blocker=blocker, matcher=ThresholdMatcher(threshold=0.0)).block(
        left, right, top_k=5
    )
    print(
        f"blocking candidates={len(cands)}  embed_failures={embedder.embed_failures}",
        flush=True,
    )

    rng = random.Random(SEED)
    rng.shuffle(cands)
    sample = cands[:N_BIG]
    print(f"matching {len(sample)} pairs with {CHAT_MODEL} ...", flush=True)

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.0, max_retries=0)
    matcher = LangChainMatcher(
        llm=llm,
        prompt=PROMPT,
        retry=RetryPolicy(max_retries=1, backoff_seconds=3.0),
        max_concurrency=CONC,
    )

    decided = 0
    errors = []
    buckets: collections.Counter = collections.Counter()
    for i in range(0, len(sample), CHUNK):
        chunk = sample[i : i + CHUNK]
        out = matcher.match(chunk)
        for pair, o in zip(chunk, out, strict=True):
            if isinstance(o, MatchDecision):
                decided += 1
            else:
                errors.append(
                    {
                        "a": pair.record_a.id,
                        "b": pair.record_b.id,
                        "reason": o.reason[:200],
                    }
                )
                buckets[_bucket(o.reason)] += 1
        json.dump(
            {
                "chat_model": CHAT_MODEL,
                "emb_model": EMB_MODEL,
                "n_sampled": len(sample),
                "decided": decided,
                "embed_failures": embedder.embed_failures,
                "match_errors": len(errors),
                "reason_buckets": dict(buckets),
                "errors": errors[:60],
            },
            open(OUT, "w"),
            indent=1,
        )
        print(
            f"  {i + len(chunk)}/{len(sample)}  decided={decided} "
            f"match_errors={len(errors)} buckets={dict(buckets)}",
            flush=True,
        )
    print(
        f"DONE  decided={decided}  match_errors={len(errors)}  "
        f"embed_failures={embedder.embed_failures}  buckets={dict(buckets)}  -> {OUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
