"""Smoke test: verify the OpenAI key + gpt-5.4-nano + text-embedding-3-large
work, including structured output (the failure-contract path). 3 tiny calls."""

import os
from typing import TypedDict

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_key() -> None:
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
    raise SystemExit("no openai_api_key in a .env on the path")


CHAT = os.environ.get("OPENAI_CHAT", "gpt-5.4-nano")
EMB = os.environ.get("OPENAI_EMB", "text-embedding-3-large")


class _Decision(TypedDict):
    is_match: bool
    confidence: float | None
    rationale: str | None


def main() -> None:
    _load_key()
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    # 1) plain chat
    try:
        llm = ChatOpenAI(model=CHAT)
        r = llm.invoke("Reply with exactly one word: pong")
        print(f"[chat]  {CHAT}: OK -> {r.content!r}")
    except Exception as e:
        print(f"[chat]  {CHAT}: FAIL -> {type(e).__name__}: {e}")

    # 2) structured output (the failure-contract path)
    try:
        llm = ChatOpenAI(model=CHAT)
        s = llm.with_structured_output(_Decision)
        out = s.invoke(
            "Do these refer to the same paper?\n"
            "A: Efficient query processing, VLDB 2003\n"
            "B: Efficient query processing in DBs, VLDB 03"
        )
        print(f"[struct] {CHAT}: OK -> {out!r}")
    except Exception as e:
        print(f"[struct] {CHAT}: FAIL -> {type(e).__name__}: {e}")

    # 3) embeddings
    try:
        emb = OpenAIEmbeddings(model=EMB)
        v = emb.embed_query("entity resolution")
        print(f"[emb]   {EMB}: OK -> dim={len(v)}")
    except Exception as e:
        print(f"[emb]   {EMB}: FAIL -> {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
