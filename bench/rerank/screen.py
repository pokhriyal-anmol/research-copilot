"""Screen a listwise LLM over the Qwen3-Reranker-4B top 20 on 55 dev questions (BENCHMARK.md, "Screen: ...").

Run from the repo root, with OPENROUTER_API_KEY in the environment:
  HF_HUB_OFFLINE=1 uv run --group bench python bench/rerank/screen.py
Replies are cached, so a run stopped by the free daily cap resumes where it left off. A stopped run writes no result.
Add --first N to score only the first N questions (logged in BENCHMARK.md as a deviation when used).
"""
import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from copilot.litsearch.metrics import recall_at_k

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
URL = "https://openrouter.ai/api/v1/chat/completions"
QUERY_SET, SPECIFIC, DEPTH = "inline_nonacl", 1, 20
CACHE = Path("bench/data/llm-cache")
OUT = "bench/results/ablations/rerank/screen-nemotron-3-ultra-top20-dev{}.json"


def rose_fell(base: dict, llm: dict, gold: dict, k: int = 5) -> tuple[int, int]:
    """How many questions' Recall@k went up and down."""
    deltas = [recall_at_k(llm[q], g, k) - recall_at_k(base[q], g, k) for q, g in gold.items()]
    return sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)


def verdict(base: dict, llm: dict, rose: int, fell: int, failures: int) -> str:
    worth = (llm["recall@5"] - base["recall@5"] >= 5.0 - 1e-9 and rose > fell
             and llm["ndcg@10"] >= base["ndcg@10"] and failures <= 2)
    return "worth a full configuration" if worth else "not worth it"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int)
    first = ap.parse_args().first
    import httpx

    from copilot.litsearch.cross import Qwen3Reranker, reorder
    from copilot.litsearch.data import load
    from copilot.litsearch.dense import DenseIndex
    from copilot.litsearch.encoders import Qwen3Encoder
    from copilot.litsearch.metrics import mean_scores
    from copilot.litsearch.rerank import ModelFailed, cached, chat, rerank
    from copilot.litsearch.split import split_questions

    data = load()
    dev, _ = split_questions([{"id": q.id, "query_set": q.query_set, "specificity": q.specificity}
                              for q in data.questions], dev_size=200, seed=13)
    questions = [q for q in data.questions if q.id in set(dev) and q.query_set == QUERY_SET and q.specificity == SPECIFIC][:first]
    index = DenseIndex(data.docs, Qwen3Encoder("Qwen/Qwen3-Embedding-8B", dtype="float16", max_length=1024))
    dense = {q.id: index.search(q.text, 200) for q in questions}
    cross = Qwen3Reranker("Qwen/Qwen3-Reranker-4B")
    cross._load = lambda: (_ for _ in ()).throw(RuntimeError("Qwen score cache miss"))  # configuration 2 is cached
    base = {q.id: reorder(dense[q.id], cross.scores(q.text, [data.docs[d] for d in dense[q.id][:100]]), 100)
            for q in questions}
    llm = cached(chat(httpx.Client(), os.environ["OPENROUTER_API_KEY"], MODEL, url=URL), MODEL, CACHE)
    reranked, failures = {}, []
    for i, q in enumerate(questions, 1):
        try:
            reranked[q.id], parsed = rerank(q.text, base[q.id], data.docs, llm, depth=DEPTH)
        except ModelFailed as exc:
            sys.exit(f"stopped at {i - 1} of {len(questions)}: {exc}. Cached replies resume on the next run.")
        if not parsed:
            failures.append(q.id)
        print(f"{i}/{len(questions)} failures {len(failures)}", flush=True)
    gold = {q.id: q.gold for q in questions}
    scores = {name: mean_scores(runs, gold, (5, 20), (10,)) for name, runs in (("qwen", base), ("llm", reranked))}
    rose, fell = rose_fell(base, reranked, gold)
    result = verdict(scores["qwen"], scores["llm"], rose, fell, len(failures))
    Path(OUT.format(f"-first{first}" if first else "")).write_text(json.dumps({"model": MODEL, "depth": DEPTH, "questions": len(questions), "scores": scores,
                               "recall@5 rose": rose, "recall@5 fell": fell, "failures": failures, "verdict": result,
                               "date": datetime.now(UTC).isoformat(timespec="seconds")}, indent=2))
    print(json.dumps(scores), f"rose {rose} fell {fell} failures {len(failures)}: {result}")


if __name__ == "__main__":
    main()
