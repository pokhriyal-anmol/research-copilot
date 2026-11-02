"""Reranker configuration 1 (BENCHMARK.md): Gemini listwise rerank of the dense 8B top 100, scored on dev.

Run from the repo root, with GEMINI_API_KEY set:
  uv run --group bench python bench/rerank/run.py              # all 200 dev questions, logged as configuration 1
  uv run --group bench python bench/rerank/run.py --limit 3 --out <dir>   # smoke test, not logged
"""
import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from copilot.litsearch.data import load
from copilot.litsearch.dense import DenseIndex
from copilot.litsearch.encoders import Qwen3Encoder
from copilot.litsearch.metrics import mean_scores
from copilot.litsearch.rerank import cached, gemini, rerank
from copilot.litsearch.split import split_questions

MODEL, DEPTH = "gemini-3.5-flash", 100
CACHE = Path("bench/data/llm-cache")
GROUPS = {  # BENCHMARK.md, citation-graph generalisation rules, also used for the reranker
    "all": None,
    "source not in corpus (inline_nonacl)": {"inline_nonacl"},
    "author-written (manual_acl, manual_iclr)": {"manual_acl", "manual_iclr"},
    "source may be in corpus (inline_acl, not counted)": {"inline_acl"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="bench/results/ablations/rerank")
    args = ap.parse_args()
    data = load()
    dev, _ = split_questions([{"id": q.id, "query_set": q.query_set, "specificity": q.specificity}
                              for q in data.questions], dev_size=200, seed=13)
    questions = [q for q in data.questions if q.id in set(dev)][: args.limit]
    index = DenseIndex(data.docs, Qwen3Encoder("Qwen/Qwen3-Embedding-8B", dtype="float16", max_length=1024))
    model = cached(gemini(httpx.Client(), os.environ["GEMINI_API_KEY"], MODEL), MODEL, CACHE)
    dense, reranked, failures, started = {}, {}, [], time.time()
    for i, q in enumerate(questions, 1):
        dense[q.id] = index.search(q.text, 200)
        reranked[q.id], parsed = rerank(q.text, dense[q.id], data.docs, model, depth=DEPTH)
        if not parsed:
            failures.append(q.id)
        print(f"{i}/{len(questions)} {time.time() - started:.0f}s failures {len(failures)}", flush=True)
    report = {"configuration": 1, "model": MODEL, "depth": DEPTH, "base": "Qwen3-Embedding-8B, protocol instruction",
              "split": "dev", "questions": len(questions), "failures": failures,
              "date": datetime.now(UTC).isoformat(timespec="seconds"), "groups": {}}
    for name, sets in GROUPS.items():
        qs = [q for q in questions if sets is None or q.query_set in sets]
        if not qs:
            continue
        gold = {q.id: q.gold for q in qs}
        report["groups"][name] = {"questions": len(qs), **{
            label: mean_scores({q.id: runs[q.id] for q in qs}, gold, (20, 100), (10,))
            for label, runs in (("dense", dense), ("reranked", reranked))}}
        d, r = report["groups"][name]["dense"], report["groups"][name]["reranked"]
        print(f"{name:52} n={len(qs):3}  R@20 {d['recall@20']:5.1f} -> {r['recall@20']:5.1f}  "
              f"nDCG@10 {d['ndcg@10']:5.1f} -> {r['ndcg@10']:5.1f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{MODEL}-vanilla-top{DEPTH}-dev.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
