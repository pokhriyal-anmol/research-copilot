"""Rerank the dense 8B top 100 on dev (BENCHMARK.md, reranker stage) and score it by question group.

Run from the repo root:
  uv run --group bench python bench/rerank/run.py --reranker qwen3-4b      # configuration 2, all 200 dev questions
  uv run --group bench python bench/rerank/run.py --reranker gemini        # configuration 1, needs GEMINI_API_KEY
  add --limit 3 --out <dir> for a smoke test, which is not logged
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
from copilot.litsearch.cross import Qwen3Reranker, reorder
from copilot.litsearch.rerank import cached, gemini, rerank
from copilot.litsearch.split import split_questions

DEPTH = 100
CONFIGS = {  # --reranker: (configuration number, model, result file stem)
    "gemini": (1, "gemini-3.5-flash", "gemini-3.5-flash-vanilla"),
    "qwen3-4b": (2, "Qwen/Qwen3-Reranker-4B", "qwen3-reranker-4b"),
}
CACHE = Path("bench/data/llm-cache")
GROUPS = {  # BENCHMARK.md, citation-graph generalisation rules, also used for the reranker
    "all": None,
    "source not in corpus (inline_nonacl)": {"inline_nonacl"},
    "author-written (manual_acl, manual_iclr)": {"manual_acl", "manual_iclr"},
    "source may be in corpus (inline_acl, not counted)": {"inline_acl"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranker", choices=CONFIGS, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="bench/results/ablations/rerank")
    args = ap.parse_args()
    data = load()
    dev, _ = split_questions([{"id": q.id, "query_set": q.query_set, "specificity": q.specificity}
                              for q in data.questions], dev_size=200, seed=13)
    questions = [q for q in data.questions if q.id in set(dev)][: args.limit]
    index = DenseIndex(data.docs, Qwen3Encoder("Qwen/Qwen3-Embedding-8B", dtype="float16", max_length=1024))
    dense = {q.id: index.search(q.text, 200) for q in questions}
    del index  # frees ~1 GB of paper vectors before the reranker loads, on a 16 GB Mac
    number, name, stem = CONFIGS[args.reranker]
    if args.reranker == "gemini":
        llm = cached(gemini(httpx.Client(), os.environ["GEMINI_API_KEY"], name), name, CACHE)
    else:
        cross = Qwen3Reranker(name)
    reranked, failures, started = {}, [], time.time()
    for i, q in enumerate(questions, 1):
        if args.reranker == "gemini":
            reranked[q.id], parsed = rerank(q.text, dense[q.id], data.docs, llm, depth=DEPTH)
        else:
            head = dense[q.id][:DEPTH]
            reranked[q.id], parsed = reorder(dense[q.id], cross.scores(q.text, [data.docs[d] for d in head]), DEPTH), True
        if not parsed:
            failures.append(q.id)
        print(f"{i}/{len(questions)} {time.time() - started:.0f}s failures {len(failures)}", flush=True)
    report = {"configuration": number, "model": name, "depth": DEPTH, "base": "Qwen3-Embedding-8B, protocol instruction",
              "split": "dev", "questions": len(questions), "failures": failures,
              "date": datetime.now(UTC).isoformat(timespec="seconds"), "groups": {}}
    for group, sets in GROUPS.items():
        qs = [q for q in questions if sets is None or q.query_set in sets]
        if not qs:
            continue
        gold = {q.id: q.gold for q in qs}
        report["groups"][group] = {"questions": len(qs), **{
            label: mean_scores({q.id: runs[q.id] for q in qs}, gold, (20, 100), (10,))
            for label, runs in (("dense", dense), ("reranked", reranked))}}
        d, r = report["groups"][group]["dense"], report["groups"][group]["reranked"]
        print(f"{group:52} n={len(qs):3}  R@20 {d['recall@20']:5.1f} -> {r['recall@20']:5.1f}  "
              f"nDCG@10 {d['ndcg@10']:5.1f} -> {r['ndcg@10']:5.1f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}-top{DEPTH}-dev.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
