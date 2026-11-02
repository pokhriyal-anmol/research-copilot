"""Method 1 screening (BENCHMARK.md): one-hop pool vs the dense top 200 on dev, no LLM.

Run from the repo root: uv run --group bench python bench/graph/one_hop_screen.py
"""
import json
from datetime import UTC, datetime
from pathlib import Path

from copilot.litsearch.data import load
from copilot.litsearch.dense import DenseIndex
from copilot.litsearch.encoders import Qwen3Encoder
from copilot.litsearch.graph import one_hop
from copilot.litsearch.metrics import mean_scores
from copilot.litsearch.split import split_questions

OUT = Path("bench/results/ablations/graph/one-hop-m50-n200-dev.json")
GROUPS = {  # BENCHMARK.md, citation-graph generalisation rules
    "all": None,
    "source not in corpus (inline_nonacl)": {"inline_nonacl"},
    "author-written (manual_acl, manual_iclr)": {"manual_acl", "manual_iclr"},
    "source may be in corpus (inline_acl, not counted)": {"inline_acl"},
}


def main() -> None:
    data = load()
    dev, _ = split_questions([{"id": q.id, "query_set": q.query_set, "specificity": q.specificity}
                              for q in data.questions], dev_size=200, seed=13)
    questions = [q for q in data.questions if q.id in set(dev)]
    index = DenseIndex(data.docs, Qwen3Encoder("Qwen/Qwen3-Embedding-8B", dtype="float16", max_length=1024))
    dense = {q.id: index.search(q.text, 200) for q in questions}
    pool = {q.id: one_hop(dense[q.id], data.citations) for q in questions}
    report = {"method": "one-hop m=50 n=200, references only, interleaved", "base": "Qwen3-Embedding-8B, protocol instruction",
              "split": "dev", "date": datetime.now(UTC).isoformat(timespec="seconds"),
              "mean_pool_size": sum(map(len, pool.values())) / len(pool), "groups": {}}
    for name, sets in GROUPS.items():
        qs = [q for q in questions if sets is None or q.query_set in sets]
        gold = {q.id: q.gold for q in qs}
        report["groups"][name] = {"questions": len(qs), **{
            label: mean_scores({q.id: runs[q.id] for q in qs}, gold, (20, 200), (10,))
            for label, runs in (("dense", dense), ("one-hop", pool))}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    for name, g in report["groups"].items():
        d, o = g["dense"], g["one-hop"]
        print(f"{name:52} n={g['questions']:3}  R@200 {d['recall@200']:5.1f} -> {o['recall@200']:5.1f}   "
              f"as-is R@20 {d['recall@20']:5.1f} -> {o['recall@20']:5.1f}  nDCG@10 {d['ndcg@10']:5.1f} -> {o['ndcg@10']:5.1f}")
    print(f"mean pool size {report['mean_pool_size']:.0f}")


if __name__ == "__main__":
    main()
