"""Score a citation-graph method against the dense 8B top 200 on dev, no LLM (BENCHMARK.md).

Run from the repo root: uv run --group bench python bench/graph/screen.py one-hop|structural|regularise-0.1|regularise-0.3|regularise-0.5
"""
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from copilot.litsearch.data import load
from copilot.litsearch.dense import DenseIndex
from copilot.litsearch.encoders import Qwen3Encoder
from copilot.litsearch.graph import one_hop, regularise, structural_rerank
from copilot.litsearch.metrics import mean_scores
from copilot.litsearch.split import split_questions

OUT = Path("bench/results/ablations/graph")

def _ids(scored):
    return [d for d, _ in scored]


METHODS = {  # name: (description, result file, reorder or expand the scored dense top 200)
    "one-hop": ("one-hop m=50 n=200, references only, interleaved", "one-hop-m50-n200-dev.json",
                lambda scored, cites: one_hop(_ids(scored), cites)),
    "structural": ("structural re-ranking: PageRank d=0.85 over citations in the top 200, RRF k=60",
                   "structural-d085-k60-dev.json", lambda scored, cites: structural_rerank(_ids(scored), cites)),
    **{f"regularise-{a}": (f"score regularisation a={a} over undirected citations in the top 200",
                           f"regularise-a{a}-dev.json", lambda scored, cites, a=a: regularise(scored, cites, a))
       for a in (0.1, 0.3, 0.5)},  # the three weights pre-registered in BENCHMARK.md
}
GROUPS = {  # BENCHMARK.md, citation-graph generalisation rules
    "all": None,
    "source not in corpus (inline_nonacl)": {"inline_nonacl"},
    "author-written (manual_acl, manual_iclr)": {"manual_acl", "manual_iclr"},
    "source may be in corpus (inline_acl, not counted)": {"inline_acl"},
}


def main(method: str) -> None:
    description, filename, apply = METHODS[method]
    data = load()
    dev, _ = split_questions([{"id": q.id, "query_set": q.query_set, "specificity": q.specificity}
                              for q in data.questions], dev_size=200, seed=13)
    questions = [q for q in data.questions if q.id in set(dev)]
    index = DenseIndex(data.docs, Qwen3Encoder("Qwen/Qwen3-Embedding-8B", dtype="float16", max_length=1024))
    scored = {q.id: index.search_scored(q.text, 200) for q in questions}
    dense = {qid: _ids(s) for qid, s in scored.items()}
    pool = {q.id: apply(scored[q.id], data.citations) for q in questions}
    report = {"method": description, "base": "Qwen3-Embedding-8B, protocol instruction",
              "split": "dev", "date": datetime.now(UTC).isoformat(timespec="seconds"),
              "mean_pool_size": sum(map(len, pool.values())) / len(pool), "groups": {}}
    for name, sets in GROUPS.items():
        qs = [q for q in questions if sets is None or q.query_set in sets]
        gold = {q.id: q.gold for q in qs}
        report["groups"][name] = {"questions": len(qs), **{
            label: mean_scores({q.id: runs[q.id] for q in qs}, gold, (20, 200), (10,))
            for label, runs in (("dense", dense), (method, pool))}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / filename).write_text(json.dumps(report, indent=2))
    for name, g in report["groups"].items():
        d, o = g["dense"], g[method]
        print(f"{name:52} n={g['questions']:3}  R@200 {d['recall@200']:5.1f} -> {o['recall@200']:5.1f}   "
              f"R@20 {d['recall@20']:5.1f} -> {o['recall@20']:5.1f}  nDCG@10 {d['ndcg@10']:5.1f} -> {o['ndcg@10']:5.1f}")
    print(f"mean pool size {report['mean_pool_size']:.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
