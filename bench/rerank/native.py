"""Score the dense 8B and the Qwen3-Reranker-4B run in LitSearch's own Table 3 metrics, from cache only.

Run from the repo root: HF_HUB_OFFLINE=1 uv run --group bench python bench/rerank/native.py
Broad questions (specificity 0) get R@20, specific ones (1) R@5 and R@20, as in arXiv 2407.18940, Table 3.
"""
import statistics as st
from copilot.litsearch.cross import Qwen3Reranker, reorder
from copilot.litsearch.data import load
from copilot.litsearch.dense import DenseIndex
from copilot.litsearch.encoders import Qwen3Encoder
from copilot.litsearch.metrics import recall_at_k
from copilot.litsearch.split import split_questions

data = load()
_, test = split_questions([{"id": q.id, "query_set": q.query_set, "specificity": q.specificity} for q in data.questions], dev_size=200, seed=13)
index = DenseIndex(data.docs, Qwen3Encoder("Qwen/Qwen3-Embedding-8B", dtype="float16", max_length=1024))
dense = {q.id: index.search(q.text, 200) for q in data.questions}
cross = Qwen3Reranker("Qwen/Qwen3-Reranker-4B")
cross._load = lambda: (_ for _ in ()).throw(RuntimeError("cache miss"))  # must not compute anything new
reranked = {q.id: reorder(dense[q.id], cross.scores(q.text, [data.docs[d] for d in dense[q.id][:100]]), 100) for q in data.questions}
print("specificity counts", {s: sum(q.specificity == s for q in data.questions) for s in (0, 1)},
      "mean gold size", {s: st.mean(len(q.gold) for q in data.questions if q.specificity == s) for s in (0, 1)})
for split, ids in (("full", None), ("test", set(test))):
    for name, runs in (("dense 8B", dense), ("+ Qwen3-Reranker-4B", reranked)):
        row = []
        for src in (("inline_acl", "inline_nonacl"), ("manual_acl", "manual_iclr"), None):
            qs = [q for q in data.questions if (ids is None or q.id in ids) and (src is None or q.query_set in src)]
            b = [recall_at_k(runs[q.id], q.gold, 20) for q in qs if q.specificity == 0]
            s5 = [recall_at_k(runs[q.id], q.gold, 5) for q in qs if q.specificity == 1]
            s20 = [recall_at_k(runs[q.id], q.gold, 20) for q in qs if q.specificity == 1]
            row += [100 * st.mean(b), 100 * st.mean(s5), 100 * st.mean(s20)]
        print(f"{split:4} {name:22}", " ".join(f"{v:5.1f}" for v in row[:3]), "|", " ".join(f"{v:5.1f}" for v in row[3:6]), "| avg broad R@20 %.1f specific R@5 %.1f" % (row[6], row[7]))
