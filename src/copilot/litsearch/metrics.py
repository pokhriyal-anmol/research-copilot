"""Recall@k and binary nDCG@k, as defined in bench/BENCHMARK.md. Stdlib only."""
import math
from collections.abc import Collection, Hashable, Mapping, Sequence


def _top_k(ranked: Sequence[Hashable], gold: Collection[Hashable], k: int) -> list[Hashable]:
    if not gold:
        raise ValueError("gold set is empty")
    return list(dict.fromkeys(ranked))[:k]  # dedupe keeping first occurrence, then cut


def recall_at_k(ranked: Sequence[Hashable], gold: Collection[Hashable], k: int) -> float:
    top = _top_k(ranked, gold, k)
    return len(set(top) & set(gold)) / len(set(gold))


def ndcg_at_k(ranked: Sequence[Hashable], gold: Collection[Hashable], k: int) -> float:
    top = _top_k(ranked, gold, k)
    dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(top) if p in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(set(gold)), k)))
    return dcg / ideal


def mean_scores(
    runs: Mapping[str, Sequence[Hashable]],
    gold: Mapping[str, Collection[Hashable]],
    recall_ks: Sequence[int],
    ndcg_ks: Sequence[int],
) -> dict[str, float]:
    """Mean over questions in gold, as percentages. A question with no run scores 0."""
    metrics = [(f"recall@{k}", recall_at_k, k) for k in recall_ks]
    metrics += [(f"ndcg@{k}", ndcg_at_k, k) for k in ndcg_ks]
    return {
        name: 100 * sum(fn(runs.get(q, []), g, k) for q, g in gold.items()) / len(gold)
        for name, fn, k in metrics
    }
