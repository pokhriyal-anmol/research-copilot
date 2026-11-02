"""Citation-graph candidate pools for LitSearch."""


def one_hop(ranking: list[int], citations: dict[int, list[int]], m: int = 50, n: int = 200) -> list[int]:
    """LitSearch's one-hop pool: each of the top m papers, then the papers it cites, without repeats, cut at n."""
    pool: dict[int, None] = {}  # insertion-ordered set
    for seed in ranking[:m]:
        for paper in (seed, *citations.get(seed, ())):
            pool.setdefault(paper)
    return list(pool)[:n]


def pagerank(papers: list[int], citations: dict[int, list[int]], damping: float = 0.85,
             iterations: int = 200) -> dict[int, float]:
    """PageRank over the citation links among `papers` only; papers citing none of them spread uniformly."""
    import numpy as np

    index = {p: i for i, p in enumerate(papers)}
    n = len(papers)
    rank = np.full(n, 1 / n)
    outs = [[index[q] for q in dict.fromkeys(citations.get(p, ())) if q in index] for p in papers]
    for _ in range(iterations):
        new = np.full(n, (1 - damping) / n)
        for j, out in enumerate(outs):
            if out:
                np.add.at(new, out, damping * rank[j] / len(out))
            else:
                new += damping * rank[j] / n
        done = np.abs(new - rank).sum() < 1e-12
        rank = new
        if done:
            break
    return dict(zip(papers, rank.tolist(), strict=True))


def structural_rerank(ranking: list[int], citations: dict[int, list[int]], rrf_k: int = 60) -> list[int]:
    """Kurland and Lee's structural re-ranking: fuse the dense order with PageRank over the candidates' citations."""
    dense = {p: r for r, p in enumerate(ranking, 1)}
    scores = pagerank(ranking, citations)
    by_pagerank = sorted(ranking, key=lambda p: (-scores[p], dense[p]))  # ties go to the better dense rank
    central = {p: r for r, p in enumerate(by_pagerank, 1)}
    fused = {p: 1 / (rrf_k + dense[p]) + 1 / (rrf_k + central[p]) for p in ranking}
    return sorted(ranking, key=lambda p: (-fused[p], dense[p]))
