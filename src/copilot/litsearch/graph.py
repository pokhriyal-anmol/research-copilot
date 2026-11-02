"""Citation-graph candidate pools for LitSearch."""


def one_hop(ranking: list[int], citations: dict[int, list[int]], m: int = 50, n: int = 200) -> list[int]:
    """LitSearch's one-hop pool: each of the top m papers, then the papers it cites, without repeats, cut at n."""
    pool: dict[int, None] = {}  # insertion-ordered set
    for seed in ranking[:m]:
        for paper in (seed, *citations.get(seed, ())):
            pool.setdefault(paper)
    return list(pool)[:n]
