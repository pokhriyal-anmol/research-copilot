"""Fixed stratified dev/test split of LitSearch questions (see bench/BENCHMARK.md)."""
import random
from collections import defaultdict


def split_questions(questions: list[dict], dev_size: int, seed: int) -> tuple[list[str], list[str]]:
    """Return (dev_ids, test_ids), each sorted. Dev seats per (query_set, specificity)
    stratum go by largest remainder of dev_size * n / total; ties break on stratum key."""
    strata: dict[tuple, list[str]] = defaultdict(list)
    for q in sorted(questions, key=lambda q: q["id"]):
        strata[(q["query_set"], q["specificity"])].append(q["id"])
    total = len(questions)
    keys = sorted(strata, key=str)
    seats = {k: dev_size * len(strata[k]) // total for k in keys}
    by_remainder = sorted(keys, key=lambda k: -(dev_size * len(strata[k]) % total))
    for k in by_remainder[: dev_size - sum(seats.values())]:
        seats[k] += 1
    rng = random.Random(seed)
    dev: set[str] = set()
    for k in keys:
        dev.update(rng.sample(strata[k], seats[k]))
    all_ids = sorted(i for ids in strata.values() for i in ids)
    return sorted(dev), [i for i in all_ids if i not in dev]
