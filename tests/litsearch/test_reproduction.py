"""The reproduction gate from bench/BENCHMARK.md, checked against saved run results.

A runner writes bench/results/<name>.json with the scores on all 597 questions. These tests
skip until that file exists, then hold it to the published figures. Never edit a target here
to make a run pass: change the code, or record a reason in the BENCHMARK.md change log.
"""
import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).parents[2] / "bench" / "results"
TOLERANCE = 2.0
# (result file, metric, published value, source)
TARGETS = [
    ("bm25-full.json", "ndcg@10", 40.2, "CASPER arXiv 2508.13394 Table 2"),
    ("qwen3-embedding-8b-full.json", "recall@20", 79.2, "PairSem arXiv 2510.09897 Table 1"),
    ("qwen3-embedding-8b-full.json", "ndcg@10", 58.0, "PairSem arXiv 2510.09897 Table 1"),
]


@pytest.mark.parametrize(("name", "metric", "published", "source"), TARGETS)
def test_baseline_reproduces_the_published_figure(name, metric, published, source):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not run yet")
    result = json.loads(path.read_text())
    assert result["questions"] == 597, "the reproduction must use all 597 questions"
    got = result["scores"][metric]
    assert abs(got - published) <= TOLERANCE, f"{metric} {got:.1f} vs {published} ({source})"
