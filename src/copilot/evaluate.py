"""Compare retrieval modes on labelled questions.

Usage: python -m copilot.evaluate [--include-drafts]
Hit means a labelled paper and page appears in the top K.
"""
import csv
import math
import sys
import time
from pathlib import Path

from copilot.db import connect
from copilot.embed import embed_query
from copilot.retrieve import MODES, search

QUESTIONS = Path("evals/questions.csv")
TOP_K = 20


def load_questions(include_drafts: bool) -> list[dict]:
    with QUESTIONS.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if include_drafts or r["reviewed"].strip().lower() == "yes"]


def margin(rate: float, n: int) -> float:
    return 1.96 * math.sqrt(rate * (1 - rate) / n) if n else 0.0


def score(hits: list[int | None]) -> tuple[float, float]:
    """Return recall@K and mean reciprocal rank from each question's first-hit rank."""
    found = [h for h in hits if h is not None]
    return len(found) / len(hits), sum(1 / h for h in found) / len(hits)


def first_hit(results: list, arxiv_id: str, page: int) -> int | None:
    for rank, passage in enumerate(results, 1):
        if (passage.arxiv_id, passage.page) == (arxiv_id, page):
            return rank
    return None


def run(include_drafts: bool) -> None:
    rows = load_questions(include_drafts)
    if not rows:
        sys.exit("No questions to score. Mark rows reviewed=yes, or pass --include-drafts.")
    conn = connect()
    embedded = [embed_query(r["question"]) for r in rows]
    print(f"{len(rows)} questions, drafts {'included' if include_drafts else 'excluded'}")
    print(f"| Mode | Recall@{TOP_K} | Margin | MRR | Mean ms |\n|---|---|---|---|---|")
    for mode in MODES:
        started = time.monotonic()
        hits = [
            first_hit(search(conn, r["question"], e, TOP_K, mode), r["arxiv_id"], int(r["page"]))
            for r, e in zip(rows, embedded, strict=True)
        ]
        ms = (time.monotonic() - started) * 1000 / len(rows)
        recall, mrr = score(hits)
        print(f"| {mode} | {recall:.3f} | ±{margin(recall, len(rows)):.3f} | {mrr:.3f} | {ms:.0f} |")


if __name__ == "__main__":
    run("--include-drafts" in sys.argv)
