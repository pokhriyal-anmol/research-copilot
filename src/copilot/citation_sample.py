"""Answer the reviewed eval questions and sample cited sentences for a person to label as supported or not.

Run with the database up and only GEMINI_API_KEY set, so the OpenRouter fallback is not used:
  env -u OPENROUTER_API_KEY DATABASE_URL=postgresql://copilot:copilot@localhost:5433/copilot \
    uv run python -m copilot.citation_sample
Answers are saved as they arrive, so a run stopped by the daily cap resumes where it left off.
"""
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

from copilot.answer import CITATION, answer
from copilot.db import connect
from copilot.embed import embed_query
from copilot.evaluate import load_questions
from copilot import llm
from copilot.llm import AllProvidersFailed
from copilot.retrieve import search

ANSWERS = Path("evals/results/2026-11-03-answers.jsonl")
SHEET = Path("evals/results/2026-11-03-citations-to-label.csv")
SAMPLE_SIZE, SEED = 50, 13
TRIES, WAIT_SECONDS = 4, 30
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def citation_pairs(text: str) -> list[tuple[str, int]]:
    """(sentence, cited number) for every distinct number each sentence cites."""
    pairs = []
    for sentence in SENTENCE_END.split(text.strip()):
        numbers = {int(n) for group in CITATION.findall(sentence) for n in group.split(",")}
        pairs += [(sentence, n) for n in sorted(numbers)]
    return pairs


def sample(items: list, size: int) -> list:
    return random.Random(SEED).sample(items, min(size, len(items)))


def answer_all() -> None:
    llm.TIMEOUT_SECONDS = 120.0  # offline run: a slow free tier must not count as a failure; the app keeps 25 s
    done = {json.loads(line)["question"] for line in ANSWERS.open()} if ANSWERS.exists() else set()
    rows = [r for r in load_questions(include_drafts=False) if r["question"] not in done]
    with connect() as conn, httpx.Client() as client, ANSWERS.open("a") as out:
        for i, r in enumerate(rows, 1):
            passages = search(conn, r["question"], embed_query(r["question"]))
            for attempt in range(TRIES):  # free tiers time out and return 503 under load
                try:
                    result = answer(client, r["question"], passages)
                    break
                except AllProvidersFailed:
                    time.sleep(WAIT_SECONDS)
            else:
                sys.exit(f"stopped after {i - 1} new answers: every provider failed {TRIES} times. Rerun to resume.")
            cited = {n: {"title": p.title, "arxiv_id": p.arxiv_id, "page": p.page, "text": p.text}
                     for n, p in result.citations.items()}
            out.write(json.dumps({"question": r["question"], "outcome": result.outcome, "text": result.text,
                                  "citations": cited, "provider": result.completion and result.completion.provider})
                      + "\n")
            out.flush()
            print(f"{i}/{len(rows)} {result.outcome}", flush=True)


def write_sheet() -> None:
    answers = [json.loads(line) for line in ANSWERS.open()]
    print("outcomes", dict(Counter(a["outcome"] for a in answers)))
    pairs = [(a, s, n) for a in answers if a["outcome"] == "answered" for s, n in citation_pairs(a["text"])]
    with SHEET.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "label (supported / partly / not)", "question", "sentence", "cited paper", "page",
                    "url", "cited passage"])
        for i, (a, s, n) in enumerate(sample(pairs, SAMPLE_SIZE), 1):
            p = a["citations"][str(n)]
            w.writerow([i, "", a["question"], s, p["title"], p["page"], f"https://arxiv.org/abs/{p['arxiv_id']}",
                        p["text"]])
    print(f"{min(SAMPLE_SIZE, len(pairs))} of {len(pairs)} citations sampled into {SHEET}")


if __name__ == "__main__":
    answer_all()
    write_sheet()
