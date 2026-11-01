from dataclasses import dataclass

import numpy as np
import psycopg

from copilot.embed import rerank_scores

CANDIDATES = 50  # per retriever, before fusion
RERANK_POOL = 30  # shortcut: larger pools rerank better but cost CPU per request
RRF_K = 60  # the constant from the original reciprocal rank fusion paper
MODES = ("vector", "keyword", "hybrid", "rerank")


@dataclass(frozen=True)
class Passage:
    arxiv_id: str
    title: str
    page: int
    text: str


COLUMNS = "c.arxiv_id, p.title, c.page, c.text"
FROM = "FROM chunks c JOIN papers p USING (arxiv_id)"


def vector_search(conn: psycopg.Connection, embedding: np.ndarray, k: int) -> list[Passage]:
    sql = f"SELECT {COLUMNS} {FROM} ORDER BY c.embedding <=> %s LIMIT %s"
    return [Passage(*row) for row in conn.execute(sql, (embedding, k)).fetchall()]


def keyword_search(conn: psycopg.Connection, question: str, k: int) -> list[Passage]:
    # OR the terms: plainto_tsquery ANDs them, which matches nothing for a full question.
    query = "to_tsquery('english', replace(plainto_tsquery('english', %s)::text, '&', '|'))"
    sql = f"""SELECT {COLUMNS} {FROM}, {query} q
              WHERE c.tsv @@ q ORDER BY ts_rank_cd(c.tsv, q) DESC LIMIT %s"""
    return [Passage(*row) for row in conn.execute(sql, (question, k)).fetchall()]


def fuse(rankings: list[list[Passage]], k: int) -> list[Passage]:
    scores: dict[Passage, float] = {}
    for ranking in rankings:
        for rank, passage in enumerate(ranking, 1):
            scores[passage] = scores.get(passage, 0.0) + 1 / (RRF_K + rank)
    return sorted(scores, key=lambda p: scores[p], reverse=True)[:k]


def rerank(question: str, passages: list[Passage], k: int) -> list[Passage]:
    if not passages:
        return []
    scores = rerank_scores(question, [p.text for p in passages])
    ranked = sorted(zip(scores, passages, strict=True), key=lambda pair: pair[0], reverse=True)
    return [p for _, p in ranked[:k]]


def search(
    conn: psycopg.Connection, question: str, embedding: np.ndarray, k: int = 8, mode: str = "rerank"
) -> list[Passage]:
    if mode == "vector":
        return vector_search(conn, embedding, k)
    if mode == "keyword":
        return keyword_search(conn, question, k)
    both = [vector_search(conn, embedding, CANDIDATES), keyword_search(conn, question, CANDIDATES)]
    if mode == "hybrid":
        return fuse(both, k)
    if mode == "rerank":
        return rerank(question, fuse(both, RERANK_POOL), k)
    raise ValueError(f"unknown mode {mode}")
