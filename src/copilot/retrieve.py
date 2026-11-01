from dataclasses import dataclass

import numpy as np
import psycopg


@dataclass(frozen=True)
class Passage:
    arxiv_id: str
    title: str
    page: int
    text: str


def search(conn: psycopg.Connection, query_embedding: np.ndarray, k: int = 8) -> list[Passage]:
    rows = conn.execute(
        """SELECT c.arxiv_id, p.title, c.page, c.text
           FROM chunks c JOIN papers p USING (arxiv_id)
           ORDER BY c.embedding <=> %s LIMIT %s""",
        (query_embedding, k),
    ).fetchall()
    return [Passage(*row) for row in rows]
