"""Usage: python -m copilot.ingest <from-date> <until-date> <max-papers>"""
import logging
import sys
import time

import httpx
import psycopg
from pypdf.errors import PyPdfError

from copilot.arxiv import Paper, chunk_pdf, parse_records
from copilot.db import connect, migrate
from copilot.embed import embed_passages

log = logging.getLogger(__name__)
OAI_URL = "https://oaipmh.arxiv.org/oai"
PDF_URL = "https://export.arxiv.org/pdf/{}"
REQUEST_GAP_SECONDS = 3  # arXiv's published limit: one request every three seconds
USER_AGENT = {"User-Agent": "research-copilot/0.1 (github.com/pokhriyal-anmol)"}


def store(conn: psycopg.Connection, paper: Paper, chunks: list[tuple[int, str]]) -> None:
    embeddings = embed_passages([text for _, text in chunks])
    with conn.transaction():
        conn.execute(
            "INSERT INTO papers VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (paper.arxiv_id, paper.title, paper.authors, paper.license),
        )
        conn.execute("DELETE FROM chunks WHERE arxiv_id = %s", (paper.arxiv_id,))
        for (page, text), embedding in zip(chunks, embeddings, strict=True):
            conn.execute(
                "INSERT INTO chunks (arxiv_id, page, text, embedding) VALUES (%s, %s, %s, %s)",
                (paper.arxiv_id, page, text, embedding),
            )


def run(start: str, until: str, limit: int) -> None:
    conn = connect()
    migrate(conn)
    params = {"verb": "ListRecords", "metadataPrefix": "arXivRaw", "from": start,
              "until": until, "set": "cs:cs:IR"}
    with httpx.Client(headers=USER_AGENT, timeout=60, follow_redirects=True) as client:
        records = client.get(OAI_URL, params=params)
        records.raise_for_status()
        for paper in parse_records(records.content)[:limit]:
            time.sleep(REQUEST_GAP_SECONDS)
            try:
                pdf = client.get(PDF_URL.format(paper.arxiv_id))
                pdf.raise_for_status()
                store(conn, paper, chunk_pdf(pdf.content))
                log.info("stored %s", paper.arxiv_id)
            except (httpx.HTTPError, PyPdfError) as exc:
                log.warning("skipped %s: %s", paper.arxiv_id, type(exc).__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
