import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from copilot.answer import PROMPT_VERSION, Answer, answer
from copilot.db import connect, migrate
from copilot.embed import embed_query
from copilot.llm import AllProvidersFailed
from copilot.retrieve import search

log = logging.getLogger(__name__)
INDEX = Path(__file__).parent / "static" / "index.html"
http_client = httpx.Client()


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@asynccontextmanager
async def lifespan(_: FastAPI):
    with connect() as conn:
        migrate(conn)
    yield


app = FastAPI(title="Research Copilot", lifespan=lifespan)


def write_trace(conn: psycopg.Connection, question: str, result: Answer | None, started: float) -> None:
    c = result.completion if result else None
    conn.execute(
        """INSERT INTO traces (question, prompt_version, outcome, provider, model,
           input_tokens, output_tokens, latency_ms) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (question, PROMPT_VERSION, result.outcome if result else "provider_failure",
         c and c.provider, c and c.model, c and c.input_tokens, c and c.output_tokens,
         int((time.monotonic() - started) * 1000)),
    )


def over_daily_cap(conn: psycopg.Connection) -> bool:
    used = conn.execute(
        "SELECT count(*) FROM traces WHERE created_at > now() - interval '1 day'"
    ).fetchone()[0]
    return used >= int(os.environ.get("DAILY_ANSWER_CAP", "200"))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX)


@app.get("/health")
def health() -> dict:
    with connect() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@app.post("/answer")
def post_answer(body: Question) -> dict:
    started = time.monotonic()
    with connect() as conn:
        if over_daily_cap(conn):
            raise HTTPException(503, "Today's free model quota is used up. Please try tomorrow.")
        try:
            result = answer(http_client, body.question, search(conn, embed_query(body.question)))
        except AllProvidersFailed:
            write_trace(conn, body.question, None, started)
            raise HTTPException(503, "The model providers are unavailable. Please retry.") from None
        write_trace(conn, body.question, result, started)
    return {
        "outcome": result.outcome,
        "text": result.text,
        "citations": [
            {"number": n, "arxiv_id": p.arxiv_id, "title": p.title, "page": p.page,
             "url": f"https://arxiv.org/abs/{p.arxiv_id}"}
            for n, p in result.citations.items()
        ],
    }
