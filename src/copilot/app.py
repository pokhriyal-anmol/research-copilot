import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import psycopg
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from copilot.answer import PROMPT_VERSION, Answer, answer
from copilot.db import connect, migrate
from copilot.embed import embed_query
from copilot.llm import AllProvidersFailed
from copilot.retrieve import MODES, search

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
INDEX = STATIC / "index.html"
SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000",
    "Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
http_client = httpx.Client()
# Render's free instance runs out of memory with the reranker loaded, so it uses hybrid there.
SEARCH_MODE = os.environ.get("SEARCH_MODE", "rerank")
if SEARCH_MODE not in MODES:
    raise ValueError(f"SEARCH_MODE must be one of {MODES}")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@asynccontextmanager
async def lifespan(_: FastAPI):
    with connect() as conn:
        migrate(conn)
    yield


app = FastAPI(title="Research Copilot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next) -> Response:
    response = await call_next(request)
    response.headers.update(SECURITY_HEADERS)
    return response


def write_trace(conn: psycopg.Connection, question: str, result: Answer | None, started: float) -> None:
    c = result.completion if result else None
    withheld = result is not None and result.outcome == "citation_check_failed"
    conn.execute(
        """INSERT INTO traces (question, prompt_version, outcome, provider, model, input_tokens,
           output_tokens, latency_ms, model_output) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (question, PROMPT_VERSION, result.outcome if result else "provider_failure",
         c and c.provider, c and c.model, c and c.input_tokens, c and c.output_tokens,
         int((time.monotonic() - started) * 1000), c.text if withheld and c else None),
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
    # No database call: frequent checks would keep Neon awake and spend its free compute.
    return {"status": "ok"}


@app.post("/answer")
def post_answer(body: Question) -> dict:
    started = time.monotonic()
    with connect() as conn:
        if over_daily_cap(conn):
            raise HTTPException(503, "Today's free model quota is used up. Please try tomorrow.")
        try:
            result = answer(http_client, body.question, search(conn, body.question, embed_query(body.question), mode=SEARCH_MODE))
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
