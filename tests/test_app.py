"""Needs TEST_DATABASE_URL, a throwaway database: these tests truncate it."""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from copilot import app as app_module
from copilot.db import connect, migrate
from copilot.llm import AllProvidersFailed, Completion

pytestmark = pytest.mark.skipif("TEST_DATABASE_URL" not in os.environ, reason="no database")
VECTOR = np.ones(384)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    with connect() as conn:
        migrate(conn)
        conn.execute("TRUNCATE papers, chunks, traces CASCADE")
        conn.execute("INSERT INTO papers VALUES ('1', 'T', 'A', 'cc')")
        conn.execute(
            "INSERT INTO chunks (arxiv_id, page, text, embedding) VALUES ('1', 4, 'x', %s)",
            (VECTOR,),
        )
    monkeypatch.setattr(app_module, "embed_query", lambda _: VECTOR)
    with TestClient(app_module.app) as test_client:
        yield test_client


def stub_llm(monkeypatch, text):
    completion = Completion(text, "groq", "m", 5, 2)
    monkeypatch.setattr("copilot.answer.complete", lambda *_: completion)


def traces():
    with connect() as conn:
        return conn.execute("SELECT outcome, provider, input_tokens FROM traces").fetchall()


def model_outputs():
    with connect() as conn:
        return [r[0] for r in conn.execute("SELECT model_output FROM traces ORDER BY id")]


def test_answer_returns_page_citation_and_writes_trace(client, monkeypatch):
    stub_llm(monkeypatch, "Claim [1].")
    body = client.post("/answer", json={"question": "what?"}).json()
    assert (body["citations"][0]["number"], body["citations"][0]["page"]) == (1, 4)
    assert traces() == [("answered", "groq", 5)]


def test_fabricated_citation_never_reaches_the_user(client, monkeypatch):
    stub_llm(monkeypatch, "Claim [7].")
    body = client.post("/answer", json={"question": "what?"}).json()
    assert body["outcome"] == "citation_check_failed" and "Claim" not in body["text"]
    assert model_outputs() == ["Claim [7]."]


def test_answered_trace_does_not_keep_model_output(client, monkeypatch):
    stub_llm(monkeypatch, "Claim [1].")
    client.post("/answer", json={"question": "what?"})
    assert model_outputs() == [None]


def test_provider_failure_is_503_and_traced(client, monkeypatch):
    def fail(*_):
        raise AllProvidersFailed

    monkeypatch.setattr("copilot.answer.complete", fail)
    assert client.post("/answer", json={"question": "what?"}).status_code == 503
    assert traces()[0][0] == "provider_failure"


def test_daily_cap_stops_model_calls(client, monkeypatch):
    monkeypatch.setenv("DAILY_ANSWER_CAP", "1")
    stub_llm(monkeypatch, "Claim [1].")
    assert client.post("/answer", json={"question": "what?"}).status_code == 200
    assert client.post("/answer", json={"question": "what?"}).status_code == 503
    assert len(traces()) == 1


def test_oversized_question_is_rejected(client):
    assert client.post("/answer", json={"question": "x" * 501}).status_code == 422


def test_page_sends_security_headers_and_loads_its_script(client):
    page = client.get("/")
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-frame-options"] == "DENY"
    assert "<script>" not in page.text  # inline script would be blocked by the policy
    assert client.get("/static/app.js").status_code == 200
