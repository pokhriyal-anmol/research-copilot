"""Needs TEST_DATABASE_URL, a throwaway database: these tests truncate it."""
import os

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient

from copilot import app as app_module
from copilot.db import connect, migrate
from copilot.llm import AllProvidersFailed, Completion

pytestmark = pytest.mark.skipif("TEST_DATABASE_URL" not in os.environ, reason="no database")
VECTOR = np.ones(384)


# The citation check now reads the text, so the stub answer must be supported by the stored passage.
PASSAGE = "Dropout of 0.1 reduced overfitting on the small training set."
SUPPORTED_ANSWER = "Dropout reduced overfitting [1]."


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    with connect() as conn:
        migrate(conn)
        conn.execute("TRUNCATE papers, chunks, traces CASCADE")
        conn.execute("INSERT INTO papers VALUES ('1', 'T', 'A', 'cc')")
        conn.execute(
            "INSERT INTO chunks (arxiv_id, page, text, embedding) VALUES ('1', 4, %s, %s)",
            (PASSAGE, VECTOR),
        )
    monkeypatch.setattr(app_module, "embed_query", lambda _: VECTOR)
    app_module.recent_by_ip.clear()
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
    stub_llm(monkeypatch, SUPPORTED_ANSWER)
    body = client.post("/answer", json={"question": "what?"}).json()
    assert (body["citations"][0]["number"], body["citations"][0]["page"]) == (1, 4)
    assert traces() == [("answered", "groq", 5)]


def test_fabricated_citation_never_reaches_the_user(client, monkeypatch):
    stub_llm(monkeypatch, "Claim [7].")
    body = client.post("/answer", json={"question": "what?"}).json()
    assert body["outcome"] == "citation_check_failed" and "Claim" not in body["text"]
    assert model_outputs() == ["Claim [7]."]


def test_answered_trace_does_not_keep_model_output(client, monkeypatch):
    stub_llm(monkeypatch, SUPPORTED_ANSWER)
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
    stub_llm(monkeypatch, SUPPORTED_ANSWER)
    assert client.post("/answer", json={"question": "what?"}).status_code == 200
    assert client.post("/answer", json={"question": "what?"}).status_code == 503
    assert len(traces()) == 1


def test_per_ip_limit_stops_one_caller_but_not_another(client, monkeypatch):
    monkeypatch.setenv("ANSWERS_PER_IP_PER_HOUR", "1")
    stub_llm(monkeypatch, SUPPORTED_ANSWER)
    first, second = ({"cf-connecting-ip": ip} for ip in ("1.1.1.1", "2.2.2.2"))
    assert client.post("/answer", json={"question": "what?"}, headers=first).status_code == 200
    assert client.post("/answer", json={"question": "what?"}, headers=first).status_code == 429
    assert client.post("/answer", json={"question": "what?"}, headers=second).status_code == 200
    assert len(traces()) == 2


def test_oversized_question_is_rejected(client):
    assert client.post("/answer", json={"question": "x" * 501}).status_code == 422


def test_page_sends_security_headers_and_loads_its_script(client):
    page = client.get("/")
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-frame-options"] == "DENY"
    assert "<script>" not in page.text  # inline script would be blocked by the policy
    assert client.get("/static/app.js").status_code == 200


def test_a_citation_carries_its_passage_text(client, monkeypatch):
    # The web app opens each citation on the exact passage it rests on.
    stub_llm(monkeypatch, SUPPORTED_ANSWER)
    assert client.post("/answer", json={"question": "what?"}).json()["citations"][0]["passage"] == PASSAGE


@pytest.fixture
def web_origin_app(monkeypatch):
    import importlib

    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    monkeypatch.setenv("WEB_ORIGINS", "https://copilot.example")
    yield importlib.reload(app_module).app
    monkeypatch.delenv("WEB_ORIGINS")
    importlib.reload(app_module)


def test_the_configured_web_origin_may_call_the_api(web_origin_app):
    with TestClient(web_origin_app) as c:
        r = c.options("/answer", headers={"Origin": "https://copilot.example", "Access-Control-Request-Method": "POST",
                                          "Access-Control-Request-Headers": "content-type"})
    assert r.headers.get("access-control-allow-origin") == "https://copilot.example"


def test_any_other_origin_is_refused(web_origin_app):
    with TestClient(web_origin_app) as c:
        r = c.options("/answer", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers


def test_startup_deletes_questions_older_than_the_retention_period(client, monkeypatch):
    with connect() as conn:
        conn.execute(
            "INSERT INTO traces (question, prompt_version, outcome, latency_ms, created_at) "
            "VALUES ('old private question', 'v1', 'answered', 1, now() - interval '100 days'), "
            "('recent question', 'v1', 'answered', 1, now() - interval '1 day')"
        )
    with TestClient(app_module.app):  # startup runs the lifespan
        pass
    with connect() as conn:
        kept = [r[0] for r in conn.execute("SELECT question FROM traces ORDER BY id")]
    assert kept == ["recent question"]


def test_database_failure_after_startup_is_a_503_not_a_bare_500(client, monkeypatch, caplog):
    def refuse(*_):
        raise psycopg.OperationalError("connection refused")

    with TestClient(app_module.app, raise_server_exceptions=False) as quiet:  # started before the database goes away
        monkeypatch.setattr(app_module, "connect", refuse)
        response = quiet.post("/answer", json={"question": "what?"})
    assert response.status_code == 503 and "database" in response.json()["detail"].lower()
    assert "database" in caplog.text.lower()


def test_answer_works_with_the_hybrid_search_mode_render_runs(client, monkeypatch):
    monkeypatch.setattr(app_module, "SEARCH_MODE", "hybrid")
    stub_llm(monkeypatch, SUPPORTED_ANSWER)
    body = client.post("/answer", json={"question": "what reduced overfitting?"}).json()
    assert body["outcome"] == "answered" and body["citations"][0]["passage"] == PASSAGE


def test_unexpected_error_is_a_json_500_with_a_trace_and_no_leak(client, monkeypatch):
    def boom(_):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(app_module, "embed_query", boom)
    with TestClient(app_module.app, raise_server_exceptions=False) as quiet:
        response = quiet.post("/answer", json={"question": "what?"})
    assert response.status_code == 500
    assert isinstance(response.json()["detail"], str) and "secret" not in response.text
    assert traces() == [("internal_error", None, None)]
