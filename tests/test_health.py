from fastapi.testclient import TestClient

from copilot import app as app_module


def test_health_answers_head_as_well_as_get():
    # Uptime monitors probe with HEAD; it returned 405 and flapped the monitor (2026-11-17).
    client = TestClient(app_module.app)  # no lifespan: /health must not need the database
    assert client.get("/health").json() == {"status": "ok"}
    assert client.head("/health").status_code == 200
