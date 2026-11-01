# Research Copilot

Answers questions about CC BY and CC0 arXiv papers, with citations checked in code.

- Setup: `uv sync`, `docker compose up -d`, copy `.env.example` to `.env`
- Tests: `TEST_DATABASE_URL=postgresql://copilot:copilot@localhost:5433/copilot_test uv run pytest` (truncates that database)
- Ingest: `DATABASE_URL=postgresql://copilot:copilot@localhost:5433/copilot uv run python -m copilot.ingest 2026-10-01 2026-10-02 20`

Rules: only store papers whose licence is in `ALLOWED_LICENSES`. Keep arXiv requests 3 seconds apart. Link citations back to arxiv.org.
