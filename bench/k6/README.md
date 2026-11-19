# k6 load test

[k6](https://k6.io) script for the API: `bench/k6/load.js`. Start the app against the local database, then:

```bash
DATABASE_URL=postgresql://copilot:copilot@localhost:5433/copilot uv run uvicorn copilot.app:app --port 8000
k6 run bench/k6/load.js            # BASE_URL=... to point elsewhere
```

It spends real Gemini/OpenRouter quota (8 answers plus up to 9 more in the burst), so do not run it against the deployed service.

## Run of 2026-11-19

k6 v2.4.0, one local uvicorn worker, local Docker Postgres (16,290 chunks), commit `3e99c2e`. Duration 2m19s, 91,777 requests (about 660/s), all checks passed. Figures are from k6's end-of-run summary; the raw summary file was not kept.

| Scenario | Load | Result | Threshold |
|---|---|---|---|
| `GET /health` | ramp to 50 users, 77,310 requests | p95 30.6 ms, max 111 ms, 0% failed | p95 < 200 ms: pass |
| `GET /` | 10 users, 35 s | all 200 | none |
| `POST /answer` | 8 sequential questions, real providers | all 200; median 6.2 s, p95 25.2 s, max 33.2 s | p95 < 40 s: pass |
| `POST /answer`, second IP | 15 requests | 9 answered, 5 got 429, 1 got 503 | none |

The 429 count matches the 10-per-IP-per-hour limit. The 503 was not investigated (daily cap or provider error; the log has no `provider failed` line).

Limits of this run: not the deployed Render/Neon service, whose latency will be higher; `/answer` was never called concurrently, because the per-IP limit stops that; one run, no repeat.
