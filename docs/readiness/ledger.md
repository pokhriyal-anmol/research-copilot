# Readiness ledger

Run by the `readiness-loop` skill. One row per gap. Pinned commit: `origin/main @ 7358f5d` (seeded 2026-11-17).

Seeded from `~/projects/ai-job-search/documents/projects/production-readiness-2026-11.md`. That audit scored the local `docs/readme` branch, 5 commits behind `origin/main`, so every row below marked "re-verify" must be checked on `origin/main` before work starts. Its gap 4 (no per-IP limit) was wrong: `origin/main` has `ANSWERS_PER_IP_PER_HOUR` (PR #7) and is dropped here.

## Open decisions for the user

- (none)

## Gaps

| ID | Priority | Failure (what goes wrong for a user or operator) | Failing check (test, drill or input) | Status | Tool chosen and why | Glue lines | Evidence |
|---|---|---|---|---|---|---|---|
| RC1 | 1 | The model cites passage [3], which was sent to it but says something else; the answer is shown as checked. The runtime check in `src/copilot/answer.py` `check()` only verifies cited numbers are in range, so an unsupported sentence passes. The GitHub description and `CLAUDE.md` say "citations checked in code", which reads as more than that. | Planted completion whose sentence cites an in-range passage that contradicts it; today `check()` returns `answered`. | closed 2026-11-17 (partial: unrelated citations caught, numeric contradiction not) | `cross-encoder/nli-deberta-v3-xsmall` quantised ONNX, run with onnxruntime and tokenizers already installed via fastembed. Base model (244 MB) also missed the contradiction, so it was dropped. | `src/copilot/support.py` 34 lines, plus 5 lines in `answer.py` | `tests/test_citation_support.py`: unrelated-citation test failed before (`answered`), passes now; with the check disabled it fails again on `assert 'answered' == 'citation_check_failed'`. Contradiction test is `xfail(strict)`. Full suite 195 passed. |
| RC2 | 2 | A prompt or model change lowers answer quality and ships anyway; nothing in CI scores answers. | Re-verify on `origin/main`: list CI jobs; plant a prompt change that breaks citations and see whether CI fails. | closed 2026-11-17 (partial: rule deletions caught at PR time; live answer quality still unscored in CI) | Plain pytest, no new dependency. promptfoo, inspect-ai and deepeval add 30+ dependencies each, and replay caches miss when the prompt changes. | 9 lines in `tests/test_answer.py` | Planted failure on origin/main: deleting the citation rule passed ruff and 195 tests. Now `test_live_prompt_keeps_every_rule_the_checks_rely_on` fails on it (`live prompt lost: ['Cite every claim ...']`). Suite 196 passed. Live eval added 2026-11-17 (see `.github/workflows/live-eval.yml`, `evals/live_eval.py`): daily, no secret, scores the deployed app on 6 answerable and 2 unanswerable fixed questions. Tests in `tests/test_live_eval.py` fail for an app that refuses everything and for one that answers everything. Not yet run from GitHub (workflow_dispatch needs the file on main). First live run found that 'capital of France' is answered legitimately from a corpus passage, so it was replaced. |
| RC3 | 3 | The live demo (Vercel) or backend (Render) is down, or the withheld-answer rate spikes, and nobody finds out until a recruiter does. Audit scored Monitoring 0 of 5. | Drill: stop the backend or point it at a bad model URL; does anything alert? | closed 2026-11-17 (partial: audit's 'no monitoring' was wrong; Vercel page was the real hole) | Existing `synthetic.yml` plus a 1-line curl step; no tool. | 1 job, 6 lines of YAML | Re-verified on origin/main: `synthetic.yml` already probed backend `/health` every 5 min and one full answer hourly, and a failed run exists (2026-11-06). Not probed: the Vercel page. Probe command run 2026-11-17: live page exit 0, 404 host exit 1, wrong page (example.com) exit 1. Verified 2026-11-17 in the owner's inbox: the 2026-11-06 failure sent 'Run failed: Synthetic checks - main (b583c87)' from notifications@github.com. `/health` skips the database by design, so a database outage shows only in the hourly answer check. Not run from GitHub yet. |
| RC4 | 4 | Remaining audit findings on Data, Model and Infrastructure were scored on the stale branch. | Re-ran those rows on `origin/main` plus RC1 to RC3 branches. | closed 2026-11-17 | Reading the code; no tool. | 0 | One row survived (RC5). Dropped or refuted: per-IP limit (exists, PR 7); 'exclude synthetic calls from the daily cap' REFUTED as a fix, since a caller could prefix `[synthetic]` to bypass the cap (synthetic and live-eval calls use about 32 of 200 a day, accepted); Infra 5 replay (retrieval is deterministic per `evals/results/2026-11-04-refusals.md`); Model 4 and Monitor 4 staleness, Infra 6 canary, Infra 7 automated rollback, Model 1 review (n/a for a static corpus and a solo repo; rollback was drilled). Not tested, no failing input found: Data 1 value checks, Data 3, Model 3, Model 6 (bench-only, no user-facing failure). |
| RC5 | 5 | User questions are stored forever (`traces.question`), with no retention rule and no notice, so free text a visitor types, including anything personal, stays in the database indefinitely. | `tests/test_app.py::test_startup_deletes_questions_older_than_the_retention_period`: a 100-day-old trace survived startup before the fix. | closed 2026-11-17 | Plain SQL delete at startup, `TRACE_RETENTION_DAYS` default 30 (user chose 30 on 2026-11-17). | 3 lines in `app.py`, 1 sentence in README | Failed before, passes after; with the delete mutated to a SELECT it fails again on `assert ['old private...ent question'] == ['recent question']`. Suite 214 passed with the database up. Takes effect on the live database at the next deploy. |

Status is one of: open, in-progress, closed, not-a-gap, not-applicable, wont-fix (with reason).

## Fault drills (used by the held-out grade)

| Drill | How to run it | Expected behaviour | Last result (date, SHA) |
|---|---|---|---|
| Unsupported citation | Planted completion, as in RC1 | Withheld or flagged, never shown as checked | Unrelated: pass. Numeric contradiction: FAIL (known, xfail). 2026-11-17 |
| Backend down | Stop the backend service or block its URL | Frontend shows the wake or error state; an alert reaches the owner | |
| Model provider error | Point the model client at an invalid endpoint | Structured error to the user, logged with request ID, no stack trace | |
| Database unreachable | Wrong `DATABASE_URL` in a test environment | Health check fails; answers refuse cleanly | |

## Held-out grades

| Date | SHA | OpenSSF Scorecard | Blind re-audit summary | Drills passed |
|---|---|---|---|---|
