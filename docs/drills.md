# Drills

Each operational check from the production bar, run once, with what was measured and what wasn't.

## Rollback, 2026-11-02

From `f43eddc` (security headers) back to `bf1dee4`, using Rollback in the Render dashboard. Target: under 5 minutes from click to a working app.

| Time (UTC) | Event | Source |
|---|---|---|
| about 10:56:05 | Rollback clicked | Operator's estimate: 4 to 5 seconds before the deploy started |
| 10:56:09 | Deploy starts | Render log |
| 10:56:27 | App startup complete | Render log |
| 10:56:31 | Render reports the service live | Render log |
| 10:59:01 | Old version confirmed serving: no security policy header, health 200 | Outside check |
| 10:59:11 | Cited answer returned, 5 citations, 9.3 s | Trace in Neon |

| 11:01:10 | Rolled forward to `f43eddc`: security headers back, answer returned | Outside check |

Click to live: about 26 seconds. Click to first verified answer: about 3 minutes 6 seconds.

Result: **pass, under the 5-minute target** on the conservative measure. The click time is an estimate, not a log entry. The timing script that should have caught the first working answer failed with a tool error, so the verified answer came 2.5 minutes after Render reported live. That gap was the checker's, not the app's, so the real recovery time is probably closer to 26 seconds, but only the 3-minute figure is proven.

A dashboard rollback turns off auto-deploy. It was turned back on afterwards, and the latest commit was deployed by hand.

Next time: start the outside checker before clicking, so it records the moment the old version starts answering.
