"""Ask the deployed app fixed questions and fail if too few answerable ones are answered
or any unanswerable one is not refused. Stdlib only. Run: python evals/live_eval.py [base_url]
Uses 8 requests; the app allows ANSWERS_PER_IP_PER_HOUR (default 10) per IP."""
import json
import sys
import urllib.request
from pathlib import Path

QUESTIONS = Path(__file__).with_name("live-questions.json")
MIN_ANSWERED_SHARE = 0.75  # 2026-11-03 run: 42 identifiable questions, 94 of 120 answered overall


def ask(url: str, question: str) -> str:
    request = urllib.request.Request(
        f"{url}/answer",
        data=json.dumps({"question": f"[synthetic] {question}"}).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=150) as response:
        return json.load(response)["outcome"]


def evaluate(url: str, questions: dict[str, list[str]]) -> list[str]:
    """Return failure messages; an empty list means pass."""
    answered = [ask(url, q) for q in questions["answerable"]]
    refused = [ask(url, q) for q in questions["unanswerable"]]
    failures = []
    share = answered.count("answered") / len(answered)
    if share < MIN_ANSWERED_SHARE:
        failures.append(f"answerable questions answered: {share:.0%}, need {MIN_ANSWERED_SHARE:.0%} ({answered})")
    if refused != ["refused"] * len(refused):
        failures.append(f"unanswerable questions should be refused, got {refused}")
    return failures


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "https://research-copilot-au2k.onrender.com"
    problems = evaluate(base, json.loads(QUESTIONS.read_text()))
    print("\n".join(problems) or "live eval passed")
    sys.exit(1 if problems else 0)
