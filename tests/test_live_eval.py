import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))
import live_eval

QUESTIONS = {"answerable": ["a1", "a2", "a3", "a4"], "unanswerable": ["u1"]}


def serve(outcome_for):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            question = json.loads(self.rfile.read(int(self.headers["content-length"])))["question"]
            body = json.dumps({"outcome": outcome_for(question)}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def run(outcome_for):
    server, url = serve(outcome_for)
    try:
        return live_eval.evaluate(url, QUESTIONS)
    finally:
        server.shutdown()


def test_healthy_app_passes():
    assert run(lambda q: "refused" if "u1" in q else "answered") == []


def test_app_that_refuses_everything_fails():
    failures = run(lambda q: "refused")
    assert len(failures) == 1 and "answerable questions answered: 0%" in failures[0]


def test_app_that_answers_everything_fails_on_the_unanswerable_question():
    failures = run(lambda q: "answered")
    assert len(failures) == 1 and "should be refused" in failures[0]


def test_withheld_citations_count_as_not_answered():
    assert run(lambda q: "citation_check_failed")
