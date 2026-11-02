"""The runner turns a search function into the result file the reproduction gate reads."""
import json

from copilot.litsearch.data import Data, Question
from copilot.litsearch.run import evaluate, save

QUESTIONS = [
    Question(id="q1", text="first", gold={1}, query_set="manual_acl", specificity=1),
    Question(id="q2", text="second", gold={2}, query_set="inline_acl", specificity=0),
]
DATA = Data(
    questions=QUESTIONS, docs={1: "A\n\nx", 2: "B", 3: "C\n\ny"}, citations={1: [], 2: [], 3: []},
    no_abstract={2}, missing_gold=set(),
)


def search(text, k):
    return [1, 3, 2] if text == "first" else [3, 1]


def test_evaluate_scores_only_the_requested_questions():
    result = evaluate(DATA, search, question_ids=["q1"], name="toy")
    assert result["questions"] == 1
    assert result["scores"]["recall@20"] == 100.0


def test_evaluate_reports_the_headline_metrics_and_recall_at_100():
    scores = evaluate(DATA, search, ["q1", "q2"], name="toy")["scores"]
    assert {"recall@20", "ndcg@10", "recall@100"} <= set(scores)
    assert scores["recall@20"] == 50.0


def test_evaluate_breaks_down_by_specificity_query_set_and_missing_abstract():
    breakdowns = evaluate(DATA, search, ["q1", "q2"], name="toy")["breakdowns"]
    assert breakdowns["specificity=1"]["recall@20"] == 100.0
    assert breakdowns["specificity=0"]["recall@20"] == 0.0
    assert breakdowns["query_set=inline_acl"]["questions"] == 1
    # q2's gold paper has no abstract.
    assert breakdowns["gold_without_abstract"]["questions"] == 1


def test_evaluate_asks_for_enough_results_to_score_recall_at_100():
    asked = []
    evaluate(DATA, lambda text, k: asked.append(k) or [], ["q1"], name="toy")
    assert asked and min(asked) >= 100


def test_save_writes_json_with_name_settings_and_date(tmp_path):
    result = evaluate(DATA, search, ["q1"], name="toy", settings={"model": "fake"})
    path = save(result, tmp_path)
    saved = json.loads(path.read_text())
    assert path.name == "toy.json"
    assert saved["settings"] == {"model": "fake"} and saved["date"]
    assert saved["scores"] == result["scores"]
