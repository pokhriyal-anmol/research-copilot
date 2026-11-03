from copilot.evaluate import first_hit, margin, score
from copilot.retrieve import Passage

RESULTS = [Passage("a", "t", 1, "x"), Passage("b", "t", 7, "y")]


def test_first_hit_needs_paper_and_page():
    assert first_hit(RESULTS, "b", 7) == 2
    assert first_hit(RESULTS, "b", 1) is None


def test_score_counts_misses():
    assert score([1, None, 2, None]) == (0.5, 0.375)


def test_margin_matches_plan_figure():
    assert round(margin(0.9, 100), 2) == 0.06


def test_draft_filter_rejects_skip_and_non_questions():
    from copilot.draft_questions import is_usable

    assert is_usable("How is recall measured?")
    assert not is_usable("SKIP")
    assert not is_usable("Recall is measured.")


def test_questions_can_come_from_another_file(tmp_path):
    from copilot.evaluate import load_questions

    other = tmp_path / "q.csv"
    other.write_text("reviewed,question\nyes,a?\nno,b?\n")
    assert [r["question"] for r in load_questions(False, other)] == ["a?"]
