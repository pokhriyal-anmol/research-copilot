from copilot.draft_questions import PROMPT, clean, is_usable


def test_a_reasoning_block_is_stripped_from_the_question():
    assert clean("<think>\n</think>What does X measure?") == "What does X measure?"
    assert clean("<think>long\nreasoning</think>\n What does X measure? ") == "What does X measure?"


def test_a_question_without_a_reasoning_block_is_unchanged():
    assert clean(" What does X measure? ") == "What does X measure?"


def test_a_leftover_tag_is_never_usable():
    assert not is_usable("</think>What does X measure?")


def test_the_prompt_asks_for_a_question_that_names_its_paper():
    # Questions written without the paper's method or system name had no single answer across 299 papers.
    assert "name" in PROMPT and "this paper" not in PROMPT
    # The first fix was read as "ask for the name"; the question has to carry the name instead.
    assert "must contain the passage's own name" in PROMPT and "Never ask what" in PROMPT


def test_drafting_never_overwrites_an_existing_question_file(tmp_path):
    # The reviewed set every published number rests on lived in the default output file.
    import pytest

    from copilot.draft_questions import run

    existing = tmp_path / "questions.csv"
    existing.write_text("reviewed,question\nyes,kept?\n")
    with pytest.raises(SystemExit, match="already exists"):
        run(5, existing)
    assert existing.read_text() == "reviewed,question\nyes,kept?\n"


def test_continuing_refuses_a_file_with_reviewed_questions(tmp_path):
    import pytest

    from copilot.draft_questions import run

    reviewed = tmp_path / "q.csv"
    reviewed.write_text("reviewed,question,arxiv_id,page,passage\nyes,kept?,1,2,x\n")
    with pytest.raises(SystemExit, match="reviewed"):
        run(5, reviewed, resume=True)
    assert reviewed.read_text() == "reviewed,question,arxiv_id,page,passage\nyes,kept?,1,2,x\n"


def test_continuing_skips_papers_that_already_have_a_draft():
    from copilot.draft_questions import pending

    rows = [("a", 2, "x"), ("b", 3, "y"), ("c", 4, "z")]
    assert pending(rows, {"b"}) == [("a", 2, "x"), ("c", 4, "z")]


def test_continued_rows_follow_the_files_own_columns(tmp_path, monkeypatch):
    # A review column added between runs must not shift appended drafts into the wrong fields.
    import csv

    from copilot import draft_questions as dq

    out = tmp_path / "q.csv"
    out.write_text("reviewed,prescreen,question,arxiv_id,page,passage\nno,ok,\"In A, what?\",a,2,x\n")
    monkeypatch.setattr(dq, "connect", lambda: type("C", (), {"execute": lambda *_: type(
        "R", (), {"fetchall": lambda _: [("a", 2, "x"), ("b", 3, "y")]})()})())
    monkeypatch.setattr(dq, "complete", lambda *_: type("T", (), {"text": "In B, why?"})())
    monkeypatch.setattr(dq, "GAP_SECONDS", 0)
    dq.run(2, out, resume=True)
    rows = list(csv.DictReader(out.open()))
    assert [(r["question"], r["arxiv_id"], r["prescreen"]) for r in rows] == [("In A, what?", "a", "ok"), ("In B, why?", "b", "")]
