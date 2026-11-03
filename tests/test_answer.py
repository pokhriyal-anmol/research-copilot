from copilot.answer import (
    PROMPT_VERSION,
    REFUSAL_TOKEN,
    SYSTEM_PROMPTS,
    build_messages,
    check,
    cited_numbers,
)
from copilot.retrieve import Passage

PASSAGES = [Passage("1", "A", 1, "x"), Passage("2", "B", 3, "y")]


def test_cited_numbers_reads_single_and_grouped():
    assert cited_numbers("a [1] b [2, 3]") == {1, 2, 3}


def test_valid_citations_return_only_cited_passages():
    result = check(None, "Claim [2].", PASSAGES)
    assert result.outcome == "answered" and result.citations == {2: PASSAGES[1]}


def test_fabricated_citation_is_withheld():
    result = check(None, "Claim [1] and [9].", PASSAGES)
    assert result.outcome == "citation_check_failed" and "[9]" not in result.text


def test_zero_is_not_a_valid_citation():
    assert check(None, "Claim [0].", PASSAGES).outcome == "citation_check_failed"


def test_uncited_answer_is_withheld():
    assert check(None, "Claim with no source.", PASSAGES).outcome == "citation_check_failed"


def test_refusal_token_refuses():
    assert check(None, REFUSAL_TOKEN, PASSAGES).outcome == "refused"


# Prompt v2 (evals/results/2026-11-04-vague-questions.md) failed its rule; v1 stays live.


def test_v1_is_the_live_prompt():
    assert PROMPT_VERSION == "v1" and build_messages("q", PASSAGES)[0]["content"] == SYSTEM_PROMPTS["v1"]


def test_v2_asks_for_each_papers_answer_when_the_question_names_no_paper():
    assert "each paper" in SYSTEM_PROMPTS["v2"] and "naming the paper" in SYSTEM_PROMPTS["v2"]


def test_v2_keeps_every_v1_rule():
    # Citations, refusal and the injection guard must survive the change.
    for line in SYSTEM_PROMPTS["v1"].splitlines():
        assert line in SYSTEM_PROMPTS["v2"]


def test_the_dropped_v2_can_still_be_run():
    assert build_messages("q", PASSAGES, version="v2")[0]["content"] == SYSTEM_PROMPTS["v2"]
