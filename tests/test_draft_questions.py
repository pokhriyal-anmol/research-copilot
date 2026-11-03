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
