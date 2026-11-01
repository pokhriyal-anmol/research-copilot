from copilot.answer import REFUSAL_TOKEN, check, cited_numbers
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
