import pytest

from copilot.answer import check
from copilot.retrieve import Passage

# Passage 1 is about BM25 recall. Passage 2 is about dropout.
PASSAGES = [
    Passage("1", "Retrieval", 4, "BM25 reached recall@10 of 0.62 on the benchmark, below dense retrieval at 0.71."),
    Passage("2", "Regularisation", 2, "Dropout of 0.1 reduced overfitting on the small training set."),
]


def test_sentence_citing_an_unrelated_passage_is_withheld():
    # In range, so the range check passes. Passage 2 says nothing about GPU memory.
    result = check(None, "The model needs 80 GB of GPU memory [2].", PASSAGES, check_support=True)
    assert result.outcome == "citation_check_failed"


@pytest.mark.xfail(strict=True, reason="xsmall NLI reads 'BM25 0.62 below dense 0.71' as entailing 'BM25 beat dense'")
def test_sentence_contradicting_its_passage_is_withheld():
    result = check(None, "BM25 beat dense retrieval on recall@10 [1].", PASSAGES, check_support=True)
    assert result.outcome == "citation_check_failed"


def test_sentence_supported_by_its_passage_is_answered():
    result = check(None, "Dropout of 0.1 reduced overfitting [2].", PASSAGES, check_support=True)
    assert result.outcome == "answered"
