"""Behaviour any first-stage retriever must have, whatever its internals."""
import numpy as np
import pytest

from copilot.litsearch.dense import DenseIndex

DOCS = {
    10: "Dense passage retrieval for open domain question answering",
    11: "BM25 and the probabilistic relevance framework",
    12: "Graph neural networks for molecule property prediction",
    13: "Retrieval augmented generation for knowledge intensive tasks",
}


def bm25_index():
    pytest.importorskip("bm25s")
    from copilot.litsearch.bm25 import BM25Index

    return BM25Index(DOCS)


def test_bm25_ranks_the_doc_matching_a_distinctive_title_first():
    assert bm25_index().search("graph neural networks molecule", k=4)[0] == 12


def test_bm25_returns_at_most_k_unique_doc_ids():
    hits = bm25_index().search("retrieval", k=2)
    assert len(hits) <= 2 and len(hits) == len(set(hits)) and set(hits) <= set(DOCS)


def test_bm25_is_deterministic():
    index = bm25_index()
    assert index.search("retrieval question answering", 4) == index.search("retrieval question answering", 4)


def test_bm25_returns_nothing_for_words_absent_from_the_corpus():
    assert bm25_index().search("zzzz qqqq", k=4) == []


def test_bm25_returns_only_docs_sharing_a_query_word():
    # Only docs 10 and 13 contain "retrieval"; zero-score docs must not pad the list.
    assert set(bm25_index().search("retrieval", k=4)) == {10, 13}


class FakeEncoder:
    """Maps each doc to a one-hot vector so the nearest doc is known in advance."""

    def __init__(self):
        self.calls = 0

    def encode_docs(self, texts):
        self.calls += 1
        return np.eye(len(texts), dtype=np.float32)

    def encode_query(self, text):
        # Halfway between docs 0 and 2, closer to 2.
        return np.array([0.3, 0.0, 0.9, 0.0], dtype=np.float32)


def test_dense_ranks_by_cosine_similarity():
    index = DenseIndex(DOCS, FakeEncoder())
    assert index.search("anything", k=2) == [12, 10]


class UnequalLengthEncoder:
    """Doc 10 points the wrong way but is long; doc 11 points the right way but is short."""

    def encode_docs(self, texts):
        return np.array([[10.0, 0.0], [1.0, 1.0], [0.0, 0.1]], dtype=np.float32)

    def encode_query(self, text):
        return np.array([1.0, 1.2], dtype=np.float32)


def test_dense_normalises_documents_so_length_does_not_win():
    # A raw dot product ranks doc 10 first (10.0 vs 2.2); cosine ranks doc 11 first.
    docs = {10: "a", 11: "b", 12: "c"}
    assert DenseIndex(docs, UnequalLengthEncoder()).search("q", k=1) == [11]


def test_dense_encodes_the_corpus_once_not_per_query():
    encoder = FakeEncoder()
    index = DenseIndex(DOCS, encoder)
    index.search("a", 2)
    index.search("b", 2)
    assert encoder.calls == 1


def test_dense_k_larger_than_corpus_returns_every_doc_once():
    hits = DenseIndex(DOCS, FakeEncoder()).search("a", k=50)
    assert sorted(hits) == sorted(DOCS)


def test_scored_search_returns_the_same_order_with_cosine_scores():
    index = DenseIndex(DOCS, FakeEncoder())
    scored = index.search_scored("anything", k=2)
    assert [d for d, _ in scored] == index.search("anything", k=2)
    assert scored[0][1] >= scored[1][1] and all(-1.0 <= s <= 1.0 for _, s in scored)
