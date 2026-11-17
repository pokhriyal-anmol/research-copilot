import os

import numpy as np
import pytest

from copilot import retrieve
from copilot.db import connect, migrate
from copilot.retrieve import Passage, fuse, rerank, search

A, B, C = (Passage(x, "t", 1, x) for x in "abc")


def test_fuse_prefers_passage_ranked_by_both():
    assert fuse([[A, B], [C, B]], 3)[0] == B


def test_fuse_respects_k():
    assert len(fuse([[A, B, C]], 2)) == 2


def test_rerank_orders_by_score(monkeypatch):
    monkeypatch.setattr(retrieve, "rerank_scores", lambda _, texts: [ord(t) for t in texts])
    assert rerank("q", [A, B, C], 2) == [C, B]


def test_rerank_handles_no_candidates():
    assert rerank("q", [], 5) == []


@pytest.mark.skipif("TEST_DATABASE_URL" not in os.environ, reason="no database")
def test_keyword_search_matches_any_term_and_survives_stopwords(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    with connect() as conn:
        migrate(conn)
        conn.execute("TRUNCATE papers, chunks, traces CASCADE")
        conn.execute("INSERT INTO papers VALUES ('1', 'T', 'A', 'cc')")
        conn.execute(
            "INSERT INTO chunks (arxiv_id, page, text, embedding) VALUES ('1', 2, %s, %s)",
            ("reranking improves retrieval", np.ones(384)),
        )
        hits = search(conn, "does reranking help zebras", np.ones(384), mode="keyword")
        assert [h.page for h in hits] == [2]
        assert search(conn, "the of and", np.ones(384), mode="keyword") == []
        assert search(conn, "zebras only", np.ones(384), mode="keyword") == []


@pytest.mark.skipif("TEST_DATABASE_URL" not in os.environ, reason="no database")
def test_connection_lets_hnsw_return_all_fusion_candidates(monkeypatch):
    # Asserts the setting: on a small table Postgres skips the index, so a row count proves nothing.
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    with connect() as conn:
        ef_search = int(conn.execute("SHOW hnsw.ef_search").fetchone()[0])
    assert ef_search >= retrieve.CANDIDATES


@pytest.mark.skipif("TEST_DATABASE_URL" not in os.environ, reason="no database")
def test_hybrid_search_fuses_vector_and_keyword_rankings(monkeypatch):
    # Render runs SEARCH_MODE=hybrid. By vector alone the order is near, mid, far; only the far chunk
    # matches the keyword, so fusing both must move it first. A hybrid that drops either list fails.
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    query = np.zeros(384)
    query[0] = 1
    mid = np.zeros(384)
    mid[0] = mid[1] = 2**-0.5
    far = np.zeros(384)
    far[1] = 1
    with connect() as conn:
        migrate(conn)
        conn.execute("TRUNCATE papers, chunks, traces CASCADE")
        conn.execute("INSERT INTO papers VALUES ('1', 'T', 'A', 'cc')")
        for page, text, embedding in [(1, "alpha", query), (2, "beta", mid), (3, "zebras", far)]:
            conn.execute(
                "INSERT INTO chunks (arxiv_id, page, text, embedding) VALUES ('1', %s, %s, %s)",
                (page, text, embedding),
            )
        vector_pages = [h.page for h in search(conn, "zebras", query, mode="vector")]
        hybrid_pages = [h.page for h in search(conn, "zebras", query, mode="hybrid")]
    assert vector_pages == [1, 2, 3]
    assert hybrid_pages == [3, 1, 2]


@pytest.mark.skipif("TEST_DATABASE_URL" not in os.environ, reason="no database")
def test_connection_gives_up_on_a_hung_database(monkeypatch):
    # An unroutable host hangs for minutes without a timeout, so /answer never replied (drill, 2026-11-17).
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    with connect() as conn:
        assert int(conn.info.get_parameters()["connect_timeout"]) <= 10
