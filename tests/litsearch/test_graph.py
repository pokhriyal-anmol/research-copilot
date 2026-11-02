"""One-hop expansion as LitSearch describes it (arXiv 2407.18940, section 4)."""
import inspect

import numpy as np
import pytest

from copilot.litsearch.graph import one_hop, pagerank, structural_rerank

CITES = {1: [10, 11], 2: [11, 12], 3: [], 10: [99], 11: [], 12: []}


def test_each_seed_is_followed_by_the_papers_it_cites():
    assert one_hop([1, 2, 3], CITES) == [1, 10, 11, 2, 12, 3]


def test_a_duplicate_keeps_its_first_place():
    # 11 is cited by seeds 1 and 2; it stays after seed 1.
    assert one_hop([1, 2], CITES).count(11) == 1
    assert one_hop([2, 1], CITES) == [2, 11, 12, 1, 10]


def test_a_seed_already_listed_as_a_reference_is_not_repeated_but_still_expanded():
    # 10 appears once, in its earlier place, and as a top-m seed its own references follow.
    assert one_hop([1, 10], CITES) == [1, 10, 11, 99]


def test_only_the_first_m_papers_are_expanded():
    # Seed 2 is past m, so its references are not added; it is dropped too, as in the paper.
    assert one_hop([1, 2], CITES, m=1) == [1, 10, 11]


def test_the_list_is_cut_at_n():
    assert one_hop([1, 2, 3], CITES, n=4) == [1, 10, 11, 2]


def test_references_are_followed_one_hop_only():
    # 10 cites 99, but 10 is a reference, not a seed.
    assert 99 not in one_hop([1], CITES)


def test_papers_that_cite_a_seed_are_not_added():
    assert one_hop([12], CITES) == [12]


def test_a_paper_missing_from_the_citation_table_counts_as_citing_nothing():
    assert one_hop([7, 1], CITES) == [7, 1, 10, 11]


def test_the_published_settings_are_the_defaults():
    params = inspect.signature(one_hop).parameters
    assert (params["m"].default, params["n"].default) == (50, 200)


# Method 2: structural re-ranking (BENCHMARK.md)


def test_pagerank_scores_sum_to_one():
    assert sum(pagerank([1, 2, 3], {1: [2], 2: [3]}).values()) == pytest.approx(1.0)


def test_pagerank_of_a_two_paper_cycle_is_even():
    scores = pagerank([1, 2], {1: [2], 2: [1]})
    assert scores[1] == pytest.approx(0.5) and scores[2] == pytest.approx(0.5)


def test_a_paper_cited_by_the_others_has_the_highest_pagerank():
    scores = pagerank([1, 2, 3, 4], {2: [1], 3: [1], 4: [1]})
    assert max(scores, key=scores.get) == 1


def test_pagerank_matches_the_closed_form_on_a_small_graph():
    # Solve r = (1-d)/N + d * M r directly, with dangling papers spreading uniformly.
    papers, links, d = [1, 2, 3], {1: [2, 3], 2: [3]}, 0.85
    n = len(papers)
    m = np.zeros((n, n))
    for j, p in enumerate(papers):
        out = [q for q in links.get(p, []) if q in papers]
        for q in out or papers:
            m[papers.index(q), j] += 1 / len(out or papers)
    expected = np.linalg.solve(np.eye(n) - d * m, np.full(n, (1 - d) / n))
    got = pagerank(papers, links)
    assert [got[p] for p in papers] == pytest.approx(list(expected))


def test_links_to_papers_outside_the_candidates_are_ignored():
    assert pagerank([1, 2], {1: [99], 2: [99]}) == pytest.approx({1: 0.5, 2: 0.5})


def test_with_no_links_among_candidates_the_dense_order_is_kept():
    assert structural_rerank([5, 4, 3, 2, 1], {5: [77], 1: [88]}) == [5, 4, 3, 2, 1]


def test_a_paper_cited_by_many_candidates_moves_up():
    ranking = [1, 2, 3, 4, 5]
    reranked = structural_rerank(ranking, {1: [5], 2: [5], 3: [5], 4: [5]})
    assert reranked.index(5) < ranking.index(5)


def test_reranking_adds_and_drops_nothing():
    ranking = list(range(1, 51))
    links = {i: [i + 1, i + 7] for i in ranking}
    assert sorted(structural_rerank(ranking, links)) == ranking


def test_structural_reranking_uses_the_published_settings():
    assert inspect.signature(pagerank).parameters["damping"].default == 0.85
    assert inspect.signature(structural_rerank).parameters["rrf_k"].default == 60
