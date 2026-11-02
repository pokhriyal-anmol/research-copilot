"""One-hop expansion as LitSearch describes it (arXiv 2407.18940, section 4)."""
from copilot.litsearch.graph import one_hop

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
    import inspect

    params = inspect.signature(one_hop).parameters
    assert (params["m"].default, params["n"].default) == (50, 200)
