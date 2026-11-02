"""The split rules come from bench/BENCHMARK.md: 200 dev, 397 test, seed 13, stratified."""
from collections import Counter

from copilot.litsearch.split import split_questions

# 597 synthetic questions shaped like LitSearch: four query sets, two specificities.
STRATA = {
    ("inline_nonacl", 1): 200, ("inline_nonacl", 0): 53, ("manual_acl", 1): 110,
    ("manual_acl", 0): 45, ("inline_acl", 1): 70, ("inline_acl", 0): 28,
    ("manual_iclr", 1): 62, ("manual_iclr", 0): 29,
}
QUESTIONS = [
    {"id": f"{qs}-{sp}-{i}", "query_set": qs, "specificity": sp}
    for (qs, sp), n in STRATA.items() for i in range(n)
]


def test_sizes_match_the_protocol():
    dev, test = split_questions(QUESTIONS, dev_size=200, seed=13)
    assert (len(dev), len(test)) == (200, 397)


def test_dev_and_test_are_disjoint_and_cover_everything():
    dev, test = split_questions(QUESTIONS, dev_size=200, seed=13)
    assert not set(dev) & set(test)
    assert set(dev) | set(test) == {q["id"] for q in QUESTIONS}


def test_same_seed_gives_the_same_split():
    assert split_questions(QUESTIONS, 200, 13) == split_questions(QUESTIONS, 200, 13)


def test_different_seed_gives_a_different_split():
    assert split_questions(QUESTIONS, 200, 13)[0] != split_questions(QUESTIONS, 200, 14)[0]


def test_split_ignores_input_order():
    shuffled = list(reversed(QUESTIONS))
    assert set(split_questions(QUESTIONS, 200, 13)[0]) == set(split_questions(shuffled, 200, 13)[0])


def test_each_stratum_is_within_one_question_of_its_fair_share():
    dev, _ = split_questions(QUESTIONS, dev_size=200, seed=13)
    by_id = {q["id"]: (q["query_set"], q["specificity"]) for q in QUESTIONS}
    got = Counter(by_id[i] for i in dev)
    for stratum, n in STRATA.items():
        assert abs(got[stratum] - 200 * n / 597) <= 1, stratum
