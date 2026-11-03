"""Expected values are hand-calculated from the definitions in bench/BENCHMARK.md."""
import math

import pytest

from copilot.litsearch.metrics import mean_scores, ndcg_at_k, recall_at_k


def test_recall_finds_the_only_gold_paper():
    assert recall_at_k([3, 1, 2], {1}, k=2) == 1.0


def test_recall_misses_gold_beyond_k():
    assert recall_at_k([3, 2, 1], {1}, k=2) == 0.0


def test_recall_is_the_fraction_of_gold_found():
    assert recall_at_k([1, 9, 8], {1, 2}, k=3) == 0.5


def test_duplicates_are_removed_before_cutting_at_k():
    # [5, 5, 6] becomes [5, 6], so both gold papers fit in the top 2.
    assert recall_at_k([5, 5, 6], {5, 6}, k=2) == 1.0


def test_ndcg_is_one_when_gold_is_first():
    assert ndcg_at_k([1, 2, 3], {1}, k=10) == 1.0


def test_ndcg_discounts_gold_at_rank_two():
    assert ndcg_at_k([9, 1], {1}, k=10) == pytest.approx(1 / math.log2(3))


def test_ndcg_with_two_gold_papers_at_ranks_one_and_three():
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([1, 9, 2], {1, 2}, k=10) == pytest.approx(expected)


def test_ndcg_ideal_uses_at_most_k_hits():
    # Three gold papers but k=2: a perfect top 2 must still score 1.0.
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=2) == 1.0


def test_ndcg_is_zero_when_gold_is_beyond_k():
    assert ndcg_at_k([9, 8, 1], {1}, k=2) == 0.0


def test_ndcg_does_not_double_count_a_duplicated_gold_paper():
    assert ndcg_at_k([1, 1], {1}, k=10) == 1.0


@pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k])
def test_empty_gold_is_an_error_not_a_score(metric):
    with pytest.raises(ValueError):
        metric([1, 2], set(), k=2)


def test_mean_scores_averages_over_questions_as_percentages():
    runs = {"a": [1, 2], "b": [9, 3]}
    gold = {"a": {1}, "b": {3}}
    scores = mean_scores(runs, gold, recall_ks=(1,), ndcg_ks=(10,))
    assert scores["recall@1"] == 50.0
    assert scores["ndcg@10"] == pytest.approx(100 * (1 + 1 / math.log2(3)) / 2)


def test_mean_scores_counts_a_question_with_no_run_as_zero():
    scores = mean_scores({"a": [1]}, {"a": {1}, "b": {2}}, recall_ks=(1,), ndcg_ks=())
    assert scores["recall@1"] == 50.0


def test_mean_scores_ignores_runs_for_questions_without_gold():
    # A run for an unknown question must not change the average.
    scores = mean_scores({"a": [1], "zzz": [1]}, {"a": {1}}, recall_ks=(1,), ndcg_ks=())
    assert scores["recall@1"] == 100.0
