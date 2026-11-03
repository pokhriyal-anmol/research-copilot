"""The listwise LLM screen's pre-registered rule (BENCHMARK.md, "Screen: listwise LLM over the Qwen top 20")."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("screen", Path(__file__).parents[2] / "bench/rerank/screen.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)

BASE = {"recall@5": 60.0, "ndcg@10": 55.0}


def verdict(**kw):
    args = {"base": BASE, "llm": {"recall@5": 66.0, "ndcg@10": 55.0}, "rose": 5, "fell": 2, "failures": 0} | kw
    return screen.verdict(**args)


def test_a_large_rise_that_keeps_ndcg_is_worth_a_configuration():
    assert verdict() == "worth a full configuration"


def test_a_rise_under_five_points_is_not_enough():
    assert verdict(llm={"recall@5": 64.9, "ndcg@10": 56.0}) == "not worth it"


def test_exactly_five_points_is_enough():
    assert verdict(llm={"recall@5": 65.0, "ndcg@10": 55.0}) == "worth a full configuration"


def test_more_questions_falling_than_rising_fails():
    assert verdict(rose=4, fell=4) == "not worth it"


def test_a_fall_in_ndcg_fails():
    assert verdict(llm={"recall@5": 70.0, "ndcg@10": 54.9}) == "not worth it"


def test_more_than_two_parse_failures_fails():
    assert verdict(failures=3) == "not worth it"
    assert verdict(failures=2) == "worth a full configuration"


def test_the_screen_covers_specific_questions_whose_source_is_not_in_the_corpus():
    assert screen.QUERY_SET == "inline_nonacl" and screen.SPECIFIC == 1 and screen.DEPTH == 20


def test_rise_and_fall_are_counted_per_question():
    base = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
    llm = {"a": [2, 1], "b": [4, 3], "c": [5, 6]}
    gold = {"a": {2}, "b": {3}, "c": {5}}
    assert screen.rose_fell(base, llm, gold, k=1) == (1, 1)
