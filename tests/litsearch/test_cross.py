"""Qwen3-Reranker scoring (BENCHMARK.md, reranker configuration 2), checked with stubs so no test loads weights."""
import numpy as np
import pytest

from copilot.litsearch.cross import (
    PREFIX,
    SUFFIX,
    Qwen3Reranker,
    build_inputs,
    format_pair,
    reorder,
    yes_probability,
)
from copilot.litsearch.encoders import TASK


class CharTokenizer:
    """One token per character, so lengths are easy to reason about."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_the_pair_follows_the_model_card_format():
    assert format_pair("Find papers", "q?", "Title\n\nAbstract") == "<Instruct>: Find papers\n<Query>: q?\n<Document>: Title\n\nAbstract"


def test_the_template_is_the_model_cards():
    assert PREFIX.startswith("<|im_start|>system\nJudge whether the Document meets the requirements")
    assert 'the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n' in PREFIX
    assert SUFFIX == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_a_long_document_is_cut_but_the_template_is_kept():
    tok = CharTokenizer()
    budget = len(PREFIX) + len(SUFFIX) + 40
    ids = build_inputs(tok, [format_pair("i", "q", "x" * 5000)], max_length=budget)[0]
    text = "".join(map(chr, ids))
    assert len(ids) == budget and text.startswith(PREFIX) and text.endswith(SUFFIX)


def test_a_short_pair_is_not_padded_or_cut():
    tok = CharTokenizer()
    middle = format_pair("i", "q", "doc")
    ids = build_inputs(tok, [middle], max_length=4096)[0]
    assert "".join(map(chr, ids)) == PREFIX + middle + SUFFIX


def test_the_score_is_the_probability_of_yes():
    p = yes_probability(np.array([[0.0, 0.0], [0.0, np.log(3.0)]]))  # columns: no, yes
    assert p == pytest.approx([0.5, 0.75])


class StubReranker(Qwen3Reranker):
    """Scores by document length, so the expected order is known, and counts model calls."""

    def __init__(self, **kw):
        super().__init__("m", tokenizer=CharTokenizer(), **kw)
        self.batches = []

    def _logits(self, batch):
        self.batches.append(len(batch))
        return np.array([[0.0, len(ids) / 1000] for ids in batch])


def test_documents_are_scored_in_batches():
    r = StubReranker(batch_size=2, cache_dir=None)
    scores = r.scores("q", ["a", "bbb", "cc"])
    assert r.batches == [2, 1] and np.argsort(-scores).tolist() == [1, 2, 0]


def test_the_protocol_instruction_is_the_default():
    assert StubReranker(cache_dir=None).instruction == TASK


def test_scores_for_a_question_are_cached(tmp_path):
    StubReranker(cache_dir=tmp_path).scores("q", ["a", "b"])
    again = StubReranker(cache_dir=tmp_path)
    again.scores("q", ["a", "b"])
    assert again.batches == []


def test_a_different_question_instruction_or_documents_miss_the_cache(tmp_path):
    StubReranker(cache_dir=tmp_path).scores("q", ["a", "b"])
    for kw, query, docs in (({}, "other", ["a", "b"]), ({"instruction": "x"}, "q", ["a", "b"]), ({}, "q", ["a", "c"])):
        r = StubReranker(cache_dir=tmp_path, **kw)
        r.scores(query, docs)
        assert r.batches


def test_a_nan_score_stops_the_run():
    class Broken(StubReranker):
        def _logits(self, batch):
            return np.full((len(batch), 2), np.nan)

    with pytest.raises(ValueError, match="not finite"):
        Broken(cache_dir=None).scores("q", ["a"])


def test_reorder_sorts_the_head_and_keeps_the_tail():
    assert reorder([10, 20, 30, 40], np.array([0.1, 0.9, 0.5]), depth=3) == [20, 30, 10, 40]


def test_reorder_breaks_ties_by_dense_rank():
    assert reorder([10, 20, 30], np.array([0.5, 0.5, 0.9]), depth=3) == [30, 10, 20]


def test_only_the_last_position_is_scored_and_padding_is_masked_on_the_left():
    # Full-vocabulary logits at every position cost ~1.5 GB a batch and swapped a 16 GB Mac.
    torch = pytest.importorskip("torch")
    seen = {}

    class FakeModel:
        device = torch.device("cpu")

        def __call__(self, **kw):
            seen.update(kw)
            logits = torch.zeros((kw["input_ids"].shape[0], 1, 10))
            logits[:, -1, 3], logits[:, -1, 4] = 1.0, 2.0  # ids 3 and 4 stand for "no" and "yes"
            return type("Out", (), {"logits": logits})()

    class Tok(CharTokenizer):
        pad_token_id = 0

    r = Qwen3Reranker("m", model=FakeModel(), tokenizer=Tok(), cache_dir=None)
    r._ids = [3, 4]
    out = r._logits([[5, 6, 7], [8]])
    assert seen["logits_to_keep"] == 1
    assert seen["input_ids"].tolist() == [[5, 6, 7], [0, 0, 8]]
    assert seen["attention_mask"].tolist() == [[1, 1, 1], [0, 0, 1]]
    assert out.tolist() == [[1.0, 2.0], [1.0, 2.0]]
