"""The Qwen3 encoder, checked with a stub model so no test downloads weights."""
from typing import ClassVar

import numpy as np
import pytest

from copilot.litsearch.encoders import TASK, Qwen3Encoder, result_name


class StubModel:
    def __init__(self):
        self.seen = []

    def encode(self, texts, **_):
        self.seen.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32)


def test_task_is_the_one_in_the_protocol():
    assert TASK == "Given a scientific literature search query, retrieve relevant papers"


def test_query_carries_the_model_card_instruction():
    model = StubModel()
    Qwen3Encoder("Qwen/Qwen3-Embedding-8B", model=model, cache_dir=None).encode_query("dense retrieval")
    assert model.seen == [[f"Instruct: {TASK}\nQuery:dense retrieval"]]


def test_documents_get_no_instruction(tmp_path):
    model = StubModel()
    Qwen3Encoder("m", model=model, cache_dir=tmp_path).encode_docs(["Title\n\nAbstract"])
    assert model.seen == [["Title\n\nAbstract"]]


def test_encode_docs_returns_one_row_per_doc(tmp_path):
    vectors = Qwen3Encoder("m", model=StubModel(), cache_dir=tmp_path).encode_docs(["a", "b", "c"])
    assert vectors.shape[0] == 3


def test_second_run_over_the_same_corpus_reads_the_cache(tmp_path):
    Qwen3Encoder("m", model=StubModel(), cache_dir=tmp_path).encode_docs(["a", "b"])
    model = StubModel()
    Qwen3Encoder("m", model=model, cache_dir=tmp_path).encode_docs(["a", "b"])
    assert model.seen == []


def test_a_changed_document_misses_the_cache(tmp_path):
    Qwen3Encoder("m", model=StubModel(), cache_dir=tmp_path).encode_docs(["a", "b"])
    model = StubModel()
    Qwen3Encoder("m", model=model, cache_dir=tmp_path).encode_docs(["a", "b changed"])
    assert model.seen


def test_a_different_model_misses_the_cache(tmp_path):
    Qwen3Encoder("m1", model=StubModel(), cache_dir=tmp_path).encode_docs(["a"])
    model = StubModel()
    Qwen3Encoder("m2", model=model, cache_dir=tmp_path).encode_docs(["a"])
    assert model.seen


@pytest.mark.parametrize(
    ("retriever", "model", "split", "name"),
    [
        ("dense", "Qwen/Qwen3-Embedding-8B", "full", "qwen3-embedding-8b-full"),
        ("dense", "IEITYuan/Yuan-embedding-2.0-en", "dev", "yuan-embedding-2.0-en-dev"),
        ("bm25", None, "test", "bm25-test"),
    ],
)
def test_result_names_match_what_the_reproduction_gate_reads(retriever, model, split, name):
    assert result_name(retriever, model, split) == name


def test_dense_without_a_model_is_an_error():
    with pytest.raises(ValueError):
        result_name("dense", None, "full")


class FakeSentenceTransformer:
    """Records how the real loader would be called, without downloading anything."""

    calls: ClassVar[list] = []

    def __init__(self, name, **kwargs):
        FakeSentenceTransformer.calls.append((name, kwargs))

    def encode(self, texts, **_):
        return np.ones((len(texts), 4), dtype=np.float32)


@pytest.fixture
def fake_loader(monkeypatch):
    st = pytest.importorskip("sentence_transformers")
    FakeSentenceTransformer.calls = []
    monkeypatch.setattr(st, "SentenceTransformer", FakeSentenceTransformer)
    return FakeSentenceTransformer.calls


def test_default_precision_loads_the_model_unchanged(fake_loader):
    Qwen3Encoder("m", cache_dir=None).encode_query("q")
    assert fake_loader == [("m", {})]


def test_half_precision_is_passed_to_the_model_loader(fake_loader):
    torch = pytest.importorskip("torch")
    encoder = Qwen3Encoder("m", cache_dir=None, dtype="float16")
    encoder.encode_query("q")
    assert fake_loader == [("m", {"model_kwargs": {"torch_dtype": torch.float16}})]


def test_unknown_precision_is_rejected_before_loading():
    with pytest.raises(ValueError):
        Qwen3Encoder("m", cache_dir=None, dtype="float8")


def test_precision_is_part_of_the_cache_key(tmp_path):
    a = Qwen3Encoder("m", model=StubModel(), cache_dir=tmp_path)
    a.encode_docs(["x"])
    b = Qwen3Encoder("m", model=StubModel(), cache_dir=tmp_path, dtype="float16")
    b.encode_docs(["x"])
    assert a.cache_key != b.cache_key
