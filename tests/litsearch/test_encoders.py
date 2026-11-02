"""The Qwen3 encoder, checked with a stub model so no test downloads weights."""
import json
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


class RecordingModel:
    def __init__(self):
        self.max_seq_length = 32768
        self.encode_kwargs = []

    def encode(self, texts, **kwargs):
        self.encode_kwargs.append(kwargs)
        return np.ones((len(texts), 4), dtype=np.float32)


def test_length_cap_and_batch_size_reach_the_model(tmp_path):
    model = RecordingModel()
    Qwen3Encoder("m", model=model, cache_dir=tmp_path, max_length=256, batch_size=4).encode_docs(["a"])
    assert model.max_seq_length == 256
    assert model.encode_kwargs[-1]["batch_size"] == 4


def test_length_cap_is_part_of_the_cache_key(tmp_path):
    a = Qwen3Encoder("m", model=StubModel(), cache_dir=tmp_path, max_length=256)
    a.encode_docs(["x"])
    b = Qwen3Encoder("m", model=StubModel(), cache_dir=tmp_path, max_length=512)
    b.encode_docs(["x"])
    assert a.cache_key != b.cache_key


def test_documents_are_encoded_in_chunks_and_keep_their_order(tmp_path):
    model = StubModel()
    model.encode = lambda texts, **_: (model.seen.append(list(texts)) or np.array([[float(t)] for t in texts]))
    encoder = Qwen3Encoder("m", model=model, cache_dir=tmp_path, chunk_size=2)
    vectors = encoder.encode_docs(["1", "2", "3", "4", "5"])
    assert [len(batch) for batch in model.seen] == [2, 2, 1]
    assert vectors[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_progress_file_reports_documents_done_out_of_total(tmp_path):
    progress = tmp_path / "progress.json"
    seen = []

    class Watching(StubModel):
        def encode(self, texts, **_):
            if progress.exists():
                seen.append(json.loads(progress.read_text())["done"])
            return super().encode(texts)

    Qwen3Encoder("Org/M", model=Watching(), cache_dir=tmp_path, chunk_size=2, progress_path=progress).encode_docs(
        ["a", "b", "c", "d", "e"]
    )
    final = json.loads(progress.read_text())
    assert (final["model"], final["done"], final["total"]) == ("Org/M", 5, 5)
    assert seen == [0, 2, 4]  # written before each chunk, so a watcher sees it move


def test_no_progress_file_unless_asked(tmp_path):
    Qwen3Encoder("m", model=StubModel(), cache_dir=tmp_path, chunk_size=2).encode_docs(["a", "b", "c"])
    assert list(tmp_path.glob("*.json")) == []


def test_device_map_is_passed_to_the_model_loader(fake_loader):
    Qwen3Encoder("m", cache_dir=None, device_map="auto").encode_query("q")
    assert fake_loader == [("m", {"model_kwargs": {"device_map": "auto"}})]


def test_device_map_and_precision_combine(fake_loader):
    torch = pytest.importorskip("torch")
    Qwen3Encoder("m", cache_dir=None, dtype="float16", device_map="auto").encode_query("q")
    assert fake_loader == [("m", {"model_kwargs": {"torch_dtype": torch.float16, "device_map": "auto"}})]


class NanModel(StubModel):
    def encode(self, texts, **_):
        out = np.ones((len(texts), 4), dtype=np.float32)
        out[0, 0] = np.nan
        return out


def test_documents_with_invalid_numbers_fail_loudly_and_are_not_cached(tmp_path):
    # fp16 on a GPU without bf16 can overflow; a NaN vector must stop the run, never be scored.
    with pytest.raises(ValueError, match="finite"):
        Qwen3Encoder("m", model=NanModel(), cache_dir=tmp_path).encode_docs(["a", "b"])
    assert list(tmp_path.glob("*.npy")) == []


def test_query_with_invalid_numbers_fails_loudly():
    with pytest.raises(ValueError, match="finite"):
        Qwen3Encoder("m", model=NanModel(), cache_dir=None).encode_query("q")
