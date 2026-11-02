"""The command line, end to end on a tiny LitSearch-shaped corpus."""
import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("bm25s")

from copilot.litsearch.run import main

QUERIES = [
    {"query_set": "manual_acl", "query": "graph neural networks for molecules", "specificity": 1,
     "quality": 2, "corpusids": [3]},
    {"query_set": "inline_acl", "query": "dense passage retrieval", "specificity": 0,
     "quality": 1, "corpusids": [1]},
    {"query_set": "manual_iclr", "query": "probabilistic relevance framework", "specificity": 1,
     "quality": 2, "corpusids": [2]},
]
CORPUS = [
    {"corpusid": 1, "title": "Dense passage retrieval", "abstract": "Open domain QA.", "citations": []},
    {"corpusid": 2, "title": "The probabilistic relevance framework", "abstract": "", "citations": [1]},
    {"corpusid": 3, "title": "Graph neural networks for molecules", "abstract": "Chemistry.",
     "citations": []},
]


@pytest.fixture
def data_dir(tmp_path):
    for sub, rows in (("query", QUERIES), ("corpus_clean", CORPUS)):
        (tmp_path / sub).mkdir()
        pd.DataFrame(rows).to_parquet(tmp_path / sub / "full-00000-of-00001.parquet")
    return tmp_path


def test_bm25_full_run_writes_the_result_file(data_dir, tmp_path):
    out = tmp_path / "results"
    main(["--retriever", "bm25", "--split", "full", "--data-dir", str(data_dir), "--out-dir", str(out)])
    result = json.loads((out / "bm25-full.json").read_text())
    assert result["questions"] == 3
    assert result["scores"]["recall@20"] == 100.0  # every gold paper is in a 3-paper corpus
    assert result["settings"]["split"] == "full" and result["settings"]["seed"] == 13


def test_dense_run_without_a_model_exits_with_an_error(data_dir, tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        main(["--retriever", "dense", "--split", "full", "--data-dir", str(data_dir),
              "--out-dir", str(tmp_path)])
    assert exit_info.value.code != 0


def test_dense_run_records_model_and_precision(data_dir, tmp_path, monkeypatch):
    from copilot.litsearch import encoders

    class Stub:
        def encode(self, texts, **_):
            return np.ones((len(texts), 4))

    monkeypatch.setattr(encoders.Qwen3Encoder, "_load", lambda self: Stub())
    out = tmp_path / "results"
    main(["--retriever", "dense", "--model", "Org/Some-Model", "--dtype", "float16", "--split", "full",
          "--data-dir", str(data_dir), "--out-dir", str(out)])
    settings = json.loads((out / "some-model-full.json").read_text())["settings"]
    assert (settings["model"], settings["dtype"]) == ("Org/Some-Model", "float16")
