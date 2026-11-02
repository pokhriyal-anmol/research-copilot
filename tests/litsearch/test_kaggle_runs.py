"""What the Kaggle script runs, with and without the uploaded 8B paper vectors (BENCHMARK.md, 2026-11-03)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("run_kaggle", Path(__file__).parents[2] / "bench/kaggle/run_kaggle.py")
run_kaggle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_kaggle)


def flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def test_without_vectors_the_cross_hardware_check_runs_before_the_8b():
    models = [flag(a, "--model") for a in run_kaggle.commands(vectors=False)]
    assert models == ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-8B"]


WEB = "Given a web search query, retrieve relevant passages that answer the query"  # Qwen's shipped query prompt


def test_with_vectors_only_the_8b_runs_first_with_the_protocol_instruction_then_qwens_own():
    argvs = run_kaggle.commands(vectors=True)
    assert [flag(a, "--model") for a in argvs] == ["Qwen/Qwen3-Embedding-8B"] * 2
    assert [flag(a, "--instruction") for a in argvs] == [None, WEB]


def test_every_run_keeps_the_settings_the_vectors_were_made_with():
    # Changing dtype or max length changes the cache key, and the uploaded vectors would be ignored.
    for argv in run_kaggle.commands(vectors=True):
        assert (flag(argv, "--dtype"), flag(argv, "--max-length"), flag(argv, "--split")) == ("float16", "1024", "full")


def test_the_two_8b_results_do_not_overwrite_each_other():
    outs = [flag(a, "--out-dir") for a in run_kaggle.commands(vectors=True)]
    assert len(set(outs)) == 2
