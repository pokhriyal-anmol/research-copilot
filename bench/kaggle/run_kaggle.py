"""Kaggle script kernel: the cross-hardware check, then Qwen3-Embedding-8B on all 597 questions.

Runs on Kaggle's GPU T4 x2 with internet on. Everything it produces lands in /kaggle/working, which
`kaggle kernels output` downloads. Each step logs to steps.log, so a failure says where it stopped.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/pokhriyal-anmol/research-copilot.git"
COMMIT = "__COMMIT__"  # filled in by push.sh, so the run uses exactly the code that was tested
WORK = Path("/kaggle/working")
SRC = Path("/kaggle/temp/research-copilot")
LOG = WORK / "steps.log"
# The private dataset litsearch-qwen3-8b-doc-vectors; when it is mounted, only questions are encoded.
INPUT = Path("/kaggle/input")
RUNS = [  # (model, dtype, device_map, instruction, results folder); the 0.6B row is the cross-hardware check
    ("Qwen/Qwen3-Embedding-0.6B", "float16", None, None, "results"),
    ("Qwen/Qwen3-Embedding-8B", "float16", "auto", None, "results"),
]
WEB = "Given a web search query, retrieve relevant passages that answer the query"  # Qwen's shipped query prompt
VECTOR_RUNS = [  # BENCHMARK.md, 2026-11-03: the protocol instruction must reproduce 81.6 / 62.8, then Qwen's own
    ("Qwen/Qwen3-Embedding-8B", "float16", "auto", None, "results"),
    ("Qwen/Qwen3-Embedding-8B", "float16", "auto", WEB, "results/qwen-default"),
]
DOWNLOAD = (
    "from huggingface_hub import snapshot_download as d; d('princeton-nlp/LitSearch', "
    "repo_type='dataset', local_dir='bench/data', allow_patterns=['query/*', 'corpus_clean/*'])"
)


def step(name: str, argv: list[str], cwd: Path | None = None) -> None:
    started = time.time()
    with LOG.open("a") as log:
        log.write(f"\n== {name}: {' '.join(argv)}\n")
        log.flush()
        code = subprocess.call(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"== {name}: exit {code} after {time.time() - started:.0f}s\n")
    if code != 0:
        sys.exit(f"{name} failed with exit {code}; see steps.log")


def commands(vectors: bool) -> list[list[str]]:
    argvs = []
    for model, dtype, device_map, instruction, out in VECTOR_RUNS if vectors else RUNS:
        argv = ["uv", "run", "--group", "bench", "python", "-m", "copilot.litsearch.run", "--retriever", "dense",
                "--model", model, "--dtype", dtype, "--max-length", "1024", "--batch-size", "8", "--split", "full",
                "--out-dir", str(WORK / out)]
        argv += ["--device-map", device_map] if device_map else []
        argvs.append(argv + (["--instruction", instruction] if instruction else []))
    return argvs


def kept(cache: Path, vectors: bool) -> list[Path]:
    # Question vectors always; paper vectors only when this run made them, not when it was given them.
    queries = [cache / "queries"] if (cache / "queries").is_dir() else []
    return queries if vectors else sorted(cache.glob("*.npy")) + queries


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    step("gpus", ["nvidia-smi"])
    step("clone", ["git", "clone", "--quiet", REPO, str(SRC)])
    step("checkout", ["git", "checkout", "--quiet", COMMIT], cwd=SRC)
    step("install uv", [sys.executable, "-m", "pip", "install", "--quiet", "uv"])
    step("install deps", ["uv", "sync", "--frozen", "--group", "bench", "--quiet"], cwd=SRC)
    step("download data", ["uv", "run", "--group", "bench", "python", "-c", DOWNLOAD], cwd=SRC)
    # The reproduction test checks saved results, not code, so it stays on the Mac.
    step("tests", ["uv", "run", "--group", "bench", "pytest", "-q", "tests/litsearch",
                   "--ignore", "tests/litsearch/test_reproduction.py"], cwd=SRC)
    found = list(INPUT.rglob("*.npy")) if INPUT.is_dir() else []
    vectors = bool(found)
    if vectors:
        (SRC / "bench/data/cache").mkdir(parents=True, exist_ok=True)
        for npy in found:
            step(f"reuse {npy.name}", ["cp", str(npy), str(SRC / "bench/data/cache" / npy.name)])
    for argv in commands(vectors):
        step(f"embed {argv[argv.index('--model') + 1]}", argv, cwd=SRC)
    for cached in kept(SRC / "bench/data/cache", vectors):  # vectors for experiments on the Mac
        step(f"keep {cached.name}", ["cp", "-r", str(cached), str(WORK / cached.name)])
    (WORK / "done.json").write_text(json.dumps({"commit": COMMIT, "finished": time.time()}))


if __name__ == "__main__":
    main()
