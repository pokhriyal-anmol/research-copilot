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
RUNS = [  # (model, dtype, device_map); the 0.6B row is the cross-hardware check in BENCHMARK.md
    ("Qwen/Qwen3-Embedding-0.6B", "float16", None),
    ("Qwen/Qwen3-Embedding-8B", "float16", "auto"),
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


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    step("gpus", ["nvidia-smi"])
    step("clone", ["git", "clone", "--quiet", REPO, str(SRC)])
    step("checkout", ["git", "checkout", "--quiet", COMMIT], cwd=SRC)
    step("install uv", [sys.executable, "-m", "pip", "install", "--quiet", "uv"])
    step("install deps", ["uv", "sync", "--frozen", "--group", "bench", "--quiet"], cwd=SRC)
    step("download data", ["uv", "run", "--group", "bench", "python", "-c", DOWNLOAD], cwd=SRC)
    step("tests", ["uv", "run", "--group", "bench", "pytest", "-q", "tests/litsearch"], cwd=SRC)
    for model, dtype, device_map in RUNS:
        argv = ["uv", "run", "--group", "bench", "python", "-m", "copilot.litsearch.run", "--retriever", "dense",
                "--model", model, "--dtype", dtype, "--max-length", "1024", "--batch-size", "8", "--split", "full",
                "--out-dir", str(WORK / "results")]
        step(f"embed {model}", [*argv, "--device-map", device_map] if device_map else argv, cwd=SRC)
    for cached in (SRC / "bench/data/cache").glob("*.npy"):  # 8B document vectors, for later experiments
        step(f"keep {cached.name}", ["cp", str(cached), str(WORK / cached.name)])
    (WORK / "done.json").write_text(json.dumps({"commit": COMMIT, "finished": time.time()}))


if __name__ == "__main__":
    main()
