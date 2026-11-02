#!/usr/bin/env bash
# Pushes the Kaggle run for the current commit. Usage: bench/kaggle/push.sh
# Needs ~/.kaggle/kaggle.json and a phone-verified Kaggle account (GPU and internet need it).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain -- src tests bench pyproject.toml uv.lock)" ]; then
  echo "Commit your changes first: the Kaggle run checks out a pushed commit." >&2
  exit 1
fi
commit=$(git rev-parse HEAD)
if ! git branch -r --contains "$commit" | grep -q .; then
  echo "Commit $commit is not on GitHub yet. Push it first." >&2
  exit 1
fi
user=$(python3 -c "import json, os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")

out=$(mktemp -d)
sed "s/__COMMIT__/$commit/" bench/kaggle/run_kaggle.py > "$out/run_kaggle.py"
cat > "$out/kernel-metadata.json" <<JSON
{
  "id": "$user/litsearch-qwen3-8b",
  "title": "litsearch-qwen3-8b",
  "code_file": "run_kaggle.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
JSON
uvx --with six --from kaggle kaggle kernels push -p "$out" --accelerator NvidiaTeslaT4
echo "Pushed commit $commit as $user/litsearch-qwen3-8b"
