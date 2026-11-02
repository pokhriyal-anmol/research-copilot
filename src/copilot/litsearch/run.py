"""Run one retriever over LitSearch questions and write the result file the gate reads."""
import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from copilot.litsearch.data import Data, Question, load
from copilot.litsearch.encoders import result_name
from copilot.litsearch.metrics import mean_scores
from copilot.litsearch.split import split_questions

SEED, DEV_SIZE, DEPTH = 13, 200, 100
RECALL_KS, NDCG_KS = (20, 100), (10,)


def _score(questions: list[Question], runs: dict[str, list[int]]) -> dict:
    gold = {q.id: q.gold for q in questions}
    return {**mean_scores(runs, gold, RECALL_KS, NDCG_KS), "questions": len(questions)}


def evaluate(
    data: Data, search: Callable[[str, int], list[int]], question_ids: list[str], name: str,
    settings: dict | None = None,
) -> dict:
    wanted = set(question_ids)
    questions = [q for q in data.questions if q.id in wanted]
    runs = {q.id: search(q.text, DEPTH) for q in questions}
    groups = {
        **{f"specificity={v}": [q for q in questions if q.specificity == v]
           for v in sorted({q.specificity for q in questions})},
        **{f"query_set={v}": [q for q in questions if q.query_set == v]
           for v in sorted({q.query_set for q in questions})},
        "gold_without_abstract": [q for q in questions if q.gold & data.no_abstract],
    }
    return {
        "name": name,
        "questions": len(questions),
        "scores": mean_scores(runs, {q.id: q.gold for q in questions}, RECALL_KS, NDCG_KS),
        "breakdowns": {g: _score(qs, runs) for g, qs in groups.items() if qs},
        "settings": settings or {},
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "missing_gold": sum(1 for q in questions if q.gold & data.missing_gold),
    }


def save(result: dict, out_dir: Path | str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{result['name']}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m copilot.litsearch.run")
    ap.add_argument("--retriever", choices=["bm25", "dense"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--progress-file", default="bench/data/progress.json")
    ap.add_argument("--device-map", default=None, help="auto splits the model across GPUs")
    ap.add_argument("--split", choices=["dev", "test", "full"], required=True)
    ap.add_argument("--data-dir", default="bench/data")
    ap.add_argument("--out-dir", default="bench/results")
    args = ap.parse_args(argv)
    try:
        name = result_name(args.retriever, args.model, args.split)
    except ValueError as e:
        ap.exit(2, f"error: {e}\n")
    data = load(args.data_dir)
    ids = [q.id for q in data.questions]
    if args.split != "full":  # full needs no split, and a tiny corpus cannot supply 200 dev seats
        dev, test = split_questions(
            [{"id": q.id, "query_set": q.query_set, "specificity": q.specificity} for q in data.questions],
            dev_size=DEV_SIZE, seed=SEED,
        )
        ids = dev if args.split == "dev" else test
    if args.retriever == "dense":
        from copilot.litsearch.dense import DenseIndex
        from copilot.litsearch.encoders import TASK, Qwen3Encoder

        encoder = Qwen3Encoder(args.model, dtype=args.dtype, max_length=args.max_length, batch_size=args.batch_size,
            progress_path=args.progress_file, device_map=args.device_map,
        )
        index = DenseIndex(data.docs, encoder)
        settings = {
            "retriever": "dense", "model": args.model, "dtype": args.dtype, "device_map": args.device_map, "max_length": args.max_length,
            "batch_size": args.batch_size, "instruction": TASK,
            "cache_key": encoder.cache_key, "seed": SEED, "split": args.split,
        }
    else:
        from copilot.litsearch.bm25 import BM25Index  # bench-only dependency

        index = BM25Index(data.docs)
        settings = {
            "retriever": "bm25", "method": index.method, "k1": index.k1, "b": index.b,
            "stopwords": index.stopwords, "seed": SEED, "split": args.split,
        }
    save(evaluate(data, index.search, ids, name, settings), args.out_dir)


if __name__ == "__main__":
    main()
