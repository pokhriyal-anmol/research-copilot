"""Load LitSearch parquet files from bench/data into plain Python structures."""
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

DATASET = "princeton-nlp/LitSearch"


@dataclass
class Question:
    id: str
    text: str
    gold: set[int]
    query_set: str
    specificity: int


@dataclass
class Data:
    questions: list[Question]
    docs: dict[int, str]
    citations: dict[int, list[int]]
    no_abstract: set[int]
    missing_gold: set[int]


def _blank(abstract: object) -> bool:
    return abstract is None or (isinstance(abstract, float) and math.isnan(abstract)) or not str(abstract).strip()


def doc_text(title: str, abstract: str | None) -> str:
    return title if _blank(abstract) else f"{title}\n\n{abstract}"


def _read(root: Path, config: str):
    import pandas as pd  # bench-only; keeps Data/Question importable without it

    files = sorted((root / config).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files in {root / config}/. Expected {root}/query/*.parquet and "
            f"{root}/corpus_clean/*.parquet from {DATASET} (configs query and corpus_clean). "
            "Download them yourself; this loader never downloads."
        )
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _qid(pos: int, text: str) -> str:
    return hashlib.sha1(f"{pos}\n{text}".encode()).hexdigest()[:10]


def load(root: Path | str = "bench/data") -> Data:
    root = Path(root)
    queries, corpus = _read(root, "query"), _read(root, "corpus_clean")
    ids = [int(c) for c in corpus["corpusid"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate corpusid in corpus_clean")
    docs, citations = {}, {}
    for cid, title, abstract, cites in zip(ids, corpus["title"], corpus["abstract"], corpus["citations"]):
        docs[cid] = doc_text(title, abstract)
        citations[cid] = [int(c) for c in cites]
    no_abstract = {cid for cid, a in zip(ids, corpus["abstract"]) if _blank(a)}
    questions = [
        Question(_qid(i, r.query), r.query, {int(c) for c in r.corpusids}, r.query_set, int(r.specificity))
        for i, r in enumerate(queries.itertuples())
    ]
    missing = {g for q in questions for g in q.gold} - docs.keys()
    return Data(questions, docs, citations, no_abstract, missing)
