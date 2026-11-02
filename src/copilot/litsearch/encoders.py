"""Qwen3-Embedding encoder with an on-disk document cache."""
import hashlib
from pathlib import Path

import numpy as np

TASK = "Given a scientific literature search query, retrieve relevant papers"
CACHE_DIR = Path("bench/data/cache")


def result_name(retriever: str, model: str | None, split: str) -> str:
    if retriever != "dense":
        return f"{retriever}-{split}"
    if not model:
        raise ValueError("--model is required for the dense retriever")
    return f"{model.rsplit('/', 1)[-1].lower()}-{split}"


class Qwen3Encoder:
    def __init__(self, model_name: str, model=None, cache_dir: Path | str | None = CACHE_DIR) -> None:
        self.model_name = model_name
        self._model = model
        self._cache = Path(cache_dir) if cache_dir is not None else None
        self.cache_key = ""

    def _load(self):
        if self._model is None:  # the only place weights are fetched
            from sentence_transformers import SentenceTransformer  # bench-only dependency

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray(self._load().encode([f"Instruct: {TASK}\nQuery:{query}"]))[0]

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        digest = hashlib.sha256(self.model_name.encode())
        for t in texts:
            digest.update(b"\0" + t.encode())
        self.cache_key = digest.hexdigest()
        path = self._cache / f"{self.cache_key}.npy" if self._cache else None
        if path and path.exists():
            return np.load(path)
        vectors = np.asarray(self._load().encode(texts))
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)
        return vectors
