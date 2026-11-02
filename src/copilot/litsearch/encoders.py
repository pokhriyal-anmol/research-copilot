"""Qwen3-Embedding encoder with an on-disk document cache."""
import hashlib
from pathlib import Path

import numpy as np

TASK = "Given a scientific literature search query, retrieve relevant papers"
CACHE_DIR = Path("bench/data/cache")
DTYPES = ("float32", "float16", "bfloat16")  # 4B and 8B only fit 16 GB at half precision


def result_name(retriever: str, model: str | None, split: str) -> str:
    if retriever != "dense":
        return f"{retriever}-{split}"
    if not model:
        raise ValueError("--model is required for the dense retriever")
    return f"{model.rsplit('/', 1)[-1].lower()}-{split}"


class Qwen3Encoder:
    def __init__(
        self, model_name: str, model=None, cache_dir: Path | str | None = CACHE_DIR, dtype: str = "float32",
        max_length: int = 512, batch_size: int = 8,
    ) -> None:
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
        self.model_name = model_name
        self.dtype = dtype
        # The model allows 32,768 tokens; uncapped, the longest batch ran out of GPU memory.
        self.max_length = max_length
        self.batch_size = batch_size
        self._capped = False
        self._model = model
        self._cache = Path(cache_dir) if cache_dir is not None else None
        self.cache_key = ""

    def _load(self):
        if self._model is None:  # the only place weights are fetched
            from sentence_transformers import SentenceTransformer  # bench-only dependency

            kwargs = {}
            if self.dtype != "float32":
                import torch

                kwargs["model_kwargs"] = {"torch_dtype": getattr(torch, self.dtype)}
            self._model = SentenceTransformer(self.model_name, **kwargs)
        if not self._capped:
            self._model.max_seq_length = self.max_length
            self._capped = True
        return self._model

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray(self._load().encode([f"Instruct: {TASK}\nQuery:{query}"], batch_size=self.batch_size))[0]

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        digest = hashlib.sha256(self.model_name.encode())
        digest.update(f"|{self.dtype}|{self.max_length}".encode())
        for t in texts:
            digest.update(b"\0" + t.encode())
        self.cache_key = digest.hexdigest()
        path = self._cache / f"{self.cache_key}.npy" if self._cache else None
        if path and path.exists():
            return np.load(path)
        vectors = np.asarray(self._load().encode(texts, batch_size=self.batch_size))
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)
        return vectors
