"""Qwen3-Embedding encoder with on-disk document and query caches."""
import hashlib
import json
from pathlib import Path

import numpy as np

TASK = "Given a scientific literature search query, retrieve relevant papers"
CACHE_DIR = Path("bench/data/cache")
PROGRESS_FILE = Path("bench/data/progress.json")  # read by the run-tracker pane
DTYPES = ("float32", "float16", "bfloat16")  # 4B and 8B only fit 16 GB at half precision


def _finite(vectors: np.ndarray) -> np.ndarray:
    # fp16 on GPUs without bf16 can overflow; a NaN vector would score silently as a miss.
    if not np.isfinite(vectors).all():
        raise ValueError("encoder returned values that are not finite (NaN or inf); try another dtype")
    return vectors


def result_name(retriever: str, model: str | None, split: str) -> str:
    if retriever != "dense":
        return f"{retriever}-{split}"
    if not model:
        raise ValueError("--model is required for the dense retriever")
    return f"{model.rsplit('/', 1)[-1].lower()}-{split}"


class Qwen3Encoder:
    def __init__(
        self, model_name: str, model=None, cache_dir: Path | str | None = CACHE_DIR, dtype: str = "float32",
        max_length: int = 512, batch_size: int = 8, chunk_size: int = 1024,
        progress_path: Path | str | None = None, device_map: str | None = None, task: str = TASK,
    ) -> None:
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
        self.model_name = model_name
        self.dtype = dtype
        # The model allows 32,768 tokens; uncapped, the longest batch ran out of GPU memory.
        self.max_length = max_length
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self._progress = Path(progress_path) if progress_path is not None else None
        self.task = task
        self.device_map = device_map  # "auto" splits a model too big for one GPU across several
        self._capped = False
        self._model = model
        self._cache = Path(cache_dir) if cache_dir is not None else None
        self.cache_key = ""

    def _load(self):
        if self._model is None:  # the only place weights are fetched
            from sentence_transformers import SentenceTransformer  # bench-only dependency

            model_kwargs = {}
            if self.dtype != "float32":
                import torch

                model_kwargs["torch_dtype"] = getattr(torch, self.dtype)
            if self.device_map is not None:
                model_kwargs["device_map"] = self.device_map
            kwargs = {"model_kwargs": model_kwargs} if model_kwargs else {}
            self._model = SentenceTransformer(self.model_name, **kwargs)
        if not self._capped:
            self._model.max_seq_length = self.max_length
            self._capped = True
        return self._model

    def encode_query(self, query: str) -> np.ndarray:
        text = f"Instruct: {self.task}\nQuery:{query}" if self.task else query
        key = hashlib.sha256(f"{self.model_name}|{self.dtype}|{self.max_length}\0{text}".encode()).hexdigest()
        path = self._cache / "queries" / f"{key}.npy" if self._cache else None
        if path and path.exists():
            return np.load(path)
        vector = _finite(np.asarray(self._load().encode([text], batch_size=self.batch_size))[0])
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vector)
        return vector

    def _report(self, done: int, total: int) -> None:
        if self._progress is not None:
            self._progress.parent.mkdir(parents=True, exist_ok=True)
            self._progress.write_text(json.dumps({"model": self.model_name, "done": done, "total": total}))

    def _encode_in_chunks(self, texts: list[str]) -> np.ndarray:
        parts = []
        for start in range(0, len(texts), self.chunk_size):
            self._report(start, len(texts))
            chunk = texts[start:start + self.chunk_size]
            parts.append(_finite(np.asarray(self._load().encode(chunk, batch_size=self.batch_size))))
        self._report(len(texts), len(texts))
        return np.concatenate(parts) if parts else np.empty((0, 0), dtype=np.float32)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        digest = hashlib.sha256(self.model_name.encode())
        digest.update(f"|{self.dtype}|{self.max_length}".encode())
        for t in texts:
            digest.update(b"\0" + t.encode())
        self.cache_key = digest.hexdigest()
        path = self._cache / f"{self.cache_key}.npy" if self._cache else None
        if path and path.exists():
            return np.load(path)
        vectors = self._encode_in_chunks(texts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)
        return vectors
