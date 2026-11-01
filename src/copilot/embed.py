from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(MODEL)


def embed_passages(texts: list[str]) -> list[np.ndarray]:
    return list(_model().passage_embed(texts))


def embed_query(text: str) -> np.ndarray:
    return next(iter(_model().query_embed(text)))
