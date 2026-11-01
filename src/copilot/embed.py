from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_BATCH = 4  # the default batch of 64 peaked at 650 MB; the host allows 512 MB


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(MODEL, threads=1)


@lru_cache(maxsize=1)
def _reranker() -> TextCrossEncoder:
    return TextCrossEncoder(RERANK_MODEL, threads=1)


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    return list(_reranker().rerank(query, texts, batch_size=RERANK_BATCH))


def embed_passages(texts: list[str]) -> list[np.ndarray]:
    return list(_model().passage_embed(texts))


def embed_query(text: str) -> np.ndarray:
    return next(iter(_model().query_embed(text)))
