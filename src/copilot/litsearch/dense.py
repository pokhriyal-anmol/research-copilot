"""Dense first-stage retriever: cosine ranking over any encoder's vectors."""
import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


class DenseIndex:
    def __init__(self, docs: dict[int, str], encoder) -> None:
        self._ids = list(docs)
        self._encoder = encoder
        self._vecs = _unit(np.asarray(encoder.encode_docs(list(docs.values())), dtype=np.float32))

    def search_scored(self, query: str, k: int) -> list[tuple[int, float]]:
        q = _unit(np.asarray(self._encoder.encode_query(query), dtype=np.float32))
        scores = self._vecs @ q
        order = np.argsort(-scores, kind="stable")[: max(k, 0)]
        return [(self._ids[i], float(scores[i])) for i in order]

    def search(self, query: str, k: int) -> list[int]:
        return [d for d, _ in self.search_scored(query, k)]
