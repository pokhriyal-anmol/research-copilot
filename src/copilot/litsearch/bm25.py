"""BM25 over title and abstract, the published keyword baseline."""
import bm25s
import numpy as np


class BM25Index:
    tokenizer = "bm25s.tokenize"
    stopwords = "en"
    stemmer = None  # PyStemmer is not a dependency
    method, k1, b = "lucene", 1.5, 0.75  # bm25s defaults

    def __init__(self, docs: dict[int, str]) -> None:
        self._ids = list(docs)
        self._bm25 = bm25s.BM25(method=self.method, k1=self.k1, b=self.b)
        self._bm25.index(bm25s.tokenize(list(docs.values()), stopwords=self.stopwords, show_progress=False), show_progress=False)

    def search(self, query: str, k: int) -> list[int]:
        words = bm25s.tokenize(query, stopwords=self.stopwords, return_ids=False, show_progress=False)[0]
        words = [w for w in words if w in self._bm25.vocab_dict]  # unseen words would raise
        if not words:
            return []
        scores = self._bm25.get_scores(words)
        order = np.lexsort((np.arange(len(scores)), -scores))  # ties: corpus order
        return [self._ids[i] for i in order[: max(k, 0)] if scores[i] > 0]
