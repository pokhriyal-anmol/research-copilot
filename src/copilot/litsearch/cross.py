"""Qwen3-Reranker as its model card runs it: P("yes") for each (question, paper) pair, with a score cache."""
import hashlib
from pathlib import Path

import numpy as np

from copilot.litsearch.encoders import CACHE_DIR, TASK, _finite

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
          'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_pair(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def build_inputs(tokenizer, middles: list[str], max_length: int) -> list[list[int]]:
    """Token ids per pair; only the pair text is cut, so the template's answer cue always survives."""
    prefix, suffix = tokenizer.encode(PREFIX, add_special_tokens=False), tokenizer.encode(SUFFIX, add_special_tokens=False)
    room = max_length - len(prefix) - len(suffix)
    return [prefix + tokenizer.encode(m, add_special_tokens=False)[:room] + suffix for m in middles]


def yes_probability(no_yes_logits: np.ndarray) -> np.ndarray:
    """Softmax over the ("no", "yes") logits at the last position, taking "yes"."""
    z = no_yes_logits - no_yes_logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def reorder(ranking: list[int], scores: np.ndarray, depth: int) -> list[int]:
    head = ranking[:depth]
    order = sorted(range(len(head)), key=lambda i: (-scores[i], i))  # ties go to the better dense rank
    return [head[i] for i in order] + ranking[depth:]


class Qwen3Reranker:
    def __init__(self, model_name: str, model=None, tokenizer=None, dtype: str = "float16", max_length: int = 1024,
                 batch_size: int = 8, cache_dir: Path | str | None = CACHE_DIR, instruction: str = TASK) -> None:
        self.model_name, self.dtype, self.max_length = model_name, dtype, max_length
        self.batch_size, self.instruction = batch_size, instruction
        self._model, self._tokenizer = model, tokenizer
        self._cache = Path(cache_dir) / "rerank" if cache_dir is not None else None

    def _load(self):
        if self._model is None:  # the only place weights are fetched
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = self._tokenizer or AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=getattr(torch, self.dtype)).to(device).eval()
            self._ids = [self._tokenizer.convert_tokens_to_ids(t) for t in ("no", "yes")]
        return self._model

    def _logits(self, batch: list[list[int]]) -> np.ndarray:
        import torch

        model = self._load()
        width = max(map(len, batch))
        pad = self._tokenizer.pad_token_id
        ids = torch.tensor([[pad] * (width - len(b)) + b for b in batch], device=model.device)
        mask = torch.tensor([[0] * (width - len(b)) + [1] * len(b) for b in batch], device=model.device)
        with torch.no_grad():
            # Only the last position: full-vocabulary logits everywhere swapped a 16 GB Mac.
            last = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1, :]
        return last[:, self._ids].float().cpu().numpy()

    def scores(self, query: str, docs: list[str]) -> np.ndarray:
        key = hashlib.sha256(f"{self.model_name}|{self.dtype}|{self.max_length}|{self.instruction}\0{query}".encode())
        for d in docs:
            key.update(b"\0" + d.encode())
        path = self._cache / f"{key.hexdigest()}.npy" if self._cache else None
        if path and path.exists():
            return np.load(path)
        if self._tokenizer is None:
            self._load()
        inputs = build_inputs(self._tokenizer, [format_pair(self.instruction, query, d) for d in docs], self.max_length)
        logits = [self._logits(inputs[i:i + self.batch_size]) for i in range(0, len(inputs), self.batch_size)]
        scores = yes_probability(_finite(np.concatenate(logits)))
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, scores)
        return scores
