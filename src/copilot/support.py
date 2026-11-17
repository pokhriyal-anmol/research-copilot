"""Checks that each cited passage supports the sentence that cites it, using a small local NLI model."""
import platform
import re
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from copilot.retrieve import Passage

MODEL_REPO = "cross-encoder/nli-deberta-v3-xsmall"  # Apache-2.0, 87 MB quantised
ENTAILMENT = 1  # label order from the model's config.json: contradiction, entailment, neutral
CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def _load() -> tuple[Tokenizer, ort.InferenceSession]:
    quantised = "model_qint8_arm64.onnx" if platform.machine() in ("arm64", "aarch64") else "model_quint8_avx2.onnx"
    tokenizer = Tokenizer.from_file(hf_hub_download(MODEL_REPO, "tokenizer.json"))
    tokenizer.enable_truncation(512)
    options = ort.SessionOptions()
    # Graph optimisation roughly doubled the load-time peak (488 vs 270 MB), which does not fit Render's 512 MB.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena = False
    options.intra_op_num_threads = 1
    return tokenizer, ort.InferenceSession(hf_hub_download(MODEL_REPO, f"onnx/{quantised}"), options)


def entails(premise: str, hypothesis: str) -> bool:
    tokenizer, session = _load()
    encoded = tokenizer.encode(premise, hypothesis)
    feed = {"input_ids": np.array([encoded.ids]), "attention_mask": np.array([encoded.attention_mask])}
    return int(session.run(None, feed)[0][0].argmax()) == ENTAILMENT


def unsupported_sentences(text: str, passages: list[Passage]) -> list[str]:
    """Cited sentences whose cited passages, taken together, do not entail them. Numbers are already range-checked."""
    from copilot.answer import cited_numbers

    bad = []
    for sentence in _SENTENCE_END.split(text):
        numbers = cited_numbers(sentence)
        if numbers and not entails(" ".join(passages[n - 1].text for n in sorted(numbers)), re.sub(r"\s+([.!?])", r"\1", CITATION.sub("", sentence)).strip()):
            bad.append(sentence)
    return bad
