"""Listwise LLM reranking as LitSearch ran it (arXiv 2407.18940, section 4), adapted from RankGPT (Sun et al., 2023)."""
import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import httpx

Model = Callable[[list[dict]], str]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
SYSTEM = "You are RankGPT, an intelligent assistant that can rank papers based on their relevance to a research query."


def build_messages(question: str, candidates: list[int], docs: dict[int, str]) -> list[dict]:
    n = len(candidates)
    papers = "\n\n".join(f"[{i}] {docs[d]}" for i, d in enumerate(candidates, 1))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"I will provide you with {n} papers, each indicated by a numerical identifier []. "
                                    f"Rank the papers based on their relevance to the research query: {question}\n\n{papers}"},
        {"role": "user", "content": f"Research query: {question}\n\nRank the {n} papers above based on their relevance to "
                                    "the research query. All the papers should be included and listed using identifiers, "
                                    "in descending order of relevance. The output format should be [] > [], e.g., "
                                    "[2] > [1]. Only respond with the ranking results, do not say any word or explain."},
    ]


def parse_ranking(reply: str, n: int) -> list[int]:
    """0-based order from a '[3] > [1]' reply; invalid and repeated ids dropped, missing ones appended in order."""
    order = list(dict.fromkeys(int(m) - 1 for m in re.findall(r"\[(\d+)\]", reply) if 1 <= int(m) <= n))
    return order + [i for i in range(n) if i not in set(order)]


def rerank(question: str, ranking: list[int], docs: dict[int, str], model: Model,
           depth: int = 100) -> tuple[list[int], bool]:
    """Reorder the top `depth` papers; the rest keep their place after them. Also says whether the reply parsed."""
    head = ranking[:depth]
    reply = model(build_messages(question, head, docs))
    parsed = bool(re.search(r"\[\d+\]", reply))
    return [head[i] for i in parse_ranking(reply, len(head))] + ranking[depth:], parsed


def cached(model: Model, model_name: str, cache_dir: Path | str) -> Model:
    """Wrap a model so each (model, prompt) is called once and saved with its reply, for reruns and audit."""
    cache = Path(cache_dir)

    def call(messages: list[dict]) -> str:
        key = hashlib.sha256(json.dumps([model_name, messages], sort_keys=True).encode()).hexdigest()
        path = cache / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["reply"]
        reply = model(messages)
        cache.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": model_name, "messages": messages, "reply": reply}))
        return reply

    return call


class ModelFailed(Exception):
    pass


def gemini(client: httpx.Client, key: str, model: str, tries: int = 6, timeout: float = 300.0,
           sleep: Callable[[float], None] = time.sleep) -> Model:
    """One pinned model, no fallback, so a run never mixes rerankers. Waits and retries on errors and empty replies."""

    def call(messages: list[dict]) -> str:
        problem = ""
        for attempt in range(tries):
            if attempt:
                sleep(min(30 * 2 ** (attempt - 1), 300))  # 30 s, 60 s, 120 s, then 300 s; free-tier limits are per minute
            try:
                response = client.post(GEMINI_URL, headers={"Authorization": f"Bearer {key}"}, timeout=timeout,
                                       json={"model": model, "messages": messages, "temperature": 0})
                response.raise_for_status()
                text = response.json()["choices"][0]["message"]["content"]
                if text:
                    return text
                problem = "empty reply"  # thinking models can spend the whole reply on reasoning
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                # Never include the exception text: the request it describes carries the key.
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                problem = f"{type(exc).__name__} {status}"
        raise ModelFailed(f"{model} failed {tries} times, last: {problem}")

    return call
