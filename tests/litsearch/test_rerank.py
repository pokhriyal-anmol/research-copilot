"""The listwise LLM reranker (BENCHMARK.md, reranker stage), checked with a fake model so no test calls an API."""
import json

from copilot.litsearch.rerank import build_messages, cached, parse_ranking, rerank

DOCS = {10: "Title A\n\nAbstract A", 20: "Title B", 30: "Title C\n\nAbstract C"}


def test_candidates_are_numbered_in_the_order_given():
    text = "\n".join(m["content"] for m in build_messages("q", [10, 20, 30], DOCS))
    assert text.index("[1] Title A") < text.index("[2] Title B") < text.index("[3] Title C")


def test_the_question_and_the_answer_format_are_in_the_prompt():
    last = build_messages("which paper did X", [10, 20], DOCS)[-1]["content"]
    assert "which paper did X" in last and "[2] > [1]" in last


def test_abstracts_are_shown():
    assert "Abstract C" in "\n".join(m["content"] for m in build_messages("q", [30], DOCS))


def test_a_clean_ranking_parses():
    assert parse_ranking("[3] > [1] > [2]", 3) == [2, 0, 1]


def test_out_of_range_and_repeated_ids_are_dropped():
    assert parse_ranking("[2] > [9] > [2] > [0] > [1]", 3) == [1, 0, 2]


def test_ids_the_model_left_out_follow_in_the_original_order():
    assert parse_ranking("[3]", 4) == [2, 0, 1, 3]


def test_an_unparseable_reply_keeps_the_original_order():
    assert parse_ranking("I cannot rank these.", 3) == [0, 1, 2]


def test_rerank_reorders_only_the_candidates_and_keeps_the_rest_after_them():
    def model(messages):
        return "[2] > [1]"

    assert rerank("q", [10, 20, 30], DOCS, model, depth=2) == ([20, 10, 30], True)


def test_rerank_reports_a_reply_it_could_not_parse():
    assert rerank("q", [10, 20], DOCS, lambda m: "no idea", depth=2) == ([10, 20], False)


def test_a_cached_reply_is_reused_without_calling_the_model(tmp_path):
    calls = []

    def model(messages):
        calls.append(messages)
        return "[1]"

    first = cached(model, "m", tmp_path)
    first([{"role": "user", "content": "x"}])
    again = cached(lambda m: "should not run", "m", tmp_path)
    assert again([{"role": "user", "content": "x"}]) == "[1]" and len(calls) == 1


def test_the_cache_is_keyed_by_model_and_prompt(tmp_path):
    cached(lambda m: "[1]", "m1", tmp_path)([{"role": "user", "content": "x"}])
    assert cached(lambda m: "[2]", "m2", tmp_path)([{"role": "user", "content": "x"}]) == "[2]"
    assert cached(lambda m: "[3]", "m1", tmp_path)([{"role": "user", "content": "y"}]) == "[3]"


def test_the_cache_stores_the_prompt_with_the_reply(tmp_path):
    # So every reply in a result can be audited later.
    cached(lambda m: "[1]", "m", tmp_path)([{"role": "user", "content": "x"}])
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["model"] == "m" and saved["reply"] == "[1]" and saved["messages"][0]["content"] == "x"


# The chat caller: one pinned model, no fallback, retries with waits.
import httpx
import pytest

from copilot.litsearch.rerank import GEMINI_URL, ModelFailed, chat


def transport(*replies):
    calls = []

    def handle(request):
        calls.append(request)
        status, content = replies[min(len(calls), len(replies)) - 1]
        body = {"choices": [{"message": {"content": content}}]} if status == 200 else {"error": "x"}
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handle), calls


def test_the_pinned_model_is_sent_at_temperature_zero():
    t, calls = transport((200, "[1]"))
    chat(httpx.Client(transport=t), "secret", "gemini-x", sleep=lambda s: None)([{"role": "user", "content": "q"}])
    body = json.loads(calls[0].content)
    assert body["model"] == "gemini-x" and body["temperature"] == 0


def test_a_rate_limit_is_retried_after_a_wait():
    t, calls = transport((429, None), (200, "[1]"))
    waits = []
    reply = chat(httpx.Client(transport=t), "secret", "m", sleep=waits.append)([{"role": "user", "content": "q"}])
    assert reply == "[1]" and len(calls) == 2 and waits


def test_an_empty_reply_is_retried_not_returned():
    t, _ = transport((200, ""), (200, "[2]"))
    assert chat(httpx.Client(transport=t), "secret", "m", sleep=lambda s: None)([]) == "[2]"


def test_it_gives_up_after_the_last_try_without_showing_the_key():
    t, calls = transport((500, None))
    with pytest.raises(ModelFailed) as info:
        chat(httpx.Client(transport=t), "secret-key", "m", tries=3, sleep=lambda s: None)([])
    assert len(calls) == 3 and "secret-key" not in str(info.value)


def test_gemini_is_the_default_endpoint():
    t, calls = transport((200, "[1]"))
    chat(httpx.Client(transport=t), "secret", "m", sleep=lambda s: None)([])
    assert str(calls[0].url) == GEMINI_URL


def test_another_openai_compatible_endpoint_can_be_pinned():
    t, calls = transport((200, "[1]"))
    url = "https://openrouter.ai/api/v1/chat/completions"
    chat(httpx.Client(transport=t), "secret", "m", url=url, sleep=lambda s: None)([])
    assert str(calls[0].url) == url
