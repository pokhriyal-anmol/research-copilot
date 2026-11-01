import httpx
import pytest

from copilot.llm import AllProvidersFailed, complete

GOOD = {"choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    for name in ("GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(name, "test")


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_falls_back_when_first_provider_is_rate_limited():
    def handler(request):
        limited = "googleapis" in request.url.host
        return httpx.Response(429) if limited else httpx.Response(200, json=GOOD)

    result = complete(client(handler), [])
    assert (result.provider, result.input_tokens) == ("openrouter", 3)


def test_falls_back_on_malformed_body():
    def handler(request):
        bad = "googleapis" in request.url.host
        return httpx.Response(200, json={"choices": []} if bad else GOOD)

    assert complete(client(handler), []).provider == "openrouter"


def test_skips_provider_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        return httpx.Response(200, json=GOOD)

    complete(client(handler), [])
    assert hosts == ["openrouter.ai"]


def test_raises_when_all_fail():
    with pytest.raises(AllProvidersFailed):
        complete(client(lambda request: httpx.Response(500)), [])


def test_falls_back_on_empty_content():
    empty = {"choices": [{"message": {"content": None}}]}

    def handler(request):
        bad = "googleapis" in request.url.host
        return httpx.Response(200, json=empty if bad else GOOD)

    assert complete(client(handler), []).provider == "openrouter"
