# this_file: tests/test_remote.py
"""Provider wire contracts; live calls are separately recorded under research/301."""

import sys
from types import SimpleNamespace

import pytest

from uubed.models import get_model
from uubed.remote import ApiBackend


class Client:
    calls = []  # noqa: RUF012 - shared test capture
    result = {}  # noqa: RUF012 - shared test response

    def __init__(self, **kwargs):
        self.calls.append(kwargs)

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: self.result)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def fake_http(monkeypatch):
    Client.calls = []
    Client.result = {"data": [{"index": 1, "embedding": [2, 3]}, {"index": 0, "embedding": [1, 2]}]}
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=Client))


@pytest.mark.parametrize(("alias", "provider"), [("jina", "jina"), ("voyage-lite", "voyage"), ("granite", "hf")])
def test_api_when_model_provider_then_exact_schema_and_timeout(alias, provider):
    if provider == "hf":
        Client.result = [[1, 2], [2, 3]]
    backend = ApiBackend(get_model(alias), provider, api_key="test", task="similarity")
    assert backend.encode(["a", "b"], "", 32) == [[1, 2], [2, 3]]
    assert Client.calls[0]["timeout"] == 60.0
    payload = Client.calls[1][1]["json"]
    if provider == "jina":
        assert payload["model"] == "jina-embeddings-v5-text-nano"
        assert payload["task"] == "text-matching"
    elif provider == "voyage":
        assert payload["model"] == "voyage-4-lite"
        assert payload["output_dimension"] == 2048
        assert payload["input_type"] is None
    else:
        assert payload["inputs"] == ["a", "b"]


def test_api_when_cloudflare_then_success_required():
    backend = ApiBackend(get_model("gemma"), "cloudflare", api_key="test", endpoint="https://example.org/model")
    Client.result = {"success": True, "result": {"data": [[1, 2], [2, 3]]}}
    assert backend.encode(["a", "b"], "prefix: ", 32) == [[1, 2], [2, 3]]
    assert Client.calls[-1][1]["json"] == {"text": ["prefix: a", "prefix: b"]}
    Client.result = {"success": False, "errors": [{"message": "no access"}]}
    with pytest.raises(ValueError, match="unsuccessful"):
        backend.encode(["a"], "", 32)


@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"index": 0, "embedding": [1]}],
        [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]}],
        [{"embedding": [1]}, {"embedding": [2]}],
    ],
)
def test_api_when_missing_or_duplicate_rows_then_rejected(data):
    Client.result = {"data": data}
    backend = ApiBackend(get_model("jina"), "jina", api_key="test")
    with pytest.raises(ValueError, match="response"):
        backend.encode(["a", "b"], "", 32)


def test_api_when_nomic_selfhost_then_exact_model_id_and_prefix():
    backend = ApiBackend(get_model("nomic"), "openai", endpoint="http://localhost:8080/v1/embeddings")
    backend.encode(["a", "b"], "search_document: ", 32)
    body = Client.calls[-1][1]["json"]
    assert body["model"] == "nomic-ai/nomic-embed-text-v2-moe"
    assert body["input"] == ["search_document: a", "search_document: b"]


def test_api_when_provider_does_not_serve_model_then_rejected():
    with pytest.raises(ValueError, match="provider"):
        ApiBackend(get_model("nomic"), "voyage", api_key="test")


def test_api_when_missing_endpoint_or_invalid_timeout_then_rejected():
    with pytest.raises(ValueError, match="endpoint"):
        ApiBackend(get_model("nomic"), "openai")
    with pytest.raises(ValueError, match="timeout"):
        ApiBackend(get_model("jina"), "jina", api_key="test", timeout=-1)
