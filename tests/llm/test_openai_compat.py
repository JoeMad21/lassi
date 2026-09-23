"""Tests for the OpenAI-compatible backend in lassi/llm/openai_compat.py (P0.4).

The backend talks to an OpenAI-compatible server (bible Model Serving): it
confirms that `/v1/models` lists the expected id before the first chat
request (Serving Rules; the default port is 8123), sends messages and sampling
unchanged, and returns the text with the server's token counts, never
fabricated ones. API keys come only from the environment variable the recipe
names (Agent Rule 12) and never reach logs, repr, errors, or the written
trial. Every request goes to the local stub server from conftest.py on
127.0.0.1; no test touches the network. Token counts, model ids, and entry
fields below are stub fixture values, not measurements.
"""

from __future__ import annotations

import html
import inspect
import json
import logging
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from lassi import llm
from lassi.core import record, store
from lassi.core.interfaces import Completion, Message, Sampling
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.llm import _http

if TYPE_CHECKING:
    from conftest import StubServer

DEFAULT_BASE_URL = llm.openai_compat.DEFAULT_BASE_URL
OpenAICompatBackend = llm.openai_compat.OpenAICompatBackend
ServingError = llm.ServingError
model_info = llm.model_info

MODEL_ID = "fixture/coder-a"
MODELS_PATH = "/v1/models"
CHAT_PATH = "/v1/chat/completions"
SAMPLING = Sampling(temperature=0.35, top_p=0.85, max_tokens=777)
E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"

PROMPT_MARKER = "prompt-marker-5b1e"
REPLY_MARKER = "reply-marker-9c3d"
REPLY_TEXT = f"```cuda\n// FILE: main.cu\nint {REPLY_MARKER};\n```\n"
MESSAGES = (
    Message(role="system", content="You translate code."),
    Message(role="user", content=f"Translate main.cpp to CUDA. {PROMPT_MARKER}\n"),
)

KEY_ENV = "LASSI_TEST_API_KEY"
SENTINEL_KEY = "lassi-test-sentinel-key-2f7d41c9"
ROTATED_KEY = "lassi-test-rotated-key-80be5a13"
# Variables an OpenAI-style client might fall back to; the backend must read only the one api_key_env names.
AMBIENT_KEY_VARS = ("OPENAI_API_KEY", "API_KEY")
# The stub replaces this marker with the Authorization header it received, as some servers do in errors.
ECHO = "<<authorization>>"
ECHO_AUTH = (ECHO, "Authorization")
REDIRECT_TARGET = "/v1/elsewhere"

PROXY_VARS = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
NO_PROXY_VARS = ("NO_PROXY", "no_proxy")

TRIAL_ID = "fixture-project/fixture-arm/fixture-suite/omp-cuda/fixture-item/run01"
RECIPE_HASH = "0123456789abcdef" * 4

_DELETE = object()


# ---------------------------------------------------------------------------
# Helpers


def chat_body(content: Any = REPLY_TEXT, *, prompt_tokens: Any = 11, completion_tokens: Any = 7) -> dict[str, Any]:
    """Return an OpenAI-style chat completion reply."""
    return {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": 18},
    }


def chat_edit(path: tuple[str | int, ...], value: Any = _DELETE) -> bytes:
    """Return the encoded chat reply with the value at `path` replaced, or deleted when no value is given."""
    body: Any = chat_body()
    *parents, last = path
    target = body
    for key in parents:
        target = target[key]
    if value is _DELETE:
        del target[last]
    else:
        target[last] = value
    return json.dumps(body).encode("utf-8")


def reply_without_usage(padding: str = "", **extra: Any) -> bytes:
    """Return an encoded chat reply with no usage block, led by a `padding` field when one is given."""
    body = {"padding": padding} if padding else {}
    body.update(chat_body(), **extra)
    del body["usage"]
    return json.dumps(body).encode("utf-8")


def nested(depth: int) -> Any:
    """Return a JSON value nesting `depth` arrays and objects, alternately, around a null."""
    value: Any = None
    for level in range(depth):
        value = [value] if level % 2 else {"k": value}
    return value


def serve_ready(stub: StubServer, reply: dict[str, Any] | None = None) -> None:
    """Serve a model list that includes MODEL_ID and a chat reply (chat_body() unless given)."""
    stub.serve_models(["other/model", MODEL_ID])
    stub.reply("POST", CHAT_PATH, json_body=chat_body() if reply is None else reply)


def make_backend(stub: StubServer, **kwargs: Any) -> OpenAICompatBackend:
    """Return a backend for MODEL_ID pointed at the stub's /v1 base URL."""
    return OpenAICompatBackend(MODEL_ID, base_url=f"{stub.url}/v1", **kwargs)


def set_proxy(monkeypatch: pytest.MonkeyPatch, proxy_url: str) -> None:
    """Point every proxy variable at `proxy_url` and remove any no-proxy list."""
    for name in NO_PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in PROXY_VARS:
        monkeypatch.setenv(name, proxy_url)


def error_texts(error: BaseException) -> list[str]:
    """Return every text an error shows: str, repr, args, and the formatted traceback with its chain."""
    texts = [str(error), repr(error), repr(error.args)]
    texts.append("".join(traceback.format_exception(type(error), error, error.__traceback__)))
    return texts


def assert_path_only(text: str, stub: StubServer) -> None:
    """Fail when `text` shows the stub's scheme, host, or port: messages and logs name the URL path only."""
    for part in ("http://", "127.0.0.1", f":{stub.port}"):
        assert part not in text, (part, text)


def make_trial(model: record.ModelInfo, response_text: str) -> record.Trial:
    """Return a one-attempt Trial whose model field is `model`."""
    attempt = record.Attempt(
        index=0, response_text=response_text, files={"main.cu": "int main() { return 0; }\n"}, stage_reached="S0"
    )
    bench = record.BenchItem(suite="fixture-suite", item="fixture-item", split="eval", direction="omp-cuda")
    return record.Trial(trial_id=TRIAL_ID, recipe_hash=RECIPE_HASH, bench_item=bench, model=model, attempts=[attempt])


@pytest.fixture
def backend(stub_server: StubServer) -> OpenAICompatBackend:
    """Return a backend without an API key, with the stub serving MODEL_ID and a chat reply."""
    serve_ready(stub_server)
    return make_backend(stub_server)


@pytest.fixture
def keyed_backend(stub_server: StubServer, monkeypatch: pytest.MonkeyPatch) -> OpenAICompatBackend:
    """Return a backend whose API key comes from KEY_ENV, which holds the sentinel key."""
    monkeypatch.setenv(KEY_ENV, SENTINEL_KEY)
    serve_ready(stub_server)
    return make_backend(stub_server, api_key_env=KEY_ENV)


# ---------------------------------------------------------------------------
# Registration and shape


def test_registered_as_llm_backend_openai_compat() -> None:
    entry = DEFAULT_REGISTRY.get("LLMBackend", "openai_compat")
    assert entry.factory is OpenAICompatBackend
    assert entry.capabilities == frozenset({"chat", "model_check"})
    assert llm.OpenAICompatBackend is OpenAICompatBackend


def test_name_capabilities_and_default_port() -> None:
    assert OpenAICompatBackend.name == "openai_compat"
    assert isinstance(OpenAICompatBackend.capabilities, frozenset)
    assert OpenAICompatBackend.capabilities == frozenset({"chat", "model_check"})
    assert DEFAULT_BASE_URL == "http://127.0.0.1:8123/v1"


def test_constructor_signature() -> None:
    params = [(p.name, p.kind, p.default) for p in inspect.signature(OpenAICompatBackend).parameters.values()]
    assert params == [
        ("model_id", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("base_url", inspect.Parameter.KEYWORD_ONLY, DEFAULT_BASE_URL),
        ("api_key_env", inspect.Parameter.KEYWORD_ONLY, None),
        ("timeout_s", inspect.Parameter.KEYWORD_ONLY, 600.0),
    ]


def test_no_parameter_takes_a_key_value() -> None:
    assert [name for name in inspect.signature(OpenAICompatBackend).parameters if "key" in name.lower()] == [
        "api_key_env"
    ]
    for attr in ("complete", "check"):
        params = inspect.signature(getattr(OpenAICompatBackend, attr)).parameters
        assert [name for name in params if "key" in name.lower()] == [], attr


def test_repr_shows_class_model_id_and_base_url(stub_server: StubServer) -> None:
    text = repr(make_backend(stub_server))
    assert "OpenAICompatBackend" in text and MODEL_ID in text and f"{stub_server.url}/v1" in text, text


def test_repr_shows_nothing_else(stub_server: StubServer) -> None:
    text = repr(make_backend(stub_server, api_key_env=KEY_ENV, timeout_s=12.5))
    assert KEY_ENV not in text and "timeout" not in text and "12.5" not in text, text


def test_construction_sends_nothing(stub_server: StubServer) -> None:
    make_backend(stub_server)
    assert stub_server.requests == []


# ---------------------------------------------------------------------------
# complete


def test_complete_returns_text_and_token_counts(stub_server: StubServer, backend: OpenAICompatBackend) -> None:
    completion = backend.complete(MESSAGES, SAMPLING)
    assert isinstance(completion, Completion)
    assert completion == Completion(text=REPLY_TEXT, prompt_tokens=11, completion_tokens=7)
    assert stub_server.calls() == [("GET", MODELS_PATH), ("POST", CHAT_PATH)]


@pytest.mark.parametrize(
    "sampling",
    [Sampling(0.35, 0.85, 777), Sampling(0.0, 1.0, 16384), Sampling(1.25, 0.05, 1)],
    ids=["typical", "greedy", "edge"],
)
def test_chat_request_carries_messages_and_sampling_exactly(
    stub_server: StubServer, backend: OpenAICompatBackend, sampling: Sampling
) -> None:
    backend.complete(MESSAGES, sampling)
    (chat,) = stub_server.requests_to("POST", CHAT_PATH)
    assert chat.body == {
        "model": MODEL_ID,
        "messages": [{"role": m.role, "content": m.content} for m in MESSAGES],
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "max_tokens": sampling.max_tokens,
        "stream": False,
    }
    assert type(chat.body["temperature"]) is float and type(chat.body["top_p"]) is float
    assert type(chat.body["max_tokens"]) is int
    assert chat.body["stream"] is False
    assert (chat.header("Content-Type") or "").split(";")[0].strip() == "application/json"


def test_messages_are_sent_unchanged(stub_server: StubServer, backend: OpenAICompatBackend) -> None:
    # Never truncate or rewrite (bible Serving Rules): a long prompt, edge whitespace, an empty turn, non-ASCII.
    long_text = "".join(f"line {i}: {'z' * 60}\n" for i in range(2500)) + "  trailing spaces  \n\n"
    messages = [
        Message("system", "  leading spaces"),
        Message("user", long_text),
        Message("assistant", ""),
        Message("user", f"caf{E_ACUTE}\r\nend"),
    ]
    backend.complete(messages, SAMPLING)
    (chat,) = stub_server.requests_to("POST", CHAT_PATH)
    assert chat.body["messages"] == [{"role": m.role, "content": m.content} for m in messages]


def test_zero_counts_and_empty_content_are_kept(stub_server: StubServer) -> None:
    serve_ready(stub_server, chat_body("", prompt_tokens=0, completion_tokens=0))
    assert make_backend(stub_server).complete(MESSAGES, SAMPLING) == Completion(
        text="", prompt_tokens=0, completion_tokens=0
    )


@pytest.mark.parametrize("status", [201, 203])
def test_any_2xx_chat_status_is_accepted(stub_server: StubServer, status: int) -> None:
    stub_server.serve_models([MODEL_ID])
    stub_server.reply("POST", CHAT_PATH, status=status, json_body=chat_body())
    assert make_backend(stub_server).complete(MESSAGES, SAMPLING) == Completion(REPLY_TEXT, 11, 7)


def test_the_first_choice_is_returned(stub_server: StubServer) -> None:
    body = chat_body()
    second = {"index": 1, "message": {"role": "assistant", "content": "second choice"}, "finish_reason": "stop"}
    body["choices"].append(second)
    serve_ready(stub_server, body)
    assert make_backend(stub_server).complete(MESSAGES, SAMPLING).text == REPLY_TEXT


def test_a_json_reply_with_a_byte_order_mark_is_read(stub_server: StubServer) -> None:
    stub_server.serve_models([MODEL_ID])
    stub_server.reply("POST", CHAT_PATH, raw=b"\xef\xbb\xbf" + json.dumps(chat_body()).encode("utf-8"))
    assert make_backend(stub_server).complete(MESSAGES, SAMPLING) == Completion(REPLY_TEXT, 11, 7)


BAD_CHAT_REPLIES = [
    pytest.param(b'{"choices": [', id="malformed-json"),
    pytest.param(b"", id="empty-body"),
    pytest.param(b"[]", id="json-array"),
    pytest.param(chat_edit(("usage",)), id="no-usage"),
    pytest.param(chat_edit(("usage",), None), id="null-usage"),
    pytest.param(chat_edit(("usage", "prompt_tokens")), id="no-prompt-tokens"),
    pytest.param(chat_edit(("usage", "completion_tokens")), id="no-completion-tokens"),
    pytest.param(chat_edit(("usage", "prompt_tokens"), None), id="null-prompt-tokens"),
    pytest.param(chat_edit(("usage", "completion_tokens"), None), id="null-completion-tokens"),
    pytest.param(chat_edit(("choices", 0, "message", "content"), None), id="null-content"),
    pytest.param(chat_edit(("choices", 0, "message", "content")), id="no-content"),
    pytest.param(chat_edit(("choices", 0, "message")), id="no-message"),
    pytest.param(chat_edit(("choices",), []), id="empty-choices"),
    pytest.param(chat_edit(("choices",)), id="no-choices"),
    pytest.param(chat_edit(("usage", "prompt_tokens"), "11"), id="string-count"),
    pytest.param(chat_edit(("usage", "prompt_tokens"), 11.0), id="float-count"),
    pytest.param(chat_edit(("usage", "completion_tokens"), -1), id="negative-count"),
    pytest.param(chat_edit(("usage", "completion_tokens"), True), id="boolean-count"),
    pytest.param(chat_edit(("choices", 0, "message", "content"), ["text"]), id="list-content"),
    pytest.param(chat_edit(("choices", 0, "message", "content"), {"text": "x"}), id="object-content"),
]


@pytest.mark.parametrize("raw", BAD_CHAT_REPLIES)
def test_a_bad_chat_reply_raises_serving_error(stub_server: StubServer, raw: bytes) -> None:
    stub_server.serve_models([MODEL_ID])
    stub_server.reply("POST", CHAT_PATH, raw=raw)
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).complete(MESSAGES, SAMPLING)
    # The message names the URL path and the status, even for a 2xx reply whose body is wrong.
    assert f"POST {CHAT_PATH} returned HTTP 200" in str(info.value), str(info.value)


@pytest.mark.parametrize("status", [400, 401, 404, 500, 503])
def test_a_non_2xx_chat_status_raises_naming_path_and_status(stub_server: StubServer, status: int) -> None:
    # Even a well-formed completion body is refused when the status is not 2xx.
    stub_server.serve_models([MODEL_ID])
    stub_server.reply("POST", CHAT_PATH, status=status, json_body=chat_body())
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).complete(MESSAGES, SAMPLING)
    message = str(info.value)
    assert f"POST {CHAT_PATH} returned HTTP {status}" in message, message
    assert_path_only(message, stub_server)


LONG_BODY = "ERR-START " + "x" * 2000 + " ERR-END"


@pytest.mark.parametrize(
    ("status", "raw", "content_type"),
    [
        pytest.param(500, LONG_BODY.encode("ascii"), "text/plain", id="status-500"),
        pytest.param(200, LONG_BODY.encode("ascii"), "text/plain", id="malformed-200"),
        pytest.param(200, reply_without_usage(LONG_BODY), "application/json", id="json-without-usage-200"),
    ],
)
def test_an_error_quotes_at_most_500_characters_of_the_body(
    stub_server: StubServer, status: int, raw: bytes, content_type: str
) -> None:
    stub_server.serve_models([MODEL_ID])
    stub_server.reply("POST", CHAT_PATH, status=status, raw=raw, content_type=content_type)
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).complete(MESSAGES, SAMPLING)
    message = str(info.value)
    assert "ERR-START" in message, message
    assert "ERR-END" not in message
    assert "x" * 491 not in message  # 500 characters of the body hold "ERR-START " and at most 490 x's
    assert f"(first 500 of {len(raw)} characters)" in message, message


def test_a_transport_error_quotes_at_most_500_characters(stub_server: StubServer) -> None:
    # A reply that is not HTTP: http.client puts the whole status line, server text, into its error.
    stub_server.serve_models([MODEL_ID])
    stub_server.reply("POST", CHAT_PATH, raw_response=("GARBAGE " + "x" * 5000 + "\r\n\r\n").encode("ascii"))
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).complete(MESSAGES, SAMPLING)
    message = str(info.value)
    assert f"POST {CHAT_PATH} got no reply" in message and "GARBAGE" in message, message
    assert "x" * 493 not in message and "(first 500 of" in message, message
    assert len(message) < 700, len(message)


def test_an_unreachable_server_raises_serving_error(dead_port: int) -> None:
    backend = OpenAICompatBackend(MODEL_ID, base_url=f"http://127.0.0.1:{dead_port}/v1")
    with pytest.raises(ServingError):
        backend.complete(MESSAGES, SAMPLING)


@pytest.mark.parametrize(("method", "path"), [("GET", MODELS_PATH), ("POST", CHAT_PATH)], ids=["models", "chat"])
def test_the_timeout_applies(stub_server: StubServer, method: str, path: str) -> None:
    serve_ready(stub_server)
    delayed = {"data": [{"id": MODEL_ID}]} if method == "GET" else chat_body()
    stub_server.reply(method, path, json_body=delayed, delay_s=30.0)
    backend = make_backend(stub_server, timeout_s=0.3)
    start = time.monotonic()
    with pytest.raises(ServingError) as info:
        backend.complete(MESSAGES, SAMPLING)
    assert time.monotonic() - start < 10.0
    message = str(info.value)
    assert f"{method} {path} got no reply" in message and "timed out" in message, message
    assert len(stub_server.requests_to(method, path)) == 1


@pytest.mark.parametrize("suffix", ["/v1", "/v1/", "/v1//"], ids=["no-slash", "trailing-slash", "two-slashes"])
def test_base_url_trailing_slash_is_removed(stub_server: StubServer, suffix: str) -> None:
    serve_ready(stub_server)
    backend = OpenAICompatBackend(MODEL_ID, base_url=stub_server.url + suffix)
    assert backend.complete(MESSAGES, SAMPLING).text == REPLY_TEXT
    assert [seen.target for seen in stub_server.requests] == [MODELS_PATH, CHAT_PATH]


def test_every_request_is_sent_as_json(stub_server: StubServer, backend: OpenAICompatBackend) -> None:
    backend.complete(MESSAGES, SAMPLING)
    types = [(seen.header("Content-Type") or "").split(";")[0].strip() for seen in stub_server.requests]
    assert types == ["application/json", "application/json"]


# ---------------------------------------------------------------------------
# base_url: a plain http(s) URL that request paths are appended to


@pytest.mark.parametrize(
    "suffix", ["/v1?q=7c1d", "/v1?", "/v1#f7c1d", "/v1/#"], ids=["query", "empty-query", "fragment", "empty-fragment"]
)
def test_a_base_url_with_a_query_or_fragment_is_refused(stub_server: StubServer, suffix: str) -> None:
    # "<base_url>/models" would land inside the query or the fragment and never reach the models endpoint.
    with pytest.raises(ValueError) as info:
        OpenAICompatBackend(MODEL_ID, base_url=stub_server.url + suffix)
    assert "query" in str(info.value) and "7c1d" not in str(info.value), str(info.value)
    assert stub_server.requests == []


@pytest.mark.parametrize("login", ["svcuser:pw-3a9c7e@", "pw-3a9c7e@", "svcuser:pw-3a9c7e:x@"])
def test_a_base_url_with_a_login_is_refused_without_quoting_it(stub_server: StubServer, login: str) -> None:
    with pytest.raises(ValueError) as info:
        OpenAICompatBackend(MODEL_ID, base_url=f"http://{login}127.0.0.1:{stub_server.port}/v1")
    assert "password" in str(info.value)
    assert "pw-3a9c7e" not in str(info.value) and "pw-3a9c7e" not in repr(info.value)
    assert stub_server.requests == []


@pytest.mark.parametrize(
    "base_url", ["ftp://127.0.0.1:8123/v1", "127.0.0.1:8123/v1", "localhost:8123/v1", "http:///v1"]
)
def test_a_base_url_that_is_not_http_with_a_host_is_refused(base_url: str) -> None:
    with pytest.raises(ValueError):
        OpenAICompatBackend(MODEL_ID, base_url=base_url)


@pytest.mark.parametrize("port", ["port-7c1d", "70000", "-1"], ids=["not-a-number", "too-large", "negative"])
def test_a_base_url_with_a_bad_port_is_refused_without_quoting_it(port: str) -> None:
    # urllib's own error quotes the port text; the refusal happens at construction and never shows it.
    with pytest.raises(ValueError) as info:
        OpenAICompatBackend(MODEL_ID, base_url=f"http://127.0.0.1:{port}/v1")
    assert "port" in str(info.value), str(info.value)
    for text in error_texts(info.value):
        assert port not in text, text


@pytest.mark.parametrize("timeout_s", [0, -1.0, float("nan"), float("inf"), True, "10", None])
def test_a_timeout_that_is_not_a_positive_number_is_refused(stub_server: StubServer, timeout_s: Any) -> None:
    with pytest.raises(ValueError) as info:
        make_backend(stub_server, timeout_s=timeout_s)
    assert "timeout_s" in str(info.value), str(info.value)


def test_a_whole_number_timeout_is_accepted(stub_server: StubServer) -> None:
    serve_ready(stub_server)
    assert make_backend(stub_server, timeout_s=5).complete(MESSAGES, SAMPLING).text == REPLY_TEXT


# ---------------------------------------------------------------------------
# Redirects are refused, never followed


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirected_chat_request_is_refused(stub_server: StubServer, status: int) -> None:
    # urllib would turn a 301/302/303 POST into a GET without the prompt; the reply would not answer the prompt.
    serve_ready(stub_server)
    stub_server.reply("POST", CHAT_PATH, status=status, headers={"Location": f"{stub_server.url}{REDIRECT_TARGET}"})
    stub_server.reply("GET", REDIRECT_TARGET, json_body=chat_body())
    stub_server.reply("POST", REDIRECT_TARGET, json_body=chat_body())
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).complete(MESSAGES, SAMPLING)
    message = str(info.value)
    assert f"POST {CHAT_PATH} returned HTTP {status}" in message and "redirect" in message, message
    assert stub_server.calls() == [("GET", MODELS_PATH), ("POST", CHAT_PATH)]


def test_a_redirected_model_list_is_refused(stub_server: StubServer) -> None:
    # A model list from another server must not vouch for the server that gets the chat request.
    stub_server.reply("GET", MODELS_PATH, status=302, headers={"Location": f"{stub_server.url}{REDIRECT_TARGET}"})
    stub_server.serve_models([MODEL_ID], path=REDIRECT_TARGET)
    stub_server.reply("POST", CHAT_PATH, json_body=chat_body())
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).complete(MESSAGES, SAMPLING)
    assert f"GET {MODELS_PATH} returned HTTP 302" in str(info.value), str(info.value)
    assert stub_server.calls() == [("GET", MODELS_PATH)]


# ---------------------------------------------------------------------------
# check: the /v1/models gate


def test_check_returns_the_served_entry(stub_server: StubServer) -> None:
    # vLLM-style servers put fields such as max_model_len in the entry; check() hands the whole entry back.
    entry = {"id": MODEL_ID, "object": "model", "owned_by": "fixture", "max_model_len": 32768}
    stub_server.serve_models(["other/model", entry])
    assert make_backend(stub_server).check() == entry
    assert stub_server.calls() == [("GET", MODELS_PATH)]


def test_check_is_cached(stub_server: StubServer, backend: OpenAICompatBackend) -> None:
    first = backend.check()
    assert backend.check() == first
    backend.complete(MESSAGES, SAMPLING)
    backend.complete(MESSAGES, SAMPLING)
    assert stub_server.calls() == [("GET", MODELS_PATH), ("POST", CHAT_PATH), ("POST", CHAT_PATH)]


def test_complete_checks_once(stub_server: StubServer, backend: OpenAICompatBackend) -> None:
    for _ in range(3):
        backend.complete(MESSAGES, SAMPLING)
    assert stub_server.calls() == [("GET", MODELS_PATH)] + [("POST", CHAT_PATH)] * 3


@pytest.mark.parametrize("first", ["missing-id", "status-503"])
def test_a_failed_check_is_not_cached(stub_server: StubServer, first: str) -> None:
    if first == "missing-id":
        stub_server.serve_models(["other/model"])
    else:
        stub_server.reply("GET", MODELS_PATH, status=503, json_body={"error": "loading"})
    backend = make_backend(stub_server)
    with pytest.raises(ServingError):
        backend.check()
    serve_ready(stub_server)
    assert backend.check()["id"] == MODEL_ID
    assert backend.complete(MESSAGES, SAMPLING).text == REPLY_TEXT
    assert stub_server.calls() == [("GET", MODELS_PATH), ("GET", MODELS_PATH), ("POST", CHAT_PATH)]


def test_complete_refuses_a_missing_id_before_any_chat_request(stub_server: StubServer) -> None:
    stub_server.serve_models(["zeta/model", "alpha/model", "mid/model"])
    stub_server.reply("POST", CHAT_PATH, json_body=chat_body())
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).complete(MESSAGES, SAMPLING)
    assert stub_server.calls() == [("GET", MODELS_PATH)]
    message = str(info.value)
    assert MODEL_ID in message, message
    positions = [message.find(served) for served in ("alpha/model", "mid/model", "zeta/model")]
    assert -1 not in positions and positions == sorted(positions), message


def test_check_refuses_naming_expected_and_served_ids(stub_server: StubServer) -> None:
    stub_server.serve_models(["zeta/model", "alpha/model"])
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).check()
    message = str(info.value)
    assert MODEL_ID in message and "alpha/model" in message and "zeta/model" in message, message
    assert message.index("alpha/model") < message.index("zeta/model"), message


def test_a_long_served_id_list_is_cut_and_counted(stub_server: StubServer) -> None:
    ids = [f"fixture/model-{index:05d}-{'z' * 60}" for index in range(3000)]
    stub_server.serve_models(ids)
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).check()
    message = str(info.value)
    assert len(message) < 1200, len(message)
    shown = [served for served in sorted(ids) if served in message]
    assert shown and shown == sorted(ids)[: len(shown)], message
    assert message.endswith(f"(and {len(ids) - len(shown)} more)"), message
    assert f"'{MODEL_ID}'" in message.partition("served ids")[0], message


def test_one_huge_served_id_is_cut(stub_server: StubServer) -> None:
    stub_server.serve_models(["h" * 100_000, "other/model"])
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).check()
    message = str(info.value)
    assert len(message) < 1200, len(message)
    assert "(first 500 of 100000 characters)" in message and message.endswith("(and 1 more)"), message


def test_a_too_deeply_nested_entry_is_refused_and_not_cached(stub_server: StubServer) -> None:
    # Deep enough for json.loads but not for copy.deepcopy: check() must raise ServingError, not RecursionError.
    deep = f'{{"data": [{{"id": "{MODEL_ID}", "x": {"[" * 600}{"]" * 600}}}]}}'
    stub_server.reply("GET", MODELS_PATH, raw=deep.encode("ascii"))
    backend = make_backend(stub_server)
    for _ in range(2):
        with pytest.raises(ServingError) as info:
            backend.check()
        message = str(info.value)
        assert f"GET {MODELS_PATH} returned HTTP 200" in message and "levels deep" in message, message
    serve_ready(stub_server)
    assert backend.complete(MESSAGES, SAMPLING).text == REPLY_TEXT
    assert stub_server.calls() == [("GET", MODELS_PATH)] * 3 + [("POST", CHAT_PATH)]


@pytest.mark.parametrize(("depth", "accepted"), [(_http.MAX_ENTRY_DEPTH - 1, True), (_http.MAX_ENTRY_DEPTH, False)])
def test_entry_nesting_is_bounded(stub_server: StubServer, depth: int, accepted: bool) -> None:
    # The entry itself is the first level, so an entry holding `depth` nested values has depth + 1 levels.
    entry = {"id": MODEL_ID, "object": "model", "x": nested(depth)}
    stub_server.serve_models([entry])
    backend = make_backend(stub_server)
    if accepted:
        assert backend.check() == entry
    else:
        with pytest.raises(ServingError):
            backend.check()


@pytest.mark.parametrize(
    "served",
    [["fixture/coder-a-instruct"], ["FIXTURE/CODER-A"], ["fixture/coder"], ["fixture/coder-a "], []],
    ids=["longer", "other-case", "prefix", "trailing-space", "none"],
)
def test_check_needs_an_exact_id(stub_server: StubServer, served: list[str]) -> None:
    stub_server.serve_models(served)
    stub_server.reply("POST", CHAT_PATH, json_body=chat_body())
    with pytest.raises(ServingError):
        make_backend(stub_server).complete(MESSAGES, SAMPLING)
    assert stub_server.requests_to("POST", CHAT_PATH) == []


@pytest.mark.parametrize(
    ("status", "raw"),
    [
        pytest.param(404, b'{"error": "not found"}', id="status-404"),
        pytest.param(500, json.dumps({"data": [{"id": MODEL_ID}]}).encode(), id="status-500-listing-the-id"),
        pytest.param(200, b"<html>not json</html>", id="malformed-json"),
        pytest.param(200, b'{"object": "list"}', id="no-data"),
        pytest.param(200, b"[]", id="json-array"),
        pytest.param(200, json.dumps({"data": MODEL_ID}).encode(), id="data-not-a-list"),
        pytest.param(200, json.dumps({"data": [{"id": 7}]}).encode(), id="id-not-a-string"),
    ],
)
def test_a_bad_models_reply_raises_before_any_chat_request(stub_server: StubServer, status: int, raw: bytes) -> None:
    stub_server.reply("GET", MODELS_PATH, status=status, raw=raw)
    stub_server.reply("POST", CHAT_PATH, json_body=chat_body())
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).complete(MESSAGES, SAMPLING)
    assert stub_server.requests_to("POST", CHAT_PATH) == []
    assert f"GET {MODELS_PATH} returned HTTP {status}" in str(info.value), str(info.value)


# ---------------------------------------------------------------------------
# Proxies: loopback traffic never goes through a proxy


def test_loopback_requests_ignore_the_proxy_environment(
    stub_server: StubServer, monkeypatch: pytest.MonkeyPatch, dead_port: int
) -> None:
    set_proxy(monkeypatch, f"http://127.0.0.1:{dead_port}")
    serve_ready(stub_server)
    backend = make_backend(stub_server)
    assert backend.complete(MESSAGES, SAMPLING).text == REPLY_TEXT
    assert [seen.target for seen in stub_server.requests] == [MODELS_PATH, CHAT_PATH]


def test_other_hosts_use_the_proxy_environment(stub_server: StubServer, monkeypatch: pytest.MonkeyPatch) -> None:
    # The stub acts as the proxy; the .invalid name is never resolved (conftest refuses such lookups).
    set_proxy(monkeypatch, stub_server.url)
    serve_ready(stub_server)
    backend = OpenAICompatBackend(MODEL_ID, base_url="http://fixture-host.invalid:8123/v1")
    assert backend.complete(MESSAGES, SAMPLING).text == REPLY_TEXT
    base = "http://fixture-host.invalid:8123"
    assert [seen.target for seen in stub_server.requests] == [f"{base}{MODELS_PATH}", f"{base}{CHAT_PATH}"]


def opener_proxies(url: str) -> list[dict[str, str]]:
    """Return the proxy table of every ProxyHandler that takes part in the opener the transport picks for `url`."""
    opener = _http._opener_for(url)
    return [handler.proxies for handler in opener.handlers if isinstance(handler, urllib.request.ProxyHandler)]


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
def test_every_loopback_host_gets_an_opener_without_proxies(monkeypatch: pytest.MonkeyPatch, host: str) -> None:
    # Checked on the opener, so no IPv6 listener and no slow localhost fallback is needed. urllib leaves an
    # empty ProxyHandler out of the opener, so no proxy handler takes part at all.
    set_proxy(monkeypatch, "http://fixture-proxy.invalid:3128")
    assert opener_proxies(f"http://{host}:8123/v1/models") == []


def test_a_remote_host_gets_an_opener_with_the_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    set_proxy(monkeypatch, "http://fixture-proxy.invalid:3128")
    (proxies,) = opener_proxies("http://fixture-host.invalid:8123/v1/models")
    assert proxies.get("http") == "http://fixture-proxy.invalid:3128", proxies


# ---------------------------------------------------------------------------
# API keys (Agent Rule 12)


def test_api_key_is_sent_as_a_bearer_header(stub_server: StubServer, keyed_backend: OpenAICompatBackend) -> None:
    keyed_backend.complete(MESSAGES, SAMPLING)
    assert [seen.header("Authorization") for seen in stub_server.requests] == [f"Bearer {SENTINEL_KEY}"] * 2


def test_no_authorization_header_without_api_key_env(
    stub_server: StubServer, backend: OpenAICompatBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No fallback to a variable the recipe did not name, such as OPENAI_API_KEY.
    for name in (KEY_ENV, *AMBIENT_KEY_VARS):
        monkeypatch.setenv(name, SENTINEL_KEY)
    backend.complete(MESSAGES, SAMPLING)
    assert [seen.header("Authorization") for seen in stub_server.requests] == [None, None]
    for seen in stub_server.requests:
        assert all(SENTINEL_KEY not in value for value in seen.headers.values()), seen.headers


def test_api_key_is_read_from_the_environment_at_request_time(
    stub_server: StubServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    serve_ready(stub_server)
    backend = make_backend(stub_server, api_key_env=KEY_ENV)  # construction reads no key
    monkeypatch.setenv(KEY_ENV, SENTINEL_KEY)
    backend.complete(MESSAGES, SAMPLING)
    monkeypatch.setenv(KEY_ENV, ROTATED_KEY)
    backend.complete(MESSAGES, SAMPLING)
    chats = stub_server.requests_to("POST", CHAT_PATH)
    assert [seen.header("Authorization") for seen in chats] == [f"Bearer {SENTINEL_KEY}", f"Bearer {ROTATED_KEY}"]


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_a_missing_api_key_variable_raises_naming_it(
    stub_server: StubServer, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(KEY_ENV, value)
    serve_ready(stub_server)
    backend = make_backend(stub_server, api_key_env=KEY_ENV)
    with pytest.raises(ServingError) as info:
        backend.complete(MESSAGES, SAMPLING)
    assert KEY_ENV in str(info.value), str(info.value)
    assert stub_server.requests_to("POST", CHAT_PATH) == []


@pytest.mark.parametrize(
    "value",
    ["sk-fixture-4e7a91c2d3", "4e7a91c2d3", "key 4e7a91c2d3", "LASSI-KEY-4e7a91c2d3", ""],
    ids=["dashed-key", "leading-digit", "space", "dash", "empty"],
)
def test_an_api_key_env_that_is_not_a_variable_name_is_refused_without_quoting_it(
    stub_server: StubServer, value: str
) -> None:
    # A key value passed by mistake as api_key_env must not reach an error, a repr, or vars().
    with pytest.raises(ValueError) as info:
        make_backend(stub_server, api_key_env=value)
    assert "api_key_env" in str(info.value), str(info.value)
    for text in error_texts(info.value):
        assert "4e7a91c2d3" not in text, text
    assert stub_server.requests == []


def test_api_key_is_not_logged(
    stub_server: StubServer, keyed_backend: OpenAICompatBackend, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="lassi.llm")
    keyed_backend.complete(MESSAGES, SAMPLING)
    records = [r for r in caplog.records if r.name.startswith("lassi")]
    messages = [r.getMessage() for r in records]
    # Every record goes to the "lassi.llm" logger itself, at DEBUG.
    assert records and all(r.name == "lassi.llm" and r.levelno == logging.DEBUG for r in records), messages
    # The method, the URL path, and the status are logged; headers, bodies, and the host never are.
    assert f"GET {MODELS_PATH} -> 200" in messages and f"POST {CHAT_PATH} -> 200" in messages, messages
    for text in [caplog.text, *messages]:
        for secret in (SENTINEL_KEY, "Bearer", PROMPT_MARKER, REPLY_MARKER, "temperature"):
            assert secret not in text, (secret, text)
        assert_path_only(text, stub_server)


def test_api_key_is_not_in_repr_or_vars(stub_server: StubServer, keyed_backend: OpenAICompatBackend) -> None:
    keyed_backend.complete(MESSAGES, SAMPLING)
    text = repr(keyed_backend)
    assert "OpenAICompatBackend" in text and MODEL_ID in text and f"{stub_server.url}/v1" in text, text
    assert SENTINEL_KEY not in text
    for name, value in vars(keyed_backend).items():
        assert SENTINEL_KEY not in repr(value) and SENTINEL_KEY not in str(value), name
    # Nor in any attribute reachable from the instance or its class, nor in a module global.
    for name in dir(keyed_backend):
        assert SENTINEL_KEY not in repr(getattr(keyed_backend, name, None)), name
    for module in (llm.openai_compat, _http):
        for name, value in vars(module).items():
            assert SENTINEL_KEY not in repr(value), (module.__name__, name)


def test_api_key_is_not_in_errors(stub_server: StubServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, SENTINEL_KEY)
    stub_server.serve_models(["other/model"])
    with pytest.raises(ServingError) as refusal:
        make_backend(stub_server, api_key_env=KEY_ENV).complete(MESSAGES, SAMPLING)
    stub_server.serve_models([MODEL_ID])
    stub_server.reply("POST", CHAT_PATH, status=401, json_body={"error": {"message": "invalid key"}})
    with pytest.raises(ServingError) as rejected:
        make_backend(stub_server, api_key_env=KEY_ENV).complete(MESSAGES, SAMPLING)
    assert "401" in str(rejected.value)
    for error in (refusal.value, rejected.value):
        for text in error_texts(error):
            assert SENTINEL_KEY not in text and "Bearer" not in text, text


# Replies that quote the key back, as some servers and gateways do. ECHO becomes "Bearer <key>" in the reply.
KEY_ECHOES = [
    pytest.param(
        "GET", MODELS_PATH, {"status": 401, "json_body": {"error": f"Incorrect API key: {ECHO}"}}, id="models-401"
    ),
    pytest.param("GET", MODELS_PATH, {"json_body": {"data": [{"id": ECHO}, {"id": "other/model"}]}}, id="served-id"),
    pytest.param("GET", MODELS_PATH, {"json_body": {"error": f"token {ECHO}"}}, id="models-200-not-a-list"),
    pytest.param(
        "POST", CHAT_PATH, {"status": 401, "json_body": {"error": {"message": f"Invalid {ECHO}"}}}, id="chat-401"
    ),
    pytest.param(
        "POST", CHAT_PATH, {"status": 500, "json_body": {"detail": f"rejected {SENTINEL_KEY}"}}, id="bare-key"
    ),
    pytest.param("POST", CHAT_PATH, {"raw": reply_without_usage(echo=ECHO)}, id="chat-200-no-usage"),
    pytest.param("POST", CHAT_PATH, {"raw": f"<p>bad {ECHO}</p>".encode(), "content_type": "text/html"}, id="not-json"),
    pytest.param("POST", CHAT_PATH, {"raw_response": f"BROKEN {ECHO}\r\n\r\n".encode()}, id="status-line"),
]


@pytest.mark.parametrize(("method", "path", "reply"), KEY_ECHOES)
def test_a_key_the_server_echoes_is_redacted_from_errors(
    stub_server: StubServer, keyed_backend: OpenAICompatBackend, method: str, path: str, reply: dict[str, Any]
) -> None:
    stub_server.reply(method, path, echo=ECHO_AUTH, **reply)
    with pytest.raises(ServingError) as info:
        keyed_backend.complete(MESSAGES, SAMPLING)
    assert stub_server.requests[-1].header("Authorization") == f"Bearer {SENTINEL_KEY}"
    assert _http.REDACTED in str(info.value), str(info.value)
    # No chained exception keeps the server's text, so no traceback printer can show the key either.
    assert info.value.__cause__ is None and info.value.__context__ is None
    for text in error_texts(info.value):
        assert SENTINEL_KEY not in text, text


def test_a_key_across_the_quote_limit_is_not_shown_in_part(
    stub_server: StubServer, keyed_backend: OpenAICompatBackend
) -> None:
    # The key is redacted before the body is cut at 500 characters, so the cut never shows a key prefix.
    body = "x" * 495 + SENTINEL_KEY
    stub_server.reply("POST", CHAT_PATH, status=401, raw=body.encode("ascii"), content_type="text/plain")
    with pytest.raises(ServingError) as info:
        keyed_backend.complete(MESSAGES, SAMPLING)
    message = str(info.value)
    assert "x" * 495 + "<reda" in message and SENTINEL_KEY[:5] not in message, message


def test_a_json_escaped_key_is_redacted(stub_server: StubServer, monkeypatch: pytest.MonkeyPatch) -> None:
    # A key with a quote or a backslash comes back JSON-escaped in a JSON error body.
    key = 'sentinel"quote\\slash/e5c0ffee'
    monkeypatch.setenv(KEY_ENV, key)
    serve_ready(stub_server)
    stub_server.reply("POST", CHAT_PATH, status=401, json_body={"error": f"bad key {key}"})
    with pytest.raises(ServingError) as info:
        make_backend(stub_server, api_key_env=KEY_ENV).complete(MESSAGES, SAMPLING)
    assert stub_server.requests[-1].header("Authorization") == f"Bearer {key}"
    for text in error_texts(info.value):
        assert "e5c0ffee" not in text, text


# A visible-ASCII key holding each character that JSON, HTML, or URL encoding escapes.
SPECIAL_KEY = "sk+e5c0/ffee=<a>&b'c\"d\\e%f"
# Ways a server may write the key back into an error body; each is valid JSON string content or page text.
ESCAPED_ECHOES = [
    pytest.param(lambda key: json.dumps(key)[1:-1].replace("/", "\\/"), id="json-slash-escaped"),
    pytest.param(
        lambda key: json.dumps(key)[1:-1].replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"),
        id="json-html-safe",
    ),
    pytest.param(lambda key: "".join(f"\\u{ord(char):04x}" for char in key), id="json-all-unicode-escaped"),
    pytest.param(lambda key: "".join(f"\\u{ord(char):04X}" for char in key), id="json-all-unicode-escaped-upper"),
    pytest.param(html.escape, id="html-escaped"),
    pytest.param(lambda key: "".join(f"&#{ord(char)};" for char in key), id="html-decimal"),
    pytest.param(lambda key: "".join(f"&#x{ord(char):X};" for char in key), id="html-hex"),
    pytest.param(lambda key: json.dumps(html.escape(key))[1:-1].replace("/", "\\/"), id="html-then-json"),
    pytest.param(lambda key: urllib.parse.quote(key, safe=""), id="percent-encoded"),
    pytest.param(lambda key: "".join(f"%{byte:02x}" for byte in key.encode("ascii")), id="percent-encoded-lower"),
]


@pytest.mark.parametrize("encode", ESCAPED_ECHOES)
def test_a_key_the_server_echoes_escaped_is_redacted(
    stub_server: StubServer, monkeypatch: pytest.MonkeyPatch, encode: Any
) -> None:
    monkeypatch.setenv(KEY_ENV, SPECIAL_KEY)
    echoed = encode(SPECIAL_KEY)
    assert echoed != SPECIAL_KEY
    stub_server.reply("GET", MODELS_PATH, status=401, raw=f'{{"error": "bad key {echoed}"}}'.encode("ascii"))
    with pytest.raises(ServingError) as info:
        make_backend(stub_server, api_key_env=KEY_ENV).check()
    assert stub_server.requests[-1].header("Authorization") == f"Bearer {SPECIAL_KEY}"
    assert f"bad key {_http.REDACTED}" in str(info.value), str(info.value)
    for text in error_texts(info.value):
        for shown in (SPECIAL_KEY, echoed, ascii(echoed)[1:-1], "e5c0"):
            assert shown not in text, (shown, text)


@pytest.mark.parametrize(
    "value",
    ["sentinel\nvalue-9d1c", "sentinel value-9d1c", "sentinel\x01value-9d1c", f"caf{E_ACUTE}-value-9d1c"],
    ids=["newline", "space", "control", "non-ascii"],
)
def test_a_key_that_cannot_be_a_header_is_refused_before_sending(
    stub_server: StubServer, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(KEY_ENV, value)
    serve_ready(stub_server)
    with pytest.raises(ServingError) as info:
        make_backend(stub_server, api_key_env=KEY_ENV).complete(MESSAGES, SAMPLING)
    assert KEY_ENV in str(info.value) and "visible ASCII" in str(info.value), str(info.value)
    assert stub_server.requests == []
    for text in error_texts(info.value):
        assert "value-9d1c" not in text, text


def test_a_header_value_http_client_refuses_is_never_quoted(stub_server: StubServer) -> None:
    # http.client puts a refused header value in its ValueError; the transport shows only the error type.
    with pytest.raises(ServingError) as info:
        _http.request_json(
            "GET", f"{stub_server.url}{MODELS_PATH}", headers={"Authorization": "Bearer a\nb-4e2a"}, timeout_s=5.0
        )
    message = str(info.value)
    assert f"GET {MODELS_PATH}" in message and "could not be sent" in message, message
    assert info.value.__cause__ is None and info.value.__context__ is None
    assert stub_server.requests == []
    for text in error_texts(info.value):
        assert "b-4e2a" not in text, text


def test_api_key_is_not_in_the_written_trial(
    stub_server: StubServer, keyed_backend: OpenAICompatBackend, tmp_path: Path
) -> None:
    completion = keyed_backend.complete(MESSAGES, SAMPLING)
    trial = make_trial(model_info(keyed_backend, SAMPLING), completion.text)
    out = store.write_trial(trial, tmp_path / "runs", store.TextStore(tmp_path / "store"))
    trial_json = (out / store.TRIAL_JSON).read_bytes()
    trial_md = (out / store.TRIAL_MD).read_bytes()
    assert json.loads(trial_json)["model"] == {
        "backend": "openai_compat",
        "id": MODEL_ID,
        "sampling": {"temperature": 0.35, "top_p": 0.85, "max_tokens": 777},
    }
    assert b"temperature 0.35, top_p 0.85, max_tokens 777" in trial_md
    written = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert {out / store.TRIAL_JSON, out / store.TRIAL_MD} <= set(written)
    for path in written:
        assert SENTINEL_KEY.encode("ascii") not in path.read_bytes(), path
