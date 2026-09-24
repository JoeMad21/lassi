"""Tests for the Ollama backend in lassi/llm/ollama.py (P0.4).

The backend serves the LASSI paper's Ollama arm (bible Model Serving, arm B0).
Before the first chat request it confirms that Ollama's OpenAI-compatible
model list at `/v1/models` holds the exact id, full tag included (Serving
Rules). It chats through `/api/chat` with the sampling parameters in
`options` (max_tokens as `num_predict`), returns the text with Ollama's
prompt_eval_count and eval_count and never fabricates a missing count, and
unloads the model through `/api/generate` with keep_alive 0 (bible LASSI quirk
table: Ollama unload before each execution). Every request goes to the local
stub server from conftest.py on 127.0.0.1; no test touches the network. Token
counts and model ids below are stub fixture values, not measurements.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import pytest

from lassi import llm
from lassi.core.interfaces import Completion, Message, Sampling
from lassi.core.registry import DEFAULT_REGISTRY

if TYPE_CHECKING:
    from conftest import StubServer

DEFAULT_BASE_URL = llm.ollama.DEFAULT_BASE_URL
OllamaBackend = llm.ollama.OllamaBackend
ServingError = llm.ServingError

MODEL_ID = "coder:7b-q8_0"
MODELS_PATH = "/v1/models"
CHAT_PATH = "/api/chat"
GENERATE_PATH = "/api/generate"
SAMPLING = Sampling(temperature=0.35, top_p=0.85, max_tokens=777)
E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"

REPLY_TEXT = "```c\n// FILE: main.c\nint main(void) { return 0; }\n```\n"
MESSAGES = (
    Message(role="system", content="You translate code."),
    Message(role="user", content="Translate main.cu to OpenMP.\n"),
)

PROXY_VARS = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
NO_PROXY_VARS = ("NO_PROXY", "no_proxy")
PROMPT_MARKER = "prompt-marker-7d20"
REDIRECT_TARGET = "/api/elsewhere"
LONG_BODY = "ERR-START " + "x" * 2000 + " ERR-END"

_DELETE = object()


# ---------------------------------------------------------------------------
# Helpers


def chat_body(content: Any = REPLY_TEXT, *, prompt_eval_count: Any = 13, eval_count: Any = 21) -> dict[str, Any]:
    """Return an Ollama /api/chat reply with stream off."""
    return {
        "model": MODEL_ID,
        "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }


def chat_edit(path: tuple[str, ...], value: Any = _DELETE) -> bytes:
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


def reply_without_counts(padding: str) -> bytes:
    """Return an encoded chat reply with no token counts, led by a `padding` field."""
    body = {"padding": padding, **chat_body()}
    del body["prompt_eval_count"], body["eval_count"]
    return json.dumps(body).encode("utf-8")


def assert_path_only(text: str, stub: StubServer) -> None:
    """Fail when `text` shows the stub's scheme, host, or port: messages and logs name the URL path only."""
    for part in ("http://", "127.0.0.1", f":{stub.port}"):
        assert part not in text, (part, text)


def serve_ready(stub: StubServer, reply: dict[str, Any] | None = None) -> None:
    """Serve a model list that includes MODEL_ID, a chat reply (chat_body() unless given), and an unload reply."""
    stub.serve_models(["coder:7b", MODEL_ID])
    stub.reply("POST", CHAT_PATH, json_body=chat_body() if reply is None else reply)
    stub.reply(
        "POST", GENERATE_PATH, json_body={"model": MODEL_ID, "response": "", "done": True, "done_reason": "unload"}
    )


def make_backend(stub: StubServer, **kwargs: Any) -> OllamaBackend:
    """Return a backend for MODEL_ID pointed at the stub."""
    return OllamaBackend(MODEL_ID, base_url=stub.url, **kwargs)


@pytest.fixture
def backend(stub_server: StubServer) -> OllamaBackend:
    """Return a backend with the stub serving MODEL_ID, a chat reply, and an unload reply."""
    serve_ready(stub_server)
    return make_backend(stub_server)


# ---------------------------------------------------------------------------
# Registration and shape


def test_registered_as_llm_backend_ollama() -> None:
    entry = DEFAULT_REGISTRY.get("LLMBackend", "ollama")
    assert entry.factory is OllamaBackend
    assert entry.capabilities == frozenset({"chat", "model_check", "unload", "unload_before_run"})
    assert llm.OllamaBackend is OllamaBackend


def test_name_capabilities_and_default_url() -> None:
    assert OllamaBackend.name == "ollama"
    assert isinstance(OllamaBackend.capabilities, frozenset)
    assert OllamaBackend.capabilities == frozenset({"chat", "model_check", "unload", "unload_before_run"})
    assert DEFAULT_BASE_URL == "http://127.0.0.1:11434"


def test_constructor_signature() -> None:
    params = [(p.name, p.kind, p.default) for p in inspect.signature(OllamaBackend).parameters.values()]
    assert params == [
        ("model_id", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("base_url", inspect.Parameter.KEYWORD_ONLY, DEFAULT_BASE_URL),
        ("timeout_s", inspect.Parameter.KEYWORD_ONLY, 600.0),
    ]


def test_repr_shows_class_model_id_and_base_url(stub_server: StubServer) -> None:
    text = repr(make_backend(stub_server))
    assert "OllamaBackend" in text and MODEL_ID in text and stub_server.url in text, text


def test_repr_shows_nothing_else(stub_server: StubServer) -> None:
    text = repr(make_backend(stub_server, timeout_s=12.5))
    assert "timeout" not in text and "12.5" not in text, text


def test_construction_sends_nothing(stub_server: StubServer) -> None:
    make_backend(stub_server)
    assert stub_server.requests == []


# ---------------------------------------------------------------------------
# complete


def test_complete_returns_text_and_token_counts(stub_server: StubServer, backend: OllamaBackend) -> None:
    completion = backend.complete(MESSAGES, SAMPLING)
    assert isinstance(completion, Completion)
    assert completion == Completion(text=REPLY_TEXT, prompt_tokens=13, completion_tokens=21)
    assert stub_server.calls() == [("GET", MODELS_PATH), ("POST", CHAT_PATH)]


@pytest.mark.parametrize(
    "sampling",
    [Sampling(0.35, 0.85, 777), Sampling(0.0, 1.0, 16384), Sampling(1.25, 0.05, 1)],
    ids=["typical", "greedy", "edge"],
)
def test_chat_request_carries_messages_and_sampling_exactly(
    stub_server: StubServer, backend: OllamaBackend, sampling: Sampling
) -> None:
    backend.complete(MESSAGES, sampling)
    (chat,) = stub_server.requests_to("POST", CHAT_PATH)
    assert chat.body == {
        "model": MODEL_ID,
        "messages": [{"role": m.role, "content": m.content} for m in MESSAGES],
        "stream": False,
        "options": {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "num_predict": sampling.max_tokens,
        },
    }
    options = chat.body["options"]
    assert type(options["temperature"]) is float and type(options["top_p"]) is float
    assert type(options["num_predict"]) is int
    assert chat.body["stream"] is False


def test_messages_are_sent_unchanged(stub_server: StubServer, backend: OllamaBackend) -> None:
    long_text = "".join(f"line {i}: {'z' * 60}\n" for i in range(2500)) + "  trailing spaces  \n\n"
    messages = [Message("user", long_text), Message("assistant", ""), Message("user", f"caf{E_ACUTE}\r\nend")]
    backend.complete(messages, SAMPLING)
    (chat,) = stub_server.requests_to("POST", CHAT_PATH)
    assert chat.body["messages"] == [{"role": m.role, "content": m.content} for m in messages]


def test_zero_counts_and_empty_content_are_kept(stub_server: StubServer) -> None:
    serve_ready(stub_server, chat_body("", prompt_eval_count=0, eval_count=0))
    assert make_backend(stub_server).complete(MESSAGES, SAMPLING) == Completion(
        text="", prompt_tokens=0, completion_tokens=0
    )


@pytest.mark.parametrize("status", [201, 203])
def test_any_2xx_chat_status_is_accepted(stub_server: StubServer, status: int) -> None:
    stub_server.serve_models([MODEL_ID])
    stub_server.reply("POST", CHAT_PATH, status=status, json_body=chat_body())
    assert make_backend(stub_server).complete(MESSAGES, SAMPLING) == Completion(REPLY_TEXT, 13, 21)


BAD_CHAT_REPLIES = [
    pytest.param(b'{"message": {', id="malformed-json"),
    pytest.param(b"", id="empty-body"),
    pytest.param(b"[]", id="json-array"),
    pytest.param(chat_edit(("prompt_eval_count",)), id="no-prompt-eval-count"),
    pytest.param(chat_edit(("eval_count",)), id="no-eval-count"),
    pytest.param(chat_edit(("prompt_eval_count",), None), id="null-prompt-eval-count"),
    pytest.param(chat_edit(("eval_count",), None), id="null-eval-count"),
    pytest.param(chat_edit(("message", "content"), None), id="null-content"),
    pytest.param(chat_edit(("message", "content")), id="no-content"),
    pytest.param(chat_edit(("message",)), id="no-message"),
    pytest.param(chat_edit(("message",), None), id="null-message"),
    pytest.param(chat_edit(("prompt_eval_count",), "13"), id="string-count"),
    pytest.param(chat_edit(("prompt_eval_count",), 13.0), id="float-count"),
    pytest.param(chat_edit(("eval_count",), -1), id="negative-count"),
    pytest.param(chat_edit(("eval_count",), True), id="boolean-count"),
    pytest.param(chat_edit(("message", "content"), ["text"]), id="list-content"),
    pytest.param(chat_edit(("message", "content"), {"text": "x"}), id="object-content"),
]


@pytest.mark.parametrize("raw", BAD_CHAT_REPLIES)
def test_a_bad_chat_reply_raises_serving_error(stub_server: StubServer, raw: bytes) -> None:
    stub_server.serve_models([MODEL_ID])
    stub_server.reply("POST", CHAT_PATH, raw=raw)
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).complete(MESSAGES, SAMPLING)
    # The message names the URL path and the status, even for a 2xx reply whose body is wrong.
    assert f"POST {CHAT_PATH} returned HTTP 200" in str(info.value), str(info.value)


@pytest.mark.parametrize("status", [400, 404, 500, 503])
def test_a_non_2xx_chat_status_raises_naming_path_and_status(stub_server: StubServer, status: int) -> None:
    stub_server.serve_models([MODEL_ID])
    stub_server.reply("POST", CHAT_PATH, status=status, json_body=chat_body())
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).complete(MESSAGES, SAMPLING)
    message = str(info.value)
    assert f"POST {CHAT_PATH} returned HTTP {status}" in message, message
    assert_path_only(message, stub_server)


@pytest.mark.parametrize(
    ("status", "raw", "content_type"),
    [
        pytest.param(500, LONG_BODY.encode("ascii"), "text/plain", id="status-500"),
        pytest.param(200, LONG_BODY.encode("ascii"), "text/plain", id="malformed-200"),
        pytest.param(200, reply_without_counts(LONG_BODY), "application/json", id="json-without-counts-200"),
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
    assert "x" * 491 not in message
    assert f"(first 500 of {len(raw)} characters)" in message, message


def test_an_unreachable_server_raises_serving_error(dead_port: int) -> None:
    with pytest.raises(ServingError):
        OllamaBackend(MODEL_ID, base_url=f"http://127.0.0.1:{dead_port}").complete(MESSAGES, SAMPLING)


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", MODELS_PATH), ("POST", CHAT_PATH), ("POST", GENERATE_PATH)],
    ids=["models", "chat", "unload"],
)
def test_the_timeout_applies(stub_server: StubServer, method: str, path: str) -> None:
    serve_ready(stub_server)
    delayed = {"data": [{"id": MODEL_ID}]} if method == "GET" else chat_body()
    stub_server.reply(method, path, json_body=delayed, delay_s=30.0)
    backend = make_backend(stub_server, timeout_s=0.3)
    start = time.monotonic()
    with pytest.raises(ServingError) as info:
        if path == GENERATE_PATH:
            backend.unload()
        else:
            backend.complete(MESSAGES, SAMPLING)
    assert time.monotonic() - start < 10.0
    message = str(info.value)
    assert f"{method} {path} got no reply" in message and "timed out" in message, message
    assert len(stub_server.requests_to(method, path)) == 1


@pytest.mark.parametrize("suffix", ["", "/", "//"], ids=["no-slash", "trailing-slash", "two-slashes"])
def test_base_url_trailing_slash_is_removed(stub_server: StubServer, suffix: str) -> None:
    serve_ready(stub_server)
    backend = OllamaBackend(MODEL_ID, base_url=stub_server.url + suffix)
    assert backend.complete(MESSAGES, SAMPLING).text == REPLY_TEXT
    backend.unload()
    targets = [seen.target for seen in stub_server.requests]
    assert targets[:2] == [MODELS_PATH, CHAT_PATH] and targets[-1] == GENERATE_PATH, targets
    assert all("//" not in target for target in targets), targets


def test_every_request_is_sent_as_json(stub_server: StubServer, backend: OllamaBackend) -> None:
    backend.complete(MESSAGES, SAMPLING)
    backend.unload()
    types = [(seen.header("Content-Type") or "").split(";")[0].strip() for seen in stub_server.requests]
    assert types == ["application/json"] * 3


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:11434?q=7c1d",
        "http://127.0.0.1:11434/#f7c1d",
        "http://user:pw-7c1d@127.0.0.1:11434",
        "ftp://127.0.0.1:11434",
        "127.0.0.1:11434",
        "http://127.0.0.1:port-7c1d",
        "http://127.0.0.1:99999",
    ],
    ids=["query", "fragment", "login", "ftp", "no-scheme", "port-not-a-number", "port-too-large"],
)
def test_a_base_url_that_is_not_a_plain_http_url_is_refused(base_url: str) -> None:
    with pytest.raises(ValueError) as info:
        OllamaBackend(MODEL_ID, base_url=base_url)
    assert "7c1d" not in str(info.value) and "7c1d" not in repr(info.value.__context__), str(info.value)


@pytest.mark.parametrize("timeout_s", [0, -1.0, float("nan"), float("inf"), True, "10", None])
def test_a_timeout_that_is_not_a_positive_number_is_refused(stub_server: StubServer, timeout_s: Any) -> None:
    with pytest.raises(ValueError) as info:
        make_backend(stub_server, timeout_s=timeout_s)
    assert "timeout_s" in str(info.value), str(info.value)


# ---------------------------------------------------------------------------
# check: Ollama's OpenAI-compatible model list


def test_check_returns_the_served_entry(stub_server: StubServer) -> None:
    entry = {"id": MODEL_ID, "object": "model", "created": 1767225600, "owned_by": "library"}
    stub_server.serve_models(["coder:7b", entry])
    assert make_backend(stub_server).check() == entry
    assert stub_server.calls() == [("GET", MODELS_PATH)]


def test_check_is_cached(stub_server: StubServer, backend: OllamaBackend) -> None:
    first = backend.check()
    assert backend.check() == first
    backend.complete(MESSAGES, SAMPLING)
    backend.complete(MESSAGES, SAMPLING)
    assert stub_server.calls() == [("GET", MODELS_PATH), ("POST", CHAT_PATH), ("POST", CHAT_PATH)]


def test_a_failed_check_is_not_cached(stub_server: StubServer) -> None:
    stub_server.serve_models(["coder:7b"])
    backend = make_backend(stub_server)
    with pytest.raises(ServingError):
        backend.check()
    serve_ready(stub_server)
    assert backend.check()["id"] == MODEL_ID
    assert backend.complete(MESSAGES, SAMPLING).text == REPLY_TEXT
    assert stub_server.calls() == [("GET", MODELS_PATH), ("GET", MODELS_PATH), ("POST", CHAT_PATH)]


def test_complete_refuses_a_missing_id_before_any_chat_request(stub_server: StubServer) -> None:
    # No served id is a substring of MODEL_ID, so each position below is where the served list names it.
    stub_server.serve_models(["zeta:1b", "alpha:1b", "mid:3b"])
    stub_server.reply("POST", CHAT_PATH, json_body=chat_body())
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).complete(MESSAGES, SAMPLING)
    assert stub_server.calls() == [("GET", MODELS_PATH)]
    message = str(info.value)
    assert MODEL_ID in message, message
    positions = [message.find(served) for served in ("alpha:1b", "mid:3b", "zeta:1b")]
    assert -1 not in positions and positions == sorted(positions), message


@pytest.mark.parametrize(
    ("model_id", "served"),
    [
        pytest.param("coder:7b", ["coder:7b-instruct", "coder:latest", "coder"], id="tag-must-match"),
        pytest.param("coder", ["coder:latest"], id="no-implicit-latest"),
        pytest.param("coder:7b", ["Coder:7B"], id="case-must-match"),
    ],
)
def test_check_needs_the_exact_id_with_its_tag(stub_server: StubServer, model_id: str, served: list[str]) -> None:
    stub_server.serve_models(served)
    stub_server.reply("POST", CHAT_PATH, json_body=chat_body())
    with pytest.raises(ServingError) as info:
        OllamaBackend(model_id, base_url=stub_server.url).complete(MESSAGES, SAMPLING)
    # Served ids may hold model_id as a substring, so look for it, quoted, before the served list.
    message = str(info.value)
    assert f"'{model_id}'" in message.partition("served ids")[0], message
    assert stub_server.requests_to("POST", CHAT_PATH) == []


def test_a_long_served_id_list_is_cut_and_counted(stub_server: StubServer) -> None:
    ids = [f"coder:{index:05d}-{'z' * 60}" for index in range(3000)]
    stub_server.serve_models(ids)
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).check()
    message = str(info.value)
    assert len(message) < 1200, len(message)
    shown = [served for served in sorted(ids) if served in message]
    assert shown and shown == sorted(ids)[: len(shown)], message
    assert message.endswith(f"(and {len(ids) - len(shown)} more)"), message


def test_a_too_deeply_nested_entry_is_refused_and_not_cached(stub_server: StubServer) -> None:
    # Deep enough for json.loads but not for copy.deepcopy: check() must raise ServingError, not RecursionError.
    deep = f'{{"data": [{{"id": "{MODEL_ID}", "x": {"[" * 600}{"]" * 600}}}]}}'
    stub_server.reply("GET", MODELS_PATH, raw=deep.encode("ascii"))
    backend = make_backend(stub_server)
    for _ in range(2):
        with pytest.raises(ServingError) as info:
            backend.check()
        assert "levels deep" in str(info.value), str(info.value)
    serve_ready(stub_server)
    assert backend.complete(MESSAGES, SAMPLING).text == REPLY_TEXT
    assert stub_server.calls() == [("GET", MODELS_PATH)] * 3 + [("POST", CHAT_PATH)]


@pytest.mark.parametrize(
    ("status", "raw"),
    [
        pytest.param(404, b"404 page not found", id="status-404"),
        pytest.param(200, b"<html>not json</html>", id="malformed-json"),
    ],
)
def test_a_bad_models_reply_raises_before_any_chat_request(stub_server: StubServer, status: int, raw: bytes) -> None:
    stub_server.reply("GET", MODELS_PATH, status=status, raw=raw)
    stub_server.reply("POST", CHAT_PATH, json_body=chat_body())
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).complete(MESSAGES, SAMPLING)
    assert f"GET {MODELS_PATH} returned HTTP {status}" in str(info.value), str(info.value)
    assert stub_server.requests_to("POST", CHAT_PATH) == []


# ---------------------------------------------------------------------------
# unload


def test_unload_posts_keep_alive_zero(stub_server: StubServer, backend: OllamaBackend) -> None:
    assert backend.unload() is None
    (unload,) = stub_server.requests_to("POST", GENERATE_PATH)
    assert unload.body == {"model": MODEL_ID, "keep_alive": 0}
    assert type(unload.body["keep_alive"]) is int
    assert stub_server.requests_to("POST", CHAT_PATH) == []


def test_unload_after_a_completion(stub_server: StubServer, backend: OllamaBackend) -> None:
    backend.complete(MESSAGES, SAMPLING)
    backend.unload()
    assert stub_server.calls()[-1] == ("POST", GENERATE_PATH)
    assert len(stub_server.requests_to("POST", GENERATE_PATH)) == 1


def test_unload_failure_raises_serving_error(stub_server: StubServer) -> None:
    stub_server.serve_models([MODEL_ID])
    stub_server.reply("POST", GENERATE_PATH, status=500, json_body={"error": "unload failed"})
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).unload()
    message = str(info.value)
    assert f"POST {GENERATE_PATH} returned HTTP 500" in message, message


# ---------------------------------------------------------------------------
# Redirects are refused, never followed


@pytest.mark.parametrize(("method", "path"), [("GET", MODELS_PATH), ("POST", CHAT_PATH)], ids=["models", "chat"])
@pytest.mark.parametrize("status", [302, 307])
def test_a_redirect_is_refused(stub_server: StubServer, method: str, path: str, status: int) -> None:
    serve_ready(stub_server)
    stub_server.reply(method, path, status=status, headers={"Location": f"{stub_server.url}{REDIRECT_TARGET}"})
    stub_server.serve_models([MODEL_ID], path=REDIRECT_TARGET)
    stub_server.reply("POST", REDIRECT_TARGET, json_body=chat_body())
    with pytest.raises(ServingError) as info:
        make_backend(stub_server).complete(MESSAGES, SAMPLING)
    message = str(info.value)
    assert f"{method} {path} returned HTTP {status}" in message and "redirect" in message, message
    assert stub_server.calls()[-1] == (method, path)
    assert all(seen.path != REDIRECT_TARGET for seen in stub_server.requests)


# ---------------------------------------------------------------------------
# Logging: the method, the URL path, and the status only


def test_requests_are_logged_by_method_path_and_status_only(
    stub_server: StubServer, backend: OllamaBackend, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="lassi.llm")
    backend.complete([Message("user", f"Translate this. {PROMPT_MARKER}")], SAMPLING)
    backend.unload()
    records = [r for r in caplog.records if r.name.startswith("lassi")]
    messages = [r.getMessage() for r in records]
    assert records and all(r.name == "lassi.llm" and r.levelno == logging.DEBUG for r in records), messages
    assert messages == [f"GET {MODELS_PATH} -> 200", f"POST {CHAT_PATH} -> 200", f"POST {GENERATE_PATH} -> 200"]
    for text in [caplog.text, *messages]:
        for body_text in (PROMPT_MARKER, "main(void)", "num_predict", "keep_alive"):
            assert body_text not in text, (body_text, text)
        assert_path_only(text, stub_server)


# ---------------------------------------------------------------------------
# Proxies: loopback traffic never goes through a proxy


def test_loopback_requests_ignore_the_proxy_environment(
    stub_server: StubServer, monkeypatch: pytest.MonkeyPatch, dead_port: int
) -> None:
    for name in NO_PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in PROXY_VARS:
        monkeypatch.setenv(name, f"http://127.0.0.1:{dead_port}")
    serve_ready(stub_server)
    backend = make_backend(stub_server)
    assert backend.complete(MESSAGES, SAMPLING).text == REPLY_TEXT
    backend.unload()
    assert [seen.target for seen in stub_server.requests] == [MODELS_PATH, CHAT_PATH, GENERATE_PATH]
