"""Tests for what the served backends record about their server (task P17.3): the entry, the version, never a key.

Bible: Model Serving (Serving Rules), Agent Rule 12; plans/p17-portable.md,
task P17.3; plans/spikes/p17-frameworks.md, Results 4 (the reply shapes) and
Consequences for the plan, P17.3; PHASE-NOTES P17 (the P17.3 points:
context length, volatile fields, versions).

The contract these tests fix:

- lassi.core.capabilities.MODEL_CHECK is "model_check"; openai_compat and
  ollama declare it. Such a backend confirms its model id with check()
  before any chat request and gives the run a serving record with serving().
- check() finds the exact id in every server's list shape (vLLM, SGLang,
  llama-server in single-model mode, furiosa-llm) and refuses an id the
  server does not list, naming both, before any chat request.
- serving() returns {base_url, model, version, version_from}. `model` is the
  entry as the server sent it, every field kept (max_model_len, null for a
  vLLM LoRA card; meta.n_ctx; max_context_len, max_prompt_len, and
  artifact_id), except `created` and `permission`, which are new on every
  GET (lassi.llm._serving.VOLATILE_FIELDS); a field the server does not send
  is absent, never filled in.
- The version comes from the route the entry's owned_by names
  (lassi.llm._serving.VERSION_ENDPOINTS): vllm GET /version `version`,
  sglang GET /server_info `version`, llamacpp GET /props `build_info`, each
  at the server root (base_url without a final /v1), with the same
  Authorization header as the /v1 requests. version_from names the request
  and field ("GET /version version") whenever one was sent, and is null when
  none was (another owned_by, or a base_url without a final /v1). A failed
  version request (an error status, a body that is not JSON, no field, a
  field that is not a string) gives version null and raises nothing, and
  neither logs nor quotes the reply.
- From SGLang's /server_info, whose reply holds api_key, admin_api_key, and
  launch_command, only `version` is read; the key appears in no record, log,
  or error. A record that would hold the API key (a server echoing it) is
  refused with a ServingError that quotes none of it.
- Ollama's serving record has its entry and no version (no route is read).
- serving_line(record) is run.md's Server text: labeled pairs, "owned_by"
  and "version" ("-" when missing or null), then each context field present
  ("-" when null). Ollama's owned_by is the model's namespace, so the label
  keeps it from reading as a server name.

Every request goes to the stub on 127.0.0.1 (tests/llm/served_stubs.py);
every served value, key, and id is SYNTHETIC. No value in this module is a
measurement.
"""

from __future__ import annotations

import copy
import importlib
import json
import logging
import traceback
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest
from served_stubs import (
    CHAT_PATH,
    COMPLETION_TOKENS,
    LLAMA_BUILD_INFO,
    MODELS_PATH,
    PROMPT_TOKENS,
    SGLANG_VERSION,
    VLLM_VERSION,
    serve_furiosa,
    serve_llama_server,
    serve_ollama,
    serve_sglang,
    serve_vllm,
    stable,
)

from lassi.core import capabilities
from lassi.core.interfaces import Message, Sampling
from lassi.llm import OllamaBackend, OpenAICompatBackend, ServingError

if TYPE_CHECKING:
    from conftest import StubServer

MODEL_ID = "coder-fixture-a"
OTHER_ID = "coder-fixture-b"
LORA_ID = "coder-fixture-lora"
OLLAMA_ID = "coder-7b-q8_0"
KEY_ENV = "LASSI_TEST_API_KEY"
SENTINEL_KEY = "lassi-test-sentinel-key-6c1f93ab"
# The stub replaces this marker with the Authorization header it received.
ECHO = "<<authorization>>"
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=64)
MESSAGES = (Message(role="user", content="Translate main.cpp to CUDA."),)
SERVERS = ("vllm", "sglang", "llama-server", "furiosa")


def serving_module() -> ModuleType:
    """Import lassi.llm._serving, failing the test clearly while it does not exist."""
    try:
        return importlib.import_module("lassi.llm._serving")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.llm._serving does not exist yet (task P17.3): {error}")


def serve(stub: StubServer, server: str, name: str = MODEL_ID, **kwargs: Any) -> dict[str, Any]:
    """Serve `name` in one server's shapes and return the entry served for it."""
    servers: dict[str, Callable[..., dict[str, Any]]] = {
        "vllm": serve_vllm,
        "sglang": serve_sglang,
        "llama-server": serve_llama_server,
        "furiosa": serve_furiosa,
    }
    return servers[server](stub, name, **kwargs)


def make_backend(stub: StubServer, name: str = MODEL_ID, **kwargs: Any) -> OpenAICompatBackend:
    """Return an openai_compat backend for `name` at the stub's /v1 base URL."""
    return OpenAICompatBackend(name, base_url=f"{stub.url}/v1", timeout_s=30.0, **kwargs)


def serving(backend: Any) -> dict[str, Any]:
    """Return backend.serving(), failing the test clearly while the method does not exist."""
    method = getattr(backend, "serving", None)
    if not callable(method):
        pytest.fail(f"{type(backend).__name__} has no serving() yet (task P17.3)")
    return method()


def error_texts(error: BaseException) -> list[str]:
    """Return every text an error shows: str, repr, args, and the formatted traceback with its chain."""
    texts = [str(error), repr(error), repr(error.args)]
    texts.append("".join(traceback.format_exception(type(error), error, error.__traceback__)))
    return texts


def keyed(stub: StubServer, monkeypatch: pytest.MonkeyPatch, name: str = MODEL_ID) -> OpenAICompatBackend:
    """Return a backend whose API key comes from KEY_ENV, which holds the sentinel key."""
    monkeypatch.setenv(KEY_ENV, SENTINEL_KEY)
    return make_backend(stub, name, api_key_env=KEY_ENV)


# ---------------------------------------------------------------------------
# The capability


def test_model_check_is_a_named_capability_of_both_served_backends() -> None:
    assert getattr(capabilities, "MODEL_CHECK", None) == "model_check"
    for cls in (OpenAICompatBackend, OllamaBackend):
        assert "model_check" in cls.capabilities, cls.__name__
        assert callable(getattr(cls, "serving", None)), f"{cls.__name__} declares model_check but has no serving()"


# ---------------------------------------------------------------------------
# The model check against each server's list


@pytest.mark.parametrize("server", SERVERS)
def test_check_finds_the_id_in_each_servers_list(stub_server: StubServer, server: str) -> None:
    entry = serve(stub_server, server)
    found = make_backend(stub_server).check()
    assert found["id"] == MODEL_ID
    assert found == entry, "check() keeps every field the server sent"


@pytest.mark.parametrize("server", SERVERS)
def test_a_model_the_server_does_not_list_is_refused_before_any_chat_request(
    stub_server: StubServer, server: str
) -> None:
    serve(stub_server, server, OTHER_ID)
    backend = make_backend(stub_server)
    with pytest.raises(ServingError) as info:
        backend.complete(MESSAGES, SAMPLING)
    message = str(info.value)
    assert MODEL_ID in message and OTHER_ID in message, message
    assert stub_server.requests_to("POST", CHAT_PATH) == []
    with pytest.raises(ServingError):
        serving(backend)
    assert [method for method, _ in stub_server.calls()] == ["GET", "GET"], "no version request after a refusal"


# ---------------------------------------------------------------------------
# The serving record: the entry as sent


@pytest.mark.parametrize("server", SERVERS)
def test_serving_keeps_the_entry_fields_as_sent(stub_server: StubServer, server: str) -> None:
    entry = serve(stub_server, server)
    record = serving(make_backend(stub_server))
    assert record["base_url"] == f"{stub_server.url}/v1"
    assert record["model"] == stable(entry)
    model = record["model"]
    if server in ("vllm", "sglang"):
        assert model["max_model_len"] == entry["max_model_len"]
    if server == "llama-server":
        assert model["meta"]["n_ctx"] == entry["meta"]["n_ctx"]
        assert "max_model_len" not in model, "a field the server does not send is never filled in"
    if server == "furiosa":
        for name in ("max_context_len", "max_prompt_len", "artifact_id"):
            assert model[name] == entry[name], name
        assert "max_model_len" not in model


def test_a_vllm_lora_card_keeps_its_null_max_model_len(stub_server: StubServer) -> None:
    serve_vllm(stub_server, MODEL_ID, loras=[LORA_ID])
    record = serving(make_backend(stub_server, LORA_ID))
    assert record["model"]["id"] == LORA_ID
    assert record["model"]["max_model_len"] is None and record["model"]["parent"] == MODEL_ID


@pytest.mark.parametrize("server", SERVERS)
def test_volatile_fields_are_left_out(stub_server: StubServer, server: str) -> None:
    entry = serve(stub_server, server)
    assert "created" in entry
    record = serving(make_backend(stub_server))
    assert "created" not in record["model"] and "permission" not in record["model"]
    assert serving_module().VOLATILE_FIELDS == frozenset({"created", "permission"})


def test_stable_entry_drops_only_the_volatile_fields_and_copies() -> None:
    module = serving_module()
    entry = {"id": "m", "created": 1, "permission": [{"id": "p"}], "meta": {"n_ctx": 8}, "parent": None}
    kept = module.stable_entry(entry)
    assert kept == {"id": "m", "meta": {"n_ctx": 8}, "parent": None}
    kept["meta"]["n_ctx"] = 9
    assert entry["meta"]["n_ctx"] == 8, "the record never shares an object with the cached entry"


def test_the_record_is_json_text(stub_server: StubServer) -> None:
    serve_vllm(stub_server, MODEL_ID)
    record = serving(make_backend(stub_server))
    assert set(record) == {"base_url", "model", "version", "version_from"}
    assert json.loads(json.dumps(record, allow_nan=False)) == record


# ---------------------------------------------------------------------------
# The version, by owned_by, at the server root


def test_vllm_version_is_read_at_the_server_root(stub_server: StubServer) -> None:
    serve_vllm(stub_server, MODEL_ID)
    record = serving(make_backend(stub_server))
    assert stub_server.calls() == [("GET", MODELS_PATH), ("GET", "/version")], "not /v1/version"
    assert (record["version"], record["version_from"]) == (VLLM_VERSION, "GET /version version")


def test_sglang_server_info_gives_only_its_version(
    stub_server: StubServer, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    serve_sglang(stub_server, MODEL_ID, api_key=SENTINEL_KEY)
    record = serving(keyed(stub_server, monkeypatch))
    assert stub_server.calls() == [("GET", MODELS_PATH), ("GET", "/server_info")]
    assert (record["version"], record["version_from"]) == (SGLANG_VERSION, "GET /server_info version")
    text = json.dumps(record)
    for field in ("api_key", "admin_api_key", "launch_command", "model_path", SENTINEL_KEY):
        assert field not in text, field
    assert all(SENTINEL_KEY not in logged.getMessage() for logged in caplog.records)


def test_llama_server_version_is_build_info_from_props(stub_server: StubServer) -> None:
    serve_llama_server(stub_server, MODEL_ID)
    record = serving(make_backend(stub_server))
    assert stub_server.calls() == [("GET", MODELS_PATH), ("GET", "/props")]
    assert (record["version"], record["version_from"]) == (LLAMA_BUILD_INFO, "GET /props build_info")


def test_an_unknown_server_sends_no_version_request(stub_server: StubServer) -> None:
    serve_furiosa(stub_server, MODEL_ID)
    record = serving(make_backend(stub_server))
    assert stub_server.calls() == [("GET", MODELS_PATH)]
    assert (record["version"], record["version_from"]) == (None, None)


def test_a_base_url_without_v1_sends_no_version_request(stub_server: StubServer) -> None:
    card = serve_vllm(stub_server, MODEL_ID)
    stub_server.reply("GET", "/serve/models", json_body={"object": "list", "data": [card]})
    backend = OpenAICompatBackend(MODEL_ID, base_url=f"{stub_server.url}/serve", timeout_s=30.0)
    record = serving(backend)
    assert stub_server.calls() == [("GET", "/serve/models")]
    assert (record["base_url"], record["version"], record["version_from"]) == (f"{stub_server.url}/serve", None, None)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        pytest.param(404, b'{"detail": "Not Found"}', id="404"),
        pytest.param(401, b'{"error": "Unauthorized"}', id="401"),
        pytest.param(200, f"version {SENTINEL_KEY}".encode("ascii"), id="not-json-holding-the-key"),
        pytest.param(200, b'{"build": "SYNTHETIC"}', id="no-version-field"),
        pytest.param(200, b'{"version": 31}', id="not-a-string"),
        pytest.param(200, b'{"version": null}', id="null"),
    ],
)
def test_a_failed_version_request_records_null_and_raises_nothing(
    stub_server: StubServer,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    status: int,
    body: bytes,
) -> None:
    caplog.set_level(logging.DEBUG)
    serve_vllm(stub_server, MODEL_ID)
    stub_server.reply("GET", "/version", status=status, raw=body)
    record = serving(keyed(stub_server, monkeypatch))
    assert (record["version"], record["version_from"]) == (None, "GET /version version"), "asked, not reported"
    assert SENTINEL_KEY not in json.dumps(record)
    assert all(SENTINEL_KEY not in logged.getMessage() for logged in caplog.records)
    assert all("Unauthorized" not in logged.getMessage() for logged in caplog.records), "the reply is never logged"


@pytest.mark.parametrize(
    ("server", "path"),
    [("vllm", "/version"), ("sglang", "/server_info"), ("llama-server", "/props")],
)
def test_the_api_key_goes_to_the_version_endpoint(
    stub_server: StubServer, monkeypatch: pytest.MonkeyPatch, server: str, path: str
) -> None:
    serve(stub_server, server)
    serving(keyed(stub_server, monkeypatch))
    (seen,) = stub_server.requests_to("GET", path)
    assert seen.header("Authorization") == f"Bearer {SENTINEL_KEY}"


def test_no_authorization_header_without_api_key_env(stub_server: StubServer) -> None:
    serve_sglang(stub_server, MODEL_ID)
    serving(make_backend(stub_server))
    assert all(seen.header("Authorization") is None for seen in stub_server.requests)


@pytest.mark.parametrize("where", ["entry", "version"])
def test_a_record_that_holds_the_api_key_is_refused(
    stub_server: StubServer, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, where: str
) -> None:
    caplog.set_level(logging.DEBUG)
    entry = serve_vllm(stub_server, MODEL_ID)
    if where == "entry":
        listing = {"object": "list", "data": [{**entry, "root": ECHO}]}
        stub_server.reply("GET", MODELS_PATH, json_body=listing, echo=(ECHO, "Authorization"))
    else:
        stub_server.reply("GET", "/version", json_body={"version": ECHO}, echo=(ECHO, "Authorization"))
    with pytest.raises(ServingError) as info:
        serving(keyed(stub_server, monkeypatch))
    for text in error_texts(info.value):
        assert SENTINEL_KEY not in text and "sentinel-key" not in text, text
    assert all(SENTINEL_KEY not in logged.getMessage() for logged in caplog.records)


def test_serving_reuses_the_checked_entry(stub_server: StubServer) -> None:
    serve_vllm(stub_server, MODEL_ID)
    backend = make_backend(stub_server)
    first = serving(backend)
    backend.complete(MESSAGES, SAMPLING)
    assert len(stub_server.requests_to("GET", MODELS_PATH)) == 1, "complete() reuses the cached entry"
    first["model"]["id"] = "changed"
    assert serving(backend)["model"]["id"] == MODEL_ID, "each record is a fresh copy"


# ---------------------------------------------------------------------------
# Ollama


def test_ollama_serving_records_its_entry_without_a_version(stub_server: StubServer) -> None:
    entry = serve_ollama(stub_server, OLLAMA_ID)
    record = serving(OllamaBackend(OLLAMA_ID, base_url=stub_server.url, timeout_s=30.0))
    assert record == {"base_url": stub_server.url, "model": stable(entry), "version": None, "version_from": None}
    assert stub_server.calls() == [("GET", MODELS_PATH)]


# ---------------------------------------------------------------------------
# Chat replies in each server's shape


@pytest.mark.parametrize("server", ["vllm", "sglang", "llama-server", "furiosa"])
def test_each_servers_chat_reply_parses(stub_server: StubServer, server: str) -> None:
    serve(stub_server, server, reply="SYNTHETIC answer\n")
    completion = make_backend(stub_server).complete(MESSAGES, SAMPLING)
    assert completion.text == "SYNTHETIC answer\n"
    assert (completion.prompt_tokens, completion.completion_tokens) == (PROMPT_TOKENS, COMPLETION_TOKENS)


# ---------------------------------------------------------------------------
# The helpers of lassi.llm._serving


@pytest.mark.parametrize(
    ("base_url", "root"),
    [
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000"),
        ("https://serve.example.invalid/v1", "https://serve.example.invalid"),
        ("http://127.0.0.1:8000/prefix/v1", "http://127.0.0.1:8000/prefix"),
        ("http://127.0.0.1:8000", None),
        ("http://127.0.0.1:8000/v10", None),
        ("http://127.0.0.1:8000/v1/extra", None),
        ("http://127.0.0.1:8000/apiv1", None),
    ],
)
def test_server_root_drops_a_final_v1_segment(base_url: str, root: str | None) -> None:
    assert serving_module().server_root(base_url) == root


def test_version_endpoints_follow_owned_by() -> None:
    module = serving_module()
    assert dict(module.VERSION_ENDPOINTS) == {
        "vllm": ("/version", "version"),
        "sglang": ("/server_info", "version"),
        "llamacpp": ("/props", "build_info"),
    }
    assert module.version_endpoint({"id": "m", "owned_by": "sglang"}) == ("/server_info", "version")
    assert module.version_endpoint({"id": "m", "owned_by": "furiosa-ai"}) is None
    assert module.version_endpoint({"id": "m"}) is None


@pytest.mark.parametrize(
    ("server", "line"),
    [
        pytest.param("vllm", f"owned_by vllm; version {VLLM_VERSION}; max_model_len 32768", id="vllm"),
        pytest.param("sglang", f"owned_by sglang; version {SGLANG_VERSION}; max_model_len 32768", id="sglang"),
        pytest.param(
            "llama-server", f"owned_by llamacpp; version {LLAMA_BUILD_INFO}; meta.n_ctx 4096", id="llama-server"
        ),
        pytest.param(
            "furiosa",
            "owned_by furiosa-ai; version -; max_context_len 131072; max_prompt_len 131072",
            id="furiosa",
        ),
    ],
)
def test_serving_line_names_server_version_and_context(stub_server: StubServer, server: str, line: str) -> None:
    serve(stub_server, server)
    record = serving(make_backend(stub_server))
    assert serving_module().serving_line(copy.deepcopy(record)) == line


def test_serving_line_for_ollama_has_no_version_or_context(stub_server: StubServer) -> None:
    serve_ollama(stub_server, OLLAMA_ID)
    record = serving(OllamaBackend(OLLAMA_ID, base_url=stub_server.url, timeout_s=30.0))
    assert serving_module().serving_line(record) == "owned_by library; version -"


def test_serving_line_shows_a_null_context_field_as_a_dash(stub_server: StubServer) -> None:
    serve_vllm(stub_server, MODEL_ID, loras=[LORA_ID])
    record = serving(make_backend(stub_server, LORA_ID))
    assert serving_module().serving_line(record) == f"owned_by vllm; version {VLLM_VERSION}; max_model_len -"
