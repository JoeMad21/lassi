"""Stub replies in the shapes of vLLM, SGLang, llama.cpp's server, furiosa-llm, and Ollama (task P17.3).

Each serve_* function configures the in-process StubServer of conftest.py
(127.0.0.1, an OS-chosen port; the autouse no_network fixture refuses every
other host) with one server's model list, its version route when it has one,
and its non-streaming chat reply, and returns the model entry it serves for
`name`. Every value served is SYNTHETIC, chosen for the tests; only the shapes
(which keys sit where) follow the sources below. The stub enforces no API key:
tests read the Authorization header it records.

The shapes are source-read in plans/spikes/p17-frameworks.md, Results 4, at
vLLM v0.31.0 (V, E/ = vllm/entrypoints/), SGLang v0.5.21 (S, E/ =
python/sglang/srt/entrypoints/), and llama.cpp v0.6.0 (L, T/ = tools/server/):

- vLLM: the list `{object, data}` (V E/serve/engine/protocol.py:117-118); a
  card {id, object, created, owned_by "vllm", root, parent, max_model_len,
  permission} (:106-113), the permission item a ModelPermission whose id and
  created are new on every GET (:91-102); a LoRA card has max_model_len null
  and its base as parent (E/openai/models/serving.py:152-165); GET /version
  gives {"version": ...} (V E/serve/instrumentator/basic.py:53-56); chat
  usage prompt_tokens, completion_tokens, total_tokens, and
  prompt_tokens_details null (E/openai/chat_completion/serving.py:1146-1160),
  and a system_fingerprint (E/serve/utils/fingerprint.py:58-84).
- SGLang: a card {id, object, created, owned_by "sglang", root (the served
  name), parent, max_model_len} with no permission (S E/openai/protocol.py:81-90;
  E/http_server.py:1898-1910); GET /server_info gives version (:867) beside the
  resolved server arguments (:862), api_key and admin_api_key among them
  (python/sglang/srt/arg_groups/fields/serving.py:153-160), and launch_command
  (:863); chat usage with reasoning_tokens at the top level of usage and no
  system_fingerprint (E/openai/usage_processor.py:26-45, :123;
  protocol.py:1246-1254).
- llama-server (single-model mode): the list {models, object, data}, data one
  entry (L T/server-context.cpp:4926-4951); the entry {id, aliases, tags,
  object, architecture, created, owned_by "llamacpp", meta{vocab_type,
  n_vocab, n_ctx, n_ctx_train, n_embd, n_params, size, ftype}}, with no
  max_model_len (:4894-4920); GET /props gives build_info (:4989,
  :5160-5170); chat usage with prompt_tokens_details.cached_tokens, the
  server's own model name, system_fingerprint, and timings
  (T/server-task.cpp:365-372, :444-457). The spike does not record the inner
  shape of `architecture` or of the `models` items; the stub's are SYNTHETIC.
- furiosa-llm: a recorded reply, not a source read. The entry's keys and
  values are those plans/spikes/p3-rngd-demo.md:117-118 recorded, except the
  id, which is the test's `name`; no version route is known.
- Ollama: the existing backend's routes (lassi/llm/ollama.py): GET
  /v1/models, POST /api/chat, and POST /api/generate for the unload. The
  owned_by value is SYNTHETIC, not source-read.

No value in this module is a measurement.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conftest import StubServer

MODELS_PATH = "/v1/models"
CHAT_PATH = "/v1/chat/completions"
OLLAMA_CHAT_PATH = "/api/chat"
OLLAMA_GENERATE_PATH = "/api/generate"

# SYNTHETIC values, in the formats the sources give.
VLLM_VERSION = "0.31.0"
SGLANG_VERSION = "0.5.21"
LLAMA_BUILD_INFO = "b11433-d8123504"
CREATED = 1790000000
VLLM_ROOT = "fixture-org/fixture-weights"
SGLANG_MODEL_PATH = "fixture-org/fixture-weights"
DEFAULT_REPLY = "SYNTHETIC reply text\n"
PROMPT_TOKENS = 11
COMPLETION_TOKENS = 7
# The furiosa-llm entry's own fields as plans/spikes/p3-rngd-demo.md:117-118 recorded them.
FURIOSA_CREATED = 1790227963
FURIOSA_ARTIFACT_ID = "d6ae6a43-6ce0-4864-aaca-eeac4340234c"
FURIOSA_MAX_LEN = 131072
OLLAMA_OWNER = "library"


def usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    """Return the three counts every server's usage block holds."""
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def chat_reply(model: str, reply: str, extra_usage: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Return a non-streaming chat completion with `reply` as the content, the usage counts, and `extra` keys."""
    return {
        "id": "chatcmpl-SYNTHETIC",
        "object": "chat.completion",
        "created": CREATED,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
        "usage": {**usage(PROMPT_TOKENS, COMPLETION_TOKENS), **extra_usage},
        **extra,
    }


def serve_vllm(
    stub: StubServer,
    name: str,
    *,
    max_model_len: int = 32768,
    version: str = VLLM_VERSION,
    loras: Sequence[str] = (),
    reply: str = DEFAULT_REPLY,
) -> dict[str, Any]:
    """Serve vLLM's model list (the base card, then a card per LoRA name), /version, and chat; return the base card."""
    cards = [vllm_card(name, max_model_len=max_model_len, parent=None, serial=0)]
    cards += [vllm_card(lora, max_model_len=None, parent=name, serial=index + 1) for index, lora in enumerate(loras)]
    stub.reply("GET", MODELS_PATH, json_body={"object": "list", "data": cards})
    stub.reply("GET", "/version", json_body={"version": version})
    fingerprint = f"vllm-{version}-0a1b2c3d"
    body = chat_reply(name, reply, {"prompt_tokens_details": None}, system_fingerprint=fingerprint)
    stub.reply("POST", CHAT_PATH, json_body=body)
    return copy.deepcopy(cards[0])


def vllm_card(name: str, *, max_model_len: int | None, parent: str | None, serial: int) -> dict[str, Any]:
    """Return one vLLM model card; a LoRA card has its base as parent and max_model_len null."""
    root = VLLM_ROOT if parent is None else f"fixture-adapters/{name}"
    return {
        "id": name,
        "object": "model",
        "created": CREATED,
        "owned_by": "vllm",
        "root": root,
        "parent": parent,
        "max_model_len": max_model_len,
        "permission": [{"id": f"modelperm-SYNTHETIC{serial:04d}", "created": CREATED}],
    }


def serve_sglang(
    stub: StubServer,
    name: str,
    *,
    context_len: int = 32768,
    version: str = SGLANG_VERSION,
    api_key: str | None = None,
    reply: str = DEFAULT_REPLY,
) -> dict[str, Any]:
    """Serve SGLang's model list, /server_info (holding `api_key` as the server would), and chat; return the card."""
    card = {
        "id": name,
        "object": "model",
        "created": CREATED,
        "owned_by": "sglang",
        "root": name,
        "parent": None,
        "max_model_len": context_len,
    }
    stub.reply("GET", MODELS_PATH, json_body={"object": "list", "data": [card]})
    command = f"python3 -m sglang.launch_server --model-path {SGLANG_MODEL_PATH} --served-model-name {name}"
    if api_key is not None:
        command += f" --api-key {api_key}"
    info = {
        "model_path": SGLANG_MODEL_PATH,
        "served_model_name": name,
        "api_key": api_key,
        "admin_api_key": api_key,
        "version": version,
        "launch_command": command,
    }
    stub.reply("GET", "/server_info", json_body=info)
    stub.reply("POST", CHAT_PATH, json_body=chat_reply(name, reply, {"reasoning_tokens": 0}))
    return copy.deepcopy(card)


def serve_llama_server(
    stub: StubServer,
    alias: str,
    *,
    n_ctx: int = 4096,
    n_ctx_train: int = 32768,
    build_info: str = LLAMA_BUILD_INFO,
    reply: str = DEFAULT_REPLY,
) -> dict[str, Any]:
    """Serve llama-server's single-model list, /props, and chat; return the data entry."""
    entry = {
        "id": alias,
        "aliases": [alias],
        "tags": [],
        "object": "model",
        "architecture": {"family": "SYNTHETIC"},
        "created": CREATED,
        "owned_by": "llamacpp",
        "meta": {
            "vocab_type": 2,
            "n_vocab": 32000,
            "n_ctx": n_ctx,
            "n_ctx_train": n_ctx_train,
            "n_embd": 1024,
            "n_params": 500000000,
            "size": 1000000000,
            "ftype": 1,
        },
    }
    listing = {"models": [{"name": alias, "model": alias}], "object": "list", "data": [entry]}
    stub.reply("GET", MODELS_PATH, json_body=listing)
    stub.reply("GET", "/props", json_body={"build_info": build_info})
    timings = {"prompt_n": PROMPT_TOKENS, "predicted_n": COMPLETION_TOKENS}
    body = chat_reply(
        alias, reply, {"prompt_tokens_details": {"cached_tokens": 0}}, system_fingerprint=build_info, timings=timings
    )
    stub.reply("POST", CHAT_PATH, json_body=body)
    return copy.deepcopy(entry)


def serve_furiosa(stub: StubServer, name: str, *, reply: str = DEFAULT_REPLY) -> dict[str, Any]:
    """Serve the model list furiosa-llm gave in the P3 spike (with `name` as the id) and chat; return the entry."""
    entry = {
        "id": name,
        "created": FURIOSA_CREATED,
        "object": "model",
        "owned_by": "furiosa-ai",
        "artifact_id": FURIOSA_ARTIFACT_ID,
        "max_prompt_len": FURIOSA_MAX_LEN,
        "max_context_len": FURIOSA_MAX_LEN,
    }
    stub.reply("GET", MODELS_PATH, json_body={"object": "list", "data": [entry]})
    stub.reply("POST", CHAT_PATH, json_body=chat_reply(name, reply, {}))
    return copy.deepcopy(entry)


def serve_ollama(stub: StubServer, name: str, *, reply: str = DEFAULT_REPLY) -> dict[str, Any]:
    """Serve Ollama's model list, its chat reply, and its unload reply; return the model entry."""
    entry = {"id": name, "object": "model", "created": CREATED, "owned_by": OLLAMA_OWNER}
    stub.reply("GET", MODELS_PATH, json_body={"object": "list", "data": [entry]})
    chat = {
        "model": name,
        "message": {"role": "assistant", "content": reply},
        "done": True,
        "prompt_eval_count": PROMPT_TOKENS,
        "eval_count": COMPLETION_TOKENS,
    }
    stub.reply("POST", OLLAMA_CHAT_PATH, json_body=chat)
    stub.reply("POST", OLLAMA_GENERATE_PATH, json_body={"model": name, "response": "", "done": True})
    return copy.deepcopy(entry)


def stable(entry: dict[str, Any]) -> dict[str, Any]:
    """Return the entry the serving record keeps: every field as served except created and permission."""
    return {key: copy.deepcopy(value) for key, value in entry.items() if key not in ("created", "permission")}
