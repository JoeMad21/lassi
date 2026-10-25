"""Tests for the opt-in sampling key llm.sampling.reasoning_effort and the length-cut empty answer (OQ-080).

Bible: Model Serving (Serving Rules, the gpt-oss line: fix reasoning effort
per arm and record it), Project Recipes (Notes: an opt-in key is off when
absent and never materialized), Design Principle 4 and Agent Rule 4 (a
departure from upstream is refused under faithful: true). plans/OWNER-QUEUE.md
OQ-080, option (a).

The contract these tests fix:

1. llm.sampling.reasoning_effort is absent by default and the loader
   materializes nothing, so a recipe without it keeps its resolved recipe and
   hash. Set, it is one of low, medium, or high, enters the resolved recipe
   as written, and changes the hash; anything else, a null and a non-string
   included, is refused at load naming the key. Under faithful: true, set in
   the recipe or inherited, it is refused at load naming the key.
2. The runner builds a backend that declares the capability
   `reasoning_effort` (lassi.core.capabilities REASONING_EFFORT) as
   factory(model.id, **config, reasoning_effort=<value>) when the key is set,
   and as before when it is not. A backend that does not declare it is
   refused, before any directory exists, naming the key and the backend.
3. openai_compat declares it and sends `"reasoning_effort": <value>` at the
   top level of the chat body only when it is set; replay declares it too
   and replays the same completions with or without it.
4. openai_compat's complete() returns a Completion with text "" and the
   server's token counts when choices[0].message.content is JSON null and
   choices[0].finish_reason is "length" (the model gave no answer within
   max_tokens). A missing content key, a null content with any other
   finish_reason or none, and missing token counts stay ServingError. The
   reasoning text is never returned or written to the run tree.

Every request goes to the stub on 127.0.0.1; the replies, reasoning texts,
token counts, and bench sources are SYNTHETIC. No value here is a measurement.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from served_stubs import CHAT_PATH
from test_served_runs import (
    FAKE_COMMIT,
    GOOD_REPLY,
    ITEM,
    KEY_ENV,
    MODEL_ID,
    SOURCES,
    SUITE_MANIFEST,
    PlainBackend,
    at_v1,
    make_registry,
    manifest,
    model,
    recipe_data,
    refused,
    run,
    serve,
)

from lassi.bench import load_suite
from lassi.core import runner as runner_module
from lassi.core.capabilities import REASONING_EFFORT, REASONING_EFFORTS
from lassi.core.interfaces import Completion, Message, Sampling
from lassi.core.recipe import Recipe, RecipeError, load_recipe
from lassi.core.registry import Registry
from lassi.llm import OpenAICompatBackend, ReplayBackend, ServingError

if TYPE_CHECKING:
    from conftest import StubServer

KEY = "llm.sampling.reasoning_effort"
MODELS_PATH = "/v1/models"
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=16384)
MESSAGES = (Message(role="user", content="Translate main.cpp. SYNTHETIC"),)
# SYNTHETIC reasoning text a length-cut reply carries beside its null content; it must never surface.
REASONING_MARKER = "reasoning-marker-SYNTHETIC-7d2e"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
REPLAY_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "replay" / "recordings"


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate, compile, and key variables, point TMPDIR at a test directory, and fake git."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(KEY_ENV, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    answers = {"rev-parse": FAKE_COMMIT + "\n", "status": ""}
    monkeypatch.setattr(runner_module, "_git", lambda *args: answers[args[0]])


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """Write the item's SYNTHETIC sources where the suite manifest lays them out; return the bench root."""
    root = tmp_path / "bench"
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, text in SOURCES.items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


# ---------------------------------------------------------------------------
# Helpers


def with_effort(data: dict[str, Any], effort: Any) -> dict[str, Any]:
    """Return `data` with llm.sampling.reasoning_effort set to `effort`."""
    data["llm"]["sampling"][REASONING_EFFORT] = effort
    return data


def load(tmp_path: Path, data: Mapping[str, Any], name: str = "loaded") -> Recipe:
    """Write `data` as <tmp_path>/<name>.yaml and load it with the test registry."""
    path = tmp_path / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return load_recipe(path, registry=make_registry())


def load_refused(tmp_path: Path, data: Mapping[str, Any], name: str = "refused") -> str:
    """Load `data`, expect a RecipeError, and return its message."""
    with pytest.raises(RecipeError) as info:
        load(tmp_path, data, name)
    return str(info.value)


def effort_registry(seen: list[dict[str, Any]]) -> Registry:
    """Return the served test registry plus "effortllm", which declares the capability and records its settings."""

    class EffortBackend(PlainBackend):
        """An LLMBackend "effortllm" that declares reasoning_effort and records the settings it was built with."""

        name = "effortllm"
        capabilities = frozenset({"chat", REASONING_EFFORT})

        def __init__(self, model_id: str, **settings: Any) -> None:
            """Keep the model id and record `settings`."""
            super().__init__(model_id)
            seen.append(dict(settings))

    registry = make_registry()
    registry.register("LLMBackend", "effortllm", EffortBackend)
    return registry


def length_cut(content: Any = None, finish_reason: Any = "length", **usage: Any) -> dict[str, Any]:
    """Return a chat reply whose answer ran out of tokens: `content` beside SYNTHETIC reasoning text."""
    message = {"role": "assistant", "content": content, "reasoning": REASONING_MARKER}
    choice: dict[str, Any] = {"index": 0, "message": message}
    if finish_reason is not ...:
        choice["finish_reason"] = finish_reason
    counts = {"prompt_tokens": 120, "completion_tokens": 16384, "total_tokens": 16504, **usage}
    return {"id": "chatcmpl-SYNTHETIC", "object": "chat.completion", "model": MODEL_ID, "choices": [choice],
            "usage": {key: value for key, value in counts.items() if value is not ...}}


def openai_backend(stub: StubServer, reply: Mapping[str, Any], **kwargs: Any) -> OpenAICompatBackend:
    """Serve MODEL_ID and `reply` on the stub; return a backend pointed at it."""
    stub.serve_models([MODEL_ID])
    stub.reply("POST", CHAT_PATH, json_body=dict(reply))
    return OpenAICompatBackend(MODEL_ID, base_url=f"{stub.url}/v1", **kwargs)


# ---------------------------------------------------------------------------
# 1. The recipe key


def test_the_capability_and_its_values() -> None:
    assert REASONING_EFFORT == "reasoning_effort"
    assert REASONING_EFFORTS == ("low", "medium", "high")


def test_absent_by_default_and_never_materialized(tmp_path: Path) -> None:
    recipe = load(tmp_path, recipe_data(model("plainllm")))
    assert REASONING_EFFORT not in recipe.data["llm"]["sampling"]
    assert REASONING_EFFORT not in recipe.canonical_yaml


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_a_set_value_enters_the_resolved_recipe_and_hash(tmp_path: Path, effort: str) -> None:
    plain = load(tmp_path, recipe_data(model("plainllm")), "plain")
    chosen = load(tmp_path, with_effort(recipe_data(model("plainllm")), effort), "chosen")
    assert chosen.data["llm"]["sampling"][REASONING_EFFORT] == effort
    assert f"    {REASONING_EFFORT}: {effort}\n" in chosen.canonical_yaml
    assert chosen.recipe_hash != plain.recipe_hash


@pytest.mark.parametrize(
    "effort",
    [None, "LOW", "Low", "minimal", "none", "", " low", 1, 0.5, True, ["low"], {"level": "low"}],
    ids=["null", "upper", "title", "minimal", "none", "empty", "space", "int", "float", "bool", "list", "mapping"],
)
def test_any_other_value_is_refused_naming_the_key(tmp_path: Path, effort: Any) -> None:
    message = load_refused(tmp_path, with_effort(recipe_data(model("plainllm")), effort))
    assert KEY in message, message


def test_a_faithful_recipe_without_the_key_loads(tmp_path: Path) -> None:
    data = recipe_data(model("plainllm"))
    data["faithful"] = True
    assert load(tmp_path, data).data["faithful"] is True


def test_refused_under_faithful_set_in_the_recipe(tmp_path: Path) -> None:
    data = with_effort(recipe_data(model("plainllm")), "low")
    data["faithful"] = True
    message = load_refused(tmp_path, data)
    assert KEY in message and "faithful: true" in message, message


@pytest.mark.parametrize("parent_sets", ["effort", "faithful"])
def test_refused_under_faithful_inherited(tmp_path: Path, parent_sets: str) -> None:
    parent = recipe_data(model("plainllm"))
    child: dict[str, Any] = {"extends": "parent.yaml"}
    if parent_sets == "effort":
        with_effort(parent, "high")
        child["faithful"] = True
    else:
        parent["faithful"] = True
        child["llm"] = {"sampling": {REASONING_EFFORT: "high"}}
    (tmp_path / "parent.yaml").write_bytes(yaml.safe_dump(parent, sort_keys=False).encode("ascii"))
    message = load_refused(tmp_path, child, "child")
    assert KEY in message and "faithful: true" in message, message


# ---------------------------------------------------------------------------
# 2. The runner and the backend capability


def test_a_backend_that_declares_it_is_built_with_the_value(tmp_path: Path, bench: Path) -> None:
    seen: list[dict[str, Any]] = []
    run_dir = run(tmp_path, bench, with_effort(recipe_data(model("effortllm")), "medium"), effort_registry(seen))
    assert manifest(run_dir)["status"] == "complete"
    assert seen == [{REASONING_EFFORT: "medium"}]
    resolved = (run_dir / "recipe.resolved.yaml").read_bytes().decode("ascii")
    assert f"    {REASONING_EFFORT}: medium\n" in resolved


def test_without_the_key_the_backend_is_built_as_before(tmp_path: Path, bench: Path) -> None:
    seen: list[dict[str, Any]] = []
    run_dir = run(tmp_path, bench, recipe_data(model("effortllm")), effort_registry(seen))
    assert manifest(run_dir)["status"] == "complete"
    assert seen == [{}]
    assert REASONING_EFFORT not in (run_dir / "recipe.resolved.yaml").read_bytes().decode("ascii")


def test_a_backend_that_does_not_declare_it_is_refused_before_any_directory(tmp_path: Path, bench: Path) -> None:
    message = str(refused(tmp_path, bench, with_effort(recipe_data(model("plainllm")), "low")))
    assert KEY in message and "'plainllm'" in message and repr(REASONING_EFFORT) in message, message


# ---------------------------------------------------------------------------
# 3. openai_compat and replay


def test_openai_compat_declares_it_and_takes_it_keyword_only() -> None:
    assert REASONING_EFFORT in OpenAICompatBackend.capabilities
    parameter = inspect.signature(OpenAICompatBackend).parameters[REASONING_EFFORT]
    assert (parameter.kind, parameter.default) == (inspect.Parameter.KEYWORD_ONLY, None)


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_openai_compat_sends_it_at_the_top_level(stub_server: StubServer, effort: str) -> None:
    backend = openai_backend(stub_server, length_cut("ok", "stop"), reasoning_effort=effort)
    backend.complete(MESSAGES, SAMPLING)
    (chat,) = stub_server.requests_to("POST", CHAT_PATH)
    assert chat.body == {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": MESSAGES[0].content}],
        "temperature": SAMPLING.temperature,
        "top_p": SAMPLING.top_p,
        "max_tokens": SAMPLING.max_tokens,
        "stream": False,
        "reasoning_effort": effort,
    }


def test_openai_compat_sends_no_field_without_it(stub_server: StubServer) -> None:
    openai_backend(stub_server, length_cut("ok", "stop")).complete(MESSAGES, SAMPLING)
    (chat,) = stub_server.requests_to("POST", CHAT_PATH)
    assert list(chat.body) == ["model", "messages", "temperature", "top_p", "max_tokens", "stream"]


@pytest.mark.parametrize("effort", ["LOW", "minimal", "", 1, True])
def test_openai_compat_refuses_another_value_at_construction(stub_server: StubServer, effort: Any) -> None:
    with pytest.raises(ValueError, match=REASONING_EFFORT):
        OpenAICompatBackend(MODEL_ID, base_url=f"{stub_server.url}/v1", reasoning_effort=effort)
    assert stub_server.requests == []


def test_the_runner_sends_it_to_openai_compat(tmp_path: Path, bench: Path, stub_server: StubServer) -> None:
    serve(stub_server, "vllm")
    run_dir = run(tmp_path, bench, with_effort(recipe_data(at_v1(stub_server)), "high"))
    assert manifest(run_dir)["status"] == "complete"
    chats = stub_server.requests_to("POST", CHAT_PATH)
    assert chats and all(chat.body["reasoning_effort"] == "high" for chat in chats)


def test_replay_declares_it_and_replays_the_same_completions() -> None:
    recordings = sorted(REPLAY_FIXTURE.glob("*.json"))
    assert recordings, REPLAY_FIXTURE
    recording = recordings[0]
    assert REASONING_EFFORT in ReplayBackend.capabilities
    plain = ReplayBackend(MODEL_ID, recording=recording)
    chosen = ReplayBackend(MODEL_ID, recording=recording, reasoning_effort="low")
    assert chosen.recording_sha256 == plain.recording_sha256
    assert chosen._completions == plain._completions
    with pytest.raises(ValueError, match=REASONING_EFFORT):
        ReplayBackend(MODEL_ID, recording=recording, reasoning_effort="max")


# ---------------------------------------------------------------------------
# 4. The length-cut empty answer


def test_a_null_content_cut_at_length_is_an_empty_answer(stub_server: StubServer) -> None:
    completion = openai_backend(stub_server, length_cut()).complete(MESSAGES, SAMPLING)
    assert completion == Completion(text="", prompt_tokens=120, completion_tokens=16384)
    assert REASONING_MARKER not in repr(completion)


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(length_cut(finish_reason="stop"), id="null-stop"),
        pytest.param(length_cut(finish_reason="content_filter"), id="null-content-filter"),
        pytest.param(length_cut(finish_reason=None), id="null-finish-reason"),
        pytest.param(length_cut(finish_reason=...), id="no-finish-reason"),
        pytest.param(length_cut(finish_reason="LENGTH"), id="null-upper-length"),
        pytest.param(length_cut(prompt_tokens=...), id="length-no-prompt-tokens"),
        pytest.param(length_cut(completion_tokens=None), id="length-null-completion-tokens"),
    ],
)
def test_other_null_contents_stay_serving_errors(stub_server: StubServer, reply: dict[str, Any]) -> None:
    with pytest.raises(ServingError) as info:
        openai_backend(stub_server, reply).complete(MESSAGES, SAMPLING)
    assert f"POST {CHAT_PATH} returned HTTP 200" in str(info.value), str(info.value)


def test_a_missing_content_cut_at_length_stays_a_serving_error(stub_server: StubServer) -> None:
    reply = length_cut()
    del reply["choices"][0]["message"]["content"]
    with pytest.raises(ServingError, match="choices"):
        openai_backend(stub_server, reply).complete(MESSAGES, SAMPLING)


def test_a_length_cut_run_completes_and_records_no_reasoning(
    tmp_path: Path, bench: Path, stub_server: StubServer
) -> None:
    serve(stub_server, "vllm")
    stub_server.reply("POST", CHAT_PATH, json_body=length_cut())
    run_dir = run(tmp_path, bench, recipe_data(at_v1(stub_server)))
    assert manifest(run_dir)["status"] == "complete"
    trial_files = sorted(run_dir.rglob("trial.json"))
    assert trial_files
    # trial.json keeps each reply as a text-store reference; every reply here is the empty answer.
    attempts = json.loads(trial_files[0].read_bytes())["attempts"]
    assert attempts and all(attempt["response_text"]["sha256"] == EMPTY_SHA256 for attempt in attempts)
    for path in run_dir.rglob("*"):
        if path.is_file():
            assert REASONING_MARKER.encode("ascii") not in path.read_bytes(), path
    assert GOOD_REPLY not in json.dumps(attempts)
