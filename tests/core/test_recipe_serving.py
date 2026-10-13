"""Tests for the server keys of a recipe's model section (task P17.3): at load, in the hash, never a key value.

Bible: Project Recipes (Notes), Model Serving (Serving Rules), Agent Rule 12;
plans/p17-portable.md, task P17.3; PHASE-NOTES P0 (the runner note on HTTP
settings) and P17 (the missing seam "Recipe HTTP settings").

The contract these tests fix:

- model.base_url (a string), model.timeout_s (a finite number above zero),
  and model.api_key_env (an environment variable name) are optional keys of
  the model section. The loader fills none in, so a recipe without them
  resolves and hashes exactly as before this task; a key that is set enters
  the canonical YAML and the recipe hash as written.
- The backend's binding carries every model key except backend and id as its
  config, beside any device section (task P17.2), so the runner can build
  the backend as factory(model.id, **config).
- openai_compat accepts all three keys and ollama base_url and timeout_s
  (their config_keys); mock and replay accept none. The registry's config
  check refuses any other, naming model.<key>.
- api_key_env holds a variable name: uppercase letters, digits, and '_', not
  starting with a digit, at most 64 characters. A value outside that shape,
  such as a key pasted in by mistake, is refused at load whatever the backend,
  naming model.api_key_env, and the message shows no part of the value (Agent
  Rule 12: a key is never stored, hashed, or written). Known cases, not a
  complete list: an all-uppercase key of 64 characters or fewer would pass.
- The model section merges key by key, so a child overrides one key and keeps
  the others.

The backend classes are the real ones, registered in a test Registry and never
constructed (loading constructs nothing); every other component is a fake that
fails when constructed. Every key value below is SYNTHETIC. No value in this
module is a measurement.
"""

from __future__ import annotations

import copy
import inspect
import traceback
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.registry import DEFAULT_REGISTRY, Binding, Registry
from lassi.llm import MockBackend, OllamaBackend, OpenAICompatBackend, ReplayBackend

TAKES_DEVICE = "takes_device"
MODEL_ID = "fixture/coder-a"
OLLAMA_ID = "coder-7b-q8_0"
KEY_ENV = "LASSI_TEST_API_KEY"
BASE_URL = "http://127.0.0.1:8000/v1"
SERVER_KEYS: dict[str, Any] = {"base_url": BASE_URL, "timeout_s": 120, "api_key_env": KEY_ENV}
CPU = {"kind": "cpu"}
BACKENDS = {"openai_compat": OpenAICompatBackend, "ollama": OllamaBackend, "mock": MockBackend, "replay": ReplayBackend}
ACCEPTED = {
    "openai_compat": frozenset({"base_url", "timeout_s", "api_key_env"}),
    "ollama": frozenset({"base_url", "timeout_s"}),
    "mock": frozenset(),
    "replay": frozenset(),
}
# The sha256 of the canonical YAML of recipe() loaded as serving.yaml, computed with lassi.core.recipe at 03441ee,
# before this task; a hash is a function of its input text, not a measurement.
WITHOUT_KEYS_HASH = "860518f22834890780add74c98e845b5b311a09e60b1b72a78471b091c8a538c"
# A shown value is never quoted whole, nor any run of this many of its characters.
WINDOW = 6

# A complete recipe that extends nothing, with fake components beside the real backends.
STANDALONE: dict[str, Any] = {
    "llm": {"sampling": {"temperature": 0.2, "top_p": 0.9}},
    "loop": {"max_corrections": 3},
    "trials": {"n": 1},
    "runs_root": "fixture-runs",
    "sandbox": {"network": False, "wall_s": 60, "mem_gb": 4},
    "report": {"trial_md": True, "parquet": True},
    "bench": {"suite": "fixture-suite", "split": "eval"},
    "directions": [{"source": "omp", "target": "cuda"}],
    "stages": ["generate"],
    "executor": {"kind": "plainexec"},
    "model": {"backend": "openai_compat", "id": MODEL_ID},
}

# SYNTHETIC key-shaped values (none is a real key) that api_key_env must refuse, each without showing any part.
KEY_VALUES = [
    pytest.param("sk-q7Zr2Lw9Xv4Tn8Kp3Ym6Hd1", id="dash"),
    pytest.param("hf_Q7zR2lW9xV4tN8kP3yM6hD1", id="lowercase-prefix"),
    pytest.param("Gq7Zr2Lw9Xv4Tn8Kp3Ym6", id="mixed-case"),
    pytest.param("7QZR2LW9XV4TN8KP3YM6HD1", id="leading-digit"),
    pytest.param(("Q7ZR2LW9XV4TN8KP3YM6HD1" * 3)[:65], id="65-characters"),
    pytest.param("Q7ZR2LW9 XV4TN8KP3YM6", id="space"),
    pytest.param("Q7ZR2LW9.XV4TN8KP3YM6", id="dot"),
    pytest.param("Q7ZR2LW9/XV4TN8+KP3YM6=", id="slash-plus-equals"),
    pytest.param("", id="empty"),
    pytest.param(31415926535897, id="integer"),
    pytest.param(["Q7ZR2LW9XV4TN8KP3YM6"], id="list"),
]


class NeverBuilt:
    """Base for every fake component: constructing one fails the test, since loading constructs nothing."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(f"component {type(self).__name__} was constructed while loading a recipe")


def fake(name: str, caps: Iterable[str] = (), *, config_keys: Iterable[str] = ()) -> type:
    """Return a fake component class with the class attributes the registry reads."""
    namespace: dict[str, Any] = {
        "__doc__": f"Fake component {name} for the server-key tests; never constructed.",
        "name": name,
        "capabilities": frozenset(caps),
        "config_keys": frozenset(config_keys),
    }
    return type(f"Fake_{name}", (NeverBuilt,), namespace)


def registry() -> Registry:
    """Return a Registry of the real backend classes, a fake backend that takes a device, and fake others."""
    found = Registry()
    for name, cls in BACKENDS.items():
        found.register("LLMBackend", name, cls)
    found.register("LLMBackend", "devllm", fake("devllm", {"chat", TAKES_DEVICE}, config_keys={"base_url"}))
    found.register("Executor", "plainexec", fake("plainexec", {"runs_code"}))
    found.register("Stage", "generate", fake("generate"))
    return found


def write(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe <directory>/<name>.yaml (ASCII, LF) and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def model(backend: str = "openai_compat", model_id: str = MODEL_ID, **keys: Any) -> dict[str, Any]:
    """Return a model section for `backend` with the given server keys."""
    return {"backend": backend, "id": model_id, **copy.deepcopy(keys)}


def recipe(**changes: Any) -> dict[str, Any]:
    """Return a copy of STANDALONE with top-level keys replaced by `changes`."""
    data = copy.deepcopy(STANDALONE)
    data.update(copy.deepcopy(changes))
    return data


def load(tmp_path: Path, data: Mapping[str, Any], name: str = "serving") -> Any:
    """Write and load `data` with the test registry and tmp_path as the only root."""
    return load_recipe(write(tmp_path, name, data), roots=[tmp_path], registry=registry())


def load_error(tmp_path: Path, data: Mapping[str, Any], name: str = "serving") -> RecipeError:
    """Return the RecipeError `data` raises; its message must name the recipe file."""
    with pytest.raises(RecipeError) as info:
        load(tmp_path, data, name)
    assert f"{name}.yaml" in str(info.value), str(info.value)
    return info.value


def backend_binding(loaded: Any) -> Binding:
    """Return the loaded recipe's one LLMBackend binding."""
    (found,) = [item for item in loaded.bindings if item.interface == "LLMBackend"]
    return found


def has_key_named(value: Any, key: str) -> bool:
    """Return True when a mapping anywhere inside `value` holds `key`."""
    if isinstance(value, dict):
        return key in value or any(has_key_named(item, key) for item in value.values())
    if isinstance(value, list):
        return any(has_key_named(item, key) for item in value)
    return False


def error_texts(error: BaseException) -> list[str]:
    """Return every text an error shows: str, repr, args, and the formatted traceback with its chain."""
    texts = [str(error), repr(error), repr(error.args)]
    texts.append("".join(traceback.format_exception(type(error), error, error.__traceback__)))
    return texts


def assert_shows_no_part(error: BaseException, value: Any) -> None:
    """Fail when any text of `error` holds `value` (as text) or any WINDOW characters of it in a row."""
    shown = value if isinstance(value, str) else str(value[0] if isinstance(value, list) else value)
    parts = [shown[start : start + WINDOW] for start in range(max(len(shown) - WINDOW + 1, 0))]
    for text in error_texts(error):
        for part in [shown, *parts]:
            if part:
                assert part not in text, (part, text)


# ---------------------------------------------------------------------------
# Loading, the canonical YAML, and the hash


def test_server_keys_load_and_enter_the_hash(tmp_path: Path) -> None:
    plain = load(tmp_path, recipe(), "serving")
    keyed = load(tmp_path, recipe(model=model(**SERVER_KEYS)), "serving")
    assert keyed.data["model"] == {"backend": "openai_compat", "id": MODEL_ID, **SERVER_KEYS}
    assert yaml.safe_load(keyed.canonical_yaml)["model"] == keyed.data["model"], "the canonical YAML holds them"
    assert keyed.recipe_hash != plain.recipe_hash
    for key, other in (("base_url", "http://127.0.0.1:8001/v1"), ("timeout_s", 121), ("api_key_env", "OTHER_KEY")):
        changed = load(tmp_path, recipe(model=model(**{**SERVER_KEYS, key: other})), "serving")
        assert changed.recipe_hash != keyed.recipe_hash, key
    again = load(tmp_path, recipe(model=model(**SERVER_KEYS)), "serving")
    assert again.recipe_hash == keyed.recipe_hash, "the same keys hash the same"


def test_server_keys_enter_the_recipe_as_written(tmp_path: Path) -> None:
    # The loader never rewrites a value: base_url keeps its trailing '/' here, though the backend drops it when built.
    slashed = load(tmp_path, recipe(model=model(base_url=BASE_URL + "/")), "serving")
    bare = load(tmp_path, recipe(model=model(base_url=BASE_URL)), "serving")
    assert slashed.data["model"]["base_url"] == BASE_URL + "/"
    assert slashed.recipe_hash != bare.recipe_hash


def test_a_recipe_without_server_keys_resolves_without_them(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe(), "serving")
    assert loaded.data["model"] == {"backend": "openai_compat", "id": MODEL_ID}, "the loader fills nothing in"
    assert not any(has_key_named(loaded.data, key) for key in SERVER_KEYS)
    assert dict(backend_binding(loaded).config) == {}
    assert loaded.recipe_hash == WITHOUT_KEYS_HASH, "the hash before this task"


@pytest.mark.parametrize(
    ("backend", "keys"),
    [
        pytest.param("openai_compat", SERVER_KEYS, id="openai_compat-all"),
        pytest.param("openai_compat", {"base_url": BASE_URL}, id="openai_compat-base_url"),
        pytest.param("openai_compat", {"timeout_s": 0.5}, id="openai_compat-timeout_s"),
        pytest.param("openai_compat", {"api_key_env": KEY_ENV}, id="openai_compat-api_key_env"),
        pytest.param("ollama", {"base_url": "http://127.0.0.1:11434", "timeout_s": 600}, id="ollama-both"),
    ],
)
def test_server_keys_reach_the_backend_binding(tmp_path: Path, backend: str, keys: Mapping[str, Any]) -> None:
    loaded = load(tmp_path, recipe(model=model(backend, OLLAMA_ID if backend == "ollama" else MODEL_ID, **keys)))
    found = backend_binding(loaded)
    assert (found.name, found.where, dict(found.config)) == (backend, "model.backend", dict(keys))


def test_server_keys_sit_beside_a_device_section_in_the_binding(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe(model=model("devllm", base_url=BASE_URL, device=CPU)))
    assert dict(backend_binding(loaded).config) == {"base_url": BASE_URL, "device": CPU}


def test_backend_config_keys_are_declared() -> None:
    for name, cls in BACKENDS.items():
        assert frozenset(getattr(cls, "config_keys", ())) == ACCEPTED[name], name
        assert DEFAULT_REGISTRY.get("LLMBackend", name).config_keys == ACCEPTED[name], name


@pytest.mark.parametrize("name", sorted(ACCEPTED))
def test_each_config_key_is_a_keyword_of_the_backend(name: str) -> None:
    # The runner builds a backend as factory(model_id, **config), so each key it accepts must be a keyword setting.
    parameters = inspect.signature(BACKENDS[name]).parameters
    for key in ACCEPTED[name]:
        assert key in parameters and parameters[key].kind is inspect.Parameter.KEYWORD_ONLY, (name, key)


# ---------------------------------------------------------------------------
# Keys a backend does not accept


def test_ollama_refuses_api_key_env_naming_the_key(tmp_path: Path) -> None:
    message = str(load_error(tmp_path, recipe(model=model("ollama", OLLAMA_ID, api_key_env=KEY_ENV))))
    assert "model.api_key_env" in message and "model.backend.api_key_env" not in message, message
    assert "LLMBackend 'ollama'" in message and "does not accept the key 'api_key_env'" in message, message
    assert "accepted keys: base_url, timeout_s" in message, "the message lists the keys ollama accepts"


@pytest.mark.parametrize("key", sorted(SERVER_KEYS))
@pytest.mark.parametrize("backend", ["mock", "replay"])
def test_mock_and_replay_refuse_server_keys_naming_the_key(tmp_path: Path, backend: str, key: str) -> None:
    message = str(load_error(tmp_path, recipe(model=model(backend, **{key: SERVER_KEYS[key]}))))
    assert f"model.{key}" in message and f"LLMBackend '{backend}'" in message, message
    assert f"does not accept the key '{key}'" in message and "accepts no config keys" in message, message


def test_a_misspelled_server_key_is_unknown(tmp_path: Path) -> None:
    message = str(load_error(tmp_path, recipe(model=model(api_key=KEY_ENV))))
    assert "unknown key model.api_key" in message, message
    for key in ("api_key_env", "backend", "base_url", "device", "id", "timeout_s"):
        assert key in message, (key, message)


# ---------------------------------------------------------------------------
# api_key_env names a variable and never holds a key


@pytest.mark.parametrize("value", KEY_VALUES)
def test_api_key_env_holding_a_key_value_is_refused_without_showing_it(tmp_path: Path, value: Any) -> None:
    error = load_error(tmp_path, recipe(model=model(api_key_env=value)))
    message = str(error)
    assert "model.api_key_env must be" in message and "not shown" in message, message
    assert_shows_no_part(error, value)


@pytest.mark.parametrize("backend", ["ollama", "mock"])
def test_a_key_value_is_refused_by_shape_whatever_the_backend(tmp_path: Path, backend: str) -> None:
    # The schema runs before the binding check, so a key value under a backend that takes no api_key_env is still
    # refused as a value that may be a key, never quoted.
    value = "sk-q7Zr2Lw9Xv4Tn8Kp3Ym6Hd1"
    error = load_error(tmp_path, recipe(model=model(backend, OLLAMA_ID, api_key_env=value)))
    assert "model.api_key_env must be" in str(error) and "not shown" in str(error), str(error)
    assert "does not accept" not in str(error), "the shape check comes before the config check"
    assert_shows_no_part(error, value)


def test_a_lowercase_variable_name_is_refused(tmp_path: Path) -> None:
    error = load_error(tmp_path, recipe(model=model(api_key_env="vllm_api_key")))
    assert "model.api_key_env must be" in str(error), str(error)
    assert "vllm_api_key" not in str(error)


@pytest.mark.parametrize("name", ["VLLM_API_KEY", "_KEY", "K", "A1_", "X" * 64, KEY_ENV])
def test_api_key_env_accepts_an_uppercase_variable_name(tmp_path: Path, name: str) -> None:
    loaded = load(tmp_path, recipe(model=model(api_key_env=name)))
    assert loaded.data["model"]["api_key_env"] == name


@pytest.mark.parametrize("key", sorted(SERVER_KEYS))
def test_a_null_server_key_is_a_required_choice_with_no_value(tmp_path: Path, key: str) -> None:
    message = str(load_error(tmp_path, recipe(model=model(**{key: None}))))
    assert f"model.{key} is a required choice with no value" in message, message


# ---------------------------------------------------------------------------
# timeout_s and base_url types


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0, id="zero"),
        pytest.param(0.0, id="zero-float"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="boolean"),
        pytest.param("600", id="string"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param(float("nan"), id="nan"),
        pytest.param([600], id="list"),
    ],
)
def test_timeout_s_must_be_a_number_above_zero(tmp_path: Path, value: Any) -> None:
    message = str(load_error(tmp_path, recipe(model=model(timeout_s=value))))
    assert "model.timeout_s must be" in message, message


@pytest.mark.parametrize("value", [0.5, 1, 600, 3600.0])
def test_timeout_s_accepts_a_number_above_zero(tmp_path: Path, value: Any) -> None:
    assert load(tmp_path, recipe(model=model(timeout_s=value))).data["model"]["timeout_s"] == value


@pytest.mark.parametrize("value", [8000, True, ["http://127.0.0.1:8000/v1"], {"host": "127.0.0.1"}])
def test_base_url_must_be_a_string(tmp_path: Path, value: Any) -> None:
    message = str(load_error(tmp_path, recipe(model=model(base_url=value))))
    assert "model.base_url must be" in message, message


# ---------------------------------------------------------------------------
# Merging


def parent_and_child(tmp_path: Path, parent: Mapping[str, Any], child: Mapping[str, Any]) -> Path:
    """Write a parent recipe and a child that extends it by name; return the child's path."""
    write(tmp_path, "parent", parent)
    return write(tmp_path, "child", {"extends": "parent", **child})


def test_a_child_overrides_one_server_key_and_keeps_the_others(tmp_path: Path) -> None:
    parent = recipe(model=model(**SERVER_KEYS))
    child = {"model": {"base_url": "http://127.0.0.1:8001/v1"}}
    loaded = load_recipe(parent_and_child(tmp_path, parent, child), roots=[tmp_path], registry=registry())
    assert loaded.data["model"] == {**model(**SERVER_KEYS), "base_url": "http://127.0.0.1:8001/v1"}
    assert dict(backend_binding(loaded).config) == {**SERVER_KEYS, "base_url": "http://127.0.0.1:8001/v1"}


def test_a_child_naming_another_backend_still_inherits_the_server_keys(tmp_path: Path) -> None:
    # model is a plain mapping, not a kind section, so a child that names ollama keeps the parent's api_key_env,
    # and ollama's config check refuses it by name; the loader never drops a key silently.
    parent = recipe(model=model(**SERVER_KEYS))
    child = {"model": {"backend": "ollama", "id": OLLAMA_ID}}
    with pytest.raises(RecipeError) as info:
        load_recipe(parent_and_child(tmp_path, parent, child), roots=[tmp_path], registry=registry())
    message = str(info.value)
    assert "model.api_key_env" in message and "does not accept the key 'api_key_env'" in message, message
