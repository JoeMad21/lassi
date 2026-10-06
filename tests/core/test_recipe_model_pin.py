"""Tests for the in-process model keys of a recipe's model section (task P17.4): model.revision and model.seed.

Bible: Project Recipes (Notes), Model Serving (Serving Rules), Agent Rule 10
(pins are commit ids); plans/p17-portable.md, task P17.4 ("a model key that
reaches the backend as P17.3's keys do and enters the recipe hash"); the
P17.4 design (section 3).

The contract these tests fix:

- model.revision (a full commit id: 40 lowercase hexadecimal characters) and
  model.seed (an integer from 0 to 2**32 - 1, never a bool) are optional keys
  of the model section, checked at load naming the key. A branch or tag
  name, a short id, or uppercase hex is not a pin.
- The loader fills neither in, so a recipe without them resolves and hashes
  exactly as before this task (the committed recipes' hashes below); a key
  that is set enters the canonical YAML and the recipe hash as written.
- The backend's binding carries them as config beside the device section, so
  the runner builds the backend as factory(model.id, revision=..., seed=...,
  device=...) as it does for P17.3's keys. Only a backend that lists them in
  config_keys accepts them (hf_local); any other is refused at load naming
  model.revision or model.seed.

The backends are fakes registered in a test Registry and never constructed,
beside the real openai_compat and mock classes; every value is SYNTHETIC. No
value in this module is a measurement.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

import lassi.core.runner  # noqa: F401  (registers every component, as `lassi run` does)
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.registry import DEFAULT_REGISTRY, Binding, Registry
from lassi.llm import MockBackend, OpenAICompatBackend

REPO = Path(__file__).resolve().parents[2]
PROJECTS = REPO / "projects"
TAKES_DEVICE = "takes_device"
MODEL_ID = "fixture/tiny-chat"
REVISION = "0123456789abcdef0123456789abcdef01234567"
OTHER_REVISION = "89abcdef0123456789abcdef0123456789abcdef"
SEED = 7
CPU = {"kind": "cpu"}
# The sha256 of each committed project recipe's canonical YAML, computed with lassi.core.recipe at 3f79562, before
# this task; a hash is a function of its input text, not a measurement. None of them sets either key.
COMMITTED_HASHES = {
    "lassi-demo/rngd-compile.yaml": "5d3ce25eba621fd4c25811d91a0addde2be573cf12f1dbf34baa231f2b67d85b",
    "lassi-demo/rngd-cpu.yaml": "b88418d427f0e1fee7e38ba9202c98f906144c8d7d73117c39a3322d417edf59",
    "lassi-demo/rngd-smoke.yaml": "09256bcdabf25e930d1ff5f3744f64546b4a6437d721ec586faebabffc98bae6",
    "lassi-demo/tier-a-rngd-maxabs.yaml": "8bdaeac1d6b46ef4293c7187a831b5f3c6b054ef2bf02e1161fd9e8f8ca4e992",
    "lassi-demo/tier-a-rngd-pcc.yaml": "575a1dfba6ab5abf20743ecc1e280c92ab92d85a83d079669c2209e38267dc59",
}

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
    "model": {"backend": "pinllm", "id": MODEL_ID, "device": dict(CPU)},
}


class NeverBuilt:
    """Base for every fake component: constructing one fails the test, since loading constructs nothing."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(f"component {type(self).__name__} was constructed while loading a recipe")


def fake(name: str, caps: Iterable[str] = (), *, config_keys: Iterable[str] = ()) -> type:
    """Return a fake component class with the class attributes the registry reads."""
    namespace: dict[str, Any] = {
        "__doc__": f"Fake component {name} for the model pin tests; never constructed.",
        "name": name,
        "capabilities": frozenset(caps),
        "config_keys": frozenset(config_keys),
    }
    return type(f"Fake_{name}", (NeverBuilt,), namespace)


def registry() -> Registry:
    """Return a Registry with a fake in-process backend that takes the keys, two real backends, and fakes."""
    found = Registry()
    found.register("LLMBackend", "pinllm", fake("pinllm", {"chat", TAKES_DEVICE}, config_keys={"revision", "seed"}))
    found.register("LLMBackend", "openai_compat", OpenAICompatBackend)
    found.register("LLMBackend", "mock", MockBackend)
    found.register("Executor", "plainexec", fake("plainexec", {"runs_code"}))
    found.register("Stage", "generate", fake("generate"))
    return found


def recipe(**model_keys: Any) -> dict[str, Any]:
    """Return a copy of STANDALONE whose model section also holds `model_keys`, each value as given (None too)."""
    data = copy.deepcopy(STANDALONE)
    for key, value in model_keys.items():
        data["model"][key] = copy.deepcopy(value)
    return data


def write(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe <directory>/<name>.yaml (ASCII, LF) and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def load(tmp_path: Path, data: Mapping[str, Any], name: str = "pinned") -> Any:
    """Write and load `data` with the test registry and tmp_path as the only root."""
    return load_recipe(write(tmp_path, name, data), roots=[tmp_path], registry=registry())


def load_error(tmp_path: Path, data: Mapping[str, Any], name: str = "pinned") -> str:
    """Return the message of the RecipeError `data` raises; it must name the recipe file."""
    with pytest.raises(RecipeError) as info:
        load(tmp_path, data, name)
    assert f"{name}.yaml" in str(info.value), str(info.value)
    return str(info.value)


def backend_binding(loaded: Any) -> Binding:
    """Return the loaded recipe's one LLMBackend binding."""
    (found,) = [item for item in loaded.bindings if item.interface == "LLMBackend"]
    return found


# ---------------------------------------------------------------------------
# Loading, the canonical YAML, and the hash


def test_revision_and_seed_enter_the_resolved_recipe_and_hash(tmp_path: Path) -> None:
    plain = load(tmp_path, recipe())
    pinned = load(tmp_path, recipe(revision=REVISION, seed=SEED))
    assert pinned.data["model"] == {"backend": "pinllm", "id": MODEL_ID, "device": CPU, "revision": REVISION,
                                    "seed": SEED}
    assert yaml.safe_load(pinned.canonical_yaml)["model"] == pinned.data["model"], "the canonical YAML holds them"
    assert pinned.recipe_hash != plain.recipe_hash
    assert load(tmp_path, recipe(revision=OTHER_REVISION, seed=SEED)).recipe_hash != pinned.recipe_hash
    assert load(tmp_path, recipe(revision=REVISION, seed=SEED + 1)).recipe_hash != pinned.recipe_hash
    assert load(tmp_path, recipe(revision=REVISION, seed=SEED)).recipe_hash == pinned.recipe_hash


def test_the_keys_reach_the_backend_binding_beside_the_device(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe(revision=REVISION, seed=SEED))
    found = backend_binding(loaded)
    assert (found.name, found.where) == ("pinllm", "model.backend")
    assert dict(found.config) == {"device": CPU, "revision": REVISION, "seed": SEED}


def test_a_recipe_without_the_keys_resolves_without_them(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe())
    assert loaded.data["model"] == {"backend": "pinllm", "id": MODEL_ID, "device": CPU}, "the loader fills nothing in"
    assert dict(backend_binding(loaded).config) == {"device": CPU}


@pytest.mark.parametrize("relative", sorted(COMMITTED_HASHES))
def test_recipes_without_the_keys_keep_their_hashes(relative: str) -> None:
    loaded = load_recipe(PROJECTS / relative)
    assert "revision" not in loaded.data.get("model", {}) and "seed" not in loaded.data.get("model", {})
    assert loaded.recipe_hash == COMMITTED_HASHES[relative], "the hash before this task"


def test_hf_local_accepts_the_keys_at_load(tmp_path: Path) -> None:
    if "hf_local" not in DEFAULT_REGISTRY.names("LLMBackend"):
        pytest.fail("importing lassi.llm registers no LLMBackend hf_local yet (task P17.4)")
    data = recipe(revision=REVISION, seed=SEED)
    data["model"]["backend"] = "hf_local"
    loaded = load_recipe(write(tmp_path, "hf", data), roots=[tmp_path], registry=_with_hf_local())
    assert dict(backend_binding(loaded).config) == {"device": CPU, "revision": REVISION, "seed": SEED}


def _with_hf_local() -> Registry:
    """Return the test registry plus the real hf_local class."""
    found = registry()
    found.register("LLMBackend", "hf_local", DEFAULT_REGISTRY.get("LLMBackend", "hf_local").factory)
    return found


# ---------------------------------------------------------------------------
# The values each key takes


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("main", id="branch"),
        pytest.param("v1.0", id="tag"),
        pytest.param(REVISION[:39], id="39-characters"),
        pytest.param(REVISION + "0", id="41-characters"),
        pytest.param(REVISION.upper(), id="uppercase"),
        pytest.param("g" * 40, id="not-hex"),
        pytest.param(" " + REVISION[1:], id="leading-space"),
        pytest.param(1234567, id="integer"),
        pytest.param([REVISION], id="list"),
    ],
)
def test_a_revision_that_is_not_a_commit_id_is_refused(tmp_path: Path, value: Any) -> None:
    message = load_error(tmp_path, recipe(revision=value, seed=SEED))
    assert "model.revision must be" in message, message


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="bool"),
        pytest.param(1.5, id="float"),
        pytest.param(7.0, id="integral-float"),
        pytest.param(-1, id="negative"),
        pytest.param(2**32, id="2-32"),
        pytest.param("7", id="string"),
    ],
)
def test_a_seed_outside_the_range_is_refused(tmp_path: Path, value: Any) -> None:
    message = load_error(tmp_path, recipe(revision=REVISION, seed=value))
    assert "model.seed must be" in message, message


@pytest.mark.parametrize("value", [0, 1, SEED, 2**32 - 1])
def test_a_seed_in_the_range_loads(tmp_path: Path, value: int) -> None:
    assert load(tmp_path, recipe(revision=REVISION, seed=value)).data["model"]["seed"] == value


@pytest.mark.parametrize("key", ["revision", "seed"])
def test_a_null_key_is_a_required_choice_with_no_value(tmp_path: Path, key: str) -> None:
    message = load_error(tmp_path, recipe(**{key: None}))
    assert f"model.{key} is a required choice with no value" in message, message


@pytest.mark.parametrize(
    ("backend", "key", "value"),
    [
        pytest.param("openai_compat", "revision", REVISION, id="openai_compat-revision"),
        pytest.param("openai_compat", "seed", SEED, id="openai_compat-seed"),
        pytest.param("mock", "revision", REVISION, id="mock-revision"),
        pytest.param("mock", "seed", SEED, id="mock-seed"),
    ],
)
def test_another_backend_refuses_revision_and_seed(tmp_path: Path, backend: str, key: str, value: Any) -> None:
    data = recipe(**{key: value})
    data["model"] = {"backend": backend, "id": MODEL_ID, key: value}
    message = load_error(tmp_path, data)
    assert f"model.{key}" in message and f"does not accept the key {key!r}" in message, message


def test_a_child_overrides_the_seed_and_keeps_the_revision(tmp_path: Path) -> None:
    write(tmp_path, "parent", recipe(revision=REVISION, seed=SEED))
    child = write(tmp_path, "child", {"extends": "parent", "model": {"seed": SEED + 1}})
    loaded = load_recipe(child, roots=[tmp_path], registry=registry())
    assert (loaded.data["model"]["revision"], loaded.data["model"]["seed"]) == (REVISION, SEED + 1)
