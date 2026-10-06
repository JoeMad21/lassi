"""Tests for device sections in recipes (task P17.2): one shape, at load, in the hash.

Bible: Project Recipes (Notes), Component Interfaces (contract rules), Agent
Rule 10; plans/p17-portable.md, the planning decision "Explicit devices
(P17.2)".

A device section is the value of the key `device` inside a component's own
recipe section: executor.device (the single form), executor.<language>.device
(a per-language kind section), oracle.device, and model.device. A component
takes one only by declaring the capability `takes_device`
(lassi.core.capabilities TAKES_DEVICE, with DEVICE_KEY "device"):

- the registry accepts the key `device` from such a component and refuses it,
  naming the key, from any other, the backend's at model.device (not
  model.backend.device); and it refuses a component that lists "device" in
  its config_keys, so no component takes a free-form device key;
- a component that declares the capability and has no section is refused at
  load, naming <section>.device: the loader never picks a device; a
  per-language entry written as a bare name cannot carry one, so its message
  says to write a kind section;
- the section is read by lassi.core.devices.parse_device at load, so a
  missing or unknown kind, or a repeated index, is a RecipeError naming the
  file and the dotted key;
- the section is part of the resolved mapping, so it enters the canonical YAML
  and the recipe hash as written (indices in their written order), with no
  default materialized for a recipe without one;
- a child's `device` replaces the inherited one whole (a device is one
  choice), so a child naming cpu under a parent's rocm with indices does not
  inherit the indices.

Recipes without a device section keep their resolved recipes and hashes:
tests/core/test_recipe.py (CHILD_HASH, the golden resolved file) and
tests/core/test_single_executor_golden.py pin them unchanged. Every component
here is a fake that fails when constructed, since loading constructs nothing.
No value in this module is a measurement.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.core import capabilities
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.registry import Binding, Registry, RegistryError

# The capability literal, so these tests collect before lassi.core.capabilities defines TAKES_DEVICE.
TAKES_DEVICE = "takes_device"

# A complete recipe that extends nothing, with fake components; each test adds or changes sections.
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
    "model": {"backend": "plainllm", "id": "fixture-model"},
}
ROCM_0 = {"kind": "rocm", "indices": [0]}
CPU = {"kind": "cpu"}
# A per-language entry whose device names one index twice.
PER_LANGUAGE_DUPLICATE = {
    "executor": {"cuda": {"kind": "devexec", "device": {"kind": "cuda", "indices": [1, 1]}}, "omp": "plainexec"}
}


class NeverBuilt:
    """Base for every fake component: constructing one fails the test, since loading constructs nothing."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(f"component {type(self).__name__} was constructed while loading a recipe")


def fake(name: str, caps: Iterable[str] = (), *, config_keys: Iterable[str] = ()) -> type:
    """Return a fake component class with the class attributes the registry reads."""
    namespace: dict[str, Any] = {
        "__doc__": f"Fake component {name} for the device-section tests; never constructed.",
        "name": name,
        "capabilities": frozenset(caps),
        "config_keys": frozenset(config_keys),
    }
    return type(f"Fake_{name}", (NeverBuilt,), namespace)


def registry() -> Registry:
    """Return a Registry of fakes: a backend, executors, and an oracle with and without takes_device."""
    found = Registry()
    found.register("LLMBackend", "devllm", fake("devllm", {"chat", TAKES_DEVICE}))
    found.register("LLMBackend", "plainllm", fake("plainllm", {"chat"}))
    found.register("Executor", "devexec", fake("devexec", {"runs_code", TAKES_DEVICE}, config_keys={"host"}))
    found.register("Executor", "plainexec", fake("plainexec", {"runs_code"}, config_keys={"host"}))
    found.register("Oracle", "devoracle", fake("devoracle", {"stdout_diff", TAKES_DEVICE}))
    found.register("Stage", "generate", fake("generate"))
    return found


def write(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe <directory>/<name>.yaml (ASCII, LF) and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def recipe(**changes: Any) -> dict[str, Any]:
    """Return a copy of STANDALONE with top-level keys replaced by `changes`."""
    data = copy.deepcopy(STANDALONE)
    data.update(copy.deepcopy(changes))
    return data


def load(tmp_path: Path, data: Mapping[str, Any], name: str = "devices") -> Any:
    """Write and load `data` with the fake registry and tmp_path as the only root."""
    return load_recipe(write(tmp_path, name, data), roots=[tmp_path], registry=registry())


def load_error(tmp_path: Path, data: Mapping[str, Any], name: str = "devices") -> str:
    """Return the RecipeError message for `data`; the message must name the recipe file."""
    with pytest.raises(RecipeError) as info:
        load(tmp_path, data, name)
    message = str(info.value)
    assert f"{name}.yaml" in message, message
    return message


def binding(loaded: Any, interface: str) -> list[Binding]:
    """Return the loaded recipe's bindings of `interface`, in binding order."""
    return [item for item in loaded.bindings if item.interface == interface]


def has_key_named(value: Any, key: str) -> bool:
    """Return True when a mapping anywhere inside `value` holds `key`."""
    if isinstance(value, dict):
        return key in value or any(has_key_named(item, key) for item in value.values())
    if isinstance(value, list):
        return any(has_key_named(item, key) for item in value)
    return False


# ---------------------------------------------------------------------------
# The capability and the registry


def test_takes_device_and_device_key_are_declared() -> None:
    assert getattr(capabilities, "TAKES_DEVICE", None) == TAKES_DEVICE
    assert getattr(capabilities, "DEVICE_KEY", None) == "device"


def test_registry_refuses_device_in_config_keys() -> None:
    for caps in ((), (TAKES_DEVICE,)):
        with pytest.raises(RegistryError) as info:
            Registry().register("Executor", "freeform", fake("freeform", caps, config_keys={"host", "device"}))
        assert "device" in str(info.value)


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        ("executor.kind", "executor.device"),
        ("executor.cuda.kind", "executor.cuda.device"),
        ("executor.cuda", "executor.cuda.device"),
        ("oracle.kind", "oracle.device"),
        ("model.backend", "model.device"),
    ],
)
def test_device_path_names_the_section_a_binding_reads(where: str, expected: str) -> None:
    from lassi.core import registry as registry_module

    device_path = getattr(registry_module, "device_path", None)
    if device_path is None:
        pytest.fail("lassi.core.registry has no device_path (task P17.2)")
    assert device_path(Binding(interface="Executor", name="x", where=where, config={})) == expected


# ---------------------------------------------------------------------------
# Where a section loads, and how it enters the hash


def test_executor_device_section_loads_and_enters_the_hash(tmp_path: Path) -> None:
    first = load(tmp_path, recipe(executor={"kind": "devexec", "device": ROCM_0}), "first")
    second = load(tmp_path, recipe(executor={"kind": "devexec", "device": {"kind": "rocm", "indices": [1]}}), "second")
    assert first.data["executor"] == {"kind": "devexec", "device": ROCM_0}
    (executor,) = binding(first, "Executor")
    assert (executor.where, dict(executor.config)) == ("executor.kind", {"device": ROCM_0})
    assert "device:" in first.canonical_yaml and "kind: rocm" in first.canonical_yaml
    assert first.recipe_hash != second.recipe_hash, "the device section enters the recipe hash"
    assert load(tmp_path, recipe(executor={"kind": "devexec", "device": ROCM_0}), "first").recipe_hash == (
        first.recipe_hash
    ), "the same section hashes the same"


def test_indices_keep_their_written_order(tmp_path: Path) -> None:
    forward = load(tmp_path, recipe(executor={"kind": "devexec", "device": {"kind": "rocm", "indices": [0, 1]}}), "a")
    backward = load(tmp_path, recipe(executor={"kind": "devexec", "device": {"kind": "rocm", "indices": [1, 0]}}), "b")
    assert backward.data["executor"]["device"]["indices"] == [1, 0], "the loader never rewrites a value"
    assert forward.recipe_hash != backward.recipe_hash


def test_a_device_section_sits_beside_other_config_keys(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe(executor={"kind": "devexec", "host": "fixture-host", "device": CPU}))
    (executor,) = binding(loaded, "Executor")
    assert dict(executor.config) == {"host": "fixture-host", "device": CPU}


def test_per_language_executor_device_section_loads(tmp_path: Path) -> None:
    section = {"cuda": {"kind": "devexec", "device": {"kind": "cuda", "indices": [0]}}, "omp": "plainexec"}
    loaded = load(tmp_path, recipe(executor=section))
    by_where = {item.where: dict(item.config) for item in binding(loaded, "Executor")}
    assert by_where == {"executor.cuda.kind": {"device": {"kind": "cuda", "indices": [0]}}, "executor.omp": {}}


def test_model_device_section_loads_for_a_backend_that_takes_one(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe(model={"backend": "devllm", "id": "fixture-model", "device": CPU}))
    assert loaded.data["model"] == {"backend": "devllm", "id": "fixture-model", "device": CPU}
    (backend,) = binding(loaded, "LLMBackend")
    assert (backend.where, dict(backend.config)) == ("model.backend", {"device": CPU})
    other = load(tmp_path, recipe(model={"backend": "devllm", "id": "fixture-model", "device": ROCM_0}), "other")
    assert other.recipe_hash != loaded.recipe_hash


def test_oracle_device_section_loads(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe(oracle={"kind": "devoracle", "device": CPU}))
    (oracle,) = binding(loaded, "Oracle")
    assert (oracle.where, dict(oracle.config)) == ("oracle.kind", {"device": CPU})


def test_two_sections_load_side_by_side(tmp_path: Path) -> None:
    data = recipe(
        model={"backend": "devllm", "id": "fixture-model", "device": ROCM_0},
        executor={"kind": "devexec", "device": {"kind": "cuda", "indices": [0, 1]}},
    )
    loaded = load(tmp_path, data)
    assert loaded.data["model"]["device"] == ROCM_0
    assert loaded.data["executor"]["device"] == {"kind": "cuda", "indices": [0, 1]}


def test_a_recipe_without_a_device_section_resolves_without_one(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe())
    assert not has_key_named(loaded.data, "device"), "the loader never fills a device in"
    assert all(dict(item.config) == {} for item in loaded.bindings)


# ---------------------------------------------------------------------------
# Refusals at load, each naming the key


def test_a_component_that_takes_a_device_needs_a_section(tmp_path: Path) -> None:
    message = load_error(tmp_path, recipe(executor={"kind": "devexec"}))
    assert "executor.device" in message and "devexec" in message, message
    message = load_error(tmp_path, recipe(executor={"kind": "devexec", "host": "fixture-host"}))
    assert "executor.device" in message, message
    message = load_error(tmp_path, recipe(model={"backend": "devllm", "id": "fixture-model"}))
    assert "model.device" in message and "devllm" in message, message
    message = load_error(tmp_path, recipe(oracle={"kind": "devoracle"}))
    assert "oracle.device" in message, message
    message = load_error(tmp_path, recipe(executor={"cuda": {"kind": "devexec"}, "omp": "plainexec"}))
    assert "executor.cuda.device" in message, message


def test_a_bare_executor_name_that_takes_a_device_is_told_to_write_a_kind_section(tmp_path: Path) -> None:
    message = load_error(tmp_path, recipe(executor={"cuda": "devexec", "omp": "plainexec"}))
    assert "executor.cuda.device" in message, message
    assert "kind section" in message, message


def test_a_device_section_is_refused_where_no_component_takes_one(tmp_path: Path) -> None:
    message = load_error(tmp_path, recipe(model={"backend": "plainllm", "id": "fixture-model", "device": CPU}))
    assert "model.device" in message and "model.backend.device" not in message, message
    assert "plainllm" in message, message
    message = load_error(tmp_path, recipe(executor={"kind": "plainexec", "device": CPU}))
    assert "executor.device" in message and "plainexec" in message, message
    section = {"cuda": {"kind": "plainexec", "device": CPU}, "omp": "plainexec"}
    message = load_error(tmp_path, recipe(executor=section))
    assert "executor.cuda.device" in message, message


@pytest.mark.parametrize(
    ("changes", "dotted"),
    [
        pytest.param(
            {"executor": {"kind": "devexec", "device": {"indices": [0]}}}, "executor.device.kind", id="no-kind"
        ),
        pytest.param(
            {"executor": {"kind": "devexec", "device": {"kind": "xpu", "indices": [0]}}},
            "executor.device.kind",
            id="unknown-kind",
        ),
        pytest.param(
            {"executor": {"kind": "devexec", "device": {"kind": "ROCM", "indices": [0]}}},
            "executor.device.kind",
            id="kind-case",
        ),
        pytest.param(
            {"executor": {"kind": "devexec", "device": {"kind": "rocm", "indices": [0, 0]}}},
            "executor.device.indices",
            id="duplicate-indices",
        ),
        pytest.param(
            {"executor": {"kind": "devexec", "device": {"kind": "rocm"}}}, "executor.device.indices", id="no-indices"
        ),
        pytest.param(
            {"executor": {"kind": "devexec", "device": {"kind": "cpu", "indices": [0]}}},
            "executor.device.indices",
            id="cpu-indices",
        ),
        pytest.param({"executor": {"kind": "devexec", "device": "cuda:0"}}, "executor.device", id="string"),
        pytest.param(
            PER_LANGUAGE_DUPLICATE,
            "executor.cuda.device.indices",
            id="per-language-duplicate",
        ),
        pytest.param(
            {"model": {"backend": "devllm", "id": "fixture-model", "device": {"kind": "mps"}}},
            "model.device.kind",
            id="model-unknown-kind",
        ),
        pytest.param(
            {"model": {"backend": "devllm", "id": "fixture-model", "device": {"indices": [0]}}},
            "model.device.kind",
            id="model-no-kind",
        ),
        pytest.param(
            {"model": {"backend": "devllm", "id": "fixture-model", "device": {"kind": "cpu", "count": 1}}},
            "model.device.count",
            id="model-unknown-key",
        ),
        pytest.param(
            {"oracle": {"kind": "devoracle", "device": {"kind": "cuda", "indices": [-1]}}},
            "oracle.device.indices",
            id="oracle-negative",
        ),
    ],
)
def test_section_errors_are_refused_at_load_naming_the_key(
    tmp_path: Path, changes: dict[str, Any], dotted: str
) -> None:
    message = load_error(tmp_path, recipe(**changes))
    assert dotted in message, message


def test_model_device_is_a_known_key_only_for_the_device_shape(tmp_path: Path) -> None:
    # model.device is in the schema; a misspelled key beside it is still unknown.
    message = load_error(tmp_path, recipe(model={"backend": "devllm", "id": "fixture-model", "devices": CPU}))
    assert "model.devices" in message, message


# ---------------------------------------------------------------------------
# Merging: a child's device replaces the inherited one whole


def parent_and_child(tmp_path: Path, parent: Mapping[str, Any], child: Mapping[str, Any]) -> Path:
    """Write a parent recipe and a child that extends it by path; return the child's path."""
    write(tmp_path, "parent", parent)
    return write(tmp_path, "child", {"extends": "parent", **child})


def test_a_child_device_section_replaces_the_inherited_one(tmp_path: Path) -> None:
    parent = recipe(
        executor={"kind": "devexec", "host": "fixture-host", "device": {"kind": "rocm", "indices": [0, 1]}},
        model={"backend": "devllm", "id": "fixture-model", "device": {"kind": "cuda", "indices": [2, 3]}},
    )
    child = {"executor": {"kind": "devexec", "device": CPU}, "model": {"device": CPU}}
    loaded = load_recipe(parent_and_child(tmp_path, parent, child), roots=[tmp_path], registry=registry())
    assert loaded.data["executor"] == {"kind": "devexec", "host": "fixture-host", "device": CPU}
    assert loaded.data["model"] == {"backend": "devllm", "id": "fixture-model", "device": CPU}


def test_a_child_per_language_device_replaces_the_inherited_one(tmp_path: Path) -> None:
    entry = {"kind": "devexec", "device": {"kind": "cuda", "indices": [0, 1]}}
    parent = recipe(executor={"cuda": entry, "omp": "plainexec"})
    child = {"executor": {"cuda": {"device": CPU}}}
    loaded = load_recipe(parent_and_child(tmp_path, parent, child), roots=[tmp_path], registry=registry())
    assert loaded.data["executor"]["cuda"] == {"kind": "devexec", "device": CPU}


def test_a_child_cannot_edit_part_of_an_inherited_device(tmp_path: Path) -> None:
    # The child's mapping is the whole device: indices alone has no kind, so the load is refused, never completed
    # from the parent's section.
    parent = recipe(executor={"kind": "devexec", "device": {"kind": "rocm", "indices": [0]}})
    child = {"executor": {"kind": "devexec", "device": {"indices": [1]}}}
    with pytest.raises(RecipeError) as info:
        load_recipe(parent_and_child(tmp_path, parent, child), roots=[tmp_path], registry=registry())
    assert "executor.device.kind" in str(info.value)


def test_a_child_without_a_device_keeps_the_inherited_one(tmp_path: Path) -> None:
    parent = recipe(executor={"kind": "devexec", "device": ROCM_0})
    child = {"executor": {"kind": "devexec", "host": "fixture-host"}}
    loaded = load_recipe(parent_and_child(tmp_path, parent, child), roots=[tmp_path], registry=registry())
    assert loaded.data["executor"] == {"kind": "devexec", "host": "fixture-host", "device": ROCM_0}
