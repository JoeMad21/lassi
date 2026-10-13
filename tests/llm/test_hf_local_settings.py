"""Tests of hf_local that need no framework extra (task P17.4): registration, settings, and pure helpers.

Bible: Model Serving (Serving Rules), Component Interfaces (LLMBackend),
Toolchain Pins (framework pins), Agent Rules 7, 10, and 12;
plans/p17-portable.md, task P17.4 and the Constraints ("Frameworks stay out
of the default environment ... lassi.llm still registers hf_local");
PHASE-NOTES P0 (the registration rule); the P17.4 design (sections 2, 3, 5,
7, 9, and 10).

These run in the default environment and with a framework extra alike: a
test that needs the framework absent hides torch and transformers with
sys.modules entries of None, and one that needs it present while it is not
installed puts stand-in modules there, which nothing here imports.

The contract these tests fix:

- Importing lassi.llm registers hf_local without torch or transformers and
  imports neither. Its capabilities are chat, model_check, and takes_device,
  and its config_keys revision and seed.
- Building hf_local without the framework raises FrameworkMissing, an
  ImportError whose message names the extra (`--extra cpu`) and the module
  that is missing; framework() raises it too, and a run is refused before any
  directory exists with a RunError naming model.device and the extra.
- hf_local requires model.revision (a full commit id), model.seed (an int from
  0 to 2**32 - 1), and model.device with one index at most, each refused
  with a ValueError naming the key; it builds no model and reads no file.
- model.id must be a Hub repository id, <owner>/<name>: an empty id, a
  one-part name, or a file path is refused with a ValueError naming
  model.id, since from_pretrained would load a local directory the id names.
- torch_device maps cpu to "cpu" and cuda or rocm index i to "cuda:i".
- generation_settings passes every sampling field explicitly: for a
  temperature above 0, sampling with temperature, top_p, top_k 0, min_p None,
  typical_p 1.0, repetition_penalty 1.0, no_repeat_ngram_size 0, num_beams 1,
  num_return_sequences 1, max_new_tokens, and the stop and pad ids; for 0,
  greedy decoding with the same fixed fields and no temperature, top_p, or
  top_k. A temperature below 0 or a top_p outside (0, 1] is a ServingError
  naming llm.sampling.<key>.
- serving_line shows an in-process record (no base_url) as "transformers
  <v>; torch <v>; revision <first 12>; max_position_embeddings <n>; device
  <d>; seed <s>".
- pyproject.toml holds torch and transformers only in the conflicting
  extras cpu, cuda, and rocm, each at the P17.1 pins, and uv.lock locks them
  while the project's own dependencies stay the control plane's.
- The runner's model-check RunError names model.base_url only for a backend
  that takes base_url.

Every value here is SYNTHETIC; no value in this module is a measurement.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import inspect
import json
import subprocess
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hf_runs import (
    CPU,
    SEED,
    FakeCpuProbe,
    clean_environment,
    make_registry,
    model_section,
    recipe_data,
    run,
)
from tiny_hf import REPO_ID, REVISION

from lassi.core.devices import DeviceSpec
from lassi.core.interfaces import Completion, LLMBackend, Message, Sampling
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.core.runner import RunError
from lassi.llm import ServingError, serving_line

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"
UV_LOCK = REPO / "uv.lock"
FRAMEWORK = ("torch", "transformers")
FRAMEWORK_MODULES = ("torch", "transformers", "tokenizers", "huggingface_hub", "safetensors")
EXTRA_HINT = "--extra cpu"
TORCH_PIN = "torch==2.14.1"
TRANSFORMERS_PIN = "transformers==5.18.0"
FLAVORS = ("cpu", "cuda", "rocm")
CONTROL_PLANE = {"pyarrow", "pyyaml", "tiktoken"}
STOPS = (2, 5)
PAD = 0
SAMPLED_KEYS = {
    "do_sample", "temperature", "top_p", "top_k", "min_p", "typical_p", "repetition_penalty", "no_repeat_ngram_size",
    "num_beams", "num_return_sequences", "max_new_tokens", "eos_token_id", "pad_token_id",
}
FIXED = {
    "min_p": None, "typical_p": 1.0, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "num_beams": 1,
    "num_return_sequences": 1,
}


def hf_local() -> ModuleType:
    """Import lassi.llm.hf_local, failing the test clearly while it does not exist."""
    try:
        return importlib.import_module("lassi.llm.hf_local")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.llm.hf_local does not exist yet (task P17.4): {error}")


def registered_hf_local() -> type:
    """Return the class lassi.llm registers as LLMBackend hf_local, failing clearly while there is none."""
    if "hf_local" not in DEFAULT_REGISTRY.names("LLMBackend"):
        pytest.fail("importing lassi.llm registers no LLMBackend hf_local yet (task P17.4)")
    return DEFAULT_REGISTRY.get("LLMBackend", "hf_local").factory


def hide_framework(monkeypatch: pytest.MonkeyPatch, names: Sequence[str] = FRAMEWORK) -> None:
    """Make `names` unimportable for the test: a None entry in sys.modules stops their import."""
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


def stand_in_framework(monkeypatch: pytest.MonkeyPatch, names: Sequence[str] = FRAMEWORK) -> None:
    """Make importlib.util.find_spec find `names`: stand-in modules for any that is not installed."""
    for name in names:
        if importlib.util.find_spec(name) is None:
            module = types.ModuleType(name)
            module.__spec__ = importlib.machinery.ModuleSpec(name, None)
            monkeypatch.setitem(sys.modules, name, module)


# A setting build() leaves out of the call, as a recipe without the key does.
_ABSENT = object()


def build(**settings: Any) -> Any:
    """Return HFLocalBackend(REPO_ID, **settings) with the valid defaults for the settings not given."""
    values: dict[str, Any] = {"device": dict(CPU), "revision": REVISION, "seed": SEED}
    values.update(settings)
    values = {key: value for key, value in values.items() if value is not _ABSENT}
    return hf_local().HFLocalBackend(REPO_ID, **values)


def python_says(code: str) -> Any:
    """Run `code` in a fresh interpreter from the repository root and return the JSON it prints."""
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# ---------------------------------------------------------------------------
# Registration without the extra (PHASE-NOTES P0, the registration rule)


def test_lassi_llm_registers_hf_local_without_the_extra() -> None:
    code = (
        "import json, sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['transformers'] = None\n"
        "import lassi.llm\n"
        "from lassi.core.registry import DEFAULT_REGISTRY\n"
        "names = DEFAULT_REGISTRY.names('LLMBackend')\n"
        "entry = DEFAULT_REGISTRY.get('LLMBackend', 'hf_local') if 'hf_local' in names else None\n"
        "print(json.dumps({'names': names, 'factory': entry and entry.factory.__name__,\n"
        "    'capabilities': entry and sorted(entry.capabilities),\n"
        "    'config_keys': entry and sorted(entry.config_keys),\n"
        "    'exported': hasattr(lassi.llm, 'HFLocalBackend') and hasattr(lassi.llm, 'hf_local')}))\n"
    )
    found = python_says(code)
    assert "hf_local" in found["names"], found
    assert found["factory"] == "HFLocalBackend"
    assert found["capabilities"] == ["chat", "model_check", "takes_device"]
    assert found["config_keys"] == ["revision", "seed"]
    assert found["exported"] is True, "lassi.llm re-exports HFLocalBackend and the module"


def test_importing_lassi_llm_imports_no_framework() -> None:
    code = (
        "import json, sys\n"
        "import lassi.llm, lassi.llm.hf_local, lassi.core.runner, lassi.cli\n"
        f"print(json.dumps(sorted(name for name in {list(FRAMEWORK_MODULES)!r} if name in sys.modules)))\n"
    )
    assert python_says(code) == [], "the framework is imported only when hf_local loads"


def test_hf_local_has_the_llm_backend_shape() -> None:
    cls = registered_hf_local()
    assert cls is hf_local().HFLocalBackend
    assert cls.name == "hf_local" and "name" in vars(cls) and "capabilities" in vars(cls)
    assert cls.capabilities == frozenset({"chat", "model_check", "takes_device"})
    assert cls.config_keys == frozenset({"revision", "seed"})
    assert list(inspect.signature(cls.complete).parameters) == [*inspect.signature(LLMBackend.complete).parameters]
    for method in ("check", "serving", "framework"):
        assert callable(getattr(cls, method, None)), method
    assert isinstance(inspect.getattr_static(cls, "framework"), (staticmethod, classmethod)), (
        "the runner calls framework() on the class, before anything is built"
    )
    parameters = inspect.signature(cls).parameters
    for key in ("device", "revision", "seed"):
        assert parameters[key].kind is inspect.Parameter.KEYWORD_ONLY, key


def test_repr_shows_the_class_the_model_and_the_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_framework(monkeypatch)
    text = repr(build())
    assert "HFLocalBackend" in text and REPO_ID in text and REVISION in text, text


# ---------------------------------------------------------------------------
# Without the extra: building and framework() name the extra; a run is refused before any directory


def test_building_without_the_extra_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    module = hf_local()
    hide_framework(monkeypatch)
    with pytest.raises(module.FrameworkMissing) as info:
        build()
    assert isinstance(info.value, ImportError)
    assert EXTRA_HINT in str(info.value) and "torch" in str(info.value), str(info.value)
    with pytest.raises(module.FrameworkMissing) as info:
        module.HFLocalBackend.framework()
    assert EXTRA_HINT in str(info.value), str(info.value)
    message = module.framework_missing_message("transformers")
    assert EXTRA_HINT in message and "transformers" in message, message


def test_a_missing_transformers_alone_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    module = hf_local()
    stand_in_framework(monkeypatch, ("torch",))
    hide_framework(monkeypatch, ("transformers",))
    with pytest.raises(module.FrameworkMissing) as info:
        build()
    assert EXTRA_HINT in str(info.value) and "transformers" in str(info.value), str(info.value)


def test_a_run_without_the_extra_is_refused_naming_the_key_and_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean_environment(tmp_path, monkeypatch)
    registry = make_registry(registered_hf_local())
    hide_framework(monkeypatch)
    probe = FakeCpuProbe()
    with pytest.raises(RunError) as info:
        run(tmp_path, registry, recipe_data(model_section()), "no-extra", probe)
    message = str(info.value)
    assert "model.device" in message and EXTRA_HINT in message and "hf_local" in message, message
    assert not (tmp_path / "runs-root").exists(), "the refusal comes before any directory exists"
    assert probe.calls == 0, "the framework build is read before the probe runs"


# ---------------------------------------------------------------------------
# The settings hf_local requires


@pytest.mark.parametrize(
    ("settings", "key"),
    [
        pytest.param({"revision": _ABSENT}, "model.revision", id="revision-missing"),
        pytest.param({"revision": None}, "model.revision", id="revision-none"),
        pytest.param({"revision": "main"}, "model.revision", id="revision-branch"),
        pytest.param({"revision": "v1.0"}, "model.revision", id="revision-tag"),
        pytest.param({"revision": REVISION[:12]}, "model.revision", id="revision-short"),
        pytest.param({"revision": REVISION.upper()}, "model.revision", id="revision-uppercase"),
        pytest.param({"revision": REVISION + "0"}, "model.revision", id="revision-41"),
        pytest.param({"seed": _ABSENT}, "model.seed", id="seed-missing"),
        pytest.param({"seed": None}, "model.seed", id="seed-none"),
        pytest.param({"seed": True}, "model.seed", id="seed-bool"),
        pytest.param({"seed": -1}, "model.seed", id="seed-negative"),
        pytest.param({"seed": 2**32}, "model.seed", id="seed-2-32"),
        pytest.param({"seed": 1.5}, "model.seed", id="seed-float"),
        pytest.param({"seed": "7"}, "model.seed", id="seed-string"),
        pytest.param({"device": _ABSENT}, "model.device", id="device-missing"),
        pytest.param({"device": None}, "model.device", id="device-none"),
        pytest.param({"device": {"kind": "gpu"}}, "model.device", id="device-kind"),
        pytest.param({"device": {"kind": "cuda", "indices": [0, 1]}}, "model.device", id="device-two-cuda"),
        pytest.param({"device": {"kind": "rocm", "indices": [1, 0]}}, "model.device", id="device-two-rocm"),
    ],
)
def test_settings_are_required_and_checked(monkeypatch: pytest.MonkeyPatch, settings: dict, key: str) -> None:
    stand_in_framework(monkeypatch)
    with pytest.raises(ValueError) as info:
        build(**settings)
    assert key in str(info.value), str(info.value)


@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param("", id="empty"),
        pytest.param("tiny-chat", id="one-part"),
        pytest.param("lassi-test/tiny-chat/extra", id="three-parts"),
        pytest.param("/srv/models/tiny-chat", id="absolute-path"),
        pytest.param("C:/models/tiny-chat", id="drive-path"),
        pytest.param("./tiny-chat", id="relative-path"),
        pytest.param("../lassi-test/tiny-chat", id="parent-path"),
        pytest.param("lassi-test/ tiny-chat", id="space"),
    ],
)
def test_a_model_id_that_is_not_a_hub_repository_id_is_refused(
    monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    stand_in_framework(monkeypatch)
    with pytest.raises(ValueError) as info:
        hf_local().HFLocalBackend(model_id, device=dict(CPU), revision=REVISION, seed=SEED)
    assert "model.id" in str(info.value), str(info.value)


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({}, id="cpu"),
        pytest.param({"seed": 0}, id="seed-0"),
        pytest.param({"seed": 2**32 - 1}, id="seed-max"),
        pytest.param({"device": {"kind": "cuda", "indices": [1]}}, id="cuda-1"),
        pytest.param({"device": {"kind": "rocm", "indices": [0]}}, id="rocm-0"),
    ],
)
def test_valid_settings_build_without_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: dict
) -> None:
    stand_in_framework(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "no-such-home"))
    built = build(**settings)
    assert built.model_id == REPO_ID
    assert not (tmp_path / "no-such-home").exists(), "construction reads and writes nothing under HF_HOME"


def test_torch_device_maps_kinds() -> None:
    torch_device = hf_local().torch_device
    assert torch_device(DeviceSpec(kind="cpu", indices=())) == "cpu"
    assert torch_device(DeviceSpec(kind="cuda", indices=(1,))) == "cuda:1"
    assert torch_device(DeviceSpec(kind="rocm", indices=(0,))) == "cuda:0", "ROCm PyTorch uses the cuda device type"
    with pytest.raises(ValueError) as info:
        torch_device(DeviceSpec(kind="cuda", indices=(0, 1)))
    assert "model.device" in str(info.value) and "2" in str(info.value), str(info.value)


# ---------------------------------------------------------------------------
# Every generation field passed explicitly


def normalized(settings: dict[str, Any]) -> dict[str, Any]:
    """Return generation settings with the stop ids as a list, however they were given."""
    found = dict(settings)
    stops = found.get("eos_token_id")
    found["eos_token_id"] = [stops] if isinstance(stops, int) else list(stops or [])
    return found


def test_generation_settings_pass_every_field() -> None:
    generation_settings = hf_local().generation_settings
    sampled = normalized(generation_settings(Sampling(temperature=0.7, top_p=0.95, max_tokens=32), STOPS, PAD))
    assert set(sampled) == SAMPLED_KEYS, sorted(sampled)
    assert sampled == {
        "do_sample": True, "temperature": 0.7, "top_p": 0.95, "top_k": 0, **FIXED, "max_new_tokens": 32,
        "eos_token_id": list(STOPS), "pad_token_id": PAD,
    }
    greedy = normalized(generation_settings(Sampling(temperature=0.0, top_p=0.9, max_tokens=16), STOPS, PAD))
    assert set(greedy) == SAMPLED_KEYS - {"temperature", "top_p", "top_k"}, sorted(greedy)
    assert greedy == {
        "do_sample": False, **FIXED, "max_new_tokens": 16, "eos_token_id": list(STOPS), "pad_token_id": PAD
    }


@pytest.mark.parametrize(
    ("sampling", "key"),
    [
        pytest.param(Sampling(temperature=-0.1, top_p=0.9, max_tokens=8), "llm.sampling.temperature", id="temp"),
        pytest.param(Sampling(temperature=0.7, top_p=0.0, max_tokens=8), "llm.sampling.top_p", id="top-p-0"),
        pytest.param(Sampling(temperature=0.7, top_p=1.5, max_tokens=8), "llm.sampling.top_p", id="top-p-1.5"),
    ],
)
def test_generation_settings_refuse_values_outside_their_range(sampling: Sampling, key: str) -> None:
    with pytest.raises(ServingError) as info:
        hf_local().generation_settings(sampling, STOPS, PAD)
    assert key in str(info.value), str(info.value)


# ---------------------------------------------------------------------------
# run.md's Server line for an in-process record


def test_serving_line_shows_an_in_process_record() -> None:
    record = {
        "model": REPO_ID,
        "revision": REVISION,
        "max_position_embeddings": 4096,
        "dtype": "float32",
        "device": "cpu",
        "threads": 4,
        "seed": SEED,
        "stop_token_ids": [2],
        "versions": {
            "torch": "2.99.0+synthetic", "transformers": "9.9.9", "tokenizers": "0.0.1", "huggingface-hub": "0.0.2",
            "safetensors": "0.0.3",
        },
    }
    assert serving_line(record) == (
        f"transformers 9.9.9; torch 2.99.0+synthetic; revision {REVISION[:12]}; max_position_embeddings 4096; "
        f"device cpu; seed {SEED}"
    )


def test_serving_line_keeps_the_served_form() -> None:
    record = {"base_url": "http://127.0.0.1:8000/v1", "model": {"owned_by": "vllm", "max_model_len": 32768},
              "version": "0.0.0-synthetic", "version_from": "GET /version version"}
    assert serving_line(record) == "owned_by vllm; version 0.0.0-synthetic; max_model_len 32768"


# ---------------------------------------------------------------------------
# The extras and their pins (Toolchain Pins, framework pins)


def read_toml(path: Path) -> dict[str, Any]:
    """Return a TOML file of the repository, read as ASCII."""
    return tomllib.loads(path.read_bytes().decode("ascii"))


def test_framework_pins_are_extras_only() -> None:
    project = read_toml(PYPROJECT)
    dependencies = " ".join(project["project"]["dependencies"])
    assert "torch" not in dependencies and "transformers" not in dependencies, "a plain uv sync installs no framework"
    extras = project["project"].get("optional-dependencies", {})
    assert set(FLAVORS) <= set(extras), f"pyproject.toml extras: {sorted(extras)}"
    for flavor in FLAVORS:
        pins = [item.split(";")[0].replace(" ", "") for item in extras[flavor]]
        assert TORCH_PIN in pins and TRANSFORMERS_PIN in pins, (flavor, pins)
    conflicts = project["tool"]["uv"]["conflicts"]
    assert [sorted(item["extra"] for item in group) for group in conflicts] == [sorted(FLAVORS)]
    sources = {entry["extra"]: entry["index"] for entry in project["tool"]["uv"]["sources"]["torch"]}
    assert set(sources) == set(FLAVORS), sources
    indexes = {entry["name"]: entry for entry in project["tool"]["uv"]["index"]}
    assert indexes[sources["cpu"]]["url"] == "https://download.pytorch.org/whl/cpu"
    assert all(indexes[name].get("explicit") is True for name in sources.values()), "torch only from its own index"


def test_the_lock_holds_the_framework_beside_the_control_plane() -> None:
    lock = read_toml(UV_LOCK)
    packages = {}
    for package in lock["package"]:
        packages.setdefault(package["name"], []).append(package)
    assert any(item["version"].startswith("2.14.1") for item in packages.get("torch", [])), "torch 2.14.1 is locked"
    assert [item["version"] for item in packages.get("transformers", [])] == ["5.18.0"]
    (own,) = packages["lassi"]
    assert {item["name"] for item in own["dependencies"]} == CONTROL_PLANE, own["dependencies"]
    assert set(FLAVORS) <= set(own.get("optional-dependencies", {})), "the lock records the three extras"


# ---------------------------------------------------------------------------
# The runner's model-check message


class CheckedInProcess:
    """An in-process LLMBackend "inproc" that declares model_check and takes no base_url; its check fails."""

    name = "inproc"
    capabilities = frozenset({"chat", "model_check"})
    config_keys = frozenset({"revision"})

    def __init__(self, model_id: str, *, revision: str | None = None) -> None:
        """Keep the model id."""
        self.model_id = model_id

    def check(self) -> dict[str, Any]:
        """Fail as a missing snapshot does."""
        raise ServingError("SYNTHETIC: the hub cache holds no snapshot at this revision")

    def serving(self) -> dict[str, Any]:
        """Fail as check() does."""
        return self.check()

    def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
        """Fail the test: the run is refused before any request."""
        raise AssertionError("a request was sent after a failed model check")


def test_the_model_check_message_names_base_url_only_for_a_backend_that_takes_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean_environment(tmp_path, monkeypatch)
    registry = make_registry(CheckedInProcess)
    data = recipe_data({"backend": "hf_local", "id": REPO_ID, "revision": REVISION})
    with pytest.raises(RunError) as info:
        run(tmp_path, registry, data, "in-process-check", FakeCpuProbe())
    message = str(info.value)
    assert "SYNTHETIC: the hub cache holds no snapshot" in message and "model.id" in message, message
    assert "base_url" not in message, "an in-process backend has no base_url to name"
    assert not (tmp_path / "runs-root").exists()
