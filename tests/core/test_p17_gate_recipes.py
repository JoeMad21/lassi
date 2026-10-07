"""Tests for gate part (a)'s run recipes (task P17.11): the hf_local CPU recipe and the four GPU refusal recipes.

Bible: Build Roadmap (P17 row, Gate part (a)), Project Recipes (Notes:
device sections, executor forms), Toolchain Pins (gate model), Component
Interfaces (contract rules, takes_device), Agent Rules 1 and 10;
plans/p17-portable.md, task P17.11 and the Constraints (nothing on alpha01
opens /dev/kfd or a /dev/dri node, a probe included).

The contract these tests fix:

- tests/fixtures/recipes/p17-cpu-hf.yaml loads with lassi's own registry
  and resolves to hf_local on kind cpu with the bible's gate model pin, one
  item of lassi-hecbench-10 (matrix-rotate, whose OpenMP and CUDA programs
  print PASS or FAIL) in both directions, executors per language {cuda:
  none, omp: native}, the proxy toolchain nvcpp-multicore for OpenMP,
  parsed diagnostics on, at most two corrections, 2048 new tokens per
  request, one trial per direction, and not faithful.
- Each tests/fixtures/recipes/p17-gpu-*.yaml loads and differs from the CPU
  recipe in exactly one device section. lassi.core.runner probe_devices,
  the runner's probe step before any component is built or any directory
  exists, refuses it with a RunError naming that section's key and kind
  index 0, from the framework build (the hf recipes: a CPU torch build) or
  from file metadata (the exec recipes: no /dev/nvidiactl; a /dev/kfd that
  exists but that access(2) denies, alpha01's case, OQ-002). The probes run
  on a SYNTHETIC host root and open nothing under it.
- The CPU recipe under the same probes gives one record, model.device on
  kind cpu, with the framework build's name and version.

probe_devices is called directly: `lassi run` checks the generated prompt
set (absent in CI) before it probes. Every framework build here is
SYNTHETIC in torch's shape; no framework is imported, no file under the
SYNTHETIC root is opened, and no value in this module is a measurement.
"""

from __future__ import annotations

import builtins
import io
import os
import pathlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.core import runner  # noqa: F401  (importing the runner registers every component)
from lassi.core.devices import FrameworkBuild
from lassi.core.recipe import load_recipe
from lassi.core.registry import DEFAULT_REGISTRY, device_path
from lassi.core.runner import RunError, probe_devices
from lassi.executors.devices import CpuProbe, CudaProbe, RocmProbe
from lassi.llm.hf_local import HFLocalBackend

REPO = Path(__file__).resolve().parents[2]
RECIPES = REPO / "tests" / "fixtures" / "recipes"
CPU_RECIPE = RECIPES / "p17-cpu-hf.yaml"
SUITE_MANIFEST = REPO / "assets" / "bench" / "lassi-hecbench-10.yaml"
BIBLE = REPO / "docs" / "BIBLE.md"

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
MODEL_REVISION = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
ITEM = "matrix-rotate"
BOTH_DIRECTIONS = [{"source": "omp", "target": "cuda"}, {"source": "cuda", "target": "omp"}]

# Each refusal recipe: the device key it names, the kind, and the reason the refusal gives on this root.
GPU_RECIPES = {
    "p17-gpu-hf-cuda.yaml": ("model.device", "cuda", "is not a CUDA build, so it cannot reach a cuda device"),
    "p17-gpu-hf-rocm.yaml": ("model.device", "rocm", "is not a HIP build, so it cannot reach a rocm device"),
    "p17-gpu-exec-cuda.yaml": ("executor.cuda.device", "cuda", "/dev/nvidiactl does not exist on this host"),
    "p17-gpu-exec-rocm.yaml": (
        "executor.cuda.device",
        "rocm",
        "/dev/kfd exists but this user may not read and write it; the GPU is not usable here",
    ),
}
# SYNTHETIC: the CPU torch build's metadata in the shape HFLocalBackend.framework() reads it.
CPU_BUILD = FrameworkBuild(name="torch", version="2.14.1+cpu", cuda=None, hip=None)
SYNTHETIC_CPU_NAME = "SYNTHETIC CPU model for the gate recipes"


def resolved(name: str) -> dict[str, Any]:
    """Return the resolved data of a gate recipe, loaded with lassi's default registry and roots."""
    return load_recipe(RECIPES / name).data


def alpha01_like_root(root: Path) -> Path:
    """Build a SYNTHETIC host root shaped like alpha01 for the probes: /dev/kfd present, no /dev/nvidiactl."""
    (root / "dev").mkdir(parents=True)
    (root / "dev" / "kfd").write_bytes(b"")
    (root / "proc").mkdir()
    (root / "proc" / "cpuinfo").write_bytes(f"processor\t: 0\nmodel name\t: {SYNTHETIC_CPU_NAME}\n".encode("ascii"))
    (root / "proc" / "meminfo").write_bytes(b"MemTotal:       1024 kB\n")
    return root


def watch_opens(monkeypatch: pytest.MonkeyPatch, root: Path) -> list[str]:
    """Record every open of a path under `root` through builtins.open, io.open, Path.open, and os.open."""
    opened: list[str] = []
    prefix = os.path.normcase(str(root))

    def recorder(original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(file: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(file, (str, os.PathLike)) and os.path.normcase(os.fspath(file)).startswith(prefix):
                opened.append(os.fspath(file))
            return original(file, *args, **kwargs)

        return wrapped

    monkeypatch.setattr(builtins, "open", recorder(builtins.open))
    monkeypatch.setattr(io, "open", recorder(io.open))
    monkeypatch.setattr(os, "open", recorder(os.open))
    original_path_open = pathlib.Path.open
    monkeypatch.setattr(pathlib.Path, "open", lambda self, *a, **k: recorder(original_path_open)(self, *a, **k))
    return opened


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], list[Path], list[str]]:
    """Return the probes on a SYNTHETIC alpha01-like root, the access(2) questions asked, and the opens under /dev.

    hf_local's framework() answers the SYNTHETIC CPU build, so no framework
    is imported.
    """
    root = alpha01_like_root(tmp_path / "host")
    asked: list[Path] = []

    def deny(path: Path) -> bool:
        asked.append(Path(path))
        return False

    monkeypatch.setattr(HFLocalBackend, "framework", staticmethod(lambda: CPU_BUILD))
    probes = {
        "cpu": CpuProbe(root=root),
        "cuda": CudaProbe(root=root, access=deny),
        "rocm": RocmProbe(root=root, access=deny),
    }
    return probes, asked, watch_opens(monkeypatch, root / "dev")


# ---------------------------------------------------------------------------
# The hf_local CPU recipe


def test_the_cpu_recipe_binds_hf_local_on_the_cpu_with_the_bible_gate_model() -> None:
    model = resolved("p17-cpu-hf.yaml")["model"]
    assert model == {
        "backend": "hf_local",
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "seed": 0,
        "device": {"kind": "cpu"},
    }


def test_the_gate_model_pin_is_the_bible_s() -> None:
    line = next(line for line in BIBLE.read_text(encoding="utf-8").splitlines() if line.startswith("- Gate model"))
    assert f"{MODEL_ID} at revision {MODEL_REVISION}" in line, line


def test_the_cpu_recipe_runs_one_passfail_item_in_both_directions() -> None:
    data = resolved("p17-cpu-hf.yaml")
    assert data["bench"]["suite"] == "lassi-hecbench-10" and data["bench"]["split"] == "eval"
    assert data["bench"]["items"] == [ITEM]
    assert data["directions"] == BOTH_DIRECTIONS
    item = yaml.safe_load(SUITE_MANIFEST.read_text(encoding="utf-8"))["items"][ITEM]
    assert item["split"] == "eval" and "omp" in item["passfail"], item


def test_the_cpu_recipe_uses_the_proxy_and_executors_per_language() -> None:
    data = resolved("p17-cpu-hf.yaml")
    assert data["executor"] == {"cuda": "none", "omp": "native"}
    assert data["toolchain"] == {"cuda": "nvcc-sm80", "omp": "nvcpp-multicore"}
    assert data["oracle"] == {"kind": "stdout_mask", "passfail": True}
    assert data["faithful"] is False, "the proxy is never faithful"


def test_the_cpu_recipe_bounds_the_model_calls() -> None:
    data = resolved("p17-cpu-hf.yaml")
    assert data["fixes"]["parsed_diagnostics"] is True, "compile text is capped so correction prompts fit"
    assert data["loop"]["max_corrections"] == 2
    assert data["llm"]["sampling"]["max_tokens"] == 2048
    assert data["trials"]["n"] == 1


def test_the_cpu_recipe_gives_one_cpu_record_under_the_probes(host: tuple[dict[str, Any], list, list]) -> None:
    probes, asked, opened = host
    recipe = load_recipe(CPU_RECIPE)
    records = probe_devices(recipe, DEFAULT_REGISTRY, probes)
    assert [(item.key, item.kind, item.indices) for item in records] == [("model.device", "cpu", [])]
    assert (records[0].framework, records[0].framework_version) == ("torch", "2.14.1+cpu")
    assert records[0].name == SYNTHETIC_CPU_NAME
    assert (asked, opened) == ([], [])


# ---------------------------------------------------------------------------
# The four GPU refusal recipes


@pytest.mark.parametrize("name", sorted(GPU_RECIPES))
def test_each_gpu_recipe_changes_exactly_one_device_section(name: str) -> None:
    key, kind, _ = GPU_RECIPES[name]
    cpu = load_recipe(CPU_RECIPE)
    gpu = load_recipe(RECIPES / name)
    cpu_devices = {device_path(item): item.config.get("device") for item in cpu.bindings}
    gpu_devices = {device_path(item): item.config.get("device") for item in gpu.bindings}
    paths = cpu_devices.keys() | gpu_devices.keys()
    changed = {path for path in paths if cpu_devices.get(path) != gpu_devices.get(path)}
    assert changed == {key}, changed
    assert gpu_devices[key] == {"kind": kind, "indices": [0]}
    others = {k: v for k, v in gpu.data.items() if k not in ("model", "executor", "project")}
    assert others == {k: v for k, v in cpu.data.items() if k not in ("model", "executor", "project")}


@pytest.mark.parametrize("name", sorted(GPU_RECIPES))
def test_each_gpu_recipe_is_refused_naming_its_key_and_kind(
    name: str, host: tuple[dict[str, Any], list, list]
) -> None:
    probes, asked, opened = host
    key, kind, reason = GPU_RECIPES[name]
    with pytest.raises(RunError) as info:
        probe_devices(load_recipe(RECIPES / name), DEFAULT_REGISTRY, probes)
    message = str(info.value)
    assert f"{key} names {kind} 0, which this host cannot provide: " in message, message
    assert reason in message, message
    assert message.endswith("a run never falls back to another device"), message
    assert opened == [], f"a probe opened a node: {opened}"


def test_only_the_exec_rocm_recipe_asks_access_about_kfd(host: tuple[dict[str, Any], list, list]) -> None:
    probes, asked, _ = host
    for name in sorted(GPU_RECIPES):
        with pytest.raises(RunError):
            probe_devices(load_recipe(RECIPES / name), DEFAULT_REGISTRY, probes)
    assert [path.name for path in asked] == ["kfd"], "only the rocm probe asks access(2), and only about /dev/kfd"


@pytest.mark.parametrize("name", sorted(GPU_RECIPES))
def test_each_gpu_recipe_names_part_a_and_part_b_in_its_header(name: str) -> None:
    text = (RECIPES / name).read_text(encoding="ascii")
    header = " ".join(line.lstrip("# ") for line in text.splitlines() if line.startswith("#"))
    assert re.search(r"gate part \(a\)", header), header
    assert "P17.G" in header and "OQ-040" in header, header
