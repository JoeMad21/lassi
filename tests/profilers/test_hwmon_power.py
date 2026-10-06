"""Tests for the rocm_smi profiler (task P17.7): amdgpu hwmon power read from sysfs, with no tool and no device node.

The file name keeps the profiler's name out of any remote command text that
names a test file (the gate refuses command text naming the SMI tools).

Bible: Component Interfaces (Profiler; supports_power), Design Principle 9,
Agent Rule 1; plans/p17-portable.md, P17.7 (the first acceptance line) and
the plan's Constraints (OQ-002: nothing opens a GPU node, and no SMI tool
starts).

The contract these tests fix (lassi.profilers.rocm_smi):

- RocmSmiProfiler, registered as Profiler "rocm_smi", declares
  supports_power and takes_device, takes the config key interval_ms, and is
  built with keyword-only device, interval_ms, source, root, and clock. Its
  device section names kind rocm and exactly one index.
- Its real source, HwmonSource(index, root), maps the index as the gpu
  executor does: lassi.executors.devices gpu_inventory("rocm", root)[index]
  names the render node /dev/dri/renderD<m> (KFD topology order, node
  directories in numeric order), of which only the name renderD<m> is used;
  the node itself is never touched. It reads
  <root>/sys/class/drm/renderD<m>/device/hwmon/<the one hwmon directory>/
  power1_average, in microwatts, as watts (int(text) / 1e6), anew at each
  read. A missing, unreadable, non-integer, or negative value is a
  TelemetryError.
- Built through the profiler: an index the inventory lacks, a GPU with no
  render node, no hwmon directory or more than one, or a first read that
  fails is TelemetryUnavailable, whose message names the host path (such as
  /sys/class/drm/renderD128/device/hwmon) and never the root it was read
  under.
- Nothing opens a file under the root's dev/, and no process starts, while
  the profiler is built, profiles a window, and reads.

Every host root is SYNTHETIC (profiler_fakes rocm_root), every reading is
SYNTHETIC text, and the paths are unverified until P17.12 and P17.13
measure a GPU host. No value in this module is a measurement.
"""

from __future__ import annotations

import inspect
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from profiler_fakes import (
    CUDA_0,
    HWMON_WATTS,
    KFD_NODES,
    NO_TICK_MS,
    ROCM_0,
    PhaseClock,
    attribute,
    hwmon_file,
    opened_under,
    power,
    profilers_module,
    rocm_root,
)

from lassi.core.record import Profile
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.executors.devices import gpu_inventory

# The host path of render minor 128's hwmon directory, as a refusal names it.
HWMON_128 = "/sys/class/drm/renderD128/device/hwmon"


def rocm_smi() -> ModuleType:
    """Return lassi.profilers.rocm_smi."""
    return profilers_module("rocm_smi")


def source(index: int, root: Path) -> Any:
    """Build the real HwmonSource for `index` under `root`."""
    return attribute(rocm_smi(), "HwmonSource")(index=index, root=root)


def profiler(root: Path, clock: Any = None, device: Mapping[str, Any] = ROCM_0, **changes: Any) -> Any:
    """Build a RocmSmiProfiler on its real source under `root`, at NO_TICK_MS unless `changes` say otherwise."""
    config: dict[str, Any] = {"device": dict(device), "interval_ms": NO_TICK_MS, "root": root, **changes}
    if clock is not None:
        config["clock"] = clock
    return attribute(rocm_smi(), "RocmSmiProfiler")(**config)


def unavailable_message(root: Path, **changes: Any) -> str:
    """Build the profiler under `root`, which must raise TelemetryUnavailable; return its message."""
    with pytest.raises(attribute(power(), "TelemetryUnavailable")) as info:
        profiler(root, **changes)
    message = str(info.value)
    assert str(root) not in message and root.as_posix() not in message, f"the message names the test root: {message}"
    return message


# ---------------------------------------------------------------------------
# The source: the index, the file, and the unit


def test_rocm_smi_reads_microwatts_as_watts_from_the_index_hwmon(tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "host")
    assert source(0, root).read_w() == 152.0
    assert source(1, root).read_w() == 75.5


def test_rocm_smi_maps_the_index_through_the_kfd_inventory(tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "host")
    inventory = gpu_inventory("rocm", root)
    assert [entry.node for entry in inventory] == ["/dev/dri/renderD128", "/dev/dri/renderD129", "/dev/dri/renderD130"]
    for entry in inventory:
        minor = int(str(entry.node).rsplit("renderD", 1)[1])
        assert source(entry.index, root).read_w() == HWMON_WATTS[minor], f"index {entry.index} reads {entry.node}"


def test_rocm_smi_reads_the_file_anew_at_each_read(tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "host")
    found = source(0, root)
    assert found.read_w() == 152.0
    hwmon_file(root, 128).write_bytes(b"160500000\n")
    assert found.read_w() == 160.5


@pytest.mark.parametrize(
    "text", ["abc\n", "\n", "-5000000\n", "1.5e8\n", "152 W\n"], ids=["word", "empty", "negative", "float", "unit"]
)
def test_rocm_smi_bad_text_is_a_failed_read(tmp_path: Path, text: str) -> None:
    root = rocm_root(tmp_path / "host")
    found = source(0, root)
    hwmon_file(root, 128).write_bytes(text.encode("ascii"))
    with pytest.raises(attribute(power(), "TelemetryError")):
        found.read_w()


def test_rocm_smi_missing_file_is_a_failed_read(tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "host")
    found = source(0, root)
    hwmon_file(root, 128).unlink()
    with pytest.raises(attribute(power(), "TelemetryError")):
        found.read_w()


# ---------------------------------------------------------------------------
# Refusals at build: no telemetry for the named GPU


def test_rocm_smi_without_hwmon_is_telemetry_unavailable(tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "host", hwmon={129: {"hwmon5": "75500000\n"}})
    assert HWMON_128 in unavailable_message(root)


def test_rocm_smi_with_two_hwmon_directories_is_telemetry_unavailable(tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "host", hwmon={128: {"hwmon3": "152000000\n", "hwmon4": "152000000\n"}})
    assert HWMON_128 in unavailable_message(root), "two hwmon directories leave the reading ambiguous"


def test_rocm_smi_without_the_power_attribute_is_telemetry_unavailable(tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "host", hwmon={128: {"hwmon3": None}})
    assert "power1_average" in unavailable_message(root)


def test_rocm_smi_on_an_index_the_inventory_lacks_is_telemetry_unavailable(tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "host")
    gpus = sum(1 for gpu_id, _ in KFD_NODES.values() if gpu_id != "0")
    unavailable_message(root, device={"kind": "rocm", "indices": [gpus]})


def test_rocm_smi_on_a_gpu_without_a_render_node_is_telemetry_unavailable(tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "host", nodes={"0": ("0", 0), "1": ("11111", None)})
    unavailable_message(root)


def test_rocm_smi_refuses_another_kind_or_two_indices_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = rocm_root(tmp_path / "host")
    opened = opened_under(monkeypatch, root)
    with pytest.raises(ValueError, match="profiler.device"):
        profiler(root, device=CUDA_0)
    with pytest.raises(ValueError, match="profiler.device"):
        profiler(root, device={"kind": "rocm", "indices": [0, 1]})
    assert opened == [], "a refused setting reads nothing"


# ---------------------------------------------------------------------------
# The profiler on the real source, and what it never does


def test_rocm_smi_profiles_a_window_in_watts_and_joules(tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "host")
    clock = PhaseClock(100.0)
    built = profiler(root, clock)
    built.start()
    hwmon_file(root, 128).write_bytes(b"160000000\n")
    clock.now = 102.0
    assert built.stop() == Profile(runtime_s=2.0, avg_power_w=156.0, energy_j=312.0)


def test_rocm_smi_opens_nothing_under_dev_and_starts_no_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = rocm_root(tmp_path / "host")
    assert (root / "dev" / "dri" / "renderD128").exists() and (root / "dev" / "kfd").exists()

    def no_process(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"a process was started: {args!r}")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    monkeypatch.setattr(os, "system", no_process)
    for name in ("posix_spawn", "posix_spawnp", "fork", "execv", "execve"):
        monkeypatch.setattr(os, name, no_process, raising=False)
    sysfs = opened_under(monkeypatch, root / "sys")
    nodes = opened_under(monkeypatch, root / "dev")
    clock = PhaseClock(1.0)
    built = profiler(root, clock)
    built.start()
    clock.now = 2.0
    built.stop()
    source(1, root).read_w()
    assert nodes == [], f"a device node was opened: {nodes}"
    assert any(path.endswith("power1_average") for path in sysfs), "the reading comes from the sysfs attribute"


def test_rocm_smi_is_registered_with_supports_power_and_takes_device() -> None:
    profilers_module()
    entry = DEFAULT_REGISTRY.get("Profiler", "rocm_smi")
    cls = attribute(rocm_smi(), "RocmSmiProfiler")
    assert entry.factory is cls and cls.name == "rocm_smi"
    assert entry.capabilities == frozenset({"supports_power", "takes_device"})
    assert entry.config_keys == frozenset({"interval_ms"}), "idle_window_s is P9's; the registry refuses it"
    parameters = inspect.signature(cls).parameters
    for key in ("device", "interval_ms", "source", "root", "clock"):
        assert parameters[key].kind is inspect.Parameter.KEYWORD_ONLY, key
