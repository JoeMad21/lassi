"""Tests for the nvml profiler (task P17.7), with a fake pynvml module and a SYNTHETIC host root.

Bible: Component Interfaces (Profiler; supports_power), Design Principle 9,
Agent Rule 1; plans/p17-portable.md, P17.7 (the first acceptance line);
plans/PHASE-NOTES.md, P17 (the registration rule).

The contract these tests fix (lassi.profilers.nvml):

- NvmlProfiler, registered as Profiler "nvml", declares supports_power and
  takes_device, takes the config key interval_ms, and is built with
  keyword-only device, interval_ms, source, root, and clock. Its device
  section names kind cuda and exactly one index, refused (ValueError naming
  profiler.device) before the vendor library is touched.
- Its real source, NvmlSource(index, root), imports pynvml (distribution
  nvidia-ml-py) only when built, never when the module is imported; without
  it the build is TelemetryUnavailable naming both names. It maps the index
  as the gpu executor does: index i is the i-th entry of
  <root>/proc/driver/nvidia/gpus in name order, which is PCI order
  (lassi.executors.devices gpu_inventory), and it takes the NVML handle with
  nvmlDeviceGetHandleByPciBusId for that entry's PCI bus id, never by NVML
  index, after nvmlInit(). read_w() is nvmlDeviceGetPowerUsage(handle) / 1000
  (milliwatts to watts); an NVMLError there is a TelemetryError. An index the
  host lacks, or an NVMLError from nvmlInit or the handle lookup, is
  TelemetryUnavailable.
- Built through the profiler, the first read is taken at build (a failed
  one is TelemetryUnavailable); a failed read during a window leaves the
  window's power null and keeps its runtime.

The fake pynvml stands in for the vendor library through sys.modules; the
host root is SYNTHETIC; every reading is SYNTHETIC. The entry directories
under the fake root write ':' as '-' (the test host's file system may refuse
':'), and each entry's information file keeps the bus id in its Bus
Location line, so the bus id is compared in one form (profiler_fakes
normalized_bus). No value in this module is a measurement.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from profiler_fakes import (
    CUDA_0,
    NO_TICK_MS,
    ROCM_0,
    PhaseClock,
    attribute,
    cuda_root,
    normalized_bus,
    power,
    profilers_module,
)

from lassi.core.record import Profile
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.executors.devices import gpu_inventory

# SYNTHETIC NVIDIA GPUs, written out of PCI order: bus id -> (model, device minor).
GPUS: dict[str, tuple[str, int]] = {
    "0000:af:00.0": ("SYNTHETIC GPU model C", 0),
    "0000:01:00.0": ("SYNTHETIC GPU model A", 2),
    "0000:3b:00.0": ("SYNTHETIC GPU model B", 1),
}
# The bus ids in name (PCI) order: index i is the i-th.
BUS_ORDER = ["0000:01:00.0", "0000:3b:00.0", "0000:af:00.0"]
# A fake power reading that raises NVMLError instead of answering.
FAIL = "fail"


@dataclass
class NvmlLog:
    """What the fake pynvml saw and how it answers: calls in order, bus ids asked for, and readings to give."""

    calls: list[str] = field(default_factory=list)
    buses: list[str] = field(default_factory=list)
    powers: list[int | str] = field(default_factory=list)
    fail: set[str] = field(default_factory=set)


def fake_pynvml(log: NvmlLog) -> ModuleType:
    """Return a stand-in pynvml module that answers from `log`; it touches no driver."""
    module = ModuleType("pynvml")
    module.__doc__ = "A SYNTHETIC stand-in for pynvml (tests/profilers/test_nvml.py)."

    class NVMLError(Exception):
        """The fake library's error, as pynvml raises NVMLError for a failed call."""

    state = {"initialized": False}

    def called(name: str) -> None:
        log.calls.append(name)
        if name in log.fail:
            raise NVMLError(f"SYNTHETIC: {name} failed")

    def nvmlInit() -> None:
        called("nvmlInit")
        state["initialized"] = True

    def nvmlDeviceGetHandleByPciBusId(bus: Any) -> tuple[str, str]:
        called("nvmlDeviceGetHandleByPciBusId")
        assert state["initialized"], "a handle was asked for before nvmlInit()"
        log.buses.append(normalized_bus(bus))
        return ("SYNTHETIC handle", normalized_bus(bus))

    def nvmlDeviceGetHandleByIndex(index: int) -> Any:
        raise AssertionError("the handle is taken by PCI bus id, never by NVML's own index order")

    def nvmlDeviceGetPowerUsage(handle: Any) -> int:
        called("nvmlDeviceGetPowerUsage")
        assert state["initialized"], "power was read before nvmlInit()"
        assert isinstance(handle, tuple) and handle[0] == "SYNTHETIC handle", handle
        assert log.powers, "power was read more often than the test scripted"
        item = log.powers.pop(0)
        if item == FAIL:
            raise NVMLError("SYNTHETIC: nvmlDeviceGetPowerUsage failed")
        return int(item)

    def nvmlShutdown() -> None:
        log.calls.append("nvmlShutdown")

    for value in (
        NVMLError,
        nvmlInit,
        nvmlDeviceGetHandleByPciBusId,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
        nvmlShutdown,
    ):
        setattr(module, value.__name__, value)
    return module


@pytest.fixture
def nvml_log(monkeypatch: pytest.MonkeyPatch) -> NvmlLog:
    """Put the fake pynvml in sys.modules for the test and return its log."""
    log = NvmlLog()
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml(log))
    return log


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Return a SYNTHETIC NVIDIA host root holding GPUS."""
    return cuda_root(tmp_path / "host", GPUS)


def nvml() -> ModuleType:
    """Return lassi.profilers.nvml."""
    return profilers_module("nvml")


def source(index: int, root: Path) -> Any:
    """Build the real NvmlSource for `index` under `root`."""
    return attribute(nvml(), "NvmlSource")(index=index, root=root)


def profiler(root: Path, clock: Any = None, device: Mapping[str, Any] = CUDA_0, **changes: Any) -> Any:
    """Build an NvmlProfiler on its real source under `root`, at NO_TICK_MS unless `changes` say otherwise."""
    config: dict[str, Any] = {"device": dict(device), "interval_ms": NO_TICK_MS, "root": root, **changes}
    if clock is not None:
        config["clock"] = clock
    return attribute(nvml(), "NvmlProfiler")(**config)


# ---------------------------------------------------------------------------
# The source: units, the handle, and the index


def test_nvml_reads_milliwatts_as_watts(nvml_log: NvmlLog, root: Path) -> None:
    nvml_log.powers = [123456]
    assert source(0, root).read_w() == pytest.approx(123.456)


def test_nvml_takes_the_handle_by_the_pci_bus_id_of_the_index(nvml_log: NvmlLog, root: Path) -> None:
    nvml_log.powers = [50000]
    found = source(1, root)
    assert nvml_log.buses == [BUS_ORDER[1]], "index 1 is the second entry in PCI order"
    assert nvml_log.calls.index("nvmlInit") < nvml_log.calls.index("nvmlDeviceGetHandleByPciBusId")
    assert found.read_w() == pytest.approx(50.0)


def test_nvml_index_order_matches_the_gpu_executor_inventory(nvml_log: NvmlLog, root: Path) -> None:
    inventory = gpu_inventory("cuda", root)
    assert len(inventory) == len(GPUS)
    for index in range(len(GPUS)):
        nvml_log.buses.clear()
        source(index, root)
        (bus,) = nvml_log.buses
        assert inventory[index].node == f"/dev/nvidia{GPUS[bus][1]}", (
            f"index {index}: NVML reads {bus}, the GPU the gpu executor binds as {inventory[index].node}"
        )


def test_nvml_read_error_is_a_telemetry_error(nvml_log: NvmlLog, root: Path) -> None:
    nvml_log.powers = [FAIL]
    with pytest.raises(attribute(power(), "TelemetryError")):
        source(0, root).read_w()


def test_nvml_without_the_binding_is_telemetry_unavailable_naming_it(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)
    unavailable = attribute(power(), "TelemetryUnavailable")
    with pytest.raises(unavailable) as info:
        source(0, root)
    assert "nvidia-ml-py" in str(info.value) and "pynvml" in str(info.value), info.value
    with pytest.raises(unavailable):
        profiler(root)


@pytest.mark.parametrize("failing", ["nvmlInit", "nvmlDeviceGetHandleByPciBusId"])
def test_nvml_init_or_handle_errors_are_telemetry_unavailable(nvml_log: NvmlLog, root: Path, failing: str) -> None:
    nvml_log.fail = {failing}
    with pytest.raises(attribute(power(), "TelemetryUnavailable")):
        source(0, root)


def test_an_index_the_host_lacks_is_telemetry_unavailable(nvml_log: NvmlLog, root: Path) -> None:
    with pytest.raises(attribute(power(), "TelemetryUnavailable")):
        source(len(GPUS), root)
    assert nvml_log.buses == [], "no handle is asked for a GPU the host does not list"


# ---------------------------------------------------------------------------
# The profiler on the real source


def test_nvml_error_on_the_first_read_is_telemetry_unavailable(nvml_log: NvmlLog, root: Path) -> None:
    nvml_log.powers = [FAIL]
    with pytest.raises(attribute(power(), "TelemetryUnavailable")):
        profiler(root)


def test_nvml_profiles_a_window_in_watts_and_joules(nvml_log: NvmlLog, root: Path) -> None:
    nvml_log.powers = [100000, 150000, 250000]
    clock = PhaseClock(100.0)
    built = profiler(root, clock)
    built.start()
    clock.now = 102.0
    assert built.stop() == Profile(runtime_s=2.0, avg_power_w=200.0, energy_j=400.0)


def test_nvml_read_error_during_a_run_leaves_power_null(nvml_log: NvmlLog, root: Path) -> None:
    nvml_log.powers = [100000, 150000, FAIL]
    clock = PhaseClock(100.0)
    built = profiler(root, clock)
    built.start()
    clock.now = 102.0
    assert built.stop() == Profile(runtime_s=2.0), "a failed read is never bridged (Agent Rule 1)"


def test_nvml_refuses_a_bad_setting_before_the_library(nvml_log: NvmlLog, root: Path) -> None:
    with pytest.raises(ValueError, match="profiler.device"):
        profiler(root, device=ROCM_0)
    with pytest.raises(ValueError, match="profiler.device"):
        profiler(root, device={"kind": "cuda", "indices": [0, 1]})
    with pytest.raises(ValueError, match="profiler.interval_ms"):
        profiler(root, interval_ms=0)
    assert nvml_log.calls == [], "a refused setting never reaches the vendor library"


def test_nvml_is_registered_with_supports_power_and_takes_device() -> None:
    profilers_module()
    entry = DEFAULT_REGISTRY.get("Profiler", "nvml")
    cls = attribute(nvml(), "NvmlProfiler")
    assert entry.factory is cls and cls.name == "nvml"
    assert entry.capabilities == frozenset({"supports_power", "takes_device"})
    assert entry.config_keys == frozenset({"interval_ms"}), "idle_window_s is P9's; the registry refuses it"
    parameters = inspect.signature(cls).parameters
    for key in ("device", "interval_ms", "source", "root", "clock"):
        assert parameters[key].kind is inspect.Parameter.KEYWORD_ONLY, key
