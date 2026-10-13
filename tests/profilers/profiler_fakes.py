"""Fakes and file helpers for the profiler tests (task P17.7).

The profilers read telemetry through a source seam
(lassi.profilers.power.PowerSource): a source's read_w() returns the current
power in watts or raises TelemetryError. The tests pass a FakeSource, which
answers from a script, through the profilers' `source` keyword, and a
PhaseClock, which the test moves by hand, through their `clock` keyword, so
every window and integral is exact. The real sources are tested on SYNTHETIC
host roots written here: the KFD topology and the amdgpu hwmon files for
rocm_smi, and the NVIDIA procfs entries for nvml (with a fake pynvml module
in tests/profilers/test_nvml.py). No GPU, driver library, or SMI tool is
touched, and no value here is a measurement.
"""

from __future__ import annotations

import builtins
import importlib
import io
import os
import pathlib
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# The thread name a power profiler's sampler runs under (lassi.profilers.power).
SAMPLER_THREAD = "lassi-power-sampler"
# An interval far longer than any test window: no sample is taken between start() and stop().
NO_TICK_MS = 60000
ROCM_0 = {"kind": "rocm", "indices": [0]}
CUDA_0 = {"kind": "cuda", "indices": [0]}
# SYNTHETIC KFD topology nodes: directory name -> (gpu_id, drm_render_minor). Node 0 is a CPU node (gpu_id 0).
# Numerically "10" comes after "2"; as text it comes before it, so index 2 tells the orders apart.
KFD_NODES: dict[str, tuple[str, int | None]] = {
    "0": ("0", 0),
    "1": ("11111", 128),
    "2": ("22222", 129),
    "10": ("33333", 130),
}
# SYNTHETIC hwmon power1_average texts, in microwatts, by render minor; each lives in its own hwmon<k> directory.
HWMON_TEXT = {128: "152000000\n", 129: "75500000\n", 130: "300250000\n"}
HWMON_WATTS = {128: 152.0, 129: 75.5, 130: 300.25}
HWMON_NAME = {128: "hwmon3", 129: "hwmon5", 130: "hwmon1"}


def profilers_module(name: str = "") -> ModuleType:
    """Import lassi.profilers or lassi.profilers.<name>, failing the test clearly while it does not exist.

    An import error about another module (a vendor library imported at
    module level, say) is not hidden: it propagates.
    """
    full = "lassi.profilers" + (f".{name}" if name else "")
    try:
        return importlib.import_module(full)
    except ModuleNotFoundError as error:
        if error.name is not None and (full == error.name or full.startswith(f"{error.name}.")):
            pytest.fail(f"{full} does not exist yet (task P17.7): {error}")
        raise


def attribute(module: ModuleType, name: str) -> Any:
    """Return module.<name>, failing the test clearly while task P17.7 has not added it."""
    if not hasattr(module, name):
        pytest.fail(f"{module.__name__} has no {name} yet (task P17.7)")
    return getattr(module, name)


def power() -> ModuleType:
    """Return lassi.profilers.power."""
    return profilers_module("power")


def telemetry_error(message: str = "SYNTHETIC: the reading failed") -> BaseException:
    """Return a TelemetryError, the exception a source raises for one failed read."""
    return attribute(power(), "TelemetryError")(message)


class PhaseClock:
    """A clock the test moves by hand: every call returns `now` until the test sets another value."""

    def __init__(self, now: float = 0.0) -> None:
        """Start at `now`."""
        self.now = now
        self.calls = 0

    def __call__(self) -> float:
        """Return the current phase's time."""
        self.calls += 1
        return self.now


class FakeSource:
    """A PowerSource that answers from a script: each read takes the next item, watts or an exception to raise.

    When the script runs out, each read returns `default`, or raises
    AssertionError when there is none. `reached` is set once `notify_after`
    reads have been made, so a test can wait for the sampler thread without
    sleeping on a guess. Reads may come from the sampler thread.
    """

    def __init__(
        self, script: Iterable[float | BaseException] = (), *, default: float | None = None, notify_after: int = 0
    ) -> None:
        """Keep the script, the default reading, and when to set `reached`."""
        self.script = list(script)
        self.default = default
        self.notify_after = notify_after
        self.reads = 0
        self.reached = threading.Event()
        self._lock = threading.Lock()

    def read_w(self) -> float:
        """Return the next scripted reading in watts, or raise the next scripted exception."""
        with self._lock:
            self.reads += 1
            if self.notify_after and self.reads >= self.notify_after:
                self.reached.set()
            item = self.script.pop(0) if self.script else self.default
        if item is None:
            raise AssertionError("the source was read more often than the test scripted")
        if isinstance(item, BaseException):
            raise item
        return item


def sampler_threads() -> list[threading.Thread]:
    """Return the live threads named as a power profiler's sampler."""
    return [thread for thread in threading.enumerate() if thread.name == SAMPLER_THREAD and thread.is_alive()]


def write_files(root: Path, files: Mapping[str, str]) -> Path:
    """Write SYNTHETIC files under `root` (relative POSIX path -> text, ASCII, LF) and return `root`."""
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def kfd_properties(minor: int | None) -> str:
    """Return a SYNTHETIC KFD node properties file: one "name value" line per property, as the kernel prints them."""
    lines = ["cpu_cores_count 0", "simd_count 304", "gfx_target_version 90402", "vendor_id 4098"]
    if minor is not None:
        lines.append(f"drm_render_minor {minor}")
    return "".join(f"{line}\n" for line in lines)


def rocm_root(
    root: Path,
    nodes: Mapping[str, tuple[str, int | None]] = KFD_NODES,
    hwmon: Mapping[int, Mapping[str, str | None]] | None = None,
) -> Path:
    """Write a SYNTHETIC AMD host root: dev/kfd, the KFD topology, a render node and its hwmon files per GPU.

    `hwmon` maps a render minor to its hwmon directories (name -> the text
    of power1_average, or None to leave the attribute out); by default each
    GPU has one directory holding HWMON_TEXT for its minor.
    """
    files = {"dev/kfd": "", "sys/module/amdgpu/version": "SYNTHETIC-6.12.12\n"}
    for name, (gpu_id, minor) in nodes.items():
        base = f"sys/class/kfd/kfd/topology/nodes/{name}"
        files[f"{base}/gpu_id"] = f"{gpu_id}\n"
        files[f"{base}/properties"] = kfd_properties(minor)
        if gpu_id != "0" and minor:
            files[f"dev/dri/renderD{minor}"] = ""
    layout = hwmon if hwmon is not None else {minor: {HWMON_NAME[minor]: text} for minor, text in HWMON_TEXT.items()}
    for minor, directories in layout.items():
        for directory, text in directories.items():
            base = f"sys/class/drm/renderD{minor}/device/hwmon/{directory}"
            files[f"{base}/name"] = "amdgpu\n"
            if text is not None:
                files[f"{base}/power1_average"] = text
    return write_files(root, files)


def hwmon_file(root: Path, minor: int) -> Path:
    """Return the power1_average file of render minor `minor` in the default rocm_root layout."""
    return root / f"sys/class/drm/renderD{minor}/device/hwmon/{HWMON_NAME[minor]}/power1_average"


def nvidia_information(model: str, bus: str, minor: int) -> str:
    """Return a SYNTHETIC proc/driver/nvidia/gpus/<bus id>/information file in the driver's layout."""
    lines = [f"Model: \t\t {model}", "IRQ:   \t\t 99", "GPU UUID: \t GPU-SYNTHETIC", "Bus Type: \t PCIe"]
    lines += [f"Bus Location: \t {bus}", f"Device Minor: \t {minor}", "GPU Firmware: \t N/A"]
    return "".join(f"{line}\n" for line in lines)


def cuda_root(root: Path, gpus: Mapping[str, tuple[str, int]]) -> Path:
    """Write a SYNTHETIC NVIDIA host root: one procfs entry and one node per GPU, and the version file.

    `gpus` maps a PCI bus id (as the driver names the entry, 0000:01:00.0)
    to (model, device minor). The entry directory writes each ':' as '-',
    since the test host's file system may refuse ':' in a name; the entry's
    information file keeps the bus id as written, in its Bus Location line.
    """
    files = {"dev/nvidiactl": "", "proc/driver/nvidia/version": "NVRM version: SYNTHETIC  999.88.77\n"}
    for bus, (model, minor) in gpus.items():
        files[f"proc/driver/nvidia/gpus/{bus.replace(':', '-')}/information"] = nvidia_information(model, bus, minor)
        files[f"dev/nvidia{minor}"] = ""
    return write_files(root, files)


def normalized_bus(value: object) -> str:
    """Return a PCI bus id as text in one form: bytes decoded, lowercase, each '-' read as ':'."""
    text = value.decode("ascii") if isinstance(value, bytes) else str(value)
    return text.lower().replace("-", ":")


def opened_under(monkeypatch: pytest.MonkeyPatch, root: Path) -> list[str]:
    """Record every file opened under `root` (builtins, io, os, and pathlib opens) and return the list."""
    opened: list[str] = []
    prefix = os.path.normcase(os.path.abspath(root))

    def note(target: Any) -> None:
        if isinstance(target, int):
            return
        path = os.path.normcase(os.path.abspath(os.fspath(target)))
        if path == prefix or path.startswith(prefix + os.sep):
            opened.append(path)

    real_open, real_os_open, real_path_open = builtins.open, os.open, pathlib.Path.open

    def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        note(file)
        return real_open(file, *args, **kwargs)

    def guarded_os_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        note(path)
        return real_os_open(path, *args, **kwargs)

    def guarded_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        note(self)
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(pathlib.Path, "open", guarded_path_open)
    return opened
