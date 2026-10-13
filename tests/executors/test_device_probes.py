"""Tests for the host device probes in lassi/executors (task P17.2).

Bible: Component Interfaces (contract rules), Result Record (provenance),
Agent Rule 1, Design Principle 9; plans/p17-portable.md, the planning
decisions "Explicit devices (P17.2)" and "Placement (Design Principle 9)";
OQ-002 (nothing opens /dev/kfd or a /dev/dri node, a probe included).

lassi.executors.devices holds one probe per device kind, each built with a
host root (default "/") and an access check (default access(2) with read
and write, which opens nothing). A probe reads file metadata (stat, access)
and procfs or sysfs text under the root only: it never opens a device node,
starts no process, and imports no framework. Importing lassi.executors
registers the three probes in lassi.core.devices.DEFAULT_PROBES by kind.

- CpuProbe (kind cpu): count os.cpu_count(); name the CPU model the native
  executor reads (lassi.executors.native, "model name" in proc/cpuinfo),
  None without one; memory_bytes from proc/meminfo's MemTotal (in kB, so
  times 1024), None without one; no driver or runtime. It never refuses.
- RocmProbe (kind rocm): refuses when dev/kfd does not exist, and when it
  exists but the access check denies read and write (alpha01, OQ-002), before
  any procfs or sysfs read; otherwise count is the KFD topology nodes under
  sys/class/kfd/kfd/topology/nodes whose gpu_id is not 0, and driver the text
  of sys/module/amdgpu/version when present.
- CudaProbe (kind cuda): refuses when dev/nvidiactl does not exist or the
  access check denies it; otherwise count is the entries under
  proc/driver/nvidia/gpus and driver the version token of the first line of
  proc/driver/nvidia/version.

The GPU paths stay unverified until P17.12 measures them; these tests use
SYNTHETIC host roots under tmp_path, so they run on Windows. Entries the real
procfs names by PCI bus id (with ':') get names Windows allows. No value in
this module is a measurement.
"""

from __future__ import annotations

import builtins
import importlib
import io
import os
import pathlib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.executors import NativeExecutor
from lassi.executors import native as native_module

SYNTHETIC_MODEL = "SYNTHETIC CPU model 9000"
HOST_CPU = "host CPU (native)"
CPUINFO = f"processor\t: 0\nvendor_id\t: SyntheticVendor\nmodel name\t: {SYNTHETIC_MODEL}\nflags\t\t: fpu\n\n"
MEMINFO = "MemTotal:       16384 kB\nMemFree:         1024 kB\nMemAvailable:    2048 kB\n"
AMDGPU_VERSION = "SYNTHETIC-6.8.5"
NVIDIA_VERSION = "999.88.77"
# The first line of proc/driver/nvidia/version in the two module flavors' layout, with a SYNTHETIC version and date.
NVIDIA_LINES = {
    "proprietary": f"NVRM version: NVIDIA UNIX x86_64 Kernel Module  {NVIDIA_VERSION}  SYNTHETIC date\n",
    "open": f"NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  {NVIDIA_VERSION}  Release Build  SYNTHETIC\n",
}


@pytest.fixture(scope="module")
def probes() -> ModuleType:
    """Import lassi.executors.devices, failing each test clearly while it does not exist."""
    try:
        return importlib.import_module("lassi.executors.devices")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.executors.devices does not exist yet (task P17.2): {error}")


@pytest.fixture(scope="module")
def core() -> ModuleType:
    """Import lassi.core.devices, failing each test clearly while it does not exist."""
    try:
        return importlib.import_module("lassi.core.devices")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.core.devices does not exist yet (task P17.2): {error}")


def write_files(root: Path, files: dict[str, str]) -> Path:
    """Write SYNTHETIC files under `root` (relative POSIX path -> text, ASCII, LF) and return `root`."""
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def rocm_root(root: Path, *, kfd: bool = True, gpu_ids: tuple[str, ...] = ("0", "11111", "22222")) -> Path:
    """Write a SYNTHETIC ROCm host root: a kfd node, KFD topology nodes with these gpu_id values, a driver."""
    files = {f"sys/class/kfd/kfd/topology/nodes/{index}/gpu_id": f"{gpu_id}\n" for index, gpu_id in enumerate(gpu_ids)}
    files["sys/module/amdgpu/version"] = f"{AMDGPU_VERSION}\n"
    if kfd:
        files["dev/kfd"] = ""
    return write_files(root, files)


def cuda_root(root: Path, *, nvidiactl: bool = True, gpus: int = 2, flavor: str = "proprietary") -> Path:
    """Write a SYNTHETIC NVIDIA host root: the control node, `gpus` entries under gpus/, and the version file."""
    files = {f"proc/driver/nvidia/gpus/gpu-{index}/information": "Model: SYNTHETIC\n" for index in range(gpus)}
    files["proc/driver/nvidia/version"] = NVIDIA_LINES[flavor] + "GCC version: SYNTHETIC\n"
    if nvidiactl:
        files["dev/nvidiactl"] = ""
    return write_files(root, files)


class Access:
    """A stand-in access check that answers `allow` and records each path it was asked about."""

    def __init__(self, allow: bool) -> None:
        """Answer `allow` to every question."""
        self.allow = allow
        self.asked: list[Path] = []

    def __call__(self, path: Path) -> bool:
        """Record the path and answer."""
        self.asked.append(Path(path))
        return self.allow


def guard_opens(monkeypatch: pytest.MonkeyPatch, root: Path) -> list[str]:
    """Record every file opened under `root` and fail any open under <root>/dev; return the record.

    builtins.open, io.open, os.open, and pathlib.Path.open are wrapped, since
    Python 3.10's pathlib opens through its own accessor, not io.open. An
    open under <root>/dev raises AssertionError and is recorded too, so a
    probe that catches the error still fails the test.
    """
    opened: list[str] = []
    prefix = os.path.normcase(os.path.abspath(root))
    devices = os.path.join(prefix, "dev")

    def note(target: Any) -> None:
        if isinstance(target, int):
            return
        path = os.path.normcase(os.path.abspath(os.fspath(target)))
        if path == prefix or path.startswith(prefix + os.sep):
            opened.append(path)
        if path == devices or path.startswith(devices + os.sep):
            raise AssertionError(f"a probe opened a device node: {path}")

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


def refusal(core: ModuleType, probe: Any) -> str:
    """Return the message of the DeviceUnavailable `probe.probe()` raises."""
    with pytest.raises(core.DeviceUnavailable) as info:
        probe.probe()
    return str(info.value)


# ---------------------------------------------------------------------------
# The cpu probe


def test_cpu_probe_reads_model_and_memory_under_the_root(
    probes: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_files(tmp_path / "host", {"proc/cpuinfo": CPUINFO, "proc/meminfo": MEMINFO})
    facts = probes.CpuProbe(root=root).probe()
    assert probes.CpuProbe(root=root).kind == "cpu"
    assert (facts.count, facts.name, facts.memory_bytes) == (os.cpu_count(), SYNTHETIC_MODEL, 16384 * 1024)
    assert (facts.driver, facts.runtime) == (None, None)
    # The model text is the native executor's reading of the same file, so the two devices compare equal.
    monkeypatch.setattr(native_module, "CPUINFO", root / "proc" / "cpuinfo")
    assert NativeExecutor().device() == f"{HOST_CPU}: {facts.name}"


def test_cpu_probe_collapses_the_model_as_the_native_executor_does(
    probes: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "processor\t: 0\nmodel name\t:   SYNTHETIC   CPU\tmodel \x07 9000  \n\n"
    root = write_files(tmp_path / "host", {"proc/cpuinfo": text})
    facts = probes.CpuProbe(root=root).probe()
    monkeypatch.setattr(native_module, "CPUINFO", root / "proc" / "cpuinfo")
    assert NativeExecutor().device() == f"{HOST_CPU}: {facts.name}"
    assert facts.name == "SYNTHETIC CPU model \\x07 9000"


@pytest.mark.parametrize(
    "files",
    [
        pytest.param({}, id="no-proc-files"),
        pytest.param({"proc/cpuinfo": "processor\t: 0\n\n", "proc/meminfo": "MemFree: 1 kB\n"}, id="no-fields"),
        pytest.param({"proc/cpuinfo": "model name\t:   \n", "proc/meminfo": "MemTotal: lots kB\n"}, id="empty-values"),
    ],
)
def test_cpu_probe_without_proc_files_still_counts(probes: ModuleType, tmp_path: Path, files: dict[str, str]) -> None:
    root = write_files(tmp_path / "host", files)
    facts = probes.CpuProbe(root=root).probe()
    assert facts.count == os.cpu_count()
    assert (facts.name, facts.memory_bytes, facts.driver, facts.runtime) == (None, None, None, None)


# ---------------------------------------------------------------------------
# The rocm probe


def test_rocm_probe_refuses_a_host_without_kfd(probes: ModuleType, core: ModuleType, tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "host", kfd=False)
    access = Access(allow=True)
    message = refusal(core, probes.RocmProbe(root=root, access=access))
    assert "kfd" in message, message
    assert access.asked == [], "a node that does not exist is not asked about"


def test_rocm_probe_refuses_kfd_that_refuses_access(
    probes: ModuleType, core: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # alpha01 (OQ-002): the node exists, but this user is not in the render group.
    root = rocm_root(tmp_path / "host")
    access = Access(allow=False)
    opened = guard_opens(monkeypatch, root)
    message = refusal(core, probes.RocmProbe(root=root, access=access))
    assert "kfd" in message, message
    assert access.asked == [root / "dev" / "kfd"]
    assert opened == [], f"the access check comes before any sysfs read: {opened}"


def test_rocm_probe_counts_gpu_nodes_and_reads_the_driver(probes: ModuleType, tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "host", gpu_ids=("0", "11111", "0", "22222", "33333"))
    access = Access(allow=True)
    probe = probes.RocmProbe(root=root, access=access)
    assert probe.kind == "rocm"
    facts = probe.probe()
    assert facts.count == 3, "CPU nodes (gpu_id 0) are not GPUs"
    assert facts.driver == AMDGPU_VERSION
    assert (facts.name, facts.memory_bytes) == (None, None), "name and memory wait for P17.12"
    assert access.asked == [root / "dev" / "kfd"]


def test_rocm_probe_without_a_driver_version_names_none(probes: ModuleType, tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "host")
    (root / "sys" / "module" / "amdgpu" / "version").unlink()
    facts = probes.RocmProbe(root=root, access=Access(allow=True)).probe()
    assert (facts.count, facts.driver) == (2, None)


def test_the_default_access_check_asks_for_read_and_write(
    probes: ModuleType, core: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = rocm_root(tmp_path / "host")
    asked: list[tuple[str, int]] = []

    def access(path: Any, mode: int, *args: Any, **kwargs: Any) -> bool:
        asked.append((os.fspath(path), mode))
        return False

    monkeypatch.setattr(os, "access", access)
    refusal(core, probes.RocmProbe(root=root))
    assert asked == [(os.fspath(root / "dev" / "kfd"), os.R_OK | os.W_OK)]


# ---------------------------------------------------------------------------
# The cuda probe


@pytest.mark.parametrize(("nvidiactl", "allow"), [(False, True), (True, False)], ids=["no-node", "access-denied"])
def test_cuda_probe_refuses_without_nvidiactl_or_access(
    probes: ModuleType,
    core: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nvidiactl: bool,
    allow: bool,
) -> None:
    root = cuda_root(tmp_path / "host", nvidiactl=nvidiactl)
    access = Access(allow=allow)
    opened = guard_opens(monkeypatch, root)
    message = refusal(core, probes.CudaProbe(root=root, access=access))
    assert "nvidiactl" in message, message
    assert access.asked == ([root / "dev" / "nvidiactl"] if nvidiactl else [])
    assert opened == [], f"a refused probe reads no procfs file: {opened}"


@pytest.mark.parametrize("flavor", sorted(NVIDIA_LINES))
def test_cuda_probe_counts_gpus_and_reads_the_driver(probes: ModuleType, tmp_path: Path, flavor: str) -> None:
    root = cuda_root(tmp_path / "host", gpus=3, flavor=flavor)
    probe = probes.CudaProbe(root=root, access=Access(allow=True))
    assert probe.kind == "cuda"
    facts = probe.probe()
    assert (facts.count, facts.driver) == (3, NVIDIA_VERSION)
    assert (facts.name, facts.memory_bytes) == (None, None), "name and memory wait for P17.12"


# ---------------------------------------------------------------------------
# No probe opens a device node; registration by kind


def test_probes_open_no_device_node(
    probes: ModuleType, core: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "host"
    rocm_root(root)
    cuda_root(root)
    write_files(root, {"proc/cpuinfo": CPUINFO, "proc/meminfo": MEMINFO, "dev/dri/renderD128": ""})
    opened = guard_opens(monkeypatch, root)
    cases: list[tuple[Callable[[], Any], bool]] = [
        (lambda: probes.CpuProbe(root=root), False),
        (lambda: probes.RocmProbe(root=root, access=Access(allow=True)), False),
        (lambda: probes.RocmProbe(root=root, access=Access(allow=False)), True),
        (lambda: probes.CudaProbe(root=root, access=Access(allow=True)), False),
        (lambda: probes.CudaProbe(root=root, access=Access(allow=False)), True),
        (lambda: probes.RocmProbe(root=tmp_path / "empty", access=Access(allow=True)), True),
        (lambda: probes.CudaProbe(root=tmp_path / "empty", access=Access(allow=True)), True),
    ]
    for make, refuses in cases:
        if refuses:
            refusal(core, make())
        else:
            make().probe()
    nodes = os.path.join(os.path.normcase(os.path.abspath(root)), "dev")
    assert not [path for path in opened if path.startswith(nodes + os.sep)], opened


def test_default_probes_cover_every_kind(probes: ModuleType, core: ModuleType) -> None:
    import lassi.executors  # noqa: F401  (importing the package registers the probes)

    registered = core.DEFAULT_PROBES
    assert sorted(registered) == ["cpu", "cuda", "rocm"]
    assert isinstance(registered["cpu"], probes.CpuProbe)
    assert isinstance(registered["rocm"], probes.RocmProbe)
    assert isinstance(registered["cuda"], probes.CudaProbe)
    assert all(probe.kind == kind for kind, probe in registered.items())


def test_the_default_probes_look_under_the_host_root(probes: ModuleType) -> None:
    for probe in (probes.CpuProbe(), probes.RocmProbe(), probes.CudaProbe()):
        assert Path(probe.root) == Path("/"), type(probe).__name__
