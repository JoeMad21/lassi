"""Host device probes, one per device kind (task P17.2; lassi.core.devices, the probe seam).

Device files are read only in lassi/executors and lassi/profilers (Design
Principle 9), so the probes the runner uses by default live here. Importing
lassi.executors registers CpuProbe, RocmProbe, and CudaProbe in
lassi.core.devices.DEFAULT_PROBES by kind.

Each probe is built with a host root (default "/") and reads only under it:
file metadata (lstat, access(2) for read and write, which opens nothing, and
directory listings) and procfs or sysfs text; the CPU count alone comes from
os.cpu_count(). No probe opens a device node, starts a process, or imports a
framework (OQ-002), so on a host whose GPU node exists but refuses this
user, such as alpha01's GPUs without the render group grant (OQ-040), a GPU
kind is refused from file metadata alone. A GPU probe checks its node before
it reads any procfs or sysfs file. A file that is missing or unreadable, or
does not parse, is an unknown value: None, or an entry not counted; a
directory that cannot be listed counts no devices, so every index is
refused. A probe raises only DeviceUnavailable. A refusal names the host
path of the node (such as /dev/kfd), never the root-joined one.

The paths are unverified until P17.12 measures them on a GPU host. The ROCm
runtime that the P17.1 spike read takes its KFD topology from
/sys/devices/virtual/kfd/kfd/topology (ROCm/rocm-systems rocm-7.2.1,
projects/rocr-runtime/libhsakmt/src/topology.c:56; plans/spikes/
p17-frameworks.md, Results 1); that /sys/class/kfd/kfd leads there, that a
node's gpu_id is 0 for a CPU node, that sys/module/amdgpu/version holds the
driver version, and the NVIDIA procfs layout (the control node nvidiactl,
one entry per GPU under proc/driver/nvidia/gpus, the driver version on the
first line of proc/driver/nvidia/version) were not read from source.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from lassi.core.devices import DeviceUnavailable, HostFacts, register_probe
from lassi.executors import native

# The host root every default probe reads under.
HOST_ROOT = Path("/")
# MemTotal in proc/meminfo, in kB.
_MEM_TOTAL = re.compile(r"^MemTotal:\s*([0-9]+)\s*kB\s*$", re.MULTILINE)
# A dotted version number, such as the driver version on the first line of proc/driver/nvidia/version.
_DOTTED_VERSION = re.compile(r"\d+(\.\d+)+")
# The KFD topology nodes, one directory per CPU or GPU node, under the root.
_KFD_NODES = ("sys", "class", "kfd", "kfd", "topology", "nodes")
_AMDGPU_VERSION = ("sys", "module", "amdgpu", "version")
_NVIDIA_GPUS = ("proc", "driver", "nvidia", "gpus")
_NVIDIA_VERSION = ("proc", "driver", "nvidia", "version")


def _accessible(path: Path) -> bool:
    """Return True when this user may read and write `path`, asked with access(2), which opens nothing."""
    return os.access(path, os.R_OK | os.W_OK)


def _read_text(path: Path) -> str | None:
    """Return the text of a procfs or sysfs file, or None when it cannot be read."""
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None


def _entries(path: Path) -> list[Path]:
    """Return the entries of a directory, sorted, or [] when it cannot be listed."""
    try:
        with os.scandir(path) as found:
            return sorted(Path(entry.path) for entry in found)
    except OSError:
        return []


def _require_node(root: Path, node: str, access: Callable[[Path], bool]) -> None:
    """Raise DeviceUnavailable unless <root>/dev/<node> exists (lstat) and `access` allows read and write.

    `access` is asked once, and only about a node that exists.
    """
    path = root / "dev" / node
    if not os.path.lexists(path):
        raise DeviceUnavailable(f"/dev/{node} does not exist on this host")
    if not access(path):
        raise DeviceUnavailable(
            f"/dev/{node} exists but this user may not read and write it; the GPU is not usable here"
        )


@dataclass(frozen=True, kw_only=True)
class CpuProbe:
    """The cpu probe: the CPU count, the model the native executor names, and the memory; it never refuses."""

    root: Path = HOST_ROOT
    kind: str = field(default="cpu", init=False)

    def probe(self) -> HostFacts:
        """Return os.cpu_count(), the model from proc/cpuinfo, and MemTotal from proc/meminfo in bytes.

        The model is the native executor's reading of the same file
        (lassi.executors.native), None when it gives none; memory_bytes is
        MemTotal (kB) times 1024, None when the line is missing or not an
        integer. driver and runtime are None.
        """
        model = native._cpu_model(self.root / "proc" / "cpuinfo")
        meminfo = _read_text(self.root / "proc" / "meminfo") or ""
        match = _MEM_TOTAL.search(meminfo)
        memory = int(match.group(1)) * 1024 if match else None
        return HostFacts(count=os.cpu_count(), name=model or None, memory_bytes=memory, driver=None, runtime=None)


@dataclass(frozen=True, kw_only=True)
class RocmProbe:
    """The rocm probe: /dev/kfd must exist and allow read and write; then the KFD GPU nodes and the driver."""

    root: Path = HOST_ROOT
    access: Callable[[Path], bool] = _accessible
    kind: str = field(default="rocm", init=False)

    def probe(self) -> HostFacts:
        """Return the count of KFD topology nodes whose gpu_id is a nonzero integer, and the amdgpu version.

        Refuses, before any sysfs read, when dev/kfd does not exist or
        access denies it. name and memory_bytes are None until P17.12.
        """
        _require_node(self.root, "kfd", self.access)
        count = sum(1 for node in _entries(self.root.joinpath(*_KFD_NODES)) if _is_gpu_node(node))
        version = _read_text(self.root.joinpath(*_AMDGPU_VERSION))
        driver = version.strip() if version and version.strip() else None
        return HostFacts(count=count, name=None, memory_bytes=None, driver=driver, runtime=None)


def _is_gpu_node(node: Path) -> bool:
    """Return True when a KFD topology node's gpu_id reads as a nonzero integer (a CPU node's is 0)."""
    text = (_read_text(node / "gpu_id") or "").strip()
    return text.isascii() and text.isdigit() and int(text) != 0


@dataclass(frozen=True, kw_only=True)
class CudaProbe:
    """The cuda probe: /dev/nvidiactl must exist and allow read and write; then the GPU count and the driver."""

    root: Path = HOST_ROOT
    access: Callable[[Path], bool] = _accessible
    kind: str = field(default="cuda", init=False)

    def probe(self) -> HostFacts:
        """Return the number of entries under proc/driver/nvidia/gpus and the driver version.

        The entries are counted, never parsed (they are named by PCI bus
        id). The driver is the first dotted number on the first line of
        proc/driver/nvidia/version, None without one. Refuses, before any
        procfs read, when dev/nvidiactl does not exist or access denies it.
        name and memory_bytes are None until P17.12.
        """
        _require_node(self.root, "nvidiactl", self.access)
        count = len(_entries(self.root.joinpath(*_NVIDIA_GPUS)))
        lines = (_read_text(self.root.joinpath(*_NVIDIA_VERSION)) or "").splitlines()
        match = _DOTTED_VERSION.search(lines[0]) if lines else None
        driver = match.group(0) if match else None
        return HostFacts(count=count, name=None, memory_bytes=None, driver=driver, runtime=None)


for _probe in (CpuProbe(), RocmProbe(), CudaProbe()):
    register_probe(_probe)
