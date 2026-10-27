"""Host device probes, one per device kind (task P17.2; lassi.core.devices, the probe seam).

Device files are read only in lassi/executors and lassi/profilers (Design
Principle 9), so the probes the runner uses by default live here. Importing
lassi.executors registers CpuProbe, RocmProbe, CudaProbe, and
TenstorrentProbe (task P17.38) in lassi.core.devices.DEFAULT_PROBES by kind.

- tenstorrent: the all-digit names in /dev/tenstorrent are the cards'
  numbers, the indices a section names (the by-id links skipped, as UMD
  skips them). A card is usable when its node exists and access(2) allows
  read and write, and its class directory /sys/class/tenstorrent/
  tenstorrent!<n> gives tt_card_type, tt_serial, tt_fw_bundle_ver, and
  tt_arc_fw_ver, the files the tt_silicon executor records; the PCI address
  is device/uevent's PCI_SLOT_NAME. The probe gives each usable card's
  description by number and why each other card is not usable
  (lassi.core.devices HostFacts by_index and refused), and the driver
  /sys/module/tenstorrent/version holds. These are the paths the tt_silicon
  executor reads on the build host's cards (plans/spikes/
  tt-silicon-executor.md).

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

The gpu executor's index-to-node mapping (task P17.5; GpuEntry,
gpu_inventory, gpu_driver, check_gpu_nodes) reads the same way, under the
same root, and opens no node; it is unverified until P17.12. Sources:
ROCm/rocm-systems rocm-7.2.1 projects/rocr-runtime/libhsakmt/src and
NVIDIA/open-gpu-kernel-modules 615.71.09 kernel-open/nvidia/nv-procfs.c.

- rocm: the KFD topology nodes under sys/class/kfd/kfd/topology/nodes
  (the directory RocmProbe counts), taken in numeric order of their
  directory names (digits only, so "10" comes after "9"); index i is the
  i-th node whose gpu_id is a nonzero integer, as RocmProbe counts. Its
  node is /dev/dri/renderD<m>, m from the properties line
  "drm_render_minor <m>" (topology.c:704-705; the thunk opens that path,
  fmm.c:2357-2358); a missing or zero value gives no node. Its name is
  gfx<major><minor><stepping>, minor and stepping in hex, decoded from the
  properties line gfx_target_version (read at topology.c:1223-1224) as
  major = v // 10000, minor = v // 100 % 100, stepping = v % 100; the
  decoding is not read from source (90402 gives gfx942).
- cuda: the entries of proc/driver/nvidia/gpus, each named by its PCI bus
  id (nv-procfs.c:1451-1453), taken in name order, which is PCI order;
  index i is the i-th entry. Its node is /dev/nvidia<m>, m from the line
  "Device Minor: <m>" of the entry's information file (nv-procfs.c:161);
  its name is that file's "Model:" line (nv-procfs.c:128).

An unreadable value is None; an index without a node is still an entry,
so the indices count as the probes count them.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
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
        driver = _amdgpu_driver(self.root)
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
        driver = _nvidia_driver(self.root)
        return HostFacts(count=count, name=None, memory_bytes=None, driver=driver, runtime=None)


def _amdgpu_driver(root: Path) -> str | None:
    """Return the stripped text of sys/module/amdgpu/version under `root`, or None when it is missing or empty."""
    version = _read_text(root.joinpath(*_AMDGPU_VERSION))
    return version.strip() if version and version.strip() else None


def _nvidia_driver(root: Path) -> str | None:
    """Return the first dotted number on the first line of proc/driver/nvidia/version under `root`, or None."""
    lines = (_read_text(root.joinpath(*_NVIDIA_VERSION)) or "").splitlines()
    match = _DOTTED_VERSION.search(lines[0]) if lines else None
    return match.group(0) if match else None


@dataclass(frozen=True)
class GpuEntry:
    """One GPU of the host's inventory: its index, its node's host path (None: none named), and its name or None."""

    index: int
    node: str | None
    name: str | None


def gpu_inventory(kind: str, root: Path) -> tuple[GpuEntry, ...]:
    """Return the GPUs of `kind` (rocm or cuda) under `root`, in index order, mapped as the module docstring says.

    Only procfs or sysfs text and directory listings are read; no node is
    opened. A root without the kind's files gives no entry. ValueError for
    another kind.
    """
    if kind == "rocm":
        return _rocm_inventory(root)
    if kind == "cuda":
        return _cuda_inventory(root)
    raise ValueError(f"no GPU inventory for kind {kind!r}; kinds: rocm, cuda")


def gpu_driver(kind: str, root: Path) -> str | None:
    """Return the driver version the kind's probe reads under `root` (amdgpu or nvidia), or None when not read."""
    if kind == "rocm":
        return _amdgpu_driver(root)
    if kind == "cuda":
        return _nvidia_driver(root)
    raise ValueError(f"no GPU driver for kind {kind!r}; kinds: rocm, cuda")


def _fields(path: Path, separator: str) -> dict[str, str]:
    """Return a procfs or sysfs file's lines split once at `separator` as stripped name -> value, first one kept."""
    found: dict[str, str] = {}
    for line in (_read_text(path) or "").splitlines():
        name, sep, value = line.strip().partition(separator)
        if sep:
            found.setdefault(name.strip(), value.strip())
    return found


def _number(text: str | None) -> int | None:
    """Return `text` as a nonnegative integer when it is ASCII digits, else None."""
    return int(text) if text and text.isascii() and text.isdigit() else None


def _rocm_inventory(root: Path) -> tuple[GpuEntry, ...]:
    """Return the KFD GPU nodes in numeric node order, each with its render node and gfx name."""
    nodes = [node for node in _entries(root.joinpath(*_KFD_NODES)) if node.name.isascii() and node.name.isdigit()]
    found: list[GpuEntry] = []
    for node in sorted(nodes, key=lambda path: int(path.name)):
        if not _is_gpu_node(node):
            continue
        properties = _fields(node / "properties", " ")
        minor = _number(properties.get("drm_render_minor"))
        target = _number(properties.get("gfx_target_version"))
        name = f"gfx{target // 10000}{target // 100 % 100:x}{target % 100:x}" if target else None
        found.append(GpuEntry(index=len(found), node=f"/dev/dri/renderD{minor}" if minor else None, name=name))
    return tuple(found)


def _cuda_inventory(root: Path) -> tuple[GpuEntry, ...]:
    """Return the NVIDIA GPUs in bus id order, each with its /dev/nvidia<minor> node and model."""
    found: list[GpuEntry] = []
    for entry in _entries(root.joinpath(*_NVIDIA_GPUS)):
        information = _fields(entry / "information", ":")
        minor = _number(information.get("Device Minor"))
        node = None if minor is None else f"/dev/nvidia{minor}"
        found.append(GpuEntry(index=len(found), node=node, name=information.get("Model") or None))
    return tuple(found)


def _dev_relative(node: str) -> str:
    """Return a node path relative to /dev: "/dev/kfd" and "kfd" both give "kfd"."""
    return node[len("/dev/") :] if node.startswith("/dev/") else node


def check_gpu_nodes(
    fixed: Sequence[str],
    entries: Sequence[GpuEntry],
    indices: Sequence[int],
    root: Path,
    access: Callable[[Path], bool],
) -> tuple[GpuEntry, ...]:
    """Return the entries of `indices`, in that order, after checking every node a run of them would bind.

    Each fixed node, then each listed index's node, must exist under `root`
    (lstat) and allow read and write (`access`, asked only about a node that
    exists); an index absent from `entries`, or one without a node, is
    refused too. Every refusal is DeviceUnavailable naming the host path
    (/dev/...) or "index N", never the root-joined path. Nothing is opened.
    """
    for node in fixed:
        _require_node(root, _dev_relative(node), access)
    by_index = {entry.index: entry for entry in entries}
    chosen: list[GpuEntry] = []
    for index in indices:
        entry = by_index.get(index)
        if entry is None:
            raise DeviceUnavailable(f"index {index} is not among this host's {len(entries)} GPU(s)")
        if entry.node is None:
            raise DeviceUnavailable(f"index {index} names no device node on this host")
        _require_node(root, _dev_relative(entry.node), access)
        chosen.append(entry)
    return tuple(chosen)


# The Tenstorrent paths under the root (task P17.38), as the tt_silicon executor reads them: the numbered card nodes,
# each card's class directory, and the driver's version.
_TT_NODES = ("dev", "tenstorrent")
_TT_CLASS = ("sys", "class", "tenstorrent")
_TT_DRIVER = ("sys", "module", "tenstorrent", "version")
# The class attributes every card must give, as the tt_silicon executor requires them; the record names each.
_TT_ATTRIBUTES = ("tt_card_type", "tt_serial", "tt_fw_bundle_ver", "tt_arc_fw_ver")


@dataclass(frozen=True)
class TenstorrentCard:
    """One Tenstorrent card as its class directory gives it: number, type, serial, firmware, and PCI address."""

    number: int
    card_type: str
    serial: str
    bundle: str
    arc: str
    address: str | None

    def describe(self) -> str:
        """Return the card in one line: type, number, serial, firmware bundle, and ARC firmware."""
        return (
            f"Tenstorrent {self.card_type} card {self.number}: serial {self.serial}, firmware bundle {self.bundle}, "
            f"ARC firmware {self.arc}"
        )


@dataclass(frozen=True)
class TenstorrentInventory:
    """The host's numbered card nodes, the usable cards by number, why each other one is not, and the driver."""

    numbers: tuple[int, ...]
    cards: dict[int, TenstorrentCard]
    refused: dict[int, str]
    driver: str | None


def _clean(text: str | None) -> str:
    """Return sysfs text with its whitespace collapsed, printable ASCII only ("?" for anything else)."""
    collapsed = " ".join((text or "").split())
    return "".join(char if " " <= char <= "~" else "?" for char in collapsed)


def _tt_card(root: Path, number: int, access: Callable[[Path], bool]) -> TenstorrentCard | str:
    """Return card `number` read from its node's metadata and class directory, or why it cannot be used.

    The node must exist (lstat) and allow read and write (`access`, which
    opens nothing); the class directory must give every _TT_ATTRIBUTES file,
    non-empty. The PCI address is the PCI_SLOT_NAME of device/uevent, None
    when not read.
    """
    node = root.joinpath(*_TT_NODES, str(number))
    if not os.path.lexists(node):
        return f"/dev/tenstorrent/{number} does not exist on this host"
    if not access(node):
        return f"/dev/tenstorrent/{number} exists, but this user may not read and write it"
    directory = root.joinpath(*_TT_CLASS, f"tenstorrent!{number}")
    values = {name: _clean(_read_text(directory / name)) for name in _TT_ATTRIBUTES}
    missing = [name for name, value in values.items() if not value]
    if missing:
        return f"/sys/class/tenstorrent/tenstorrent!{number} gives no {', '.join(missing)}"
    address = _fields(directory / "device" / "uevent", "=").get("PCI_SLOT_NAME") or None
    return TenstorrentCard(number, values["tt_card_type"], values["tt_serial"], values["tt_fw_bundle_ver"],
                           values["tt_arc_fw_ver"], _clean(address) or None)


def tenstorrent_inventory(root: Path, access: Callable[[Path], bool]) -> TenstorrentInventory:
    """Return the host's Tenstorrent cards under `root`, read from file metadata and sysfs text only.

    The numbered nodes are the all-digit names in /dev/tenstorrent (the by-id
    links skipped), in numeric order. DeviceUnavailable, naming the host
    path, when /dev/tenstorrent is missing or holds no numbered node. No node
    is opened and no process starts (OQ-002).
    """
    directory = root.joinpath(*_TT_NODES)
    if not directory.is_dir():
        raise DeviceUnavailable("/dev/tenstorrent does not exist on this host, so there is no Tenstorrent card")
    numbers = sorted(int(path.name) for path in _entries(directory) if path.name.isascii() and path.name.isdigit())
    if not numbers:
        raise DeviceUnavailable("/dev/tenstorrent holds no numbered card node on this host")
    cards: dict[int, TenstorrentCard] = {}
    refused: dict[int, str] = {}
    for number in numbers:
        found = _tt_card(root, number, access)
        if isinstance(found, str):
            refused[number] = found
        else:
            cards[number] = found
    driver = _clean(_read_text(root.joinpath(*_TT_DRIVER))) or None
    return TenstorrentInventory(numbers=tuple(numbers), cards=cards, refused=refused, driver=driver)


@dataclass(frozen=True, kw_only=True)
class TenstorrentProbe:
    """The tenstorrent probe (task P17.38): each card by its /dev/tenstorrent number, from sysfs and metadata only."""

    root: Path = HOST_ROOT
    access: Callable[[Path], bool] = _accessible
    kind: str = field(default="tenstorrent", init=False)

    def probe(self) -> HostFacts:
        """Return the count of numbered nodes, each usable card's description, the refused ones, and the driver.

        name, memory_bytes, and runtime are None: the card's memory is not
        read from sysfs, and the runtime is the framework's own build.
        """
        inventory = tenstorrent_inventory(self.root, self.access)
        return HostFacts(
            count=len(inventory.numbers), name=None, memory_bytes=None, driver=inventory.driver, runtime=None,
            by_index={number: card.describe() for number, card in inventory.cards.items()},
            refused=dict(inventory.refused),
        )


for _probe in (CpuProbe(), RocmProbe(), CudaProbe(), TenstorrentProbe()):
    register_probe(_probe)
