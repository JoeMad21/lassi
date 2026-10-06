"""The gpu executor, registered as Executor "gpu" (task P17.5; bible Execution Backends, gpu rows; Sandbox).

It runs a built GPU artifact (CUDA, HIP, or an OpenMP offload program) with
its inputs as arguments, only through lassi.executors.sandbox (Agent Rule
6), under the native executor's rules: the workdir inside the runs root,
the hidden roots, the harness and toolchains root read-only, and the output
files (lassi.executors.native). Its recipe section names the GPU with a
device section (task P17.2), `executor: {kind: gpu, device: {kind: rocm,
indices: [0]}}`, or one per language; kind cuda means NVIDIA and kind rocm
AMD, so there is no vendor key, and kind cpu is refused.

Each run's SandboxSpec carries a DeviceExposure: the vendor table's fixed
nodes and each listed index's node, each bound at its own path in the
private /dev, and the table's /sys directories, bound read-only. No other
run exposes a node: native, ttsim, and compile runs keep SETUP_SCRIPT and
its six nodes. The program's environment is the native executor's (PATH,
LANG, TMPDIR, OMP_NUM_THREADS) plus the table's visibility variables, set
to the in-sandbox indices 0,...,n-1, and its settings. With only the
listed render nodes bound, the ROCm thunk skips a GPU whose render node
fails to open with ENOENT or EPERM rather than failing (topology.c:715-718;
fmm.c:2358-2365), so the runtime numbers the listed GPUs from 0 in host
order and a host index above n-1 would select nothing; the same is assumed,
unverified, for CUDA, whose driver is closed source. CUDA_DEVICE_ORDER=
PCI_BUS_ID makes CUDA's order the PCI order the index mapping uses.

The tables are unverified until P17.12 measures a GPU host. Sources:

- NVIDIA (NVIDIA/nvidia-modprobe 615.71.09): /dev/nvidiactl
  (modprobe-utils/nvidia-modprobe-utils.h:40), /dev/nvidia-uvm and
  /dev/nvidia-uvm-tools (nvidia-modprobe-utils.c:61-62), /dev/nvidia<minor>
  per index (nvidia-modprobe-utils.h:39). Not bound: /dev/nvidia-modeset
  (display), the MIG nodes under /dev/nvidia-caps, and the nvswitch and
  nvlink nodes (nvidia-modprobe-utils.h:41, :44-49). No /sys directory is
  bound until P17.12 measures what CUDA reads; /proc/driver/nvidia is a
  procfs entry, which the sandbox's own /proc shows (inferred). The
  visibility variable CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER come from
  the CUDA documentation, not from source.
- AMD (ROCm/rocm-systems rocm-7.2.1, projects/rocr-runtime/libhsakmt/src
  and runtime/hsa-runtime, projects/clr): /dev/kfd (openclose.c:48,
  opened at :192), /dev/dri/renderD<minor> per index (fmm.c:2357-2358),
  and the KFD topology /sys/devices/virtual/kfd/kfd/topology (topology.c:56,
  :63-65); not bound: /dev/udmabuf, opened only with HSA_USE_UDMABUF
  (openclose.c:49, :207-211). ROCR_VISIBLE_DEVICES is read at
  core/util/flag.h:109-110 and HIP_VISIBLE_DEVICES declared at
  rocclr/utils/flags.hpp:154.

Construction reads the host's GPU inventory under `root` with
lassi.executors.devices (gpu_inventory, gpu_driver, check_gpu_nodes): file
metadata and procfs or sysfs text only. Nothing in this module opens a
device node or starts a process (OQ-002); the sandbox's setup binds the
nodes without opening them, and only the program inside may open them. A
fixed or listed node this host lacks or refuses this user, or an index
the inventory lacks or that has no node, raises DeviceUnavailable, which
the runner reports before any directory exists. device() formats what
construction read.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from lassi.core.capabilities import TAKES_DEVICE
from lassi.core.devices import DeviceSpec, parse_device
from lassi.core.interfaces import Limits, RunResult
from lassi.core.registry import register
from lassi.executors import devices
from lassi.executors.native import (
    _TOOLCHAINS_VARIABLE,
    _absolute_variable,
    _checked_workdir,
    _environment_roots,
    _printable,
    _program_environment,
    _regular_files,
    _resolved,
)
from lassi.executors.sandbox import DeviceBind, DeviceExposure, Sandbox, SandboxSpec, SandboxUnavailableError
from lassi.executors.workdir import runs_root


@dataclass(frozen=True)
class VendorTable:
    """What a run on one vendor's GPUs is given (unverified until P17.12; module docstring for the sources).

    `kind` is the device kind; `fixed_nodes` the host nodes every run binds;
    `sys_dirs` the /sys directories bound read-only at their own paths;
    `visibility` the variables set to the in-sandbox indices; `settings`
    the other variables set.
    """

    kind: str
    fixed_nodes: tuple[str, ...]
    sys_dirs: tuple[str, ...]
    visibility: tuple[str, ...]
    settings: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


NVIDIA_TABLE = VendorTable(
    kind="cuda",
    fixed_nodes=("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"),
    sys_dirs=(),
    visibility=("CUDA_VISIBLE_DEVICES",),
    settings=MappingProxyType({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}),
)
AMD_TABLE = VendorTable(
    kind="rocm",
    fixed_nodes=("/dev/kfd",),
    sys_dirs=("/sys/devices/virtual/kfd/kfd/topology",),
    visibility=("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"),
)
# The vendor table of each GPU kind.
TABLES: Mapping[str, VendorTable] = MappingProxyType({"cuda": NVIDIA_TABLE, "rocm": AMD_TABLE})
# The kernel module whose version device() names, by kind.
_DRIVER_MODULES = {"cuda": "nvidia", "rocm": "amdgpu"}


def _printable_line(text: str) -> str:
    """Return `text` with its whitespace collapsed and every other character outside printable ASCII escaped."""
    return "".join(_printable(character) for character in " ".join(text.split()))


@register("Executor", "gpu")
class GpuExecutor:
    """Runs a built GPU artifact with its inputs as arguments, only inside the sandbox, on the GPUs it names.

    `device` is the recipe's device section (kind cuda or rocm, and
    indices); the config key `harness` is the native executor's. The keyword
    settings `hidden_roots`, `toolchains`, and `sandbox` are the native
    executor's; `root` is the host root the inventory is read under, and
    `access` the read-and-write check asked about each node (access(2),
    which opens nothing), as the P17.2 probes take them.
    """

    name = "gpu"
    capabilities = frozenset({"runs_code", "sandboxed", TAKES_DEVICE})
    config_keys = frozenset({"harness"})

    def __init__(
        self,
        *,
        device: Mapping[str, object],
        harness: str | None = None,
        hidden_roots: Sequence[str] | None = None,
        toolchains: str | None = None,
        sandbox: Sandbox | None = None,
        root: Path = devices.HOST_ROOT,
        access: Callable[[Path], bool] = devices._accessible,
    ) -> None:
        """Parse the device section, then read and check this host's nodes for it.

        A section that names no GPU (kind cpu, an unknown kind, no indices,
        a repeated index) is a ValueError naming the key, raised before any
        file is read. A fixed or listed node that is missing or refused, or
        an index the inventory lacks or that has no node, raises
        DeviceUnavailable (devices.check_gpu_nodes).
        """
        section = parse_device(device, "device")
        if section.kind not in TABLES:
            raise ValueError(f"device.kind: the gpu executor runs on kind cuda or rocm, not {section.kind}")
        if isinstance(hidden_roots, (str, os.PathLike)):
            raise ValueError(f"hidden_roots must be a sequence of paths, got {hidden_roots!r}")
        self.section: DeviceSpec = section
        self.table: VendorTable = TABLES[section.kind]
        self.harness: Path | None = None if harness is None else Path(harness)
        self.hidden_roots: tuple[Path, ...] | None = (
            None if hidden_roots is None else tuple(Path(item) for item in hidden_roots)
        )
        self.toolchains: Path | None = None if toolchains is None else Path(toolchains)
        self.sandbox: Sandbox = Sandbox() if sandbox is None else sandbox
        host = Path(root)
        inventory = devices.gpu_inventory(section.kind, host)
        self.entries = devices.check_gpu_nodes(self.table.fixed_nodes, inventory, section.indices, host, access)
        self.driver: str | None = devices.gpu_driver(section.kind, host)

    def device(self) -> str:
        """Return "gpu (<kind>): index <i> <name>[, ...]; <module> driver <version>" from what construction read.

        "name not read" and "driver not read" stand for unknown values; the
        line is printable ASCII. It opens no file and starts no process.
        """
        gpus = ", ".join(
            f"index {entry.index} {_printable_line(entry.name or '') or 'name not read'}" for entry in self.entries
        )
        version = _printable_line(self.driver or "") or "not read"
        return f"gpu ({self.section.kind}): {gpus}; {_DRIVER_MODULES[self.section.kind]} driver {version}"

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Run `artifact` with `inputs` in the sandbox, with this executor's GPUs exposed, and return its RunResult.

        The rules are NativeExecutor.run's (the workdir, the hidden roots,
        the refusals before anything runs, output_files, the flags); the
        spec adds the exposure (_exposure) and the environment adds the
        table's variables (_gpu_environment).
        """
        artifact = Path(artifact)
        if not artifact.is_absolute():
            raise ValueError(f"the artifact must be an absolute path, got {str(artifact)!r}")
        if isinstance(inputs, str):
            raise ValueError(f"inputs must be a sequence of strings, got {inputs!r}")
        spec = self._spec(_checked_workdir(artifact), limits)
        before = _regular_files(spec.workdir)
        result = self.sandbox.run(spec, [str(artifact), *inputs], limits)
        after = _regular_files(spec.workdir)
        return RunResult(
            exit_code=result.returncode,
            hang=result.hang,
            stdout=result.stdout,
            stderr=result.stderr,
            output_files={relative: after[relative] for relative in sorted(after) if relative not in before},
            wall_s=result.wall_s,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            workdir_incomplete=result.workdir_incomplete,
        )

    def _spec(self, workdir: Path, limits: Limits) -> SandboxSpec:
        """Return the run's SandboxSpec: the native executor's mounts plus the exposure and the GPU variables."""
        roots = _environment_roots() if self.hidden_roots is None else self.hidden_roots
        if not roots:
            raise SandboxUnavailableError(
                "no hidden root: set $LASSI_SCRATCH or $HOME, or pass hidden_roots; nothing ran"
            )
        toolchains = self.toolchains
        if toolchains is None:
            toolchains = _absolute_variable(_TOOLCHAINS_VARIABLE, "mounted read-only in the sandbox")
        return SandboxSpec(
            workdir=workdir,
            hidden_roots=(*(_resolved(root) for root in roots), runs_root().resolve()),
            harness=None if self.harness is None else _resolved(self.harness),
            toolchains=None if toolchains is None else _resolved(toolchains),
            environment=_gpu_environment(limits, self.table, len(self.entries)),
            devices=_exposure(self.table, self.entries),
        )


def _exposure(table: VendorTable, entries: Sequence[devices.GpuEntry]) -> DeviceExposure:
    """Return the exposure of a run: the fixed nodes, then each entry's node, and the table's /sys directories.

    Each path is bound at its own path inside.
    """
    nodes = [*table.fixed_nodes, *(entry.node for entry in entries if entry.node is not None)]
    return DeviceExposure(
        nodes=tuple(DeviceBind(PurePosixPath(node), PurePosixPath(node)) for node in nodes),
        sys_dirs=tuple(DeviceBind(PurePosixPath(path), PurePosixPath(path)) for path in table.sys_dirs),
    )


def _gpu_environment(limits: Limits, table: VendorTable, count: int) -> dict[str, str]:
    """Return the program's environment: the native executor's, each visibility variable = "0,...,count-1", settings.

    Nothing comes from the caller's environment.
    """
    visible = ",".join(str(index) for index in range(count))
    return {**_program_environment(limits), **dict.fromkeys(table.visibility, visible), **table.settings}
