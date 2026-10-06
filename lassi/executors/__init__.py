"""Executors: the Executor components (bible Component Interfaces, Executor row; Execution Backends).

Importing this package registers the executors in
lassi.core.registry.DEFAULT_REGISTRY:

- "none" (none.NoneExecutor): compile only; it runs nothing.
- "native" (native.NativeExecutor): runs a CPU artifact.
- "ttsim" (ttsim.TtsimExecutor): runs a TT host program on ttsim, the
  pinned Wormhole simulator (task P4.11).

Each executor names the device its programs run on with device(), one
non-empty line of printable ASCII with no leading or trailing blank, which
starts no process (task P4.5): "none (compile only)" for none, the host
CPU with its model for native, and ttsim with its pin and the pinned
tt-metal, as a simulator, for ttsim.

Importing it also registers the host device probes (devices.CpuProbe,
RocmProbe, and CudaProbe; task P17.2) in lassi.core.devices.DEFAULT_PROBES
by kind. They read file metadata and procfs or sysfs text only and never
open a device node.

Every executor that runs generated code runs it only through
lassi.executors.sandbox (Agent Rule 6), whose names are re-exported here:
Sandbox, SandboxSpec, SandboxResult, SandboxUnavailableError,
sandbox_command, and classify. lassi.executors.workdir gives each attempt
its build directory under the runs root.
"""

from lassi.executors import devices, native, none, sandbox, ttsim
from lassi.executors.native import NativeExecutor
from lassi.executors.none import NoneExecutor
from lassi.executors.sandbox import (
    Sandbox,
    SandboxResult,
    SandboxSpec,
    SandboxUnavailableError,
    classify,
    sandbox_command,
)
from lassi.executors.ttsim import TtsimExecutor

__all__ = [
    "NativeExecutor",
    "NoneExecutor",
    "Sandbox",
    "SandboxResult",
    "SandboxSpec",
    "SandboxUnavailableError",
    "TtsimExecutor",
    "classify",
    "devices",
    "native",
    "none",
    "sandbox",
    "sandbox_command",
    "ttsim",
]
