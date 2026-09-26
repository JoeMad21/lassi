"""Executors: the Executor components (bible Component Interfaces, Executor row; Execution Backends).

Importing this package registers the executors in
lassi.core.registry.DEFAULT_REGISTRY:

- "none" (none.NoneExecutor): compile only; it runs nothing.
- "native" (native.NativeExecutor): runs a CPU artifact.

Each executor names the device its programs run on with device(), one
non-empty line of printable ASCII with no leading or trailing blank, which
starts no process (task P4.5): "none (compile only)" for none, and the host
CPU with its model for native.

Every executor that runs generated code runs it only through
lassi.executors.sandbox (Agent Rule 6), whose names are re-exported here:
Sandbox, SandboxSpec, SandboxResult, SandboxUnavailableError,
sandbox_command, and classify. lassi.executors.workdir gives each attempt
its build directory under the runs root.
"""

from lassi.executors import native, none, sandbox
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

__all__ = [
    "NativeExecutor",
    "NoneExecutor",
    "Sandbox",
    "SandboxResult",
    "SandboxSpec",
    "SandboxUnavailableError",
    "classify",
    "native",
    "none",
    "sandbox",
    "sandbox_command",
]
