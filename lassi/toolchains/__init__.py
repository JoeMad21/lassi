"""Toolchains: the Toolchain components (bible Component Interfaces, Toolchain row and contract rules).

A toolchain turns files into an artifact plus diagnostics parsed into
Diagnostic records (severity, code, file, line, column, message, stage). The
compiler's raw stderr is kept as an attachment, the file STDERR_ATTACHMENT in
the workdir, and never consumed downstream.

Importing this package registers two presets in
lassi.core.registry.DEFAULT_REGISTRY under the interface "Toolchain":

- "nvcc-sm80" (nvcc.NvccSm80): CUDA with nvcc for sm_80.
- "nvcpp-cc80" (nvcpp.NvcppCc80): OpenMP offload with nvc++ for cc80.

Toolchain bindings carry no config. The runner builds each one with the
pinned compiler, factory(executable=<$LASSI_TOOLCHAINS/<name>@<pin> path>),
and records that path (Agent Rule 10); a bare factory() finds "nvcc" or
"nvc++" on PATH, which is for tests and local checks only. Other keyword
settings: `runner` (a CommandRunner; None means subprocess_runner) and
`timeout_s` (default 600.0).

Tests inject a fake CommandRunner, so no compiler runs in them. The runner
contract (CommandResult, CommandRunner, subprocess_runner) and
STDERR_ATTACHMENT are defined in lassi.toolchains._base and exported here.
"""

from __future__ import annotations

from lassi.toolchains import nvcc, nvcpp
from lassi.toolchains._base import STDERR_ATTACHMENT, CommandResult, CommandRunner, subprocess_runner
from lassi.toolchains.nvcc import NvccSm80, NvccToolchain
from lassi.toolchains.nvcpp import NvcppCc80, NvcppToolchain

__all__ = [
    "STDERR_ATTACHMENT",
    "CommandResult",
    "CommandRunner",
    "NvccSm80",
    "NvccToolchain",
    "NvcppCc80",
    "NvcppToolchain",
    "nvcc",
    "nvcpp",
    "subprocess_runner",
]
