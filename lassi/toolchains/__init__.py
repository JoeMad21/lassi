"""Toolchains: the Toolchain components (bible Component Interfaces, Toolchain row and contract rules).

A toolchain turns files into an artifact plus diagnostics parsed into
Diagnostic records (severity, code, file, line, column, message, stage). The
compiler's raw stderr is kept as an attachment, the file STDERR_ATTACHMENT in
the workdir, and never consumed downstream.

Importing this package registers two presets in
lassi.core.registry.DEFAULT_REGISTRY under the interface "Toolchain":

- "nvcc-sm80" (nvcc.NvccSm80): CUDA with nvcc for sm_80.
- "nvcpp-cc80" (nvcpp.NvcppCc80): OpenMP offload with nvc++ for cc80.

Toolchain bindings carry no config. Each preset declares its pin as class
attributes: PIN, the pin file stem (toolchains/<PIN>.pin, read by
lassi.toolchains.pins), and PIN_BIN, the compiler's path under the pin's
install prefix, which may name {COMPILER_SUBDIR} from the pin file. The
stage runner (lassi.core.runner) builds each one with the pinned compiler
and a clean environment, factory(executable=<toolchains root>/<PREFIX_NAME>/
<PIN_BIN>, runner=EnvRunner(env)), and records the pin and that path (Agent
Rule 10); the toolchains root is $LASSI_TOOLCHAINS. A bare factory() finds
"nvcc" or "nvc++" on PATH, which is for tests and local checks only. Other
keyword settings: `runner` (a CommandRunner; None means subprocess_runner)
and `timeout_s` (default 600.0).

Tests inject a fake CommandRunner, so no compiler runs in them. The runner
contract (CommandResult, CommandRunner, subprocess_runner, EnvRunner) and
STDERR_ATTACHMENT are defined in lassi.toolchains._base and exported here.
"""

from __future__ import annotations

from lassi.toolchains import nvcc, nvcpp
from lassi.toolchains._base import STDERR_ATTACHMENT, CommandResult, CommandRunner, EnvRunner, subprocess_runner
from lassi.toolchains.nvcc import NvccSm80, NvccToolchain
from lassi.toolchains.nvcpp import NvcppCc80, NvcppToolchain

__all__ = [
    "STDERR_ATTACHMENT",
    "CommandResult",
    "CommandRunner",
    "EnvRunner",
    "NvccSm80",
    "NvccToolchain",
    "NvcppCc80",
    "NvcppToolchain",
    "nvcc",
    "nvcpp",
    "subprocess_runner",
]
