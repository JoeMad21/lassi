"""Toolchains: the Toolchain components (bible Component Interfaces, Toolchain row and contract rules).

A toolchain turns files into an artifact plus diagnostics parsed into
Diagnostic records (severity, code, file, line, column, message, stage). The
compiler's raw stderr is kept as an attachment, the file STDERR_ATTACHMENT in
the workdir, and never consumed downstream.

Importing this package registers five presets in
lassi.core.registry.DEFAULT_REGISTRY under the interface "Toolchain":

- "nvcc-sm80" (nvcc.NvccSm80): CUDA with nvcc for sm_80.
- "nvcpp-cc80" (nvcpp.NvcppCc80): OpenMP offload with nvc++ for cc80.
- "nvcpp-multicore" (nvcpp.NvcppMulticore): OpenMP target regions built
  with nvc++ -mp=multicore to run on the host CPU, the Harness Contract's
  proxy; never in faithful recipes, never a source of runtime numbers.
- "gcc-native" (gcc.GccNative): C++ for the host CPU with the host g++ and
  the native row's flags (-O3 -fopenmp).
- "ttmetal-host" (ttmetal_build.TtMetalHost): a TT host program with the
  host clang++-20 against the pinned tt-metal tree, with the pinned build's
  flags; kernel sources are placed for the kernel JIT, never compiled.

Toolchain bindings carry no config. Each preset declares its pin as class
attributes: PIN, the pin file stem (toolchains/<PIN>.pin, read by
lassi.toolchains.pins), and PIN_BIN, the compiler's path under the pin's
install prefix, which may name {COMPILER_SUBDIR} from the pin file. The
stage runner (lassi.core.runner) builds each one with the pinned compiler
and a clean environment, factory(executable=<toolchains root>/<PREFIX_NAME>/
<PIN_BIN>, runner=SandboxedCompileRunner(...)), so every compile runs in
the sandbox (lassi.executors.sandbox, P0.20), and records the pin and that
path (Agent Rule 10); the toolchains root is $LASSI_TOOLCHAINS. A preset
with PIN and no PIN_BIN (gcc-native, ttmetal-host) uses a host compiler the
project does not install: the executable is the pin's EXECUTABLE, an
absolute path, as given. One that also defines check_tree (ttmetal-host)
builds against the pin's installed tree, which the runner checks with it
and passes as factory(..., tree=<resolved toolchains root>/<PREFIX_NAME>).
A bare
factory() finds "nvcc", "nvc++", or "g++" on PATH, which is for tests and
local checks only; ttmetal-host always needs a tree. Other
keyword settings: `runner` (a CommandRunner; None means subprocess_runner)
and `timeout_s` (default 600.0).

Tests inject a fake CommandRunner, so no compiler runs in them. The runner
contract (CommandResult, CommandRunner, subprocess_runner, EnvRunner) and
STDERR_ATTACHMENT are defined in lassi.toolchains._base and exported here,
with capped_runner, the sandbox's default runner for programs, which caps
stdout and stderr at OUTPUT_CAP_BYTES, and CappedRunner, the same at a cap
of its own, which the compile sandbox uses far above any compiler output
(subprocess_runner and EnvRunner keep all of it).
"""

from __future__ import annotations

from lassi.toolchains import gcc, nvcc, nvcpp, ttmetal_build
from lassi.toolchains._base import (
    STDERR_ATTACHMENT,
    CappedRunner,
    CommandResult,
    CommandRunner,
    EnvRunner,
    capped_runner,
    subprocess_runner,
)
from lassi.toolchains.gcc import GccNative
from lassi.toolchains.nvcc import NvccSm80, NvccToolchain
from lassi.toolchains.nvcpp import NvcppCc80, NvcppToolchain
from lassi.toolchains.ttmetal_build import TtMetalHost

__all__ = [
    "STDERR_ATTACHMENT",
    "CappedRunner",
    "CommandResult",
    "CommandRunner",
    "EnvRunner",
    "GccNative",
    "NvccSm80",
    "NvccToolchain",
    "NvcppCc80",
    "NvcppToolchain",
    "TtMetalHost",
    "capped_runner",
    "gcc",
    "nvcc",
    "nvcpp",
    "subprocess_runner",
    "ttmetal_build",
]
