"""The HIP toolchain adapter, registered as Toolchain "hipcc-gfx942" (bible Component Interfaces, Toolchain).

HipccGfx942 builds HIP sources for the gfx942 target with one hipcc call,
which compiles the host code and the gfx942 device code and links the host
program:

    <hipcc> --offload-arch=gfx942 -Wall -O3 -o main <sources>

Nothing runs the program here: a recipe binds the toolchain under the
language hip and pairs it with the executor none, the bible's compile-only
tier (Execution Backends), as nvcc-sm80 builds for sm_80 without a GPU.
hipcc given no target runs rocm_agent_enumerator, a device-query program,
to choose one, for --version alone too, before it prints its version
(ROCm/llvm-project tag therock-7.12, amd/hipcc/src/hipBin_amd.h:475,
:568-581, :779-792, :914-916; read from source, not run). So every command
names the target: each build, and the --version check, which runs
`<hipcc> --offload-arch=gfx942 --version` (VERSION_ARGS, which
lassi.core.runner build_toolchain reads), and hipcc is never left to start
amdgpu-arch, offload-arch, rocm_agent_enumerator, or rocminfo (OQ-002).
The target is a module constant, OFFLOAD_ARCH, from which HIPCC_FLAGS and
VERSION_ARGS are built; it is never an ARCH attribute or a constructor
setting, so no subclass or capture override builds another target. The
flags are HIPCC_FLAGS [DESIGN]: the target, then -Wall and -O3, carried
from the host -Wall and -O3 of LASSI's nvcc compile flags (bible Source
Papers, LASSI, Compile flags).

The sources are the built files ending in .hip, in sorted order; headers,
harness files, and files with other suffixes are written beside them and
never compiled. Whether this hipcc builds a .cu or .cpp file as HIP was not
measured, so neither is a source.

The compiler is the build host's hipcc, pinned by version and path in
toolchains/hipcc.pin (PIN "hipcc"): the class declares no PIN_BIN, so the
stage runner takes the pin's EXPECT_VERSION-checked EXECUTABLE as the
compiler, as it does for gcc-native (lassi.core.runner build_toolchain).
The pin names no variable: a compile gets only the compile sandbox's PATH,
LANG=C, LC_ALL=C, and private TMPDIR, with no ROCm variable and no
environment script sourced (bible Host Facts). It declares hip (the offload
model, as nvcc-sm80 declares cuda), diagnostics, and emits_warnings.

parse_diagnostics reads the stderr of hipcc's clang with the clang patterns
ttmetal-host uses (lassi.toolchains._stderr CLANG_DRIVER and CLANG_PLACE),
then the shared GNU ld pattern, since which linker the host link uses is
not pinned by a flag:

- A driver line, `<tool>: <severity>: <message>` from clang, clang++, or
  ld.lld with an optional version suffix, has no place, and a fatal error
  is an error. It is tried first, since its message may quote a name the
  model chose.
- `<file>:<line>:<column>: <severity>: <message>` is one Diagnostic, with
  a leading "./" dropped from the file and the column kept only on a built
  file; `fatal error` is an error and a trailing ` [<option>]` is the code.
- A GNU ld "undefined reference to" line is an error, its place kept only
  as a built file (UNDEFINED_REFERENCE).
- Nothing comes from the include chain, the source echo and caret lines,
  or the "N warnings generated when compiling for gfx942." and "... for
  host." summaries. A diagnostic that the device pass and the host pass
  both print parses twice, as stderr shows it.

The patterns are pinned on SYNTHETIC lines until the fixtures captured on
the build host (tests/toolchains/fixtures/hipcc) are read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lassi.core.record import Diagnostic
from lassi.core.registry import register
from lassi.toolchains._base import OUTPUT, CommandRunner, CompilerToolchain
from lassi.toolchains._stderr import CLANG_DRIVER, CLANG_PLACE, NO_FILES, UNDEFINED_REFERENCE, parse_stderr

# The one target every build names; toolchains/hipcc.pin records the same FLAGS.
OFFLOAD_ARCH = "gfx942"
HIPCC_FLAGS = (f"--offload-arch={OFFLOAD_ARCH}", "-Wall", "-O3")

# Tried in this order on each line: the driver pattern before the place pattern, and the linker pattern last
# because it is the loosest.
_PATTERNS = (CLANG_DRIVER, CLANG_PLACE, UNDEFINED_REFERENCE)


def parse_diagnostics(stderr: str, files: Mapping[str, str] = NO_FILES) -> list[Diagnostic]:
    """Return the compile-stage Diagnostics in hipcc's `stderr`, in order (see the module docstring).

    `files` (relative path -> text) are the files built: a place line keeps
    its column only on one of them, and a linker place is kept only as one
    of them. Lines that match no pattern are skipped.
    """
    return parse_stderr(stderr, files, _PATTERNS)


@register("Toolchain", "hipcc-gfx942")
class HipccGfx942(CompilerToolchain):
    """Builds HIP sources for gfx942 with the pinned host hipcc; the program is linked and never run here."""

    name = "hipcc-gfx942"
    capabilities = frozenset({"hip", "diagnostics", "emits_warnings"})
    SOURCE_SUFFIXES = (".hip",)
    # The pinned host hipcc: toolchains/hipcc.pin, whose EXECUTABLE is an absolute path on the build host. No
    # PIN_BIN, since the compiler is not installed under the toolchains root.
    PIN = "hipcc"
    # The --version check's arguments after the executable: the target first, so hipcc never runs
    # rocm_agent_enumerator to choose one (see the module docstring).
    VERSION_ARGS = (f"--offload-arch={OFFLOAD_ARCH}", "--version")

    def __init__(
        self, *, executable: str = "hipcc", runner: CommandRunner | None = None, timeout_s: float = 600.0
    ) -> None:
        """Keep the hipcc executable, the command runner (None means subprocess_runner), and the timeout."""
        super().__init__(executable=executable, runner=runner, timeout_s=timeout_s)

    def command(self, sources: Sequence[str]) -> list[str]:
        """Return the hipcc command line: the gfx942 target and flags, -o main, then `sources` as given."""
        return [self.executable, *HIPCC_FLAGS, "-o", OUTPUT, *sources]

    def parse(self, stderr: str, files: Mapping[str, str]) -> list[Diagnostic]:
        """Return the Diagnostics in hipcc's `stderr` (see parse_diagnostics)."""
        return parse_diagnostics(stderr, files)
