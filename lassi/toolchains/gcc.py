"""The host g++ toolchain adapter, registered as Toolchain "gcc-native" (bible Component Interfaces, Toolchain).

GccNative compiles C++ for the host CPU with one g++ command carrying the
native row's flags (bible Execution Backends), in this order:

    g++ -O3 -fopenmp -o main <sources>

The sources are the built files ending in .cpp, .cc, or .cxx, in sorted
order; headers and harness files are written beside them and never compiled
as sources. A .c file is not a source here, since g++ would compile it as
C++. The compiler is the build host's own g++, pinned by version and path in
toolchains/gcc.pin (PIN "gcc"): the class declares no PIN_BIN, so the stage
runner takes the pin's EXPECT_VERSION-checked EXECUTABLE as the compiler
instead of a path under the toolchains root (lassi.core.runner
build_toolchain). It declares `diagnostics` and `emits_warnings`, and no
offload capability: OpenMP here runs on the host.

parse_diagnostics reads GCC's stderr, with the patterns shared with the
other adapters (lassi.toolchains._stderr) and one of its own:

- `<file>:<line>:<column>: <severity>: <message>` is one Diagnostic, with
  severity error, warning, or note as printed and `fatal error` read as an
  error; a trailing ` [<option>]` whose option starts with "-" (such as
  `[-Wunused-variable]` or `[-fpermissive]`) becomes the code. File and line
  are kept as printed. The column is GCC's as printed and is kept only for
  a built file (fold_built_column); for a system header, a harness file, or
  a parse with no files it is None. GCC counts display columns by default,
  so on a line holding a tab or a character outside ASCII the column is not
  a character index.
- A driver line, `<driver>: <severity>: <message>` from g++, gcc, cc1plus,
  or cc1 with an optional version suffix (such as "g++-12: fatal error:
  Killed signal terminated program cc1plus"), has no place; it is tried
  before the GCC style pattern, since its message may quote a name the
  model chose.
- The linker's undefined reference and the collect2 line are errors, read
  as the shared linker patterns read them (a place only as a built file).
- Nothing comes from the context lines ("In file included from ...", its
  "from ..." continuations, "<file>: In function ...", "At global scope",
  "In instantiation of ...", "... required from here"), from the source
  echo, caret, and fix-it lines, or from "compilation terminated.".
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence

from lassi.core.record import Diagnostic
from lassi.core.registry import register
from lassi.toolchains._base import OUTPUT, CommandRunner, CompilerToolchain
from lassi.toolchains._stderr import (
    COLLECT2,
    GCC,
    NO_FILES,
    UNDEFINED_REFERENCE,
    LinePattern,
    compile_diagnostic,
    fold_built_column,
    parse_stderr,
)

# The native row's flags (bible Execution Backends), in order; toolchains/gcc.pin records the same FLAGS.
NATIVE_FLAGS = ("-O3", "-fopenmp")

# A GCC driver or compiler proper line with no place, for example (SYNTHETIC, in GCC's format)
#   g++-12: fatal error: Killed signal terminated program cc1plus
# The name may carry a version suffix (g++-12) and a target prefix (x86_64-linux-gnu-g++-12). A GCC style line on a
# built file reads "<file>:<line>:<column>: ...", so a file named like a driver never matches this pattern.
_DRIVER = re.compile(
    r"(?:[A-Za-z0-9_.]+-)*(?:g\+\+|gcc|cc1plus|cc1)(?:-[0-9]+(?:\.[0-9]+)*)?: "
    r"(?P<severity>fatal error|error|warning|note): (?P<message>.*)"
)
_DRIVER_SEVERITY = {"fatal error": "error", "error": "error", "warning": "warning", "note": "note"}


def _driver(match: re.Match[str]) -> Diagnostic:
    """Return the Diagnostic for a driver line, which names no file; a fatal error is an error."""
    return compile_diagnostic(_DRIVER_SEVERITY[match["severity"]], match["message"])


# Tried in this order on each line: the driver pattern before the GCC style one (see _DRIVER), and the linker
# pattern last because it is the loosest.
_PATTERNS = (
    LinePattern(_DRIVER, _driver),
    dataclasses.replace(GCC, fold=fold_built_column),
    COLLECT2,
    UNDEFINED_REFERENCE,
)


def parse_diagnostics(stderr: str, files: Mapping[str, str] = NO_FILES) -> list[Diagnostic]:
    """Return the compile-stage Diagnostics in g++'s `stderr`, in order (see the module docstring).

    `files` (relative path -> text) are the files built: a GCC style
    diagnostic keeps its column only on one of them, and a linker place is
    kept only as one of them. Lines that match no pattern are skipped.
    """
    return parse_stderr(stderr, files, _PATTERNS)


@register("Toolchain", "gcc-native")
class GccNative(CompilerToolchain):
    """Compiles C++ for the host CPU with the pinned host g++ and the native row's flags."""

    name = "gcc-native"
    capabilities = frozenset({"diagnostics", "emits_warnings"})
    SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx")
    # The pinned host g++: toolchains/gcc.pin, whose EXECUTABLE is an absolute path on the build host. No PIN_BIN,
    # since the compiler is not installed under the toolchains root.
    PIN = "gcc"

    def __init__(
        self, *, executable: str = "g++", runner: CommandRunner | None = None, timeout_s: float = 600.0
    ) -> None:
        """Keep the g++ executable, the command runner (None means subprocess_runner), and the timeout."""
        super().__init__(executable=executable, runner=runner, timeout_s=timeout_s)

    def command(self, sources: Sequence[str]) -> list[str]:
        """Return the g++ command line: the native flags, -o main, then `sources` as given."""
        return [self.executable, *NATIVE_FLAGS, "-o", OUTPUT, *sources]

    def parse(self, stderr: str, files: Mapping[str, str]) -> list[Diagnostic]:
        """Return the Diagnostics in g++'s `stderr` (see parse_diagnostics)."""
        return parse_diagnostics(stderr, files)
