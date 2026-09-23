"""The nvcc toolchain adapter, registered as Toolchain "nvcc-sm80" (bible Component Interfaces, Toolchain).

NvccToolchain compiles CUDA sources with one nvcc command carrying the LASSI
compile flags from the bible (Source Papers, LASSI), unchanged:

    nvcc -std=c++14 -Xcompiler -Wall -arch=<ARCH> -O3 -o main <sources>

NvccSm80 is the preset with ARCH "sm_80". parse_diagnostics reads the stderr
of the CUDA 12 EDG front end, the host GCC (through -Xcompiler -Wall), the
nvcc driver, ptxas, and the linker; the patterns below each show a sample
line.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from lassi.core.record import Diagnostic
from lassi.core.registry import register
from lassi.toolchains._base import OUTPUT, CommandRunner, CompilerToolchain
from lassi.toolchains._stderr import (
    COLLECT2,
    EDG_SEVERITY,
    GCC,
    NO_FILES,
    UNDEFINED_REFERENCE,
    LinePattern,
    compile_diagnostic,
    error_from_message,
    parse_stderr,
)

# EDG front end: "<file>(<line>): <severity>[ #<number>[-D]]: <message>", for example
#   main.cu(8): error: identifier "undefined_var" is undefined
#   kernels/scale.cuh(5): warning #177-D: variable "unused" was declared but never referenced
#   main.cu(1): catastrophic error: cannot open source file "kernels/missing.cuh"
# The code is the number with any "-D"; a remark is a note, and a catastrophic, command-line, or
# internal error is an error. The file holds no ": ", so a GCC line whose message contains "(3): error: "
# is not read as EDG. Continuation lines, a source echo, and a caret line may follow.
_EDG = re.compile(
    r"(?P<file>(?:(?!: ).)+?)\((?P<line>[0-9]+)\): "
    r"(?P<severity>catastrophic error|command-line error|internal error|error|warning|remark)"
    r"(?: #(?P<code>[0-9]+(?:-D)?))?"
    r": (?P<message>.*)"
)

# The nvcc driver: "nvcc fatal   : <message>", for example
#   nvcc fatal   : Unsupported gpu architecture 'compute_80'
_DRIVER_FATAL = re.compile(r"nvcc fatal\s*:\s*(?P<message>.*)")

# ptxas: "ptxas <severity> : <message>" with varying spacing, for example
#   ptxas error   : Entry function '_Z6reducePKfPfi' uses too much shared data (0x10000 bytes, 0xc000 max)
# A ptxas fatal is an error.
_PTXAS = re.compile(r"ptxas\s+(?P<severity>error|warning|fatal)\s*:\s*(?P<message>.*)")
_PTXAS_SEVERITY = {"error": "error", "warning": "warning", "fatal": "error"}


def _edg(match: re.Match[str]) -> Diagnostic:
    """Return the Diagnostic for an EDG line; parse_stderr sets the column from the echo and caret."""
    return compile_diagnostic(
        EDG_SEVERITY[match["severity"]],
        match["message"],
        code=match["code"],
        file=match["file"],
        line=int(match["line"]),
    )


def _ptxas(match: re.Match[str]) -> Diagnostic:
    """Return the Diagnostic for a ptxas line, which names no file."""
    return compile_diagnostic(_PTXAS_SEVERITY[match["severity"]], match["message"])


# Tried in this order on each line; the linker pattern is last because it is the loosest.
_PATTERNS = (
    LinePattern(_EDG, _edg, echoed=True),
    GCC,
    LinePattern(_DRIVER_FATAL, error_from_message),
    LinePattern(_PTXAS, _ptxas),
    COLLECT2,
    UNDEFINED_REFERENCE,
)


def parse_diagnostics(stderr: str, files: Mapping[str, str] = NO_FILES) -> list[Diagnostic]:
    """Return the compile-stage Diagnostics in nvcc's `stderr`, in order.

    `files` (relative path -> text) are the files built; an EDG diagnostic
    gets its column by aligning the source echo with its line in them.
    Lines that match no pattern are skipped: error summaries, Remark lines,
    GCC context and source lines, and blank lines.
    """
    return parse_stderr(stderr, files, _PATTERNS)


class NvccToolchain(CompilerToolchain):
    """Compiles CUDA with nvcc; a preset subclass sets `name` and ARCH, the GPU architecture."""

    ARCH = ""
    SOURCE_SUFFIXES = (".cu", ".cpp", ".cc", ".cxx", ".c")
    capabilities = frozenset({"cuda", "emits_warnings", "diagnostics"})

    def __init__(
        self, *, executable: str = "nvcc", runner: CommandRunner | None = None, timeout_s: float = 600.0
    ) -> None:
        """Keep the nvcc executable, the command runner (None means subprocess_runner), and the timeout."""
        super().__init__(executable=executable, runner=runner, timeout_s=timeout_s)

    def command(self, sources: Sequence[str]) -> list[str]:
        """Return the nvcc command line: the LASSI compile flags, -o main, then `sources` as given."""
        flags = ["-std=c++14", "-Xcompiler", "-Wall", f"-arch={self.ARCH}", "-O3"]
        return [self.executable, *flags, "-o", OUTPUT, *sources]

    def parse(self, stderr: str, files: Mapping[str, str]) -> list[Diagnostic]:
        """Return the Diagnostics in nvcc's `stderr` (see parse_diagnostics)."""
        return parse_diagnostics(stderr, files)


@register("Toolchain", "nvcc-sm80")
class NvccSm80(NvccToolchain):
    """nvcc for sm_80 GPUs, with the LASSI compile flags."""

    name = "nvcc-sm80"
    ARCH = "sm_80"
    # The pinned nvcc: toolchains/cuda.pin, at <toolchains root>/<PREFIX_NAME>/bin/nvcc (see lassi.toolchains).
    PIN = "cuda"
    PIN_BIN = "bin/nvcc"
