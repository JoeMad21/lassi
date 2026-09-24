"""The nvcc toolchain adapter, registered as Toolchain "nvcc-sm80" (bible Component Interfaces, Toolchain).

NvccToolchain compiles CUDA sources with one nvcc command carrying the LASSI
compile flags from the bible (Source Papers, LASSI), unchanged:

    nvcc -std=c++14 -Xcompiler -Wall -arch=<ARCH> -O3 -o main <sources>

NvccSm80 is the preset with ARCH "sm_80". parse_diagnostics reads the stderr
of the CUDA 12 EDG front end, the host GCC (through -Xcompiler -Wall), the
nvcc driver, ptxas, and the linker; the patterns below each show a sample
line, from the captures in tests/toolchains/fixtures/ where one shows it.
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
    CUDA_TOOL_SEVERITY,
    EDG_SEVERITY,
    GCC,
    NO_FILES,
    UNDEFINED_REFERENCE,
    LinePattern,
    compile_diagnostic,
    cuda_tool_diagnostic,
    error_from_message,
    fold_edg_lines,
    parse_stderr,
)

# EDG front end: "<file>(<line>): <severity>[ #<number>[-D]]: <message>", for example (captures
# nvcc_undefined_identifier and nvcc_warning_177; an error carries no number)
#   main.cu(7): error: identifier "undefined_var" is undefined
#   kernels/scale.cuh(5): warning #177-D: variable "unused" was declared but never referenced
# and, a severity and a number without "-D" that no capture shows yet,
#   main.cu(1): catastrophic error: cannot open source file "kernels/missing.cuh"
#   main.cu(3): error #20: identifier "x" is undefined
# The code is the number with any "-D"; a remark is a note, and a catastrophic, command-line, or
# internal error is an error. The file holds no ": ", so a GCC line whose message contains "(3): error: "
# is not read as EDG. Continuation lines, a source echo, and a caret line may follow. Each diagnostic
# appears once, although nvcc runs EDG for the device and the host pass.
_EDG = re.compile(
    r"(?P<file>(?:(?!: ).)+?)\((?P<line>[0-9]{1,10})\): "
    r"(?P<severity>catastrophic error|command-line error|internal error|error|warning|remark)"
    r"(?: #(?P<code>[0-9]+(?:-D)?))?"
    r": (?P<message>.*)"
)

# The nvcc driver: "nvcc fatal   : <message>", for example (capture nvcc_fatal)
#   nvcc fatal   : Value 'sm_35' is not defined for option 'gpu-architecture'
_DRIVER_FATAL = re.compile(r"nvcc fatal\s*:\s*(?P<message>.*)")

# ptxas: "ptxas <severity> : <message>" with varying spacing, for example (capture nvcc_ptxas_error)
#   ptxas error   : Entry function '_Z6reducePKfPfi' uses too much shared data (0x40000 bytes, 0x29000 max)
# A ptxas fatal is an error; fixture nvcc_ptxas_inline_asm (below) shows one, and no fixture shows a ptxas warning
# yet.
_PTXAS = re.compile(r"ptxas\s+(?P<severity>error|warning|fatal)\s*:\s*(?P<message>.*)")

# ptxas with a place in the PTX it reads: "ptxas <PTX file>, line <line>; <severity> : <message>", with varying
# spacing before the ":", for example (fixture nvcc_ptxas_inline_asm, from inline PTX asm with an unknown modifier,
# in the clean capture rx 20260923-211958-desktop-8r113ei-p0-core-d221; the workdir shortened with "...")
#   ptxas /mnt/nvme10/.../@lassi-tmp/tmpxft_00000002_00000000-6_main.ptx, line 28; error   : Unknown modifier '.bogus'
#   ptxas fatal   : Ptx assembly aborted due to errors
# The PTX file is the one nvcc wrote in the compile's private TMPDIR (@lassi-tmp), and its line counts in that PTX;
# nvcc never gets a .ptx source, so neither indexes a built file. File and line are None, and the message keeps
# the place as printed, with ": " for "; <severity> : ": "<PTX file>, line 28: Unknown modifier '.bogus'". The
# fatal line after it is a _PTXAS line, a second error. The line is read up to 10 digits, so a longer number is
# no place and the line matches no pattern. The place is the shortest text that ends in ", line <line>" before
# "; <severity>"; a blank run before the ":" follows one place only, so the match stays linear in the line length.
# The place never starts with a padded severity or info word and a ":", so a line such as "ptxas info    : ..."
# (printed only with -v or --resource-usage, which the preset does not pass) is never read as a place. The path is
# kept as printed, although it names the run's workdir: it is the tool's text, and P0.17 keeps the PTX location in
# the message for the reader.
_PTXAS_PLACE = re.compile(
    r"ptxas (?!(?:info|error|warning|fatal)\s*:)"
    r"(?P<place>.+?, line [0-9]{1,10}); (?P<severity>error|warning|fatal)\s*:\s*(?P<message>.*)"
)


def _edg(match: re.Match[str]) -> Diagnostic:
    """Return the Diagnostic for an EDG line; parse_stderr sets the column from the echo and caret."""
    return compile_diagnostic(
        EDG_SEVERITY[match["severity"]],
        match["message"],
        code=match["code"],
        file=match["file"],
        line=int(match["line"]),
    )


def _ptxas_place(match: re.Match[str]) -> Diagnostic:
    """Return the Diagnostic for a ptxas line with a PTX place: no file or line, the place kept in the message."""
    return compile_diagnostic(CUDA_TOOL_SEVERITY[match["severity"]], f"{match['place']}: {match['message']}")


# A host GCC line keeps its file, line, flag, and message, but never its column. GCC reads a .cu file, and every
# header it includes, twice: its preprocessor reads the file as it is, so a preprocessor column (a missing
# include, an #error) indexes the named file, and its compiler reads the host code cudafe1 regenerated, so a
# compiler column counts in that text. Both echo the named file's line from disk, so nothing in stderr tells
# the two apart. Capture nvcc_host_gcc_warning shows the second:
#   main.cu:16:19: warning: comparison of integer expressions of different signedness: ... [-Wsign-compare]
#      16 |     for (int i = 0; i < host.size(); i++) {
#         |                 ~~^~~~~~~~~~~~~
# The echo is main.cu line 16 exactly, yet column 19 is the '<' of the regenerated line, which lost its
# indent; in main.cu column 19 is the ';' and the '<' is column 23. A .cpp or .c source goes to GCC without
# cudafe1, but no capture shows a GCC line for one yet, and cudafe1 regenerates it with any .cu file that
# #includes it, so its column is dropped too.
def _gcc_without_column(match: re.Match[str]) -> Diagnostic:
    """Return the Diagnostic for a host GCC line with no column (see above)."""
    return dataclasses.replace(GCC.build(match), column=None)


# Tried in this order on each line; the linker pattern is last because it is the loosest. The two ptxas patterns
# and the driver pattern come first: each needs its tool's name at the start of the line, and a ptxas or driver
# message, or the PTX name (nvcc names it after the source stem, which the model chooses), may hold text shaped
# like an EDG or GCC place, such as "x(3): error: " or "a:1:2: error: ". Tried later, the EDG or GCC pattern
# would read that text as a file and line that index no built file. An EDG or GCC line is read as one of them
# only when a built file is itself named "ptxas ..." or "nvcc fatal...", and its place then becomes None, the
# safe side of the rule.
_PATTERNS = (
    LinePattern(_PTXAS, cuda_tool_diagnostic),
    LinePattern(_PTXAS_PLACE, _ptxas_place),
    LinePattern(_DRIVER_FATAL, error_from_message),
    LinePattern(_EDG, _edg, fold=fold_edg_lines),
    LinePattern(GCC.regex, _gcc_without_column),
    COLLECT2,
    UNDEFINED_REFERENCE,
)


def parse_diagnostics(stderr: str, files: Mapping[str, str] = NO_FILES) -> list[Diagnostic]:
    """Return the compile-stage Diagnostics in nvcc's `stderr`, in order.

    `files` (relative path -> text) are the files built; an EDG diagnostic
    gets its column by aligning the source echo with its line in them, and a
    linker place is kept only as a built file. A GCC diagnostic never has a
    column (see _gcc_without_column), and a ptxas place in its PTX stays in
    the message, never in file and line. Lines that match no pattern are
    skipped: error summaries, Remark lines, GCC context and source lines,
    the linker's "in function" lines, and blank lines.
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
