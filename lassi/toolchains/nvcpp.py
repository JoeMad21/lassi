"""The nvc++ toolchain adapter, registered as Toolchain "nvcpp-cc80" (bible Component Interfaces, Toolchain).

NvcppToolchain compiles OpenMP offload sources with one nvc++ command
carrying the LASSI compile flags from the bible (Source Papers, LASSI),
unchanged:

    nvc++ -Wall -O3 -Minfo -mp=gpu -gpu=<GPU> -o main <sources>

NvcppCc80 is the preset with GPU "cc80". parse_diagnostics reads the stderr
of the NVHPC EDG front end, the NVC++ backend, the nvc++ driver, GCC style
and linker lines (nvc++ runs the link step itself), and nvlink, the device
linker; the patterns below each show a sample line, from the captures in
tests/toolchains/fixtures/ where one shows it. The -Minfo report goes to
stderr too and matches no pattern, so it yields no diagnostics; for example
(capture nvcpp_minfo_clean)
    saxpy(int, float, float const*, float*):
          4, #omp target teams distribute parallel for
              4, Generating "nvkernel__Z5saxpyifPKfPf_F1L4_2" GPU kernel
          6, Loop not vectorized/parallelized: not countable
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
    EDG_SEVERITY,
    GCC,
    NO_FILES,
    UNDEFINED_REFERENCE,
    LinePattern,
    compile_diagnostic,
    cuda_tool_diagnostic,
    fold_edg_lines,
    parse_stderr,
)

# EDG front end: '"<file>", line <line>: <severity>: <message>', with an optional trailing
# " [<tag>]" that becomes the code, for example (captures nvcpp_edg_error, nvcpp_edg_warning, and
# nvcpp_missing_include; a warning carries its tag, an error none)
#   "main.cpp", line 7: error: identifier "undefined_var" is undefined
#   "main.cpp", line 13: warning: variable "unused" was declared but never referenced [declared_but_not_referenced]
#   "main.cpp", line 2: catastrophic error: cannot open source file "kernels/scale.h"
# A remark is a note, and a catastrophic, command-line, or internal error is an error (NVHPC's EDG
# front end preprocesses too, so a missing header is a catastrophic error). Continuation lines, a
# source echo, and a caret line may follow.
_EDG = re.compile(
    r'"(?P<file>[^"]+)", line (?P<line>[0-9]{1,10}): '
    r"(?P<severity>catastrophic error|command-line error|internal error|error|warning|remark): "
    r"(?P<message>.*?)"
    r"(?: \[(?P<code>[A-Za-z_][A-Za-z0-9_]*)\])?"
)

# NVC++ backend: "NVC++-<S>-<nnnn>-<message>", with an optional trailing "(<file>: <line>)" after one or more
# blanks, for example (captures nvcpp_backend_error and nvcpp_fatal_abort, both messages shortened with "...")
#   NVC++-S-1101-The maximum stack size ... is limited to 524288 bytes: 1048676 (main.cpp: 4)
#   NVC++-F-0000-Internal compiler error. child tinfo ... outlining function for host    1198  (main.cpp: 8)
# The code is "<S>-<nnnn>". S is I (a note), W (a warning), S (severe, an error), or F (fatal, an error).
# The message is the text before the blanks, as printed, so an internal error keeps its internal number and
# its inner blanks. The look-behind lets only the first blank of a run start the separator, which keeps the
# match linear in the line length.
_BACKEND = re.compile(
    r"NVC\+\+-(?P<severity>[IWSF])-(?P<number>[0-9]+)-"
    r"(?P<message>.*?)"
    r"(?:(?<=\S)\s+\((?P<file>[^()]+): (?P<line>[0-9]{1,10})\))?"
)
_BACKEND_SEVERITY = {"I": "note", "W": "warning", "S": "error", "F": "error"}

# nvlink, the CUDA device linker: "nvlink <severity> : <message>" with varying spacing, for example (exploratory
# capture nvcpp_nvlink_error, from an OpenMP target region calling a function with no device definition, in
# dirty-tree rx 20260923-202909-desktop-8r113ei-p0-core-09c6; the workdir and toolchain paths shortened with "...")
#   nvlink error   : Undefined reference to '_Z5twicef' in '/mnt/nvme10/.../@lassi-tmp/nvc++8c0N0-iy_9.o'
#   pgacclnk: child process exit status 2: /mnt/nvme10/joseph_ufl/toolchains/nvhpc@24.11/.../bin/tools/nvdd
# nvlink names a symbol and an object, never a source line, and the object is a temporary one in the compile's
# private TMPDIR (@lassi-tmp). So the Diagnostic has no file, line, or column, and the message is the text after
# the ":" as printed, the object path included; a path ending in "<file>:<number>" is never read as a place. A
# fatal is an error; no capture shows an nvlink warning or fatal yet. The pgacclnk line after it matches no pattern.
# It is tried before the GCC style pattern: an asm label can make a symbol name any text, "a:1:2: error: b" too.
# The object path is kept as printed, although it names the run's workdir: it is the tool's text.
_NVLINK = re.compile(r"nvlink\s+(?P<severity>error|warning|fatal)\s*:\s*(?P<message>.*)")

# The nvc++ driver: "nvc++-<Severity>-<message>", with no place, for example (plans/spikes/p0-toolchains-verify.md
# run 2a, rx 20260923-045538-desktop-8r113ei-p0-core-bde2 from a clean commit, before the CUDA home was set; then
# the P0.15 exploratory dirty-tree probe rx 20260923-104313-desktop-8r113ei-p0-core-2b22, whose source crashed the
# front end by accident; both shortened with "...")
#   nvc++-Error-A CUDA toolkit matching the current driver version (0) or a supported older version (11.8) was ...
#   nvc++-Fatal-/mnt/nvme10/joseph_ufl/toolchains/nvhpc@24.11/.../bin/tools/nvcpfe TERMINATED by signal 11
# The Diagnostic has no file, line, column, or code, and the message is the text after the second "-" as printed.
# Error and Fatal are errors; Warning is a warning, and no capture shows one yet. No source compiled with the preset
# flags gave a driver line without crashing a compiler (P0.17). The line holds no number, so no 10-digit bound
# applies. It is tried before the GCC style pattern, since a driver message may quote a source name the model chose,
# and that name may hold "a:1:2: error: b". A GCC style line is read as a driver line only when a built file is
# itself named "nvc++-Error-..." or the like, and its place then becomes None, the safe side of the rule.
_DRIVER = re.compile(r"nvc\+\+-(?P<severity>Error|Fatal|Warning)-(?P<message>.*)")
_DRIVER_SEVERITY = {"Error": "error", "Fatal": "error", "Warning": "warning"}


def _edg(match: re.Match[str]) -> Diagnostic:
    """Return the Diagnostic for an EDG line; parse_stderr sets the column from the echo and caret."""
    return compile_diagnostic(
        EDG_SEVERITY[match["severity"]],
        match["message"],
        code=match["code"],
        file=match["file"],
        line=int(match["line"]),
    )


def _driver(match: re.Match[str]) -> Diagnostic:
    """Return the Diagnostic for an nvc++ driver line, which names no file."""
    return compile_diagnostic(_DRIVER_SEVERITY[match["severity"]], match["message"])


def _backend(match: re.Match[str]) -> Diagnostic:
    """Return the Diagnostic for an NVC++ backend line, which never gives a column."""
    line = match["line"]
    return compile_diagnostic(
        _BACKEND_SEVERITY[match["severity"]],
        match["message"],
        code=f"{match['severity']}-{match['number']}",
        file=match["file"],
        line=None if line is None else int(line),
    )


# GCC style lines: nvc++ compiles the built files itself, with no regenerated source in between, so a GCC style
# line on a built file keeps its column as printed; no capture shows one yet. Any other place gets no column. An
# exploratory probe (nvcpp_probe_asm_int in dirty-tree rx 20260923-104618-desktop-8r113ei-p0-core-173d, not a
# fixture) showed LLVM's assembler rejecting an inline asm instruction, with an echo and a caret line (no
# gutter) that match no pattern after it:
#   <inline asm>:1:2: error: invalid instruction mnemonic 'bogus.op.s32'
# Its line and column index the asm string, not a built file; the file and line stay as printed.
def _gcc_column(
    diagnostic: Diagnostic, lines: list[str], index: int, files: Mapping[str, str], patterns: Sequence[LinePattern]
) -> tuple[Diagnostic, int]:
    """Keep GCC's column only when the named file is a key of `files`, else set it to None; consume no line."""
    if diagnostic.file in files:
        return diagnostic, index
    return dataclasses.replace(diagnostic, column=None), index


# Tried in this order on each line; the linker pattern is last because it is the loosest. The driver and nvlink
# patterns come before the GCC style one (see each).
_PATTERNS = (
    LinePattern(_EDG, _edg, fold=fold_edg_lines),
    LinePattern(_BACKEND, _backend),
    LinePattern(_DRIVER, _driver),
    LinePattern(_NVLINK, cuda_tool_diagnostic),
    dataclasses.replace(GCC, fold=_gcc_column),
    COLLECT2,
    UNDEFINED_REFERENCE,
)


def parse_diagnostics(stderr: str, files: Mapping[str, str] = NO_FILES) -> list[Diagnostic]:
    """Return the compile-stage Diagnostics in nvc++'s `stderr`, in order.

    `files` (relative path -> text) are the files built; an EDG diagnostic
    gets its column by aligning the source echo with its line in them, a
    GCC style diagnostic keeps its column only on one of them (_gcc_column),
    and a linker place is kept only as one of them; an nvlink or driver
    diagnostic has no place. Lines that match no pattern are skipped: the
    file name header nvc++ prints for each of several sources (such as
    "helper.cpp:"), error summaries, Remark lines, the -Minfo report, the
    compiler's closing status line (such as "NVC++/x86-64 Linux 24.11-0:
    compilation aborted"), the linker's "in function" line, the "pgacclnk:
    child process exit status 1: /usr/bin/ld" line and the one naming nvdd
    after an nvlink error (each only restates that a tool failed and says
    no "error:", unlike the "collect2: error: ..." line, which is read as an
    error), and blank lines.
    """
    return parse_stderr(stderr, files, _PATTERNS)


class NvcppToolchain(CompilerToolchain):
    """Compiles OpenMP offload C and C++ with nvc++; a preset subclass sets `name` and GPU, the -gpu target."""

    GPU = ""
    SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx", ".c")
    capabilities = frozenset({"openmp_offload", "emits_warnings", "diagnostics"})

    def __init__(
        self, *, executable: str = "nvc++", runner: CommandRunner | None = None, timeout_s: float = 600.0
    ) -> None:
        """Keep the nvc++ executable, the command runner (None means subprocess_runner), and the timeout."""
        super().__init__(executable=executable, runner=runner, timeout_s=timeout_s)

    def command(self, sources: Sequence[str]) -> list[str]:
        """Return the nvc++ command line: the LASSI compile flags, -o main, then `sources` as given."""
        flags = ["-Wall", "-O3", "-Minfo", "-mp=gpu", f"-gpu={self.GPU}"]
        return [self.executable, *flags, "-o", OUTPUT, *sources]

    def parse(self, stderr: str, files: Mapping[str, str]) -> list[Diagnostic]:
        """Return the Diagnostics in nvc++'s `stderr` (see parse_diagnostics)."""
        return parse_diagnostics(stderr, files)


@register("Toolchain", "nvcpp-cc80")
class NvcppCc80(NvcppToolchain):
    """nvc++ for cc80 GPUs, with the LASSI compile flags."""

    name = "nvcpp-cc80"
    GPU = "cc80"
    # The pinned nvc++: toolchains/nvhpc.pin, at <toolchains root>/<PREFIX_NAME>/<COMPILER_SUBDIR>/nvc++.
    PIN = "nvhpc"
    PIN_BIN = "{COMPILER_SUBDIR}/nvc++"
