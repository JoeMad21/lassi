"""The nvc++ toolchain adapter, registered as Toolchain "nvcpp-cc80" (bible Component Interfaces, Toolchain).

NvcppToolchain compiles OpenMP offload sources with one nvc++ command
carrying the LASSI compile flags from the bible (Source Papers, LASSI),
unchanged:

    nvc++ -Wall -O3 -Minfo -mp=gpu -gpu=<GPU> -o main <sources>

NvcppCc80 is the preset with GPU "cc80". parse_diagnostics reads the stderr
of the NVHPC EDG front end, the NVC++ backend, and GCC style and linker lines
(nvc++ runs the link step itself); the patterns below each show a sample
line. The -Minfo report matches no pattern, so it yields no diagnostics.
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
    parse_stderr,
)

# EDG front end: '"<file>", line <line>: <severity>: <message>', with an optional trailing
# " [<tag>]" that becomes the code, for example
#   "main.cpp", line 8: error: identifier "undefined_var" is undefined
#   "main.cpp", line 14: warning: variable "unused" was declared but never referenced [declared_but_not_referenced]
#   "main.cpp", line 1: catastrophic error: cannot open source file "kernels/missing.h"
# A remark is a note, and a catastrophic, command-line, or internal error is an error (NVHPC's EDG
# front end preprocesses too, so a missing header is a catastrophic error). Continuation lines, a
# source echo, and a caret line may follow.
_EDG = re.compile(
    r'"(?P<file>[^"]+)", line (?P<line>[0-9]+): '
    r"(?P<severity>catastrophic error|command-line error|internal error|error|warning|remark): "
    r"(?P<message>.*?)"
    r"(?: \[(?P<code>[A-Za-z_][A-Za-z0-9_]*)\])?"
)

# NVC++ backend: "NVC++-<S>-<nnnn>-<message>", with an optional trailing " (<file>: <line>)", for example
#   NVC++-S-0155-Compiler failed to translate accelerator region (see -Minfo messages): ... (main.cpp: 21)
#   NVC++-F-0704-Compilation aborted due to previous errors.
# The code is "<S>-<nnnn>". S is I (a note), W (a warning), S (severe, an error), or F (fatal, an error).
_BACKEND = re.compile(
    r"NVC\+\+-(?P<severity>[IWSF])-(?P<number>[0-9]+)-"
    r"(?P<message>.*?)"
    r"(?: \((?P<file>[^()]+): (?P<line>[0-9]+)\))?"
)
_BACKEND_SEVERITY = {"I": "note", "W": "warning", "S": "error", "F": "error"}


def _edg(match: re.Match[str]) -> Diagnostic:
    """Return the Diagnostic for an EDG line; parse_stderr sets the column from the echo and caret."""
    return compile_diagnostic(
        EDG_SEVERITY[match["severity"]],
        match["message"],
        code=match["code"],
        file=match["file"],
        line=int(match["line"]),
    )


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


# Tried in this order on each line; the linker pattern is last because it is the loosest.
_PATTERNS = (
    LinePattern(_EDG, _edg, echoed=True),
    LinePattern(_BACKEND, _backend),
    GCC,
    COLLECT2,
    UNDEFINED_REFERENCE,
)


def parse_diagnostics(stderr: str, files: Mapping[str, str] = NO_FILES) -> list[Diagnostic]:
    """Return the compile-stage Diagnostics in nvc++'s `stderr`, in order.

    `files` (relative path -> text) are the files built; an EDG diagnostic
    gets its column by aligning the source echo with its line in them.
    Lines that match no pattern are skipped: error summaries, Remark lines,
    the -Minfo report, the compiler's closing status line, and blank lines.
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
