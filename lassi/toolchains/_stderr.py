"""Shared pieces of the compiler stderr parsers in lassi.toolchains.nvcc and lassi.toolchains.nvcpp.

A parser reads stderr line by line (split on "\\n" only) and tries each
LinePattern in order against the line with trailing whitespace removed; the
first match becomes one Diagnostic, and a line that matches no pattern is
skipped. A GCC source echo or caret line (a line-number gutter ending in
"|") is skipped without trying the patterns, so echoed source text never
becomes a diagnostic. Every Diagnostic has stage "compile" (bible Result
Record, Diagnostic).

A pattern may carry a `fold`, which gets the Diagnostic built from the
match, may read the lines after it and the built files, and returns the
Diagnostic to keep. EDG front ends may print indented continuation lines
under a diagnostic (such as "argument types are: (int *)"), then the
offending source line and a caret line. fold_edg_lines appends the
continuation lines to the message after "; ", and when a source echo and a
caret line (spaces, one "^") follow, it consumes both and takes the column
from edg_column; otherwise the column stays None. The linker pattern's fold
keeps the linker's place only as a built file (_fold_linker_place), and an
adapter may fold a GCC style line to decide whether to keep GCC's column.

A line or column number is read only up to 10 digits, enough for any C or
C++ line number (#line allows at most 2147483647). Model-controlled text can
reach stderr, and int() refuses a digit string past 4300 digits, so a longer
number never becomes a place.

This module also holds what both adapters share: the EDG severity words,
the severity words of the CUDA tools (ptxas under nvcc, nvlink under nvc++),
and the patterns for host GCC style lines and the linker lines.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from lassi.core.record import Diagnostic

# The default `files` of parse_diagnostics: no file text, so no EDG column, no GCC column, and no linker place.
NO_FILES: Mapping[str, str] = MappingProxyType({})

# The caret line under an EDG source echo: spaces, one "^", for example "            ^".
_CARET = re.compile(r"[ \t]*\^\s*")
# The start of a GCC source echo or caret line, a line-number gutter, for example (capture
# nvcc_host_gcc_warning)
#   "   16 |     for (int i = 0; i < host.size(); i++) {"  and  "      |                 ~~^~~~~~~~~~~~~"
_GCC_GUTTER = re.compile(r"[ \t]*[0-9]*[ \t]*\|")

# What a pattern does with the lines after its match: it gets the Diagnostic built from the match, the stderr
# lines, the index of the line after the match, the files, and the patterns, and returns the Diagnostic to keep
# with the index of the first line it leaves for the parser.
Fold = Callable[[Diagnostic, list[str], int, Mapping[str, str], Sequence["LinePattern"]], tuple[Diagnostic, int]]


@dataclass(frozen=True)
class LinePattern:
    """One stderr line format: a regular expression matched against the whole line, and its Diagnostic builder.

    `fold`, when set, reads the lines after a match (see Fold): fold_edg_lines
    for EDG formats, whose line may be followed by continuation lines, a
    source echo, and a caret line; the linker's check of its place; or an
    adapter's check of a GCC column.
    """

    regex: re.Pattern[str]
    build: Callable[[re.Match[str]], Diagnostic]
    fold: Fold | None = None


def compile_diagnostic(
    severity: str,
    message: str,
    *,
    code: str | None = None,
    file: str | None = None,
    line: int | None = None,
    column: int | None = None,
) -> Diagnostic:
    """Return a compile-stage Diagnostic."""
    return Diagnostic(
        stage="compile", severity=severity, code=code, file=file, line=line, column=column, message=message
    )


def parse_stderr(stderr: str, files: Mapping[str, str], patterns: Sequence[LinePattern]) -> list[Diagnostic]:
    """Return the Diagnostics in `stderr`, in order; `files` (path -> text) lets a fold check a column."""
    lines = stderr.split("\n")
    diagnostics: list[Diagnostic] = []
    index = 0
    while index < len(lines):
        found = _line_match(lines[index], patterns)
        index += 1
        if found is None:
            continue
        pattern, match = found
        diagnostic = pattern.build(match)
        if pattern.fold is not None:
            diagnostic, index = pattern.fold(diagnostic, lines, index, files, patterns)
        diagnostics.append(diagnostic)
    return diagnostics


def _line_match(line: str, patterns: Sequence[LinePattern]) -> tuple[LinePattern, re.Match[str]] | None:
    """Return the first pattern that matches `line` without trailing whitespace, with its match, or None.

    A GCC gutter line (source echo or caret) is never tried against the patterns.
    """
    if _GCC_GUTTER.match(line):
        return None
    stripped = line.rstrip()
    for pattern in patterns:
        match = pattern.regex.fullmatch(stripped)
        if match is not None:
            return pattern, match
    return None


def fold_edg_lines(
    diagnostic: Diagnostic, lines: list[str], index: int, files: Mapping[str, str], patterns: Sequence[LinePattern]
) -> tuple[Diagnostic, int]:
    """Fold the lines after an EDG diagnostic into it; return it with the index of the first line left.

    From `index` on, indented continuation lines are appended to the message
    after "; " until a source echo and caret line pair, which is consumed and
    gives the column, or until any other line, which is left for the parser.
    """
    extra: list[str] = []
    column = None
    while index < len(lines):
        if index + 1 < len(lines) and _CARET.fullmatch(lines[index + 1]):
            column = edg_column(files, diagnostic, lines[index], lines[index + 1])
            index += 2
            break
        if not _is_continuation(lines[index], patterns):
            break
        extra.append(lines[index].strip())
        index += 1
    message = "; ".join([diagnostic.message, *extra])
    return dataclasses.replace(diagnostic, column=column, message=message), index


def _is_continuation(line: str, patterns: Sequence[LinePattern]) -> bool:
    """Return True for an indented, non-blank line that is no caret, GCC gutter, or diagnostic line."""
    return (
        line[:1] in (" ", "\t")
        and bool(line.strip())
        and _CARET.fullmatch(line) is None
        and _GCC_GUTTER.match(line) is None
        and _line_match(line, patterns) is None
    )


@functools.lru_cache(maxsize=8)
def _split_lines(text: str) -> tuple[str, ...]:
    """Return `text` split on "\\n" only; cached, since every diagnostic on one file asks for the same text."""
    return tuple(text.split("\n"))


def _file_line(files: Mapping[str, str], diagnostic: Diagnostic) -> str | None:
    """Return line `diagnostic.line` of `files[diagnostic.file]` without trailing whitespace, or None.

    The text is split on "\\n" only, so the "\\r" of a CRLF line goes with the
    trailing whitespace. None when the file text is not given or the line is
    out of range.
    """
    text = files.get(diagnostic.file) if diagnostic.file is not None else None
    if text is None or diagnostic.line is None:
        return None
    real_lines = _split_lines(text)
    if not 1 <= diagnostic.line <= len(real_lines):
        return None
    return real_lines[diagnostic.line - 1].rstrip()


def edg_column(files: Mapping[str, str], diagnostic: Diagnostic, echo: str, caret: str) -> int | None:
    """Return the 1-based column an EDG caret points at, by aligning the source echo with the real line.

    The real line is line `diagnostic.line` of `files[diagnostic.file]`, split
    on "\\n" only. When the echo, trailing whitespace stripped, ends with the
    real line, trailing whitespace stripped, the echo prefix is the part
    before it and the column is caret index - prefix length + 1. The column is
    None when the file text is not given, the line is out of range or blank,
    the echo does not end with it, or the result would be below 1.

    Out of range is about lines only. The caret may sit past the end of a
    matched line and that column is kept as the compiler gives it: for a
    missing header nvc++ puts it after the closing quote (capture
    nvcpp_missing_include, main.cpp line 2 of 26 characters, column 27):
      "main.cpp", line 2: catastrophic error: cannot open source file "kernels/scale.h"
        #include "kernels/scale.h"
                                  ^
    """
    real = _file_line(files, diagnostic)
    shown = echo.rstrip()
    if not real or not shown.endswith(real):
        return None
    column = caret.index("^") - (len(shown) - len(real)) + 1
    return column if column >= 1 else None


# The Diagnostic severity of each EDG severity word, shared by both EDG patterns.
EDG_SEVERITY: Mapping[str, str] = MappingProxyType(
    {
        "catastrophic error": "error",
        "command-line error": "error",
        "internal error": "error",
        "error": "error",
        "warning": "warning",
        "remark": "note",
    }
)


def error_from_message(match: re.Match[str]) -> Diagnostic:
    """Return an error with only the match's `message` group set, for lines that name no file."""
    return compile_diagnostic("error", match["message"])


# The Diagnostic severity of each severity word the CUDA tools print after their name, as in "ptxas error   : ..."
# and "nvlink error   : ..."; a fatal is an error.
CUDA_TOOL_SEVERITY: Mapping[str, str] = MappingProxyType({"error": "error", "warning": "warning", "fatal": "error"})


def cuda_tool_diagnostic(match: re.Match[str]) -> Diagnostic:
    """Return the Diagnostic for a CUDA tool line from its `severity` and `message` groups; it names no file."""
    return compile_diagnostic(CUDA_TOOL_SEVERITY[match["severity"]], match["message"])


# Host GCC style: "<file>:<line>:<column>: <severity>: <message>", with an optional trailing
# " [<flag>]" that becomes the code, for example (capture nvcc_host_gcc_warning, message shortened)
#   main.cu:16:19: warning: comparison of integer expressions of different signedness: ... [-Wsign-compare]
# and, a format no capture shows yet,
#   main.cu:3:10: fatal error: kernels/missing.cuh: No such file or directory
# A source echo and a caret line in a gutter follow; parse_stderr skips them. The column is kept here as
# printed, and each adapter decides whether to keep it (lassi.toolchains.nvcc never does; lassi.toolchains.nvcpp
# does only for a built file).
_GCC = re.compile(
    r"(?P<file>.+?):(?P<line>[0-9]{1,10}):(?P<column>[0-9]{1,10}): "
    r"(?P<severity>fatal error|error|warning|note): "
    r"(?P<message>.*?)"
    r"(?: \[(?P<code>-[^\]\s]+)\])?"
)
_GCC_SEVERITY = {"fatal error": "error", "error": "error", "warning": "warning", "note": "note"}


def _gcc(match: re.Match[str]) -> Diagnostic:
    """Return the Diagnostic for a GCC style line; a fatal error is an error, and column 0 (unknown) is None."""
    return compile_diagnostic(
        _GCC_SEVERITY[match["severity"]],
        match["message"],
        code=match["code"],
        file=match["file"],
        line=int(match["line"]),
        column=int(match["column"]) or None,
    )


# The collect2 link driver: "collect2: error: <message>", for example (capture nvcc_linker_error)
#   collect2: error: ld returned 1 exit status
_COLLECT2 = re.compile(r"collect2: error: (?P<message>.*)")

# The linker: any line holding "undefined reference to"; the message is the text after the last ": "
# (the last one that still leaves the phrase in the message), for example (fixtures nvcc_linker_error and
# nvcpp_linker_error, from the clean capture rx 20260923-211958-desktop-8r113ei-p0-core-d221; the second with its
# workdir path and the helper's signature shortened with "...")
#   tmpxft_00000002_00000000-6_main.cudafe1.cpp:(.text.startup+0x2c): undefined reference to `helper(float*, int)'
#   /mnt/nvme10/joseph_ufl/lassi-runs/.../work/nvcpp_linker_error/main.cpp:21: undefined reference to `helper(...)'
# The place is the text before that ": ", after any earlier ": " (such as a "/usr/bin/ld: " in front). When it
# is "<path>:<line>", as in the second sample, it gives the file and line, and _fold_linker_place keeps them only
# as a built file; a temporary file with a section offset, as in the first sample, gives neither. The
# "/usr/bin/ld: <object>: in function `main':" line before it matches no pattern. The regex only finds the
# phrase; _undefined_reference cuts the text with str.rfind, which stays linear in the line length however many
# ": " the line holds.
_LINKER_PHRASE = "undefined reference to"
_UNDEFINED_REFERENCE = re.compile(r".*undefined reference to.*")
_LINKER_PLACE = re.compile(r"(?P<file>[^:]+):(?P<line>[0-9]{1,10})")


def _undefined_reference(match: re.Match[str]) -> Diagnostic:
    """Return the error for a linker line: the text after the last ": " before the last "undefined reference to".

    A place "<path>:<line>" just before that ": " gives the file and line,
    which _fold_linker_place then checks against the built files.
    """
    line = match[0]
    cut = line.rfind(": ", 0, line.rfind(_LINKER_PHRASE))
    if cut < 0:
        return compile_diagnostic("error", line)
    start = line.rfind(": ", 0, cut)
    place = _LINKER_PLACE.fullmatch(line, 0 if start < 0 else start + 2, cut)
    if place is None:
        return compile_diagnostic("error", line[cut + 2 :])
    return compile_diagnostic("error", line[cut + 2 :], file=place["file"], line=int(place["line"]))


def _built_file(path: str | None, files: Mapping[str, str]) -> str | None:
    """Return the longest key of `files` that `path` equals or ends with after a "/", or None."""
    if path is None:
        return None
    keys = [key for key in files if path == key or path.endswith("/" + key)]
    return max(keys, key=len, default=None)


def _fold_linker_place(
    diagnostic: Diagnostic, lines: list[str], index: int, files: Mapping[str, str], patterns: Sequence[LinePattern]
) -> tuple[Diagnostic, int]:
    """Keep a linker Diagnostic's file and line only as a built file and a line in it; consume no line.

    The path names a built file when it is a key of `files` or ends with "/"
    and a key; the longest such key becomes the file, so an absolute path in
    the build directory becomes the relative name in `files`. File and line
    become None when no key matches, no files are given, or the line is past
    the end of that file.
    """
    located = dataclasses.replace(diagnostic, file=_built_file(diagnostic.file, files))
    if located.file is None or _file_line(files, located) is None:
        return dataclasses.replace(diagnostic, file=None, line=None), index
    return located, index


GCC = LinePattern(_GCC, _gcc)
COLLECT2 = LinePattern(_COLLECT2, error_from_message)
UNDEFINED_REFERENCE = LinePattern(_UNDEFINED_REFERENCE, _undefined_reference, fold=_fold_linker_place)
