"""Shared pieces of the compiler stderr parsers in lassi.toolchains.nvcc and lassi.toolchains.nvcpp.

A parser reads stderr line by line (split on "\\n" only) and tries each
LinePattern in order against the line with trailing whitespace removed; the
first match becomes one Diagnostic, and a line that matches no pattern is
skipped. A GCC source echo or caret line (a line-number gutter ending in
"|") is skipped without trying the patterns, so echoed source text never
becomes a diagnostic. Every Diagnostic has stage "compile" (bible Result
Record, Diagnostic).

EDG front ends may print indented continuation lines under a diagnostic
(such as "argument types are: (int *)"), then the offending source line and
a caret line. For a pattern marked `echoed`, the continuation lines are
appended to the message after "; ", and when a source echo and a caret line
(spaces, one "^") follow, both are consumed and the column comes from
edg_column; otherwise the column stays None.

This module also holds what both adapters share: the EDG severity words,
and the patterns for host GCC style lines and the linker lines.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from lassi.core.record import Diagnostic

# The default `files` of parse_diagnostics: no file text, so no EDG column.
NO_FILES: Mapping[str, str] = MappingProxyType({})

# The caret line under an EDG source echo: spaces, one "^", for example "            ^".
_CARET = re.compile(r"[ \t]*\^\s*")
# The start of a GCC source echo or caret line, a line-number gutter, for example
#   "   21 |     for (int i = 0; i < n; i++) {"  and  "      |                     ~~^~~"
_GCC_GUTTER = re.compile(r"[ \t]*[0-9]*[ \t]*\|")


@dataclass(frozen=True)
class LinePattern:
    """One stderr line format: a regular expression matched against the whole line, and its Diagnostic builder.

    `echoed` marks EDG formats, whose line may be followed by a source echo
    and a caret line that give the column.
    """

    regex: re.Pattern[str]
    build: Callable[[re.Match[str]], Diagnostic]
    echoed: bool = False


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
    """Return the Diagnostics in `stderr`, in order; `files` (path -> text) gives EDG columns."""
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
        if pattern.echoed:
            diagnostic, index = _fold_edg_lines(diagnostic, lines, index, files, patterns)
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


def _fold_edg_lines(
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


def edg_column(files: Mapping[str, str], diagnostic: Diagnostic, echo: str, caret: str) -> int | None:
    """Return the 1-based column an EDG caret points at, by aligning the source echo with the real line.

    The real line is line `diagnostic.line` of `files[diagnostic.file]`, split
    on "\\n" only. When the echo, trailing whitespace stripped, ends with the
    real line, trailing whitespace stripped, the echo prefix is the part
    before it and the column is caret index - prefix length + 1. The column is
    None when the file text is not given, the line is out of range or blank,
    the echo does not end with it, or the result would be below 1.
    """
    text = files.get(diagnostic.file) if diagnostic.file is not None else None
    if text is None or diagnostic.line is None:
        return None
    real_lines = text.split("\n")
    if not 1 <= diagnostic.line <= len(real_lines):
        return None
    real = real_lines[diagnostic.line - 1].rstrip()
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


# Host GCC style: "<file>:<line>:<column>: <severity>: <message>", with an optional trailing
# " [<flag>]" that becomes the code, for example
#   main.cu:21:23: warning: comparison of integer expressions of different signedness: ... [-Wsign-compare]
#   main.cu:3:10: fatal error: kernels/missing.cuh: No such file or directory
_GCC = re.compile(
    r"(?P<file>.+?):(?P<line>[0-9]+):(?P<column>[0-9]+): "
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


# The collect2 link driver: "collect2: error: <message>", for example
#   collect2: error: ld returned 1 exit status
_COLLECT2 = re.compile(r"collect2: error: (?P<message>.*)")

# The linker: any line holding "undefined reference to"; the message is the text after the last ": "
# (the last one that still leaves the phrase in the message), for example
#   main.cpp:(.text+0x51): undefined reference to `helper(float*, int)'
# The regex only finds the phrase; _undefined_reference cuts the message with str.rfind, which stays
# linear in the line length however many ": " the line holds.
_LINKER_PHRASE = "undefined reference to"
_UNDEFINED_REFERENCE = re.compile(r".*undefined reference to.*")


def _undefined_reference(match: re.Match[str]) -> Diagnostic:
    """Return the error for a linker line: the text after the last ": " before the last "undefined reference to"."""
    line = match[0]
    cut = line.rfind(": ", 0, line.rfind(_LINKER_PHRASE))
    return compile_diagnostic("error", line if cut < 0 else line[cut + 2 :])


GCC = LinePattern(_GCC, _gcc)
COLLECT2 = LinePattern(_COLLECT2, error_from_message)
UNDEFINED_REFERENCE = LinePattern(_UNDEFINED_REFERENCE, _undefined_reference)
