"""The OpenMP-construct tag of the Harness Contract (task P17.40): read_openmp_tags.

A translation into OpenMP C++ on the host CPU is judged by its outputs
alone, and the pinned host g++ builds every C++ program with -fopenmp, so a
candidate that returns the serial program unchanged would pass binary_io
(plans/spikes/bale.md, section 7.2). This reading tags such a candidate and
never fails it: the Toolchain gcc-openmp (lassi.toolchains.gcc) declares the
capability static_tags (lassi.core.capabilities) and answers with
read_openmp_tags, and compile_loop appends the tags to the attempt's
diagnostics after every build that gave a program. Like a guard's
diagnostics, a tag is parse-stage, its code starts with "guard-", so it
never enters a correction prompt, and it changes no stage, correction, end
reason, guard, or score.

What is read: the model's files whose suffix is one of SUFFIXES, in sorted
path order, each lexed by lassi.toolchains._cxx_scan (comments are blanks,
`#if 0` and `#if false` regions are dropped, and string literals are
tokens). The harness files and files of other suffixes are never read, and
nothing is compiled, run, or opened: only the `files` argument is read.

An OpenMP construct is:

- a directive line, outside a dropped region, whose tokens after `#` are
  `pragma` and then `omp` (spaces around `#` and between the words allowed);
- or the identifier `_Pragma` followed by `(` and a string literal whose
  text, after its opening quote and any spaces, starts with the word omp.

A comment, a string literal elsewhere, a dropped region, `#pragma ompx`,
and an OpenMP runtime call (omp_get_num_threads) are not constructs: the
tag asks whether the program directs any parallel work, and a runtime call
alone directs none.

The reading: () when any file read holds a construct; otherwise one
parse-stage warning, code NO_OPENMP_CODE, at the first file read (none
when no file was read), saying the program holds no OpenMP construct and is
tagged, not failed. A file the lexer cannot read (an unterminated comment
or literal; lassi.toolchains._cxx_scan Unreadable) is skipped, and when no
construct is found elsewhere the reading is one parse-stage note, code
NOT_CHECKED_CODE, naming the first such file and the reason, in place of
the warning. A defect of the reader gives that note too, naming the
exception's class, so a reading never stops a run.

Limits, known cases, not complete: no macro is expanded, so a pragma made
by a macro other than `_Pragma` written in place is missed and the program
is tagged; every arm of a conditional other than `#if 0` and `#if false` is
read, so a construct under `#ifdef _OPENMP` counts; and a construct in
code that never runs (an unused function) counts.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath

from lassi.core.record import Diagnostic
from lassi.toolchains._cxx_scan import IDENT, LITERAL, Lexed, Unreadable, lex

# The suffixes of the model's files the reading lexes: C and C++ sources and headers.
SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".inl", ".ipp")
NO_OPENMP_CODE = "guard-openmp-none"
NOT_CHECKED_CODE = "guard-openmp-not-checked"
NO_OPENMP_MESSAGE = (
    "OpenMP-construct tag: no OpenMP construct (a #pragma omp directive or _Pragma(\"omp ...\")) in the program's "
    "files, so it runs serially; tagged, not failed (Harness Contract)"
)


def read_openmp_tags(files: Mapping[str, str]) -> tuple[Diagnostic, ...]:
    """Return the OpenMP-construct tags of the program built from the model's `files` (the module docstring).

    `files` maps each build-directory path to its text and is not changed.
    Every input gives a reading: a defect in the reader gives the
    not-checked note naming the exception's class.
    """
    try:
        return _reading(files)
    except Exception as error:  # The reading never stops a run (module docstring)
        return (_note(f"the reader failed with {type(error).__name__}"),)


def _reading(files: Mapping[str, str]) -> tuple[Diagnostic, ...]:
    """Return the reading by the module docstring's rules."""
    read = [path for path in sorted(files) if PurePosixPath(path).suffix in SUFFIXES]
    unreadable: tuple[str, str] | None = None
    for path in read:
        try:
            lexed = lex(files[path], path)
        except Unreadable as error:
            unreadable = unreadable or (path, error.reason)
            continue
        if _has_construct(lexed):
            return ()
    if unreadable is not None:
        path, reason = unreadable
        return (_note(f"{path} could not be read ({reason})", path),)
    return (
        Diagnostic(stage="parse", severity="warning", code=NO_OPENMP_CODE, file=read[0] if read else None,
                   message=NO_OPENMP_MESSAGE),
    )


def _has_construct(lexed: Lexed) -> bool:
    """Return True when the lexed file holds a `#pragma omp` directive or a `_Pragma("omp ...")` in code."""
    for directive in lexed.directives:
        words = [token.text for token in directive.tokens[1:3]]
        if words == ["pragma", "omp"]:
            return True
    tokens = lexed.tokens
    for index, token in enumerate(tokens[:-2]):
        if token.kind == IDENT and token.text == "_Pragma" and tokens[index + 1].text == "(":
            literal = tokens[index + 2]
            if literal.kind == LITERAL and _omp_literal(literal.text):
                return True
    return False


def _omp_literal(text: str) -> bool:
    """Return True for a string literal whose text, after its opening quote and spaces, starts with the word omp."""
    quote = text.find('"')
    if quote < 0:
        return False
    body = text[quote + 1:].lstrip(" \t")
    return body.startswith("omp") and not (body[3:4].isalnum() or body[3:4] == "_")


def _note(reason: str, path: str | None = None) -> Diagnostic:
    """Return the not-checked note for `reason`, at `path` when the reason names a file."""
    message = f"OpenMP-construct tag not checked: {reason} (Harness Contract)"
    return Diagnostic(stage="parse", severity="note", code=NOT_CHECKED_CODE, file=path, message=message)
