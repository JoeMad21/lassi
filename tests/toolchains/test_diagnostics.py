"""Tests for the nvcc and nvc++ toolchain adapters in lassi/toolchains/ (P0.5).

A Toolchain turns files into an artifact plus diagnostics parsed into records
of (severity, code, file, line, column, message, stage); the raw stderr is kept
as an attachment and never consumed downstream (bible Component Interfaces,
Toolchain row and contract rules; Result Record, Diagnostic). The two presets
are the toolchain bindings of the bible's lassi-repro recipe, `nvcc-sm80` and
`nvcpp-cc80`, and their command lines carry the LASSI compile flags from the
bible's Source Papers section unchanged.

No test here runs a compiler. parse_diagnostics is called on the hand-written
stderr files in tests/toolchains/fixtures/ (see the README there), and build
gets a fake command runner that records its call and returns canned stderr.
subprocess_runner is exercised with the current Python interpreter only. The
sources below are what the EDG echo and caret lines in the fixtures point
into, so every expected column is exact. No value in this module is a
measurement.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from lassi import toolchains
from lassi.core.capabilities import Component
from lassi.core.interfaces import BuildResult
from lassi.core.record import Diagnostic
from lassi.core.registry import DEFAULT_REGISTRY

# The package imports both adapter modules, which registers the two presets.
nvcc = toolchains.nvcc
nvcpp = toolchains.nvcpp
CommandResult = toolchains.CommandResult
subprocess_runner = toolchains.subprocess_runner

REPO = Path(__file__).resolve().parents[2]
BIBLE = REPO / "docs" / "BIBLE.md"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
OUTPUT = "main"
ATTACHMENT = "compile.stderr"
E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"

NVCC_FLAGS = ("-std=c++14", "-Xcompiler", "-Wall", "-arch=sm_80", "-O3")
NVCPP_FLAGS = ("-Wall", "-O3", "-Minfo", "-mp=gpu", "-gpu=cc80")
NVCC_SUFFIXES = (".cu", ".cpp", ".cc", ".cxx", ".c")
NVCPP_SUFFIXES = (".cpp", ".cc", ".cxx", ".c")

# ---------------------------------------------------------------------------
# Sources the fixtures point into (line numbers in the comments)

# Line 8 uses an identifier that does not exist; nvcc_undefined_identifier points at column 27.
NVCC_BROKEN = (
    "#include <cstdio>\n"
    '#include "kernels/scale.cuh"\n'
    "\n"
    "__global__ void saxpy(int n, float a, const float *x, float *y)\n"
    "{\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i < n) {\n"
    "        y[i] = a * x[i] + undefined_var;\n"
    "    }\n"
    "}\n"
)
NVCC_MAIN = NVCC_BROKEN.replace("a * x[i] + undefined_var", "scale(x[i], a) + y[i]")
# Line 5 declares a variable it never uses; nvcc_warning_177 points at column 11.
SCALE_CUH = (
    "#pragma once\n"
    "\n"
    "__device__ inline float scale(float v, float s)\n"
    "{\n"
    "    float unused = 0.0f;\n"
    "    return v * s;\n"
    "}\n"
)
# Line 8 uses an identifier that does not exist; nvcpp_edg_error points at column 23.
NVCPP_BROKEN = (
    "#include <cstdio>\n"
    "#include <vector>\n"
    "\n"
    "void saxpy(int n, float a, const float *x, float *y)\n"
    "{\n"
    "#pragma omp target teams distribute parallel for map(to: x[0:n]) map(tofrom: y[0:n])\n"
    "  for (int i = 0; i < n; i++) {\n"
    "    y[i] = a * x[i] + undefined_var;\n"
    "  }\n"
    "}\n"
)
# Line 14 declares a variable it never uses; nvcpp_edg_warning points at column 7.
NVCPP_MAIN = (
    NVCPP_BROKEN.replace("a * x[i] + undefined_var", "a * x[i] + y[i]")
    + "\n"
    + "int main()\n"
    + "{\n"
    + "  int unused = 0;\n"
    + "  std::vector<float> x(1024, 1.0f), y(1024, 2.0f);\n"
    + "  saxpy(1024, 2.0f, x.data(), y.data());\n"
    + '  std::printf("%f\\n", y[0]);\n'
    + "  return 0;\n"
    + "}\n"
)

# Line 2 includes a header the response did not return; nvcpp_missing_include points at column 10.
NVCPP_INCLUDES = '#include <cstdio>\n#include "kernels/scale.h"\n\nint main() { return 0; }\n'

NVCC_BROKEN_FILES = {"main.cu": NVCC_BROKEN, "kernels/scale.cuh": SCALE_CUH}
NVCC_FILES = {"main.cu": NVCC_MAIN, "kernels/scale.cuh": SCALE_CUH}
NVCPP_BROKEN_FILES = {"main.cpp": NVCPP_BROKEN}
NVCPP_FILES = {"main.cpp": NVCPP_MAIN}


def compile_diag(
    severity: str, code: str | None, file: str | None, line: int | None, column: int | None, message: str
) -> Diagnostic:
    """Return a compile-stage Diagnostic with every other field given explicitly."""
    return Diagnostic(
        stage="compile", severity=severity, code=code, file=file, line=line, column=column, message=message
    )


# ---------------------------------------------------------------------------
# The fixtures and the exact Diagnostic list each one parses into


@dataclass(frozen=True)
class FixtureCase:
    """One stderr fixture: the adapter module that parses it, the files it points into, and the expected list.

    `edg` marks fixtures whose diagnostics are all EDG lines with a source
    echo and caret, so their columns need the file text.
    """

    module: ModuleType
    files: Mapping[str, str]
    expected: list[Diagnostic]
    edg: bool = False


FIXTURE_CASES: dict[str, FixtureCase] = {
    "nvcc_undefined_identifier": FixtureCase(
        nvcc,
        NVCC_BROKEN_FILES,
        [compile_diag("error", None, "main.cu", 8, 27, 'identifier "undefined_var" is undefined')],
        edg=True,
    ),
    "nvcc_warning_177": FixtureCase(
        nvcc,
        NVCC_FILES,
        [
            compile_diag(
                "warning", "177-D", "kernels/scale.cuh", 5, 11, 'variable "unused" was declared but never referenced'
            )
        ],
        edg=True,
    ),
    "nvcc_host_gcc_warning": FixtureCase(
        nvcc,
        {},
        [
            compile_diag(
                "warning",
                "-Wsign-compare",
                "main.cu",
                21,
                23,
                "comparison of integer expressions of different signedness: 'int' and 'size_t' "
                "{aka 'long unsigned int'}",
            )
        ],
    ),
    "nvcc_fatal": FixtureCase(
        nvcc, {}, [compile_diag("error", None, None, None, None, "Unsupported gpu architecture 'compute_80'")]
    ),
    "nvcc_ptxas_error": FixtureCase(
        nvcc,
        {},
        [
            compile_diag(
                "error",
                None,
                None,
                None,
                None,
                "Entry function '_Z6reducePKfPfi' uses too much shared data (0x10000 bytes, 0xc000 max)",
            )
        ],
    ),
    "nvcc_linker_error": FixtureCase(
        nvcc,
        {},
        [
            compile_diag("error", None, None, None, None, "undefined reference to `helper(float*, int)'"),
            compile_diag("error", None, None, None, None, "ld returned 1 exit status"),
        ],
    ),
    "nvcc_clean": FixtureCase(nvcc, {}, []),
    "nvcpp_edg_error": FixtureCase(
        nvcpp,
        NVCPP_BROKEN_FILES,
        [compile_diag("error", None, "main.cpp", 8, 23, 'identifier "undefined_var" is undefined')],
        edg=True,
    ),
    "nvcpp_edg_warning": FixtureCase(
        nvcpp,
        NVCPP_FILES,
        [
            compile_diag(
                "warning",
                "declared_but_not_referenced",
                "main.cpp",
                14,
                7,
                'variable "unused" was declared but never referenced',
            )
        ],
        edg=True,
    ),
    "nvcpp_backend_error": FixtureCase(
        nvcpp,
        {},
        [
            compile_diag(
                "error",
                "S-0155",
                "main.cpp",
                21,
                None,
                "Compiler failed to translate accelerator region (see -Minfo messages): "
                "Could not find allocated-variable index for symbol - tmp",
            )
        ],
    ),
    "nvcpp_fatal_abort": FixtureCase(
        nvcpp, {}, [compile_diag("error", "F-0704", None, None, None, "Compilation aborted due to previous errors.")]
    ),
    "nvcpp_minfo_clean": FixtureCase(nvcpp, {}, []),
    "nvcpp_linker_error": FixtureCase(
        nvcpp, {}, [compile_diag("error", None, None, None, None, "undefined reference to `helper(float*, int)'")]
    ),
    "nvcpp_missing_include": FixtureCase(
        nvcpp,
        {"main.cpp": NVCPP_INCLUDES},
        [compile_diag("error", None, "main.cpp", 2, 10, 'cannot open source file "kernels/scale.h"')],
        edg=True,
    ),
}
EDG_FIXTURES = sorted(name for name, case in FIXTURE_CASES.items() if case.edg)


def fixture_text(name: str) -> str:
    """Return a stderr fixture exactly as stored: its bytes decoded as UTF-8, with no newline translation."""
    return (FIXTURES / f"{name}.stderr").read_bytes().decode("utf-8")


# ---------------------------------------------------------------------------
# The fixtures themselves


def test_every_fixture_file_has_a_case() -> None:
    assert sorted(path.stem for path in FIXTURES.glob("*.stderr")) == sorted(FIXTURE_CASES)
    assert (FIXTURES / "README.md").is_file()


@pytest.mark.parametrize("name", sorted(FIXTURE_CASES))
def test_fixture_is_raw_ascii_stderr(name: str) -> None:
    raw = (FIXTURES / f"{name}.stderr").read_bytes()
    assert raw.isascii(), name
    assert b"\r" not in raw, f"{name} must be stored with LF line endings"
    if raw:
        assert raw.endswith(b"\n"), name
        assert not raw.startswith((b"#", b"//")), f"{name} starts with a comment; fixtures are raw stderr"


# ---------------------------------------------------------------------------
# parse_diagnostics on the fixtures


@pytest.mark.parametrize("name", sorted(FIXTURE_CASES))
def test_fixture_parses_into_the_exact_diagnostics(name: str) -> None:
    case = FIXTURE_CASES[name]
    assert case.module.parse_diagnostics(fixture_text(name), case.files) == case.expected


@pytest.mark.parametrize("name", EDG_FIXTURES)
def test_edg_column_needs_the_file_text(name: str) -> None:
    # Without the file text the echo cannot be aligned, so the column is None and the rest is unchanged.
    case = FIXTURE_CASES[name]
    expected = [dataclasses.replace(diagnostic, column=None) for diagnostic in case.expected]
    assert case.module.parse_diagnostics(fixture_text(name)) == expected
    assert case.module.parse_diagnostics(fixture_text(name), {}) == expected


@pytest.mark.parametrize("module", [nvcc, nvcpp], ids=["nvcc", "nvcpp"])
def test_empty_stderr_parses_to_nothing(module: ModuleType) -> None:
    assert module.parse_diagnostics("") == []
    assert module.parse_diagnostics("\n\n") == []


# ---------------------------------------------------------------------------
# parse_diagnostics: one pattern at a time


NVCC_LINES = [
    pytest.param(
        'main.cu(3): error #20: identifier "blockIdy" is undefined\n',
        [compile_diag("error", "20", "main.cu", 3, None, 'identifier "blockIdy" is undefined')],
        id="edg-error-number",
    ),
    pytest.param(
        "main.cu(14): remark #186-D: pointless comparison of unsigned integer with zero\n",
        [compile_diag("note", "186-D", "main.cu", 14, None, "pointless comparison of unsigned integer with zero")],
        id="edg-remark-is-note",
    ),
    pytest.param(
        "main.cu:3:10: fatal error: kernels/missing.cuh: No such file or directory\ncompilation terminated.\n",
        [compile_diag("error", None, "main.cu", 3, 10, "kernels/missing.cuh: No such file or directory")],
        id="gcc-fatal-error",
    ),
    pytest.param(
        "/usr/local/cuda/include/crt/host_config.h:143:2: error: #error -- unsupported GNU version! "
        "gcc versions later than 12 are not supported!\n",
        [
            compile_diag(
                "error",
                None,
                "/usr/local/cuda/include/crt/host_config.h",
                143,
                2,
                "#error -- unsupported GNU version! gcc versions later than 12 are not supported!",
            )
        ],
        id="gcc-error-absolute-file",
    ),
    pytest.param(
        "main.cu:30:17: warning: format '%d' expects argument of type 'int', but argument 2 has type "
        "'size_t' {aka 'long unsigned int'} [-Wformat=]\n"
        "main.cu:30:24: note: format string is defined here\n",
        [
            compile_diag(
                "warning",
                "-Wformat=",
                "main.cu",
                30,
                17,
                "format '%d' expects argument of type 'int', but argument 2 has type 'size_t' "
                "{aka 'long unsigned int'}",
            ),
            compile_diag("note", None, "main.cu", 30, 24, "format string is defined here"),
        ],
        id="gcc-warning-flag-with-equals-and-note",
    ),
    pytest.param(
        "ptxas warning : Stack size for entry function '_Z6kernelPf' cannot be statically determined\n",
        [
            compile_diag(
                "warning",
                None,
                None,
                None,
                None,
                "Stack size for entry function '_Z6kernelPf' cannot be statically determined",
            )
        ],
        id="ptxas-warning",
    ),
    pytest.param(
        "ptxas fatal   : Unresolved extern function '_Z6helperPfi'\n",
        [compile_diag("error", None, None, None, None, "Unresolved extern function '_Z6helperPfi'")],
        id="ptxas-fatal-is-error",
    ),
    pytest.param(
        "nvcc fatal   : Don't know what to do with 'kernels/scale.cuh'\n",
        [compile_diag("error", None, None, None, None, "Don't know what to do with 'kernels/scale.cuh'")],
        id="driver-fatal",
    ),
    pytest.param(
        'main.cu(2): catastrophic error: cannot open source file "kernels/scale.cuh"\n',
        [compile_diag("error", None, "main.cu", 2, None, 'cannot open source file "kernels/scale.cuh"')],
        id="edg-catastrophic-error",
    ),
    pytest.param(
        "main.cu(40): internal error: assertion failed in lower_il\n",
        [compile_diag("error", None, "main.cu", 40, None, "assertion failed in lower_il")],
        id="edg-internal-error",
    ),
    pytest.param(
        "/usr/bin/ld: main.cpp:(.text+0x51): undefined reference to `x'\n",
        [compile_diag("error", None, None, None, None, "undefined reference to `x'")],
        id="linker-message-after-the-last-colon",
    ),
    pytest.param(
        "main.cu:7:0: warning: ignoring '#pragma unroll' [-Wunknown-pragmas]\n",
        [compile_diag("warning", "-Wunknown-pragmas", "main.cu", 7, None, "ignoring '#pragma unroll'")],
        id="gcc-column-zero-is-none",
    ),
    pytest.param(
        "main.cu:12:9: error: static assertion failed: size(3): error: too small\n",
        [compile_diag("error", None, "main.cu", 12, 9, "static assertion failed: size(3): error: too small")],
        id="gcc-message-that-looks-like-edg",
    ),
]

NVCPP_LINES = [
    pytest.param(
        '"kernels/scale.h", line 3: warning: variable "t" was set but never used [set_but_not_used]\n',
        [
            compile_diag(
                "warning", "set_but_not_used", "kernels/scale.h", 3, None, 'variable "t" was set but never used'
            )
        ],
        id="edg-warning-tag",
    ),
    pytest.param(
        '"main.cpp", line 9: error: expected a ";"\n',
        [compile_diag("error", None, "main.cpp", 9, None, 'expected a ";"')],
        id="edg-error-no-tag",
    ),
    pytest.param(
        "NVC++-W-0155-Accelerator region ignored; see -Minfo messages (main.cpp: 12)\n",
        [compile_diag("warning", "W-0155", "main.cpp", 12, None, "Accelerator region ignored; see -Minfo messages")],
        id="backend-warning",
    ),
    pytest.param(
        "NVC++-I-0035-Predefined intrinsic max loses intrinsic property (main.cpp: 7)\n",
        [compile_diag("note", "I-0035", "main.cpp", 7, None, "Predefined intrinsic max loses intrinsic property")],
        id="backend-info-is-note",
    ),
    pytest.param(
        "NVC++-F-0000-Internal compiler error. unsupported procedure (main.cpp: 40)\n",
        [compile_diag("error", "F-0000", "main.cpp", 40, None, "Internal compiler error. unsupported procedure")],
        id="backend-fatal-with-file",
    ),
    pytest.param(
        "NVC++-S-0155-Invalid accelerator region: branching into or out of region is not allowed\n",
        [
            compile_diag(
                "error",
                "S-0155",
                None,
                None,
                None,
                "Invalid accelerator region: branching into or out of region is not allowed",
            )
        ],
        id="backend-severe-without-file",
    ),
    pytest.param(
        "main.cpp:3:10: fatal error: kernels/missing.h: No such file or directory\n",
        [compile_diag("error", None, "main.cpp", 3, 10, "kernels/missing.h: No such file or directory")],
        id="gcc-fatal-error",
    ),
    pytest.param(
        "collect2: error: ld returned 1 exit status\n",
        [compile_diag("error", None, None, None, None, "ld returned 1 exit status")],
        id="collect2",
    ),
    pytest.param(
        '"main.cpp", line 5: remark: loop was vectorized\n',
        [compile_diag("note", None, "main.cpp", 5, None, "loop was vectorized")],
        id="edg-remark-is-note",
    ),
    pytest.param(
        '"main.cpp", line 3: internal error: assertion failed at: "lower_il.c", line 12\n',
        [compile_diag("error", None, "main.cpp", 3, None, 'assertion failed at: "lower_il.c", line 12')],
        id="edg-internal-error",
    ),
    pytest.param(
        '"main.cpp", line 1: command-line error: invalid macro definition: N=\n',
        [compile_diag("error", None, "main.cpp", 1, None, "invalid macro definition: N=")],
        id="edg-command-line-error",
    ),
    pytest.param(
        "NVC++-S-0155-Invalid accelerator region (main.cpp: 21)   \t\n",
        [compile_diag("error", "S-0155", "main.cpp", 21, None, "Invalid accelerator region")],
        id="backend-trailing-blanks-keep-the-file",
    ),
    pytest.param(
        "/usr/bin/ld: main.cpp:(.text+0x51): undefined reference to `x'\n",
        [compile_diag("error", None, None, None, None, "undefined reference to `x'")],
        id="linker-message-after-the-last-colon",
    ),
]


@pytest.mark.parametrize(("stderr", "expected"), NVCC_LINES)
def test_nvcc_pattern(stderr: str, expected: list[Diagnostic]) -> None:
    assert nvcc.parse_diagnostics(stderr) == expected


@pytest.mark.parametrize(("stderr", "expected"), NVCPP_LINES)
def test_nvcpp_pattern(stderr: str, expected: list[Diagnostic]) -> None:
    assert nvcpp.parse_diagnostics(stderr) == expected


SKIPPED_LINES = [
    pytest.param(nvcc, '1 error detected in the compilation of "main.cu".', id="nvcc-error-summary"),
    pytest.param(nvcc, '2 errors detected in the compilation of "main.cu".', id="nvcc-errors-summary"),
    pytest.param(
        nvcc, 'Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"', id="nvcc-remark"
    ),
    pytest.param(nvcc, "main.cu: In function 'void launch(float*, int)':", id="nvcc-gcc-in-function"),
    pytest.param(nvcc, "compilation terminated.", id="nvcc-gcc-terminated"),
    pytest.param(nvcc, "   21 |     for (int i = 0; i < n; i++) {", id="nvcc-gcc-echo"),
    pytest.param(nvcc, "      |                     ~~^~~", id="nvcc-gcc-caret"),
    # Echoed source text never becomes a diagnostic, even when it looks like a linker or GCC line.
    pytest.param(nvcc, '   21 |     printf("undefined reference to %d", name);', id="nvcc-gcc-echo-linker-phrase"),
    pytest.param(nvcc, '    5 |     const char* msg = "a.c:1:2: error: boom";', id="nvcc-gcc-echo-gcc-line"),
    pytest.param(nvcc, '12345 | x("main.cu(3): error: y");', id="nvcc-gcc-echo-edg-line"),
    pytest.param(nvcpp, '   21 |     printf("undefined reference to %d", name);', id="nvcpp-gcc-echo-linker-phrase"),
    pytest.param(nvcpp, '    5 |     const char* msg = "a.c:1:2: error: boom";', id="nvcpp-gcc-echo-gcc-line"),
    pytest.param(
        nvcc, "/usr/bin/ld: /tmp/tmpxft_00002d4c_00000000-11_main.o: in function `main':", id="nvcc-ld-context"
    ),
    pytest.param(nvcpp, '1 error detected in the compilation of "main.cpp".', id="nvcpp-error-summary"),
    pytest.param(
        nvcpp,
        'Remark: individual warnings can be suppressed with "--diag_suppress <warning-name>"',
        id="nvcpp-remark",
    ),
    pytest.param(nvcpp, "NVC++/x86-64 Linux PLACEHOLDER: compilation completed with severe errors", id="nvcpp-summary"),
    pytest.param(nvcpp, "saxpy(int, float, const float *, float *):", id="nvcpp-minfo-function"),
    pytest.param(nvcpp, "      6, #omp target teams distribute parallel for", id="nvcpp-minfo-region"),
    pytest.param(
        nvcpp, "     21, Accelerator restriction: size of the GPU copy of tmp is unknown", id="nvcpp-minfo-note"
    ),
    pytest.param(nvcpp, "         Generating map(tofrom:y[:n])", id="nvcpp-minfo-map"),
    pytest.param(nvcpp, "pgacclnk: child process exit status 1: /usr/bin/ld", id="nvcpp-link-status"),
]


@pytest.mark.parametrize(("module", "line"), SKIPPED_LINES)
def test_lines_matching_no_pattern_are_skipped(module: ModuleType, line: str) -> None:
    assert module.parse_diagnostics(line + "\n") == []
    assert module.parse_diagnostics(line) == []


@pytest.mark.parametrize("module", [nvcc, nvcpp], ids=["nvcc", "nvcpp"])
def test_a_gcc_warning_keeps_its_echo_out_of_the_diagnostics(module: ModuleType) -> None:
    # A build that exits 0 with this warning must not carry a false linker error from the echoed source.
    stderr = (
        "main.cu:21:37: warning: format '%d' expects argument of type 'int', but argument 2 has type "
        "'const char*' [-Wformat=]\n"
        '   21 |     printf("undefined reference to %d", name);\n'
        "      |                                    ~^\n"
        "      |                                     |\n"
        "      |                                     int\n"
    )
    message = "format '%d' expects argument of type 'int', but argument 2 has type 'const char*'"
    assert module.parse_diagnostics(stderr) == [compile_diag("warning", "-Wformat=", "main.cu", 21, 37, message)]


@pytest.mark.parametrize("module", [nvcc, nvcpp], ids=["nvcc", "nvcpp"])
def test_a_long_line_with_many_colons_parses_in_linear_time(module: ModuleType) -> None:
    # Model code can put one very long line in stderr. Both lines below took seconds with a pattern that
    # backtracked over every ": "; a linear parser takes milliseconds.
    no_phrase = "c ? a : " * 32_768
    with_phrase = "x: undefined reference to `y'" + ": a" * 40_000
    began = time.monotonic()
    assert module.parse_diagnostics(no_phrase + "\n") == []
    assert module.parse_diagnostics(with_phrase + "\n") == [
        compile_diag("error", None, None, None, None, with_phrase[len("x: ") :])
    ]
    assert time.monotonic() - began < 2.0


# ---------------------------------------------------------------------------
# The EDG echo, caret, and column rule


EDG_TOOLS = {
    "nvcc": (nvcc, "main.cu", "{file}({line}): {severity}: {message}"),
    "nvcpp": (nvcpp, "main.cpp", '"{file}", line {line}: {severity}: {message}'),
}


def edg_header(tool: str, line: int, message: str, severity: str = "error") -> str:
    """Return one EDG diagnostic line, with its newline, in the format of `tool` for that tool's main file."""
    _, file, pattern = EDG_TOOLS[tool]
    return pattern.format(file=file, line=line, severity=severity, message=message) + "\n"


def edg_parse(tool: str, stderr: str, text: str | None) -> list[Diagnostic]:
    """Parse `stderr` with the adapter of `tool`; `text`, when given, is the text of that tool's main file."""
    module, file, _ = EDG_TOOLS[tool]
    return module.parse_diagnostics(stderr, {} if text is None else {file: text})


def edg_error(tool: str, line: int, column: int | None, message: str) -> Diagnostic:
    """Return the expected error Diagnostic for an EDG line of `tool` in its main file."""
    return compile_diag("error", None, EDG_TOOLS[tool][1], line, column, message)


TOOLS = sorted(EDG_TOOLS)


@pytest.mark.parametrize("tool", TOOLS)
def test_column_uses_the_echo_prefix_length(tool: str) -> None:
    # The echo has a 4-space prefix here: caret index 12 - prefix 4 + 1 = column 9, the 'b' in 'int a = b;'.
    stderr = edg_header(tool, 2, 'identifier "b" is undefined') + "    int a = b;\n" + " " * 12 + "^\n"
    expected = [edg_error(tool, 2, 9, 'identifier "b" is undefined')]
    assert edg_parse(tool, stderr, "int x;\nint a = b;\n") == expected


@pytest.mark.parametrize("tool", TOOLS)
def test_column_ignores_trailing_whitespace_and_cr(tool: str) -> None:
    # A CRLF file and trailing blanks still align: caret index 10 - prefix 2 + 1 = column 9.
    stderr = edg_header(tool, 2, 'identifier "b" is undefined') + "  int a = b;   \n" + " " * 10 + "^\n"
    expected = [edg_error(tool, 2, 9, 'identifier "b" is undefined')]
    assert edg_parse(tool, stderr, "int x;\r\nint a = b; \t\r\n") == expected


@pytest.mark.parametrize("tool", TOOLS)
def test_column_is_none_when_the_echo_does_not_match_the_file(tool: str) -> None:
    stderr = edg_header(tool, 2, 'identifier "b" is undefined') + "  int a = b;\n" + " " * 10 + "^\n"
    expected = [edg_error(tool, 2, None, 'identifier "b" is undefined')]
    assert edg_parse(tool, stderr, "int x;\nint a = c;\n") == expected


@pytest.mark.parametrize("tool", TOOLS)
def test_column_is_none_past_the_end_of_the_file(tool: str) -> None:
    stderr = edg_header(tool, 9, 'identifier "b" is undefined') + "  int a = b;\n" + " " * 10 + "^\n"
    expected = [edg_error(tool, 9, None, 'identifier "b" is undefined')]
    assert edg_parse(tool, stderr, "int x;\nint a = b;\n") == expected


@pytest.mark.parametrize("tool", TOOLS)
def test_echo_and_caret_lines_are_consumed(tool: str) -> None:
    # 'nme' is a typo for 'name'. The echoed source line contains the linker pattern "undefined reference to",
    # so it must be consumed with the caret line and never become a diagnostic of its own.
    text = '#include <cstdio>\n\nvoid report(const char *name)\n{\n    printf("undefined reference to %s", nme);\n}\n'
    stderr = (
        edg_header(tool, 5, 'identifier "nme" is undefined')
        + '      printf("undefined reference to %s", nme);\n'
        + " " * 42
        + "^\n"
        + "\n"
    )
    assert edg_parse(tool, stderr, text) == [edg_error(tool, 5, 41, 'identifier "nme" is undefined')]
    assert edg_parse(tool, stderr, None) == [edg_error(tool, 5, None, 'identifier "nme" is undefined')]


@pytest.mark.parametrize("tool", TOOLS)
def test_diagnostics_without_echo_lines_are_all_kept(tool: str) -> None:
    stderr = edg_header(tool, 2, 'identifier "a" is undefined') + edg_header(tool, 3, 'identifier "b" is undefined')
    expected = [
        edg_error(tool, 2, None, 'identifier "a" is undefined'),
        edg_error(tool, 3, None, 'identifier "b" is undefined'),
    ]
    assert edg_parse(tool, stderr, "int x;\nint y = a;\nint z = b;\n") == expected


@pytest.mark.parametrize("tool", TOOLS)
def test_an_echo_of_a_blank_line_is_consumed_without_a_column(tool: str) -> None:
    stderr = edg_header(tool, 2, 'expected a "}"') + "\n" + "  ^\n" + "\n"
    assert edg_parse(tool, stderr, "int f() {\n\n") == [edg_error(tool, 2, None, 'expected a "}"')]


OVERLOAD = 'no instance of overloaded function "foo" matches the argument list'
OVERLOAD_SOURCE = "void foo(float *p);\nint main() {\n  int *p = 0;\n  foo(p);\n}\n"


@pytest.mark.parametrize("tool", TOOLS)
def test_continuation_lines_join_the_message_and_the_column_still_counts(tool: str) -> None:
    # EDG prints supplementary lines between the diagnostic and the echo; the echo prefix is 2 here,
    # so caret index 4 - 2 + 1 = column 3, the 'f' in '  foo(p);'.
    stderr = (
        edg_header(tool, 4, OVERLOAD)
        + "            argument types are: (int *)\n"
        + "            object type is: Bar\n"
        + "    foo(p);\n"
        + "    ^\n"
        + "\n"
    )
    message = f"{OVERLOAD}; argument types are: (int *); object type is: Bar"
    assert edg_parse(tool, stderr, OVERLOAD_SOURCE) == [edg_error(tool, 4, 3, message)]


@pytest.mark.parametrize("tool", TOOLS)
def test_continuation_lines_without_an_echo_still_join_the_message(tool: str) -> None:
    summary = '1 error detected in the compilation of "main.cu".\n'
    stderr = edg_header(tool, 4, OVERLOAD) + "            argument types are: (int *)\n" + "\n" + summary
    message = f"{OVERLOAD}; argument types are: (int *)"
    assert edg_parse(tool, stderr, OVERLOAD_SOURCE) == [edg_error(tool, 4, None, message)]


@pytest.mark.parametrize("tool", TOOLS)
def test_continuation_lines_stop_at_the_next_diagnostic(tool: str) -> None:
    stderr = (
        edg_header(tool, 4, OVERLOAD)
        + "            argument types are: (int *)\n"
        + edg_header(tool, 5, 'identifier "q" is undefined')
        + "  int p = q;\n"
        + "          ^\n"
    )
    assert edg_parse(tool, stderr, None) == [
        edg_error(tool, 4, None, f"{OVERLOAD}; argument types are: (int *)"),
        edg_error(tool, 5, None, 'identifier "q" is undefined'),
    ]


@pytest.mark.parametrize("tool", TOOLS)
def test_diagnostics_keep_stderr_order(tool: str) -> None:
    stderr = (
        edg_header(tool, 3, 'variable "u" was declared but never referenced', severity="warning")
        + "  int u;\n"
        + "      ^\n"
        + "\n"
        + edg_header(tool, 1, 'identifier "q" is undefined')
        + "  int p = q;\n"
        + "          ^\n"
    )
    file = EDG_TOOLS[tool][1]
    expected = [
        compile_diag("warning", None, file, 3, 5, 'variable "u" was declared but never referenced'),
        edg_error(tool, 1, 9, 'identifier "q" is undefined'),
    ]
    assert edg_parse(tool, stderr, "int p = q;\n\nint u;\n") == expected


# ---------------------------------------------------------------------------
# Command lines


def test_nvcc_command_line_is_exact() -> None:
    argv = nvcc.NvccSm80().command(["kernels/extra.cu", "main.cu"])
    assert type(argv) is list
    assert argv == [
        "nvcc",
        "-std=c++14",
        "-Xcompiler",
        "-Wall",
        "-arch=sm_80",
        "-O3",
        "-o",
        "main",
        "kernels/extra.cu",
        "main.cu",
    ]


def test_nvcpp_command_line_is_exact() -> None:
    argv = nvcpp.NvcppCc80().command(["main.cpp", "util/io.c"])
    assert type(argv) is list
    assert argv == ["nvc++", "-Wall", "-O3", "-Minfo", "-mp=gpu", "-gpu=cc80", "-o", "main", "main.cpp", "util/io.c"]


def test_executable_setting_replaces_only_the_first_word() -> None:
    assert nvcc.NvccSm80(executable="/opt/cuda-pin/bin/nvcc").command(["main.cu"]) == [
        "/opt/cuda-pin/bin/nvcc",
        *NVCC_FLAGS,
        "-o",
        OUTPUT,
        "main.cu",
    ]
    assert nvcpp.NvcppCc80(executable="/opt/nvhpc-pin/bin/nvc++").command(["main.cpp"]) == [
        "/opt/nvhpc-pin/bin/nvc++",
        *NVCPP_FLAGS,
        "-o",
        OUTPUT,
        "main.cpp",
    ]


def test_arch_and_gpu_class_attributes_drive_the_flags() -> None:
    class NvccOther(nvcc.NvccToolchain):
        """A preset with another arch, built only here and never registered."""

        name = "fixture-nvcc-other"
        ARCH = "sm_90"

    class NvcppOther(nvcpp.NvcppToolchain):
        """A preset with another GPU, built only here and never registered."""

        name = "fixture-nvcpp-other"
        GPU = "cc90"

    nvcc_argv = ["nvcc", "-std=c++14", "-Xcompiler", "-Wall", "-arch=sm_90", "-O3", "-o", "main", "main.cu"]
    nvcpp_argv = ["nvc++", "-Wall", "-O3", "-Minfo", "-mp=gpu", "-gpu=cc90", "-o", "main", "main.cpp"]
    assert NvccOther().command(["main.cu"]) == nvcc_argv
    assert NvcppOther().command(["main.cpp"]) == nvcpp_argv


def bible_compile_flags() -> list[str]:
    """Return the two compile commands from the bible's LASSI section, as written between backticks."""
    lines = BIBLE.read_text(encoding="utf-8").splitlines()
    start = lines.index("### LASSI")
    line = next(text for text in lines[start:] if text.startswith("Compile flags:"))
    return re.findall(r"`([^`]+)`", line)


def bible_repro_toolchains() -> dict[str, str]:
    """Return the `toolchain` binding of the bible's projects/lassi-repro/recipe.yaml block."""
    lines = BIBLE.read_text(encoding="utf-8").splitlines()
    start = lines.index("# projects/lassi-repro/recipe.yaml")
    end = lines.index("```", start)
    return yaml.safe_load("\n".join(lines[start:end]))["toolchain"]


def test_command_lines_carry_the_bible_compile_flags_unchanged() -> None:
    nvcc_words, nvcpp_words = (command.split() for command in bible_compile_flags())
    assert nvcc.NvccSm80().command(["main.cu"]) == [*nvcc_words, "-o", OUTPUT, "main.cu"]
    assert nvcpp.NvcppCc80().command(["main.cpp"]) == [*nvcpp_words, "-o", OUTPUT, "main.cpp"]


def test_the_repro_recipe_toolchains_are_the_two_presets() -> None:
    bindings = bible_repro_toolchains()
    assert bindings == {"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"}
    assert DEFAULT_REGISTRY.get("Toolchain", bindings["cuda"]).factory is nvcc.NvccSm80
    assert DEFAULT_REGISTRY.get("Toolchain", bindings["omp"]).factory is nvcpp.NvcppCc80


# ---------------------------------------------------------------------------
# Registration and construction


@dataclass(frozen=True)
class Preset:
    """One registered preset and what the build tests expect of it."""

    name: str
    factory: type
    capabilities: frozenset[str]
    executable: str
    flags: tuple[str, ...]
    suffixes: tuple[str, ...]
    tool_names: tuple[str, ...]  # any of these may name the tool in the exit-status message
    main: str
    header: str
    error_fixture: str
    warning_fixture: str


PRESETS = {
    "nvcc-sm80": Preset(
        name="nvcc-sm80",
        factory=nvcc.NvccSm80,
        capabilities=frozenset({"cuda", "emits_warnings", "diagnostics"}),
        executable="nvcc",
        flags=NVCC_FLAGS,
        suffixes=NVCC_SUFFIXES,
        tool_names=("nvcc",),
        main="main.cu",
        header="kernels/scale.cuh",
        error_fixture="nvcc_undefined_identifier",
        warning_fixture="nvcc_warning_177",
    ),
    "nvcpp-cc80": Preset(
        name="nvcpp-cc80",
        factory=nvcpp.NvcppCc80,
        capabilities=frozenset({"openmp_offload", "emits_warnings", "diagnostics"}),
        executable="nvc++",
        flags=NVCPP_FLAGS,
        suffixes=NVCPP_SUFFIXES,
        tool_names=("nvc++", "nvcpp-cc80"),
        main="main.cpp",
        header="kernels/scale.h",
        error_fixture="nvcpp_edg_error",
        warning_fixture="nvcpp_edg_warning",
    ),
}
PRESET_NAMES = sorted(PRESETS)


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_preset_is_registered_with_its_capabilities(preset: str) -> None:
    case = PRESETS[preset]
    entry = DEFAULT_REGISTRY.get("Toolchain", case.name)
    assert entry.factory is case.factory
    assert entry.capabilities == case.capabilities
    assert case.factory.name == case.name
    assert case.factory.capabilities == case.capabilities
    assert case.name in DEFAULT_REGISTRY.names("Toolchain")


def test_presets_subclass_the_adapters() -> None:
    assert issubclass(nvcc.NvccSm80, nvcc.NvccToolchain)
    assert nvcc.NvccSm80.ARCH == "sm_80"
    assert hasattr(nvcc.NvccToolchain, "ARCH")
    assert issubclass(nvcpp.NvcppCc80, nvcpp.NvcppToolchain)
    assert nvcpp.NvcppCc80.GPU == "cc80"
    assert hasattr(nvcpp.NvcppToolchain, "GPU")


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_factory_with_no_arguments_builds_the_preset(preset: str) -> None:
    # Toolchain bindings carry no config, so the runner builds them as factory().
    case = PRESETS[preset]
    tool = DEFAULT_REGISTRY.get("Toolchain", case.name).factory()
    assert type(tool) is case.factory
    assert isinstance(tool, Component)
    assert tool.name == case.name
    assert tool.command(["x.c"]) == [case.executable, *case.flags, "-o", OUTPUT, "x.c"]
    assert list(inspect.signature(case.factory.build).parameters) == ["self", "files", "workdir"]


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_factory_with_no_arguments_builds_through_subprocess_runner(
    preset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # runner=None means subprocess_runner; a fake stands in for it here, so no compiler starts.
    case = PRESETS[preset]
    fake = FakeRunner(creates_output=True)
    monkeypatch.setattr("lassi.toolchains._base.subprocess_runner", fake)
    workdir = make_workdir(tmp_path)
    result = DEFAULT_REGISTRY.get("Toolchain", case.name).factory().build({case.main: "int x;\n"}, workdir)
    assert [(call.argv, call.cwd, call.timeout_s) for call in fake.calls] == [
        ([case.executable, *case.flags, "-o", OUTPUT, case.main], workdir, 600.0)
    ]
    assert result == BuildResult(artifact=workdir / OUTPUT, diagnostics=[], stderr_ref=ATTACHMENT)


# ---------------------------------------------------------------------------
# build with a fake runner


@dataclass(frozen=True)
class Call:
    """One call the fake runner received, with the workdir's files (relative path -> bytes) at that moment."""

    argv: list[str]
    cwd: Path
    timeout_s: float
    files: dict[str, bytes]


@dataclass
class FakeRunner:
    """A CommandRunner that records each call and returns canned output; it never starts a process.

    With `creates_output` set it writes the artifact file into the workdir,
    the way a compiler that got far enough would.
    """

    returncode: int = 0
    stderr: str = ""
    stdout: str = ""
    creates_output: bool = False
    calls: list[Call] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the call, optionally write the artifact, and return the canned CommandResult."""
        root = Path(cwd)
        snapshot = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        self.calls.append(Call(argv=list(argv), cwd=root, timeout_s=timeout_s, files=snapshot))
        if self.creates_output:
            (root / OUTPUT).write_bytes(b"\x7fELF placeholder artifact written by the fake runner")
        return CommandResult(returncode=self.returncode, stdout=self.stdout, stderr=self.stderr)


def make_workdir(tmp_path: Path) -> Path:
    """Return an empty build directory inside the test's temporary directory."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    return workdir


def exit_status_check(case: Preset, diagnostic: Diagnostic, status: str) -> None:
    """Assert that `diagnostic` is the exit-status fallback error naming the tool and `status`."""
    assert (diagnostic.stage, diagnostic.severity, diagnostic.code) == ("compile", "error", "exit-status")
    assert (diagnostic.file, diagnostic.line, diagnostic.column) == (None, None, None)
    assert any(name in diagnostic.message for name in case.tool_names), diagnostic.message
    assert status in diagnostic.message, diagnostic.message


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_build_writes_every_file_exactly_before_running(preset: str, tmp_path: Path) -> None:
    case = PRESETS[preset]
    files = {
        case.main: "int main() {\r\n  return 0;\r\n}\r\n",
        case.header: f"// caf{E_ACUTE}\n#define N 4\n",
        "empty.h": "",
        "notes/no_final_newline.txt": "last line",
    }
    runner = FakeRunner(creates_output=True)
    workdir = make_workdir(tmp_path)
    case.factory(runner=runner).build(files, workdir)
    expected = {path: text.encode("utf-8") for path, text in files.items()}
    assert len(runner.calls) == 1
    assert runner.calls[0].files == expected
    for path, raw in expected.items():
        assert (workdir / path).read_bytes() == raw, path


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_build_passes_only_sources_in_sorted_order(preset: str, tmp_path: Path) -> None:
    case = PRESETS[preset]
    names = ["z.cu", "b.cpp", "src/k/m.cuh", "c.cc", "g.h", "d.cxx", "Makefile", "e.c", "h.hpp", "src/k/m.cu"]
    names += ["notes.txt", "src/k/a.cpp", "f.cuh"]
    files = {name: "int x;\n" for name in names}
    runner = FakeRunner(creates_output=True)
    case.factory(runner=runner).build(files, make_workdir(tmp_path))
    sources = sorted(name for name in names if name.endswith(case.suffixes))
    assert sources == (
        ["b.cpp", "c.cc", "d.cxx", "e.c", "src/k/a.cpp", "src/k/m.cu", "z.cu"]
        if preset == "nvcc-sm80"
        else ["b.cpp", "c.cc", "d.cxx", "e.c", "src/k/a.cpp"]
    )
    assert runner.calls[0].argv == [case.executable, *case.flags, "-o", OUTPUT, *sources]


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_build_runs_in_the_workdir_with_the_timeout(preset: str, tmp_path: Path) -> None:
    case = PRESETS[preset]
    default_runner, set_runner = FakeRunner(), FakeRunner()
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    case.factory(runner=default_runner).build({case.main: "int x;\n"}, first)
    pinned = case.factory(executable="/pinned/bin/tool", runner=set_runner, timeout_s=42.5)
    pinned.build({case.main: "int x;\n"}, second)
    assert (default_runner.calls[0].cwd, default_runner.calls[0].timeout_s) == (first, 600.0)
    assert (set_runner.calls[0].cwd, set_runner.calls[0].timeout_s) == (second, 42.5)
    assert set_runner.calls[0].argv == ["/pinned/bin/tool", *case.flags, "-o", OUTPUT, case.main]


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_build_keeps_raw_stderr_byte_for_byte_as_the_attachment(preset: str, tmp_path: Path) -> None:
    case = PRESETS[preset]
    stderr = f"first line\r\nsecond line caf{E_ACUTE}\n\n\tlast line without newline"
    workdir = make_workdir(tmp_path)
    runner = FakeRunner(stderr=stderr, stdout="STDOUT-MARKER\n", creates_output=True)
    result = case.factory(runner=runner).build({case.main: "int x;\n"}, workdir)
    assert toolchains.STDERR_ATTACHMENT == ATTACHMENT
    assert result.stderr_ref == ATTACHMENT
    assert (workdir / ATTACHMENT).read_bytes() == stderr.encode("utf-8")  # stderr only, never stdout
    assert "STDOUT-MARKER" not in repr(result)


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_build_parses_diagnostics_with_the_files_and_keeps_stderr_out(preset: str, tmp_path: Path) -> None:
    case = PRESETS[preset]
    fixture = FIXTURE_CASES[case.error_fixture]
    stderr = fixture_text(case.error_fixture)
    workdir = make_workdir(tmp_path)
    result = case.factory(runner=FakeRunner(returncode=1, stderr=stderr)).build(fixture.files, workdir)
    assert isinstance(result, BuildResult)
    assert [f.name for f in dataclasses.fields(result)] == ["artifact", "diagnostics", "stderr_ref"]
    assert result.diagnostics == fixture.expected  # the exact columns show the files reached the parser
    assert result.artifact is None
    assert result.stderr_ref == ATTACHMENT
    assert (workdir / ATTACHMENT).read_bytes() == stderr.encode("utf-8")
    # Lines the parser skips (the source echo and the summary) appear nowhere in the result.
    shown = repr(result)
    assert "undefined_var;" not in shown
    assert "detected in the compilation" not in shown
    assert all(stderr not in diagnostic.message for diagnostic in result.diagnostics)


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_clean_build_returns_the_artifact_and_an_empty_attachment(preset: str, tmp_path: Path) -> None:
    case = PRESETS[preset]
    workdir = make_workdir(tmp_path)
    result = case.factory(runner=FakeRunner(creates_output=True)).build({case.main: "int x;\n"}, workdir)
    assert result == BuildResult(artifact=workdir / OUTPUT, diagnostics=[], stderr_ref=ATTACHMENT)
    assert (workdir / ATTACHMENT).read_bytes() == b""


@pytest.mark.parametrize(
    ("returncode", "creates_output", "has_artifact", "codes"),
    [
        (0, True, True, []),
        (0, False, False, ["no-artifact"]),
        (1, True, False, ["exit-status"]),
        (-1, True, False, ["exit-status"]),
    ],
    ids=["success", "success-without-file", "failure-with-file", "timeout-with-file"],
)
@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_artifact_only_on_status_zero_when_the_file_exists(
    preset: str, returncode: int, creates_output: bool, has_artifact: bool, codes: list[str], tmp_path: Path
) -> None:
    # A build without an artifact failed, so it always carries an error.
    case = PRESETS[preset]
    workdir = make_workdir(tmp_path)
    runner = FakeRunner(returncode=returncode, creates_output=creates_output)
    result = case.factory(runner=runner).build({case.main: "int x;\n"}, workdir)
    assert result.artifact == (workdir / OUTPUT if has_artifact else None)
    assert [diagnostic.code for diagnostic in result.diagnostics] == codes
    assert all(diagnostic.severity == "error" for diagnostic in result.diagnostics)


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_status_zero_without_an_artifact_is_an_error(preset: str, tmp_path: Path) -> None:
    case = PRESETS[preset]
    result = case.factory(runner=FakeRunner(returncode=0)).build({case.main: "int x;\n"}, make_workdir(tmp_path))
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert (diagnostic.stage, diagnostic.severity, diagnostic.code) == ("compile", "error", "no-artifact")
    assert (diagnostic.file, diagnostic.line, diagnostic.column) == (None, None, None)
    assert any(name in diagnostic.message for name in case.tool_names), diagnostic.message
    assert f"'{OUTPUT}'" in diagnostic.message, diagnostic.message
    assert (result.artifact, result.stderr_ref) == (None, ATTACHMENT)


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_a_stale_artifact_is_removed_before_the_compiler_runs(preset: str, tmp_path: Path) -> None:
    # Only a program the compiler wrote in this build can become the artifact.
    case = PRESETS[preset]
    workdir = make_workdir(tmp_path)
    (workdir / OUTPUT).write_bytes(b"stale program from an earlier attempt")
    runner = FakeRunner(returncode=0)
    result = case.factory(runner=runner).build({case.main: "int x;\n"}, workdir)
    assert OUTPUT not in runner.calls[0].files
    assert not (workdir / OUTPUT).exists()
    assert result.artifact is None
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["no-artifact"]


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_warnings_on_success_add_no_error(preset: str, tmp_path: Path) -> None:
    case = PRESETS[preset]
    fixture = FIXTURE_CASES[case.warning_fixture]
    runner = FakeRunner(stderr=fixture_text(case.warning_fixture), creates_output=True)
    result = case.factory(runner=runner).build(fixture.files, make_workdir(tmp_path))
    assert result.diagnostics == fixture.expected
    assert result.artifact is not None


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_exit_status_fallback_when_stderr_is_empty(preset: str, tmp_path: Path) -> None:
    case = PRESETS[preset]
    result = case.factory(runner=FakeRunner(returncode=137)).build({case.main: "int x;\n"}, make_workdir(tmp_path))
    assert len(result.diagnostics) == 1
    exit_status_check(case, result.diagnostics[0], "137")
    assert result.artifact is None


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_exit_status_fallback_follows_parsed_warnings(preset: str, tmp_path: Path) -> None:
    # A failed build never has an empty error list, even when stderr held only warnings.
    case = PRESETS[preset]
    fixture = FIXTURE_CASES[case.warning_fixture]
    runner = FakeRunner(returncode=2, stderr=fixture_text(case.warning_fixture))
    result = case.factory(runner=runner).build(fixture.files, make_workdir(tmp_path))
    assert result.diagnostics[:-1] == fixture.expected
    exit_status_check(case, result.diagnostics[-1], "2")


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_no_exit_status_error_when_an_error_was_parsed(preset: str, tmp_path: Path) -> None:
    case = PRESETS[preset]
    fixture = FIXTURE_CASES[case.error_fixture]
    runner = FakeRunner(returncode=2, stderr=fixture_text(case.error_fixture))
    result = case.factory(runner=runner).build(fixture.files, make_workdir(tmp_path))
    assert result.diagnostics == fixture.expected
    assert all(diagnostic.code != "exit-status" for diagnostic in result.diagnostics)


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_timeout_fallback_says_timed_out(preset: str, tmp_path: Path) -> None:
    case = PRESETS[preset]
    runner = FakeRunner(returncode=-1, stderr="partial output\ntimed out after 600 s\n")
    result = case.factory(runner=runner).build({case.main: "int x;\n"}, make_workdir(tmp_path))
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert (diagnostic.stage, diagnostic.severity, diagnostic.code) == ("compile", "error", "exit-status")
    assert "timed out" in diagnostic.message, diagnostic.message
    assert any(name in diagnostic.message for name in case.tool_names), diagnostic.message
    assert result.artifact is None


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_a_timeout_is_reported_even_after_a_parsed_error(preset: str, tmp_path: Path) -> None:
    # The model should learn that the compile was stopped, not only see the errors printed before it.
    case = PRESETS[preset]
    fixture = FIXTURE_CASES[case.error_fixture]
    stderr = fixture_text(case.error_fixture) + "timed out after 600 s\n"
    runner = FakeRunner(returncode=-1, stderr=stderr)
    result = case.factory(runner=runner).build(fixture.files, make_workdir(tmp_path))
    assert result.diagnostics[:-1] == fixture.expected
    last = result.diagnostics[-1]
    assert (last.code, last.severity) == ("exit-status", "error")
    assert "timed out" in last.message, last.message


def test_a_missing_header_reaches_the_model_as_a_located_error(tmp_path: Path) -> None:
    # nvc++ reports a missing header as an EDG catastrophic error; it replaces the exit-status fallback.
    fixture = FIXTURE_CASES["nvcpp_missing_include"]
    runner = FakeRunner(returncode=2, stderr=fixture_text("nvcpp_missing_include"))
    result = nvcpp.NvcppCc80(runner=runner).build(fixture.files, make_workdir(tmp_path))
    assert result.diagnostics == fixture.expected
    assert result.artifact is None


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_no_sources_returns_an_error_without_running(preset: str, tmp_path: Path) -> None:
    case = PRESETS[preset]
    files = {case.header: "#pragma once\n", "README.md": "notes\n", "legacy.cuh": "int x;\n"}
    if preset == "nvcpp-cc80":
        files["legacy.cu"] = "__global__ void k() {}\n"  # .cu is not an nvc++ source
    runner = FakeRunner()
    workdir = make_workdir(tmp_path)
    result = case.factory(runner=runner).build(files, workdir)
    assert runner.calls == []
    assert (result.artifact, result.stderr_ref) == (None, "")
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    fields = (diagnostic.stage, diagnostic.severity, diagnostic.code, diagnostic.file, diagnostic.line)
    assert fields == ("compile", "error", "no-sources", None, None)
    assert diagnostic.column is None
    missing = [suffix for suffix in case.suffixes if suffix not in diagnostic.message]
    assert not missing, diagnostic.message
    assert not (workdir / ATTACHMENT).exists()
    for path, text in files.items():
        assert (workdir / path).read_bytes() == text.encode("utf-8"), path


# Refused paths; "{tmp}" becomes the test's temporary directory, so a wrongly written file stays inside it.
BAD_BUILD_PATHS = [
    pytest.param("../escape.cu", id="dotdot"),
    pytest.param("sub/../../escape2.cu", id="dotdot-nested"),
    pytest.param("{tmp}/absolute.cu", id="absolute"),
    pytest.param("src\\main.cu", id="backslash"),
    pytest.param("./main.cu", id="dot-slash"),
    pytest.param("src//main.cu", id="empty-segment"),
    pytest.param("", id="empty"),
    # A source path starting with "-" or "@" would reach the compiler command line as an option.
    pytest.param("-optf=a.txt,b.c", id="nvcc-options-file"),
    pytest.param("-x.cu", id="leading-dash"),
    pytest.param("@rsp.cpp", id="response-file"),
    pytest.param("-Xcompiler=-specs=evil.c", id="host-compiler-flag"),
]


@pytest.mark.parametrize("template", BAD_BUILD_PATHS)
@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_bad_path_raises_before_anything_is_written(preset: str, template: str, tmp_path: Path) -> None:
    case = PRESETS[preset]
    bad = template.replace("{tmp}", tmp_path.as_posix())
    files = {"a.cu": "int a;\n", "a.cpp": "int a;\n", bad: "int bad;\n", "z.c": "int z;\n"}
    runner = FakeRunner(creates_output=True)
    workdir = make_workdir(tmp_path)
    with pytest.raises(ValueError):
        case.factory(runner=runner).build(files, workdir)
    assert runner.calls == []
    assert sorted(workdir.rglob("*")) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["work"]


# File sets whose paths each pass the path check but that the workdir cannot hold, with the paths refused.
UNWRITABLE = [
    pytest.param({"kernels": "x\n", "kernels/a.cuh": "y\n"}, ["kernels"], id="file-then-directory"),
    pytest.param({"kernels/a.cuh": "y\n", "kernels": "x\n"}, ["kernels"], id="directory-then-file"),
    pytest.param({"a/b": "x\n", "a/b/c/d.h": "y\n"}, ["a/b"], id="nested-directory"),
    pytest.param({"main": "x\n"}, ["main"], id="output-name"),
    pytest.param({"main/x.h": "x\n"}, ["main/x.h"], id="output-name-as-directory"),
    pytest.param({"compile.stderr": "x\n"}, ["compile.stderr"], id="attachment-name"),
    pytest.param({"compile.stderr/a.h": "x\n"}, ["compile.stderr/a.h"], id="attachment-name-as-directory"),
    pytest.param({"main/x.h": "x\n", "main": "y\n"}, ["main", "main/x.h"], id="several"),
]


@pytest.mark.parametrize(("extra", "refused"), UNWRITABLE)
@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_a_file_set_the_workdir_cannot_hold_is_a_bad_path_error(
    preset: str, extra: dict[str, str], refused: list[str], tmp_path: Path
) -> None:
    # The model named these files, so the problem goes back to it as an error; nothing is written or run.
    case = PRESETS[preset]
    files = {case.main: "int x;\n", **extra}
    runner = FakeRunner(creates_output=True)
    workdir = make_workdir(tmp_path)
    result = case.factory(runner=runner).build(files, workdir)
    assert runner.calls == []
    assert sorted(workdir.rglob("*")) == []
    assert (result.artifact, result.stderr_ref) == (None, "")
    fields = [(d.stage, d.severity, d.code, d.file, d.line, d.column) for d in result.diagnostics]
    assert fields == [("compile", "error", "bad-path", path, None, None) for path in refused]
    for diagnostic in result.diagnostics:
        assert repr(diagnostic.file) in diagnostic.message, diagnostic.message
        assert "Rename" in diagnostic.message, diagnostic.message


# ---------------------------------------------------------------------------
# The package: CommandResult, STDERR_ATTACHMENT, subprocess_runner


def test_command_result_is_a_frozen_record() -> None:
    result = CommandResult(returncode=0, stdout="o", stderr="e")
    assert [f.name for f in dataclasses.fields(CommandResult)] == ["returncode", "stdout", "stderr"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.returncode = 1  # type: ignore[misc]


def test_stderr_attachment_name() -> None:
    assert toolchains.STDERR_ATTACHMENT == "compile.stderr"


def test_subprocess_runner_captures_status_and_utf8_output(tmp_path: Path) -> None:
    code = (
        "import os, sys; sys.stdout.write(os.getcwd()); "
        "sys.stderr.buffer.write(b'caf\\xc3\\xa9 bad\\xff'); sys.stderr.flush(); sys.exit(3)"
    )
    result = subprocess_runner((sys.executable, "-c", code), tmp_path, 60.0)
    assert isinstance(result, CommandResult)
    assert result.returncode == 3
    assert Path(result.stdout).samefile(tmp_path)
    assert result.stderr == f"caf{E_ACUTE} bad\N{REPLACEMENT CHARACTER}"


def test_subprocess_runner_passes_argv_without_a_shell(tmp_path: Path) -> None:
    literal = "a b;c|d %PATH% $HOME `x` & e"
    code = "import sys; sys.stdout.write(sys.argv[1])"
    result = subprocess_runner([sys.executable, "-c", code, literal], tmp_path, 60.0)
    assert result == CommandResult(returncode=0, stdout=literal, stderr="")


def test_subprocess_runner_timeout_keeps_partial_stderr(tmp_path: Path) -> None:
    code = "import sys, time; sys.stderr.buffer.write(b'started\\n'); sys.stderr.flush(); time.sleep(60)"
    began = time.monotonic()
    result = subprocess_runner([sys.executable, "-c", code], tmp_path, 2.0)
    assert time.monotonic() - began < 30
    assert result.returncode == -1
    lines = result.stderr.rstrip("\n").split("\n")
    assert lines[0] == "started", result.stderr
    assert re.fullmatch(r"timed out after 2(\.0)? s", lines[-1]), result.stderr


def test_subprocess_runner_timeout_stops_the_processes_the_command_started(tmp_path: Path) -> None:
    # A compiler driver starts its stages as child processes that share its stderr. Here the child would
    # write a marker file 4 s in; the runner must return at its timeout and take the child down with it.
    marker = tmp_path / "late.txt"
    child = f"import time; time.sleep(4); open({marker.as_posix()!r}, 'w').write('late')"
    code = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "sys.stderr.write('started\\n'); sys.stderr.flush(); time.sleep(60)"
    )
    began = time.monotonic()
    result = subprocess_runner([sys.executable, "-c", code], tmp_path, 2.0)
    assert time.monotonic() - began < 3.5
    assert result.returncode == -1
    lines = result.stderr.rstrip("\n").split("\n")
    assert lines[0].rstrip("\r") == "started", result.stderr
    assert re.fullmatch(r"timed out after 2(\.0)? s", lines[-1]), result.stderr
    time.sleep(max(0.0, began + 5.5 - time.monotonic()))
    assert not marker.exists()
