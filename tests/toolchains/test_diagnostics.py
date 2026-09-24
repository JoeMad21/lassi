"""Tests for the nvcc and nvc++ toolchain adapters in lassi/toolchains/ (P0.5, P0.15, P0.17).

A Toolchain turns files into an artifact plus diagnostics parsed into records
of (severity, code, file, line, column, message, stage); the raw stderr is kept
as an attachment and never consumed downstream (bible Component Interfaces,
Toolchain row and contract rules; Result Record, Diagnostic). The two presets
are the toolchain bindings of the bible's lassi-repro recipe, `nvcc-sm80` and
`nvcpp-cc80`, and their command lines carry the LASSI compile flags from the
bible's Source Papers section unchanged.

The .stderr files in tests/toolchains/fixtures/ are raw stderr captured on the
build host with the pinned compilers under LC_ALL=C, one scenario each, copied
byte for byte; captures.json there is the capture's provenance manifest, and
fixtures/sources/<scenario>/ holds the files each scenario compiled (see the
README there). Each expected Diagnostic list below is derived by hand from the
raw stderr and the Toolchain contract, never copied from a parser's output;
the comment above each case gives the reasoning. The one-pattern sample lines
either come from a capture or say that no capture shows their format yet.

P0.17 adds the ptxas place format (a PTX file and line) and the nvlink
format, with fixtures nvcc_ptxas_inline_asm and nvcpp_nvlink_error from the
clean-commit capture that all 16 fixtures come from (rx
20260923-211958-desktop-8r113ei-p0-core-d221). It also adds the nvc++ driver
format (nvc++-Error-, nvc++-Fatal-), which no scenario produces without
crashing a compiler; its line tests use the lines on record
(NVCPP_DRIVER_ERROR, NVCPFE_CRASH_STDERR).

No test here runs a compiler: parse_diagnostics reads the fixtures, build gets
a fake command runner that records its call and returns canned stderr, and
subprocess_runner runs the current Python interpreter only. No value in this
module is a measurement; the captures record compiler output, not performance.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

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
SOURCES = FIXTURES / "sources"
CAPTURES = FIXTURES / "captures.json"
OUTPUT = "main"
ATTACHMENT = "compile.stderr"
E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"

NVCC_FLAGS = ("-std=c++14", "-Xcompiler", "-Wall", "-arch=sm_80", "-O3")
NVCPP_FLAGS = ("-Wall", "-O3", "-Minfo", "-mp=gpu", "-gpu=cc80")
NVCC_SUFFIXES = (".cu", ".cpp", ".cc", ".cxx", ".c")
NVCPP_SUFFIXES = (".cpp", ".cc", ".cxx", ".c")


def scenario_files(name: str) -> dict[str, str]:
    """Return the files scenario `name` compiled: fixtures/sources/<name>/, relative POSIX path -> text.

    The text is the file's bytes decoded as UTF-8 with no newline translation,
    as build() writes it.
    """
    tree = SOURCES / name
    paths = sorted(path for path in tree.rglob("*") if path.is_file())
    return {path.relative_to(tree).as_posix(): path.read_bytes().decode("utf-8") for path in paths}


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
    """One captured stderr fixture: the adapter module that parses it and the Diagnostics it must give.

    The files it is parsed with are its scenario's source tree (scenario_files).
    `without_files`, when set, is what it must give with no file text, where
    more than the columns change; by default only every column becomes None.
    """

    module: ModuleType
    expected: list[Diagnostic]
    without_files: list[Diagnostic] | None = None


SIGN_COMPARE = (
    "comparison of integer expressions of different signedness: 'int' and 'std::vector<float>::size_type' "
    "{aka 'long unsigned int'}"
)
HELPER_UNDEFINED = "undefined reference to `helper(float*, int)'"
VTABLE_UNDEFINED = "undefined reference to `vtable for Foo'"
STACK_LIMIT = "The maximum stack size for a GPU kernel or procedure is limited to 524288 bytes: 1048676"
# As printed, with the four blanks before the internal number; only the blanks before "(main.cpp: 8)" go.
TINFO_ICE = "Internal compiler error. child tinfo should have been created at outlining function for host    1198"

# P0.17. The sample lines of the ptxas place and nvlink formats, copied from fixtures nvcc_ptxas_inline_asm and
# nvcpp_nvlink_error (clean-commit capture rx 20260923-211958-desktop-8r113ei-p0-core-d221);
# test_the_p017_sample_lines_are_lines_of_their_fixtures checks each against its fixture. Each names that run's
# absolute workdir, so a recapture changes them together with the fixtures.
# The temporary .ptx that nvcc wrote in the compile's private TMPDIR, and the temporary object nvc++ gave nvlink.
PTX_FILE = (
    "/mnt/nvme10/joseph_ufl/lassi-runs/fixture-captures/20260923-211958-desktop-8r113ei-p0-core-d221/work/"
    "nvcc_ptxas_inline_asm/@lassi-tmp/tmpxft_00000002_00000000-6_main.ptx"
)
NVLINK_OBJECT = (
    "/mnt/nvme10/joseph_ufl/lassi-runs/fixture-captures/20260923-211958-desktop-8r113ei-p0-core-d221/work/"
    "nvcpp_nvlink_error/@lassi-tmp/nvc++bcdFQVDqP8.o"
)
PTXAS_PLACE_LINE = "ptxas " + PTX_FILE + ", line 28; error   : Unknown modifier '.bogus'"
PTXAS_ABORTED_LINE = "ptxas fatal   : Ptx assembly aborted due to errors"
NVLINK_LINE = "nvlink error   : Undefined reference to '_Z5twicef' in '" + NVLINK_OBJECT + "'"
NVDD_STATUS_LINE = (
    "pgacclnk: child process exit status 2: "
    "/mnt/nvme10/joseph_ufl/toolchains/nvhpc@24.11/Linux_x86_64/24.11/compilers/bin/tools/nvdd"
)
# The ptxas place names the temporary .ptx and a line of it, which index no built file. So file and line are None,
# and the message keeps the place as printed in front of ptxas's message. The ": " stands where ptxas printed
# "; error   : ", because the severity has its own field.
PTXAS_PLACE_MESSAGE = PTX_FILE + ", line 28: Unknown modifier '.bogus'"
# nvlink names the symbol and a temporary object, no source line. The message is the text after "nvlink error   : ",
# as printed, the object path included.
NVLINK_MESSAGE = "Undefined reference to '_Z5twicef' in '" + NVLINK_OBJECT + "'"
# The shortened form of the .ptx path for the sample lines that no capture shows (the workdir becomes /w).
SHORT_PTX = "/w/@lassi-tmp/tmpxft_00000002_00000000-6_main.ptx"
# nvcc names its PTX after the source stem, which the model chooses. These stems hold an EDG and a GCC style place.
EDG_STEM_PTX = "/w/@lassi-tmp/tmpxft_00000002_00000000-6_x(3): error: y.ptx"
GCC_STEM_PTX = "/w/@lassi-tmp/tmpxft_00000002_00000000-6_ab:1:2: error: b.ptx"

# P0.17, the nvc++ driver format (nvc++-Error-, nvc++-Fatal-). No scenario produces it from sources with the preset
# flags without crashing a compiler, so its line tests use the two lines on record, copied verbatim. The first is
# from a clean-commit run, plans/spikes/p0-toolchains-verify.md run 2a (rx
# 20260923-045538-desktop-8r113ei-p0-core-bde2), before the CUDA home was set. The second is the whole stderr (200
# bytes) of scenario nvcpp_backend_error in the P0.15 exploratory dirty-tree probe rx
# 20260923-104313-desktop-8r113ei-p0-core-2b22, whose source of that time crashed the front end by accident. That
# probe is exploratory and never [MEASURED].
NVCPP_DRIVER_ERROR = (
    "nvc++-Error-A CUDA toolkit matching the current driver version (0) or a supported older version (11.8) was not "
    "installed with this HPC SDK."
)
NVCPFE = "/mnt/nvme10/joseph_ufl/toolchains/nvhpc@24.11/Linux_x86_64/24.11/compilers/bin/tools/nvcpfe"
NVCPFE_CRASH_STDERR = (
    "NVC++-S-0155-for must have the ordered clause specified  (main.cpp: 32)\n"
    "nvc++-Fatal-" + NVCPFE + " TERMINATED by signal 11\n"
)
# The Diagnostics of NVCPFE_CRASH_STDERR: the backend error keeps its place as printed, and the driver fatal is an
# error with no place whose message is the text after "nvc++-Fatal-", the toolchain path included.
NVCPFE_CRASH_DIAGNOSTICS = [
    compile_diag("error", "S-0155", "main.cpp", 32, None, "for must have the ordered clause specified"),
    compile_diag("error", None, None, None, None, NVCPFE + " TERMINATED by signal 11"),
]

FIXTURE_CASES: dict[str, FixtureCase] = {
    # main.cu line 7 is "        y[i] = a * x[i] + undefined_var;". The echo adds 2 blanks in front and the
    # caret is at index 28, so the column is 28 - 2 + 1 = 27, the 'u'. nvcc prints no number for an EDG error,
    # so there is no code. The blank line and the "1 error detected" summary are no diagnostics.
    "nvcc_undefined_identifier": FixtureCase(
        nvcc, [compile_diag("error", None, "main.cu", 7, 27, 'identifier "undefined_var" is undefined')]
    ),
    # kernels/scale.cuh line 5 is "    float unused = 0.0f;": caret index 12 - echo prefix 2 + 1 = column 11, the
    # 'u'. The code is the number with its "-D". The Remark line and the blank lines are no diagnostics, and
    # the warning appears once.
    "nvcc_warning_177": FixtureCase(
        nvcc,
        [
            compile_diag(
                "warning", "177-D", "kernels/scale.cuh", 5, 11, 'variable "unused" was declared but never referenced'
            )
        ],
    ),
    # GCC names main.cu:16:19 with the flag [-Wsign-compare] as the code. Its echo is line 16 of main.cu as the
    # file holds it, but GCC counted column 19 in the host code cudafe1 regenerated, where the line has no
    # indent: there the '<' is column 19, in main.cu it is column 23, and column 19 of main.cu is the ';'. A
    # column must index the named file, and nothing in stderr shows which text GCC counted in, so under nvcc
    # a GCC column is always None; file, line, flag, and message stay. The "In function" line and the echo
    # and caret lines are no diagnostics. The build exited 0.
    "nvcc_host_gcc_warning": FixtureCase(
        nvcc, [compile_diag("warning", "-Wsign-compare", "main.cu", 16, None, SIGN_COMPARE)]
    ),
    # The driver names no file; the message is the text after "nvcc fatal   : ".
    "nvcc_fatal": FixtureCase(
        nvcc,
        [compile_diag("error", None, None, None, None, "Value 'sm_35' is not defined for option 'gpu-architecture'")],
    ),
    # ptxas names the mangled kernel but no file or line; the message is the text after "ptxas error   : ".
    "nvcc_ptxas_error": FixtureCase(
        nvcc,
        [
            compile_diag(
                "error",
                None,
                None,
                None,
                None,
                "Entry function '_Z6reducePKfPfi' uses too much shared data (0x40000 bytes, 0x29000 max)",
            )
        ],
    ),
    # P0.17. Two lines. The first is "ptxas <PTX_FILE>, line 28; error   : Unknown modifier '.bogus'": ptxas names
    # the temporary main.ptx nvcc wrote under @lassi-tmp, the compile's private TMPDIR, and line 28 of it. That file
    # is nvcc's own PTX, never a built file (nvcc gets no .ptx source, and main.cu has 19 lines), so file and line
    # are None and the place stays in the message, with ": " where ptxas printed "; error   : "
    # (PTXAS_PLACE_MESSAGE). The second, "ptxas fatal   : Ptx assembly aborted due to errors", is a second error, as
    # a ptxas fatal is; its message is the text after "ptxas fatal   : ". ptxas gives no column or code. The build
    # exited 255.
    "nvcc_ptxas_inline_asm": FixtureCase(
        nvcc,
        [
            compile_diag("error", None, None, None, None, PTXAS_PLACE_MESSAGE),
            compile_diag("error", None, None, None, None, "Ptx assembly aborted due to errors"),
        ],
    ),
    # The "in function `main'" context line is no diagnostic. The undefined reference names a temporary cudafe1
    # file of this run and a section offset, no built file and no line, so it has no file or line; its message
    # is the text after the last ": ". The collect2 line is a second error.
    "nvcc_linker_error": FixtureCase(
        nvcc,
        [
            compile_diag("error", None, None, None, None, HELPER_UNDEFINED),
            compile_diag("error", None, None, None, None, "ld returned 1 exit status"),
        ],
    ),
    # Empty stderr.
    "nvcc_clean": FixtureCase(nvcc, []),
    # main.cpp line 7 is "        y[i] = a * x[i] + undefined_var;": caret index 28 - echo prefix 2 + 1 = 27.
    # The blank line and the summary are no diagnostics.
    "nvcpp_edg_error": FixtureCase(
        nvcpp, [compile_diag("error", None, "main.cpp", 7, 27, 'identifier "undefined_var" is undefined')]
    ),
    # main.cpp line 13 is "    int unused = 0;": caret index 10 - echo prefix 2 + 1 = column 9, the 'u'. The
    # trailing [declared_but_not_referenced] is the code. The Remark line and the whole -Minfo report after it
    # (function names, numbered region lines, "Generating map(...)" lines) are no diagnostics.
    "nvcpp_edg_warning": FixtureCase(
        nvcpp,
        [
            compile_diag(
                "warning",
                "declared_but_not_referenced",
                "main.cpp",
                13,
                9,
                'variable "unused" was declared but never referenced',
            )
        ],
    ),
    # NVC++-S-1101: S (severe) is an error, and the code is "S-1101". The file and line come from the trailing
    # "(main.cpp: 4)"; line 4 is as printed (the line before the pragma, as the -Minfo report numbers the
    # region). A backend line gives no column. The -Minfo report and the closing summary are no diagnostics.
    "nvcpp_backend_error": FixtureCase(nvcpp, [compile_diag("error", "S-1101", "main.cpp", 4, None, STACK_LIMIT)]),
    # NVC++-F-0000: F (fatal) is an error, and the code is "F-0000". Two blanks separate the message from
    # "(main.cpp: 8)", which gives file and line; the message is the text before them, as printed. The
    # "compilation aborted" summary is no diagnostic.
    "nvcpp_fatal_abort": FixtureCase(nvcpp, [compile_diag("error", "F-0000", "main.cpp", 8, None, TINFO_ICE)]),
    # The -Minfo report alone yields nothing.
    "nvcpp_minfo_clean": FixtureCase(nvcpp, []),
    # The -Minfo report, the "in function `main'" context line, and the closing pgacclnk status line are no
    # diagnostics. The undefined reference names main.cpp by its absolute path in this run's workdir, which
    # ends with "/main.cpp", a built file, and line 21, the call "    helper(y, 1024);" in main.cpp: so the file
    # is main.cpp and the line 21. The linker gives no column, and the message is the text after the last ": ".
    # Without the file text nothing shows that the path is a built file, so file and line are None.
    "nvcpp_linker_error": FixtureCase(
        nvcpp,
        [compile_diag("error", None, "main.cpp", 21, None, HELPER_UNDEFINED)],
        without_files=[compile_diag("error", None, None, None, None, HELPER_UNDEFINED)],
    ),
    # P0.17. Thirteen lines. The file headers "helper.cpp:" and "main.cpp:" that nvc++ prints for each of several
    # sources, the -Minfo report (the function names, the numbered region and loop lines, the "Generating map(...)"
    # lines), and the closing "pgacclnk: child process exit status 2: .../tools/nvdd" line are no diagnostics. The
    # one error is the line "nvlink error   : Undefined reference to '_Z5twicef' in '<NVLINK_OBJECT>'". nvlink
    # names the mangled twice(float) and a temporary object under @lassi-tmp, no built file and no line, so file,
    # line, and column are None, and there is no code. The message is the text after "nvlink error   : ", the
    # object path included (NVLINK_MESSAGE). The build exited 2.
    "nvcpp_nvlink_error": FixtureCase(nvcpp, [compile_diag("error", None, None, None, None, NVLINK_MESSAGE)]),
    # main.cpp line 2 is '#include "kernels/scale.h"' (26 characters). The caret is at index 28 of the echo,
    # after the closing quote: 28 - echo prefix 2 + 1 = column 27, one past the end of the line, kept as the
    # compiler gives it since the echo matches the line. A catastrophic error is an error. The summary and
    # "Compilation terminated." are no diagnostics.
    "nvcpp_missing_include": FixtureCase(
        nvcpp, [compile_diag("error", None, "main.cpp", 2, 27, 'cannot open source file "kernels/scale.h"')]
    ),
}

# The preset each adapter module registers, which the capture built for that module's scenarios.
PRESET_OF: dict[ModuleType, type] = {nvcc: nvcc.NvccSm80, nvcpp: nvcpp.NvcppCc80}


def fixture_text(name: str) -> str:
    """Return a stderr fixture exactly as stored: its bytes decoded as UTF-8, with no newline translation."""
    return (FIXTURES / f"{name}.stderr").read_bytes().decode("utf-8")


def load_captures() -> dict[str, Any]:
    """Return captures.json, the provenance manifest of the capture the fixtures were copied from."""
    raw = CAPTURES.read_bytes()
    assert raw.isascii(), f"{CAPTURES} is not plain ASCII"
    return json.loads(raw.decode("ascii"))


# ---------------------------------------------------------------------------
# The fixtures themselves and their capture record


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


def test_the_capture_ran_from_a_clean_commit_and_covers_every_fixture() -> None:
    captures = load_captures()
    assert captures["dirty"] is False
    assert captures["snapshot_of"] is None
    assert re.fullmatch(r"[0-9a-f]{40}", captures["commit"]), captures["commit"]
    assert captures["rx_run_id"], "the capture must come from a recorded rx run"
    assert captures["host"] == "alpha01"
    assert sorted(captures["scenarios"]) == sorted(FIXTURE_CASES)


@pytest.mark.parametrize("name", sorted(FIXTURE_CASES))
def test_fixture_is_the_captured_stderr_byte_for_byte(name: str) -> None:
    entry = load_captures()["scenarios"][name]
    raw = (FIXTURES / f"{name}.stderr").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["stderr_sha256"], name
    assert len(raw) == entry["stderr_bytes"], name


@pytest.mark.parametrize("name", sorted(FIXTURE_CASES))
def test_the_capture_compiled_the_scenario_source_tree(name: str) -> None:
    # The recorded argv is the adapter's own command over the tree's sources, so the fixture belongs to these files.
    entry = load_captures()["scenarios"][name]
    preset = PRESET_OF[FIXTURE_CASES[name].module]
    assert entry["toolchain"] == preset.name, name
    adapter = type(f"{preset.__name__}Capture", (preset,), dict(entry["overrides"]))
    sources = sorted(path for path in scenario_files(name) if path.endswith(preset.SOURCE_SUFFIXES))
    assert sources, name
    assert entry["argv"] == adapter(executable=entry["argv"][0]).command(sources), name


@pytest.mark.parametrize("name", sorted(FIXTURE_CASES))
def test_a_failed_capture_parses_into_an_error_and_a_clean_one_into_none(name: str) -> None:
    status = load_captures()["scenarios"][name]["exit_status"]
    errors = [diagnostic for diagnostic in FIXTURE_CASES[name].expected if diagnostic.severity == "error"]
    assert bool(errors) == (status != 0), (name, status)


def test_readme_records_the_capture_without_placeholder() -> None:
    # One table row per fixture holds its file name, the capture's exit status, and the rx id; the README names
    # the commit the capture ran from and points to captures.json.
    text = (FIXTURES / "README.md").read_text(encoding="utf-8")
    captures = load_captures()
    assert "PLACEHOLDER" not in text
    assert captures["commit"][:7] in text
    assert "captures.json" in text
    rows = [line for line in text.splitlines() if line.startswith("|")]
    for name, entry in sorted(captures["scenarios"].items()):
        matching = [row for row in rows if f"{name}.stderr" in row]
        assert len(matching) == 1, f"{name}: {len(matching)} table rows"
        cells = [cell.strip().strip("`") for cell in matching[0].strip().strip("|").split("|")]
        assert str(entry["exit_status"]) in cells, (name, cells)
        assert any(captures["rx_run_id"] in cell for cell in cells), (name, cells)


# ---------------------------------------------------------------------------
# parse_diagnostics on the fixtures


@pytest.mark.parametrize("name", sorted(FIXTURE_CASES))
def test_fixture_parses_into_the_exact_diagnostics(name: str) -> None:
    case = FIXTURE_CASES[name]
    assert case.module.parse_diagnostics(fixture_text(name), scenario_files(name)) == case.expected


@pytest.mark.parametrize("name", sorted(FIXTURE_CASES))
def test_no_column_or_linker_place_without_the_file_text(name: str) -> None:
    # Every column and linker place in the fixtures needs the file text to be checked against, so without it
    # the column is None, a linker place is gone (without_files), and the rest is unchanged.
    case = FIXTURE_CASES[name]
    expected = case.without_files
    if expected is None:
        expected = [dataclasses.replace(diagnostic, column=None) for diagnostic in case.expected]
    assert case.module.parse_diagnostics(fixture_text(name)) == expected
    assert case.module.parse_diagnostics(fixture_text(name), {}) == expected


def test_the_p017_sample_lines_are_lines_of_their_fixtures() -> None:
    # The oracle for the P0.17 sample lines: each is a whole line of its fixture, which
    # test_fixture_is_the_captured_stderr_byte_for_byte checks against the sha256 the capture recorded.
    assert fixture_text("nvcc_ptxas_inline_asm").splitlines() == [PTXAS_PLACE_LINE, PTXAS_ABORTED_LINE]
    assert fixture_text("nvcpp_nvlink_error").splitlines()[-2:] == [NVLINK_LINE, NVDD_STATUS_LINE]
    run = load_captures()["rx_run_id"]
    assert run in PTX_FILE and run in NVLINK_OBJECT


@pytest.mark.parametrize("module", [nvcc, nvcpp], ids=["nvcc", "nvcpp"])
def test_empty_stderr_parses_to_nothing(module: ModuleType) -> None:
    assert module.parse_diagnostics("") == []
    assert module.parse_diagnostics("\n\n") == []


# ---------------------------------------------------------------------------
# parse_diagnostics: one pattern at a time


NVCC_LINES = [
    # From capture nvcc_undefined_identifier: nvcc prints an EDG error without a number.
    pytest.param(
        'main.cu(7): error: identifier "undefined_var" is undefined\n',
        [compile_diag("error", None, "main.cu", 7, None, 'identifier "undefined_var" is undefined')],
        id="edg-error",
    ),
    # From capture nvcc_warning_177: an EDG warning carries its number with "-D".
    pytest.param(
        'kernels/scale.cuh(5): warning #177-D: variable "unused" was declared but never referenced\n',
        [
            compile_diag(
                "warning", "177-D", "kernels/scale.cuh", 5, None, 'variable "unused" was declared but never referenced'
            )
        ],
        id="edg-warning-number",
    ),
    # No capture shows an EDG number without "-D" from nvcc yet.
    pytest.param(
        'main.cu(3): error #20: identifier "x" is undefined\n',
        [compile_diag("error", "20", "main.cu", 3, None, 'identifier "x" is undefined')],
        id="edg-number-without-d",
    ),
    # No capture shows an EDG remark from nvcc yet.
    pytest.param(
        "main.cu(14): remark #186-D: pointless comparison of unsigned integer with zero\n",
        [compile_diag("note", "186-D", "main.cu", 14, None, "pointless comparison of unsigned integer with zero")],
        id="edg-remark-is-note",
    ),
    # From capture nvcc_host_gcc_warning. nvcc keeps no GCC column (see the GCC column tests).
    pytest.param(
        "main.cu:16:19: warning: " + SIGN_COMPARE + " [-Wsign-compare]\n",
        [compile_diag("warning", "-Wsign-compare", "main.cu", 16, None, SIGN_COMPARE)],
        id="gcc-warning-flag",
    ),
    # No capture shows the GCC lines below from nvcc yet; nvcc keeps no GCC column.
    pytest.param(
        "main.cu:3:10: fatal error: kernels/missing.cuh: No such file or directory\ncompilation terminated.\n",
        [compile_diag("error", None, "main.cu", 3, None, "kernels/missing.cuh: No such file or directory")],
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
                None,
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
                None,
                "format '%d' expects argument of type 'int', but argument 2 has type 'size_t' "
                "{aka 'long unsigned int'}",
            ),
            compile_diag("note", None, "main.cu", 30, None, "format string is defined here"),
        ],
        id="gcc-warning-flag-with-equals-and-note",
    ),
    pytest.param(
        "main.cu:12:9: error: static assertion failed: size(3): error: too small\n",
        [compile_diag("error", None, "main.cu", 12, None, "static assertion failed: size(3): error: too small")],
        id="gcc-message-that-looks-like-edg",
    ),
    # From capture nvcc_ptxas_error.
    pytest.param(
        "ptxas error   : Entry function '_Z6reducePKfPfi' uses too much shared data (0x40000 bytes, 0x29000 max)\n",
        [
            compile_diag(
                "error",
                None,
                None,
                None,
                None,
                "Entry function '_Z6reducePKfPfi' uses too much shared data (0x40000 bytes, 0x29000 max)",
            )
        ],
        id="ptxas-error",
    ),
    # No capture shows a ptxas warning or fatal line yet.
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
    # P0.17, the first line of capture nvcc_ptxas_inline_asm (PTXAS_PLACE_LINE). A PTX place indexes no built file,
    # so file and line are None and the message keeps the place (PTXAS_PLACE_MESSAGE).
    pytest.param(
        PTXAS_PLACE_LINE + "\n",
        [compile_diag("error", None, None, None, None, PTXAS_PLACE_MESSAGE)],
        id="ptxas-ptx-file-and-line",
    ),
    # P0.17, from exploratory scenario nvcc_ptxas_ptx_error (dirty-tree rx
    # 20260923-202434-desktop-8r113ei-p0-core-0586). The unknown instruction 'bogus.op.s32 %0, %0;' gave two
    # errors on one PTX line, and each is kept. Only the path is shortened: the workdir becomes /w (SHORT_PTX).
    pytest.param(
        "ptxas " + SHORT_PTX + ", line 28; error   : Unknown modifier '.op'\n"
        "ptxas " + SHORT_PTX + ", line 28; error   : Not a name of any known instruction: 'bogus'\n"
        "ptxas fatal   : Ptx assembly aborted due to errors\n",
        [
            compile_diag("error", None, None, None, None, SHORT_PTX + ", line 28: Unknown modifier '.op'"),
            compile_diag(
                "error", None, None, None, None, SHORT_PTX + ", line 28: Not a name of any known instruction: 'bogus'"
            ),
            compile_diag("error", None, None, None, None, "Ptx assembly aborted due to errors"),
        ],
        id="ptxas-two-errors-on-one-ptx-line",
    ),
    # No capture shows the two lines below yet. The first is a ptxas warning with a PTX place, whose severity comes
    # from the line. The second has the longest line number the parser reads, 10 digits.
    pytest.param(
        "ptxas " + SHORT_PTX + ", line 57; warning : Instruction 'vote' without '.sync' is deprecated\n",
        [
            compile_diag(
                "warning",
                None,
                None,
                None,
                None,
                SHORT_PTX + ", line 57: Instruction 'vote' without '.sync' is deprecated",
            )
        ],
        id="ptxas-ptx-warning",
    ),
    pytest.param(
        "ptxas " + SHORT_PTX + ", line 2147483647; error   : Unknown modifier '.bogus'\n",
        [compile_diag("error", None, None, None, None, SHORT_PTX + ", line 2147483647: Unknown modifier '.bogus'")],
        id="ptxas-ptx-line-ten-digits",
    ),
    # No capture shows a ptxas fatal with a PTX place yet. A fatal is an error, as on a ptxas line with no place.
    pytest.param(
        "ptxas " + SHORT_PTX + ", line 28; fatal   : Parsing error near '.bogus': syntax error\n",
        [
            compile_diag(
                "error", None, None, None, None, SHORT_PTX + ", line 28: Parsing error near '.bogus': syntax error"
            )
        ],
        id="ptxas-ptx-fatal-is-error",
    ),
    # No capture shows this line. The place is the shortest text that ends in ", line <line>" before
    # "; <severity>", so a message that holds that text again stays whole.
    pytest.param(
        "ptxas " + SHORT_PTX + ", line 28; error   : x, line 3; error : y\n",
        [compile_diag("error", None, None, None, None, SHORT_PTX + ", line 28: x, line 3; error : y")],
        id="ptxas-ptx-shortest-place",
    ),
    # No capture shows the lines below. Both ptxas patterns are tried before the EDG and GCC ones, so text shaped
    # like an EDG or GCC place in a ptxas message, or in the source stem nvcc puts in its PTX name (EDG_STEM_PTX,
    # GCC_STEM_PTX), never becomes a file and line. They stay None, and the message keeps the text whole.
    pytest.param(
        "ptxas error   : Unresolved extern function 'a:1:2: error: b'\n",
        [compile_diag("error", None, None, None, None, "Unresolved extern function 'a:1:2: error: b'")],
        id="ptxas-message-with-a-gcc-place",
    ),
    pytest.param(
        "ptxas " + SHORT_PTX + ", line 28; error   : Unknown symbol 'a:1:2: error: b'\n",
        [compile_diag("error", None, None, None, None, SHORT_PTX + ", line 28: Unknown symbol 'a:1:2: error: b'")],
        id="ptxas-ptx-message-with-a-gcc-place",
    ),
    pytest.param(
        "ptxas " + EDG_STEM_PTX + ", line 28; error   : Unknown modifier '.bogus'\n",
        [compile_diag("error", None, None, None, None, EDG_STEM_PTX + ", line 28: Unknown modifier '.bogus'")],
        id="ptxas-ptx-stem-with-an-edg-place",
    ),
    pytest.param(
        "ptxas " + GCC_STEM_PTX + ", line 28; error   : Unknown modifier '.bogus'\n",
        [compile_diag("error", None, None, None, None, GCC_STEM_PTX + ", line 28: Unknown modifier '.bogus'")],
        id="ptxas-ptx-stem-with-a-gcc-place",
    ),
    # The driver format is the one capture nvcc_fatal shows; this message is another driver fatal.
    pytest.param(
        "nvcc fatal   : Don't know what to do with 'kernels/scale.cuh'\n",
        [compile_diag("error", None, None, None, None, "Don't know what to do with 'kernels/scale.cuh'")],
        id="driver-fatal",
    ),
    # No capture shows this line. A driver fatal may quote a file the model named, and the name may hold a GCC style
    # place. The driver pattern is tried before the GCC one, so the line has no file or line and the message is whole.
    pytest.param(
        "nvcc fatal   : Could not open input file 'ab:1:2: error: b.cu'\n",
        [compile_diag("error", None, None, None, None, "Could not open input file 'ab:1:2: error: b.cu'")],
        id="driver-fatal-with-a-gcc-place",
    ),
    # No capture shows a catastrophic or internal EDG error from nvcc yet.
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
    # From capture nvcc_linker_error: the undefined reference names a temporary cudafe1 file and a section
    # offset, with no "/usr/bin/ld: " in front; the collect2 line follows.
    pytest.param(
        "tmpxft_00000002_00000000-6_main.cudafe1.cpp:(.text.startup+0x2c): " + HELPER_UNDEFINED + "\n"
        "collect2: error: ld returned 1 exit status\n",
        [
            compile_diag("error", None, None, None, None, HELPER_UNDEFINED),
            compile_diag("error", None, None, None, None, "ld returned 1 exit status"),
        ],
        id="linker-message-after-the-last-colon",
    ),
    # No capture shows this yet: a reference from a data section, with two ": " before the phrase.
    pytest.param(
        "/usr/bin/ld: main.o:(.data.rel.ro+0x10): " + VTABLE_UNDEFINED + "\n",
        [compile_diag("error", None, None, None, None, VTABLE_UNDEFINED)],
        id="linker-two-colons-before-the-phrase",
    ),
]

NVCPP_LINES = [
    # The EDG format and the trailing tag are the ones capture nvcpp_edg_warning shows; this is another tag.
    pytest.param(
        '"kernels/scale.h", line 3: warning: variable "t" was set but never used [set_but_not_used]\n',
        [
            compile_diag(
                "warning", "set_but_not_used", "kernels/scale.h", 3, None, 'variable "t" was set but never used'
            )
        ],
        id="edg-warning-tag",
    ),
    # From capture nvcpp_edg_error.
    pytest.param(
        '"main.cpp", line 7: error: identifier "undefined_var" is undefined\n',
        [compile_diag("error", None, "main.cpp", 7, None, 'identifier "undefined_var" is undefined')],
        id="edg-error-no-tag",
    ),
    # From capture nvcpp_missing_include: nvc++ reports a missing header as an EDG catastrophic error.
    pytest.param(
        '"main.cpp", line 2: catastrophic error: cannot open source file "kernels/scale.h"\n',
        [compile_diag("error", None, "main.cpp", 2, None, 'cannot open source file "kernels/scale.h"')],
        id="edg-catastrophic-error",
    ),
    # No capture shows these EDG severities from nvc++ yet.
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
    # From capture nvcpp_backend_error: one blank before "(main.cpp: 4)".
    pytest.param(
        "NVC++-S-1101-" + STACK_LIMIT + " (main.cpp: 4)\n",
        [compile_diag("error", "S-1101", "main.cpp", 4, None, STACK_LIMIT)],
        id="backend-severe-with-file",
    ),
    # From capture nvcpp_fatal_abort: two blanks before "(main.cpp: 8)". Any run of blanks separates the file,
    # and the message keeps the text before them as printed, internal number included.
    pytest.param(
        "NVC++-F-0000-" + TINFO_ICE + "  (main.cpp: 8)\n",
        [compile_diag("error", "F-0000", "main.cpp", 8, None, TINFO_ICE)],
        id="backend-fatal-internal-error",
    ),
    # No capture shows the backend lines below yet.
    pytest.param(
        "NVC++-W-0155-Accelerator region ignored; see -Minfo messages (main.cpp: 12)\n",
        [compile_diag("warning", "W-0155", "main.cpp", 12, None, "Accelerator region ignored; see -Minfo messages")],
        id="backend-warning",
    ),
    pytest.param(
        "NVC++-W-0155-Accelerator region ignored; see -Minfo messages\t(main.cpp: 12)\n",
        [compile_diag("warning", "W-0155", "main.cpp", 12, None, "Accelerator region ignored; see -Minfo messages")],
        id="backend-tab-before-the-file",
    ),
    pytest.param(
        "NVC++-I-0035-Predefined intrinsic max loses intrinsic property (main.cpp: 7)\n",
        [compile_diag("note", "I-0035", "main.cpp", 7, None, "Predefined intrinsic max loses intrinsic property")],
        id="backend-info-is-note",
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
        "NVC++-S-0155-Invalid accelerator region (main.cpp: 21)   \t\n",
        [compile_diag("error", "S-0155", "main.cpp", 21, None, "Invalid accelerator region")],
        id="backend-trailing-blanks-keep-the-file",
    ),
    # No capture shows a collect2 line from nvc++ yet; its captured link failure ends with a pgacclnk line.
    pytest.param(
        "collect2: error: ld returned 1 exit status\n",
        [compile_diag("error", None, None, None, None, "ld returned 1 exit status")],
        id="collect2",
    ),
    # From capture nvcpp_linker_error: the linker names main.cpp by its absolute workdir path and a line. With
    # no files the path is no built file, so file and line are None (see the linker place tests).
    pytest.param(
        "/mnt/nvme10/joseph_ufl/lassi-runs/fixture-captures/20260923-211958-desktop-8r113ei-p0-core-d221/work/"
        "nvcpp_linker_error/main.cpp:21: " + HELPER_UNDEFINED + "\n",
        [compile_diag("error", None, None, None, None, HELPER_UNDEFINED)],
        id="linker-message-after-the-last-colon",
    ),
    # No capture shows this yet: a reference from a data section, with two ": " before the phrase.
    pytest.param(
        "/usr/bin/ld: main.o:(.data.rel.ro+0x10): " + VTABLE_UNDEFINED + "\n",
        [compile_diag("error", None, None, None, None, VTABLE_UNDEFINED)],
        id="linker-two-colons-before-the-phrase",
    ),
    # No committed capture shows this. Exploratory probe nvcpp_probe_asm_int (dirty-tree rx
    # 20260923-104618-desktop-8r113ei-p0-core-173d) showed LLVM's assembler naming the inline asm string, whose
    # line and column index no built file, so there is no column; its echo and caret lines match no pattern.
    pytest.param(
        "<inline asm>:1:2: error: invalid instruction mnemonic 'bogus.op.s32'\n"
        "        bogus.op.s32 %edi, %edi;\n"
        "        ^~~~~~~~~~~~\n",
        [compile_diag("error", None, "<inline asm>", 1, None, "invalid instruction mnemonic 'bogus.op.s32'")],
        id="llvm-inline-asm-error",
    ),
    # P0.17, the last two lines of capture nvcpp_nvlink_error (NVLINK_LINE, NVDD_STATUS_LINE). nvlink names a
    # temporary object and no line, so file and line are None (NVLINK_MESSAGE). The pgacclnk line after it names
    # nvdd and only restates the failure, so it is no diagnostic, like the one naming /usr/bin/ld.
    pytest.param(
        NVLINK_LINE + "\n" + NVDD_STATUS_LINE + "\n",
        [compile_diag("error", None, None, None, None, NVLINK_MESSAGE)],
        id="nvlink-undefined-reference",
    ),
    # No capture shows an nvlink warning yet; its severity comes from the line.
    pytest.param(
        "nvlink warning : Stack size for entry function 'nvkernel__Z5scaleiPKfPf_F1L6_2' cannot be statically "
        "determined\n",
        [
            compile_diag(
                "warning",
                None,
                None,
                None,
                None,
                "Stack size for entry function 'nvkernel__Z5scaleiPKfPf_F1L6_2' cannot be statically determined",
            )
        ],
        id="nvlink-warning",
    ),
    # No capture shows this line. An asm label ('float twice(float) asm("...");') can make a symbol name any text,
    # a GCC style place included, and nvlink quotes it. The line is still nvlink's: no file or line, message whole.
    pytest.param(
        "nvlink error   : Undefined reference to 'a.cpp:1:2: error: b' in '/w/@lassi-tmp/nvc++Vcnd0BZwBe.o'\n",
        [
            compile_diag(
                "error",
                None,
                None,
                None,
                None,
                "Undefined reference to 'a.cpp:1:2: error: b' in '/w/@lassi-tmp/nvc++Vcnd0BZwBe.o'",
            )
        ],
        id="nvlink-symbol-with-a-gcc-place",
    ),
    # No capture shows an nvlink fatal yet. A fatal is an error, as for ptxas.
    pytest.param(
        "nvlink fatal   : Could not open input file '/w/@lassi-tmp/nvc++Vcnd0BZwBe.o'\n",
        [compile_diag("error", None, None, None, None, "Could not open input file '/w/@lassi-tmp/nvc++Vcnd0BZwBe.o'")],
        id="nvlink-fatal-is-error",
    ),
    # P0.17, the nvc++ driver line of plans/spikes/p0-toolchains-verify.md run 2a (NVCPP_DRIVER_ERROR). The driver
    # names no file or line, so the Diagnostic has none, and the message is the text after "nvc++-Error-".
    pytest.param(
        NVCPP_DRIVER_ERROR + "\n",
        [compile_diag("error", None, None, None, None, NVCPP_DRIVER_ERROR.removeprefix("nvc++-Error-"))],
        id="driver-error",
    ),
    # P0.17, the stderr of exploratory probe rx 20260923-104313-desktop-8r113ei-p0-core-2b22 (NVCPFE_CRASH_STDERR):
    # a backend error, then the driver's fatal line naming the front end that crashed. A driver fatal is an error.
    pytest.param(NVCPFE_CRASH_STDERR, NVCPFE_CRASH_DIAGNOSTICS, id="driver-fatal-after-a-backend-error"),
    # No capture shows a driver warning yet; its severity comes from the line.
    pytest.param(
        "nvc++-Warning-CUDA_HOME has been deprecated. Please, use NVHPC_CUDA_HOME instead.\n",
        [
            compile_diag(
                "warning", None, None, None, None, "CUDA_HOME has been deprecated. Please, use NVHPC_CUDA_HOME instead."
            )
        ],
        id="driver-warning",
    ),
    # No capture shows this line; its message is made up. A driver message may quote a source name, and the model
    # names the sources, so it may hold a GCC style place. The driver pattern is tried before the GCC style one, so
    # the line is still the driver's: no file or line, and the message whole.
    pytest.param(
        "nvc++-Error-Unable to access file ab:1:2: error: b.cpp\n",
        [compile_diag("error", None, None, None, None, "Unable to access file ab:1:2: error: b.cpp")],
        id="driver-message-with-a-gcc-place",
    ),
]


@pytest.mark.parametrize(("stderr", "expected"), NVCC_LINES)
def test_nvcc_pattern(stderr: str, expected: list[Diagnostic]) -> None:
    assert nvcc.parse_diagnostics(stderr) == expected


@pytest.mark.parametrize(("stderr", "expected"), NVCPP_LINES)
def test_nvcpp_pattern(stderr: str, expected: list[Diagnostic]) -> None:
    assert nvcpp.parse_diagnostics(stderr) == expected


SKIPPED_LINES = [
    # From the captures.
    pytest.param(nvcc, '1 error detected in the compilation of "main.cu".', id="nvcc-error-summary"),
    pytest.param(
        nvcc, 'Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"', id="nvcc-remark"
    ),
    pytest.param(nvcc, "main.cu: In function 'int main()':", id="nvcc-gcc-in-function"),
    pytest.param(nvcc, "   16 |     for (int i = 0; i < host.size(); i++) {", id="nvcc-gcc-echo"),
    pytest.param(nvcc, "      |                 ~~^~~~~~~~~~~~~", id="nvcc-gcc-caret"),
    pytest.param(
        nvcc,
        "/usr/bin/ld: /mnt/nvme10/joseph_ufl/lassi-runs/fixture-captures/20260923-211958-desktop-8r113ei-p0-core-d221"
        "/work/nvcc_linker_error/@lassi-tmp/tmpxft_00000002_00000000-11_main.o: in function `main':",
        id="nvcc-ld-context",
    ),
    pytest.param(nvcpp, '1 error detected in the compilation of "main.cpp".', id="nvcpp-error-summary"),
    pytest.param(
        nvcpp,
        '1 catastrophic error detected in the compilation of "main.cpp".',
        id="nvcpp-catastrophic-summary",
    ),
    pytest.param(nvcpp, "Compilation terminated.", id="nvcpp-terminated"),
    pytest.param(
        nvcpp,
        'Remark: individual warnings can be suppressed with "--diag_suppress <warning-name>"',
        id="nvcpp-remark",
    ),
    pytest.param(nvcpp, "NVC++/x86-64 Linux 24.11-0: compilation completed with severe errors", id="nvcpp-summary"),
    pytest.param(nvcpp, "NVC++/x86-64 Linux 24.11-0: compilation aborted", id="nvcpp-summary-aborted"),
    pytest.param(nvcpp, "saxpy(int, float, float const*, float*):", id="nvcpp-minfo-function"),
    pytest.param(nvcpp, "main:", id="nvcpp-minfo-main"),
    pytest.param(nvcpp, "      4, #omp target teams distribute parallel for", id="nvcpp-minfo-region"),
    pytest.param(
        nvcpp, '          4, Generating "nvkernel__Z5saxpyifPKfPf_F1L4_2" GPU kernel', id="nvcpp-minfo-kernel"
    ),
    pytest.param(
        nvcpp,
        "          6, Loop parallelized across teams and threads(128), schedule(static)",
        id="nvcpp-minfo-parallel",
    ),
    pytest.param(nvcpp, "      6, Loop not vectorized/parallelized: not countable", id="nvcpp-minfo-colon"),
    pytest.param(nvcpp, "         Generating map(to:x[:n]) ", id="nvcpp-minfo-map"),
    pytest.param(nvcpp, "     15, Generated vector simd code for the loop", id="nvcpp-minfo-simd"),
    pytest.param(
        nvcpp,
        "/usr/bin/ld: /mnt/nvme10/joseph_ufl/lassi-runs/fixture-captures/20260923-211958-desktop-8r113ei-p0-core-d221"
        "/work/nvcpp_linker_error/@lassi-tmp/nvc++vc-2bVcpD9.o: in function `main':",
        id="nvcpp-ld-context",
    ),
    pytest.param(nvcpp, "pgacclnk: child process exit status 1: /usr/bin/ld", id="nvcpp-link-status"),
    # From capture nvcpp_nvlink_error (P0.17): the file header nvc++ prints for each of several
    # sources, an -Minfo line with ": " in it, and the pgacclnk line naming nvdd rather than ld.
    pytest.param(nvcpp, "helper.cpp:", id="nvcpp-file-header"),
    pytest.param(nvcpp, "      8, Loop not vectorized/parallelized: contains call", id="nvcpp-minfo-contains-call"),
    pytest.param(nvcpp, NVDD_STATUS_LINE, id="nvcpp-nvdd-status"),
    # No capture shows these yet.
    pytest.param(nvcc, '2 errors detected in the compilation of "main.cu".', id="nvcc-errors-summary"),
    pytest.param(nvcc, "compilation terminated.", id="nvcc-gcc-terminated"),
    pytest.param(
        nvcpp, "     21, Accelerator restriction: size of the GPU copy of tmp is unknown", id="nvcpp-minfo-note"
    ),
    # No capture shows these either: ptxas prints info lines only with -v or --resource-usage, which the preset does
    # not pass. An info line is no diagnostic, and one holding ", line <line>; error : " is still no PTX place.
    pytest.param(nvcc, "ptxas info    : Used 8 registers, 360 bytes cmem[0]", id="nvcc-ptxas-info"),
    pytest.param(
        nvcc, "ptxas info    : Function properties for f, line 3; error : y", id="nvcc-ptxas-info-with-a-place"
    ),
    # Echoed source text never becomes a diagnostic, even when it looks like a linker or GCC line.
    pytest.param(nvcc, '   21 |     printf("undefined reference to %d", name);', id="nvcc-gcc-echo-linker-phrase"),
    pytest.param(nvcc, '    5 |     const char* msg = "a.c:1:2: error: boom";', id="nvcc-gcc-echo-gcc-line"),
    pytest.param(nvcc, '12345 | x("main.cu(3): error: y");', id="nvcc-gcc-echo-edg-line"),
    pytest.param(nvcpp, '   21 |     printf("undefined reference to %d", name);', id="nvcpp-gcc-echo-linker-phrase"),
    pytest.param(nvcpp, '    5 |     const char* msg = "a.c:1:2: error: boom";', id="nvcpp-gcc-echo-gcc-line"),
]


@pytest.mark.parametrize(("module", "line"), SKIPPED_LINES)
def test_lines_matching_no_pattern_are_skipped(module: ModuleType, line: str) -> None:
    assert module.parse_diagnostics(line + "\n") == []
    assert module.parse_diagnostics(line) == []


@pytest.mark.parametrize("module", [nvcc, nvcpp], ids=["nvcc", "nvcpp"])
def test_a_gcc_warning_keeps_its_echo_out_of_the_diagnostics(module: ModuleType) -> None:
    # A build that exits 0 with this warning must not carry a false linker error from the echoed source. With
    # no files neither adapter keeps a GCC column.
    stderr = (
        "main.cu:21:37: warning: format '%d' expects argument of type 'int', but argument 2 has type "
        "'const char*' [-Wformat=]\n"
        '   21 |     printf("undefined reference to %d", name);\n'
        "      |                                    ~^\n"
        "      |                                     |\n"
        "      |                                     int\n"
    )
    message = "format '%d' expects argument of type 'int', but argument 2 has type 'const char*'"
    assert module.parse_diagnostics(stderr) == [compile_diag("warning", "-Wformat=", "main.cu", 21, None, message)]


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


def test_a_backend_line_with_a_long_blank_run_parses_in_linear_time() -> None:
    # Any run of blanks may separate "(main.cpp: 3)" from the message (capture nvcpp_fatal_abort has two). A
    # separator pattern that starts at every blank of a long run rescans the run each time and took seconds here.
    message = "a" + " " * 50_000 + "b"
    began = time.monotonic()
    assert nvcpp.parse_diagnostics(f"NVC++-S-0155-{message} (main.cpp: 3)\n") == [
        compile_diag("error", "S-0155", "main.cpp", 3, None, message)
    ]
    assert time.monotonic() - began < 2.0


def test_a_long_ptxas_place_line_parses_in_linear_time() -> None:
    # P0.17. Model-written inline PTX reaches a ptxas message, so a ptxas line can be very long. Each case below
    # repeats a piece of the place format: places with no severity after them, a long blank run before the ":"
    # (ptxas pads its severity word), and a message that holds many "; error   : ". A linear parser takes milliseconds.
    # A blank run with no ":" after it makes the pattern give the run back one blank at a time before it fails, and
    # a padded info word makes the lookahead that refuses a severity word read the run once.
    places = "ptxas " + f"{SHORT_PTX}, line 28; " * 20_000
    blank_run = f"ptxas {SHORT_PTX}, line 28; error" + " " * 50_000 + ": x"
    blank_run_no_colon = f"ptxas {SHORT_PTX}, line 28; error" + " " * 50_000 + "x"
    info_blank_run = "ptxas info" + " " * 50_000 + f": x {SHORT_PTX}, line 28; error : y"
    separators = "a; error   : b" * 20_000
    began = time.monotonic()
    assert nvcc.parse_diagnostics(places + "\n") == []
    assert nvcc.parse_diagnostics(blank_run + "\n") == [
        compile_diag("error", None, None, None, None, f"{SHORT_PTX}, line 28: x")
    ]
    assert nvcc.parse_diagnostics(blank_run_no_colon + "\n") == []
    assert nvcc.parse_diagnostics(info_blank_run + "\n") == []
    assert nvcc.parse_diagnostics(f"ptxas {SHORT_PTX}, line 28; error   : {separators}\n") == [
        compile_diag("error", None, None, None, None, f"{SHORT_PTX}, line 28: {separators}")
    ]
    assert time.monotonic() - began < 2.0


def test_a_long_nvlink_line_parses_in_linear_time() -> None:
    # P0.17. A symbol name and an object path reach an nvlink message, so an nvlink line can be very long. The cases
    # below are a long blank run before the ":", a message with many ": ", and a line that repeats the tool and
    # severity words with no ":" after them.
    blank_run = "nvlink error" + " " * 50_000 + ": x"
    blank_run_no_colon = "nvlink error" + " " * 50_000 + "x"
    message = "Undefined reference to '_Z1fv' in '" + "x: " * 40_000 + "o'"
    no_colon = "nvlink error " * 30_000
    began = time.monotonic()
    assert nvcpp.parse_diagnostics(blank_run + "\n") == [compile_diag("error", None, None, None, None, "x")]
    assert nvcpp.parse_diagnostics(blank_run_no_colon + "\n") == []
    assert nvcpp.parse_diagnostics(f"nvlink error   : {message}\n") == [
        compile_diag("error", None, None, None, None, message)
    ]
    assert nvcpp.parse_diagnostics(no_colon + "\n") == []
    assert time.monotonic() - began < 2.0


def test_a_long_nvcpp_driver_line_parses_in_linear_time() -> None:
    # P0.17. A driver message can quote model-chosen text, so a driver line can be very long. The cases below are a
    # message with many GCC style places, one with a long blank run, and a line that repeats the driver prefix with
    # no "-" after the severity word.
    places = "a:1:2: error: b " * 20_000
    blanks = "a" + " " * 50_000 + "b"
    no_dash = "nvc++-Error " * 30_000
    began = time.monotonic()
    assert nvcpp.parse_diagnostics(f"nvc++-Fatal-{places}\n") == [
        compile_diag("error", None, None, None, None, places.rstrip())
    ]
    assert nvcpp.parse_diagnostics(f"nvc++-Error-{blanks}\n") == [compile_diag("error", None, None, None, None, blanks)]
    assert nvcpp.parse_diagnostics(no_dash + "\n") == []
    assert time.monotonic() - began < 2.0


# ---------------------------------------------------------------------------
# The GCC column
#
# Under nvcc a GCC diagnostic never keeps GCC's column. GCC preprocesses a .cu file as it is but compiles the
# host code cudafe1 regenerated from it, and echoes the line from disk either way (capture
# nvcc_host_gcc_warning), so nothing in stderr shows whether a column indexes the named file. nvc++ compiles
# the built files itself: it keeps GCC's column for a built file (a key of `files`) and drops it anywhere
# else. No committed capture shows a GCC style line from nvc++ yet.

# A host .cpp file; line 5 declares a variable it never uses, and GCC's column 9 is the 'u'.
HOST_CPP = "#include <cstdio>\n\nint main()\n{\n    int u = 0;\n    return 0;\n}\n"
UNUSED = "unused variable 'u'"


def gcc_unused_stderr(column: int = 9) -> str:
    """Return GCC's -Wunused-variable warning at main.cpp:5:<column>, with its source echo and caret lines."""
    return f"main.cpp:5:{column}: warning: {UNUSED} [-Wunused-variable]\n    5 |     int u = 0;\n      |         ^\n"


def gcc_unused(column: int | None) -> Diagnostic:
    """Return the Diagnostic of gcc_unused_stderr with `column`."""
    return compile_diag("warning", "-Wunused-variable", "main.cpp", 5, column, UNUSED)


@pytest.mark.parametrize(
    "files",
    [{}, {"main.cpp": HOST_CPP}, {"main.cpp": HOST_CPP.replace("int u = 0;", "int u = 1;")}],
    ids=["no-files", "echo-is-the-file-line", "echo-differs"],
)
def test_nvcc_gcc_column_is_always_none(files: dict[str, str]) -> None:
    # No capture shows a GCC line for a host .cpp source built by nvcc yet; even an echo equal to the file line
    # keeps no column (see the next test for why).
    assert nvcc.parse_diagnostics(gcc_unused_stderr(), files) == [gcc_unused(None)]


def test_nvcc_gcc_column_from_cudafe1_code_is_none_although_the_echo_is_the_file_line() -> None:
    # Capture nvcc_host_gcc_warning: GCC echoes line 16 of main.cu exactly as the file holds it, yet its column
    # 19 counts in cudafe1's text, where the line lost its indent; in main.cu it would point at the ';', not at
    # the '<' GCC marks (column 23). So an echo equal to the file line does not show that a column is right.
    files = scenario_files("nvcc_host_gcc_warning")
    stderr = fixture_text("nvcc_host_gcc_warning")
    echo = stderr.split("\n")[2].split("| ", 1)[1]
    assert echo == files["main.cu"].split("\n")[15]
    assert nvcc.parse_diagnostics(stderr, files) == [
        compile_diag("warning", "-Wsign-compare", "main.cu", 16, None, SIGN_COMPARE)
    ]


@pytest.mark.parametrize(
    ("files", "column"),
    [
        pytest.param({}, None, id="no-files"),
        pytest.param({"other.cpp": HOST_CPP}, None, id="not-a-built-file"),
        pytest.param({"main.cpp": HOST_CPP}, 9, id="built-file"),
        # nvc++ builds the file as it is, so the column is kept as printed without comparing the echo.
        pytest.param({"main.cpp": HOST_CPP.replace("int u = 0;", "int u = 1;")}, 9, id="built-file-echo-differs"),
    ],
)
def test_nvcpp_gcc_column_is_kept_only_on_a_built_file(files: dict[str, str], column: int | None) -> None:
    assert nvcpp.parse_diagnostics(gcc_unused_stderr(), files) == [gcc_unused(column)]


def test_nvcpp_gcc_column_zero_is_none() -> None:
    # GCC prints column 0 when it has no column; on a built file, where nvc++ keeps GCC's column, it is None.
    assert nvcpp.parse_diagnostics(gcc_unused_stderr(column=0), {"main.cpp": HOST_CPP}) == [gcc_unused(None)]


def test_nvcpp_inline_asm_error_has_no_column_with_the_files() -> None:
    # Exploratory probe nvcpp_probe_asm_int (see NVCPP_LINES): '<inline asm>' is no built file, so even with the
    # files there is no column; file and line stay as printed.
    stderr = "<inline asm>:1:2: error: invalid instruction mnemonic 'bogus.op.s32'\n"
    assert nvcpp.parse_diagnostics(stderr, {"main.cpp": HOST_CPP}) == [
        compile_diag("error", None, "<inline asm>", 1, None, "invalid instruction mnemonic 'bogus.op.s32'")
    ]


# ---------------------------------------------------------------------------
# The linker's place
#
# An undefined reference may name the source file and line of the call (capture nvcpp_linker_error). The place
# is kept only as a built file: the path is a key of `files` or ends with "/" and a key (the longest key wins),
# and the line is in that file. Otherwise file and line are None. The linker gives no column.

# Line 7 calls the undefined helper.
LINKED = "#include <cstdio>\n\nvoid helper(float *data, int n);\n\nint main()\n{\n    helper(0, 1);\n}\n"


@pytest.mark.parametrize(
    ("place", "files", "file", "line"),
    [
        # As in capture nvcpp_linker_error, with a shorter workdir.
        pytest.param("/work/run/main.cpp:7", {"main.cpp": LINKED}, "main.cpp", 7, id="absolute-path"),
        # No capture shows the three places below yet.
        pytest.param("main.cpp:7", {"main.cpp": LINKED}, "main.cpp", 7, id="relative-path"),
        pytest.param("/usr/bin/ld: /work/run/main.cpp:7", {"main.cpp": LINKED}, "main.cpp", 7, id="ld-in-front"),
        pytest.param(
            "/work/run/src/main.cpp:7",
            {"main.cpp": "int x;\n", "src/main.cpp": LINKED},
            "src/main.cpp",
            7,
            id="longest-key",
        ),
        pytest.param("/work/run/xmain.cpp:7", {"main.cpp": LINKED}, None, None, id="no-path-boundary"),
        pytest.param("/work/run/other.cpp:7", {"main.cpp": LINKED}, None, None, id="not-a-built-file"),
        pytest.param("/work/run/main.cpp:99", {"main.cpp": LINKED}, None, None, id="line-past-the-end"),
        pytest.param("/work/run/main.cpp:7", {}, None, None, id="no-files"),
        # As in capture nvcc_linker_error: a temporary file and a section offset.
        pytest.param(
            "tmpxft_00000002_00000000-6_main.cudafe1.cpp:(.text.startup+0x2c)",
            {"main.cpp": LINKED},
            None,
            None,
            id="section-offset",
        ),
    ],
)
@pytest.mark.parametrize("module", [nvcc, nvcpp], ids=["nvcc", "nvcpp"])
def test_linker_place_is_kept_only_as_a_built_file(
    module: ModuleType, place: str, files: dict[str, str], file: str | None, line: int | None
) -> None:
    stderr = f"{place}: {HELPER_UNDEFINED}\n"
    assert module.parse_diagnostics(stderr, files) == [compile_diag("error", None, file, line, None, HELPER_UNDEFINED)]


def test_a_ptx_place_is_no_built_file_even_when_a_built_file_has_its_name() -> None:
    # P0.17. ptxas reads the PTX nvcc wrote under @lassi-tmp, the compile's private TMPDIR, which build() never
    # writes, and nvcc is never given a .ptx source. So a built file at the workdir root with the same name, even one
    # with a line 28, is another file. File and line stay None, and the place stays in the message.
    files = {"main.cu": "int x;\n", "tmpxft_00000002_00000000-6_main.ptx": "// PLACEHOLDER\n" * 40}
    assert nvcc.parse_diagnostics(PTXAS_PLACE_LINE + "\n", files) == [
        compile_diag("error", None, None, None, None, PTXAS_PLACE_MESSAGE)
    ]


# ---------------------------------------------------------------------------
# Numbers too long to be a line

HUGE = "5" * 5000


@pytest.mark.parametrize(
    ("module", "stderr", "expected"),
    [
        pytest.param(nvcc, f"main.cu({HUGE}): error: x\n", [], id="nvcc-edg-line"),
        pytest.param(nvcc, f"main.cu:1:{HUGE}: error: x\n", [], id="nvcc-gcc-column"),
        pytest.param(nvcc, f"main.cu:{HUGE}:1: error: x\n", [], id="nvcc-gcc-line"),
        pytest.param(nvcpp, f'"main.cpp", line {HUGE}: error: x\n', [], id="nvcpp-edg-line"),
        pytest.param(nvcpp, f"main.cpp:1:{HUGE}: error: x\n", [], id="nvcpp-gcc-column"),
        pytest.param(
            nvcpp,
            f"NVC++-S-0155-x (main.cpp: {HUGE})\n",
            [compile_diag("error", "S-0155", None, None, None, f"x (main.cpp: {HUGE})")],
            id="nvcpp-backend-line",
        ),
        pytest.param(
            nvcpp,
            f"/work/main.cpp:{HUGE}: {HELPER_UNDEFINED}\n",
            [compile_diag("error", None, None, None, None, HELPER_UNDEFINED)],
            id="nvcpp-linker-line",
        ),
        # P0.17. The PTX place is part of the ptxas place format, as the place is part of the EDG and GCC formats.
        # A line number past 10 digits is no place, so the line is not read as that format, and no other pattern
        # reads it. The 10-digit line is in NVCC_LINES (ptxas-ptx-line-ten-digits).
        pytest.param(nvcc, f"ptxas {SHORT_PTX}, line {HUGE}; error   : x\n", [], id="nvcc-ptxas-ptx-line"),
        # P0.17. nvlink names objects, never a source line. An object path that ends like "<built file>:<number>"
        # stays in the message, and no place is read from it.
        pytest.param(
            nvcpp,
            f"nvlink error   : Undefined reference to 'x' in '/work/main.cpp:{HUGE}'\n",
            [compile_diag("error", None, None, None, None, f"Undefined reference to 'x' in '/work/main.cpp:{HUGE}'")],
            id="nvcpp-nvlink-object",
        ),
        # P0.17. The nvc++ driver line names no place. A GCC style place in its message stays in the message, and no
        # line or column is read from it, however long its numbers.
        pytest.param(
            nvcpp,
            f"nvc++-Error-main.cpp:{HUGE}:1: error: x\n",
            [compile_diag("error", None, None, None, None, f"main.cpp:{HUGE}:1: error: x")],
            id="nvcpp-driver-message",
        ),
    ],
)
def test_a_number_past_ten_digits_is_never_a_line_or_column(
    module: ModuleType, stderr: str, expected: list[Diagnostic]
) -> None:
    # Model-controlled text can reach stderr, and int() refuses a digit string past 4300 digits. Every line
    # and column is read up to 10 digits, enough for any #line (at most 2147483647), so parsing never raises.
    assert module.parse_diagnostics(stderr, {"main.cu": "int x;\n", "main.cpp": "int x;\n"}) == expected


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
    # Line 9 of a two-line file: the rule is about lines; a caret past the end of a line is the next test.
    stderr = edg_header(tool, 9, 'identifier "b" is undefined') + "  int a = b;\n" + " " * 10 + "^\n"
    expected = [edg_error(tool, 9, None, 'identifier "b" is undefined')]
    assert edg_parse(tool, stderr, "int x;\nint a = b;\n") == expected


@pytest.mark.parametrize("tool", TOOLS)
def test_a_caret_one_past_the_end_of_the_line_keeps_its_column(tool: str) -> None:
    # As in capture nvcpp_missing_include: the caret sits after the closing quote. Line 2 has 14 characters and
    # the echo prefix is 2, so caret index 16 gives column 15, which the echo match keeps as the compiler gave it.
    stderr = edg_header(tool, 2, 'cannot open source file "x.h"') + '  #include "x.h"\n' + " " * 16 + "^\n"
    expected = [edg_error(tool, 2, 15, 'cannot open source file "x.h"')]
    assert edg_parse(tool, stderr, '#include <a>\n#include "x.h"\n') == expected


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
    # so caret index 4 - 2 + 1 = column 3, the 'f' in '  foo(p);'. No capture shows continuation lines yet.
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
    result = case.factory(runner=FakeRunner(returncode=1, stderr=stderr)).build(
        scenario_files(case.error_fixture), workdir
    )
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
    result = case.factory(runner=runner).build(scenario_files(case.warning_fixture), make_workdir(tmp_path))
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
    result = case.factory(runner=runner).build(scenario_files(case.warning_fixture), make_workdir(tmp_path))
    assert result.diagnostics[:-1] == fixture.expected
    exit_status_check(case, result.diagnostics[-1], "2")


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_no_exit_status_error_when_an_error_was_parsed(preset: str, tmp_path: Path) -> None:
    case = PRESETS[preset]
    fixture = FIXTURE_CASES[case.error_fixture]
    runner = FakeRunner(returncode=2, stderr=fixture_text(case.error_fixture))
    result = case.factory(runner=runner).build(scenario_files(case.error_fixture), make_workdir(tmp_path))
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
    result = case.factory(runner=runner).build(scenario_files(case.error_fixture), make_workdir(tmp_path))
    assert result.diagnostics[:-1] == fixture.expected
    last = result.diagnostics[-1]
    assert (last.code, last.severity) == ("exit-status", "error")
    assert "timed out" in last.message, last.message


def test_a_missing_header_reaches_the_model_as_a_located_error(tmp_path: Path) -> None:
    # Capture nvcpp_missing_include: nvc++ reports a missing header as an EDG catastrophic error, which replaces
    # the exit-status fallback.
    fixture = FIXTURE_CASES["nvcpp_missing_include"]
    runner = FakeRunner(returncode=2, stderr=fixture_text("nvcpp_missing_include"))
    result = nvcpp.NvcppCc80(runner=runner).build(scenario_files("nvcpp_missing_include"), make_workdir(tmp_path))
    assert result.diagnostics == fixture.expected
    assert result.artifact is None


@pytest.mark.parametrize("name", ["nvcc_ptxas_inline_asm", "nvcpp_nvlink_error"])
def test_a_p017_capture_builds_without_the_exit_status_error(name: str, tmp_path: Path) -> None:
    # P0.17 acceptance: each capture parses into its errors, so the build never falls back to "exit-status". The
    # fake compiler returns the fixture with the exit status captures.json records for it (255 for ptxas under
    # nvcc, 2 for nvlink under nvc++).
    case = FIXTURE_CASES[name]
    status = load_captures()["scenarios"][name]["exit_status"]
    assert status != 0, name
    runner = FakeRunner(returncode=status, stderr=fixture_text(name))
    result = PRESET_OF[case.module](runner=runner).build(scenario_files(name), make_workdir(tmp_path))
    assert result.diagnostics == case.expected
    assert all(diagnostic.code != "exit-status" for diagnostic in result.diagnostics)
    assert result.artifact is None


@pytest.mark.parametrize(
    ("stderr", "status", "expected"),
    [
        pytest.param(
            NVCPP_DRIVER_ERROR + "\n",
            1,
            [compile_diag("error", None, None, None, None, NVCPP_DRIVER_ERROR.removeprefix("nvc++-Error-"))],
            id="driver-error",
        ),
        pytest.param(NVCPFE_CRASH_STDERR, 2, NVCPFE_CRASH_DIAGNOSTICS, id="driver-fatal"),
    ],
)
def test_an_nvcpp_driver_line_replaces_the_exit_status_error(
    stderr: str, status: int, expected: list[Diagnostic], tmp_path: Path
) -> None:
    # P0.17 acceptance: the driver's own message reaches the model as the error, not the "exit-status" fallback.
    # The status is the one on record: 1 in plans/spikes/p0-toolchains-verify.md run 2a, 2 in the probe capture.
    runner = FakeRunner(returncode=status, stderr=stderr)
    result = nvcpp.NvcppCc80(runner=runner).build({"main.cpp": "int main() { return 0; }\n"}, make_workdir(tmp_path))
    assert result.diagnostics == expected
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
    # P0.16 R4 added the two truncation flags, which default to False (tests/toolchains/test_runner_caps.py).
    fields = ["returncode", "stdout", "stderr", "stdout_truncated", "stderr_truncated"]
    assert [f.name for f in dataclasses.fields(CommandResult)] == fields
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
