"""Tests for the GCC diagnostics of the native C++ toolchain `gcc-native` (task P4.5).

Bible: Component Interfaces (Toolchain row and contract rules: diagnostics
parsed into (severity, code, file, line, column, message, stage), raw
stderr kept as an attachment), Result Record (Diagnostic), Toolchain Pins,
Agent Rules 1 and 10.

Two halves.

Parser rules, pinned now on SYNTHETIC lines hand-written in GCC's format
(none of them captured), through the registered toolchain's parse(stderr,
files):

- `<file>:<line>:<column>: <severity>: <message>` is one Diagnostic, stage
  compile. Severity: error, warning, and note as printed; `fatal error` is
  an error. The message is the text after `<severity>: ` as printed (a
  nested note keeps its leading blanks), without a trailing
  ` [<option>]`, which becomes the code when the bracket starts with "-"
  (`-Wunused-variable`, `-Werror=unused-variable`, `-fpermissive`); any
  other trailing bracket stays in the message and the code is None.
- File and line are kept as printed. GCC counts columns in the named file
  itself, so the column is kept when that file is a built file (a key of
  `files`) and is None otherwise (a system header, the harness header, or
  a parse with no files).
- No Diagnostic comes from context lines ("In file included from ...", its
  "from ..." continuation, "<file>: In function ...", "At global scope",
  "In instantiation of ...", "... required from here"), from the source
  echo, caret, and fix-it lines, or from "compilation terminated.".
- The linker's undefined reference and the collect2 line are errors with no
  place, as the shared linker patterns read them; a driver line such as
  "g++-12: fatal error: <message>" is an error with no place.

Captured fixtures, in tests/toolchains/fixtures/gcc/ (captured on alpha01 from clean
commit 94368a4 in rx 20260925-195420-desktop-8r113ei-detached-94368a41-e825; see README.md):

- scenarios.json and sources/<case>/ (written here) name the six cases:
  gcc_syntax_error, gcc_undeclared_identifier, gcc_wall_warning,
  gcc_note_chain, gcc_header_error, and gcc_clean. The capture tool reads
  that set with `--fixtures tests/toolchains/fixtures/gcc`.
- <case>.stderr: the raw stderr of one compile, copied byte for byte from
  the capture's compile.stderr, plain ASCII with LF line endings.
- <case>.json: an object {"derivation": <plain ASCII text saying how the
  list was read from the stderr>, "diagnostics": [<Diagnostic as an object
  with exactly the keys stage, severity, code, file, line, column,
  message>]}, derived by hand from the stderr, never copied from the
  parser's output.
- captures.json: the capture tool's manifest.json, copied byte for byte;
  README.md: the capture's provenance, one table row per case.

The fixture tests check the capture record (clean commit, the build host,
the gcc pin, the argv the adapter builds), the exact parse of each stderr,
and what each case must show, read from its source tree: the error or
warning names the right built file, its line holds the named identifier,
and its column points at it.

No test runs a compiler; the parser reads strings and files. No value in
this module is a measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

import lassi.toolchains  # noqa: F401  (importing the package registers every toolchain preset)
from lassi.core.record import Diagnostic
from lassi.core.registry import DEFAULT_REGISTRY, RegistryError
from lassi.toolchains import CommandResult
from lassi.toolchains.pins import read_pin

REPO = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO / "tools"
TOOL_MODULE = "capture_toolchain_fixtures"
GCC_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gcc"
SCENARIOS = GCC_FIXTURES / "scenarios.json"
SOURCES = GCC_FIXTURES / "sources"
CAPTURES = GCC_FIXTURES / "captures.json"
README = GCC_FIXTURES / "README.md"

NAME = "gcc-native"
PIN_NAME = "gcc"
CASES = (
    "gcc_syntax_error",
    "gcc_undeclared_identifier",
    "gcc_wall_warning",
    "gcc_note_chain",
    "gcc_header_error",
    "gcc_clean",
)
DIAGNOSTIC_KEYS = frozenset({"stage", "severity", "code", "file", "line", "column", "message"})
STAND_IN = "/usr/bin/g++-12"


def gcc_class() -> type:
    """Return the Toolchain class registered as gcc-native; fail the test clearly while it is missing."""
    try:
        return DEFAULT_REGISTRY.get("Toolchain", NAME).factory
    except RegistryError as error:
        pytest.fail(f"no Toolchain is registered as {NAME!r} ({error}); task P4.5 adds it to lassi.toolchains")


def refuse_to_run(argv: Any, cwd: Path, timeout_s: float) -> CommandResult:
    """A CommandRunner for parse-only tests: running anything fails the test."""
    raise AssertionError(f"a parse test ran a command: {argv!r}")


def parse(stderr: str, files: dict[str, str] | None = None) -> list[Diagnostic]:
    """Return what the gcc-native toolchain parses from `stderr` with the built `files` (none by default)."""
    toolchain = gcc_class()(executable=STAND_IN, runner=refuse_to_run)
    return toolchain.parse(stderr, files or {})


def diag(
    severity: str, code: str | None, file: str | None, line: int | None, column: int | None, message: str
) -> Diagnostic:
    """Return a compile-stage Diagnostic with every other field given explicitly."""
    return Diagnostic(
        stage="compile", severity=severity, code=code, file=file, line=line, column=column, message=message
    )


# ---------------------------------------------------------------------------
# Parser rules on SYNTHETIC lines in GCC's format (not captured)

# A SYNTHETIC main.cpp of 20 lines; the parser never checks a column against it, it only needs the file to be built.
MAIN = "".join(f"// SYNTHETIC line {number}\n" for number in range(1, 21))
SCALE_H = "".join(f"// SYNTHETIC header line {number}\n" for number in range(1, 9))


def test_an_error_on_a_built_file_keeps_file_line_and_column() -> None:
    stderr = (
        "main.cpp: In function 'int main()':\n"
        "main.cpp:7:30: error: 'undefined_var' was not declared in this scope\n"
        "    7 |         y[i] = 2.0f * y[i] + undefined_var;\n"
        "      |                              ^~~~~~~~~~~~~\n"
    )
    assert parse(stderr, {"main.cpp": MAIN}) == [
        diag("error", None, "main.cpp", 7, 30, "'undefined_var' was not declared in this scope")
    ]


def test_without_the_built_files_the_column_is_none() -> None:
    stderr = "main.cpp:7:30: error: 'undefined_var' was not declared in this scope\n"
    assert parse(stderr) == [diag("error", None, "main.cpp", 7, None, "'undefined_var' was not declared in this scope")]


@pytest.mark.parametrize(
    ("line", "severity", "code", "message"),
    [
        pytest.param(
            "main.cpp:7:9: warning: unused variable 'unused' [-Wunused-variable]",
            "warning",
            "-Wunused-variable",
            "unused variable 'unused'",
            id="warning-flag",
        ),
        pytest.param(
            "main.cpp:7:9: error: unused variable 'unused' [-Werror=unused-variable]",
            "error",
            "-Werror=unused-variable",
            "unused variable 'unused'",
            id="werror-flag",
        ),
        pytest.param(
            "main.cpp:3:14: error: invalid conversion from 'void*' to 'int*' [-fpermissive]",
            "error",
            "-fpermissive",
            "invalid conversion from 'void*' to 'int*'",
            id="f-flag",
        ),
        pytest.param(
            "main.cpp:12:18: warning: array subscript 4 is above array bounds of 'float [4]' [-Warray-bounds]",
            "warning",
            "-Warray-bounds",
            "array subscript 4 is above array bounds of 'float [4]'",
            id="bracket-inside-quotes-stays",
        ),
        pytest.param(
            "main.cpp:3:6: note: SYNTHETIC message that ends in a bracket [with T = int]",
            "note",
            None,
            "SYNTHETIC message that ends in a bracket [with T = int]",
            id="bracket-that-is-no-option-stays",
        ),
        pytest.param(
            "main.cpp:5:1: error: expected ',' or ';' before 'for'",
            "error",
            None,
            "expected ',' or ';' before 'for'",
            id="no-bracket",
        ),
    ],
)
def test_a_trailing_option_in_brackets_is_the_code(line: str, severity: str, code: str | None, message: str) -> None:
    (found,) = parse(line + "\n", {"main.cpp": MAIN})
    assert (found.severity, found.code, found.message) == (severity, code, message)
    assert (found.stage, found.file) == ("compile", "main.cpp")


def test_a_note_chain_gives_one_note_each_and_a_nested_note_keeps_its_leading_blanks() -> None:
    stderr = (
        "main.cpp: In function 'int main()':\n"
        "main.cpp:18:10: error: no matching function for call to 'scale(int [4], int)'\n"
        "   18 |     scale(values, 4);\n"
        "      |     ~~~~~^~~~~~~~~~~\n"
        "main.cpp:4:6: note: candidate: 'void scale(float*, int)'\n"
        "    4 | void scale(float *x, int n) {\n"
        "      |      ^~~~~\n"
        "main.cpp:4:19: note:   no known conversion for argument 1 from 'int [4]' to 'float*'\n"
        "    4 | void scale(float *x, int n) {\n"
        "      |            ~~~~~~~^\n"
    )
    assert parse(stderr, {"main.cpp": MAIN}) == [
        diag("error", None, "main.cpp", 18, 10, "no matching function for call to 'scale(int [4], int)'"),
        diag("note", None, "main.cpp", 4, 6, "candidate: 'void scale(float*, int)'"),
        diag("note", None, "main.cpp", 4, 19, "  no known conversion for argument 1 from 'int [4]' to 'float*'"),
    ]


def test_an_error_in_an_included_header_names_the_header_and_the_include_chain_gives_nothing() -> None:
    stderr = (
        "In file included from kernels/inner.h:1,\n"
        "                 from main.cpp:2:\n"
        "kernels/scale.h: In function 'void scale(float*, int)':\n"
        "kernels/scale.h:6:17: error: 'factor' was not declared in this scope\n"
        "    6 |         x[i] *= factor;\n"
        "      |                 ^~~~~~\n"
    )
    files = {"main.cpp": MAIN, "kernels/scale.h": SCALE_H}
    message = "'factor' was not declared in this scope"
    assert parse(stderr, files) == [diag("error", None, "kernels/scale.h", 6, 17, message)]


def test_a_fatal_error_is_an_error_and_compilation_terminated_gives_nothing() -> None:
    stderr = (
        "main.cpp:2:10: fatal error: kernels/missing.h: No such file or directory\n"
        '    2 | #include "kernels/missing.h"\n'
        "      |          ^~~~~~~~~~~~~~~~~~~\n"
        "compilation terminated.\n"
    )
    assert parse(stderr, {"main.cpp": MAIN}) == [
        diag("error", None, "main.cpp", 2, 10, "kernels/missing.h: No such file or directory")
    ]


@pytest.mark.parametrize(
    "path",
    ["/usr/include/c++/12/bits/stl_vector.h", "lassi_io.h"],
    ids=["system-header", "harness-header"],
)
def test_a_file_that_is_not_built_keeps_file_and_line_but_no_column(path: str) -> None:
    stderr = f"{path}:1123:7: note: candidate: 'void push_back(const float&)'\n"
    assert parse(stderr, {"main.cpp": MAIN}) == [
        diag("note", None, path, 1123, None, "candidate: 'void push_back(const float&)'")
    ]


@pytest.mark.parametrize(
    "line",
    [
        "In file included from main.cpp:2:",
        "                 from main.cpp:3:",
        "main.cpp: In function 'int main()':",
        "main.cpp: At global scope:",
        "main.cpp: In instantiation of 'void f(T) [with T = int]':",
        "main.cpp:9:6:   required from here",
        "compilation terminated.",
        "  +++ |+#include <cstdio>",
        "    3 | // a.cpp:1:2: error: SYNTHETIC text in a source echo",
        "      |     ^~~~",
        "",
    ],
    ids=[
        "included-from",
        "from-continuation",
        "in-function",
        "at-global-scope",
        "in-instantiation",
        "required-from-here",
        "compilation-terminated",
        "fix-it",
        "source-echo",
        "caret",
        "blank",
    ],
)
def test_context_echo_and_fix_it_lines_give_no_diagnostic(line: str) -> None:
    assert parse(line + "\n", {"main.cpp": MAIN}) == []


def test_the_linker_lines_are_errors_with_no_place() -> None:
    stderr = (
        "/usr/bin/ld: @lassi-tmp/ccSYNTH01.o: in function `main':\n"
        "main.cpp:(.text.startup+0x1b): undefined reference to `helper(float*, int)'\n"
        "collect2: error: ld returned 1 exit status\n"
    )
    assert parse(stderr, {"main.cpp": MAIN}) == [
        diag("error", None, None, None, None, "undefined reference to `helper(float*, int)'"),
        diag("error", None, None, None, None, "ld returned 1 exit status"),
    ]


@pytest.mark.parametrize("driver", ["g++-12", "g++"])
def test_a_driver_fatal_error_is_an_error_with_no_place(driver: str) -> None:
    stderr = f"{driver}: fatal error: Killed signal terminated program cc1plus\ncompilation terminated.\n"
    assert parse(stderr, {"main.cpp": MAIN}) == [
        diag("error", None, None, None, None, "Killed signal terminated program cc1plus")
    ]


# ---------------------------------------------------------------------------
# The captured fixtures


def load_json(path: Path) -> Any:
    """Return a plain ASCII JSON file's value; fail the test clearly while the file is missing."""
    if not path.is_file():
        pytest.fail(f"{path} does not exist; task P4.5 captures the gcc fixtures on the build host")
    raw = path.read_bytes()
    assert raw.isascii(), f"{path} is not plain ASCII"
    assert b"\r" not in raw, f"{path} has a CR"
    return json.loads(raw.decode("ascii"))


def stderr_bytes(case: str) -> bytes:
    """Return the captured stderr of `case` as stored; fail the test clearly while it is missing."""
    path = GCC_FIXTURES / f"{case}.stderr"
    if not path.is_file():
        pytest.fail(f"{path} does not exist; task P4.5 captures it with tools/capture_toolchain_fixtures.py")
    return path.read_bytes()


def source_files(case: str) -> dict[str, str]:
    """Return the files `case` compiled: relative POSIX path -> text, with no newline translation."""
    tree = SOURCES / case
    paths = sorted(path for path in tree.rglob("*") if path.is_file())
    return {path.relative_to(tree).as_posix(): path.read_bytes().decode("utf-8") for path in paths}


def expected(case: str) -> list[Diagnostic]:
    """Return the Diagnostic list <case>.json holds, after checking its form."""
    data = load_json(GCC_FIXTURES / f"{case}.json")
    assert isinstance(data, dict) and set(data) == {"derivation", "diagnostics"}, f"{case}.json: object form"
    assert isinstance(data["derivation"], str) and data["derivation"].strip(), f"{case}.json: say how it was read"
    items = data["diagnostics"]
    assert isinstance(items, list)
    for item in items:
        assert isinstance(item, dict) and set(item) == DIAGNOSTIC_KEYS, f"{case}.json: {item!r}"
    return [Diagnostic(**item) for item in items]


def captured(case: str) -> list[Diagnostic]:
    """Return what the parser reads from the captured stderr of `case` with its source tree as the built files."""
    return parse(stderr_bytes(case).decode("utf-8"), source_files(case))


def entry(case: str) -> dict[str, Any]:
    """Return the capture manifest's entry for `case`."""
    scenarios = load_json(CAPTURES)["scenarios"]
    assert case in scenarios, f"captures.json has no entry for {case}"
    return scenarios[case]


def line_of(case: str, file: str, text: str) -> int:
    """Return the 1-based number of the one line of `file` in the source tree of `case` that holds `text`."""
    lines = source_files(case)[file].split("\n")
    numbers = [number for number, line in enumerate(lines, start=1) if text in line]
    assert len(numbers) == 1, f"{case}/{file}: {text!r} must be on exactly one line"
    return numbers[0]


def points_at(case: str, diagnostic: Diagnostic, text: str) -> bool:
    """Return True when the diagnostic's column, in its own file of the source tree, starts `text`."""
    if diagnostic.file is None or diagnostic.line is None or diagnostic.column is None:
        return False
    line = source_files(case)[diagnostic.file].split("\n")[diagnostic.line - 1]
    return line[diagnostic.column - 1 :].startswith(text)


def tool() -> Any:
    """Import and return tools/capture_toolchain_fixtures.py."""
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    return importlib.import_module(TOOL_MODULE)


def test_the_scenario_set_names_the_six_cases_each_with_its_source_tree() -> None:
    scenarios = load_json(SCENARIOS)
    assert sorted(scenarios) == sorted(CASES)
    for case in CASES:
        assert scenarios[case]["toolchain"] == NAME, case
        assert (SOURCES / case / "main.cpp").is_file(), case
    assert sorted(path.name for path in SOURCES.iterdir()) == sorted(CASES)


def test_the_capture_tool_reads_the_gcc_scenario_set() -> None:
    loaded = tool().load_scenarios(SCENARIOS, SOURCES)
    assert sorted(loaded) == sorted(CASES)
    assert {scenario.toolchain for scenario in loaded.values()} == {NAME}


def test_the_capture_tool_offers_a_fixtures_directory_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as done:
        tool().main(["--help"])
    assert done.value.code == 0
    assert "--fixtures" in capsys.readouterr().out, "the tool captures another fixture set with --fixtures DIR"


def test_the_capture_tool_captures_the_set_the_fixtures_option_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tool()
    for name in ("TMPDIR", "LASSI_RUNS_ROOT"):
        (tmp_path / name).mkdir()
        monkeypatch.setenv(name, str(tmp_path / name))
    monkeypatch.delenv("LASSI_SCRATCH", raising=False)
    monkeypatch.delenv("LASSI_TOOLCHAINS", raising=False)
    chosen: list[str] = []

    def fake_capture(scenarios: Any, out: Path, root: Any, shown: bool) -> dict[str, Any]:
        chosen.extend(scenario.name for scenario in scenarios)
        out.mkdir(parents=True, exist_ok=True)
        return {"scenarios": {}}

    monkeypatch.setattr(module, "capture", fake_capture)
    status = module.main(["--fixtures", str(GCC_FIXTURES), "--out", str(tmp_path / "out")])
    assert status == 0
    assert sorted(chosen) == sorted(CASES), "--fixtures DIR captures DIR/scenarios.json over DIR/sources"


def test_every_captured_file_belongs_to_a_case() -> None:
    stems = sorted(path.stem for path in GCC_FIXTURES.glob("*.stderr"))
    assert stems == sorted(CASES), "one <case>.stderr per case, and no other"
    manifests = {SCENARIOS.name, CAPTURES.name}
    lists = sorted(path.stem for path in GCC_FIXTURES.glob("*.json") if path.name not in manifests)
    assert lists == sorted(CASES), "one <case>.json per case, and no other"


@pytest.mark.parametrize("case", CASES)
def test_the_fixture_is_raw_ascii_stderr(case: str) -> None:
    raw = stderr_bytes(case)
    assert raw.isascii(), case
    assert b"\r" not in raw, f"{case} must be stored with LF line endings"
    if raw:
        assert raw.endswith(b"\n"), case
        assert not raw.startswith((b"#", b"//")), f"{case} starts with a comment; fixtures are raw stderr"


def test_the_capture_ran_from_a_clean_commit_on_the_build_host_with_the_pinned_gpp() -> None:
    captures = load_json(CAPTURES)
    assert captures["dirty"] is False
    assert captures["snapshot_of"] is None
    assert re.fullmatch(r"[0-9a-f]{40}", captures["commit"]), captures["commit"]
    assert captures["rx_run_id"], "the capture comes from a recorded rx run"
    assert captures["host"] == "alpha01"
    assert sorted(captures["scenarios"]) == sorted(CASES)
    assert sorted(captures["toolchains"]) == [NAME]
    record = captures["toolchains"][NAME]
    pin = read_pin(PIN_NAME)
    assert record["pins"] == {PIN_NAME: pin["VERSION"]}
    assert record["executable"] == pin["EXECUTABLE"]
    assert record["version_exit_status"] == 0
    assert pin["EXPECT_VERSION"] in "\n".join(record["version"])
    assert record["locale"] == {"LANG": "C", "LC_ALL": "C"}


@pytest.mark.parametrize("case", CASES)
def test_the_fixture_is_the_captured_stderr_byte_for_byte(case: str) -> None:
    raw = stderr_bytes(case)
    assert hashlib.sha256(raw).hexdigest() == entry(case)["stderr_sha256"], case
    assert len(raw) == entry(case)["stderr_bytes"], case


@pytest.mark.parametrize("case", CASES)
def test_the_capture_compiled_the_case_source_tree_with_the_adapter_command(case: str) -> None:
    record = entry(case)
    assert record["toolchain"] == NAME and record["overrides"] == {}
    preset = gcc_class()
    sources = sorted(path for path in source_files(case) if path.endswith(preset.SOURCE_SUFFIXES))
    assert sources, case
    assert record["argv"][0] == read_pin(PIN_NAME)["EXECUTABLE"]
    assert record["argv"] == preset(executable=record["argv"][0], runner=refuse_to_run).command(sources)


@pytest.mark.parametrize("case", CASES)
def test_the_captured_stderr_parses_into_the_hand_derived_list(case: str) -> None:
    assert captured(case) == expected(case)
    assert entry(case)["diagnostics"] == len(expected(case))


@pytest.mark.parametrize("case", CASES)
def test_without_the_built_files_only_the_columns_change(case: str) -> None:
    bare = parse(stderr_bytes(case).decode("utf-8"))
    assert bare == [dataclasses.replace(item, column=None) for item in expected(case)]


@pytest.mark.parametrize("case", CASES)
def test_a_failed_capture_parses_into_an_error_and_a_clean_one_into_none(case: str) -> None:
    status = entry(case)["exit_status"]
    errors = [item for item in expected(case) if item.severity == "error"]
    assert bool(errors) == (status != 0), (case, status)


def test_the_clean_case_has_empty_stderr_and_exit_status_zero() -> None:
    assert stderr_bytes("gcc_clean") == b""
    assert entry("gcc_clean")["exit_status"] == 0
    assert expected("gcc_clean") == []


def test_the_syntax_error_is_an_error_on_main_cpp() -> None:
    found = captured("gcc_syntax_error")
    errors = [item for item in found if item.severity == "error"]
    assert errors, "a syntax error gives at least one error"
    assert {item.file for item in found} == {"main.cpp"}, "every diagnostic names the one built file"
    lines = source_files("gcc_syntax_error")["main.cpp"].split("\n")
    assert all(item.line is not None and 1 <= item.line <= len(lines) for item in found)


def test_the_undeclared_identifier_error_points_at_the_identifier() -> None:
    case = "gcc_undeclared_identifier"
    errors = [item for item in captured(case) if item.severity == "error" and "undefined_var" in item.message]
    assert errors, "the error names the undeclared identifier"
    assert errors[0].file == "main.cpp"
    assert errors[0].line == line_of(case, "main.cpp", "undefined_var;")
    assert points_at(case, errors[0], "undefined_var"), "the column counts in main.cpp and points at the name"


def test_the_wall_warning_names_its_flag_and_points_at_the_variable() -> None:
    case = "gcc_wall_warning"
    assert entry(case)["exit_status"] == 0, "a warning alone does not fail the build"
    warnings = [item for item in captured(case) if item.severity == "warning" and item.code == "-Wunused-variable"]
    assert len(warnings) == 1, "the pragma turns on -Wunused-variable, which names the unused variable once"
    (warning,) = warnings
    assert warning.file == "main.cpp"
    assert warning.line == line_of(case, "main.cpp", "int unused")
    assert points_at(case, warning, "unused")


def test_the_note_chain_follows_the_error_with_a_note_per_candidate() -> None:
    case = "gcc_note_chain"
    found = captured(case)
    assert found and found[0].severity == "error", "the chain starts with the error"
    assert (found[0].file, found[0].line) == ("main.cpp", line_of(case, "main.cpp", "scale(values, 4);"))
    notes = [item for item in found[1:] if item.severity == "note"]
    assert len(notes) >= 2, "each overload is a candidate note"
    assert {item.file for item in notes} == {"main.cpp"}
    candidate_lines = {line_of(case, "main.cpp", "void scale(float"), line_of(case, "main.cpp", "void scale(double")}
    assert candidate_lines <= {item.line for item in notes}, "a note points at each candidate's declaration"


def test_the_header_error_names_the_header_and_not_the_include_line() -> None:
    case = "gcc_header_error"
    found = captured(case)
    errors = [item for item in found if item.severity == "error"]
    assert errors and {item.file for item in errors} == {"kernels/scale.h"}
    first = errors[0]
    assert first.line == line_of(case, "kernels/scale.h", "factor;")
    assert points_at(case, first, "factor")
    include_line = line_of(case, "main.cpp", '#include "kernels/scale.h"')
    assert not [item for item in found if (item.file, item.line) == ("main.cpp", include_line)], (
        "the include chain is context, never a diagnostic"
    )


def test_the_readme_records_the_capture_without_placeholder() -> None:
    if not README.is_file():
        pytest.fail(f"{README} does not exist; task P4.5 records the capture's provenance there")
    text = README.read_text(encoding="ascii")
    captures = load_json(CAPTURES)
    assert "PLACEHOLDER" not in text
    assert captures["commit"][:7] in text and captures["rx_run_id"] in text
    assert "captures.json" in text
    rows = [line for line in text.splitlines() if line.startswith("|")]
    for case in CASES:
        row = [line for line in rows if f"`{case}.stderr`" in line]
        assert len(row) == 1, f"README has one table row for {case}"
        assert f"| {entry(case)['exit_status']} |" in row[0], f"the row of {case} gives its exit status"
