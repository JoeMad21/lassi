"""Tests for the host compile diagnostics of the tt-metal host toolchain `ttmetal-host` (task P4.10).

Bible: Component Interfaces (Toolchain row and contract rules: diagnostics
parsed into (severity, code, file, line, column, message, stage), raw
stderr kept as an attachment), Result Record (Diagnostic), Toolchain Pins,
Agent Rules 1 and 10.

The host compiler is the pinned clang++-20 (toolchains/tt-metal.pin
EXECUTABLE), linking with ld.lld through -fuse-ld=lld (the pinned build's
LINK_FLAGS). Two halves.

Parser rules, pinned now on SYNTHETIC lines hand-written in clang's and
lld's formats (none of them captured), through the registered toolchain's
parse(stderr, files):

- `<file>:<line>:<column>: <severity>: <message>` is one Diagnostic, stage
  compile, as for GCC: error, warning, and note as printed, `fatal error`
  read as an error, and a trailing ` [<option>]` whose option starts with
  "-" (clang's `[-Werror,-Wunused-parameter]` included) becomes the code.
  File and line are kept as printed, except that a leading "./" is
  dropped: clang may name a header it found beside the including file that
  way, and without it the name is the built file's. The column is kept only
  on a built file (a key of `files`), and is None for a header of the
  pinned tree, a harness file, or a parse with no files.
- No Diagnostic comes from the include chain ("In file included from
  ..."), the source echo and caret lines, the "N errors generated." summary,
  or lld's ">>> " context lines.
- lld's `ld.lld: <severity>: <message>` and the clang driver's
  `clang++-20: <severity>: <message>` (also clang++ and clang, with an
  optional version suffix) are Diagnostics with no place; they are tried
  before the place pattern, since their message may quote a name the model
  chose.

Captured fixtures, in tests/toolchains/fixtures/ttmetal/, captured on
alpha01 from a clean commit with `tools/capture_toolchain_fixtures.py
--fixtures tests/toolchains/fixtures/ttmetal` (task P4.10, after the
toolchain exists):

- scenarios.json and sources/<case>/ (written with these tests) name the
  eight cases in CASES. Every source is a small host program written for
  these fixtures; none is upstream text. ttm_clean also holds a kernel under
  kernels/ that includes a device header, so its clean build shows that the
  host compiler never builds a kernel source.
- <case>.stderr: the raw stderr of one compile, copied byte for byte from
  the capture's compile.stderr, plain ASCII with LF line endings.
- <case>.json: {"derivation": <plain ASCII text saying how the list was
  read from the stderr>, "diagnostics": [<Diagnostic as an object with
  exactly the keys stage, severity, code, file, line, column, message>]},
  derived by hand from the stderr, never copied from the parser's output.
- captures.json: the capture tool's manifest.json, copied byte for byte;
  README.md: the capture's provenance, one table row per case.

The fixture tests check the capture record (clean commit, the build host,
the tt-metal pin, the argv the adapter builds against the tree the capture
used), the exact parse of each stderr, and what each case must show, read
from its source tree. No test runs a compiler. No value in this module is a
measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
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
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ttmetal"
SCENARIOS = FIXTURES / "scenarios.json"
SOURCES = FIXTURES / "sources"
CAPTURES = FIXTURES / "captures.json"
README = FIXTURES / "README.md"

NAME = "ttmetal-host"
PIN_NAME = "tt-metal"
PREFIX = "tt-metal@5280a9cf"
CASES = (
    "ttm_clean",
    "ttm_header_error",
    "ttm_linker_error",
    "ttm_missing_header",
    "ttm_note_chain",
    "ttm_syntax_error",
    "ttm_undeclared_identifier",
    "ttm_unused_parameter",
)
DIAGNOSTIC_KEYS = frozenset({"stage", "severity", "code", "file", "line", "column", "message"})
STAND_IN = "/usr/bin/clang++-20"
STAND_IN_TREE = "/mnt/nvme10/joseph_ufl/toolchains/tt-metal@5280a9cf"
SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx")
# The include word that names the tree's host API directory in the command; the capture's tree is read from it.
API_INCLUDE_SUFFIX = "/tt_metal/api"


def ttmetal_class() -> type:
    """Return the Toolchain class registered as ttmetal-host; fail the test clearly while it is missing."""
    try:
        return DEFAULT_REGISTRY.get("Toolchain", NAME).factory
    except RegistryError as error:
        pytest.fail(f"no Toolchain is registered as {NAME!r} ({error}); task P4.10 adds it to lassi.toolchains")


def refuse_to_run(argv: Any, cwd: Path, timeout_s: float) -> CommandResult:
    """A CommandRunner for parse-only tests: running anything fails the test."""
    raise AssertionError(f"a parse test ran a command: {argv!r}")


def make(executable: str = STAND_IN, tree: str = STAND_IN_TREE) -> Any:
    """Return a ttmetal-host toolchain that may parse but never run."""
    return ttmetal_class()(executable=executable, runner=refuse_to_run, tree=Path(tree))


def parse(stderr: str, files: dict[str, str] | None = None) -> list[Diagnostic]:
    """Return what the ttmetal-host toolchain parses from `stderr` with the built `files` (none by default)."""
    return make().parse(stderr, files or {})


def diag(
    severity: str, code: str | None, file: str | None, line: int | None, column: int | None, message: str
) -> Diagnostic:
    """Return a compile-stage Diagnostic with every other field given explicitly."""
    return Diagnostic(
        stage="compile", severity=severity, code=code, file=file, line=line, column=column, message=message
    )


# ---------------------------------------------------------------------------
# Parser rules on SYNTHETIC lines in clang's and lld's formats (not captured)

# A SYNTHETIC main.cpp of 20 lines; the parser never checks a column against it, it only needs the file to be built.
MAIN = "".join(f"// SYNTHETIC line {number}\n" for number in range(1, 21))
SCALE_H = "".join(f"// SYNTHETIC header line {number}\n" for number in range(1, 9))
# A header of the pinned tree, as clang prints it: an absolute path under the install.
TREE_HEADER = f"{STAND_IN_TREE}/tt_metal/api/tt-metalium/host_api.hpp"


def test_an_error_on_a_built_file_keeps_file_line_and_column() -> None:
    stderr = (
        "main.cpp:7:18: error: use of undeclared identifier 'undefined_var'\n"
        "    7 |         total += undefined_var;\n"
        "      |                  ^\n"
        "1 error generated.\n"
    )
    assert parse(stderr, {"main.cpp": MAIN}) == [
        diag("error", None, "main.cpp", 7, 18, "use of undeclared identifier 'undefined_var'")
    ]


def test_without_the_built_files_the_column_is_none() -> None:
    stderr = "main.cpp:7:18: error: use of undeclared identifier 'undefined_var'\n"
    assert parse(stderr) == [diag("error", None, "main.cpp", 7, None, "use of undeclared identifier 'undefined_var'")]


@pytest.mark.parametrize(
    ("line", "severity", "code", "message"),
    [
        pytest.param(
            "main.cpp:4:39: error: unused parameter 'factor' [-Werror,-Wunused-parameter]",
            "error",
            "-Werror,-Wunused-parameter",
            "unused parameter 'factor'",
            id="werror-pair",
        ),
        pytest.param(
            "main.cpp:5:9: warning: unused variable 'unused' [-Wunused-variable]",
            "warning",
            "-Wunused-variable",
            "unused variable 'unused'",
            id="warning-flag",
        ),
        pytest.param(
            "main.cpp:5:18: error: expected ';' at end of declaration",
            "error",
            None,
            "expected ';' at end of declaration",
            id="no-bracket",
        ),
    ],
)
def test_a_trailing_option_in_brackets_is_the_code(line: str, severity: str, code: str | None, message: str) -> None:
    (found,) = parse(line + "\n", {"main.cpp": MAIN})
    assert (found.severity, found.code, found.message) == (severity, code, message)
    assert (found.stage, found.file) == ("compile", "main.cpp")


def test_a_fatal_error_is_an_error() -> None:
    stderr = (
        "main.cpp:3:10: fatal error: 'tt-metalium/p410_no_such_header.hpp' file not found\n"
        "    3 | #include <tt-metalium/p410_no_such_header.hpp>\n"
        "      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        "1 error generated.\n"
    )
    assert parse(stderr, {"main.cpp": MAIN}) == [
        diag("error", None, "main.cpp", 3, 10, "'tt-metalium/p410_no_such_header.hpp' file not found")
    ]


def test_a_note_chain_gives_one_note_per_candidate() -> None:
    stderr = (
        "main.cpp:18:5: error: no matching function for call to 'scale'\n"
        "   18 |     scale(values, 4);\n"
        "      |     ^~~~~\n"
        "main.cpp:4:6: note: candidate function not viable: no known conversion from 'int[4]' to 'float *' "
        "for 1st argument\n"
        "    4 | void scale(float *x, int n) {\n"
        "      |      ^     ~~~~~~~~\n"
        "main.cpp:10:6: note: candidate function not viable: no known conversion from 'int[4]' to 'double *' "
        "for 1st argument\n"
        "1 error generated.\n"
    )
    first = "candidate function not viable: no known conversion from 'int[4]' to 'float *' for 1st argument"
    second = "candidate function not viable: no known conversion from 'int[4]' to 'double *' for 1st argument"
    assert parse(stderr, {"main.cpp": MAIN}) == [
        diag("error", None, "main.cpp", 18, 5, "no matching function for call to 'scale'"),
        diag("note", None, "main.cpp", 4, 6, first),
        diag("note", None, "main.cpp", 10, 6, second),
    ]


@pytest.mark.parametrize("printed", ["host/scale.h", "./host/scale.h"], ids=["as-built", "dot-slash"])
def test_an_error_in_an_included_header_names_the_header_and_the_include_chain_gives_nothing(printed: str) -> None:
    stderr = (
        "In file included from main.cpp:4:\n"
        "In file included from ./host/inner.h:1:\n"
        f"{printed}:5:20: error: use of undeclared identifier 'factor'\n"
        "    5 |     return value * factor;\n"
        "      |                    ^\n"
        "1 error generated.\n"
    )
    files = {"main.cpp": MAIN, "host/scale.h": SCALE_H}
    assert parse(stderr, files) == [diag("error", None, "host/scale.h", 5, 20, "use of undeclared identifier 'factor'")]


@pytest.mark.parametrize("path", [TREE_HEADER, "lassi_io.h"], ids=["pinned-tree-header", "harness-header"])
def test_a_file_that_is_not_built_keeps_file_and_line_but_no_column(path: str) -> None:
    message = "candidate function not viable: requires 0 arguments, but 1 was provided"
    stderr = f"{path}:88:9: note: {message}\n"
    assert parse(stderr, {"main.cpp": MAIN}) == [diag("note", None, path, 88, None, message)]


@pytest.mark.parametrize(
    "line",
    [
        "In file included from main.cpp:2:",
        "In file included from host/inner.h:1:",
        "1 error generated.",
        "2 errors generated.",
        "1 warning and 2 errors generated.",
        "1 warning generated.",
        "    5 |     int total = 0",
        "      |                  ^",
        "      |                  ;",
        "    3 | // a.cpp:1:2: error: SYNTHETIC text in a source echo",
        ">>> referenced by main.cpp",
        ">>>               @lassi-tmp/main-5a1b2c.o:(main)",
        "",
    ],
    ids=[
        "included-from",
        "included-from-nested",
        "one-error-generated",
        "errors-generated",
        "warnings-and-errors-generated",
        "warning-generated",
        "source-echo",
        "caret",
        "fix-it",
        "echo-holding-a-diagnostic",
        "lld-referenced-by",
        "lld-object",
        "blank",
    ],
)
def test_context_echo_and_summary_lines_give_no_diagnostic(line: str) -> None:
    assert parse(line + "\n", {"main.cpp": MAIN}) == []


def test_an_lld_undefined_symbol_and_the_driver_line_are_errors_with_no_place() -> None:
    stderr = (
        "ld.lld: error: undefined symbol: helper(int)\n"
        ">>> referenced by main.cpp\n"
        ">>>               @lassi-tmp/main-5a1b2c.o:(main)\n"
        "clang++-20: error: linker command failed with exit code 1 (use -v to see invocation)\n"
    )
    assert parse(stderr, {"main.cpp": MAIN}) == [
        diag("error", None, None, None, None, "undefined symbol: helper(int)"),
        diag("error", None, None, None, None, "linker command failed with exit code 1 (use -v to see invocation)"),
    ]


@pytest.mark.parametrize("driver", ["clang++-20", "clang-20", "clang++", "clang"])
def test_a_driver_error_is_an_error_with_no_place(driver: str) -> None:
    stderr = f"{driver}: error: no such file or directory: 'a.cpp:1:2: error: SYNTHETIC'\n"
    assert parse(stderr, {"main.cpp": MAIN}) == [
        diag("error", None, None, None, None, "no such file or directory: 'a.cpp:1:2: error: SYNTHETIC'")
    ]


def test_an_lld_warning_is_a_warning_with_no_place() -> None:
    message = "SYNTHETIC linker warning text"
    assert parse(f"ld.lld: warning: {message}\n", {"main.cpp": MAIN}) == [
        diag("warning", None, None, None, None, message)
    ]


# ---------------------------------------------------------------------------
# The captured fixtures


def load_json(path: Path) -> Any:
    """Return a plain ASCII JSON file's value; fail the test clearly while the file is missing."""
    if not path.is_file():
        pytest.fail(f"{path} does not exist; task P4.10 captures the ttmetal fixtures on the build host")
    raw = path.read_bytes()
    assert raw.isascii(), f"{path} is not plain ASCII"
    assert b"\r" not in raw, f"{path} has a CR"
    return json.loads(raw.decode("ascii"))


def stderr_bytes(case: str) -> bytes:
    """Return the captured stderr of `case` as stored; fail the test clearly while it is missing."""
    path = FIXTURES / f"{case}.stderr"
    if not path.is_file():
        pytest.fail(f"{path} does not exist; task P4.10 captures it with tools/capture_toolchain_fixtures.py")
    return path.read_bytes()


def source_files(case: str) -> dict[str, str]:
    """Return the files `case` compiled: relative POSIX path -> text, with no newline translation."""
    tree = SOURCES / case
    paths = sorted(path for path in tree.rglob("*") if path.is_file())
    return {path.relative_to(tree).as_posix(): path.read_bytes().decode("utf-8") for path in paths}


def host_sources(case: str) -> list[str]:
    """Return the host sources of `case` in sorted order: C++ sources outside any kernels/ directory."""
    return sorted(
        path
        for path in source_files(case)
        if path.endswith(SOURCE_SUFFIXES) and "kernels" not in PurePosixPath(path).parts[:-1]
    )


def expected(case: str) -> list[Diagnostic]:
    """Return the Diagnostic list <case>.json holds, after checking its form."""
    data = load_json(FIXTURES / f"{case}.json")
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


def test_the_scenario_set_names_the_cases_each_with_its_source_tree() -> None:
    scenarios = load_json(SCENARIOS)
    assert sorted(scenarios) == sorted(CASES)
    for case in CASES:
        assert scenarios[case]["toolchain"] == NAME, case
        assert (SOURCES / case / "main.cpp").is_file(), case
        assert host_sources(case) == ["main.cpp"], f"{case} has one host source"
    assert sorted(path.name for path in SOURCES.iterdir()) == sorted(CASES)


def test_the_clean_case_holds_a_kernel_the_host_compiler_could_not_build() -> None:
    kernels = [path for path in source_files("ttm_clean") if "kernels" in PurePosixPath(path).parts[:-1]]
    assert kernels, "ttm_clean shows that a kernel source is never a host source"
    assert all('#include "dataflow_api.h"' in source_files("ttm_clean")[path] for path in kernels)


def test_the_fixture_sources_are_plain_ascii_with_lf() -> None:
    for path in sorted(SOURCES.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            assert raw.isascii() and b"\r" not in raw and raw.endswith(b"\n"), path


def test_the_capture_tool_reads_the_ttmetal_scenario_set() -> None:
    loaded = tool().load_scenarios(SCENARIOS, SOURCES)
    assert sorted(loaded) == sorted(CASES)
    assert {scenario.toolchain for scenario in loaded.values()} == {NAME}


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
    status = module.main(["--fixtures", str(FIXTURES), "--out", str(tmp_path / "out")])
    assert status == 0
    assert sorted(chosen) == sorted(CASES), "--fixtures DIR captures DIR/scenarios.json over DIR/sources"


def test_every_captured_file_belongs_to_a_case() -> None:
    stems = sorted(path.stem for path in FIXTURES.glob("*.stderr"))
    assert stems == sorted(CASES), "one <case>.stderr per case, and no other; task P4.10 captures them"
    manifests = {SCENARIOS.name, CAPTURES.name}
    lists = sorted(path.stem for path in FIXTURES.glob("*.json") if path.name not in manifests)
    assert lists == sorted(CASES), "one <case>.json per case, and no other"


@pytest.mark.parametrize("case", CASES)
def test_the_fixture_is_raw_ascii_stderr(case: str) -> None:
    raw = stderr_bytes(case)
    assert raw.isascii(), case
    assert b"\r" not in raw, f"{case} must be stored with LF line endings"
    if raw:
        assert raw.endswith(b"\n"), case
        assert not raw.startswith((b"#", b"//")), f"{case} starts with a comment; fixtures are raw stderr"


def test_the_capture_ran_from_a_clean_commit_on_the_build_host_with_the_pinned_clang() -> None:
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
    assert record["environment"] == ["LANG", "LC_ALL", "PATH"], "no loader or tt-metal variable reaches a compile"


@pytest.mark.parametrize("case", CASES)
def test_the_fixture_is_the_captured_stderr_byte_for_byte(case: str) -> None:
    raw = stderr_bytes(case)
    assert hashlib.sha256(raw).hexdigest() == entry(case)["stderr_sha256"], case
    assert len(raw) == entry(case)["stderr_bytes"], case


def capture_tree(argv: list[str]) -> str:
    """Return the tree the captured command built against, read from its -I<tree>/tt_metal/api word."""
    words = [word for word in argv if word.startswith("-I") and word.endswith(API_INCLUDE_SUFFIX)]
    trees = [word[2 : -len(API_INCLUDE_SUFFIX)] for word in words if "/build_Release/" not in word]
    assert len(trees) == 1, argv
    return trees[0]


@pytest.mark.parametrize("case", CASES)
def test_the_capture_compiled_the_case_source_tree_with_the_adapter_command(case: str) -> None:
    record = entry(case)
    assert record["toolchain"] == NAME and record["overrides"] == {}
    argv = record["argv"]
    assert argv[0] == read_pin(PIN_NAME)["EXECUTABLE"]
    tree = capture_tree(argv)
    assert PurePosixPath(tree).name == PREFIX, "the capture built against the pinned install"
    assert argv == make(executable=argv[0], tree=tree).command(host_sources(case))


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
    assert stderr_bytes("ttm_clean") == b""
    assert entry("ttm_clean")["exit_status"] == 0
    assert expected("ttm_clean") == []


def test_the_syntax_error_is_an_error_on_main_cpp_at_the_declaration() -> None:
    case = "ttm_syntax_error"
    errors = [item for item in captured(case) if item.severity == "error"]
    assert errors, "a syntax error gives at least one error"
    assert errors[0].file == "main.cpp"
    assert errors[0].line == line_of(case, "main.cpp", "int total = 0")


def test_the_undeclared_identifier_error_points_at_the_identifier() -> None:
    case = "ttm_undeclared_identifier"
    errors = [item for item in captured(case) if item.severity == "error" and "undefined_var" in item.message]
    assert errors, "the error names the undeclared identifier"
    assert errors[0].file == "main.cpp"
    assert errors[0].line == line_of(case, "main.cpp", "undefined_var;")
    assert points_at(case, errors[0], "undefined_var"), "the column counts in main.cpp and points at the name"


def test_the_missing_header_is_an_error_at_the_include_line() -> None:
    case = "ttm_missing_header"
    errors = [item for item in captured(case) if item.severity == "error"]
    assert errors and errors[0].file == "main.cpp"
    assert errors[0].line == line_of(case, "main.cpp", "p410_no_such_header.hpp")
    assert "p410_no_such_header.hpp" in errors[0].message


def test_the_unused_parameter_is_a_warning_and_the_build_succeeds() -> None:
    # OQ-027, option (b): the host build keeps -Wall and -Wunused-parameter but not -Werror, so the warning stays a
    # warning (counted in df-v0's W at S4 or S5) and the program is built.
    case = "ttm_unused_parameter"
    assert entry(case)["exit_status"] == 0, "without -Werror the warning does not fail the build"
    found = [item for item in captured(case) if item.code is not None and "-Wunused-parameter" in item.code]
    assert len(found) == 1, "the pinned -Wunused-parameter names the unused parameter once"
    (item,) = found
    assert item.severity == "warning" and item.code == "-Wunused-parameter" and item.file == "main.cpp"
    assert item.line == line_of(case, "main.cpp", "int factor")
    assert points_at(case, item, "factor")


def test_the_note_chain_follows_the_error_with_a_note_per_candidate() -> None:
    case = "ttm_note_chain"
    found = captured(case)
    assert found and found[0].severity == "error", "the chain starts with the error"
    assert (found[0].file, found[0].line) == ("main.cpp", line_of(case, "main.cpp", "scale(values, 4);"))
    notes = [item for item in found[1:] if item.severity == "note"]
    assert {item.file for item in notes} == {"main.cpp"}
    candidate_lines = {line_of(case, "main.cpp", "void scale(float"), line_of(case, "main.cpp", "void scale(double")}
    assert candidate_lines <= {item.line for item in notes}, "a note points at each candidate's declaration"


def test_the_header_error_names_the_header_and_not_the_include_line() -> None:
    case = "ttm_header_error"
    found = captured(case)
    errors = [item for item in found if item.severity == "error"]
    assert errors and {item.file for item in errors} == {"host/scale.h"}
    first = errors[0]
    assert first.line == line_of(case, "host/scale.h", "factor;")
    assert points_at(case, first, "factor")
    include_line = line_of(case, "main.cpp", '#include "host/scale.h"')
    assert not [item for item in found if (item.file, item.line) == ("main.cpp", include_line)], (
        "the include chain is context, never a diagnostic"
    )


def test_the_linker_error_names_the_symbol_with_no_place() -> None:
    case = "ttm_linker_error"
    assert entry(case)["exit_status"] != 0
    errors = [item for item in captured(case) if item.severity == "error"]
    symbol = [item for item in errors if "helper" in item.message]
    assert symbol, "the undefined symbol is an error naming helper"
    assert all(item.file is None and item.line is None for item in symbol), "a linker error has no source place"


def test_the_readme_records_the_capture_without_placeholder() -> None:
    if not README.is_file():
        pytest.fail(f"{README} does not exist; task P4.10 records the capture's provenance there")
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
