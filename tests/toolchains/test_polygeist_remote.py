"""Remote tests of the Polygeist install (cgeist) on the build host (task P5.5).

Bible: Toolchain Pins (MLIR stacks; the --version check against
EXPECT_VERSION in the compile sandbox), Language Frontends (C and C++ rows),
IR Strategy (cpu-mlir and llvm-ir rows), Agent Rules 5, 6, 7, 10, and 11;
plans/p5-ir.md, P5.5; plans/spikes/p5-mlir-pins.md, Results 1. The local
contract is in test_polygeist_pins.py.

1. The installed prefix, $LASSI_TOOLCHAINS/<PREFIX_NAME>, is the pinned
   install: its lassi-install.txt names polygeist and COMMIT on its first
   line and holds LLVM_COMMIT, and its lassi-fetched-sources.txt equals
   toolchains/polygeist-fetched-sources.txt line for line, which by then
   holds the first install's record rather than PLACEHOLDER. The build
   tree's CMakeCache.txt (beside the tools' bin directory) puts
   POLYGEIST_PGO_DEFAULT_DATA_DIR under $LASSI_SCRATCH, and cgeist holds no
   trace of the default /var/tmp/polygeist path.
2. cgeist, polygeist-opt, mlir-opt, and mlir-translate print
   EXPECT_VERSION, and that LLVM's clang prints LLVM_CLANG_EXPECT, each run
   with --version in the compile sandbox. clang's builtin headers lie in <bin>/../lib/clang/18,
   where cgeist finds them (tools/cgeist/Lib/clang-mlir.cc:6027-6031).
3. A small C program and a small C++ program written here, each a whole
   program with its main, go through the CPU route: `cgeist -S`, which
   stops at MLIR (without -S cgeist writes a /tmp file,
   tools/cgeist/driver.cc:1200-1202), raises it to affine or scf with no
   unstructured cf branch; that LLVM's mlir-opt lowers it to the llvm
   dialect (LOWERING); `mlir-translate --mlir-to-llvmir` gives LLVM IR; that
   LLVM's clang builds it; and it runs natively, as does the same source
   built by that clang, and both print what the Python oracle computes.
   Each step's exit status and first error line are recorded as JSON in
   $LASSI_REPORT_DIR/polygeist-routes.json (an evidence run sets it, inside
   $LASSI_RUNS_ROOT), else in pytest's temp directory.
4. EXPLORATORY, for P5.7, and not a measurement: each Tier A item's C++
   reference, read through the bench registry for purpose eval (never
   training, prompt tuning, or harvest; Agent Rule 5), goes through
   `cgeist -S`; its exit status, first error line, and the structured ops it
   reached are recorded in polygeist-tier-a-exploratory.json beside the
   routes' record. Nothing is asserted about them beyond one record per
   item, and no source or IR text is recorded.

Every tool runs in the compile sandbox (lassi.executors.sandbox
SandboxedCompileRunner) with PATH, LANG, and LC_ALL only, its build dir a
fresh directory under $LASSI_RUNS_ROOT that is removed afterwards. The
programs written here run only through NativeExecutor, in the sandbox
(Agent Rule 6); nothing opens a device. The programs are SYNTHETIC, hold no
upstream text, and their IR is never committed (Agent Rule 11).

The tests are marked `remote` and skip unless the host can run them (Linux,
the sandbox tools on PATH, a reachable user systemd manager, $LASSI_SCRATCH,
$LASSI_RUNS_ROOT, and $LASSI_TOOLCHAINS set, TMPDIR inside $LASSI_SCRATCH,
the pin present, and the install with its five tools in place); with
LASSI_REQUIRE_SANDBOX=1 they fail instead, so a silent skip never passes for
evidence. Run them after the install job with `uv run tools/rx.py run --
'<command>'` (Agent Rule 17: with the owner's approval), the command holding
`LASSI_REQUIRE_SANDBOX=1 LASSI_REPORT_DIR=<dir under $LASSI_RUNS_ROOT> uv run
pytest -q -p no:cacheprovider -m remote tests/toolchains/test_polygeist_remote.py`.
No value here is a measurement.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from lassi.bench.registry import Direction, Suite, load_suite
from lassi.core.interfaces import Limits
from lassi.executors.native import NativeExecutor
from lassi.executors.sandbox import SandboxedCompileRunner
from lassi.toolchains import CommandResult
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
PIN_NAME = "polygeist"
TRACKED_FETCHED = REPO / "toolchains" / "polygeist-fetched-sources.txt"
TREE_FETCHED_NAME = "lassi-fetched-sources.txt"
RECORD_NAME = "lassi-install.txt"
ROUTES_NAME = "polygeist-routes.json"
EXPLORATORY_NAME = "polygeist-tier-a-exploratory.json"
TIER_A = REPO / "assets" / "bench" / "tt-pairs-v0.yaml"
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
# The roots every compile hides (lassi.core.runner), and the PATH a compile gets.
HIDDEN_VARIABLES = ("HOME", "LASSI_SCRATCH", "LASSI_RUNS_ROOT")
SANDBOX_PATH = "/usr/bin:/bin"
TIMEOUT_S = 600.0
LIMITS = Limits(wall_s=30.0, memory_mb=512, cpus=1)
# Each built tool by pin key, with the pin key of the version text it prints.
VERSIONS = (
    ("CGEIST", "EXPECT_VERSION"),
    ("POLYGEIST_OPT", "EXPECT_VERSION"),
    ("MLIR_OPT", "EXPECT_VERSION"),
    ("MLIR_TRANSLATE", "EXPECT_VERSION"),
    ("LLVM_CLANG", "LLVM_CLANG_EXPECT"),
)
# clang's major version, which names its resource directory (cmake/Modules/GetClangResourceDir.cmake:18 at
# LLVM 26eb4285; LLVM_VERSION_MAJOR 18, llvm/CMakeLists.txt:11-12).
CLANG_MAJOR = "18"
# The lowering to the llvm dialect, every pass upstream at LLVM 26eb4285 (mlir/include/mlir/Conversion/Passes.td:
# lower-affine :38, convert-scf-to-cf :870, convert-arith-to-llvm :119, convert-index-to-llvm :613,
# finalize-memref-to-llvm :720, convert-func-to-llvm :359, convert-cf-to-llvm :273, reconcile-unrealized-casts :844).
# Every function and call stays inside one module, so the default memref calling convention serves. The translation
# after it is mlir-translate's --mlir-to-llvmir (mlir/lib/Target/LLVMIR/ConvertToLLVMIR.cpp:27).
LOWERING = (
    "--lower-affine",
    "--convert-scf-to-cf",
    "--convert-arith-to-llvm",
    "--convert-index-to-llvm",
    "--finalize-memref-to-llvm",
    "--convert-func-to-llvm",
    "--convert-cf-to-llvm",
    "--reconcile-unrealized-casts",
)
# The structured levels a raise reaches (bible Language Frontends: C to affine, scf, memref; C++ to scf and memref),
# and the unstructured branches it must not leave.
STRUCTURED = ("affine.for", "affine.parallel", "scf.for", "scf.parallel", "scf.while", "scf.if")
LOOPS = ("affine.for", "affine.parallel", "scf.for", "scf.parallel", "scf.while")
UNSTRUCTURED = ("cf.br", "cf.cond_br")

# SYNTHETIC: a whole C program written for these tests, a prefix sum weighted by position over a static array.
C_PROGRAM = """/* SYNTHETIC: written for the P5.5 remote tests; not from any upstream file. */
#include <stdio.h>

#define LASSI_N 16

static int lassi_x[LASSI_N];
static int lassi_y[LASSI_N];

void lassi_c_kernel(int n, const int *x, int *y) {
  for (int i = 0; i < n; i++) {
    int acc = 0;
    for (int j = 0; j <= i; j++) {
      acc += x[j] * (j + 1);
    }
    y[i] = acc;
  }
}

int main(void) {
  for (int i = 0; i < LASSI_N; i++) {
    lassi_x[i] = (i * 7) % 11 - 5;
  }
  lassi_c_kernel(LASSI_N, lassi_x, lassi_y);
  for (int i = 0; i < LASSI_N; i++) {
    printf("%d\\n", lassi_y[i]);
  }
  return 0;
}
"""
# SYNTHETIC: a whole C++ program in cgeist's C-like subset (Language Frontends, C++ row): a namespace, constexpr
# constants, a function template, and an inline helper, over static arrays; no class, lambda, or exception.
CXX_PROGRAM = """// SYNTHETIC: written for the P5.5 remote tests; not from any upstream file.
#include <cstdio>

namespace lassi_probe {

constexpr int kCount = 12;
constexpr int kWidth = 3;

template <typename T>
T weigh(T value, T weight) {
  return value * weight + static_cast<T>(1);
}

inline int clamp_low(int value, int low) { return value < low ? low : value; }

void window_sum(int n, const int *x, int *y) {
  for (int i = 0; i < n; ++i) {
    int acc = 0;
    for (int j = clamp_low(i - kWidth + 1, 0); j <= i; ++j) {
      acc += weigh<int>(x[j], 3);
    }
    y[i] = acc;
  }
}

}  // namespace lassi_probe

static int lassi_in[lassi_probe::kCount];
static int lassi_out[lassi_probe::kCount];

int main() {
  for (int i = 0; i < lassi_probe::kCount; ++i) {
    lassi_in[i] = (i * 5) % 9 - 4;
  }
  lassi_probe::window_sum(lassi_probe::kCount, lassi_in, lassi_out);
  for (int i = 0; i < lassi_probe::kCount; ++i) {
    std::printf("%d\\n", lassi_out[i]);
  }
  return 0;
}
"""


def c_expected() -> str:
    """Return what C_PROGRAM prints, computed here from its definition (the oracle)."""
    x = [(i * 7) % 11 - 5 for i in range(16)]
    return "".join(f"{sum(x[j] * (j + 1) for j in range(i + 1))}\n" for i in range(16))


def cxx_expected() -> str:
    """Return what CXX_PROGRAM prints, computed here from its definition (the oracle)."""
    x = [(i * 5) % 9 - 4 for i in range(12)]
    return "".join(f"{sum(3 * x[j] + 1 for j in range(max(i - 2, 0), i + 1))}\n" for i in range(12))


# Each program: its file name, its text, the clang driver words that build it, and its oracle.
PROGRAMS: dict[str, tuple[str, str, tuple[str, ...], Callable[[], str]]] = {
    "c": ("lassi_probe.c", C_PROGRAM, (), c_expected),
    "cxx": ("lassi_probe.cpp", CXX_PROGRAM, ("--driver-mode=g++",), cxx_expected),
}
# Each program's route record, by language (run_route).
Routes = dict[str, dict[str, object]]


def read_pin() -> dict[str, str]:
    """Return toolchains/polygeist.pin."""
    return pins_module.read_pin(PIN_NAME)


def install_prefix() -> Path:
    """Return the pinned install prefix under the resolved toolchains root."""
    return Path(os.environ["LASSI_TOOLCHAINS"]).resolve() / read_pin()["PREFIX_NAME"]


def tool(key: str) -> Path:
    """Return the installed tool the pin names under `key`."""
    return install_prefix() / read_pin()[key]


def host_problem() -> str:
    """Return why this host cannot run the tests, or "" when it can."""
    if not sys.platform.startswith("linux"):
        return f"the sandbox needs Linux, not {sys.platform}; run it through `uv run tools/rx.py run`"
    missing = [name for name in TOOLS if shutil.which(name) is None]
    if missing:
        return f"not on PATH: {', '.join(missing)}"
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if not runtime or not (Path(runtime) / "bus").exists():
        return "no user systemd manager: $XDG_RUNTIME_DIR/bus does not exist"
    for name in ("LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS", "TMPDIR"):
        if not os.environ.get(name):
            return f"{name} is not set"
    scratch, tmpdir = Path(os.environ["LASSI_SCRATCH"]).resolve(), Path(os.environ["TMPDIR"]).resolve()
    if scratch not in tmpdir.parents:
        return "TMPDIR does not lie inside $LASSI_SCRATCH"
    if not pins_module.pin_path(PIN_NAME).is_file():
        return f"{pins_module.pin_path(PIN_NAME)} does not exist"
    if not (install_prefix() / RECORD_NAME).is_file():
        return f"{install_prefix()} is not an install of the pinned Polygeist"
    for key, _ in VERSIONS:
        if not tool(key).is_file():
            return f"{tool(key)} ({key} in toolchains/polygeist.pin) is not installed"
    return ""


PROBLEM = host_problem()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can run the Polygeist tools"),
]


@pytest.fixture(autouse=True)
def host_ready() -> None:
    """Fail, rather than skip, when LASSI_REQUIRE_SANDBOX=1 asks for a real sandbox and the host cannot run one."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")


@pytest.fixture(scope="module")
def compile_runner() -> SandboxedCompileRunner:
    """Return the compile sandbox runner every tool runs through, as a stage's compiles do."""
    hidden = [Path(os.environ[name]) for name in HIDDEN_VARIABLES if os.environ.get(name)]
    return SandboxedCompileRunner(
        environment={"PATH": SANDBOX_PATH, "LANG": "C", "LC_ALL": "C"},
        toolchains=Path(os.environ["LASSI_TOOLCHAINS"]),
        hidden_roots=hidden,
    )


def fresh_base(prefix: str) -> Path:
    """Create and return a fresh directory under $LASSI_RUNS_ROOT."""
    return Path(tempfile.mkdtemp(prefix=prefix, dir=os.environ["LASSI_RUNS_ROOT"]))


@pytest.fixture
def workdir() -> Iterator[Path]:
    """Create a fresh build directory under $LASSI_RUNS_ROOT; remove it afterwards."""
    base = fresh_base("lassi-polygeist.")
    try:
        build = base / "build"
        build.mkdir()
        yield build
    finally:
        shutil.rmtree(base)


def run_tool(runner: SandboxedCompileRunner, workdir: Path, key: str, args: list[str]) -> CommandResult:
    """Run the installed tool `key` with `args` in the compile sandbox, in `workdir`."""
    return runner([str(tool(key)), *args], workdir, TIMEOUT_S)


def first_error(result: CommandResult) -> str:
    """Return the first line of a tool's stderr that holds "error", in printable ASCII, at most 300 characters."""
    for line in result.stderr.splitlines():
        if "error" in line:
            return re.sub(r"[^ -~]", "?", line.strip())[:300]
    return ""


def step(name: str, result: CommandResult) -> dict[str, object]:
    """Return one route step's record: its exit status and first error line."""
    return {f"{name}_status": result.returncode, f"{name}_error": first_error(result)}


def levels(module: str) -> dict[str, bool]:
    """Return which structured ops, unstructured branches, and polygeist ops a module's text holds."""
    found = {op: op in module for op in (*STRUCTURED, *UNSTRUCTURED)}
    found["polygeist"] = "polygeist." in module
    return found


def entry_lines(text: str) -> list[str]:
    """Return the lines of a fetched-sources list that are not blank and not # comments, trailing spaces removed."""
    return [line.rstrip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def report_path(name: str, fallback: Path) -> Path:
    """Return where a record goes: $LASSI_REPORT_DIR (inside $LASSI_RUNS_ROOT), else `fallback`."""
    named = os.environ.get("LASSI_REPORT_DIR", "")
    if not named:
        assert not REQUIRE, "an evidence run (LASSI_REQUIRE_SANDBOX=1) sets LASSI_REPORT_DIR for the records"
        return fallback / name
    directory = Path(named).resolve()
    assert Path(os.environ["LASSI_RUNS_ROOT"]).resolve() in directory.parents, "the record stays in the runs root"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def write_report(name: str, records: object, fallback: Path) -> None:
    """Write `records` as ASCII JSON, with the pin's two commits, to report_path(name, fallback)."""
    pin = read_pin()
    report = {"commit": pin["COMMIT"], "llvm_commit": pin["LLVM_COMMIT"], "records": records}
    report_path(name, fallback).write_bytes((json.dumps(report, indent=2) + "\n").encode("ascii"))


# The install.


def test_the_install_is_the_pinned_one_with_the_recorded_fetched_sources() -> None:
    """The install record names the pinned commits, and the configure's fetches equal the tracked list."""
    prefix, pin = install_prefix(), read_pin()
    record = (prefix / RECORD_NAME).read_text(encoding="ascii")
    assert record.splitlines()[0].split()[:2] == [PIN_NAME, pin["COMMIT"]]
    assert pin["LLVM_COMMIT"] in record
    tracked = entry_lines(TRACKED_FETCHED.read_text(encoding="ascii"))
    assert tracked != ["PLACEHOLDER"], "copy the first install's list into the tracked one before these tests"
    assert entry_lines((prefix / TREE_FETCHED_NAME).read_text(encoding="ascii")) == tracked


def test_the_pgo_directory_lies_in_the_scratch_root() -> None:
    """The build tree caches POLYGEIST_PGO_DEFAULT_DATA_DIR under $LASSI_SCRATCH; cgeist holds no /var/tmp/polygeist."""
    cache = tool("CGEIST").parent.parent / "CMakeCache.txt"
    lines = cache.read_text(encoding="utf-8", errors="replace").splitlines()
    values = [line.split("=", 1)[1] for line in lines if line.startswith("POLYGEIST_PGO_DEFAULT_DATA_DIR:")]
    assert len(values) == 1, values
    scratch = Path(os.environ["LASSI_SCRATCH"]).resolve()
    assert scratch in Path(values[0]).resolve().parents, values[0]
    assert b"/var/tmp/polygeist" not in tool("CGEIST").read_bytes()


@pytest.mark.parametrize(("key", "expect"), VERSIONS, ids=[key for key, _ in VERSIONS])
def test_each_tool_prints_its_pinned_version(
    compile_runner: SandboxedCompileRunner, workdir: Path, key: str, expect: str
) -> None:
    """Each tool's --version, in the compile sandbox, prints the pin's text for it (Agent Rule 10)."""
    result = run_tool(compile_runner, workdir, key, ["--version"])
    assert result.returncode == 0, result.stderr
    assert read_pin()[expect] in result.stdout + result.stderr


def test_clang_builtin_headers_lie_where_cgeist_finds_them() -> None:
    """cgeist and clang share one bin directory, and <bin>/../lib/clang/18/include holds the builtin headers."""
    assert tool("CGEIST").parent == tool("LLVM_CLANG").parent
    include = tool("CGEIST").parent.parent / "lib" / "clang" / CLANG_MAJOR / "include"
    assert (include / "stddef.h").is_file(), include


# The CPU route for a C program and a C++ program written here.


def build_and_run(
    runner: SandboxedCompileRunner, here: Path, name: str, driver: tuple[str, ...], source: str
) -> dict[str, object]:
    """Build `source` in `here` with that LLVM's clang into <name>/program, run it natively; return the record."""
    (here / name).mkdir()
    built = run_tool(runner, here, "LLVM_CLANG", [*driver, source, "-o", f"{name}/program"])
    record = step(f"{name}_build", built)
    if built.returncode != 0:
        return record
    run = NativeExecutor().run(here / name / "program", [], LIMITS)
    return {**record, f"{name}_exit": run.exit_code, f"{name}_hang": run.hang, f"{name}_stdout": run.stdout}


def run_route(runner: SandboxedCompileRunner, base: Path, language: str) -> dict[str, object]:
    """Take one program through cgeist -S, mlir-opt, mlir-translate, and clang, beside its source build."""
    name, text, driver, _ = PROGRAMS[language]
    here = base / language
    here.mkdir()
    (here / name).write_bytes(text.encode("ascii"))
    record: dict[str, object] = {"program": language, **build_and_run(runner, here, "source", driver, name)}
    raised = run_tool(runner, here, "CGEIST", [name, "-S", "-o", "raised.mlir"])
    record.update(step("cgeist", raised))
    if raised.returncode != 0:
        return record
    record["levels"] = levels((here / "raised.mlir").read_text(encoding="utf-8"))
    lowered = run_tool(runner, here, "MLIR_OPT", ["raised.mlir", *LOWERING, "-o", "lowered.mlir"])
    record.update(step("lower", lowered))
    if lowered.returncode != 0:
        return record
    translated = run_tool(runner, here, "MLIR_TRANSLATE", ["--mlir-to-llvmir", "lowered.mlir", "-o", "raised.ll"])
    record.update(step("translate", translated))
    if translated.returncode != 0:
        return record
    return {**record, **build_and_run(runner, here, "raised", driver, "raised.ll")}


@pytest.fixture(scope="module")
def routes(
    compile_runner: SandboxedCompileRunner, tmp_path_factory: pytest.TempPathFactory
) -> Routes:
    """Run both programs' routes once, write their record (ROUTES_NAME), and return it by language."""
    base = fresh_base("lassi-polygeist-routes.")
    try:
        records = {language: run_route(compile_runner, base, language) for language in PROGRAMS}
    finally:
        shutil.rmtree(base)
    write_report(ROUTES_NAME, records, tmp_path_factory.mktemp("polygeist-routes"))
    return records


@pytest.mark.parametrize("language", list(PROGRAMS))
def test_cgeist_raises_each_program_to_affine_or_scf(routes: Routes, language: str) -> None:
    """cgeist -S stops at MLIR with a structured loop and no unstructured cf branch (Language Frontends)."""
    record = routes[language]
    assert record["cgeist_status"] == 0, json.dumps(record, indent=2)
    found = record["levels"]
    assert any(found[op] for op in LOOPS), json.dumps(record, indent=2)
    assert not any(found[op] for op in UNSTRUCTURED), json.dumps(record, indent=2)


@pytest.mark.parametrize("language", list(PROGRAMS))
def test_each_raised_program_lowers_through_that_llvm_and_builds(routes: Routes, language: str) -> None:
    """That LLVM's mlir-opt and mlir-translate take the raised module to LLVM IR, and its clang builds it."""
    record = routes[language]
    for name in ("cgeist", "lower", "translate", "raised_build"):
        assert record.get(f"{name}_status") == 0, json.dumps(record, indent=2)


@pytest.mark.parametrize("language", list(PROGRAMS))
def test_each_raised_program_prints_its_source_builds_output(routes: Routes, language: str) -> None:
    """Both builds exit 0 natively and print the same lines, which are the oracle's."""
    record, expected = routes[language], PROGRAMS[language][3]()
    assert record.get("source_build_status") == 0, json.dumps(record, indent=2)
    assert (record.get("source_exit"), record.get("source_hang")) == (0, False), json.dumps(record, indent=2)
    assert record["source_stdout"] == expected, "the source build itself prints the oracle's output"
    assert (record.get("raised_exit"), record.get("raised_hang")) == (0, False), json.dumps(record, indent=2)
    assert record["raised_stdout"] == record["source_stdout"]


# EXPLORATORY: Tier A's C++ references through cgeist, for P5.7.


def explore_item(runner: SandboxedCompileRunner, workdir: Path, suite: Suite, name: str) -> dict[str, object]:
    """Raise one Tier A item's C++ reference with cgeist -S in its own directory; return its record (no text)."""
    here = workdir / name
    here.mkdir()
    files = suite.reference_target(name, Direction(source="tt", target="cpp"), REPO, purpose="eval")
    support = suite.support_files(name, REPO, purpose="eval", language="cpp")
    for file, text in {**support, **files}.items():
        (here / file).write_bytes(text.encode("utf-8"))
    results = []
    for file in sorted(files):
        raised = run_tool(runner, here, "CGEIST", [file, "-S", "-I", ".", "-o", f"{file}.mlir"])
        entry: dict[str, object] = {"file": file, **step("cgeist", raised)}
        if raised.returncode == 0:
            entry["levels"] = levels((here / f"{file}.mlir").read_text(encoding="utf-8"))
        results.append(entry)
    return {"item": name, "files": results}


def test_exploratory_tier_a_cpp_references_through_cgeist(
    compile_runner: SandboxedCompileRunner, workdir: Path, tmp_path: Path
) -> None:
    """EXPLORATORY, not a measurement: one record per Tier A item, for P5.7; nothing about cgeist is asserted."""
    suite = load_suite(TIER_A)
    records = [explore_item(compile_runner, workdir, suite, name) for name in sorted(suite.items)]
    labeled = {"label": "EXPLORATORY: cgeist -S on each Tier A C++ reference, for P5.7", "items": records}
    write_report(EXPLORATORY_NAME, labeled, tmp_path)
    assert [record["item"] for record in records] == sorted(suite.items)
