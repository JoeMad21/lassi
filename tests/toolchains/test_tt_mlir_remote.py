"""Remote tests of the tt-mlir install at the joint pin, on the build host (task P5.4).

Bible: Toolchain Pins (the joint pin; MLIR stacks; the --version check
against EXPECT_VERSION in the compile sandbox), IR Strategy (the ttkernel
row), Target Rules, Agent Rules 6, 7, 10, and 11; plans/p5-ir.md, P5.4;
plans/spikes/p5-mlir-pins.md, Results 2 (what the ttkernel-to-EmitC
conversion handles). The local contract is in test_tt_mlir_pins.py.

1. The installed prefix, $LASSI_TOOLCHAINS/<PREFIX_NAME>, is the pinned
   install: its lassi-install.txt names tt-mlir and COMMIT on its first
   line, and its lassi-fetched-sources.txt equals
   toolchains/tt-mlir-fetched-sources.txt line for line, which by then holds
   the first install's record rather than PLACEHOLDER.
2. `ttmlir-opt --version`, run in the compile sandbox, exits 0 and prints
   the pin's EXPECT_VERSION; `ttmlir-translate --help` lists both
   translations the routes below use.
3. ttmlir-opt parses and verifies a small ttkernel module written here, and
   prints it in custom form that parses again; a module written here that
   breaks a verifier rule is refused.
4. That module goes through `--convert-ttkernel-to-emitc` and
   `ttmlir-translate --ttkernel-to-cpp`, tt-mlir's own kernel route (its
   tests at the pin, test/ttmlir/Translate/TTKernel), to a kernel_main.
5. A module written here with scf.for, scf.if, arith, a memref.global
   constant table read with memref.get_global, and a func.call to a helper
   goes through each route of ROUTES. Each route's result is recorded: the
   exit status of each step, its first error line, and which parts the C++
   keeps (the kernel, the helper, the table's values, the loop, the branch,
   the NOC read). By source, `--ttkernel-to-cpp` drops module-level globals
   and helper functions (lib/Target/TTKernel/TTKernelToCpp.cpp:361-386 at
   the pin; the spike, Results 2), so that is recorded, not asserted. The
   acceptance is asserted: at least one route through the ttkernel-to-EmitC
   conversion and `ttmlir-translate --mlir-to-cpp` exits 0 with C++ that
   keeps every part. The record is written as JSON to
   $LASSI_REPORT_DIR/tt-mlir-routes.json (an evidence run sets it, inside
   $LASSI_RUNS_ROOT), else into pytest's temp directory.

Every tool runs in the compile sandbox (lassi.executors.sandbox
SandboxedCompileRunner) with PATH, LANG, and LC_ALL only, its build dir a
fresh directory under $LASSI_RUNS_ROOT that is removed afterwards. Nothing
runs a built program and no command names a device (Agent Rules 6 and 9).
The modules are SYNTHETIC, written here in custom assembly form only (Agent
Rule 11), and hold no upstream text.

The tests are marked `remote` and skip unless the host can run them (Linux,
the sandbox tools on PATH, a reachable user systemd manager, $LASSI_SCRATCH,
$LASSI_RUNS_ROOT, and $LASSI_TOOLCHAINS set, TMPDIR inside $LASSI_SCRATCH,
the pin present, and the install with both tools in place); with
LASSI_REQUIRE_SANDBOX=1 they fail instead, so a silent skip never passes for
evidence. Run them after the install job with `uv run tools/rx.py run --
'<command>'` (Agent Rule 17: with the owner's approval), the command holding
`LASSI_REQUIRE_SANDBOX=1 LASSI_REPORT_DIR=<dir under $LASSI_RUNS_ROOT> uv run
pytest -q -p no:cacheprovider -m remote tests/toolchains/test_tt_mlir_remote.py`.
No value here is a measurement.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from lassi.executors.sandbox import SandboxedCompileRunner
from lassi.toolchains import CommandResult
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
TOOLCHAINS_DIR = REPO / "toolchains"
PIN_NAME = "tt-mlir"
TRACKED_FETCHED = TOOLCHAINS_DIR / "tt-mlir-fetched-sources.txt"
TREE_FETCHED_NAME = "lassi-fetched-sources.txt"
RECORD_NAME = "lassi-install.txt"
REPORT_NAME = "tt-mlir-routes.json"
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
# The roots every compile hides (lassi.core.runner), and the PATH a compile gets.
HIDDEN_VARIABLES = ("HOME", "LASSI_SCRATCH", "LASSI_RUNS_ROOT")
SANDBOX_PATH = "/usr/bin:/bin"
TIMEOUT_S = 300.0

# SYNTHETIC: a data-movement kernel written for these tests; the op forms are those of ttkernel at the pin
# (custom assembly form only).
VERIFY_MODULE = """// SYNTHETIC: written for the P5.4 remote tests; not from any upstream file.
func.func @lassi_probe_reader() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %bank = arith.constant 0 : index
  %src = arith.constant 524288 : i32
  %dst = arith.constant 131072 : i32
  %bytes = arith.constant 64 : i32
  %addr = ttkernel.get_noc_addr(%bank, %bank, %src) : (index, index, i32) -> !ttkernel.noc_addr
  ttkernel.noc_async_read(%addr, %dst, %bytes) : (!ttkernel.noc_addr, i32, i32) -> ()
  ttkernel.noc_async_read_barrier() : () -> ()
  func.return
}
"""
# SYNTHETIC: the same kernel returning a value its signature does not declare, which func.return's verifier refuses.
BROKEN_MODULE = VERIFY_MODULE.replace("  func.return\n", "  func.return %src : i32\n")
# SYNTHETIC: the forms the TT raiser emits (P5.8): scf.for and scf.if, arith, a constant table held as a
# memref.global and read with memref.get_global, and a func.call to a helper in the same module.
TABLE_VALUES = ("1234567", "2345671", "3456712", "4567123")
ROUTE_MODULE = """// SYNTHETIC: written for the P5.4 remote tests; not from any upstream file.
module {
  memref.global "private" constant @lassi_table : memref<4xi32> = dense<[%s]>
  func.func private @lassi_scale(%%x: i32) -> i32 {
    %%three = arith.constant 3 : i32
    %%y = arith.muli %%x, %%three : i32
    func.return %%y : i32
  }
  func.func @lassi_reader() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    %%table = memref.get_global @lassi_table : memref<4xi32>
    %%c0 = arith.constant 0 : index
    %%c1 = arith.constant 1 : index
    %%c4 = arith.constant 4 : index
    %%one = arith.constant 1 : i32
    %%zero = arith.constant 0 : i32
    %%dst = arith.constant 131072 : i32
    %%bytes = arith.constant 64 : i32
    scf.for %%i = %%c0 to %%c4 step %%c1 {
      %%v = memref.load %%table[%%i] : memref<4xi32>
      %%s = func.call @lassi_scale(%%v) : (i32) -> i32
      %%low = arith.andi %%s, %%one : i32
      %%odd = arith.cmpi ne, %%low, %%zero : i32
      scf.if %%odd {
        %%addr = ttkernel.get_noc_addr(%%c0, %%c0, %%s) : (index, index, i32) -> !ttkernel.noc_addr
        ttkernel.noc_async_read(%%addr, %%dst, %%bytes) : (!ttkernel.noc_addr, i32, i32) -> ()
      }
    }
    ttkernel.noc_async_read_barrier() : () -> ()
    func.return
  }
}
""" % ", ".join(TABLE_VALUES)
# The routes tried, each (name, ttmlir-opt passes, ttmlir-translate translation). The first is tt-mlir's own kernel
# route; the second converts the kernel alone (the conversion visits thread-attributed functions only,
# lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp:949-952); the third adds LLVM 4efe170d's module-level EmitC
# conversions (mlir/include/mlir/Conversion/Passes.td:167, :437, :857, :1107), memref first so the table is an
# emitc.global before the kernel's get_global becomes an emitc.get_global, which must name one
# (mlir/lib/Dialect/EmitC/IR/EmitC.cpp:1301-1307). The run records what each does.
ROUTES = (
    ("ttkernel-to-cpp", ["--convert-ttkernel-to-emitc"], "--ttkernel-to-cpp"),
    ("mlir-to-cpp", ["--convert-ttkernel-to-emitc"], "--mlir-to-cpp"),
    (
        "mlir-to-cpp-module",
        ["--convert-memref-to-emitc", "--convert-ttkernel-to-emitc", "--convert-arith-to-emitc",
         "--convert-scf-to-emitc", "--convert-func-to-emitc"],
        "--mlir-to-cpp",
    ),
)


def install_prefix() -> Path:
    """Return the pinned install prefix under the resolved toolchains root."""
    return Path(os.environ["LASSI_TOOLCHAINS"]).resolve() / pins_module.read_pin(PIN_NAME)["PREFIX_NAME"]


def tool(key: str) -> Path:
    """Return the installed tool the pin names under `key` (TTMLIR_OPT or TTMLIR_TRANSLATE)."""
    return install_prefix() / pins_module.read_pin(PIN_NAME)[key]


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
        return f"{install_prefix()} is not an install of the pinned tt-mlir"
    for key in ("TTMLIR_OPT", "TTMLIR_TRANSLATE"):
        if not tool(key).is_file():
            return f"{tool(key)} ({key} in toolchains/tt-mlir.pin) is not installed"
    return ""


PROBLEM = host_problem()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can run the tt-mlir tools"),
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


@pytest.fixture
def workdir() -> Iterator[Path]:
    """Create a fresh build directory under $LASSI_RUNS_ROOT; remove it afterwards."""
    base = Path(tempfile.mkdtemp(prefix="lassi-tt-mlir.", dir=os.environ["LASSI_RUNS_ROOT"]))
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


def entry_lines(text: str) -> list[str]:
    """Return the lines of a fetched-sources list that are not blank and not # comments, trailing spaces removed."""
    return [line.rstrip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def test_the_install_is_the_pinned_one_with_the_recorded_fetched_sources() -> None:
    """The install record names the pinned commit, and the configure's fetches equal the tracked list."""
    prefix = install_prefix()
    first = (prefix / RECORD_NAME).read_text(encoding="ascii").splitlines()[0].split()
    assert first[:2] == [PIN_NAME, pins_module.read_pin(PIN_NAME)["COMMIT"]]
    tracked = entry_lines(TRACKED_FETCHED.read_text(encoding="ascii"))
    assert tracked != ["PLACEHOLDER"], "copy the first install's list into the tracked one before these tests"
    assert entry_lines((prefix / TREE_FETCHED_NAME).read_text(encoding="ascii")) == tracked


def test_ttmlir_opt_prints_the_pinned_version(compile_runner: SandboxedCompileRunner, workdir: Path) -> None:
    """ttmlir-opt --version in the compile sandbox prints the pin's EXPECT_VERSION (Agent Rule 10)."""
    result = run_tool(compile_runner, workdir, "TTMLIR_OPT", ["--version"])
    assert result.returncode == 0, result.stderr
    assert pins_module.read_pin(PIN_NAME)["EXPECT_VERSION"] in result.stdout + result.stderr


def test_ttmlir_translate_lists_both_cpp_translations(compile_runner: SandboxedCompileRunner, workdir: Path) -> None:
    """The kernel translation (TTKernelToCppRegistration.cpp:22-23) and upstream's (ttmlir-translate.cpp:46)."""
    result = run_tool(compile_runner, workdir, "TTMLIR_TRANSLATE", ["--help"])
    assert result.returncode == 0, result.stderr
    assert "--ttkernel-to-cpp" in result.stdout and "--mlir-to-cpp" in result.stdout


def test_ttmlir_opt_verifies_a_ttkernel_module_written_here(
    compile_runner: SandboxedCompileRunner, workdir: Path
) -> None:
    """The module parses and verifies, and its printed custom form parses and verifies again."""
    (workdir / "probe.mlir").write_bytes(VERIFY_MODULE.encode("ascii"))
    first = run_tool(compile_runner, workdir, "TTMLIR_OPT", ["probe.mlir", "-o", "printed.mlir"])
    assert first.returncode == 0, first.stderr
    printed = (workdir / "printed.mlir").read_text(encoding="utf-8")
    assert "ttkernel.noc_async_read" in printed and '"ttkernel.' not in printed, "custom form, not generic"
    again = run_tool(compile_runner, workdir, "TTMLIR_OPT", ["printed.mlir", "-o", "again.mlir"])
    assert again.returncode == 0, again.stderr


def test_ttmlir_opt_refuses_a_module_that_does_not_verify(
    compile_runner: SandboxedCompileRunner, workdir: Path
) -> None:
    """A func.return with a value its function does not declare fails verification with an error."""
    (workdir / "broken.mlir").write_bytes(BROKEN_MODULE.encode("ascii"))
    result = run_tool(compile_runner, workdir, "TTMLIR_OPT", ["broken.mlir", "-o", "out.mlir"])
    assert result.returncode != 0
    assert "error" in result.stderr


def test_a_kernel_translates_through_ttkernel_to_cpp(compile_runner: SandboxedCompileRunner, workdir: Path) -> None:
    """tt-mlir's own kernel route: the conversion, then --ttkernel-to-cpp, gives a kernel_main with the NOC read."""
    (workdir / "probe.mlir").write_bytes(VERIFY_MODULE.encode("ascii"))
    converted = run_tool(compile_runner, workdir, "TTMLIR_OPT",
                         ["--convert-ttkernel-to-emitc", "probe.mlir", "-o", "emitc.mlir"])
    assert converted.returncode == 0, converted.stderr
    translated = run_tool(compile_runner, workdir, "TTMLIR_TRANSLATE",
                          ["--ttkernel-to-cpp", "emitc.mlir", "-o", "kernel.cpp"])
    assert translated.returncode == 0, translated.stderr
    cpp = (workdir / "kernel.cpp").read_text(encoding="utf-8")
    assert "kernel_main" in cpp and "noc_async_read" in cpp


def kept_parts(cpp: str) -> dict[str, bool]:
    """Return which parts of ROUTE_MODULE a route's C++ keeps."""
    return {
        "kernel": "lassi_reader" in cpp or "kernel_main" in cpp,
        "helper": "lassi_scale" in cpp,
        "table": all(value in cpp for value in TABLE_VALUES),
        "loop": "for (" in cpp,
        "branch": "if (" in cpp,
        "noc_read": "noc_async_read" in cpp,
    }


def run_route(runner: SandboxedCompileRunner, workdir: Path, route: tuple[str, list[str], str]) -> dict[str, object]:
    """Take ROUTE_MODULE through one route in its own directory; return the route's record."""
    name, passes, translation = route
    here = workdir / name
    here.mkdir()
    (here / "module.mlir").write_bytes(ROUTE_MODULE.encode("ascii"))
    record: dict[str, object] = {"route": name, "passes": passes, "translation": translation}
    converted = run_tool(runner, here, "TTMLIR_OPT", [*passes, "module.mlir", "-o", "emitc.mlir"])
    record.update(opt_status=converted.returncode, opt_error=first_error(converted))
    if converted.returncode != 0:
        return {**record, "translate_status": None, "translate_error": "", "keeps": {}}
    translated = run_tool(runner, here, "TTMLIR_TRANSLATE", [translation, "emitc.mlir", "-o", "kernel.cpp"])
    record.update(translate_status=translated.returncode, translate_error=first_error(translated))
    cpp = (here / "kernel.cpp").read_text(encoding="utf-8") if translated.returncode == 0 else ""
    return {**record, "keeps": kept_parts(cpp) if cpp else {}}


def report_path(tmp_path: Path) -> Path:
    """Return where the routes' record goes: $LASSI_REPORT_DIR (inside $LASSI_RUNS_ROOT), else pytest's temp dir."""
    named = os.environ.get("LASSI_REPORT_DIR", "")
    if not named:
        assert not REQUIRE, "an evidence run (LASSI_REQUIRE_SANDBOX=1) sets LASSI_REPORT_DIR for the routes' record"
        return tmp_path / REPORT_NAME
    directory = Path(named).resolve()
    assert Path(os.environ["LASSI_RUNS_ROOT"]).resolve() in directory.parents, "the record stays in the runs root"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / REPORT_NAME


def test_each_route_records_what_it_keeps_or_drops(
    compile_runner: SandboxedCompileRunner, workdir: Path, tmp_path: Path
) -> None:
    """Every route is recorded; one through the conversion and --mlir-to-cpp keeps the whole module."""
    records = [run_route(compile_runner, workdir, route) for route in ROUTES]
    pin = pins_module.read_pin(PIN_NAME)
    report = {"commit": pin["COMMIT"], "expect_version": pin["EXPECT_VERSION"], "routes": records}
    report_path(tmp_path).write_bytes((json.dumps(report, indent=2) + "\n").encode("ascii"))
    complete = [
        record for record in records
        if record["translation"] == "--mlir-to-cpp" and record["translate_status"] == 0
        and record["keeps"] and all(record["keeps"].values())
    ]
    assert complete, json.dumps(records, indent=2)
