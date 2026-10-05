"""Remote tests of the native C++ toolchain `gcc-native` on the build host (task P4.5).

Bible: Toolchain Pins (the --version check against EXPECT_VERSION in the
compile sandbox before the first build), Execution Backends (native row),
Harness Contract (lassi_io.h), Sandbox (compiles and runs), Agent Rules 6,
7, and 10.

- build_toolchain("gcc-native", $LASSI_TOOLCHAINS) runs the pinned host g++
  (toolchains/gcc.pin EXECUTABLE) with --version through the compile
  sandbox (SandboxedCompileRunner), which must print the pin's
  EXPECT_VERSION; a copy of the pin with another EXPECT_VERSION is refused
  there.
- A tiny C++ program, written here and not generated, includes
  assets/harness/c/lassi_io.h as a harness file, fills an f32 array in an
  OpenMP parallel loop, and writes it with lassi_io_write. It builds with
  the toolchain's own build() under the native flags with an empty
  compile.stderr, runs through NativeExecutor (so only in the sandbox), and
  its output file reads back with lassi.harness.lassi_io.read_array as the
  exact values (0.5 times each index is exact in f32).
- A program with a syntax error fails to build, and the live stderr parses
  into an error on the built file.

The tests are marked `remote` and skip unless the host can run them (Linux,
the sandbox tools on PATH, a reachable user systemd manager, $LASSI_SCRATCH,
$LASSI_RUNS_ROOT, and $LASSI_TOOLCHAINS set, TMPDIR inside $LASSI_SCRATCH,
and the pin's EXECUTABLE present); with LASSI_REQUIRE_SANDBOX=1 they fail
instead, so a silent skip never passes for evidence. Run them with
`uv run tools/rx.py run -- '<command>'`, the command being
`LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/toolchains/test_gcc_native_remote.py`.
Every file they write lies in a temp directory under $LASSI_RUNS_ROOT or
pytest's temp directory (under TMPDIR on the scratch disk), removed
afterwards (Agent Rule 7). No value here is a measurement.
"""

from __future__ import annotations

import os
import shutil
import struct
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from lassi.core.interfaces import Limits
from lassi.core.runner import BuiltToolchain, RunError, build_toolchain
from lassi.executors.native import NativeExecutor
from lassi.executors.sandbox import SandboxedCompileRunner
from lassi.harness.lassi_io import LassiArray, read_array
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
HEADER = REPO / "assets" / "harness" / "c" / "lassi_io.h"
GCC_PIN = REPO / "toolchains" / "gcc.pin"
NAME = "gcc-native"
PIN_NAME = "gcc"
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
LIMITS = Limits(wall_s=30.0, memory_mb=512, cpus=2)
COUNT = 8
EXPECTED_Y = struct.pack(f"<{COUNT}f", *(0.5 * index for index in range(COUNT)))

PROGRAM = r"""// A test program for the native toolchain: fill y in an OpenMP loop and write it with lassi_io.h.
#include <cstdint>
#include <cstdio>

#include "lassi_io.h"

int main() {
    float y[8];
#pragma omp parallel for
    for (int i = 0; i < 8; ++i) {
        y[i] = 0.5f * static_cast<float>(i);
    }
    const std::uint64_t dims[1] = {8u};
    if (lassi_io_write("y.lassiio", "y", LASSI_IO_F32, 1u, dims, y) != 0) {
        return 1;
    }
    std::printf("wrote y\n");
    return 0;
}
"""

BROKEN = "int main() {\n    int total = 0\n    return total;\n}\n"


def host_problem() -> str:
    """Return why this host cannot run the tests, or "" when it can."""
    if not sys.platform.startswith("linux"):
        return f"the sandbox needs Linux, not {sys.platform}; run it through `uv run tools/rx.py run`"
    missing = [tool for tool in TOOLS if shutil.which(tool) is None]
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
    if not GCC_PIN.is_file():
        return f"{GCC_PIN} does not exist"
    executable = pins_module.read_pin(PIN_NAME).get("EXECUTABLE", "")
    if not executable or not Path(executable).is_file():
        return f"the pinned g++ {executable!r} is not on this host"
    return ""


PROBLEM = host_problem()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can run the native toolchain"),
]


@pytest.fixture(autouse=True)
def host_ready() -> None:
    """Fail, rather than skip, when LASSI_REQUIRE_SANDBOX=1 asks for a real sandbox and the host cannot run one."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")


@pytest.fixture(scope="module")
def built() -> BuiltToolchain:
    """Build gcc-native as a run builds it: the pinned g++, checked by --version in the compile sandbox."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")
    return build_toolchain(NAME, Path(os.environ["LASSI_TOOLCHAINS"]))


@pytest.fixture
def workdir() -> Iterator[Path]:
    """Create a fresh attempt build directory under $LASSI_RUNS_ROOT; remove it afterwards."""
    base = Path(tempfile.mkdtemp(prefix="lassi-gcc-native.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        build = base / "attempt00" / "build"
        build.mkdir(parents=True)
        yield build
    finally:
        shutil.rmtree(base)


def test_the_version_check_ran_the_pinned_gpp_in_the_compile_sandbox(built: BuiltToolchain) -> None:
    pin = pins_module.read_pin(PIN_NAME)
    assert built.executable == pin["EXECUTABLE"]
    assert built.version_status == 0
    assert pin["EXPECT_VERSION"] in "\n".join(built.version)
    assert built.pins == {PIN_NAME: pin}
    assert isinstance(built.toolchain.runner, SandboxedCompileRunner), "every compile runs in the compile sandbox"


def test_a_program_with_the_lassi_io_header_builds_and_runs_natively(built: BuiltToolchain, workdir: Path) -> None:
    harness = {"lassi_io.h": HEADER.read_text(encoding="ascii")}
    result = built.toolchain.build({"main.cpp": PROGRAM}, workdir, harness=harness)
    stderr = (workdir / "compile.stderr").read_text(encoding="utf-8")
    assert result.artifact == workdir / "main", f"the program did not build:\n{stderr}"
    assert result.diagnostics == [] and stderr == "", f"a clean build prints nothing:\n{stderr}"
    run = NativeExecutor().run(result.artifact, [], LIMITS)
    assert (run.exit_code, run.hang) == (0, False), run
    assert run.stdout == "wrote y\n"
    assert sorted(run.output_files) == ["y.lassiio"]
    assert read_array(run.output_files["y.lassiio"]) == LassiArray("y", "f32", (COUNT,), EXPECTED_Y)


def test_a_syntax_error_parses_into_an_error_on_the_built_file(built: BuiltToolchain, workdir: Path) -> None:
    result = built.toolchain.build({"main.cpp": BROKEN}, workdir)
    assert result.artifact is None
    errors = [item for item in result.diagnostics if item.severity == "error"]
    assert any(item.file == "main.cpp" and item.code != "exit-status" for item in errors), result.diagnostics
    assert (workdir / "compile.stderr").read_text(encoding="utf-8").strip(), "the raw stderr is kept"


def test_a_gpp_whose_version_is_not_the_pin_is_refused_in_the_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pins_dir = tmp_path / "pins"
    pins_dir.mkdir()
    for path in (REPO / "toolchains").glob("*.pin"):
        shutil.copyfile(path, pins_dir / path.name)
    text = (pins_dir / "gcc.pin").read_text(encoding="ascii")
    expected = pins_module.read_pin(PIN_NAME)["EXPECT_VERSION"]
    changed = text.replace(expected, "PLACEHOLDER version text that no compiler prints")
    assert changed != text
    (pins_dir / "gcc.pin").write_text(changed, encoding="ascii", newline="\n")
    monkeypatch.setattr(pins_module, "PINS_DIR", pins_dir)
    with pytest.raises(RunError, match="EXPECT_VERSION"):
        build_toolchain(NAME, Path(os.environ["LASSI_TOOLCHAINS"]))
