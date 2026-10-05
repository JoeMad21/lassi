"""Remote tests of lassi_io.h against the Python reader and writer on the build host (task P4.3).

Bible: Harness Contract (programs read inputs and write outputs as binary
files through lassi_io.h; the harness owns inputs and comparison), Frontend
Rules (harness bindings), Sandbox (compiles and runs), Agent Rules 6 and 7.

Two small programs written here, not generated code, include
assets/harness/c/lassi_io.h (its API is stated in test_lassi_io_header.py,
the file format in test_lassi_io_format.py):

- main.cpp, compiled as C++17 (-std=c++17 -O2 -Wall -Wextra -pedantic
  -Werror). `main describe <file>...` reads each file and prints one line
  per file: name, dtype code, item size, rank, then the dims; it exits 1 at
  the first file it cannot read. `main transform <x> <h> <s> <e>` reads an
  f32 array x, a bf16 array h, and two files of any dtype, s and e, and
  writes y.lassiio (name y: x times 2 in f32), h_copy.lassiio (name h_copy:
  the bf16 bit patterns copied), and s_copy.lassiio and e_copy.lassiio (the
  arrays copied under the names <name>_copy).
- main.c, compiled as C99 (-std=c99 -O2 -Wall -Wextra -pedantic -Werror),
  prints one line per required dtype (name, its LASSI_IO_ constant,
  lassi_io_itemsize of it) and copies argv[1] to argv[2] under the name
  c_copy.

Each program includes the header twice, so its include guard is used.
Warnings are errors because the header is compiled into every candidate
program, where its warnings would reach the diagnostics fed back to the
model. The compilers are the host's g++ and gcc on SANDBOX_PATH, run in the
compile sandbox (SandboxedCompileRunner) with the header copied into the
build dir as a harness file; the g++ pin is P4.5's.

The programs run through NativeExecutor.run, so only in the sandbox (Agent
Rule 6). The Python side writes the inputs with write_array, reads the
outputs with read_array, and checks them against values it computes itself
(doubling an f32 is exact short of overflow, subnormals included); every
file the C or C++ side writes must be byte-identical to what write_array
writes for the same array. The C++ reader must refuse a bad magic, version,
or dtype, data one byte short or long, and a truncated header with exit
status 1, no stdout, and a stderr naming the file.

The tests are marked `remote` and skip unless the host can run them (Linux,
the sandbox tools on PATH, a reachable user systemd manager, $LASSI_SCRATCH,
$LASSI_RUNS_ROOT, and $LASSI_TOOLCHAINS set, TMPDIR inside $LASSI_SCRATCH,
and g++ and gcc on SANDBOX_PATH); with LASSI_REQUIRE_SANDBOX=1 they fail
instead, so a silent skip never passes for evidence. Run them with
`uv run tools/rx.py run -- 'LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/harness/test_lassi_io_remote.py'`.
Every file they write lies in a temp directory under $LASSI_RUNS_ROOT that
they remove afterwards (Agent Rule 7). No value here is a measurement.
"""

from __future__ import annotations

import importlib
import math
import os
import shutil
import struct
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from lassi.core.interfaces import Limits, RunResult
from lassi.executors.native import NativeExecutor
from lassi.executors.sandbox import SANDBOX_PATH, SandboxedCompileRunner
from lassi.toolchains import CommandResult

REPO = Path(__file__).resolve().parents[2]
HEADER = REPO / "assets" / "harness" / "c" / "lassi_io.h"
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk")
COMPILERS = ("g++", "gcc")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
COMPILE_ENVIRONMENT = {"PATH": SANDBOX_PATH, "LANG": "C", "LC_ALL": "C"}
COMPILE_TIMEOUT_S = 300.0
CPP_FLAGS = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic", "-Werror"]
C_FLAGS = ["-std=c99", "-O2", "-Wall", "-Wextra", "-pedantic", "-Werror"]
LIMITS = Limits(wall_s=30.0, memory_mb=512, cpus=1)

# name -> (code, itemsize), as test_lassi_io_format.py states; main.c prints them in this order.
REQUIRED = {
    "f32": (1, 4),
    "f16": (2, 2),
    "bf16": (3, 2),
    "i32": (4, 4),
    "u32": (5, 4),
    "i8": (6, 1),
    "u8": (7, 1),
    "i64": (8, 8),
    "f64": (9, 8),
}
# Six elements per dtype, as raw little-endian bit patterns (the same as test_lassi_io_format.py).
PATTERNS = {
    "f32": struct.pack("<6I", 0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001),
    "f16": struct.pack("<6H", 0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x0001),
    "bf16": struct.pack("<6H", 0x7FC0, 0x7F81, 0xFF80, 0x8000, 0x0001, 0x3F80),
    "i32": struct.pack("<6i", 0, -1, 2**31 - 1, -(2**31), 1, -2),
    "u32": struct.pack("<6I", 0, 1, 2**32 - 1, 2**31, 0xDEADBEEF, 7),
    "i8": struct.pack("<6b", 0, -1, 127, -128, 1, -2),
    "u8": struct.pack("<6B", 0, 1, 255, 128, 127, 42),
    "i64": struct.pack("<6q", 0, -1, 2**63 - 1, -(2**63), 1, -2),
    "f64": struct.pack(
        "<6Q", 0, 1 << 63, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000001, 0x0000000000000001
    ),
}

# transform's inputs: an f32 3x4 array with signed zeros, infinities, two subnormals, and values that double exactly;
# a bf16 2x4 array of NaNs (one with a payload and the sign set), infinities, -0, a subnormal, 1.0, and the largest
# finite value; an i64 scalar; and an empty u8 array.
X_DATA = struct.pack(
    "<12f", 0.0, -0.0, 1.0, -1.5, 3.25, 0.1, math.inf, -math.inf, 2.0**-149, 2.0**-127, 1.0e30, -65504.0
)
Y_DATA = struct.pack("<12f", *(2.0 * value for value in struct.unpack("<12f", X_DATA)))
H_DATA = struct.pack("<8H", 0x7FC0, 0xFFC1, 0x7F80, 0xFF80, 0x8000, 0x0001, 0x3F80, 0x7F7F)
S_DATA = struct.pack("<q", -(2**62) + 5)

CPP_SOURCE = r"""// A test program for lassi_io.h: describe files, or double an f32 array and copy three more.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "lassi_io.h"
#include "lassi_io.h"

namespace {

std::uint64_t count_of(const lassi_io_array &array) {
    std::uint64_t count = 1;
    for (std::uint32_t i = 0; i < array.rank; ++i) {
        count *= array.dims[i];
    }
    return count;
}

int describe(int count, char **paths) {
    for (int i = 0; i < count; ++i) {
        lassi_io_array array{};
        if (lassi_io_read(paths[i], &array) != 0) {
            return 1;
        }
        std::printf("%s %u %u %u", array.name, static_cast<unsigned>(array.dtype),
                    static_cast<unsigned>(lassi_io_itemsize(array.dtype)), static_cast<unsigned>(array.rank));
        for (std::uint32_t d = 0; d < array.rank; ++d) {
            std::printf(" %llu", static_cast<unsigned long long>(array.dims[d]));
        }
        std::printf("\n");
        lassi_io_free(&array);
    }
    return 0;
}

int copy_as(const lassi_io_array &array, const char *path) {
    const std::string name = std::string(array.name) + "_copy";
    return lassi_io_write(path, name.c_str(), array.dtype, array.rank, array.dims, array.data);
}

int transform(char **paths) {
    lassi_io_array x{};
    lassi_io_array h{};
    lassi_io_array s{};
    lassi_io_array e{};
    if (lassi_io_read(paths[0], &x) != 0 || lassi_io_read(paths[1], &h) != 0 ||
        lassi_io_read(paths[2], &s) != 0 || lassi_io_read(paths[3], &e) != 0) {
        return 1;
    }
    const std::uint32_t f32 = static_cast<std::uint32_t>(LASSI_IO_F32);
    const std::uint32_t bf16 = static_cast<std::uint32_t>(LASSI_IO_BF16);
    if (x.dtype != f32 || h.dtype != bf16) {
        std::fprintf(stderr, "unexpected dtypes %u and %u\n", static_cast<unsigned>(x.dtype),
                     static_cast<unsigned>(h.dtype));
        return 3;
    }
    const float *xs = static_cast<const float *>(x.data);
    std::vector<float> y(static_cast<std::size_t>(count_of(x)));
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] = xs[i] * 2.0f;
    }
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(count_of(h)));
    std::memcpy(bits.data(), h.data, bits.size() * sizeof(std::uint16_t));
    int status = lassi_io_write("y.lassiio", "y", f32, x.rank, x.dims, y.data());
    status |= lassi_io_write("h_copy.lassiio", "h_copy", bf16, h.rank, h.dims, bits.data());
    status |= copy_as(s, "s_copy.lassiio");
    status |= copy_as(e, "e_copy.lassiio");
    lassi_io_free(&x);
    lassi_io_free(&h);
    lassi_io_free(&s);
    lassi_io_free(&e);
    return status == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc >= 2 && std::strcmp(argv[1], "describe") == 0) {
        return describe(argc - 2, argv + 2);
    }
    if (argc == 6 && std::strcmp(argv[1], "transform") == 0) {
        return transform(argv + 2);
    }
    std::fprintf(stderr, "usage: main describe <file>... | main transform <x> <h> <s> <e>\n");
    return 2;
}
"""

C_SOURCE = r"""/* A test program for lassi_io.h in C99: dtype codes and item sizes, then one copy. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "lassi_io.h"
#include "lassi_io.h"

static void show(const char *name, uint32_t code) {
    printf("%s %u %u\n", name, (unsigned)code, (unsigned)lassi_io_itemsize(code));
}

int main(int argc, char **argv) {
    lassi_io_array array;
    int status;
    show("f32", LASSI_IO_F32);
    show("f16", LASSI_IO_F16);
    show("bf16", LASSI_IO_BF16);
    show("i32", LASSI_IO_I32);
    show("u32", LASSI_IO_U32);
    show("i8", LASSI_IO_I8);
    show("u8", LASSI_IO_U8);
    show("i64", LASSI_IO_I64);
    show("f64", LASSI_IO_F64);
    if (argc != 3) {
        fprintf(stderr, "usage: main <in> <out>\n");
        return 2;
    }
    memset(&array, 0, sizeof array);
    if (lassi_io_read(argv[1], &array) != 0) {
        return 1;
    }
    status = lassi_io_write(argv[2], "c_copy", array.dtype, array.rank, array.dims, array.data);
    lassi_io_free(&array);
    return status == 0 ? 0 : 1;
}
"""


def host_problem() -> str:
    """Return why this host cannot compile and run the programs in the sandbox, or "" when it can."""
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
    if not Path(os.environ["LASSI_TOOLCHAINS"]).is_dir():
        return "$LASSI_TOOLCHAINS is not a directory"
    missing = [compiler for compiler in COMPILERS if shutil.which(compiler, path=SANDBOX_PATH) is None]
    if missing:
        return f"not on the sandbox's PATH {SANDBOX_PATH}: {', '.join(missing)}"
    return ""


PROBLEM = host_problem()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can run the sandbox"),
]


def lassi_io() -> ModuleType:
    """Import and return lassi.harness.lassi_io."""
    return importlib.import_module("lassi.harness.lassi_io")


def hidden_roots() -> list[Path]:
    """Return $HOME, $LASSI_SCRATCH, and $LASSI_RUNS_ROOT (those set), which a compile hides."""
    roots: list[Path] = []
    for name in ("HOME", "LASSI_SCRATCH", "LASSI_RUNS_ROOT"):
        value = os.environ.get(name, "")
        if value and Path(value) not in roots:
            roots.append(Path(value))
    return roots


def compile_program(build: Path, source_name: str, source: str, compiler: str, flags: list[str]) -> CommandResult:
    """Put the header and `source` in `build` and compile them to `build`/main in the compile sandbox."""
    build.mkdir(parents=True)
    if HEADER.is_file():
        shutil.copyfile(HEADER, build / "lassi_io.h")
    (build / source_name).write_text(source, encoding="ascii", newline="\n")
    executable = shutil.which(compiler, path=SANDBOX_PATH)
    assert executable is not None, f"no {compiler} on {SANDBOX_PATH}"
    runner = SandboxedCompileRunner(
        environment=COMPILE_ENVIRONMENT,
        toolchains=Path(os.environ["LASSI_TOOLCHAINS"]),
        hidden_roots=hidden_roots(),
    )
    return runner([executable, *flags, source_name, "-o", "main"], build, COMPILE_TIMEOUT_S)


@pytest.fixture(autouse=True)
def host_ready() -> None:
    """Fail, rather than skip, when LASSI_REQUIRE_SANDBOX=1 asks for a real sandbox and the host cannot run one."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")


@pytest.fixture(scope="module")
def programs() -> Iterator[dict[str, tuple[Path, CommandResult]]]:
    """Compile main.cpp and main.c once; yield language -> (artifact, compile result); remove them afterwards."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")
    base = Path(tempfile.mkdtemp(prefix="lassi-io-build.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        built = {}
        for language, source_name, source, compiler, flags in [
            ("cpp", "main.cpp", CPP_SOURCE, "g++", CPP_FLAGS),
            ("c", "main.c", C_SOURCE, "gcc", C_FLAGS),
        ]:
            build = base / language / "build"
            built[language] = (build / "main", compile_program(build, source_name, source, compiler, flags))
        yield built
    finally:
        shutil.rmtree(base)


@pytest.fixture
def workdir() -> Iterator[Path]:
    """Create a fresh run directory under $LASSI_RUNS_ROOT; remove it afterwards."""
    base = Path(tempfile.mkdtemp(prefix="lassi-io-run.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        build = base / "attempt00" / "build"
        build.mkdir(parents=True)
        yield build
    finally:
        shutil.rmtree(base)


def installed(programs: dict[str, tuple[Path, CommandResult]], language: str, workdir: Path) -> Path:
    """Copy the compiled program into `workdir` and return its path; fail with the compiler's stderr if none."""
    artifact, result = programs[language]
    assert result.returncode == 0 and artifact.is_file(), (
        f"the {language} program did not compile (status {result.returncode}):\n{result.stderr}"
    )
    target = workdir / "main"
    shutil.copy2(artifact, target)
    return target


def run(program: Path, args: list[str]) -> RunResult:
    """Run `program` with `args` in the sandbox through the native executor."""
    return NativeExecutor().run(program, args, LIMITS)


def check_outputs(result: RunResult, expected: dict[str, tuple[str, str, tuple[int, ...], bytes]], twins: Path) -> None:
    """Check the run wrote exactly `expected` (file -> array), each file byte-identical to write_array's."""
    module = lassi_io()
    assert sorted(result.output_files) == sorted(expected), result
    twins.mkdir(parents=True, exist_ok=True)
    for file, (name, dtype, shape, data) in expected.items():
        path = Path(result.output_files[file])
        assert module.read_array(path) == module.LassiArray(name, dtype, shape, data), file
        module.write_array(twins / file, name, dtype, shape, data)
        assert path.read_bytes() == (twins / file).read_bytes(), f"{file}: the bytes differ from write_array's"


# ---------------------------------------------------------------------------
# Compiling


@pytest.mark.parametrize("language", ["cpp", "c"])
def test_the_header_compiles_with_warnings_as_errors(language: str, programs: dict) -> None:
    artifact, result = programs[language]
    assert result.returncode == 0, f"status {result.returncode}:\n{result.stderr}"
    assert artifact.is_file()


# ---------------------------------------------------------------------------
# The C++ side


def test_the_cpp_reader_describes_every_dtype_python_writes(programs: dict, workdir: Path) -> None:
    program = installed(programs, "cpp", workdir)
    module = lassi_io()
    files, expected = [], []
    for dtype, (code, itemsize) in REQUIRED.items():
        module.write_array(workdir / f"{dtype}.lassiio", f"arr_{dtype}", dtype, (2, 3), PATTERNS[dtype])
        files.append(f"{dtype}.lassiio")
        expected.append(f"arr_{dtype} {code} {itemsize} 2 2 3")
    module.write_array(workdir / "scalar.lassiio", "s", "i64", (), S_DATA)
    module.write_array(workdir / "empty.lassiio", "e", "u8", (0, 5), b"")
    files += ["scalar.lassiio", "empty.lassiio"]
    expected += ["s 8 8 0", "e 7 1 2 0 5"]
    result = run(program, ["describe", *files])
    assert result.exit_code == 0, result
    assert result.stdout.splitlines() == expected
    assert result.output_files == {}


def test_the_cpp_program_round_trips_f32_bf16_scalar_and_empty_files(programs: dict, workdir: Path) -> None:
    program = installed(programs, "cpp", workdir)
    module = lassi_io()
    module.write_array(workdir / "x.lassiio", "x", "f32", (3, 4), X_DATA)
    module.write_array(workdir / "h.lassiio", "h", "bf16", (2, 4), H_DATA)
    module.write_array(workdir / "s.lassiio", "s", "i64", (), S_DATA)
    module.write_array(workdir / "e.lassiio", "e", "u8", (0, 5), b"")
    result = run(program, ["transform", "x.lassiio", "h.lassiio", "s.lassiio", "e.lassiio"])
    assert result.exit_code == 0, result
    expected = {
        "y.lassiio": ("y", "f32", (3, 4), Y_DATA),
        "h_copy.lassiio": ("h_copy", "bf16", (2, 4), H_DATA),
        "s_copy.lassiio": ("s_copy", "i64", (), S_DATA),
        "e_copy.lassiio": ("e_copy", "u8", (0, 5), b""),
    }
    check_outputs(result, expected, workdir.parents[1] / "python")


def test_the_cpp_reader_refuses_bad_files_and_names_them(programs: dict, workdir: Path) -> None:
    program = installed(programs, "cpp", workdir)
    module = lassi_io()
    module.write_array(workdir / "valid.lassiio", "x", "f32", (2,), struct.pack("<2f", 1.0, -2.0))
    valid = (workdir / "valid.lassiio").read_bytes()
    control = run(program, ["describe", "valid.lassiio"])
    assert (control.exit_code, control.stdout) == (0, "x 1 4 1 2\n"), control
    bad = {
        "bad_magic.lassiio": b"X" + valid[1:],
        "bad_version.lassiio": valid[:8] + struct.pack("<I", 2) + valid[12:],
        "bad_dtype.lassiio": valid[:12] + struct.pack("<I", 0) + valid[16:],
        "short_data.lassiio": valid[:-1],
        "long_data.lassiio": valid + b"\x00",
        "truncated_header.lassiio": valid[:12],
    }
    for file, data in bad.items():
        (workdir / file).write_bytes(data)
    wrong = {}
    for file in bad:
        result = run(program, ["describe", file])
        if result.exit_code != 1 or result.stdout or file not in result.stderr:
            wrong[file] = (result.exit_code, result.stdout, result.stderr[-500:])
    assert not wrong, f"files not refused with status 1, no stdout, and a stderr naming them: {wrong}"


# ---------------------------------------------------------------------------
# The C side


def test_the_c_program_agrees_on_codes_and_sizes_and_copies_a_file(programs: dict, workdir: Path) -> None:
    program = installed(programs, "c", workdir)
    module = lassi_io()
    assert {name: (module.DTYPES[name].code, module.DTYPES[name].itemsize) for name in REQUIRED} == REQUIRED
    module.write_array(workdir / "h.lassiio", "h", "bf16", (2, 4), H_DATA)
    result = run(program, ["h.lassiio", "c_copy.lassiio"])
    assert result.exit_code == 0, result
    assert result.stdout.splitlines() == [f"{name} {code} {size}" for name, (code, size) in REQUIRED.items()]
    check_outputs(result, {"c_copy.lassiio": ("c_copy", "bf16", (2, 4), H_DATA)}, workdir.parents[1] / "python")
