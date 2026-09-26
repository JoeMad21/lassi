"""Tests for the tt-metal host toolchain `ttmetal-host` and its pin (task P4.10).

Bible: Component Interfaces (Toolchain row and contract rules), Toolchain
Pins (the joint pin; Install; the host compiler pin; the --version check
before the first build), Harness Contract (device kernels compile at first
launch), Repository Layout (toolchains/), Sandbox (compiles), Agent Rules 1,
7, and 10; plans/PHASE-NOTES.md, P4 (the JIT cache rule; the CMake-fetched
packages).

The design these tests fix:

Registration. Importing lassi.toolchains registers the Toolchain
"ttmetal-host" (module lassi.toolchains.ttmetal_build, the name Repository
Layout gives). It declares `diagnostics` and no offload capability. Its PIN
is "tt-metal" and it declares no PIN_BIN: the compiler is the build host's
clang++-20, pinned by path as the host g++ is (toolchains/gcc.pin), so a
trial it builds records toolchain_pins.tt_metal.

Flags: the pinned build's own line, recorded once in the pin. The pinned
tree records how it built the gate's example,
metal_example_add_2_integers_in_riscv, in build_Release/build.ninja: one
object, compiled by `/usr/bin/clang++-20 $DEFINES $INCLUDES $FLAGS ... -c`
and linked by `/usr/bin/clang++-20 $FLAGS $LINK_FLAGS $in -o $TARGET_FILE
$LINK_PATH $LINK_LIBRARIES` (rules.ninja, rx 20260925-223734-exec-4611),
with the values rx 20260925-223402-exec-32ce read. The five Tier A examples
use the same values; the two matmul ones add only Matmul::Common's include
directory (exec 4611). Rejected: parsing build.ninja at every construction
(a generated file outside the repository, so nothing here could test what
is read) and configuring with CMake against the tree (a configure per
build, and CPM would fetch). Chosen: toolchains/tt-metal.pin records those
values, as the bible says a pin records its build flags, in five keys whose
values are argv words separated by spaces, `{TREE}` standing for the
installed tree, <resolved toolchains root>/<PREFIX_NAME>:

- HOST_DEFINES: the example's DEFINES without OVERRIDE_KERNEL_PREFIX, which
  only the programming examples' CMakeLists adds. Each of the six examples
  defines it itself when it is unset (#ifndef OVERRIDE_KERNEL_PREFIX, exec
  4611), as "" where that exec printed the value (loopback, eltwise_sfpu;
  the remote tests check all six), so the kernel paths a program names are
  relative to its working directory.
- HOST_INCLUDES: as recorded; "-isystem" is a word of its own.
- HOST_FLAGS: as recorded without -Werror (OQ-027, option (b), applied under
  the owner's standing direction of 2026-09-26): -Wall, -Wunused-parameter,
  and every other word stay, so a warning in a host program stays a warning
  and counts in df-v0's W as it does for gcc-native. "-Xclang" is a word of
  its own.
- HOST_LINK_FLAGS: the LINK_FLAGS without CMake's dependency file (the
  "-Xlinker --dependency-file=..." pair).
- HOST_LINK_LIBRARIES: the rpath, then the libraries, each an absolute path
  under {TREE}/build_Release (build.ninja names them relative to the build
  directory).

The pin also gains EXECUTABLE, /usr/bin/clang++-20 (the compiler rules.ninja
names, and `command -v clang++-20` in exec 32ce), and EXPECT_VERSION, which
holds the pin's CLANG_EXPECT and lies in the banner exec 32ce printed.

The command is one compile and link, with {TREE} replaced by the tree in
POSIX form:

    <EXECUTABLE> HOST_DEFINES HOST_INCLUDES -idirafter . HOST_FLAGS
        HOST_LINK_FLAGS -o main <host sources> HOST_LINK_LIBRARIES

"-idirafter ." is the one word the pinned build lacks: the build directory
is searched after every pinned directory for an angle-bracket include
(a quoted include searches the including file's directory first), and
a header placed beside the program is found at the path the program
names in either include form (the remote tests place the matmul
examples' Matmul::Common header that way). The toolchain reads the flags
from the pin when it is constructed.

No loader variable. The link gives the program a RUNPATH naming the tree's
four library directories, as the pinned example has (readelf -d, exec
32ce), and the compile environment is the runner's clean one (PATH, LANG=C,
LC_ALL=C), with no LD_LIBRARY_PATH. Nothing runs at build time, so the build
sets no TT_METAL_* variable; the executor that runs a program sets
TT_METAL_RUNTIME_ROOT to the tree and TT_METAL_CACHE to a LASSI-owned
directory (PHASE-NOTES P4; P4.11), and the toolchain keeps the tree as its
attribute `tree` for that.

Host and kernel sources. A C++ source (.cpp, .cc, .cxx) with a directory
named `kernels` in its path is a kernel source: written at its path in the
build directory and never given to the host compiler, since the kernel JIT
compiles it at first launch (Harness Contract). Every other C++ source is a
host source, compiled in sorted order; headers and harness files are never
sources. The artifact is <build dir>/main, and a run's working directory is
the artifact's directory (lassi.executors.native; P4.11's executor runs the
same way). At the pin, tt-metal looks a kernel up in the working directory
first, then TT_METAL_KERNEL_PATH, the system kernel directory, and the
runtime root (tt_metal/impl/kernels/kernel.cpp, as read for P4.9 in
plans/spikes/p4-ttsim-runtime.md), so a kernel written at the path the
program names is the one the JIT finds.

The pin check, in lassi.core.runner.build_toolchain before the first build:
first, before any process starts, the tree must be an install of the pin:

- <root>/<PREFIX_NAME> holds lassi-install.txt (not only
  lassi-install.unfinished), and its first line names the pin's COMMIT;
- its lassi-cpm-sources.txt equals, line for line, the tracked copy
  toolchains/tt-metal-cpm-sources.txt of the list P4.2 recorded
  (results/p4-tt-install/summary.md, CMake-fetched packages, read by rx
  exec 20260925-173747-exec-fd38). The copy is read from
  lassi.toolchains.pins.PINS_DIR, beside the pin, and its "#" lines are
  comments that cite that record. results/ holds evidence: a refusal that
  parsed a summary's code block would hang on a report's layout, while a
  copy under toolchains/ is a pin, which Toolchain Pins already governs (a
  change is a Decision Log entry).

Then, as for every pinned compiler, `<EXECUTABLE> --version` runs once
through the compile runner and must print EXPECT_VERSION. Each refusal is a
RunError.

No compiler runs here. The build tests give the toolchain a fake command
runner; the build_toolchain tests replace lassi.core.runner's
SandboxedCompileRunner with a fake whose --version answer is a PLACEHOLDER
banner, point the pin reader at a copy of toolchains/ whose tt-metal.pin
EXECUTABLE is an empty stand-in file, and build a stand-in tree holding the
install record and CPM list P4.2 recorded, with an empty stand-in for every
directory and library the pinned flags name. The real build runs on the
build host in test_ttmetal_host_remote.py. No value here is a measurement:
the flags, paths, and banner are what the named rx execs printed.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

import lassi.toolchains  # noqa: F401  (importing the package registers every toolchain preset)
from lassi.core import runner as runner_module
from lassi.core.interfaces import BuildResult
from lassi.core.record import Diagnostic
from lassi.core.registry import DEFAULT_REGISTRY, RegistryError
from lassi.core.runner import RunError, build_toolchain
from lassi.toolchains import CommandResult
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
TOOLCHAINS_DIR = REPO / "toolchains"
TT_METAL_PIN = TOOLCHAINS_DIR / "tt-metal.pin"
CPM_LIST_NAME = "tt-metal-cpm-sources.txt"
CPM_LIST = TOOLCHAINS_DIR / CPM_LIST_NAME
INSTALL_SUMMARY = REPO / "results" / "p4-tt-install" / "summary.md"

NAME = "ttmetal-host"
MODULE = "lassi.toolchains.ttmetal_build"
PIN_NAME = "tt-metal"
PREFIX = "tt-metal@5280a9cf"
COMMIT = "5280a9cfb00998fd49667a29523d03aee905c129"
INSTALL_RECORD = "lassi-install.txt"
UNFINISHED_RECORD = "lassi-install.unfinished"
TREE_CPM_LIST = "lassi-cpm-sources.txt"
OUTPUT = "main"
ATTACHMENT = "compile.stderr"
PLACEHOLDER_TREE = "{TREE}"

# The installed tree on the build host (results/p4-tt-install/summary.md, the install record's "local" line).
HOST_TREE = "/mnt/nvme10/joseph_ufl/toolchains/tt-metal@5280a9cf"
# The compiler the pinned build ran (rules.ninja, rx 20260925-223734-exec-4611) and `command -v clang++-20`
# printed (rx 20260925-223402-exec-32ce).
HOST_CLANG = "/usr/bin/clang++-20"
# The first line `clang++-20 --version` printed on the build host in rx 20260925-223402-exec-32ce.
RECORDED_BANNER = "Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)"
# The --version text the fake compile runner prints: a PLACEHOLDER line, then the pin's EXPECT_VERSION.
BANNER_LEAD = "PLACEHOLDER banner of a fake compile runner, not a compiler"

# The gate example's build line, as rx 20260925-223402-exec-32ce read it from build_Release/build.ninja, with the
# tree written {TREE} and the CPM cache {TREE}/.cpmcache. Left out, as the module docstring says why: the define
# -DOVERRIDE_KERNEL_PREFIX=\"tt_metal/programming_examples/\", the link words -Xlinker
# --dependency-file=tt_metal/programming_examples/add_2_integers_in_riscv/CMakeFiles/
# metal_example_add_2_integers_in_riscv.dir/link.d, and -Werror, which followed -Wall (OQ-027, option (b)). The
# link FLAGS, -O3 -DNDEBUG, are the first two FLAGS.
DEFINES = (
    "-DENCHANTUM_ENABLE_MSVC_SPEEDUP=1",
    "-DFMT_HEADER_ONLY=1",
    "-DSPDLOG_FMT_EXTERNAL",
    "-DSPDLOG_FWRITE_UNLOCKED",
    "-DTRACY_IMPORTS",
    "-DTRACY_TIMER_FALLBACK",
    "-DTT_ENABLE_LIGHT_METAL_TRACE=1",
)
INCLUDES = (
    "-I{TREE}/tt_stl/.",
    "-I{TREE}/tt_metal/api",
    "-I{TREE}/build_Release/tt_metal/api",
    "-I{TREE}/tt_metal/hostdevcommon/api",
    "-I{TREE}",
    "-isystem",
    "{TREE}/.cpmcache/reflect/f93e77475670eaeacf332927dfe8b50e3f3812e0",
    "-isystem",
    "{TREE}/.cpmcache/enchantum/2fb7ab238e36c101b9848892ddb6382276b65837/enchantum/include",
    "-isystem",
    "{TREE}/.cpmcache/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include",
    "-isystem",
    "{TREE}/.cpmcache/fmt/69912fb6b71fcb1f7e5deca191a2bb4748c4e7b6/include",
    "-isystem",
    "{TREE}/.cpmcache/tt-logger/87c1a5f2e9d2dd011200eb49c86426c26dec719e/include",
    "-isystem",
    "{TREE}/.cpmcache/spdlog/b1c2586bb5c35a7929362e87f62433eb68206873/include",
    "-isystem",
    "{TREE}/tt_metal/third_party/umd/device/api",
    "-isystem",
    "{TREE}/tt_metal/third_party/tracy/public",
)
FLAGS = (
    "-O3",
    "-DNDEBUG",
    "-std=c++20",
    "-fPIE",
    "-fsized-deallocation",
    "-gz",
    "-march=x86-64-v3",
    "-fPIC",
    "-pipe",
    "-fvisibility-inlines-hidden",
    "-Wall",
    "-Wconditional-uninitialized",
    "-Wno-deprecated-declarations",
    "-Xclang",
    "-fno-pch-timestamp",
    "-Wunused-parameter",
    "-Wno-c++11-narrowing",
    "-Wno-int-to-pointer-cast",
)
LINK_FLAGS = ("-Wl,--compress-debug-sections=zlib", "-fuse-ld=lld")
RPATH_DIRS = (
    "{TREE}/build_Release/tt_metal",
    "{TREE}/build_Release/tt_stl",
    "{TREE}/build_Release/lib",
    "{TREE}/build_Release/tt_metal/third_party/umd/device",
)
LINK_LIBRARIES = (
    "-Wl,-rpath," + ":".join(RPATH_DIRS),
    "{TREE}/build_Release/tt_metal/libtt_metal.so",
    "{TREE}/build_Release/tt_stl/libtt_stl.so",
    "{TREE}/build_Release/lib/libtracy.so.0.10.0",
    "-ldl",
    "{TREE}/build_Release/tt_metal/third_party/umd/device/libdevice.so",
)
HOST_KEYS = {
    "HOST_DEFINES": DEFINES,
    "HOST_INCLUDES": INCLUDES,
    "HOST_FLAGS": FLAGS,
    "HOST_LINK_FLAGS": LINK_FLAGS,
    "HOST_LINK_LIBRARIES": LINK_LIBRARIES,
}
# The word the toolchain adds after the pinned include directories: the build directory, searched last.
BUILD_DIR_INCLUDE = ("-idirafter", ".")
KERNEL_DIR = "kernels"
SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx")


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables, give HOME and TMPDIR test directories, so no test reads the real host layout."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "LD_LIBRARY_PATH", "TT_METAL_RUNTIME_ROOT"):
        monkeypatch.delenv(name, raising=False)
    for name in ("home", "compile-tmp"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "compile-tmp"))


def ttmetal_class() -> type:
    """Return the Toolchain class registered as ttmetal-host; fail the test clearly while it is missing."""
    try:
        return DEFAULT_REGISTRY.get("Toolchain", NAME).factory
    except RegistryError as error:
        pytest.fail(f"no Toolchain is registered as {NAME!r} ({error}); task P4.10 adds it to lassi.toolchains")


def pin_value(key: str) -> str:
    """Return one value of toolchains/tt-metal.pin; fail the test clearly while the pin lacks it."""
    pin = pins_module.read_pin(PIN_NAME)
    if not pin.get(key):
        pytest.fail(f"toolchains/tt-metal.pin has no {key}; task P4.10 records the host build there")
    return pin[key]


def fill(words: Sequence[str], tree: str) -> list[str]:
    """Return `words` with {TREE} replaced by `tree`."""
    return [word.replace(PLACEHOLDER_TREE, tree) for word in words]


def expected_command(executable: str, tree: str, sources: Sequence[str]) -> list[str]:
    """Return the command the toolchain must run for `sources` against the tree `tree` (POSIX form)."""
    return [
        executable,
        *DEFINES,
        *fill(INCLUDES, tree),
        *BUILD_DIR_INCLUDE,
        *FLAGS,
        *LINK_FLAGS,
        "-o",
        OUTPUT,
        *sources,
        *fill(LINK_LIBRARIES, tree),
    ]


def summary_block(heading: str) -> list[str]:
    """Return the non-blank lines of the first code block under `## <heading>` in the P4.2 install summary."""
    text = INSTALL_SUMMARY.read_text(encoding="utf-8")
    section = text.split(f"\n## {heading}\n", 1)[1]
    block = section.split("```\n", 2)[1]
    return [line.rstrip() for line in block.splitlines() if line.strip() and not line.startswith("[rx] ")]


def recorded_cpm_lines() -> list[str]:
    """Return the 25 lines of lassi-cpm-sources.txt that P4.2 recorded (rx exec 20260925-173747-exec-fd38)."""
    return summary_block("CMake-fetched packages")


def recorded_install_lines() -> list[str]:
    """Return the lines of lassi-install.txt that P4.2 recorded, without the exec's own status line."""
    return summary_block("Install record")


def tracked_cpm_lines() -> list[str]:
    """Return the non-comment, non-blank lines of toolchains/tt-metal-cpm-sources.txt; fail while it is missing."""
    if not CPM_LIST.is_file():
        pytest.fail(f"{CPM_LIST} does not exist; task P4.10 tracks P4.2's CMake-fetched packages list there")
    lines = CPM_LIST.read_text(encoding="ascii").splitlines()
    return [line.rstrip() for line in lines if line.strip() and not line.startswith("#")]


def is_kernel(path: str) -> bool:
    """Return True for a file under a directory named kernels, which the host compiler never builds."""
    return KERNEL_DIR in PurePosixPath(path).parts[:-1]


# ---------------------------------------------------------------------------
# Registration and capabilities


def test_the_package_registers_ttmetal_host() -> None:
    assert NAME in DEFAULT_REGISTRY.names("Toolchain"), f"importing lassi.toolchains registers {NAME!r}"
    cls = ttmetal_class()
    assert cls.name == NAME
    assert cls.__module__ == MODULE, "the module Repository Layout names, ttmetal_build"


def test_it_declares_diagnostics_and_offloads_nothing() -> None:
    ttmetal_class()
    capabilities = DEFAULT_REGISTRY.get("Toolchain", NAME).capabilities
    assert "diagnostics" in capabilities, "compile_loop and baseline require Toolchain capability 'diagnostics'"
    assert "openmp_offload" not in capabilities


def test_it_is_pinned_by_the_tt_metal_pin_and_uses_the_host_clang() -> None:
    cls = ttmetal_class()
    assert getattr(cls, "PIN", None) == PIN_NAME, "the class names its pin file toolchains/tt-metal.pin"
    assert getattr(cls, "PIN_BIN", None) is None, "the compiler is the host clang++-20, not a file under the tree"


# ---------------------------------------------------------------------------
# The pin: the host compiler and the pinned build's flags


def test_the_pin_names_the_host_clang_by_path() -> None:
    executable = pin_value("EXECUTABLE")
    assert executable == HOST_CLANG, "the compiler the pinned build ran (rules.ninja, rx exec 4611)"
    assert PurePosixPath(executable).name == pin_value("CLANG_CXX"), "the clang++ the install checked"


def test_the_expected_version_is_the_pinned_clang_in_the_recorded_banner() -> None:
    expected = pin_value("EXPECT_VERSION")
    assert pin_value("CLANG_EXPECT") in expected, "the host compiler version the install pinned (clang 20.1.8)"
    assert expected in RECORDED_BANNER, "the text clang++-20 --version printed on the build host (rx exec 32ce)"


@pytest.mark.parametrize("key", sorted(HOST_KEYS))
def test_the_pin_records_the_gate_examples_build_line(key: str) -> None:
    assert pin_value(key).split() == list(HOST_KEYS[key]), f"{key} is the pinned build's line, read in rx exec 32ce"


def test_the_pin_leaves_out_the_example_define_and_the_dependency_file() -> None:
    for key in HOST_KEYS:
        value = pin_value(key)
        assert "OVERRIDE_KERNEL_PREFIX" not in value, "a define only the programming examples' CMakeLists adds"
        assert "--dependency-file" not in value and "-Xlinker" not in value, "CMake's dependency file stays out"
        assert '"' not in value and "\\" not in value, "every word is a plain word, with no quoting"
        assert "-Werror" not in value.split(), "OQ-027, option (b): a warning in a host program stays a warning"


def test_every_linked_library_lies_in_an_rpath_directory_of_the_tree() -> None:
    words = pin_value("HOST_LINK_LIBRARIES").split()
    rpaths = [word for word in words if word.startswith("-Wl,-rpath,")]
    assert len(rpaths) == 1, "one rpath, so the program needs no loader variable"
    directories = rpaths[0][len("-Wl,-rpath,") :].split(":")
    assert all(directory.startswith(PLACEHOLDER_TREE + "/") for directory in directories), directories
    libraries = [word for word in words if not word.startswith("-")]
    assert libraries, "the link names the tree's libraries by path"
    for library in libraries:
        assert str(PurePosixPath(library).parent) in directories, f"{library} is found through the rpath"


def test_every_cpm_include_directory_is_a_recorded_cpm_source() -> None:
    sources = {line.split()[0] for line in tracked_cpm_lines()}
    marker = PLACEHOLDER_TREE + "/.cpmcache/"
    found = [word[len(marker) :] for word in pin_value("HOST_INCLUDES").split() if word.startswith(marker)]
    assert found, "the host build includes CMake-fetched packages"
    for directory in found:
        name, key = directory.split("/")[:2]
        assert f"{name}/{key}" in sources, f"{directory} is not in {CPM_LIST_NAME}"


# ---------------------------------------------------------------------------
# The tracked CPM list


def test_the_tracked_cpm_list_is_the_install_record() -> None:
    recorded = recorded_cpm_lines()
    assert len(recorded) == 25, "P4.2 recorded 25 CMake-fetched packages"
    assert tracked_cpm_lines() == recorded, "a line for line copy of the list in results/p4-tt-install/summary.md"


def test_the_tracked_cpm_list_is_plain_ascii_and_cites_its_record() -> None:
    tracked_cpm_lines()
    raw = CPM_LIST.read_bytes()
    assert raw.isascii() and b"\r" not in raw and raw.endswith(b"\n")
    comments = [line for line in raw.decode("ascii").splitlines() if line.startswith("#")]
    assert any("results/p4-tt-install" in line for line in comments), "a comment names the record it copies"


# ---------------------------------------------------------------------------
# The command line and build()


@dataclass
class FakeRunner:
    """A CommandRunner that records each call and answers with `status` and `stderr`; it runs nothing.

    On status 0 it writes a PLACEHOLDER artifact `main` in the workdir, as a
    compiler would.
    """

    status: int = 0
    stderr: str = ""
    calls: list[tuple[list[str], Path]] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the call and return the scripted result."""
        self.calls.append((list(argv), Path(cwd)))
        if self.status == 0:
            (Path(cwd) / OUTPUT).write_bytes(b"PLACEHOLDER artifact of a fake compile\n")
        return CommandResult(self.status, "", self.stderr)


def make_toolchain(runner: FakeRunner, tree: Path | None = None) -> Any:
    """Return a ttmetal-host toolchain built as the runner builds it: executable, runner, and tree given."""
    return ttmetal_class()(executable=HOST_CLANG, runner=runner, tree=tree or Path(HOST_TREE))


def test_the_command_is_the_pinned_compile_and_link_line() -> None:
    command = make_toolchain(FakeRunner()).command(["main.cpp"])
    assert command == expected_command(HOST_CLANG, HOST_TREE, ["main.cpp"])


def test_the_command_takes_its_flags_from_the_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ttmetal_class()
    pins_dir = copy_pins(tmp_path / "pins")
    pin = pins_module.read_pin(PIN_NAME)
    extra = "-DPLACEHOLDER_P410_FLAG=1"
    write_pin(pins_dir, {**pin, "HOST_FLAGS": f"{pin.get('HOST_FLAGS', '')} {extra}".strip()})
    monkeypatch.setattr(pins_module, "PINS_DIR", pins_dir)
    command = make_toolchain(FakeRunner()).command(["main.cpp"])
    assert extra in command, "the flags come from toolchains/tt-metal.pin, read when the toolchain is built"


def test_build_compiles_host_sources_only_and_places_kernels_at_their_paths(tmp_path: Path) -> None:
    runner = FakeRunner()
    workdir = tmp_path / "build"
    workdir.mkdir()
    kernels = {
        "kernels/dataflow/reader.cpp": '#include "dataflow_api.h"\nvoid kernel_main() {}\n',
        "kernels/compute/add.cpp": "// SYNTHETIC compute kernel\n",
        "p410_example/kernels/writer.cpp": "// SYNTHETIC kernel under an example directory\n",
    }
    files = {
        "main.cpp": '#include "host/util.h"\nint main() { return util(); }\n',
        "host/util.cpp": "int util() { return 0; }\n",
        "host/util.h": "int util();\n",
        **kernels,
    }
    harness = {"lassi_io.h": "/* SYNTHETIC stand-in for the harness header */\n"}
    result = make_toolchain(runner).build(files, workdir, harness=harness)
    assert isinstance(result, BuildResult)
    ((argv, cwd),) = runner.calls
    assert cwd == workdir
    assert argv == expected_command(HOST_CLANG, HOST_TREE, ["host/util.cpp", "main.cpp"]), (
        "host sources in sorted order; kernel sources and headers are never given to the host compiler"
    )
    assert result.artifact == workdir / OUTPUT
    for path, text in kernels.items():
        placed = result.artifact.parent / path
        assert placed.read_bytes() == text.encode("utf-8"), f"{path} lies at its path in the run's working directory"
    assert (workdir / "lassi_io.h").is_file()
    assert (workdir / ATTACHMENT).read_bytes() == b""


def test_a_build_with_only_kernel_sources_runs_nothing(tmp_path: Path) -> None:
    runner = FakeRunner()
    workdir = tmp_path / "build"
    workdir.mkdir()
    files = {"kernels/dataflow/reader.cpp": "// SYNTHETIC kernel\n"}
    result = make_toolchain(runner).build(files, workdir)
    assert runner.calls == [], "a kernel is never a host source"
    assert result.artifact is None
    assert [item.code for item in result.diagnostics] == ["no-sources"]
    assert (workdir / "kernels" / "dataflow" / "reader.cpp").is_file()


def test_a_failed_build_keeps_the_raw_stderr_and_parses_its_error(tmp_path: Path) -> None:
    # SYNTHETIC stderr in clang's format (not captured): line 6 of main.cpp is "        total += undefined_var;",
    # and clang's column 18 is the 'u', counted in main.cpp itself.
    stderr = (
        "main.cpp:6:18: error: use of undeclared identifier 'undefined_var'\n"
        "    6 |         total += undefined_var;\n"
        "      |                  ^\n"
        "1 error generated.\n"
    )
    runner = FakeRunner(status=1, stderr=stderr)
    workdir = tmp_path / "build"
    workdir.mkdir()
    main = (
        "#include <tt-metalium/host_api.hpp>\n\nint main() {\n    int total = 0;\n"
        "    for (int i = 0; i < 4; ++i) {\n        total += undefined_var;\n    }\n    return total;\n}\n"
    )
    result = make_toolchain(runner).build({"main.cpp": main}, workdir)
    assert result.artifact is None
    assert result.diagnostics == [
        Diagnostic(
            stage="compile",
            severity="error",
            code=None,
            file="main.cpp",
            line=6,
            column=18,
            message="use of undeclared identifier 'undefined_var'",
        )
    ]
    assert (workdir / ATTACHMENT).read_bytes() == stderr.encode("ascii")


# ---------------------------------------------------------------------------
# build_toolchain: the pin check before the first build


@dataclass
class CompileRunnerLog:
    """What the fake compile runners saw: each construction's keywords and each command, in order."""

    constructed: list[dict[str, Any]] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)
    banner: str = ""
    status: int = 0


def fake_compile_runner(log: CompileRunnerLog) -> type:
    """Return a stand-in for lassi.executors.sandbox.SandboxedCompileRunner that runs nothing and records calls."""

    class FakeCompileRunner:
        """Answers --version with the log's banner and status; any other command fails the test."""

        def __init__(self, **keywords: Any) -> None:
            log.constructed.append(dict(keywords))

        def spec(self, build: Path) -> None:
            """Accept any build dir layout; the real sandbox checks it on the build host."""

        def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
            log.calls.append(list(argv))
            assert list(argv[1:]) == ["--version"], f"only the --version check may run here, not {argv!r}"
            return CommandResult(log.status, log.banner, "")

    return FakeCompileRunner


def copy_pins(pins_dir: Path) -> Path:
    """Copy every file of toolchains/ into `pins_dir`, so the pin and the tracked CPM list are read from there."""
    pins_dir.mkdir(parents=True)
    for path in TOOLCHAINS_DIR.iterdir():
        if path.is_file():
            shutil.copyfile(path, pins_dir / path.name)
    return pins_dir


def write_pin(directory: Path, pairs: Mapping[str, str]) -> None:
    """Write `pairs` as a tt-metal.pin in `directory`, one quoted KEY="value" line each."""
    text = "# SYNTHETIC copy of toolchains/tt-metal.pin for a test\n" + "".join(
        f'{key}="{value}"\n' for key, value in pairs.items()
    )
    (directory / f"{PIN_NAME}.pin").write_bytes(text.encode("ascii"))


def lines_text(lines: Sequence[str]) -> bytes:
    """Return `lines` as ASCII text with LF line endings and a final newline."""
    return "".join(f"{line}\n" for line in lines).encode("ascii")


def make_tree(tree: Path) -> Path:
    """Build a stand-in install at `tree`: P4.2's install record and CPM list, and every path the flags name.

    The directories and libraries the pinned flags name are empty
    stand-ins, so an implementation may check that they exist.
    """
    tree.mkdir(parents=True)
    (tree / INSTALL_RECORD).write_bytes(lines_text(recorded_install_lines()))
    (tree / TREE_CPM_LIST).write_bytes(lines_text(recorded_cpm_lines()))
    posix = tree.as_posix()
    for word in fill(INCLUDES, posix):
        if word != "-isystem":
            Path(word[2:] if word.startswith("-I") else word).mkdir(parents=True, exist_ok=True)
    for directory in fill(RPATH_DIRS, posix):
        Path(directory).mkdir(parents=True, exist_ok=True)
    for word in fill(LINK_LIBRARIES, posix):
        if not word.startswith("-"):
            Path(word).write_bytes(b"")
    return tree


@dataclass(frozen=True)
class PinnedTree:
    """A copy of toolchains/ with a stand-in EXECUTABLE, a toolchains root holding a stand-in tree, and the log."""

    pin: dict[str, str]
    executable: Path
    root: Path
    tree: Path
    pins_dir: Path
    log: CompileRunnerLog


@pytest.fixture
def pinned_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PinnedTree:
    """Point the pin reader at a copy of toolchains/ whose EXECUTABLE is an empty stand-in; fake the compile runner."""
    ttmetal_class()
    real = pins_module.read_pin(PIN_NAME)
    executable = tmp_path / "host-bin" / PurePosixPath(real.get("EXECUTABLE", HOST_CLANG)).name
    executable.parent.mkdir()
    executable.write_bytes(b"")
    pins_dir = copy_pins(tmp_path / "pins")
    write_pin(pins_dir, {**real, "EXECUTABLE": str(executable)})
    monkeypatch.setattr(pins_module, "PINS_DIR", pins_dir)
    log = CompileRunnerLog(banner=f"{BANNER_LEAD}\n{real.get('EXPECT_VERSION', '')}\n")
    monkeypatch.setattr(runner_module, "SandboxedCompileRunner", fake_compile_runner(log))
    root = tmp_path / "toolchains-root"
    tree = make_tree(root / real["PREFIX_NAME"])
    return PinnedTree(pins_module.read_pin(PIN_NAME), executable, root, tree, pins_dir, log)


def test_build_toolchain_checks_the_pinned_clang_version_once_before_any_build(pinned_tree: PinnedTree) -> None:
    built = build_toolchain(NAME, pinned_tree.root)
    assert pinned_tree.log.calls == [[str(pinned_tree.executable), "--version"]], (
        "the --version check runs once, through the compile runner, before the first build"
    )
    assert built.executable == str(pinned_tree.executable), "the executable is the pin's EXECUTABLE, as given"
    assert built.toolchain.executable == str(pinned_tree.executable)
    assert built.pins == {PIN_NAME: pinned_tree.pin}, "a trial records toolchain_pins.tt_metal"
    assert built.version_status == 0
    assert pinned_tree.pin["EXPECT_VERSION"] in "\n".join(built.version)
    assert type(built.toolchain.runner).__name__ == "FakeCompileRunner", "builds use the compile runner too"


def test_build_toolchain_gives_the_toolchain_the_resolved_tree(pinned_tree: PinnedTree) -> None:
    built = build_toolchain(NAME, pinned_tree.root)
    tree = pinned_tree.root.resolve() / PREFIX
    assert Path(built.toolchain.tree) == tree
    command = built.toolchain.command(["main.cpp"])
    assert command == expected_command(str(pinned_tree.executable), tree.as_posix(), ["main.cpp"])


def test_build_toolchain_gives_clang_no_loader_or_tt_metal_variable(
    pinned_tree: PinnedTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/PLACEHOLDER/not/a/library/dir")
    monkeypatch.setenv("TT_METAL_RUNTIME_ROOT", "/PLACEHOLDER/not/the/tree")
    built = build_toolchain(NAME, pinned_tree.root)
    assert built.environment is not None
    assert set(built.environment) == {"PATH", "LANG", "LC_ALL"}, "the program finds its libraries by its rpath"
    assert (built.environment["LANG"], built.environment["LC_ALL"]) == ("C", "C")


def test_build_toolchain_refuses_a_clang_that_is_not_the_pinned_version(pinned_tree: PinnedTree) -> None:
    pinned_tree.log.banner = f"{BANNER_LEAD}\nPLACEHOLDER some other clang version\n"
    with pytest.raises(RunError, match="EXPECT_VERSION"):
        build_toolchain(NAME, pinned_tree.root)


def break_install(tree: Path, problem: str) -> None:
    """Change the stand-in tree so it is not the pinned install."""
    record = tree / INSTALL_RECORD
    if problem == "no-tree":
        shutil.rmtree(tree)
    elif problem == "unfinished":
        record.rename(tree / UNFINISHED_RECORD)
    elif problem == "other-commit":
        text = record.read_text(encoding="ascii")
        assert COMMIT in text.splitlines()[0]
        record.write_bytes(text.replace(COMMIT, "0" * 40, 1).encode("ascii"))
    else:
        raise AssertionError(problem)


@pytest.mark.parametrize("problem", ["no-tree", "unfinished", "other-commit"])
def test_build_toolchain_refuses_a_tree_that_is_not_the_pinned_install(pinned_tree: PinnedTree, problem: str) -> None:
    break_install(pinned_tree.tree, problem)
    with pytest.raises(RunError) as caught:
        build_toolchain(NAME, pinned_tree.root)
    if problem != "no-tree":
        assert INSTALL_RECORD in str(caught.value), "the refusal names the install record"
    assert pinned_tree.log.calls == [], "the tree is checked before any process starts"


def drift_cpm(tree: Path, problem: str) -> None:
    """Change the stand-in tree's lassi-cpm-sources.txt so it no longer matches the tracked list."""
    path = tree / TREE_CPM_LIST
    lines = recorded_cpm_lines()
    if problem == "changed-line":
        assert lines[0].endswith("changed=0")
        lines[0] = lines[0][: -len("changed=0")] + "changed=1"
    elif problem == "missing-line":
        lines = lines[:-1]
    elif problem == "extra-line":
        lines.append("p410-placeholder/0000 git 0000000000000000000000000000000000000000 changed=0")
    elif problem == "missing-file":
        path.unlink()
        return
    else:
        raise AssertionError(problem)
    path.write_bytes(lines_text(lines))


@pytest.mark.parametrize("problem", ["changed-line", "missing-line", "extra-line", "missing-file"])
def test_build_toolchain_refuses_cpm_sources_that_differ_from_the_tracked_list(
    pinned_tree: PinnedTree, problem: str
) -> None:
    drift_cpm(pinned_tree.tree, problem)
    with pytest.raises(RunError, match=TREE_CPM_LIST):
        build_toolchain(NAME, pinned_tree.root)
    assert pinned_tree.log.calls == [], "the CMake-fetched packages are compared before any process starts"


def test_build_toolchain_refuses_without_the_tracked_cpm_list(pinned_tree: PinnedTree) -> None:
    (pinned_tree.pins_dir / CPM_LIST_NAME).unlink(missing_ok=True)
    with pytest.raises(RunError, match=CPM_LIST_NAME):
        build_toolchain(NAME, pinned_tree.root)
    assert pinned_tree.log.calls == []


def test_build_toolchain_refuses_without_a_toolchains_root(pinned_tree: PinnedTree) -> None:
    with pytest.raises(RunError, match="LASSI_TOOLCHAINS"):
        build_toolchain(NAME, None)
    assert pinned_tree.log.calls == [] and pinned_tree.log.constructed == []
