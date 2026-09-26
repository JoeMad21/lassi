"""Tests for the native C++ toolchain `gcc-native` and its pin toolchains/gcc.pin (task P4.5).

Bible: Execution Backends (native row: `g++ -O3 -fopenmp`), Toolchain Pins
(pin files; the --version check against EXPECT_VERSION in the compile
sandbox before the first build), Component Interfaces (Toolchain row and
contract rules), Result Record (Trial.toolchain_pins), Agent Rules 1 and 10.

The contract these tests fix (plans/p4-ttsim.md, P4.5, and its planning
decision "Executors per language"):

- Importing lassi.toolchains registers a Toolchain "gcc-native" that builds
  C++ for the host CPU with the host g++. It declares `diagnostics` (so
  compile_loop and baseline may bind it) and `emits_warnings`, and not
  `openmp_offload`: it offloads nothing. Its class attribute PIN is "gcc".
- toolchains/gcc.pin pins the host g++ by version and path; the compiler is
  not installed by the project, so there is no toolchains/gcc.sh and no
  PREFIX_NAME. Keys: NAME=gcc; VERSION, the g++-12 version the P4.1 spike
  recorded on the build host (12.3.0, plans/spikes/p4-tt-pins.md, Host
  build prerequisites); EXECUTABLE, the absolute path of that g++ on the
  build host, outside the scratch disk; FLAGS, the native row's flags
  "-O3 -fopenmp", in that order; and EXPECT_VERSION, text its --version
  prints, which holds VERSION.
- The command line is `<EXECUTABLE> <FLAGS...> -o main <sources...>`, the
  sources in sorted order; headers and harness files are written beside
  them and never compiled as sources. Raw stderr stays in compile.stderr,
  and GCC's diagnostics parse into Diagnostic records (the parser rules are
  in test_gcc_diagnostics.py).
- lassi.core.runner.build_toolchain("gcc-native", root) builds it with the
  pin's EXECUTABLE (an absolute path on the host that runs, outside the
  toolchains root, which is accepted for this pin) and the clean compile
  environment (PATH, LANG=C, LC_ALL=C: GCC then quotes in plain ASCII), and
  runs `<EXECUTABLE> --version` once through the compile runner before any
  build; output without EXPECT_VERSION, a missing or relative EXECUTABLE,
  and a pin without EXPECT_VERSION are RunErrors, the last two before any
  command runs.
- A trial built by gcc-native records its pin: lassi.core.record gains the
  Trial.toolchain_pins field `gcc`.

No compiler runs here. The build tests give the toolchain a fake command
runner; the build_toolchain tests replace lassi.core.runner's
SandboxedCompileRunner with a fake whose --version answer is a PLACEHOLDER
banner, and point the pin reader at a temporary copy of gcc.pin whose
EXECUTABLE is an empty stand-in file. The real --version check and a real
compile run on the build host in test_gcc_native_remote.py. No value in this
module is a measurement; 12.3.0 is the version the spike read from the
host's package list.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

import lassi.toolchains  # noqa: F401  (importing the package registers every toolchain preset)
from lassi.core import record as record_module
from lassi.core import runner as runner_module
from lassi.core.interfaces import BuildResult
from lassi.core.record import Diagnostic
from lassi.core.registry import DEFAULT_REGISTRY, RegistryError
from lassi.core.runner import RunError, build_toolchain
from lassi.toolchains import CommandResult
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
BIBLE = REPO / "docs" / "BIBLE.md"
TOOLCHAINS_DIR = REPO / "toolchains"
GCC_PIN = TOOLCHAINS_DIR / "gcc.pin"

NAME = "gcc-native"
PIN_NAME = "gcc"
# The native row's flags (bible Execution Backends: `g++ -O3 -fopenmp`), in order.
NATIVE_FLAGS = ["-O3", "-fopenmp"]
# g++-12 12.3.0: the host fact of the P4.1 spike (plans/spikes/p4-tt-pins.md, Host build prerequisites, from
# the package list read in rx 20260924-185249-exec-1540). Not a measurement of anything.
PINNED_VERSION = "12.3.0"
SCRATCH_ROOT = "/mnt/nvme10/joseph_ufl/"
OUTPUT = "main"
ATTACHMENT = "compile.stderr"
# A stand-in executable path for the tests that never run a compiler.
STAND_IN = "/usr/bin/g++-12"
# The --version text the fake compile runner prints: a PLACEHOLDER line, then the pin's EXPECT_VERSION.
BANNER_LEAD = "PLACEHOLDER banner of a fake compile runner, not a compiler"


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables, give HOME and TMPDIR test directories, so no test reads the real host layout."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "CPATH", "CPLUS_INCLUDE_PATH"):
        monkeypatch.delenv(name, raising=False)
    for name in ("home", "compile-tmp"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "compile-tmp"))


def gcc_class() -> type:
    """Return the Toolchain class registered as gcc-native; fail the test clearly while it is missing."""
    try:
        return DEFAULT_REGISTRY.get("Toolchain", NAME).factory
    except RegistryError as error:
        pytest.fail(f"no Toolchain is registered as {NAME!r} ({error}); task P4.5 adds it to lassi.toolchains")


def gcc_pin() -> dict[str, str]:
    """Return toolchains/gcc.pin as the runner reads it; fail the test clearly while it is missing."""
    if not GCC_PIN.is_file():
        pytest.fail(f"{GCC_PIN} does not exist; task P4.5 pins the host g++ there")
    return pins_module.read_pin(PIN_NAME)


# ---------------------------------------------------------------------------
# Registration and capabilities


def test_the_package_registers_gcc_native() -> None:
    assert NAME in DEFAULT_REGISTRY.names("Toolchain"), f"importing lassi.toolchains registers {NAME!r}"
    assert gcc_class().name == NAME


def test_it_declares_diagnostics_and_warnings_and_offloads_nothing() -> None:
    gcc_class()
    capabilities = DEFAULT_REGISTRY.get("Toolchain", NAME).capabilities
    assert "diagnostics" in capabilities, "compile_loop and baseline require Toolchain capability 'diagnostics'"
    assert "emits_warnings" in capabilities
    assert "openmp_offload" not in capabilities, "the host g++ builds OpenMP for the host CPU; it offloads nothing"


def test_it_is_pinned_by_the_gcc_pin_file() -> None:
    assert getattr(gcc_class(), "PIN", None) == PIN_NAME, "the class names its pin file toolchains/gcc.pin"


# ---------------------------------------------------------------------------
# The pin file


def test_the_pin_records_name_version_executable_flags_and_expected_version() -> None:
    pin = gcc_pin()
    assert pin.get("NAME") == PIN_NAME
    assert pin.get("VERSION") == PINNED_VERSION, "the version the P4.1 spike recorded for g++-12 on the build host"
    executable = pin.get("EXECUTABLE", "")
    assert PurePosixPath(executable).is_absolute(), f"EXECUTABLE is an absolute path on the build host: {executable!r}"
    assert re.fullmatch(r"g\+\+(-[0-9]+)?", PurePosixPath(executable).name), executable
    assert pin.get("FLAGS", "").split() == NATIVE_FLAGS, "FLAGS are the native row's flags, in order"
    expected = pin.get("EXPECT_VERSION", "")
    assert expected.strip(), "EXPECT_VERSION is the text the pinned g++ prints for --version"
    assert PINNED_VERSION in expected, "the expected --version text names the pinned version"


def test_the_pin_names_a_host_compiler_that_the_project_does_not_install() -> None:
    pin = gcc_pin()
    assert "PREFIX_NAME" not in pin, "the host g++ has no install prefix under $LASSI_TOOLCHAINS"
    assert not (TOOLCHAINS_DIR / "gcc.sh").exists(), "the host g++ is pinned by version and path, not installed"
    assert not pin["EXECUTABLE"].startswith(SCRATCH_ROOT), "EXECUTABLE is the host's g++, not a scratch install"


def test_the_pin_file_is_plain_ascii_with_lf_and_comments() -> None:
    gcc_pin()
    raw = GCC_PIN.read_bytes()
    assert raw.isascii() and b"\r" not in raw
    lines = raw.decode("ascii").splitlines()
    assert any(line.startswith("#") for line in lines), "the pin file says where its values come from"


def native_row() -> str:
    """Return the native row of the bible's Execution Backends table."""
    rows = [line for line in BIBLE.read_text(encoding="utf-8").splitlines() if line.startswith("| native |")]
    assert len(rows) == 1, "the bible's Execution Backends table has one native row"
    return rows[0]


def test_the_pinned_flags_are_the_native_rows_flags() -> None:
    commands = [text.split() for text in re.findall(r"`([^`]+)`", native_row()) if text.startswith("g++ ")]
    assert commands, "the native row names a g++ command"
    assert commands[0][1:] == NATIVE_FLAGS, "the bible's native row gives g++ -O3 -fopenmp"
    assert gcc_pin()["FLAGS"].split() == commands[0][1:]


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


def make_toolchain(runner: FakeRunner, executable: str = STAND_IN) -> Any:
    """Return a gcc-native toolchain built as the runner builds a pinned one: executable and runner given."""
    return gcc_class()(executable=executable, runner=runner)


def test_the_command_is_the_executable_the_native_flags_the_output_then_the_sources() -> None:
    toolchain = make_toolchain(FakeRunner())
    command = toolchain.command(["a.cpp", "main.cpp"])
    assert command == [STAND_IN, *NATIVE_FLAGS, "-o", OUTPUT, "a.cpp", "main.cpp"]


def test_the_command_flags_are_the_pins_flags() -> None:
    command = make_toolchain(FakeRunner()).command(["main.cpp"])
    assert command[1:-3] == gcc_pin()["FLAGS"].split()


def test_build_compiles_the_cpp_sources_in_sorted_order_and_writes_headers_and_harness_beside_them(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    workdir = tmp_path / "build"
    workdir.mkdir()
    files = {
        "main.cpp": '#include "kernels/scale.h"\n#include "lassi_io.h"\nint main() { return 0; }\n',
        "util.cc": "int util() { return 1; }\n",
        "kernels/scale.h": "inline int scale(int x) { return 2 * x; }\n",
    }
    harness = {"lassi_io.h": "/* SYNTHETIC stand-in for the harness header */\n"}
    result = make_toolchain(runner).build(files, workdir, harness=harness)
    assert isinstance(result, BuildResult)
    ((argv, cwd),) = runner.calls
    assert cwd == workdir
    assert argv == [STAND_IN, *NATIVE_FLAGS, "-o", OUTPUT, "main.cpp", "util.cc"], "headers are never sources"
    assert result.artifact == workdir / OUTPUT
    assert result.diagnostics == []
    assert (workdir / "lassi_io.h").read_text(encoding="utf-8") == harness["lassi_io.h"]
    assert (workdir / "kernels" / "scale.h").is_file()
    assert (workdir / ATTACHMENT).read_bytes() == b"", "the raw stderr attachment is kept, empty for a clean build"


def test_a_failed_build_keeps_the_raw_stderr_and_parses_its_error(tmp_path: Path) -> None:
    # SYNTHETIC stderr in GCC's format (not captured): line 3 of main.cpp is "    x = 1;", and GCC's column 5 is
    # the 'x', counted in main.cpp itself.
    stderr = (
        "main.cpp: In function 'int main()':\n"
        "main.cpp:3:5: error: 'x' was not declared in this scope\n"
        "    3 |     x = 1;\n"
        "      |     ^\n"
    )
    runner = FakeRunner(status=1, stderr=stderr)
    workdir = tmp_path / "build"
    workdir.mkdir()
    files = {"main.cpp": "int main() {\n    int y = 0;\n    x = 1;\n    return y;\n}\n"}
    result = make_toolchain(runner).build(files, workdir)
    assert result.artifact is None
    assert result.diagnostics == [
        Diagnostic(
            stage="compile",
            severity="error",
            code=None,
            file="main.cpp",
            line=3,
            column=5,
            message="'x' was not declared in this scope",
        )
    ]
    assert (workdir / ATTACHMENT).read_bytes() == stderr.encode("ascii")


# ---------------------------------------------------------------------------
# build_toolchain: the pinned host g++, checked by --version in the compile runner before any build


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


@dataclass(frozen=True)
class PinnedHost:
    """A temporary pin directory holding gcc.pin, its stand-in executable, a toolchains root, and the runner log."""

    pin: dict[str, str]
    executable: Path
    root: Path
    log: CompileRunnerLog


def write_pin(directory: Path, pairs: Mapping[str, str]) -> None:
    """Write `pairs` as a gcc.pin in `directory`, one quoted KEY="value" line each."""
    text = "# SYNTHETIC copy of toolchains/gcc.pin for a test\n" + "".join(
        f'{key}="{value}"\n' for key, value in pairs.items()
    )
    (directory / "gcc.pin").write_bytes(text.encode("ascii"))


@pytest.fixture
def pinned_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PinnedHost:
    """Point the pin reader at a copy of gcc.pin whose EXECUTABLE is an empty stand-in; fake the compile runner."""
    real = gcc_pin()
    executable = tmp_path / "host-bin" / PurePosixPath(real["EXECUTABLE"]).name
    executable.parent.mkdir()
    executable.write_bytes(b"")
    pin = {**real, "EXECUTABLE": str(executable)}
    pins_dir = tmp_path / "pins"
    pins_dir.mkdir()
    write_pin(pins_dir, pin)
    monkeypatch.setattr(pins_module, "PINS_DIR", pins_dir)
    log = CompileRunnerLog(banner=f"{BANNER_LEAD}\n{real['EXPECT_VERSION']}\n")
    monkeypatch.setattr(runner_module, "SandboxedCompileRunner", fake_compile_runner(log))
    root = tmp_path / "toolchains-root"
    root.mkdir()
    return PinnedHost(pins_module.read_pin(PIN_NAME), executable, root, log)


def test_build_toolchain_checks_the_pinned_gpp_version_once_before_any_build(pinned_host: PinnedHost) -> None:
    built = build_toolchain(NAME, pinned_host.root)
    assert pinned_host.log.calls == [[str(pinned_host.executable), "--version"]], (
        "the --version check runs once, through the compile runner, before the first build"
    )
    assert built.executable == str(pinned_host.executable), "the executable is the pin's EXECUTABLE, as given"
    assert built.toolchain.executable == str(pinned_host.executable)
    assert built.pins == {PIN_NAME: pinned_host.pin}
    assert built.version_status == 0
    assert pinned_host.pin["EXPECT_VERSION"] in "\n".join(built.version)
    assert len(pinned_host.log.constructed) == 1, "one compile runner serves the check and every build"
    assert type(built.toolchain.runner).__name__ == "FakeCompileRunner", "builds use the compile runner too"


def test_build_toolchain_gives_gpp_the_clean_c_locale_environment(pinned_host: PinnedHost) -> None:
    built = build_toolchain(NAME, pinned_host.root)
    assert built.environment is not None
    assert set(built.environment) == {"PATH", "LANG", "LC_ALL"}
    assert (built.environment["LANG"], built.environment["LC_ALL"]) == ("C", "C"), "GCC quotes in ASCII under C"


def test_build_toolchain_refuses_a_gpp_that_is_not_the_pinned_version(pinned_host: PinnedHost) -> None:
    pinned_host.log.banner = f"{BANNER_LEAD}\nPLACEHOLDER some other g++ version\n"
    with pytest.raises(RunError, match="EXPECT_VERSION"):
        build_toolchain(NAME, pinned_host.root)


def test_build_toolchain_refuses_a_failed_version_check(pinned_host: PinnedHost) -> None:
    pinned_host.log.status = 1
    with pytest.raises(RunError):
        build_toolchain(NAME, pinned_host.root)


@pytest.mark.parametrize("problem", ["missing", "relative"])
def test_build_toolchain_refuses_a_missing_or_relative_executable_before_anything_runs(
    pinned_host: PinnedHost, tmp_path: Path, problem: str
) -> None:
    executable = str(tmp_path / "no-such-dir" / "g++-12") if problem == "missing" else "g++-12"
    write_pin(pins_module.PINS_DIR, {**pinned_host.pin, "EXECUTABLE": executable})
    with pytest.raises(RunError) as caught:
        build_toolchain(NAME, pinned_host.root)
    assert "g++-12" in str(caught.value), "the refusal names the executable"
    assert pinned_host.log.calls == [], "nothing runs before the executable is known to be the pinned one"


def test_build_toolchain_refuses_a_pin_without_expect_version_before_anything_runs(pinned_host: PinnedHost) -> None:
    pin = {key: value for key, value in pinned_host.pin.items() if key != "EXPECT_VERSION"}
    write_pin(pins_module.PINS_DIR, pin)
    with pytest.raises(RunError, match="EXPECT_VERSION"):
        build_toolchain(NAME, pinned_host.root)
    assert pinned_host.log.calls == []


# ---------------------------------------------------------------------------
# The Result Record


def test_a_trial_records_the_gcc_pin_in_its_toolchain_pins() -> None:
    assert "gcc" in record_module.TOOLCHAIN_PIN_NAMES, "Trial.toolchain_pins gains the field gcc (Result Record)"
    pins = record_module.ToolchainPins(**{"gcc": PINNED_VERSION})
    assert pins.gcc == PINNED_VERSION


# ---------------------------------------------------------------------------
# Refusals the commit audit of P4.5 found covered only by its probes


def test_build_toolchain_refuses_a_host_pin_without_executable_before_anything_runs(pinned_host: PinnedHost) -> None:
    pin = {key: value for key, value in pinned_host.pin.items() if key != "EXECUTABLE"}
    write_pin(pins_module.PINS_DIR, pin)
    with pytest.raises(RunError, match="EXECUTABLE"):
        build_toolchain(NAME, pinned_host.root)
    assert pinned_host.log.calls == [] and pinned_host.log.constructed == []


def test_build_toolchain_refuses_gcc_native_without_a_toolchains_root(pinned_host: PinnedHost) -> None:
    # The clean_environment fixture unsets LASSI_TOOLCHAINS, so a run would pass no root; the compile sandbox
    # exposes the toolchains root read-only for every compile, the host compiler's included.
    with pytest.raises(RunError, match="LASSI_TOOLCHAINS"):
        build_toolchain(NAME, None)
    assert pinned_host.log.calls == [] and pinned_host.log.constructed == []


def test_build_toolchain_refuses_a_relative_toolchains_root_for_gcc_native(pinned_host: PinnedHost) -> None:
    with pytest.raises(RunError, match="absolute"):
        build_toolchain(NAME, Path("relative-toolchains-root"))
    assert pinned_host.log.calls == [] and pinned_host.log.constructed == []
