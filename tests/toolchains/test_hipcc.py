"""Tests for the HIP toolchain `hipcc-gfx942`, its pin toolchains/hipcc.pin, and its captured fixtures (task P17.6).

Bible: Toolchain Pins (a host compiler pinned by version and path; the
--version check against EXPECT_VERSION in the compile sandbox before the
first build), Execution Backends (none row: compile only; gpu (AMD) row),
Component Interfaces (Toolchain row and contract rules), Sandbox (the
compile environment's fixed allowlist), Host Facts (ROCm), Result Record
(Trial.toolchain_pins), Agent Rules 1 and 10; OQ-002 (nothing opens an AMD
device), OQ-039 (a) and OQ-043 entry 15 (the owner allowed AMD work that
opens no GPU device).

The contract these tests fix (plans/p17-portable.md, P17.6, part 1; the bible's
Toolchain Pins, the HIP build):

- Importing lassi.toolchains registers a Toolchain "hipcc-gfx942", the class
  HipccGfx942 in lassi.toolchains.hipcc, which a recipe binds under the
  language hip. It declares exactly hip, diagnostics, and emits_warnings.
  Its sources are the built files ending in .hip, in sorted order; headers,
  harness files, and files with other suffixes are written beside them and
  never compiled.
- One hipcc call compiles the host code and the gfx942 device code and links
  the host program, and nothing runs it (compile-only, the executor none):
  `<EXECUTABLE> --offload-arch=gfx942 -Wall -O3 -o main <sources>`. The
  module holds OFFLOAD_ARCH ("gfx942") and HIPCC_FLAGS; the target is always
  named, in every build and in the --version check (the class's
  VERSION_ARGS, `--offload-arch=gfx942 --version`), since hipcc given none
  runs rocm_agent_enumerator to choose one (read from hipcc's source), and
  no command names a device-query program (OQ-002). The class has no ARCH
  attribute and its constructor takes no target, so nothing can build
  another target. Every other pinned toolchain's check stays
  `<executable> --version`.
- toolchains/hipcc.pin pins the build host's hipcc as toolchains/gcc.pin
  pins g++: exactly NAME, VERSION, EXECUTABLE, FLAGS, and EXPECT_VERSION, no
  PREFIX_NAME and no install script, and no key that names a variable, so
  PREFIX_VARIABLES and the sandbox's ENVIRONMENT_NAMES gain no HIP or ROCm
  name and a compile gets exactly PATH, LANG=C, and LC_ALL=C (plus its
  private TMPDIR). No value of the pin expands anything or names the ROCm
  environment script.
- lassi.core.runner.build_toolchain("hipcc-gfx942", root) builds it with the
  pin's EXECUTABLE and runs `<EXECUTABLE> --offload-arch=gfx942 --version`
  once through the compile runner before any build; output without EXPECT_VERSION, a failed
  check, a missing or relative EXECUTABLE, a pin without EXECUTABLE or
  EXPECT_VERSION, and a missing or relative toolchains root are RunErrors,
  all but the first two before any command runs.
- A trial built by it records the pin as Trial.toolchain_pins.hipcc: the
  runner names the field after the pin file, and the field is appended
  after gcc.
- parse_diagnostics reads clang's stderr with the patterns ttmetal-host
  uses, moved unchanged into lassi.toolchains._stderr as CLANG_DRIVER and
  CLANG_PLACE, hipcc's own DEVICE_LLD for the device link's `lld:` line,
  and the shared GNU ld UNDEFINED_REFERENCE pattern. Part 1 pinned the rules
  on SYNTHETIC lines hand-written in clang's and GNU ld's formats; part 2
  checks them against the captured fixtures.
- tests/toolchains/fixtures/hipcc holds the capture scenarios, CASES, each a
  small main.hip written for the fixtures; hip_clean is also the program the
  remote test keeps (test_hipcc_remote.py).
- Part 2: the same directory holds each case's stderr, captured on alpha01
  from the clean part 1 commit f03fa62 (rx run
  20261006-121638-desktop-8r113ei-detached-f03fa62a-961a) and copied byte
  for byte as <case>.stderr; captures.json,
  the capture's manifest.json copied byte for byte; <case>.json, the
  Diagnostic list derived by hand from the stderr (never from the parser's
  output), with a `derivation` field; and README.md. The tests check the
  parse against each list and the design's eight points: the clean case's
  empty stderr; a warning printed by both passes and an error printed once;
  the summary lines; the host and device target errors with their notes;
  the host link's ld.lld lines and the per-run text in lld's context lines;
  the device link's `lld:` prefix; hipcc's own "failed to execute:" line;
  and plain ASCII with LF.

No compiler runs here. The build tests give the toolchain a fake command
runner; the build_toolchain tests replace lassi.core.runner's
SandboxedCompileRunner with a fake whose --version answer is a PLACEHOLDER
line before the pin's EXPECT_VERSION, and point the pin reader at a
temporary copy of hipcc.pin whose EXECUTABLE is an empty stand-in file. The
real check and a real build run on the build host in test_hipcc_remote.py.
No value in this module is a measurement: the version and the banner line
are what P17.1 read on the build host (plans/spikes/p17-frameworks.md, Host
state and Results 5), and the fixtures record compiler output, not
performance.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import pytest
import yaml

import lassi.core.runner  # noqa: F401  (importing the runner registers every component a recipe binds)
import lassi.toolchains  # noqa: F401  (importing the package registers every toolchain preset)
from lassi.core import record as record_module
from lassi.core import runner as runner_module
from lassi.core.interfaces import BuildResult
from lassi.core.recipe import load_recipe
from lassi.core.record import Diagnostic
from lassi.core.registry import DEFAULT_REGISTRY, RegistryError
from lassi.core.runner import RunError, build_toolchain
from lassi.executors import sandbox as sandbox_module
from lassi.toolchains import CommandResult, ttmetal_build
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
BIBLE = REPO / "docs" / "BIBLE.md"
TOOLCHAINS_DIR = REPO / "toolchains"
HIPCC_PIN = TOOLCHAINS_DIR / "hipcc.pin"
SMOKE = REPO / "tests" / "fixtures" / "recipes" / "p0-smoke.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hipcc"
SCENARIOS = FIXTURES / "scenarios.json"
SOURCES = FIXTURES / "sources"
TOOLS_DIR = REPO / "tools"

MODULE = "lassi.toolchains.hipcc"
CLASS_NAME = "HipccGfx942"
NAME = "hipcc-gfx942"
PIN_NAME = "hipcc"
LANGUAGE = "hip"
TARGET = "gfx942"
FLAGS = ["--offload-arch=gfx942", "-Wall", "-O3"]
# The --version check's arguments after the executable: the target first, since hipcc given no target runs
# rocm_agent_enumerator before it prints its version (ROCm/llvm-project tag therock-7.12,
# amd/hipcc/src/hipBin_amd.h:475, :568-581, :779-792; read from source).
VERSION_ARGS = ["--offload-arch=gfx942", "--version"]
CAPABILITIES = frozenset({"hip", "diagnostics", "emits_warnings"})
# The build host's hipcc, from P17.1 (plans/spikes/p17-frameworks.md, Host state: rx 20261005-194413-exec-00bf, and
# Results 5: the first line `hipcc --version` printed in the sandbox in rx
# 20261005-204140-desktop-8r113ei-detached-e805c6d4-54d2). Host facts, not measurements of anything here.
PINNED_VERSION = "7.12.60610-2bd1678d3d"
PINNED_EXECUTABLE = "/opt/rocm/core-7.12/bin/hipcc"
PINNED_EXPECT_VERSION = "HIP version: 7.12.60610-2bd1678d3d"
PIN_KEYS = frozenset({"NAME", "VERSION", "EXECUTABLE", "FLAGS", "EXPECT_VERSION"})
SCRATCH_ROOT = "/mnt/nvme10/"
OUTPUT = "main"
ATTACHMENT = "compile.stderr"
# A stand-in executable path for the tests that never run a compiler.
STAND_IN = "/opt/rocm/core-7.12/bin/hipcc"
# The --version text the fake compile runner prints: a PLACEHOLDER line, then the pin's EXPECT_VERSION.
BANNER_LEAD = "PLACEHOLDER banner of a fake compile runner, not a compiler"
# Programs that ask the host which GPUs it has (OQ-002): no command may start one (P17.1 found the first three in
# the ROCm tree, rx 20261005-194413-exec-00bf).
DEVICE_QUERY_PROGRAMS = frozenset(
    {"amdgpu-arch", "offload-arch", "rocm_agent_enumerator", "rocminfo", "rocm-smi", "amd-smi"}
)
CASES = (
    "hip_clean",
    "hip_device_undefined",
    "hip_host_calls_device",
    "hip_kernel_calls_host",
    "hip_linker_error",
    "hip_missing_header",
    "hip_undeclared_identifier",
    "hip_unused_variable",
)
# Word parts that would name a HIP or ROCm compile variable in the sandbox's allowlist.
ROCM_NAME_PARTS = ("HIP", "ROCM", "ROCR", "HSA", "AMD")


def hipcc_module() -> ModuleType:
    """Return lassi.toolchains.hipcc; fail the test clearly while it is missing."""
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as error:
        pytest.fail(f"{MODULE} does not exist ({error}); task P17.6 adds it")


def hipcc_class() -> type:
    """Return the Toolchain class registered as hipcc-gfx942; fail the test clearly while it is missing."""
    try:
        return DEFAULT_REGISTRY.get("Toolchain", NAME).factory
    except RegistryError as error:
        pytest.fail(f"no Toolchain is registered as {NAME!r} ({error}); task P17.6 adds it to lassi.toolchains")


def hipcc_pin() -> dict[str, str]:
    """Return toolchains/hipcc.pin as the runner reads it; fail the test clearly while it is missing."""
    if not HIPCC_PIN.is_file():
        pytest.fail(f"{HIPCC_PIN} does not exist; task P17.6 pins the build host's hipcc there")
    return pins_module.read_pin(PIN_NAME)


def refuse_to_run(argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
    """A CommandRunner for tests that must not run anything: a call fails the test."""
    raise AssertionError(f"a test that runs nothing ran {list(argv)!r}")


def diag(
    severity: str, code: str | None, file: str | None, line: int | None, column: int | None, message: str
) -> Diagnostic:
    """Return a compile-stage Diagnostic with every other field given explicitly."""
    return Diagnostic(
        stage="compile", severity=severity, code=code, file=file, line=line, column=column, message=message
    )


# ---------------------------------------------------------------------------
# Registration, capabilities, and the module


def test_the_package_registers_hipcc_gfx942() -> None:
    assert NAME in DEFAULT_REGISTRY.names("Toolchain"), f"importing lassi.toolchains registers {NAME!r}"
    assert hipcc_class().name == NAME


def test_it_declares_hip_diagnostics_and_warnings_and_nothing_else() -> None:
    hipcc_class()
    capabilities = DEFAULT_REGISTRY.get("Toolchain", NAME).capabilities
    assert capabilities == CAPABILITIES, "hip, as nvcc-sm80 declares cuda; diagnostics for compile_loop; no guard"


def test_it_is_pinned_as_a_host_compiler_by_the_hipcc_pin_file() -> None:
    cls = hipcc_class()
    assert getattr(cls, "PIN", None) == PIN_NAME, "the class names its pin file toolchains/hipcc.pin"
    assert getattr(cls, "PIN_BIN", None) is None, "the compiler is the build host's hipcc, not a file under the root"
    assert getattr(cls, "check_tree", None) is None, "it builds against no installed tree"


def test_the_module_names_the_target_the_flags_and_the_class() -> None:
    module = hipcc_module()
    assert module.OFFLOAD_ARCH == TARGET
    assert list(module.HIPCC_FLAGS) == FLAGS, "the target first, then -Wall -O3"
    assert getattr(module, CLASS_NAME) is hipcc_class()
    assert callable(module.parse_diagnostics)


def test_the_package_exports_the_class_and_the_module() -> None:
    assert getattr(lassi.toolchains, CLASS_NAME, None) is hipcc_class()
    assert CLASS_NAME in lassi.toolchains.__all__ and "hipcc" in lassi.toolchains.__all__
    assert lassi.toolchains.hipcc is hipcc_module()


def test_nothing_can_build_another_target() -> None:
    cls = hipcc_class()
    assert not hasattr(cls, "ARCH"), "no ARCH attribute, so no subclass or capture override changes the target"
    parameters = set(inspect.signature(cls).parameters)
    assert parameters <= {"executable", "runner", "timeout_s"}, (
        f"the target is never a constructor setting: {parameters}"
    )


def test_the_version_check_names_the_target_from_the_module_constant() -> None:
    module, cls = hipcc_module(), hipcc_class()
    assert list(cls.VERSION_ARGS) == VERSION_ARGS, "the target first, then --version"
    assert cls.VERSION_ARGS[0] == f"--offload-arch={module.OFFLOAD_ARCH}", "built from OFFLOAD_ARCH, not a copy"
    assert runner_module._version_argv(cls, STAND_IN) == [STAND_IN, *VERSION_ARGS], "the runner runs that argv"


def test_every_other_pinned_toolchain_still_checks_plain_version() -> None:
    names = [name for name in DEFAULT_REGISTRY.names("Toolchain") if name != NAME]
    pinned = [name for name in names if getattr(DEFAULT_REGISTRY.get("Toolchain", name).factory, "PIN", None)]
    expected = {"gcc-native", "nvcc-sm80", "nvcpp-cc80", "nvcpp-multicore", "ttmetal-host"}
    assert expected <= set(pinned), f"the registry's pinned toolchains: {pinned}"
    for name in pinned:
        factory = DEFAULT_REGISTRY.get("Toolchain", name).factory
        assert runner_module._version_argv(factory, STAND_IN) == [STAND_IN, "--version"], name


def test_the_sources_are_hip_files_only() -> None:
    assert hipcc_class().SOURCE_SUFFIXES == (".hip",), "whether this hipcc builds .cu or .cpp as HIP is not measured"


def test_a_recipe_binds_it_under_the_language_hip(tmp_path: Path) -> None:
    hipcc_class()
    data = yaml.safe_load(SMOKE.read_bytes().decode("ascii"))
    data["toolchain"] = {**data["toolchain"], LANGUAGE: NAME}
    path = tmp_path / "hip-binding.yaml"
    path.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    recipe = load_recipe(path)
    bound = [(item.name, item.where) for item in recipe.bindings if item.interface == "Toolchain"]
    assert (NAME, f"toolchain.{LANGUAGE}") in bound


# ---------------------------------------------------------------------------
# The pin file


def test_the_pin_records_exactly_name_version_executable_flags_and_expected_version() -> None:
    pin = hipcc_pin()
    assert set(pin) == PIN_KEYS, f"the pin's keys are exactly {sorted(PIN_KEYS)}, got {sorted(pin)}"
    assert pin["NAME"] == PIN_NAME
    assert pin["VERSION"] == PINNED_VERSION, "the HIP version with its build hash, as hipcc --version printed it"
    assert pin["EXECUTABLE"] == PINNED_EXECUTABLE
    assert pin["FLAGS"].split() == FLAGS
    assert pin["EXPECT_VERSION"] == PINNED_EXPECT_VERSION, "the whole first banner line"
    assert pin["VERSION"] in pin["EXPECT_VERSION"]


def test_the_pin_names_a_host_compiler_that_the_project_does_not_install() -> None:
    pin = hipcc_pin()
    assert "PREFIX_NAME" not in pin, "the build host's hipcc has no install prefix under $LASSI_TOOLCHAINS"
    assert not (TOOLCHAINS_DIR / "hipcc.sh").exists(), "hipcc is pinned by version and path, not installed"
    executable = PurePosixPath(pin["EXECUTABLE"])
    assert executable.is_absolute() and executable.name == "hipcc"
    assert not pin["EXECUTABLE"].startswith(SCRATCH_ROOT), "EXECUTABLE is the host's ROCm, not a scratch install"


def test_the_pin_sets_no_variable_and_expands_nothing() -> None:
    pin = hipcc_pin()
    assert pins_module.linked_prefixes(pin) == {}, "the build needs no variable (P17.1, Results 5)"
    assert not set(pin) & set(pins_module.PREFIX_VARIABLES), "no key that sets a compile variable"
    for key, value in pin.items():
        assert "$" not in value and "`" not in value, f"{key} is a literal value, never a shell expansion"
        assert "rocm_env" not in value, f"{key}: no build sources the ROCm environment script (Host Facts)"


def test_the_pin_file_is_plain_ascii_with_lf_and_comments() -> None:
    hipcc_pin()
    raw = HIPCC_PIN.read_bytes()
    assert raw.isascii() and b"\r" not in raw and raw.endswith(b"\n")
    lines = raw.decode("ascii").splitlines()
    assert any(line.startswith("#") for line in lines), "the pin file says where its values come from"


def test_the_pinned_flags_are_the_modules_flags() -> None:
    assert hipcc_pin()["FLAGS"].split() == list(hipcc_module().HIPCC_FLAGS)


def test_no_hip_or_rocm_variable_joins_the_compile_allowlist() -> None:
    # A new name in the compile sandbox is a Sandbox edit with a Decision Log entry; P17.6 needs none. The GPU
    # visibility names are on the allowlist for the gpu executor (task P17.5) and a compile refuses them.
    names = set(sandbox_module.ENVIRONMENT_NAMES) - set(sandbox_module.GPU_VISIBILITY_NAMES)
    found = sorted(name for name in names if any(part in name for part in ROCM_NAME_PARTS))
    assert not found, f"a HIP or ROCm name reaches a compile: {found}"
    linked = sorted(set(pins_module.PREFIX_VARIABLES.values()))
    assert not [name for name in linked if any(part in name for part in ROCM_NAME_PARTS)], linked


# ---------------------------------------------------------------------------
# The command line and build()


@dataclass
class FakeRunner:
    """A CommandRunner that records each call and answers with `status` and `stderr`; it runs nothing.

    On status 0 it writes a PLACEHOLDER artifact `main` in the workdir, as
    hipcc would.
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


def make_toolchain(runner: Any, executable: str = STAND_IN) -> Any:
    """Return a hipcc-gfx942 toolchain built as the runner builds a pinned one: executable and runner given."""
    return hipcc_class()(executable=executable, runner=runner)


def test_the_command_is_the_executable_the_flags_the_output_then_the_sources() -> None:
    command = make_toolchain(refuse_to_run).command(["a.hip", "main.hip"])
    assert command == [STAND_IN, *FLAGS, "-o", OUTPUT, "a.hip", "main.hip"]


def test_the_command_flags_are_the_pins_flags() -> None:
    command = make_toolchain(refuse_to_run).command(["main.hip"])
    assert command[1:-3] == hipcc_pin()["FLAGS"].split()


@pytest.mark.parametrize("sources", [["main.hip"], ["a.hip", "kernels.hip", "main.hip"]])
def test_every_command_names_gfx942_once_and_no_device_query_program(sources: list[str]) -> None:
    command = make_toolchain(refuse_to_run).command(sources)
    targets = [word for word in command if "offload-arch" in word or "amdgpu-target" in word]
    assert targets == [f"--offload-arch={TARGET}"], "the target is named once, explicitly, never chosen by a probe"
    assert "native" not in command and not any(word.endswith("=native") for word in command)
    named = [word for word in command if PurePosixPath(word).name in DEVICE_QUERY_PROGRAMS]
    assert not named, f"a command never starts a device-query program (OQ-002): {named}"
    assert not any("rocm_env" in word for word in command)


def test_build_compiles_the_hip_sources_in_sorted_order_and_writes_the_rest_beside_them(tmp_path: Path) -> None:
    runner = FakeRunner()
    workdir = tmp_path / "build"
    workdir.mkdir()
    files = {
        "main.hip": '#include "kernels/scale.h"\n#include "lassi_io.h"\nint main() { return 0; }\n',
        "util.hip": "int util() { return 1; }\n",
        "kernels/scale.h": "inline int scale(int x) { return 2 * x; }\n",
        "other.cu": "int other() { return 2; }\n",
        "helper.cpp": "int helper() { return 3; }\n",
    }
    harness = {"lassi_io.h": "/* SYNTHETIC stand-in for the harness header */\n"}
    result = make_toolchain(runner).build(files, workdir, harness=harness)
    assert isinstance(result, BuildResult)
    ((argv, cwd),) = runner.calls
    assert cwd == workdir
    assert argv == [STAND_IN, *FLAGS, "-o", OUTPUT, "main.hip", "util.hip"], "only .hip files are sources"
    assert result.artifact == workdir / OUTPUT
    assert result.diagnostics == []
    for path in ("kernels/scale.h", "other.cu", "helper.cpp"):
        assert (workdir / path).read_text(encoding="utf-8") == files[path], f"{path} is written beside the sources"
    assert (workdir / "lassi_io.h").read_text(encoding="utf-8") == harness["lassi_io.h"]
    assert (workdir / ATTACHMENT).read_bytes() == b"", "the raw stderr attachment is kept, empty for a clean build"


def test_a_build_without_a_hip_source_runs_nothing_and_says_so(tmp_path: Path) -> None:
    workdir = tmp_path / "build"
    workdir.mkdir()
    result = make_toolchain(refuse_to_run).build({"main.cu": "int main() { return 0; }\n"}, workdir)
    assert result.artifact is None
    (error,) = result.diagnostics
    assert (error.severity, error.code) == ("error", "no-sources")
    assert ".hip" in error.message


def test_a_failed_build_keeps_the_raw_stderr_and_parses_its_error(tmp_path: Path) -> None:
    # SYNTHETIC stderr in clang's format (not captured): line 7 of the case's main.hip is
    # "        y[i] = y[i] * undefined_var;", and clang's column 23 is the 'u', counted in main.hip itself.
    stderr = (
        "main.hip:7:23: error: use of undeclared identifier 'undefined_var'\n"
        "    7 |         y[i] = y[i] * undefined_var;\n"
        "      |                       ^\n"
        "1 error generated when compiling for gfx942.\n"
    )
    runner = FakeRunner(status=1, stderr=stderr)
    workdir = tmp_path / "build"
    workdir.mkdir()
    source = (SOURCES / "hip_undeclared_identifier" / "main.hip").read_bytes().decode("ascii")
    assert source.split("\n")[6] == "        y[i] = y[i] * undefined_var;"
    result = make_toolchain(runner).build({"main.hip": source}, workdir)
    assert result.artifact is None
    message = "use of undeclared identifier 'undefined_var'"
    assert result.diagnostics == [diag("error", None, "main.hip", 7, 23, message)]
    assert (workdir / ATTACHMENT).read_bytes() == stderr.encode("ascii")


# ---------------------------------------------------------------------------
# build_toolchain: the pinned host hipcc, checked by --version in the compile runner before any build


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
        """Answers the --version check with the log's banner and status; any other command fails the test."""

        def __init__(self, **keywords: Any) -> None:
            log.constructed.append(dict(keywords))

        def spec(self, build: Path) -> None:
            """Accept any build dir layout; the real sandbox checks it on the build host."""

        def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
            log.calls.append(list(argv))
            assert [word for word in argv if "offload-arch" in word] == [f"--offload-arch={TARGET}"], (
                f"the --version check names the target exactly once, so hipcc never looks for a GPU: {argv!r}"
            )
            assert list(argv[1:]) == VERSION_ARGS, f"only the --version check may run here, not {argv!r}"
            return CommandResult(log.status, log.banner, "")

    return FakeCompileRunner


@dataclass(frozen=True)
class PinnedHost:
    """A temporary pin directory holding hipcc.pin, its stand-in executable, a toolchains root, and the runner log."""

    pin: dict[str, str]
    executable: Path
    root: Path
    log: CompileRunnerLog


def write_pin(directory: Path, pairs: Mapping[str, str]) -> None:
    """Write `pairs` as a hipcc.pin in `directory`, one quoted KEY="value" line each."""
    text = "# SYNTHETIC copy of toolchains/hipcc.pin for a test\n" + "".join(
        f'{key}="{value}"\n' for key, value in pairs.items()
    )
    (directory / "hipcc.pin").write_bytes(text.encode("ascii"))


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables, give HOME and TMPDIR test directories, so no test reads the real host layout."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "HIP_PATH", "ROCM_PATH", "HIP_PLATFORM"):
        monkeypatch.delenv(name, raising=False)
    for name in ("home", "compile-tmp"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "compile-tmp"))


@pytest.fixture
def pinned_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PinnedHost:
    """Point the pin reader at a copy of hipcc.pin whose EXECUTABLE is an empty stand-in; fake the compile runner."""
    real = hipcc_pin()
    hipcc_class()
    executable = tmp_path / "rocm-bin" / PurePosixPath(real["EXECUTABLE"]).name
    executable.parent.mkdir()
    executable.write_bytes(b"")
    pin = {**real, "EXECUTABLE": str(executable)}
    pins_dir = tmp_path / "pins"
    pins_dir.mkdir()
    write_pin(pins_dir, pin)
    monkeypatch.setattr(pins_module, "PINS_DIR", pins_dir)
    log = CompileRunnerLog(banner=f"{BANNER_LEAD}\n{real['EXPECT_VERSION']}\nPLACEHOLDER clang line\n")
    monkeypatch.setattr(runner_module, "SandboxedCompileRunner", fake_compile_runner(log))
    root = tmp_path / "toolchains-root"
    root.mkdir()
    return PinnedHost(pins_module.read_pin(PIN_NAME), executable, root, log)


def test_build_toolchain_checks_the_pinned_hipcc_version_once_before_any_build(pinned_host: PinnedHost) -> None:
    built = build_toolchain(NAME, pinned_host.root)
    assert pinned_host.log.calls == [[str(pinned_host.executable), *VERSION_ARGS]], (
        "the --version check runs once, with the target named, through the compile runner, before the first build"
    )
    assert built.executable == str(pinned_host.executable), "the executable is the pin's EXECUTABLE, as given"
    assert built.toolchain.executable == str(pinned_host.executable)
    assert built.pins == {PIN_NAME: pinned_host.pin}
    assert built.version_status == 0
    assert pinned_host.pin["EXPECT_VERSION"] in "\n".join(built.version)
    assert len(pinned_host.log.constructed) == 1, "one compile runner serves the check and every build"
    assert type(built.toolchain.runner).__name__ == "FakeCompileRunner", "builds use the compile runner too"


def test_build_toolchain_gives_hipcc_only_the_clean_c_locale_environment(pinned_host: PinnedHost) -> None:
    built = build_toolchain(NAME, pinned_host.root)
    assert built.environment is not None
    assert set(built.environment) == {"PATH", "LANG", "LC_ALL"}, "no HIP_PATH, ROCM_PATH, or other ROCm variable"
    assert (built.environment["LANG"], built.environment["LC_ALL"]) == ("C", "C")
    (keywords,) = pinned_host.log.constructed
    assert keywords["environment"] == built.environment, "the compile sandbox gets that environment and no other"


def test_build_toolchain_refuses_a_hipcc_that_is_not_the_pinned_version(pinned_host: PinnedHost) -> None:
    pinned_host.log.banner = f"{BANNER_LEAD}\nHIP version: PLACEHOLDER-other-version\n"
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
    executable = str(tmp_path / "no-such-dir" / "hipcc") if problem == "missing" else "hipcc"
    write_pin(pins_module.PINS_DIR, {**pinned_host.pin, "EXECUTABLE": executable})
    with pytest.raises(RunError) as caught:
        build_toolchain(NAME, pinned_host.root)
    assert "hipcc" in str(caught.value), "the refusal names the executable"
    assert pinned_host.log.calls == [], "nothing runs before the executable is known to be the pinned one"


@pytest.mark.parametrize("key", ["EXPECT_VERSION", "EXECUTABLE"])
def test_build_toolchain_refuses_a_pin_without_a_key_before_anything_runs(pinned_host: PinnedHost, key: str) -> None:
    write_pin(pins_module.PINS_DIR, {name: value for name, value in pinned_host.pin.items() if name != key})
    with pytest.raises(RunError, match=key):
        build_toolchain(NAME, pinned_host.root)
    assert pinned_host.log.calls == []


def test_build_toolchain_refuses_hipcc_without_an_absolute_toolchains_root(pinned_host: PinnedHost) -> None:
    # The clean_environment fixture unsets LASSI_TOOLCHAINS; the compile sandbox exposes the root for every compile.
    with pytest.raises(RunError, match="LASSI_TOOLCHAINS"):
        build_toolchain(NAME, None)
    with pytest.raises(RunError, match="absolute"):
        build_toolchain(NAME, Path("relative-toolchains-root"))
    assert pinned_host.log.calls == [] and pinned_host.log.constructed == []


# ---------------------------------------------------------------------------
# The Result Record


def test_a_trial_records_the_hipcc_pin_in_its_own_field() -> None:
    names = record_module.TOOLCHAIN_PIN_NAMES
    assert PIN_NAME in names, "Trial.toolchain_pins gains the field hipcc (Result Record)"
    assert names.index(PIN_NAME) == names.index("gcc") + 1, "hipcc is appended after gcc"
    assert "rocm" in names, "rocm stays, unused, for a ROCm the project would pin as a whole"
    pins = record_module.ToolchainPins(hipcc=PINNED_VERSION)
    assert pins.hipcc == PINNED_VERSION and pins.rocm is None


def test_the_runner_records_the_hipcc_pin_under_the_pin_files_name() -> None:
    versions = runner_module._pin_versions([{PIN_NAME: hipcc_pin()}])
    assert versions == record_module.ToolchainPins(hipcc=PINNED_VERSION)


# ---------------------------------------------------------------------------
# The parser, on SYNTHETIC lines in clang's and GNU ld's formats (none captured)

# A built main.hip of 12 lines, so a place on lines 1 to 12 is a line of a built file.
BUILT = {"main.hip": "".join(f"// SYNTHETIC line {number}\n" for number in range(1, 13))}


def parse(stderr: str, files: Mapping[str, str] | None = None) -> list[Diagnostic]:
    """Return what the hipcc-gfx942 toolchain parses from `stderr` with the built `files` (none by default)."""
    return make_toolchain(refuse_to_run).parse(stderr, dict(files or {}))


def test_clang_place_lines_parse_with_the_column_only_on_a_built_file() -> None:
    stderr = (
        "main.hip:6:23: error: use of undeclared identifier 'undefined_var'\n"
        "main.hip:5:9: warning: unused variable 'unused' [-Wunused-variable]\n"
        "main.hip:3:10: fatal error: 'p176_no_such_header.h' file not found\n"
        "./main.hip:8:5: note: 'device_scale' declared here\n"
        "/opt/rocm/core-7.12/include/hip/amd_detail/SYNTHETIC.h:120:3: note: SYNTHETIC note in a ROCm header\n"
    )
    assert parse(stderr, BUILT) == [
        diag("error", None, "main.hip", 6, 23, "use of undeclared identifier 'undefined_var'"),
        diag("warning", "-Wunused-variable", "main.hip", 5, 9, "unused variable 'unused'"),
        diag("error", None, "main.hip", 3, 10, "'p176_no_such_header.h' file not found"),
        diag("note", None, "main.hip", 8, 5, "'device_scale' declared here"),
        diag(
            "note",
            None,
            "/opt/rocm/core-7.12/include/hip/amd_detail/SYNTHETIC.h",
            120,
            None,
            "SYNTHETIC note in a ROCm header",
        ),
    ]
    assert parse("main.hip:6:23: error: x\n") == [diag("error", None, "main.hip", 6, None, "x")], "no files, no column"


@pytest.mark.parametrize(
    ("line", "severity", "message"),
    [
        (
            "clang++: error: linker command failed with exit code 1 (use -v to see invocation)",
            "error",
            "linker command failed with exit code 1 (use -v to see invocation)",
        ),
        ("clang: error: SYNTHETIC driver error", "error", "SYNTHETIC driver error"),
        ("clang-22: fatal error: SYNTHETIC fatal driver error", "error", "SYNTHETIC fatal driver error"),
        ("ld.lld: error: undefined symbol: helper(int)", "error", "undefined symbol: helper(int)"),
    ],
)
def test_driver_and_lld_lines_parse_with_no_place(line: str, severity: str, message: str) -> None:
    assert parse(line + "\n", BUILT) == [diag(severity, None, None, None, None, message)]


def test_gnu_ld_undefined_references_parse_as_errors() -> None:
    stderr = (
        "/usr/bin/ld: /SYNTHETIC/tmp/main-0123ab.o: in function `main':\n"
        "main.hip:(.text+0x1c): undefined reference to `helper(int)'\n"
        "/usr/bin/ld: /SYNTHETIC/attempt00/build/main.hip:7: undefined reference to `helper(int)'\n"
    )
    assert parse(stderr, BUILT) == [
        diag("error", None, None, None, None, "undefined reference to `helper(int)'"),
        diag("error", None, "main.hip", 7, None, "undefined reference to `helper(int)'"),
    ]


def test_summaries_include_chains_echoes_and_carets_parse_to_nothing() -> None:
    stderr = (
        "In file included from main.hip:2:\n"
        "1 warning generated when compiling for gfx942.\n"
        "1 warning generated when compiling for host.\n"
        "2 errors generated when compiling for host.\n"
        "        y[i] = y[i] * undefined_var;\n"
        "                      ^\n"
        "    6 |         y[i] = y[i] * undefined_var;\n"
        "      |                       ^~~~~~~~~~~~~\n"
        "\n"
    )
    assert parse(stderr, BUILT) == []


def test_a_diagnostic_printed_by_both_passes_parses_twice() -> None:
    # The parse is faithful to stderr: a warning the device pass and the host pass both print is two Diagnostics.
    line = "main.hip:5:9: warning: unused variable 'unused' [-Wunused-variable]\n"
    summary = "1 warning generated when compiling for {}.\n"
    stderr = line + summary.format("gfx942") + line + summary.format("host")
    once = diag("warning", "-Wunused-variable", "main.hip", 5, 9, "unused variable 'unused'")
    assert parse(stderr, BUILT) == [once, once]


def test_parse_diagnostics_is_what_the_toolchain_parses() -> None:
    stderr = "main.hip:6:23: error: use of undeclared identifier 'undefined_var'\nclang++: error: SYNTHETIC\n"
    assert hipcc_module().parse_diagnostics(stderr, BUILT) == parse(stderr, BUILT)
    assert hipcc_module().parse_diagnostics(stderr) == parse(stderr), "files default to none"


def test_the_clang_patterns_are_shared_with_ttmetal_host_unchanged() -> None:
    from lassi.toolchains import _stderr

    driver, place = getattr(_stderr, "CLANG_DRIVER", None), getattr(_stderr, "CLANG_PLACE", None)
    assert isinstance(driver, _stderr.LinePattern) and isinstance(place, _stderr.LinePattern), (
        "ttmetal-host's clang patterns move into lassi.toolchains._stderr as CLANG_DRIVER and CLANG_PLACE"
    )
    assert ttmetal_build._PATTERNS == (driver, place), "ttmetal-host keeps exactly those patterns, in that order"
    stderr = (
        "main.hip:6:23: error: use of undeclared identifier 'undefined_var'\n"
        "./main.hip:8:5: note: 'device_scale' declared here\n"
        "/opt/rocm/core-7.12/include/SYNTHETIC.h:2:1: warning: SYNTHETIC [-Wsynthetic]\n"
        "clang++-20: error: linker command failed with exit code 1 (use -v to see invocation)\n"
        "ld.lld: error: undefined symbol: helper(int)\n"
        "1 error generated.\n"
    )
    assert parse(stderr, BUILT) == ttmetal_build.parse_diagnostics(stderr, BUILT), "the same rules for clang lines"


# ---------------------------------------------------------------------------
# The capture scenarios (tests/toolchains/fixtures/hipcc; the captures are checked in the next section)


def scenarios() -> dict[str, Any]:
    """Return scenarios.json of the hipcc fixture set."""
    data = json.loads(SCENARIOS.read_bytes().decode("ascii"))
    assert isinstance(data, dict)
    return data


def test_the_scenarios_are_the_eight_cases_each_built_by_hipcc_gfx942() -> None:
    raw = SCENARIOS.read_bytes()
    assert raw.isascii() and b"\r" not in raw
    data = scenarios()
    assert tuple(sorted(data)) == CASES
    for name, entry in data.items():
        assert set(entry) == {"scenario", "toolchain"}, f"{name}: no override, so the target stays gfx942"
        assert entry["toolchain"] == NAME
        description = entry["scenario"]
        assert isinstance(description, str) and description.strip() and "\n" not in description, name


def test_each_case_is_one_small_main_hip_of_plain_ascii_with_lf() -> None:
    assert sorted(path.name for path in SOURCES.iterdir()) == list(CASES), "one source tree per scenario"
    for name in CASES:
        files = sorted(path.relative_to(SOURCES / name).as_posix() for path in (SOURCES / name).rglob("*"))
        assert files == ["main.hip"], f"{name}: {files}"
        raw = (SOURCES / name / "main.hip").read_bytes()
        assert raw.isascii() and b"\r" not in raw and raw.endswith(b"\n"), name
        text = raw.decode("ascii")
        assert text.startswith("// "), f"{name}: the first line says what the case is for"
        assert "#include <hip/hip_runtime.h>" in text, name
        assert not re.search(r"hip(Set|Get)Device\b|hipGetDeviceCount|hipGetDeviceProperties", text), (
            f"{name}: no source names or selects a device"
        )


def test_the_capture_tool_reads_the_set_with_no_override() -> None:
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    tool = importlib.import_module("capture_toolchain_fixtures")
    loaded = tool.load_scenarios(SCENARIOS, SOURCES)
    assert tuple(loaded) == CASES
    for name, scenario in loaded.items():
        assert (scenario.toolchain, dict(scenario.overrides)) == (NAME, {}), name
        assert list(scenario.files) == ["main.hip"], name


# ---------------------------------------------------------------------------
# The captured fixtures (part 2): tests/toolchains/fixtures/hipcc, from the clean capture on alpha01

CAPTURES = FIXTURES / "captures.json"
README = FIXTURES / "README.md"
DIAGNOSTIC_KEYS = frozenset({"stage", "severity", "code", "file", "line", "column", "message"})
# The capture's run, from the clean part 1 commit f03fa62.
CAPTURE_RX_ID = "20261006-121638-desktop-8r113ei-detached-f03fa62a-961a"
# The line hipcc itself prints after a failed clang call, before the clang++ command it ran (every failed case).
HIPCC_FAILED_PREFIX = "failed to execute:"
# The capture's workdir root on the build host: per-run text, which changes on a recapture.
CAPTURE_WORKDIR = f"/mnt/nvme10/joseph_ufl/lassi-runs/fixture-captures/{CAPTURE_RX_ID}/work/"
# The clang summary line each pass prints after its diagnostics, for example "1 warning generated when compiling for
# gfx942." (capture hip_unused_variable); group 1 is the pass.
SUMMARY = re.compile(r"[0-9]+ (?:warnings?|errors?)(?: and [0-9]+ errors?)? generated when compiling for (\S+)\.")
# The case the part 1 parser read short: its manifest count is that parser's, which read no "lld:" line.
PART_1_COUNTS = {"hip_device_undefined": 1}
# The failed compiles (not links): each stops after the gfx942 pass.
COMPILE_ERROR_CASES = (
    "hip_host_calls_device",
    "hip_kernel_calls_host",
    "hip_missing_header",
    "hip_undeclared_identifier",
)


def load_json(path: Path) -> Any:
    """Return a plain ASCII JSON file's value; fail the test clearly while the file is missing."""
    if not path.is_file():
        pytest.fail(f"{path} does not exist; task P17.6 part 2 copies the capture from the build host")
    raw = path.read_bytes()
    assert raw.isascii(), f"{path} is not plain ASCII"
    assert b"\r" not in raw, f"{path} has a CR"
    return json.loads(raw.decode("ascii"))


def stderr_bytes(case: str) -> bytes:
    """Return the captured stderr of `case` as stored; fail the test clearly while it is missing."""
    path = FIXTURES / f"{case}.stderr"
    if not path.is_file():
        pytest.fail(f"{path} does not exist; task P17.6 part 2 copies it byte for byte from the capture")
    return path.read_bytes()


def stderr_lines(case: str) -> list[str]:
    """Return the non-empty lines of the captured stderr of `case`, split on "\\n" only."""
    return [line for line in stderr_bytes(case).decode("ascii").split("\n") if line]


def source_files(case: str) -> dict[str, str]:
    """Return the files `case` compiled: relative POSIX path -> text, with no newline translation."""
    tree = SOURCES / case
    paths = sorted(path for path in tree.rglob("*") if path.is_file())
    return {path.relative_to(tree).as_posix(): path.read_bytes().decode("ascii") for path in paths}


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
    return parse(stderr_bytes(case).decode("ascii"), source_files(case))


def entry(case: str) -> dict[str, Any]:
    """Return the capture manifest's entry for `case`."""
    cases = load_json(CAPTURES)["scenarios"]
    assert case in cases, f"captures.json has no entry for {case}"
    return cases[case]


def summaries(case: str) -> list[str]:
    """Return the pass each clang summary line in the captured stderr of `case` names, in order."""
    return [match[1] for line in stderr_lines(case) if (match := SUMMARY.fullmatch(line))]


def line_of(case: str, text: str) -> int:
    """Return the 1-based number of the one line of main.hip in the source tree of `case` that holds `text`."""
    lines = source_files(case)["main.hip"].split("\n")
    numbers = [number for number, line in enumerate(lines, start=1) if text in line]
    assert len(numbers) == 1, f"{case}/main.hip: {text!r} must be on exactly one line"
    return numbers[0]


def points_at(case: str, diagnostic: Diagnostic, text: str) -> bool:
    """Return True when the diagnostic's column, in main.hip of the source tree, starts `text`."""
    if diagnostic.file != "main.hip" or diagnostic.line is None or diagnostic.column is None:
        return False
    line = source_files(case)["main.hip"].split("\n")[diagnostic.line - 1]
    return line[diagnostic.column - 1 :].startswith(text)


def test_every_captured_file_belongs_to_a_case() -> None:
    stems = sorted(path.stem for path in FIXTURES.glob("*.stderr"))
    assert stems == list(CASES), "one <case>.stderr per case, and no other"
    manifests = {SCENARIOS.name, CAPTURES.name}
    lists = sorted(path.stem for path in FIXTURES.glob("*.json") if path.name not in manifests)
    assert lists == list(CASES), "one <case>.json per case, and no other"


@pytest.mark.parametrize("case", CASES)
def test_the_fixture_is_raw_ascii_stderr_with_lf(case: str) -> None:
    # Check 8: the capture ran with LANG=C and LC_ALL=C, so clang's quotes are plain ASCII.
    raw = stderr_bytes(case)
    assert raw.isascii(), case
    assert b"\r" not in raw, f"{case} must be stored with LF line endings"
    if raw:
        assert raw.endswith(b"\n"), case
        assert not raw.startswith((b"#", b"//")), f"{case} starts with a comment; fixtures are raw stderr"


def test_the_capture_ran_from_a_clean_commit_on_the_build_host_with_the_pinned_hipcc() -> None:
    captures = load_json(CAPTURES)
    assert captures["dirty"] is False
    assert captures["snapshot_of"] is None
    assert re.fullmatch(r"[0-9a-f]{40}", captures["commit"]), captures["commit"]
    assert captures["rx_run_id"] == CAPTURE_RX_ID
    assert captures["host"] == "alpha01"
    assert sorted(captures["scenarios"]) == list(CASES)
    assert sorted(captures["toolchains"]) == [NAME]
    record = captures["toolchains"][NAME]
    pin = hipcc_pin()
    assert record["pins"] == {PIN_NAME: pin["VERSION"]}
    assert record["pin_files"] == {PIN_NAME: pin}, "the capture used the pin as it is now"
    assert record["executable"] == pin["EXECUTABLE"]
    assert record["version_exit_status"] == 0
    assert record["version"][0] == pin["EXPECT_VERSION"]
    assert record["locale"] == {"LANG": "C", "LC_ALL": "C"}
    assert record["environment"] == ["LANG", "LC_ALL", "PATH"], "no ROCm or loader variable reaches a compile"


@pytest.mark.parametrize("case", CASES)
def test_the_fixture_is_the_captured_stderr_byte_for_byte(case: str) -> None:
    raw = stderr_bytes(case)
    assert hashlib.sha256(raw).hexdigest() == entry(case)["stderr_sha256"], case
    assert len(raw) == entry(case)["stderr_bytes"], case


@pytest.mark.parametrize("case", CASES)
def test_the_capture_built_the_case_source_with_the_adapter_command(case: str) -> None:
    record = entry(case)
    assert record["toolchain"] == NAME and record["overrides"] == {}
    assert record["argv"] == make_toolchain(refuse_to_run, executable=hipcc_pin()["EXECUTABLE"]).command(["main.hip"])


@pytest.mark.parametrize("case", CASES)
def test_the_captured_stderr_parses_into_the_hand_derived_list(case: str) -> None:
    assert captured(case) == expected(case)


@pytest.mark.parametrize("case", CASES)
def test_the_manifest_counts_are_the_part_1_parsers(case: str) -> None:
    # The capture counted with the part 1 parser, which read no "lld:" line; every other count is the hand list's.
    lld_lines = [line for line in stderr_lines(case) if line.startswith("lld: ")]
    assert entry(case)["diagnostics"] == PART_1_COUNTS.get(case, len(expected(case)))
    assert entry(case)["diagnostics"] == len(expected(case)) - len(lld_lines), case


@pytest.mark.parametrize("case", CASES)
def test_without_the_built_files_only_the_columns_change(case: str) -> None:
    bare = parse(stderr_bytes(case).decode("ascii"))
    assert bare == [dataclasses.replace(item, column=None) for item in expected(case)]


@pytest.mark.parametrize("case", CASES)
def test_a_failed_capture_parses_into_an_error_and_a_clean_one_into_none(case: str) -> None:
    status = entry(case)["exit_status"]
    errors = [item for item in expected(case) if item.severity == "error"]
    assert bool(errors) == (status != 0), (case, status)


def test_the_clean_case_has_empty_stderr_and_exit_status_zero() -> None:
    # Check 1: no driver warning, no summary line, and no hipcc line on a clean build.
    assert stderr_bytes("hip_clean") == b""
    assert entry("hip_clean")["exit_status"] == 0
    assert expected("hip_clean") == []


def test_the_warning_case_exits_0_with_the_warning_printed_by_both_passes() -> None:
    # Checks 2 and 3: a warning in a kernel is printed by the gfx942 pass and again by the host pass, each followed
    # by its summary line; the parse keeps both, as stderr shows them, and the summaries give nothing.
    case = "hip_unused_variable"
    assert entry(case)["exit_status"] == 0, "without -Werror the warning does not fail the build"
    assert summaries(case) == [TARGET, "host"]
    found = captured(case)
    assert len(found) == 2 and found[0] == found[1], "the same warning, once per pass"
    item = found[0]
    assert (item.severity, item.code, item.file) == ("warning", "-Wunused-variable", "main.hip")
    assert item.line == line_of(case, "int unused = 0;")
    assert points_at(case, item, "unused")


@pytest.mark.parametrize("case", COMPILE_ERROR_CASES)
def test_a_compile_error_is_printed_once_by_the_gfx942_pass_and_no_host_pass_prints(case: str) -> None:
    # Check 2: the build stops after the gfx942 pass, so an error is printed once, even in code both passes compile.
    assert summaries(case) == [TARGET]
    errors = [item for item in captured(case) if item.severity == "error"]
    assert len(errors) == 1, case


def test_every_summary_line_parses_to_nothing() -> None:
    # Check 3: "N warning(s)/error(s) generated when compiling for gfx942." and "... for host." match no pattern.
    lines = [line for case in CASES for line in stderr_lines(case) if "generated when compiling for" in line]
    assert lines, "the captures hold summary lines"
    for line in lines:
        assert SUMMARY.fullmatch(line), line
        assert parse(line + "\n", BUILT) == [], line


def test_the_host_calls_device_error_and_its_note_point_at_the_call_and_the_declaration() -> None:
    # Check 4: a host function calling a __device__ function; clang's one summary names the gfx942 pass.
    case = "hip_host_calls_device"
    error, note = captured(case)
    assert (error.severity, error.file, error.code) == ("error", "main.hip", None)
    assert error.line == line_of(case, "float value = device_scale(1.0f);")
    assert points_at(case, error, "device_scale")
    assert (note.severity, note.file) == ("note", "main.hip")
    assert note.line == line_of(case, "__device__ float device_scale")
    assert points_at(case, note, "device_scale")
    assert "call to __device__ function from __host__ function" in note.message


def test_the_kernel_calls_host_error_and_its_note_point_at_the_call_and_the_declaration() -> None:
    # Check 4: a kernel calling a plain host function.
    case = "hip_kernel_calls_host"
    error, note = captured(case)
    assert (error.severity, error.file, error.code) == ("error", "main.hip", None)
    assert error.line == line_of(case, "y[i] = host_scale(y[i]);")
    assert points_at(case, error, "host_scale")
    assert (note.severity, note.file) == ("note", "main.hip")
    assert note.line == line_of(case, "float host_scale(float x)")
    assert points_at(case, note, "host_scale")
    assert "call to __host__ function from __global__ function" in note.message


def test_the_undeclared_identifier_and_the_missing_header_point_at_the_name() -> None:
    (undeclared,) = captured("hip_undeclared_identifier")
    assert undeclared.line == line_of("hip_undeclared_identifier", "undefined_var;")
    assert points_at("hip_undeclared_identifier", undeclared, "undefined_var")
    (missing,) = captured("hip_missing_header")
    assert missing.severity == "error", "a fatal error is an error"
    assert missing.line == line_of("hip_missing_header", '#include "p176_no_such_header.h"')
    assert points_at("hip_missing_header", missing, '"p176_no_such_header.h"')


def test_the_host_link_is_ld_lld_and_the_driver_names_the_linker_command() -> None:
    # Check 5: the host link's linker is ld.lld, never GNU ld; the driver line is clang++'s.
    case = "hip_linker_error"
    lines = stderr_lines(case)
    assert lines[0] == "ld.lld: error: undefined symbol: helper(int)"
    assert "clang++: error: linker command failed with exit code 1 (use -v to see invocation)" in lines
    assert not [line for line in lines if "undefined reference to" in line or line.startswith("/usr/bin/ld")]
    symbol, driver = captured(case)
    assert symbol == diag("error", None, None, None, None, "undefined symbol: helper(int)")
    assert driver.message.startswith("linker command failed")


def test_the_device_link_is_reported_by_lld_without_the_ld_prefix() -> None:
    # Check 6: the gfx942 device link fails in lld, which names itself "lld:", not "ld.lld:"; the part 1 parser
    # missed that line, and DEVICE_LLD reads it. The clang++ driver calls the step amdgcn-link.
    case = "hip_device_undefined"
    lines = stderr_lines(case)
    assert lines[0] == "lld: error: undefined hidden symbol: device_helper(int)"
    assert "clang++: error: amdgcn-link command failed with exit code 1 (use -v to see invocation)" in lines
    assert entry(case)["exit_status"] == 1, "the build fails, so there is no program"
    symbol, driver = captured(case)
    assert symbol == diag("error", None, None, None, None, "undefined hidden symbol: device_helper(int)")
    assert driver.message.startswith("amdgcn-link command failed")
    assert hipcc_module().DEVICE_LLD in hipcc_module()._PATTERNS


def test_ttmetal_hosts_parse_of_lld_lines_is_unchanged() -> None:
    # DEVICE_LLD is hipcc's own: ttmetal-host still reads no "lld:" line, and both read an "ld.lld:" line alike.
    device = "lld: error: undefined hidden symbol: device_helper(int)\n"
    host = "ld.lld: error: undefined symbol: helper(int)\n"
    assert ttmetal_build.parse_diagnostics(device) == []
    assert ttmetal_build.parse_diagnostics(host) == parse(host)
    assert hipcc_module().DEVICE_LLD not in ttmetal_build._PATTERNS


@pytest.mark.parametrize(
    ("line", "severity", "message"),
    [
        ("lld: warning: SYNTHETIC device link warning", "warning", "SYNTHETIC device link warning"),
        (
            "lld: error: main.hip:1:2: error: SYNTHETIC quoted place",
            "error",
            "main.hip:1:2: error: SYNTHETIC quoted place",
        ),
    ],
)
def test_a_device_lld_line_parses_with_no_place(line: str, severity: str, message: str) -> None:
    # SYNTHETIC lines in the captured lld format; the second shows the lld pattern is tried before the place pattern.
    assert parse(line + "\n", BUILT) == [diag(severity, None, None, None, None, message)]


def test_only_the_linker_cases_name_the_capture_workdir_and_only_in_lld_context_lines() -> None:
    # Check 5: the per-run text, the workdir (it holds the rx id) and clang's random temporary object, is in the
    # ">>> " lines of the two linker cases and gives no diagnostic.
    naming = {case: [line for line in stderr_lines(case) if CAPTURE_WORKDIR in line] for case in CASES}
    assert sorted(case for case, lines in naming.items() if lines) == ["hip_device_undefined", "hip_linker_error"]
    for case, lines in naming.items():
        for line in lines:
            assert line.startswith(">>> ") and "/@lassi-tmp/main-" in line, (case, line)
            assert parse(line + "\n", source_files(case)) == []


@pytest.mark.parametrize("case", CASES)
def test_hipcc_prints_one_line_of_its_own_after_a_failed_clang_call(case: str) -> None:
    # Check 7: every failed build ends with hipcc's "failed to execute:<clang++ command>", which parses to nothing;
    # no line starts with "hipcc" or names clang-offload-bundler.
    lines = stderr_lines(case)
    own = [line for line in lines if line.startswith(HIPCC_FAILED_PREFIX)]
    assert not [line for line in lines if line.startswith("hipcc") or "clang-offload-bundler" in line], case
    if entry(case)["exit_status"] == 0:
        assert own == [], case
        return
    assert own == [lines[-1]], f"{case}: hipcc's line is the last one"
    command = own[0][len(HIPCC_FAILED_PREFIX) :].split()
    assert command[0] == "/opt/rocm/core-7.12/lib/llvm/bin/clang++"
    assert f"--offload-arch={TARGET}" in command and command[-1] == "main.hip"
    assert parse(own[0] + "\n", source_files(case)) == []


def test_the_readme_records_the_capture_without_placeholder() -> None:
    if not README.is_file():
        pytest.fail(f"{README} does not exist; task P17.6 part 2 records the capture's provenance there")
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


# ---------------------------------------------------------------------------
# The remote test's trace reader, on SYNTHETIC lines in strace -f's format (its OQ-002 evidence; none captured)

# Each execve names its program first and its argv after it; the shell's argv carries --offload-arch=gfx942, which
# must never read as the program offload-arch. The third execve is a failed attempt, which still counts.
SYNTHETIC_TRACE = (
    '101 execve("/bin/sh", ["/bin/sh", "-c", "\\"$@\\"; exit \\"$?\\"", "sh", "/opt/rocm/core-7.12/bin/hipcc", '
    '"--offload-arch=gfx942"], 0x7ffd /* 4 vars */) = 0\n'
    '101 openat(AT_FDCWD, "/dev/null", O_RDONLY|O_CLOEXEC) = 3\n'
    '102 execve("/opt/rocm/core-7.12/bin/hipcc", ["/opt/rocm/core-7.12/bin/hipcc", "--offload-arch=gfx942", "-o", '
    '"main", "main.hip"], 0x5f /* 4 vars */ <unfinished ...>\n'
    "102 <... execve resumed>) = 0\n"
    '103 execve("/usr/local/bin/amdgpu-arch", ["amdgpu-arch"], 0x1 /* 4 vars */) = -1 ENOENT (No such file or '
    "directory)\n"
    '104 newfstatat(AT_FDCWD, "/sys/devices/system/cpu/online", {st_mode=S_IFREG|0444, st_size=4096, ...}, 0) = 0\n'
    '104 readlink("/proc/self/exe", "/opt/rocm/core-7.12/lib/llvm/bin/clang-22", 4095) = 41\n'
    '104 openat(AT_FDCWD, "main.hip", O_RDONLY) = 3\n'
    '105 execveat(AT_FDCWD, "/opt/rocm/core-7.12/lib/llvm/bin/lld", ["lld"], 0x1 /* 4 vars */, 0) = 0\n'
    '106 openat(AT_FDCWD, "/dev/kfd", O_RDWR|O_CLOEXEC) = -1 ENOENT (No such file or directory)\n'
)


def test_the_remote_trace_reader_reads_programs_by_path_and_never_by_argv() -> None:
    remote = importlib.import_module("test_hipcc_remote")
    programs, paths = remote.traced_calls(SYNTHETIC_TRACE)
    assert programs == [
        "/bin/sh",
        "/opt/rocm/core-7.12/bin/hipcc",
        "/usr/local/bin/amdgpu-arch",
        "/opt/rocm/core-7.12/lib/llvm/bin/lld",
    ]
    assert paths == [
        "/dev/null",
        "/sys/devices/system/cpu/online",
        "/proc/self/exe",
        "/opt/rocm/core-7.12/lib/llvm/bin/clang-22",
        "/dev/kfd",
    ]
    queries = [program for program in programs if PurePosixPath(program).name in remote.DEVICE_QUERY_PROGRAMS]
    assert queries == ["/usr/local/bin/amdgpu-arch"]
    assert [path for path in paths if path.startswith(remote.GPU_PATH_PREFIXES)] == ["/dev/kfd"]
    assert not "/usr/lib/x86_64-linux-gnu/libdrm_amdgpu.so.1".startswith(remote.GPU_PATH_PREFIXES)


# A clang the pinned hipcc starts, as a version check or a build would start it (SYNTHETIC).
SYNTHETIC_CLANG = (
    '107 execve("/opt/rocm/core-7.12/lib/llvm/bin/clang", ["clang", "--offload-arch=gfx942", "--version"], 0x1 '
    "/* 4 vars */) = 0\n"
)


def test_the_remote_trace_check_refuses_a_device_query_program_a_gpu_path_and_an_empty_trace() -> None:
    remote = importlib.import_module("test_hipcc_remote")
    lines = SYNTHETIC_TRACE.splitlines(keepends=True)
    clean = "".join(line for line in lines if "amdgpu-arch" not in line and "/dev/kfd" not in line)
    remote.check_trace(clean + SYNTHETIC_CLANG, "a SYNTHETIC trace")
    with pytest.raises(AssertionError, match="device-query program"):
        remote.check_trace(SYNTHETIC_TRACE + SYNTHETIC_CLANG, "a SYNTHETIC trace")
    with pytest.raises(AssertionError, match="GPU path"):
        remote.check_trace("".join(line for line in lines if "amdgpu-arch" not in line) + SYNTHETIC_CLANG, "a trace")
    with pytest.raises(AssertionError, match="covers the pinned hipcc"):
        remote.check_trace("", "an empty trace")


# ---------------------------------------------------------------------------
# The bible's ROCm guidance, reconciled (the main session applies the bible edit to the mirror before review)


def bible_lines() -> list[str]:
    """Return the lines of the bible's mirror, docs/BIBLE.md."""
    return BIBLE.read_bytes().decode("utf-8").splitlines()


def test_the_bible_no_longer_tells_a_build_to_source_the_rocm_environment_script() -> None:
    text = "\n".join(bible_lines())
    assert "Source `rocm_env.sh`" not in text, "the gpu (AMD) key settings name the pinned hipcc instead"
    assert "Source `/mnt/nvme10/john_ufl/rocm_env.sh`" not in text, "Host Facts: no build sources it"


def test_the_bibles_gpu_amd_row_and_toolchain_pins_name_the_hip_build() -> None:
    rows = [line for line in bible_lines() if line.startswith("| gpu (AMD) |")]
    assert len(rows) == 1, "the bible's Execution Backends table has one gpu (AMD) row"
    assert NAME in rows[0] and f"--offload-arch={TARGET}" in rows[0]
    pinned = [line for line in bible_lines() if "toolchains/hipcc.pin" in line and line.startswith("- ")]
    assert pinned, "Toolchain Pins has a bullet for the HIP build's pin"
    assert any(f"`{hipcc_pin()['FLAGS']}`" in line for line in pinned), "the bullet gives the pin's FLAGS"
