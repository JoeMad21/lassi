"""Tests for sandboxed compiles of generated sources (P0.20; bible Sandbox, Toolchain Pins; Agent Rules 6, 7, 10, 12).

P0.20 runs every compile of model-generated sources through the P0.16
sandbox, lassi.executors.sandbox, instead of a bare process that holds the
caller's environment. The interface these tests pin:

- SandboxSpec.environment (default None). None keeps the program defaults of
  P0.16 (PATH=SANDBOX_PATH, HOME=<workdir>, LANG=C.UTF-8, TMPDIR=/tmp) and
  the exact P0.16 command. A mapping gives the program exactly those
  variables and nothing else. Its names come from ENVIRONMENT_NAMES, a fixed
  frozenset allowlist that holds PATH, LANG, LC_ALL, TMPDIR, and every
  variable a pin names (NVHPC_CUDA_HOME), and never HOME, a loader or
  compiler-flag variable, or a credential. Its values are strings without
  NUL. Anything else raises ValueError. Each variable reaches the command as
  one NAME=value element after the constant setup script, so no shell parses
  it, and the command otherwise stays the P0.16 command.
- SandboxedCompileRunner(environment=..., toolchains=..., hidden_roots=...,
  runner=None), a CommandRunner: (argv, cwd, timeout_s) -> CommandResult.
  `environment` is what every compile gets (no HOME and no TMPDIR), kept as a
  copy; `toolchains` and `hidden_roots` are kept too. spec(workdir) returns
  the SandboxSpec of a compile in that build directory, a pure function of
  it: the build directory as the workdir, the toolchains root exposed
  read-only, no harness, the hidden roots, and the environment plus TMPDIR, a
  private directory under the build directory. limits(timeout_s) returns its
  Limits: the toolchain's timeout is the wall limit, and the memory limit
  never halves the workdir disk cap (workdir_cap_bytes). A call creates the
  private TMPDIR, runs the compile through the sandbox command of that spec
  and those limits (so prlimit --core=1 covers it), and returns the
  compiler's status, stdout, and stderr, whole far past a program run's
  OUTPUT_CAP_BYTES (see Output below). A compile that
  reaches the wall limit returns -1 and ends stderr with a "timed out" line,
  as the toolchains' own runners do. A sandbox that cannot run the compile
  raises SandboxUnavailableError, so nothing ever compiles outside it.
- Output (review finding): with runner=None each stream is kept whole up to
  COMPILE_OUTPUT_CAP_BYTES, far above any compile's output and far above a
  program run's OUTPUT_CAP_BYTES; past it the runner keeps the stream's head
  and tail (lassi.toolchains CappedRunner), so a compiler that prints without
  end cannot fill the host's memory, and a "lassi-sandbox:" line at the end
  of stderr says the stream was cut. Nothing is lost silently otherwise
  either: a compile killed before its wall limit (review finding) gets a line
  naming the limits that kill (memory, CPU time), and one whose whole
  sandbox was killed gets a line saying the build dir may lack its outputs.

The limit that A1 states is checked as well: files outside $HOME, the
scratch root, and the runs root keep the host user's read permission inside
a compile, as they do for program runs.

No test here starts a process or a sandbox. The sandbox's runner is a fake,
or subprocess.Popen is replaced by a fake that answers as the sandbox setup
would (the ready line, the program's output, the done line). The paths are
made up unless a test creates them under tmp_path. No value here is a
measurement.
"""

from __future__ import annotations

import dataclasses
import inspect
import io
import math
import os
import re
import subprocess
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.core.interfaces import Limits
from lassi.toolchains import CommandResult, NvccSm80
from lassi.toolchains._base import OUTPUT_CAP_BYTES
from lassi.toolchains.pins import PREFIX_VARIABLES

ROOT = Path(Path(__file__).resolve().anchor)
SCRATCH = ROOT / "scratch"
HOME = ROOT / "home" / "user"
RUNS = SCRATCH / "lassi-runs"
TOOLCHAINS = SCRATCH / "toolchains"
WORK = RUNS / "runs" / "run-1" / "trial-a" / "attempt00" / "build"
OTHER_TRIAL = RUNS / "runs" / "run-1" / "trial-b" / "attempt00" / "build"
NVCC = TOOLCHAINS / "cuda@12.6.3" / "bin" / "nvcc"
# The fixed part of a compile environment, as the stage runner builds it, and the nvc++ one with its linked prefix.
COMPILE_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
NVCPP_ENV = {**COMPILE_ENV, "NVHPC_CUDA_HOME": (TOOLCHAINS / "cuda@12.6.3").as_posix()}
FLAGS = ["-std=c++14", "-Xcompiler", "-Wall", "-arch=sm_80", "-O3", "-o", "main", "main.cu"]
PROGRAM = ["./main", "--size", "8"]
LIMITS = Limits(wall_s=600.0, memory_mb=4096, cpus=1)
# Names the allowlist must hold, and names it must never hold (a loader or compiler-flag variable, a credential).
REQUIRED_NAMES = ("PATH", "LANG", "LC_ALL", "TMPDIR", "NVHPC_CUDA_HOME")
FORBIDDEN_NAMES = (
    "HOME",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "CPATH",
    "LASSI_TEST_API_KEY",
    "XDG_RUNTIME_DIR",
)
# Compiler stderr as an EDG error prints it (capture nvcc_undefined_identifier holds one such line).
STDERR_TEXT = 'main.cu(7): error: identifier "undefined_var" is undefined\n'
READY_LINE = "lassi-sandbox-ready\n"
# The directories the sandbox covers with a tmpfs, where it refuses a workdir.
PRIVATE_OR_SYSTEM = ("/tmp", "/var/tmp", "/dev/shm", "/run", "/var", "/sys")


def done_line(program_s: float) -> str:
    """Return the setup's done line with clock readings `program_s` apart, as /proc/uptime prints them."""
    ended = 100000 + math.floor(round(program_s * 100, 6))
    return f"lassi-sandbox-done 1000.00 {ended // 100}.{ended % 100:02d}\n"


@pytest.fixture
def sandbox() -> ModuleType:
    """Return the lassi.executors.sandbox module (imported here so each test shows a missing name clearly)."""
    from lassi.executors import sandbox

    return sandbox


def sample_spec(sandbox: ModuleType, **overrides: object) -> Any:
    """Return a SandboxSpec with the sample build dir, the scratch and home roots hidden, and the toolchains root."""
    values: dict[str, object] = {"workdir": WORK, "hidden_roots": (SCRATCH, HOME), "toolchains": TOOLCHAINS}
    values.update(overrides)
    return sandbox.SandboxSpec(**values)


def compile_runner(
    sandbox: ModuleType,
    runner: Any = None,
    *,
    environment: dict[str, str] | None = None,
    toolchains: Path = TOOLCHAINS,
    hidden_roots: Sequence[Path] = (HOME, SCRATCH, RUNS),
) -> Any:
    """Return a SandboxedCompileRunner with the sample compile environment, toolchains root, and hidden roots."""
    chosen = dict(COMPILE_ENV if environment is None else environment)
    return sandbox.SandboxedCompileRunner(
        environment=chosen, toolchains=toolchains, hidden_roots=tuple(hidden_roots), runner=runner
    )


def script_of(command: list[str]) -> str:
    """Return the script element of a sandbox command: the element after "sh", "-c"."""
    start = command.index("sh")
    assert command[start + 1] == "-c", command
    return command[start + 2]


def is_subsequence(short: Sequence[str], long: Sequence[str]) -> bool:
    """Return True when every element of `short` appears in `long` in the same order."""
    remaining = iter(long)
    return all(any(part == other for other in remaining) for part in short)


def covered(path: Path, roots: Sequence[Path]) -> bool:
    """Return True when `path` is one of `roots` or lies under one."""
    return any(root == path or root in path.parents for root in roots)


def visible_inside(spec: Any, path: Path) -> bool:
    """Return whether `path` can be read inside a sandbox built from `spec`, as the setup lays out the view.

    The workdir, the harness, and the toolchains root are bound at their own
    paths; each hidden root is covered by an empty tmpfs apart from the path
    skeleton to those three; any other path keeps the host user's read
    permission (the limit A1 states). The private and system directories are
    left out here, since no test path lies under them.
    """
    exposures = [item for item in (spec.workdir, spec.harness, spec.toolchains) if item is not None]
    if covered(path, [Path(item) for item in exposures]):
        return True
    return not covered(path, [Path(root) for root in spec.hidden_roots])


@dataclass(frozen=True)
class RealLayout:
    """Directories a test creates under tmp_path: home, scratch, a toolchains root, and a build dir in scratch."""

    home: Path
    scratch: Path
    toolchains: Path
    workdir: Path

    @property
    def hidden(self) -> tuple[Path, ...]:
        """Return the roots a compile hides: home and scratch (the runs root lies under scratch)."""
        return (self.home, self.scratch)

    def compile_argv(self) -> list[str]:
        """Return the nvcc command line the adapter would run, with the pinned path under this toolchains root."""
        return [str(self.toolchains / "cuda@12.6.3" / "bin" / "nvcc"), *FLAGS]


@pytest.fixture
def real(tmp_path: Path) -> RealLayout:
    """Create home, scratch, a toolchains root, and a build dir under tmp_path; skip where the sandbox refuses them."""
    text = tmp_path.resolve().as_posix()
    blocked = [name for name in PRIVATE_OR_SYSTEM if text == name or text.startswith(name + "/")]
    if blocked:
        pytest.skip(f"the sandbox refuses a build dir under {blocked[0]}; point TMPDIR at the scratch disk")
    home, scratch = tmp_path / "home", tmp_path / "scratch"
    toolchains, workdir = scratch / "toolchains", scratch / "lassi-runs" / "trial" / "attempt00" / "build"
    for directory in (home, toolchains, workdir):
        directory.mkdir(parents=True)
    return RealLayout(home=home, scratch=scratch, toolchains=toolchains, workdir=workdir)


@dataclass
class FakeSandboxRunner:
    """Stands in for the compile sandbox's command runner: records each call and answers as the setup would.

    The answer's stderr is the ready line, `stderr`, and a done line whose
    clock readings are `program_s` apart, unless `ready` is False (a setup
    that failed before the program) or `done` is False (the whole sandbox
    was killed before its done line). `error` is raised instead when set, and
    `on_call` runs at the start of each call. Nothing is started.
    """

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    program_s: float = 0.5
    ready: bool = True
    done: bool = True
    error: OSError | None = None
    on_call: Callable[[], None] | None = None
    calls: list[tuple[list[str], Path, float]] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the call and return the canned result, or raise the canned error."""
        self.calls.append((list(argv), Path(cwd), timeout_s))
        if self.on_call is not None:
            self.on_call()
        if self.error is not None:
            raise self.error
        stderr = self.stderr
        if self.ready:
            stderr = READY_LINE + stderr + (done_line(self.program_s) if self.done else "")
        return CommandResult(returncode=self.returncode, stdout=self.stdout, stderr=stderr)


# A compiler diagnostic line that a flood repeats (the form of capture nvcc_warning_177).
FLOOD_LINE = b'main.cu(1): warning #177-D: variable "v" was declared but never referenced\n'


class Flood(io.RawIOBase):
    """A readable stream of `prefix`, then `size` bytes of FLOOD_LINE lines, then `suffix`, made as it is read.

    Nothing holds the whole flood, so a test can offer far more bytes than a
    runner may keep; `read_total` counts what was read.
    """

    def __init__(self, prefix: bytes, size: int, suffix: bytes) -> None:
        """Keep the parts; nothing is generated yet."""
        super().__init__()
        self.prefix, self.size, self.suffix = prefix, size, suffix
        self.sent = 0
        self.read_total = 0

    def readable(self) -> bool:
        """Return True: the stream is read like a pipe."""
        return True

    def read1(self, size: int = -1) -> bytes:
        """Return the next chunk: the prefix, then up to `size` bytes of the flood, then the suffix, then b""."""
        if self.prefix:
            chunk, self.prefix = self.prefix, b""
        elif self.sent < self.size:
            count = min(size if size > 0 else 1 << 16, self.size - self.sent)
            chunk = (FLOOD_LINE * (count // len(FLOOD_LINE) + 1))[:count]
            self.sent += count
        else:
            chunk, self.suffix = self.suffix, b""
        self.read_total += len(chunk)
        return chunk

    def read(self, size: int = -1) -> bytes:
        """Return the next chunk, as read1 does."""
        return self.read1(size)


class FinishedSandbox:
    """What the fake subprocess.Popen returns: a sandbox command that has already exited; nothing was started."""

    # A pid no process has, so a kill (only after a timeout, which never happens here) finds nothing.
    pid = 2**31 - 1

    def __init__(self, argv: list[str], returncode: int, stdout: bytes, stderr: bytes) -> None:
        """Keep the exit status and offer the output as readable byte streams."""
        self.args = argv
        self.returncode = returncode
        self.stdin = None
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)

    def communicate(self, input: Any = None, timeout: float | None = None) -> tuple[bytes, bytes]:
        """Return what is left of both streams."""
        return self.stdout.read(), self.stderr.read()

    def poll(self) -> int:
        """Return the exit status."""
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        """Return the exit status."""
        return self.returncode

    def kill(self) -> None:
        """Do nothing: no process was started."""

    def __enter__(self) -> FinishedSandbox:
        """Return self, as Popen does."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Do nothing on exit."""


# ---------------------------------------------------------------------------
# SandboxSpec.environment: a validated allowlist, passed as positional arguments


def test_the_spec_environment_defaults_to_none_the_program_defaults(sandbox: ModuleType) -> None:
    assert "environment" in {item.name for item in dataclasses.fields(sandbox.SandboxSpec)}
    spec = sandbox.SandboxSpec(workdir=WORK, hidden_roots=(SCRATCH,))
    assert spec.environment is None


def test_environment_names_are_a_fixed_allowlist_without_home_loader_flag_or_secret_variables(
    sandbox: ModuleType,
) -> None:
    names = sandbox.ENVIRONMENT_NAMES
    assert isinstance(names, frozenset), "a fixed allowlist that nothing can extend at run time"
    assert set(REQUIRED_NAMES) <= names
    assert set(PREFIX_VARIABLES.values()) <= names, "every variable a pin names can reach its compile"
    assert not names & set(FORBIDDEN_NAMES), sorted(names & set(FORBIDDEN_NAMES))
    assert all(re.fullmatch(r"[A-Z][A-Z0-9_]*", name) for name in names), sorted(names)


def test_a_spec_keeps_an_allowed_environment(sandbox: ModuleType) -> None:
    environment = {**NVCPP_ENV, "TMPDIR": (WORK / ".tmp").as_posix()}
    spec = sample_spec(sandbox, environment=environment)
    assert dict(spec.environment) == environment
    environment["CPATH"] = "/leaked"
    assert "CPATH" not in spec.environment, "the spec keeps its own copy"


@pytest.mark.parametrize("name", [*FORBIDDEN_NAMES, "PATH=/bin", "path", ""])
def test_a_spec_refuses_a_variable_outside_the_allowlist(sandbox: ModuleType, name: str) -> None:
    with pytest.raises(ValueError) as refused:
        sample_spec(sandbox, environment={**COMPILE_ENV, name: "/value"})
    if name:
        assert name in str(refused.value), str(refused.value)


@pytest.mark.parametrize("value", ["/usr/bin\0/evil", 5, None, b"/usr/bin"], ids=["nul", "int", "none", "bytes"])
def test_a_spec_refuses_a_value_that_is_not_a_string_without_nul(sandbox: ModuleType, value: object) -> None:
    with pytest.raises(ValueError):
        sample_spec(sandbox, environment={**COMPILE_ENV, "PATH": value})


def test_a_spec_refuses_an_environment_given_as_one_string(sandbox: ModuleType) -> None:
    with pytest.raises(ValueError):
        sample_spec(sandbox, environment="PATH=/usr/bin")


def test_program_runs_keep_the_p0_16_command_when_no_environment_is_given(sandbox: ModuleType) -> None:
    default = sandbox.sandbox_command(sample_spec(sandbox), PROGRAM, LIMITS)
    assert sandbox.sandbox_command(sample_spec(sandbox, environment=None), PROGRAM, LIMITS) == default


def test_an_environment_reaches_the_command_as_one_name_value_element_each_after_the_setup_script(
    sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Agent Rule 12: only the allowlisted variables enter, and each value is a positional argument that no shell
    # parses, whatever it holds.
    for name in sandbox.PASSED_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LASSI_TEST_API_KEY", "sk-lassi-test-secret")
    environment = {
        "PATH": "/opt/a b/bin:/usr/bin:$(reboot)",
        "LANG": "C",
        "LC_ALL": "C",
        "TMPDIR": (WORK / "t `id`;\n'q' \"$HOME\"").as_posix(),
        "NVHPC_CUDA_HOME": (TOOLCHAINS / "cuda@12.6.3 && rm -rf x").as_posix(),
    }
    compile_argv = [str(NVCC), *FLAGS]
    default = sandbox.sandbox_command(sample_spec(sandbox), compile_argv, LIMITS)
    command = sandbox.sandbox_command(sample_spec(sandbox, environment=environment), compile_argv, LIMITS)
    assert script_of(command) == sandbox.SETUP_SCRIPT, "the setup script stays one constant"
    assert command[:5] == ["prlimit", "--core=1", "--", "env", "-i"]
    assert command[-len(compile_argv) :] == compile_argv, "the compiler argv stays the command's tail"
    assert is_subsequence(default, command), "the environment only adds elements to the P0.16 command"
    start = command.index(sandbox.SETUP_SCRIPT)
    end = len(command) - len(compile_argv)
    for name, value in environment.items():
        holders = [index for index, part in enumerate(command) if part == f"{name}={value}"]
        assert len(holders) == 1 and start < holders[0] < end, (name, holders)
    assert not [part for part in command if "sk-lassi-test-secret" in part]


# ---------------------------------------------------------------------------
# SandboxedCompileRunner: the spec and limits of one compile


def test_the_compile_runner_keeps_a_copy_of_its_environment_and_its_roots(sandbox: ModuleType) -> None:
    environment = dict(COMPILE_ENV)
    runner = sandbox.SandboxedCompileRunner(
        environment=environment, toolchains=TOOLCHAINS, hidden_roots=(HOME, SCRATCH)
    )
    environment["CPATH"] = "/leaked"
    assert runner.environment == COMPILE_ENV
    assert Path(runner.toolchains).resolve() == TOOLCHAINS.resolve()
    assert [Path(root).resolve() for root in runner.hidden_roots] == [HOME.resolve(), SCRATCH.resolve()]


def test_a1_a_compile_exposes_its_build_dir_and_the_toolchains_and_hides_home_scratch_and_the_runs_root(
    sandbox: ModuleType,
) -> None:
    spec = compile_runner(sandbox).spec(WORK)
    assert isinstance(spec, sandbox.SandboxSpec)
    assert Path(spec.workdir) == WORK.resolve()
    assert Path(spec.toolchains) == TOOLCHAINS.resolve()
    assert spec.harness is None, "a compile gets no harness"
    roots = [Path(root) for root in spec.hidden_roots]
    for root in (HOME, SCRATCH, RUNS):
        assert covered(root.resolve(), roots), root


@pytest.mark.parametrize(
    ("path", "visible"),
    [
        (HOME / ".ssh" / "id_ed25519", False),
        (HOME / "notes" / "secret.h", False),
        (SCRATCH / "tmp" / "secret.h", False),
        (SCRATCH / "hf" / "token", False),
        (OTHER_TRIAL / "main.cu", False),
        (RUNS / "runs" / "run-1" / "provenance.json", False),
        (WORK / "kernels" / "local.h", True),
        (NVCC.parent.parent / "include" / "cuda_runtime.h", True),
        (ROOT / "usr" / "include" / "stdio.h", True),
    ],
    ids=[
        "home-key",
        "home-header",
        "scratch-file",
        "scratch-token",
        "another-trial",
        "run-manifest",
        "build-dir",
        "toolchains",
        "elsewhere-keeps-host-read-permission",
    ],
)
def test_a1_an_absolute_include_resolves_only_in_the_build_dir_the_toolchains_or_outside_the_hidden_roots(
    sandbox: ModuleType, path: Path, visible: bool
) -> None:
    # A generated `#include "<absolute path>"` reads that path inside the compile's view: under $HOME, the
    # scratch root, and the runs root only the build dir and the toolchains root exist.
    spec = compile_runner(sandbox).spec(WORK)
    assert visible_inside(spec, path.resolve()) is visible


@pytest.mark.parametrize("environment", [COMPILE_ENV, NVCPP_ENV], ids=["nvcc", "nvcpp"])
def test_a2_a3_a_compile_gets_its_environment_and_a_private_tmpdir_under_its_build_dir_and_no_home(
    sandbox: ModuleType, environment: dict[str, str]
) -> None:
    runner = compile_runner(sandbox, environment=environment)
    spec = runner.spec(WORK)
    given = dict(spec.environment)
    tmpdir = Path(given.pop("TMPDIR"))
    assert given == environment, "exactly the compile environment and TMPDIR"
    assert "HOME" not in spec.environment
    assert WORK.resolve() in tmpdir.resolve().parents, "a private TMPDIR under its own build dir"
    assert visible_inside(spec, tmpdir.resolve())
    other = Path(dict(runner.spec(OTHER_TRIAL).environment)["TMPDIR"])
    assert OTHER_TRIAL.resolve() in other.resolve().parents
    assert runner.spec(WORK) == spec, "spec(workdir) depends on the build dir alone"


@pytest.mark.parametrize("name", ["HOME", "LD_PRELOAD", "NVCC_PREPEND_FLAGS", "CPATH", "LASSI_TEST_API_KEY"])
def test_a2_the_compile_runner_refuses_home_and_every_variable_outside_the_allowlist(
    sandbox: ModuleType, name: str
) -> None:
    with pytest.raises(ValueError):
        runner = compile_runner(sandbox, environment={**COMPILE_ENV, name: "/value"})
        runner.spec(WORK)


@pytest.mark.parametrize("timeout_s", [600.0, 12.5])
def test_a_compiles_limits_take_the_toolchain_timeout_as_wall_time_and_keep_its_disk_cap_whole(
    sandbox: ModuleType, timeout_s: float
) -> None:
    runner = compile_runner(sandbox)
    limits = runner.limits(timeout_s)
    assert isinstance(limits, Limits)
    assert limits.wall_s == timeout_s
    for name in ("memory_mb", "cpus"):
        value = getattr(limits, name)
        assert isinstance(value, int) and not isinstance(value, bool) and value >= 1, (name, value)
    spec = runner.spec(WORK)
    assert sandbox.workdir_cap_bytes(spec, limits) == spec.disk_mb << 20, "the memory limit must not halve it"


# ---------------------------------------------------------------------------
# SandboxedCompileRunner: one compile


def test_a5_a_compile_runs_as_the_sandbox_command_of_its_spec_under_prlimit_core_1(
    sandbox: ModuleType, real: RealLayout
) -> None:
    fake = FakeSandboxRunner(returncode=2, stderr=STDERR_TEXT)
    runner = compile_runner(sandbox, fake, toolchains=real.toolchains, hidden_roots=real.hidden)
    argv = real.compile_argv()
    result = runner(argv, real.workdir, 600.0)
    ((command, cwd, _timeout),) = fake.calls
    assert command == sandbox.sandbox_command(runner.spec(real.workdir), argv, runner.limits(600.0))
    assert command[:3] == ["prlimit", "--core=1", "--"], "a compiler crash stores no core with the host handler"
    assert command[-len(argv) :] == argv
    assert cwd.resolve() == real.workdir.resolve()
    assert result == CommandResult(returncode=2, stdout="", stderr=STDERR_TEXT)


def test_a3_the_private_tmpdir_exists_when_the_compile_starts(sandbox: ModuleType, real: RealLayout) -> None:
    seen: list[bool] = []
    fake = FakeSandboxRunner()
    runner = compile_runner(sandbox, fake, toolchains=real.toolchains, hidden_roots=real.hidden)
    tmpdir = Path(dict(runner.spec(real.workdir).environment)["TMPDIR"])
    fake.on_call = lambda: seen.append(tmpdir.is_dir())
    runner(real.compile_argv(), real.workdir, 600.0)
    assert seen == [True]


@pytest.mark.parametrize("status", [0, 1, 2, 255, 139])
def test_the_compilers_status_and_output_pass_through(sandbox: ModuleType, real: RealLayout, status: int) -> None:
    fake = FakeSandboxRunner(returncode=status, stdout="built\n", stderr=STDERR_TEXT)
    runner = compile_runner(sandbox, fake, toolchains=real.toolchains, hidden_roots=real.hidden)
    result = runner(real.compile_argv(), real.workdir, 600.0)
    assert (result.returncode, result.stdout, result.stderr) == (status, "built\n", STDERR_TEXT)
    assert not result.stdout_truncated and not result.stderr_truncated


def test_a_compile_that_reaches_the_wall_limit_returns_minus_one_with_a_timed_out_line(
    sandbox: ModuleType, real: RealLayout
) -> None:
    fake = FakeSandboxRunner(returncode=124, stderr="partial output\n", program_s=5.0)
    runner = compile_runner(sandbox, fake, toolchains=real.toolchains, hidden_roots=real.hidden)
    result = runner(real.compile_argv(), real.workdir, 5.0)
    assert result.returncode == -1, "the toolchains' timeout status, which build() reports as a timeout"
    assert result.stderr.startswith("partial output\n")
    assert result.stderr.rstrip("\n").splitlines()[-1].startswith("timed out")
    toolchain = NvccSm80(executable=real.compile_argv()[0], runner=runner, timeout_s=5.0)
    workdir = real.workdir.parent / "attempt01"
    workdir.mkdir()
    built = toolchain.build({"main.cu": "int main() { return 0; }\n"}, workdir)
    assert built.artifact is None
    assert [d.code for d in built.diagnostics if "timed out" in d.message] == ["exit-status"], built.diagnostics


def test_a_sandbox_that_cannot_run_the_compile_raises_and_nothing_compiles_unsandboxed(
    sandbox: ModuleType, real: RealLayout
) -> None:
    failed = FakeSandboxRunner(returncode=1, stderr="mount: permission denied\n", ready=False)
    runner = compile_runner(sandbox, failed, toolchains=real.toolchains, hidden_roots=real.hidden)
    with pytest.raises(sandbox.SandboxUnavailableError):
        runner(real.compile_argv(), real.workdir, 600.0)
    assert [call[0][:3] for call in failed.calls] == [["prlimit", "--core=1", "--"]], "one sandboxed try, no other"
    missing = FakeSandboxRunner(error=FileNotFoundError(2, "No such file or directory", "prlimit"))
    runner = compile_runner(sandbox, missing, toolchains=real.toolchains, hidden_roots=real.hidden)
    with pytest.raises(sandbox.SandboxUnavailableError):
        runner(real.compile_argv(), real.workdir, 600.0)
    assert len(missing.calls) == 1


def test_a_compile_keeps_all_of_the_compilers_output_past_the_program_output_cap(
    sandbox: ModuleType, real: RealLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The P0.15 contract keeps compiler stderr whole; the 1 MiB cap is for generated programs only.
    line = b'main.cu(%d): warning #177-D: variable "v%d" was declared but never referenced\n'
    stderr = b"".join(line % (number, number) for number in range(1, 30000))
    stdout = b"o" * (OUTPUT_CAP_BYTES + 12345)
    assert len(stderr) > OUTPUT_CAP_BYTES + 65536
    started: list[list[str]] = []

    def popen(args: Any, *more: Any, **kwargs: Any) -> FinishedSandbox:
        argv = [os.fsdecode(item) for item in args]
        started.append(argv)
        assert argv[:3] == ["prlimit", "--core=1", "--"], f"only the sandbox command may start, not {argv[:3]}"
        wrapped = READY_LINE.encode("ascii") + stderr + done_line(0.5).encode("ascii")
        return FinishedSandbox(argv, 0, stdout, wrapped)

    monkeypatch.setattr(subprocess, "Popen", popen)
    runner = compile_runner(sandbox, None, toolchains=real.toolchains, hidden_roots=real.hidden)
    result = runner(real.compile_argv(), real.workdir, 600.0)
    assert len(started) == 1
    assert result.stderr.encode("utf-8") == stderr
    assert result.stdout.encode("utf-8") == stdout
    assert (result.returncode, result.stdout_truncated, result.stderr_truncated) == (0, False, False)


# ---------------------------------------------------------------------------
# Output past the compile output cap, and a compile a limit killed: never lost silently (review finding)


def test_the_compile_output_cap_is_documented_and_far_above_any_compile_and_the_program_cap(
    sandbox: ModuleType,
) -> None:
    # The largest compiler stderr in the fixtures and the layout app was 3210 bytes (exploratory, see the spike);
    # the cap keeps a compile's output whole far past that, and far past a program run's 1 MiB.
    cap = sandbox.COMPILE_OUTPUT_CAP_BYTES
    assert isinstance(cap, int) and cap >= 32 * OUTPUT_CAP_BYTES
    runner = compile_runner(sandbox)
    assert getattr(runner.runner, "cap_bytes", None) == cap, "the default runner keeps at most the documented cap"


def flooding_popen(floods: list[Flood], size: int) -> Callable[..., FinishedSandbox]:
    """Return a fake Popen whose sandbox command prints the ready line, `size` bytes of FLOOD_LINE, the done line."""

    def popen(args: Any, *more: Any, **kwargs: Any) -> FinishedSandbox:
        argv = [os.fsdecode(item) for item in args]
        assert argv[:3] == ["prlimit", "--core=1", "--"], f"only the sandbox command may start, not {argv[:3]}"
        process = FinishedSandbox(argv, 1, b"", b"")
        flood = Flood(READY_LINE.encode("ascii"), size, done_line(0.5).encode("ascii"))
        process.stderr = flood  # type: ignore[assignment]
        floods.append(flood)
        return process

    return popen


def test_f1_a_flooding_compile_keeps_a_bounded_head_and_tail_and_says_so(
    sandbox: ModuleType, real: RealLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A generated source can make the compiler print without end (a diagnostic per macro expansion). The host
    # runner sits outside the sandbox's memory limit, so it must keep a bounded part and say what it cut.
    cap = 256 << 10
    size = 64 * cap
    monkeypatch.setattr(sandbox, "COMPILE_OUTPUT_CAP_BYTES", cap)
    floods: list[Flood] = []
    monkeypatch.setattr(subprocess, "Popen", flooding_popen(floods, size))
    runner = compile_runner(sandbox, None, toolchains=real.toolchains, hidden_roots=real.hidden)
    tracemalloc.start()
    try:
        result = runner(real.compile_argv(), real.workdir, 600.0)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    (flood,) = floods
    assert flood.read_total >= size, "the runner drains the whole flood, so the compiler never blocks on its pipe"
    assert result.returncode == 1, "the done line survives in the kept tail, so the compiler's status comes back"
    assert (result.stdout_truncated, result.stderr_truncated) == (False, True)
    body, note = result.stderr.rstrip("\n").rsplit("\n", 1)
    assert len(body.encode("utf-8")) < cap
    assert body.startswith(FLOOD_LINE.decode("ascii")), "the head of the compiler's stderr is kept"
    assert note.startswith("lassi-sandbox:") and "stderr" in note and str(cap) in note, note
    assert peak < size // 4, f"the runner held {peak} bytes of a {size}-byte flood"


def test_f1_build_records_the_cut_in_compile_stderr(
    sandbox: ModuleType, real: RealLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap = 128 << 10
    monkeypatch.setattr(sandbox, "COMPILE_OUTPUT_CAP_BYTES", cap)
    monkeypatch.setattr(subprocess, "Popen", flooding_popen([], 16 * cap))
    runner = compile_runner(sandbox, None, toolchains=real.toolchains, hidden_roots=real.hidden)
    toolchain = NvccSm80(executable=real.compile_argv()[0], runner=runner, timeout_s=600.0)
    built = toolchain.build({"main.cu": "int main() { return 0; }\n"}, real.workdir)
    kept = (real.workdir / "compile.stderr").read_bytes()
    assert len(kept) < cap + 1024
    assert kept.rstrip(b"\n").rsplit(b"\n", 1)[-1].startswith(b"lassi-sandbox:")
    assert built.artifact is None


def test_f5_a_compile_killed_before_its_wall_limit_names_the_limits_that_kill(
    sandbox: ModuleType, real: RealLayout
) -> None:
    fake = FakeSandboxRunner(returncode=137, stderr="partial output\n", program_s=3.0)
    runner = compile_runner(sandbox, fake, toolchains=real.toolchains, hidden_roots=real.hidden)
    result = runner(real.compile_argv(), real.workdir, 600.0)
    assert result.returncode == 137, "the status stays the compiler's, so build() reports it"
    assert result.stderr.startswith("partial output\n")
    last = result.stderr.rstrip("\n").splitlines()[-1]
    assert last.startswith("lassi-sandbox:") and "killed" in last, last
    assert f"{sandbox.COMPILE_MEMORY_MB} MiB" in last and "CPU" in last, last
    toolchain = NvccSm80(executable=real.compile_argv()[0], runner=runner, timeout_s=600.0)
    workdir = real.workdir.parent / "attempt01"
    workdir.mkdir()
    toolchain.build({"main.cu": "int main() { return 0; }\n"}, workdir)
    assert (workdir / "compile.stderr").read_text(encoding="utf-8").rstrip("\n").splitlines()[-1] == last


def test_f5_a_compile_whose_whole_sandbox_was_killed_says_the_build_dir_may_be_incomplete(
    sandbox: ModuleType, real: RealLayout
) -> None:
    # The runner saw a death by SIGKILL (-9) and no done line: the setup died too, before its copy-back.
    fake = FakeSandboxRunner(returncode=-9, stderr="partial output\n", done=False)
    runner = compile_runner(sandbox, fake, toolchains=real.toolchains, hidden_roots=real.hidden)
    result = runner(real.compile_argv(), real.workdir, 600.0)
    assert result.returncode == 137
    notes = [line for line in result.stderr.splitlines() if line.startswith("lassi-sandbox:")]
    assert len(notes) == 2, result.stderr
    assert "killed" in notes[0] and "build dir" in notes[1], notes


def test_a_compile_that_ends_normally_gets_no_sandbox_line(sandbox: ModuleType, real: RealLayout) -> None:
    for status in (0, 2, 139):
        fake = FakeSandboxRunner(returncode=status, stderr=STDERR_TEXT)
        runner = compile_runner(sandbox, fake, toolchains=real.toolchains, hidden_roots=real.hidden)
        assert runner(real.compile_argv(), real.workdir, 600.0).stderr == STDERR_TEXT, status


# ---------------------------------------------------------------------------
# Documentation


def test_the_module_docstring_documents_the_compile_sandbox_and_the_allowlist(sandbox: ModuleType) -> None:
    doc = sandbox.__doc__ or ""
    for term in ("SandboxedCompileRunner", "ENVIRONMENT_NAMES", "COMPILE_OUTPUT_CAP_BYTES"):
        assert term in doc, term
    # With an environment, env's own failures (125, and 126 or 127 when it cannot run the program) can surface as
    # the program's status too (review finding).
    for status in ("125", "126", "127"):
        assert status in doc, status


def test_the_compile_runner_states_what_it_hides_and_the_read_permission_limit(sandbox: ModuleType) -> None:
    doc = inspect.getdoc(sandbox.SandboxedCompileRunner) or ""
    for term in ("HOME", "scratch root", "runs root", "TMPDIR", "read permission", "COMPILE_OUTPUT_CAP_BYTES"):
        assert term in doc, term
    for name in ("spec", "limits"):
        assert inspect.getdoc(getattr(sandbox.SandboxedCompileRunner, name)), name
