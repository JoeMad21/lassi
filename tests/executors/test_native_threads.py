"""Tests for the native executor's bound on a program's OpenMP threads (DEMO.2).

Bible: Execution Backends (executor native, Sandbox; the Harness Contract
bullet on the `nvc++ -mp=multicore` proxy), Agent Rules 6 and 12. The DEMO.1
spike (plans/spikes/demo-multicore-proxy.md) found that on a build host with
many CPUs the OpenMP runtime starts one thread per CPU, passes the sandbox's
TasksMax, and aborts. The native executor bounds a program's OpenMP threads
by Limits.cpus: it hands the sandbox a program environment
(SandboxSpec.environment) that is the sandbox's default program environment
with OMP_NUM_THREADS=<Limits.cpus> added.

The environment these tests pin, variable by variable:

- PATH=SANDBOX_PATH, LANG=C.UTF-8, and TMPDIR=/tmp, the values the sandbox
  default gives a program run today;
- OMP_NUM_THREADS=<Limits.cpus>, a thread count: not a secret, not a loader
  variable, and not a compiler flag, so it keeps Agent Rule 12;
- no HOME. This is the one variable that changes: the sandbox default sets
  HOME=<workdir>, but SandboxSpec.environment never holds HOME (the P0.20
  allowlist, which tests/executors/test_sandbox_compile.py pins), so a run
  with the bound has no HOME. Its working directory is still the workdir.

Nothing from the caller's environment enters, a caller's OMP_NUM_THREADS
included: the bound comes from Limits.cpus only. The variables reach the
sandbox command as one NAME=value element each after the sandbox's
environment marker, right before the artifact and its inputs, so no shell
parses them. Since the program then runs as `env -i -- NAME=value...
<artifact>`, an artifact path that holds "=" is refused with ValueError
before anything starts. The fixed part of the environment is checked
against the program defaults SETUP_SCRIPT's own `env -i` line gives, so the
two copies cannot drift apart silently.

Every test uses a fake Sandbox or a trapped subprocess.Popen, so no sandbox
and no program ever starts; the artifacts are placeholder files that are
never executed. No value in this module is a measurement.
"""

from __future__ import annotations

import inspect
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

from lassi.core.interfaces import Limits

LIMITS = Limits(wall_s=2.5, memory_mb=256, cpus=2)
# The caller's variables a test sets; none of them may reach the program.
SECRET = "sk-lassi-native-threads-test-secret"
CALLER_ENV = {
    "OMP_NUM_THREADS": "999",
    "LASSI_TEST_API_KEY": SECRET,
    "LD_PRELOAD": "/caller/evil.so",
    "LD_LIBRARY_PATH": "/caller/lib",
    "LANG": "de_DE.UTF-8",
    "PATH": "/caller/bin:/usr/bin",
}


@pytest.fixture
def native() -> ModuleType:
    """Return the lassi.executors.native module (imported here so each test shows a missing module clearly)."""
    from lassi.executors import native

    return native


@pytest.fixture
def sandbox() -> ModuleType:
    """Return the lassi.executors.sandbox module."""
    from lassi.executors import sandbox

    return sandbox


@pytest.fixture(autouse=True)
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make tmp_path/runs the runs root, tmp_path/scratch the one hidden root, and unset $HOME and $LASSI_TOOLCHAINS."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(runs))
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("LASSI_TOOLCHAINS", raising=False)
    return runs


@dataclass
class FakeSandbox:
    """A stand-in for lassi.executors.sandbox.Sandbox that records each spec and returns a canned result."""

    result: object
    specs: list[object] = field(default_factory=list)

    def run(self, spec: object, argv: Sequence[str], limits: Limits) -> object:
        """Record the spec and return the canned result; nothing is executed."""
        self.specs.append(spec)
        return self.result


def canned(sandbox: ModuleType) -> object:
    """Return a SandboxResult for a normal exit with status 0."""
    return sandbox.SandboxResult(returncode=0, stdout="", stderr="", wall_s=0.25, hang=False, killed=False)


def make_artifact(tmp_path: Path) -> Path:
    """Return a placeholder artifact in a fresh build directory under the runs root; it is never executed."""
    workdir = tmp_path / "runs" / "attempt00" / "build"
    workdir.mkdir(parents=True)
    artifact = workdir / "main"
    artifact.write_bytes(b"placeholder artifact, never executed\n")
    return artifact


def expected_environment(sandbox: ModuleType, cpus: int) -> dict[str, str]:
    """Return the exact program environment the native executor must pass for `cpus` CPUs (see the module docstring)."""
    return {"PATH": sandbox.SANDBOX_PATH, "LANG": "C.UTF-8", "TMPDIR": "/tmp", "OMP_NUM_THREADS": str(cpus)}


def spec_environment(native: ModuleType, sandbox: ModuleType, tmp_path: Path, limits: Limits) -> dict[str, str]:
    """Run a placeholder artifact through the native executor with a fake sandbox; return its spec's environment."""
    fake = FakeSandbox(result=canned(sandbox))
    native.NativeExecutor(sandbox=fake).run(make_artifact(tmp_path), ["--size", "8"], limits)
    assert len(fake.specs) == 1
    environment = fake.specs[0].environment
    assert environment is not None, "the native executor must set the program's environment (the thread bound)"
    return dict(environment)


@pytest.mark.parametrize("cpus", [1, 2, 16, 256])
def test_the_program_environment_is_the_default_plus_the_thread_bound(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, cpus: int
) -> None:
    limits = Limits(wall_s=LIMITS.wall_s, memory_mb=LIMITS.memory_mb, cpus=cpus)
    assert spec_environment(native, sandbox, tmp_path, limits) == expected_environment(sandbox, cpus)


def test_nothing_from_the_callers_environment_enters_not_even_its_thread_count(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in CALLER_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "caller-tmp"))
    environment = spec_environment(native, sandbox, tmp_path, LIMITS)
    assert environment == expected_environment(sandbox, LIMITS.cpus)
    assert not [value for value in environment.values() if SECRET in value]


def test_the_spec_admits_the_thread_bound_and_keeps_it(sandbox: ModuleType, tmp_path: Path) -> None:
    workdir = make_artifact(tmp_path).parent
    environment = expected_environment(sandbox, 16)
    spec = sandbox.SandboxSpec(workdir=workdir, hidden_roots=(tmp_path / "scratch",), environment=environment)
    assert dict(spec.environment) == environment


def test_the_sandbox_command_gives_the_program_exactly_that_environment_before_its_argv(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in CALLER_ENV.items():
        monkeypatch.setenv(name, value)
    seen: list[list[str]] = []

    def trap(argv: Sequence[str], *args: object, **kwargs: object) -> None:
        """Record the argv and refuse to start anything, as if the first tool were missing."""
        seen.append(list(argv))
        raise FileNotFoundError(2, "No such file or directory", str(argv[0]))

    monkeypatch.setattr(subprocess, "Popen", trap)
    artifact = make_artifact(tmp_path)
    argv = [str(artifact), "--size", "8"]
    with pytest.raises(sandbox.SandboxUnavailableError):
        native.NativeExecutor().run(artifact, argv[1:], LIMITS)
    assert len(seen) == 1
    command = seen[0]
    assert command[:3] == ["prlimit", "--core=1", "--"], command[:6]
    assert sandbox.SETUP_SCRIPT in command, "the setup script stays the one constant"
    environment = expected_environment(sandbox, LIMITS.cpus)
    elements = [f"{name}={environment[name]}" for name in sorted(environment)]
    tail = [sandbox.ENVIRONMENT_MARKER, *elements, *argv]
    assert command[-len(tail) :] == tail, command[-len(tail) :]
    assert not [part for part in command if SECRET in part or "/caller/" in part]


@pytest.mark.parametrize(
    ("runs_name", "artifact_name"),
    [("runs=root", "main"), ("runs", "main=1")],
    ids=["equals-in-runs-root", "equals-in-artifact-name"],
)
def test_an_artifact_path_that_holds_an_equals_sign_raises_before_anything_starts(
    native: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runs_name: str,
    artifact_name: str,
) -> None:
    # env would read such a path as a variable, so the sandbox refuses it (DEMO.2: every native run sets one).
    runs = tmp_path / runs_name
    runs.mkdir(exist_ok=True)
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(runs))
    workdir = runs / "attempt00" / "build"
    workdir.mkdir(parents=True)
    artifact = workdir / artifact_name
    artifact.write_bytes(b"placeholder artifact, never executed\n")
    started: list[list[str]] = []

    def trap(argv: Sequence[str], *args: object, **kwargs: object) -> None:
        """Record the argv and fail the test: nothing may start."""
        started.append(list(argv))
        raise AssertionError("subprocess.Popen was called for an artifact path that holds '='")

    monkeypatch.setattr(subprocess, "Popen", trap)
    with pytest.raises(ValueError, match="may not hold '='"):
        native.NativeExecutor().run(artifact, ["--size", "8"], LIMITS)
    assert started == []


def setup_script_program_defaults(sandbox: ModuleType) -> dict[str, str]:
    """Return the program defaults of SETUP_SCRIPT's own `env -i` line (the P0.16 command), as NAME -> value.

    The line's PATH="$PATH" is the setup's PATH, which the outer `env -i`
    sets to SANDBOX_PATH (the sandbox module docstring), so it is returned
    as SANDBOX_PATH; HOME="$workdir" is returned as it is written.
    """
    lines = [line for line in sandbox.SETUP_SCRIPT.splitlines() if line.startswith("unshare ") and " env -i " in line]
    assert len(lines) == 1, lines
    defaults: dict[str, str] = {}
    for word in lines[0].split(" env -i ", 1)[1].split():
        if word == "\\":
            break
        name, _, value = word.partition("=")
        defaults[name] = value.strip('"')
    assert defaults.pop("PATH") == "$PATH", lines[0]
    return {"PATH": sandbox.SANDBOX_PATH, **defaults}


def test_the_fixed_part_is_the_setup_scripts_own_program_defaults_without_home(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path
) -> None:
    defaults = setup_script_program_defaults(sandbox)
    assert defaults.pop("HOME") == "$workdir", "the default gives HOME=<workdir>; a native run has no HOME"
    environment = spec_environment(native, sandbox, tmp_path, LIMITS)
    assert environment.pop("OMP_NUM_THREADS") == str(LIMITS.cpus)
    assert environment == defaults


def test_run_documents_the_thread_bound_and_the_home_change(native: ModuleType) -> None:
    # The run docstring, not the module's: the module docstring already names $HOME as a hidden root.
    doc = " ".join((inspect.getdoc(native.NativeExecutor.run) or "").split())
    assert "OMP_NUM_THREADS" in doc, "run documents the thread bound"
    assert "cpus" in doc, "and where the bound comes from (Limits.cpus)"
    assert "HOME" in doc, "and the one default variable a bounded run no longer gets"
