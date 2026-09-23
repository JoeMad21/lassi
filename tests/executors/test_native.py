"""Tests for the native executor, registered as Executor "native" (P0.10).

The native executor runs a CPU artifact (bible Execution Backends, executor
native) and, like every executor, runs it only through the sandbox module
(Agent Rule 6): the workdir is the artifact's directory, the argv is the
artifact then its inputs, and the read-only roots default to $LASSI_SCRATCH
and $HOME. It reports the files the run created as output files (bible
Component Interfaces, Executor contract rules).

Every test uses a fake Sandbox, a Sandbox with a fake runner, or a trapped
subprocess.Popen, so no sandbox and no generated code ever starts. The
artifacts are placeholder files that are never executed. No value in this
module is a measurement.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

from lassi.core import registry
from lassi.core.interfaces import Limits, RunResult
from lassi.toolchains import CommandResult

LIMITS = Limits(wall_s=2.5, memory_mb=256, cpus=2)
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
PROJECT_NAMES = ("lassi-repro", "lassi-ee", "lassi-df", "hecbench", "qwen", "wizardcoder", "a100", "mi300x")


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
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make tmp_path the runs root ($LASSI_RUNS_ROOT), where the artifacts of these tests live."""
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(tmp_path))
    return tmp_path


@dataclass(frozen=True)
class SandboxCall:
    """One call a fake Sandbox received."""

    spec: object
    argv: list[str]
    limits: Limits


@dataclass
class FakeSandbox:
    """A stand-in for lassi.executors.sandbox.Sandbox that records each run and returns a canned SandboxResult.

    `effect`, when set, runs against the spec during the call, the way a
    program would write files into its workdir. Nothing is executed.
    """

    result: object
    effect: Callable[[object], None] | None = None
    calls: list[SandboxCall] = field(default_factory=list)

    def run(self, spec: object, argv: Sequence[str], limits: Limits) -> object:
        """Record the call, apply the effect, and return the canned result."""
        self.calls.append(SandboxCall(spec=spec, argv=list(argv), limits=limits))
        if self.effect is not None:
            self.effect(spec)
        return self.result


@dataclass
class FakeRunner:
    """A CommandRunner that records each argv and returns exit status 0; it never starts a process."""

    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the argv and return an empty successful CommandResult."""
        self.calls.append(list(argv))
        return CommandResult(returncode=0, stdout="", stderr="")


def canned(sandbox: ModuleType, **overrides: object) -> object:
    """Return a SandboxResult for a normal exit with status 0, with any field overridden."""
    values: dict[str, object] = {"returncode": 0, "stdout": "", "stderr": "", "wall_s": 0.25}
    values.update({"hang": False, "killed": False})
    values.update(overrides)
    return sandbox.SandboxResult(**values)


def make_artifact(tmp_path: Path) -> Path:
    """Return a placeholder artifact file in a fresh build directory; it is never executed."""
    workdir = tmp_path / "attempt00" / "build"
    workdir.mkdir(parents=True)
    artifact = workdir / "main"
    artifact.write_bytes(b"placeholder artifact, never executed\n")
    return artifact


def set_roots(monkeypatch: pytest.MonkeyPatch, scratch: Path | None, home: Path | None) -> None:
    """Set or clear $LASSI_SCRATCH and $HOME for one test."""
    for name, value in (("LASSI_SCRATCH", scratch), ("HOME", home)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(value))


def public_defs(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    """Return public top-level classes and functions, plus the public methods of public classes."""
    found: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, *FUNCTION_NODES)) or node.name.startswith("_"):
            continue
        found.append((node.name, node))
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, FUNCTION_NODES) and not item.name.startswith("_"):
                    found.append((f"{node.name}.{item.name}", item))
    return found


def is_typed(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True when every parameter except self or cls, and the return value, are annotated."""
    args = node.args
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    params += [a for a in (args.vararg, args.kwarg) if a is not None]
    params = [p for p in params if p.arg not in ("self", "cls")]
    return node.returns is not None and all(p.annotation is not None for p in params)


# ---------------------------------------------------------------------------
# Registration and construction


def test_registered_as_executor_native(native: ModuleType) -> None:
    entry = registry.DEFAULT_REGISTRY.get("Executor", "native")
    assert entry.factory is native.NativeExecutor
    assert entry.capabilities == frozenset({"runs_code", "sandboxed"})
    assert entry.config_keys == frozenset({"harness"})
    assert native.NativeExecutor.name == "native"


def test_factory_works_with_no_arguments_and_no_environment(
    native: ModuleType, sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_roots(monkeypatch, None, None)
    executor = registry.DEFAULT_REGISTRY.get("Executor", "native").factory()
    assert executor.name == "native"
    assert isinstance(executor.sandbox, sandbox.Sandbox)


def test_signatures_match_the_contract(native: ModuleType) -> None:
    init = inspect.signature(native.NativeExecutor).parameters
    assert list(init) == ["harness", "readonly_roots", "sandbox"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None for p in init.values())
    assert list(inspect.signature(native.NativeExecutor.run).parameters) == ["self", "artifact", "inputs", "limits"]


def imports_native(tree: ast.Module) -> bool:
    """Return True when a package __init__ imports lassi.executors.native, in any absolute or relative spelling."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name == "lassi.executors.native" for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if (node.level, module) in ((0, "lassi.executors.native"), (1, "native")):
                return True
            if (node.level, module) in ((0, "lassi.executors"), (1, "")):
                if any(alias.name == "native" for alias in node.names):
                    return True
    return False


def test_package_imports_and_re_exports_native(native: ModuleType) -> None:
    import lassi.executors as package

    # Importing lassi.executors alone must register "native", so __init__ itself imports the module.
    assert imports_native(ast.parse(Path(package.__file__).read_text(encoding="ascii")))
    assert package.native is native
    assert package.NativeExecutor is native.NativeExecutor
    assert "NativeExecutor" in package.__all__


def test_module_is_documented_typed_ascii_and_names_no_project(native: ModuleType) -> None:
    raw = Path(native.__file__).read_bytes()
    assert raw.isascii()
    source = raw.decode("ascii")
    assert not [name for name in PROJECT_NAMES if name in source.lower()]
    tree = ast.parse(source)
    assert ast.get_docstring(tree)
    missing_doc = [name for name, node in public_defs(tree) if not ast.get_docstring(node)]
    assert not missing_doc, missing_doc
    untyped = [name for name, node in public_defs(tree) if isinstance(node, FUNCTION_NODES) and not is_typed(node)]
    assert not untyped, untyped


# ---------------------------------------------------------------------------
# What run hands to the sandbox


def test_run_hands_the_sandbox_the_workdir_argv_and_limits(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch, home = tmp_path / "scratch", tmp_path / "home"
    set_roots(monkeypatch, scratch, home)
    artifact = make_artifact(tmp_path)
    fake = FakeSandbox(result=canned(sandbox))
    harness = tmp_path / "assets" / "harness"
    inputs = ["--size", "8", "two words", "$HOME; `id`"]
    native.NativeExecutor(harness=str(harness), sandbox=fake).run(artifact, inputs, LIMITS)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert isinstance(call.spec, sandbox.SandboxSpec)
    assert call.spec.workdir == artifact.parent
    assert call.spec.harness == harness
    assert call.spec.readonly_roots == (scratch, home)
    assert call.argv == [str(artifact), *inputs]
    assert call.limits == LIMITS


def test_harness_defaults_to_none(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_roots(monkeypatch, tmp_path / "scratch", None)
    fake = FakeSandbox(result=canned(sandbox))
    native.NativeExecutor(sandbox=fake).run(make_artifact(tmp_path), [], LIMITS)
    assert fake.calls[0].spec.harness is None
    assert fake.calls[0].argv == [str(tmp_path / "attempt00" / "build" / "main")]


@pytest.mark.parametrize(
    ("scratch", "home", "expected"),
    [
        ("scratch", "home", ("scratch", "home")),
        ("scratch", None, ("scratch",)),
        (None, "home", ("home",)),
        ("same", "same", ("same",)),
    ],
    ids=["both", "scratch-only", "home-only", "deduplicated"],
)
def test_readonly_roots_default_to_scratch_then_home(
    native: ModuleType,
    sandbox: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scratch: str | None,
    home: str | None,
    expected: tuple[str, ...],
) -> None:
    set_roots(monkeypatch, None if scratch is None else tmp_path / scratch, None if home is None else tmp_path / home)
    fake = FakeSandbox(result=canned(sandbox))
    native.NativeExecutor(sandbox=fake).run(make_artifact(tmp_path), [], LIMITS)
    assert fake.calls[0].spec.readonly_roots == tuple(tmp_path / name for name in expected)


def test_explicit_readonly_roots_replace_the_environment(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_roots(monkeypatch, tmp_path / "scratch", tmp_path / "home")
    roots = [str(tmp_path / "ro-a"), str(tmp_path / "ro-b")]
    fake = FakeSandbox(result=canned(sandbox))
    native.NativeExecutor(readonly_roots=roots, sandbox=fake).run(make_artifact(tmp_path), [], LIMITS)
    assert fake.calls[0].spec.readonly_roots == (tmp_path / "ro-a", tmp_path / "ro-b")


def test_no_readonly_root_means_sandbox_unavailable_and_nothing_runs(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_roots(monkeypatch, None, None)
    artifact = make_artifact(tmp_path)
    fake = FakeSandbox(result=canned(sandbox))
    with pytest.raises(sandbox.SandboxUnavailableError):
        native.NativeExecutor(sandbox=fake).run(artifact, [], LIMITS)
    assert fake.calls == []
    runner = FakeRunner()
    with pytest.raises(sandbox.SandboxUnavailableError):
        native.NativeExecutor(sandbox=sandbox.Sandbox(runner=runner)).run(artifact, [], LIMITS)
    assert runner.calls == []


def test_a_relative_artifact_is_refused_before_anything_runs(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_roots(monkeypatch, tmp_path / "scratch", tmp_path / "home")
    fake = FakeSandbox(result=canned(sandbox))
    with pytest.raises(ValueError):
        native.NativeExecutor(sandbox=fake).run(Path("build") / "main", [], LIMITS)
    assert fake.calls == []


@pytest.mark.parametrize(
    ("scratch", "home"),
    [("relative/scratch", None), ("relative/scratch", "ABSOLUTE"), ("ABSOLUTE", "relative/home")],
    ids=["relative-scratch", "relative-scratch-absolute-home", "relative-home"],
)
def test_a_relative_root_variable_means_sandbox_unavailable_and_nothing_runs(
    native: ModuleType,
    sandbox: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scratch: str,
    home: str | None,
) -> None:
    for name, value in (("LASSI_SCRATCH", scratch), ("HOME", home)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(tmp_path / name.lower()) if value == "ABSOLUTE" else value)
    fake = FakeSandbox(result=canned(sandbox))
    with pytest.raises(sandbox.SandboxUnavailableError, match="absolute"):
        native.NativeExecutor(sandbox=fake).run(make_artifact(tmp_path), [], LIMITS)
    assert fake.calls == []


def test_an_artifact_outside_the_runs_root_is_refused_before_anything_runs(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_roots(monkeypatch, tmp_path / "scratch", tmp_path / "home")
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(tmp_path / "runs"))
    fake = FakeSandbox(result=canned(sandbox))
    with pytest.raises(ValueError, match="runs root"):
        native.NativeExecutor(sandbox=fake).run(make_artifact(tmp_path), [], LIMITS)
    with pytest.raises(ValueError, match="runs root"):
        native.NativeExecutor(sandbox=fake).run(tmp_path / "runs" / "main", [], LIMITS)
    assert fake.calls == []


@pytest.mark.parametrize("value", [None, "", "relative/runs"], ids=["unset", "empty", "relative"])
def test_no_usable_runs_root_is_refused_before_anything_runs(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    set_roots(monkeypatch, tmp_path / "scratch", tmp_path / "home")
    if value is None:
        monkeypatch.delenv("LASSI_RUNS_ROOT", raising=False)
    else:
        monkeypatch.setenv("LASSI_RUNS_ROOT", value)
    fake = FakeSandbox(result=canned(sandbox))
    with pytest.raises(ValueError, match="LASSI_RUNS_ROOT"):
        native.NativeExecutor(sandbox=fake).run(make_artifact(tmp_path), [], LIMITS)
    assert fake.calls == []


def test_a_workdir_linked_out_of_the_runs_root_is_refused(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_roots(monkeypatch, tmp_path / "scratch", tmp_path / "home")
    runs, outside = tmp_path / "runs", tmp_path / "outside"
    runs.mkdir()
    outside.mkdir()
    (outside / "main").write_bytes(b"placeholder artifact, never executed\n")
    try:
        os.symlink(outside, runs / "attempt00", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"cannot create a directory symlink here: {exc}")
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(runs))
    fake = FakeSandbox(result=canned(sandbox))
    with pytest.raises(ValueError, match="runs root"):
        native.NativeExecutor(sandbox=fake).run(runs / "attempt00" / "main", [], LIMITS)
    assert fake.calls == []


def test_default_sandbox_runs_through_the_sandbox_command_only(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    set_roots(monkeypatch, scratch, None)
    seen: list[list[str]] = []

    def trap(argv: Sequence[str], *args: object, **kwargs: object) -> None:
        """Record the argv and refuse to start anything, as if the first tool were missing."""
        seen.append(list(argv))
        raise FileNotFoundError(2, "No such file or directory", str(argv[0]))

    monkeypatch.setattr(subprocess, "Popen", trap)
    artifact = make_artifact(tmp_path)
    with pytest.raises(sandbox.SandboxUnavailableError):
        native.NativeExecutor().run(artifact, ["x"], LIMITS)
    expected_spec = sandbox.SandboxSpec(workdir=artifact.parent, readonly_roots=(scratch,))
    assert seen == [sandbox.sandbox_command(expected_spec, [str(artifact), "x"], LIMITS)]


# ---------------------------------------------------------------------------
# What run returns


def test_output_files_are_only_the_new_files(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_roots(monkeypatch, tmp_path / "scratch", tmp_path / "home")
    artifact = make_artifact(tmp_path)
    workdir = artifact.parent
    (workdir / "sub").mkdir()
    (workdir / "sub" / "old.txt").write_text("old\n", encoding="ascii")
    (workdir / "input.bin").write_bytes(b"\x00\x01")

    def program_writes(spec: object) -> None:
        """Write new files, rewrite an old one, and make an empty directory, the way a program run might."""
        (workdir / "out.bin").write_bytes(b"\x02")
        (workdir / "sub" / "new.txt").write_text("new\n", encoding="ascii")
        (workdir / "deep" / "er").mkdir(parents=True)
        (workdir / "deep" / "er" / "x.dat").write_bytes(b"x")
        (workdir / "empty-dir").mkdir()
        (workdir / "input.bin").write_bytes(b"\x03\x04\x05")

    fake = FakeSandbox(result=canned(sandbox), effect=program_writes)
    result = native.NativeExecutor(sandbox=fake).run(artifact, [], LIMITS)
    assert dict(result.output_files) == {
        "out.bin": workdir / "out.bin",
        "sub/new.txt": workdir / "sub" / "new.txt",
        "deep/er/x.dat": workdir / "deep" / "er" / "x.dat",
    }
    assert all(path.is_absolute() for path in result.output_files.values())


def test_symbolic_links_the_run_creates_are_not_output_files(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_roots(monkeypatch, tmp_path / "scratch", tmp_path / "home")
    artifact = make_artifact(tmp_path)
    workdir = artifact.parent
    secret = tmp_path / "host-secret.txt"
    secret.write_text("secret\n", encoding="ascii")
    try:
        os.symlink(secret, tmp_path / "probe-link")
    except OSError as exc:
        pytest.skip(f"cannot create a symlink here: {exc}")

    def program_links(spec: object) -> None:
        """Link a host file outside the workdir into it and write one regular file, the way a program might."""
        os.symlink(secret, workdir / "leak")
        (workdir / "out.txt").write_text("out\n", encoding="ascii")

    fake = FakeSandbox(result=canned(sandbox), effect=program_links)
    result = native.NativeExecutor(sandbox=fake).run(artifact, [], LIMITS)
    assert dict(result.output_files) == {"out.txt": workdir / "out.txt"}


def test_no_new_file_means_no_output_files(
    native: ModuleType, sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_roots(monkeypatch, tmp_path / "scratch", tmp_path / "home")
    result = native.NativeExecutor(sandbox=FakeSandbox(result=canned(sandbox))).run(make_artifact(tmp_path), [], LIMITS)
    assert dict(result.output_files) == {}


@pytest.mark.parametrize(
    ("returncode", "hang", "killed"),
    [(0, False, False), (3, False, False), (124, True, False), (137, False, True), (-1, True, False)],
)
def test_run_result_fields_map_from_the_sandbox_result(
    native: ModuleType,
    sandbox: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    hang: bool,
    killed: bool,
) -> None:
    set_roots(monkeypatch, tmp_path / "scratch", tmp_path / "home")
    outcome = canned(sandbox, returncode=returncode, stdout="out\n", stderr="err\n", wall_s=1.25)
    outcome = dataclasses.replace(outcome, hang=hang, killed=killed)
    result = native.NativeExecutor(sandbox=FakeSandbox(result=outcome)).run(make_artifact(tmp_path), [], LIMITS)
    assert result == RunResult(
        exit_code=returncode, hang=hang, stdout="out\n", stderr="err\n", output_files={}, wall_s=1.25
    )
