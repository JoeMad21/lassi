"""Tests for the sandbox module lassi.executors.sandbox (P0.10).

The sandbox module is the only module that runs generated code (Agent Rule 6;
bible Sandbox, Component Interfaces Executor contract rules). Its mechanism
comes from plans/spikes/p0-sandbox.md and both of its addenda: systemd-run
--user --scope for the memory, task, and backstop limits, wrapping unshare
-rinmpfu --mount-proc for isolation, a constant setup script that remounts
everything read-only except the per-trial workdir and forbids nested user
namespaces, and nice -n 19, prlimit --cpu, and an innermost timeout
--kill-after for CPU priority, CPU time, and wall time.

Every test that runs something uses a fake CommandRunner (or a trapped
subprocess.Popen), so no sandbox and no generated code ever starts. The
setup-script tests run SETUP_SCRIPT under sh with stub commands on a PATH
that holds only the stubs, and with the script's write to
/proc/sys/user/max_user_namespaces pointed at a file under tmp_path: nothing
is mounted, no limit is written (not even the sandbox user namespace's), and the program argv is only
recorded, never run. The paths are made up and never touched unless a test
creates them under tmp_path. No value in this module is a measurement: the
limits are sample inputs, and the classify table is the P0.10 contract's
mapping, which reads the exit statuses the spike recorded (timeout 124, a
limit kill 137, the backstop 143 or 137, the runner's own timeout -1)
together with the elapsed wall time.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import inspect
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import ModuleType

import pytest

from lassi.core.interfaces import Limits
from lassi.toolchains import CommandResult

ROOT = Path(Path(__file__).resolve().anchor)
SCRATCH = ROOT / "scratch"
HOME = ROOT / "home" / "user"
WORK = SCRATCH / "runs" / "trial" / "attempt00" / "build"
HARNESS = SCRATCH / "repo" / "assets" / "harness"
PROGRAM = ["./main", "--size", "8"]
ONE_SECOND = Limits(wall_s=1.0, memory_mb=64, cpus=1)
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
PROJECT_NAMES = ("lassi-repro", "lassi-ee", "lassi-df", "hecbench", "qwen", "wizardcoder", "a100", "mi300x")
# The executor modules the P0.10 contract names; any other module found in the package is scanned too.
NAMED_EXECUTOR_MODULES = ("__init__", "native", "none", "sandbox", "workdir")
RUNNER_NAMES = ("CommandResult", "CommandRunner", "subprocess_runner")
# The only modules in lassi/ that may start a process, and why.
PROCESS_MODULES = {
    "lassi.executors.sandbox": "runs generated code, only inside the sandbox",
    "lassi.toolchains._base": "defines subprocess_runner, which runs compilers, never generated code",
    "lassi.toolchains": "re-exports subprocess_runner",
}
# The exact statement that ends SETUP_SCRIPT, with its line continuations joined.
EXEC_STATEMENT = (
    'exec env -i PATH="$PATH" HOME="$workdir" LANG=C.UTF-8 TMPDIR=/tmp'
    " nice -n 19"
    " setpriv --no-new-privs --inh-caps=-all --bounding-set=-all --"
    ' prlimit --cpu="$cpu" --core=0 --'
    ' timeout --kill-after="$kill_after" "$wall" "$@"'
)
# The sandbox user namespace's limit, which SETUP_SCRIPT sets to 0 while it still holds capabilities (the host
# value is kept per user namespace and never changes), and the exact line that does it.
USERNS_LIMIT = "/proc/sys/user/max_user_namespaces"
USERNS_LINE = f"echo 0 > {USERNS_LIMIT}"


@pytest.fixture
def sandbox() -> ModuleType:
    """Return the lassi.executors.sandbox module (imported here so each test shows a missing module clearly)."""
    from lassi.executors import sandbox

    return sandbox


@dataclass
class Clock:
    """A fake time.monotonic: it stands still until a fake runner advances it."""

    now: float = 1000.0

    def __call__(self) -> float:
        """Return the current fake time in seconds."""
        return self.now


@dataclass(frozen=True)
class Call:
    """One call a fake runner received."""

    argv: list[str]
    cwd: Path
    timeout_s: float


@dataclass
class FakeRunner:
    """A CommandRunner that records each call and returns a canned result; it never starts a process.

    With `clock` set, each call advances the fake clock by `elapsed_s`, so the
    sandbox measures exactly that wall time. With `error` set, each call
    raises it instead of returning. With `ready` True (the default) the
    returned stderr starts with the setup script's ready line, as it does
    whenever the sandbox reached the program; False stands for a setup that
    failed, and `stderr` is then all there is.
    """

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    elapsed_s: float = 0.0
    clock: Clock | None = None
    error: BaseException | None = None
    ready: bool = True
    calls: list[Call] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the call, advance the clock, and return the canned CommandResult or raise the canned error."""
        from lassi.executors.sandbox import READY_MARKER

        self.calls.append(Call(argv=list(argv), cwd=Path(cwd), timeout_s=timeout_s))
        if self.clock is not None:
            self.clock.now += self.elapsed_s
        if self.error is not None:
            raise self.error
        stderr = READY_MARKER + "\n" + self.stderr if self.ready else self.stderr
        return CommandResult(returncode=self.returncode, stdout=self.stdout, stderr=stderr)


def install_clock(monkeypatch: pytest.MonkeyPatch, sandbox: ModuleType) -> Clock:
    """Replace time.monotonic (and a monotonic imported by name into the module) with a fake clock."""
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    if hasattr(sandbox, "monotonic"):
        monkeypatch.setattr(sandbox, "monotonic", clock)
    return clock


def trap_popen(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Replace subprocess.Popen with a trap that records argv and raises FileNotFoundError; nothing starts."""
    seen: list[list[str]] = []

    def trap(argv: Sequence[str], *args: object, **kwargs: object) -> None:
        """Record the argv and refuse to start anything, as if the first tool were missing."""
        seen.append(list(argv))
        raise FileNotFoundError(2, "No such file or directory", str(argv[0]))

    monkeypatch.setattr(subprocess, "Popen", trap)
    return seen


def sample_spec(sandbox: ModuleType, **overrides: object) -> object:
    """Return a SandboxSpec with the sample workdir, the scratch and home roots, and the harness."""
    values: dict[str, object] = {"workdir": WORK, "readonly_roots": (SCRATCH, HOME), "harness": HARNESS}
    values.update(overrides)
    return sandbox.SandboxSpec(**values)


def script_of(command: list[str]) -> str:
    """Return the script element of a sandbox command: the element after "sh", "-c"."""
    start = command.index("sh")
    assert command[start + 1] == "-c", command
    return command[start + 2]


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


def is_os_process_call(name: str) -> bool:
    """Return True for the os functions that start or replace a process: system, popen, exec*, spawn*, fork*."""
    return name in ("system", "popen", "posix_spawn", "posix_spawnp") or name.startswith(("exec", "spawn", "fork"))


def process_references(tree: ast.Module) -> list[str]:
    """Return every way a module could start a process: subprocess, subprocess_runner, Popen, os.exec*, and so on.

    Docstrings and comments are not code, so a module may mention these names
    in prose; only imports, names, and attributes count.
    """
    found: list[str] = []
    process_names = ("subprocess_runner", "Popen", "create_subprocess_exec", "create_subprocess_shell")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if alias.name.split(".")[0] in ("subprocess", "pty")]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [alias.name for alias in node.names]
            if node.module.split(".")[0] in ("subprocess", "pty"):
                found.append(node.module)
            found += [name for name in names if name in process_names]
            if node.module == "os":
                found += [f"os.{name}" for name in names if is_os_process_call(name)]
        elif isinstance(node, ast.Name) and node.id in process_names:
            found.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in process_names:
            found.append(node.attr)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
            if is_os_process_call(node.attr):
                found.append(f"os.{node.attr}")
    return found


def top_level_names(tree: ast.Module) -> set[str]:
    """Return the names a module defines at top level: classes, functions, and assignment targets."""
    names = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, *FUNCTION_NODES))}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        names |= {target.id for target in targets if isinstance(target, ast.Name)}
    return names


@dataclass(frozen=True)
class StubRun:
    """What SETUP_SCRIPT did under stub commands: its status, its stderr, each mount call, and the exec argv.

    `exec_argv` is the argument list the script's final `exec env` received,
    or None when the script never got that far. `userns_limit` is what the
    script wrote in place of /proc/sys/user/max_user_namespaces, or None
    when it wrote nothing there.
    """

    returncode: int
    stderr: str
    mounts: list[list[str]]
    exec_argv: list[str] | None
    userns_limit: str | None


def read_records(path: Path) -> list[list[str]]:
    """Return the records a stub wrote: one per line, fields separated by the unit separator (octal 037)."""
    if not path.exists():
        return []
    text = path.read_bytes().decode("ascii")
    return [line.split("\x1f")[:-1] for line in text.split("\n") if line]


def write_stub(directory: Path, name: str, body: str) -> None:
    """Write an executable /bin/sh stub named `name` into `directory`."""
    path = directory / name
    path.write_text("#!/bin/sh\n" + body, encoding="ascii", newline="\n")
    path.chmod(0o755)


def stub_script(sandbox: ModuleType, userns_target: Path) -> str:
    """Return SETUP_SCRIPT with its one write to /proc/sys/user/max_user_namespaces pointed at `userns_target`."""
    assert sandbox.SETUP_SCRIPT.count(USERNS_LIMIT) == 1
    assert "'" not in str(userns_target)
    return sandbox.SETUP_SCRIPT.replace(USERNS_LIMIT, f"'{userns_target.as_posix()}'")


def run_setup_with_stubs(
    sandbox: ModuleType,
    tmp_path: Path,
    layout: list[str],
    *,
    fail_remount_of: str = "",
    missing: str = "",
    userns_target: Path | None = None,
) -> StubRun:
    """Run SETUP_SCRIPT under sh with only stub commands on PATH; nothing is mounted and no program runs.

    The stub `mount` records its arguments and succeeds, except that it exits
    32 on `-o remount,bind,ro <fail_remount_of>`. The stub `env` records its
    arguments and exits 0 without running them. The stubs nice, setpriv,
    prlimit, and timeout exist only for the script's PATH check and exit 99
    if called. `missing` names a stub to leave out. The script's write to
    /proc/sys/user/max_user_namespaces goes to `userns_target` (by default a
    file under tmp_path), so no real limit is ever written. PATH holds
    nothing but the stubs, so no real mount can ever run; the test skips as
    root, and when sh is absent.
    """
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh is not on PATH")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("never run the setup script as root, even with stubs")
    assert not re.search(r"/s?bin/", sandbox.SETUP_SCRIPT), "the script must find every command through PATH"
    target = tmp_path / "max_user_namespaces" if userns_target is None else userns_target
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    mount_log, env_log = tmp_path / "mount.log", tmp_path / "env.log"
    assert "'" not in f"{mount_log}{env_log}{fail_remount_of}"
    fail = f'[ "$1 $2 $3" = \'-o remount,bind,ro {fail_remount_of}\' ] && exit 32\n' if fail_remount_of else ""
    record = "printf '%s\\037' \"$@\" >> '{log}'\nprintf '\\n' >> '{log}'\n"
    write_stub(stubs, "mount", record.format(log=mount_log.as_posix()) + fail + "exit 0\n")
    write_stub(stubs, "env", record.format(log=env_log.as_posix()) + "exit 0\n")
    for name in ("nice", "setpriv", "prlimit", "timeout"):
        write_stub(stubs, name, "exit 99\n")
    if missing:
        (stubs / missing).unlink()
    argv = [shell, "-c", stub_script(sandbox, target), "sh", *layout]
    done = subprocess.run(argv, env={"PATH": str(stubs)}, capture_output=True, timeout=60, check=False)
    exec_records = read_records(env_log)
    return StubRun(
        returncode=done.returncode,
        stderr=done.stderr.decode("utf-8", errors="replace"),
        mounts=read_records(mount_log),
        exec_argv=exec_records[0] if exec_records else None,
        userns_limit=target.read_text(encoding="ascii") if target.is_file() else None,
    )


def stub_layout(tmp_path: Path) -> tuple[list[str], dict[str, str]]:
    """Return a positional layout for SETUP_SCRIPT with two roots and a harness, and its named values.

    Only the workdir exists (the script cds into it); the stub mount never
    touches the other paths. The limits are cpu 6, wall 3, kill-after 2.
    """
    workdir = tmp_path / "work"
    workdir.mkdir()
    names = {"workdir": workdir, "harness": tmp_path / "harness", "root_a": tmp_path / "ro-a"}
    names["root_b"] = tmp_path / "ro-b"
    values = {key: path.as_posix() for key, path in names.items()}
    layout = [values["workdir"], values["harness"], "2", values["root_a"], values["root_b"], "6", "3", "2"]
    return [*layout, "./main", "one two"], values


# ---------------------------------------------------------------------------
# Constants, dataclasses, and the module itself


def test_constants_match_the_contract(sandbox: ModuleType) -> None:
    assert sandbox.KILL_AFTER_S == 2
    assert sandbox.OUTER_MARGIN_S == 5
    assert isinstance(sandbox.SETUP_SCRIPT, str) and sandbox.SETUP_SCRIPT.strip()


def test_sandbox_spec_is_frozen_with_defaults(sandbox: ModuleType) -> None:
    spec = sandbox.SandboxSpec(workdir=WORK, readonly_roots=(SCRATCH,))
    assert dataclasses.is_dataclass(spec)
    assert spec.harness is None
    assert spec.tasks_max == 256
    assert spec.readonly_roots == (SCRATCH,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.workdir = HOME  # type: ignore[misc]


def test_sandbox_result_fields_and_error_type(sandbox: ModuleType) -> None:
    names = [f.name for f in dataclasses.fields(sandbox.SandboxResult)]
    assert names == ["returncode", "stdout", "stderr", "wall_s", "hang", "killed"]
    result = sandbox.SandboxResult(returncode=0, stdout="o", stderr="e", wall_s=0.5, hang=False, killed=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.hang = True  # type: ignore[misc]
    assert issubclass(sandbox.SandboxUnavailableError, RuntimeError)


def test_module_docstring_states_the_mechanism_and_cites_the_spike(sandbox: ModuleType) -> None:
    doc = sandbox.__doc__ or ""
    assert "plans/spikes/p0-sandbox.md" in doc
    assert "systemd-run" in doc
    assert "unshare" in doc


def test_module_is_documented_typed_ascii_and_names_no_project(sandbox: ModuleType) -> None:
    raw = Path(sandbox.__file__).read_bytes()
    assert raw.isascii()
    source = raw.decode("ascii")
    assert not [name for name in PROJECT_NAMES if name in source.lower()]
    tree = ast.parse(source)
    assert ast.get_docstring(tree)
    missing_doc = [name for name, node in public_defs(tree) if not ast.get_docstring(node)]
    assert not missing_doc, missing_doc
    untyped = [name for name, node in public_defs(tree) if isinstance(node, FUNCTION_NODES) and not is_typed(node)]
    assert not untyped, untyped


def test_reuses_the_toolchains_runner_contract_without_copying_it(sandbox: ModuleType) -> None:
    tree = ast.parse(Path(sandbox.__file__).read_text(encoding="ascii"))
    assert not top_level_names(tree) & set(RUNNER_NAMES)
    assert "subprocess_runner" in process_references(tree)


def test_package_re_exports_the_sandbox_names(sandbox: ModuleType) -> None:
    import lassi.executors as package

    for name in ("Sandbox", "SandboxSpec", "SandboxResult", "SandboxUnavailableError", "sandbox_command", "classify"):
        assert getattr(package, name) is getattr(sandbox, name), name
        assert name in package.__all__, name


# ---------------------------------------------------------------------------
# sandbox_command: the exact argv


@pytest.mark.parametrize(
    ("overrides", "limits", "properties", "layout"),
    [
        (
            {},
            Limits(wall_s=2.5, memory_mb=512, cpus=2),
            ["MemoryMax=512M", "MemorySwapMax=0", "TasksMax=256", "RuntimeMaxSec=10", "TimeoutStopSec=1"],
            [str(WORK), str(HARNESS), "2", str(SCRATCH), str(HOME), "5", "3", "2"],
        ),
        (
            {"readonly_roots": (SCRATCH,), "harness": None, "tasks_max": 64},
            Limits(wall_s=0.5, memory_mb=1, cpus=3),
            ["MemoryMax=1M", "MemorySwapMax=0", "TasksMax=64", "RuntimeMaxSec=8", "TimeoutStopSec=1"],
            [str(WORK), "", "1", str(SCRATCH), "2", "1", "2"],
        ),
        (
            {"readonly_roots": (HOME, SCRATCH)},
            Limits(wall_s=10.0, memory_mb=2048, cpus=4),
            ["MemoryMax=2048M", "MemorySwapMax=0", "TasksMax=256", "RuntimeMaxSec=17", "TimeoutStopSec=1"],
            [str(WORK), str(HARNESS), "2", str(HOME), str(SCRATCH), "40", "10", "2"],
        ),
    ],
    ids=["ceil-wall-and-cpu", "sub-second-no-harness", "whole-seconds-roots-in-order"],
)
def test_sandbox_command_is_exact(
    sandbox: ModuleType, overrides: dict[str, object], limits: Limits, properties: list[str], layout: list[str]
) -> None:
    command = sandbox.sandbox_command(sample_spec(sandbox, **overrides), PROGRAM, limits)
    expected = ["systemd-run", "--user", "--scope", "--quiet", *[part for p in properties for part in ("-p", p)]]
    expected += ["unshare", "-rinmpfu", "--mount-proc", "sh", "-c", sandbox.SETUP_SCRIPT, "sh", *layout, *PROGRAM]
    assert isinstance(command, list)
    assert command == expected
    assert all(isinstance(part, str) for part in command)


def test_sandbox_command_keeps_the_program_argv_verbatim(sandbox: ModuleType) -> None:
    program = ("python3", "-c", "print('hi')", "", "--flag=a b")
    command = sandbox.sandbox_command(sample_spec(sandbox), program, ONE_SECOND)
    assert command[-len(program) :] == list(program)


# ---------------------------------------------------------------------------
# SETUP_SCRIPT: constant, positional, and in the contract's order


def test_setup_script_never_substitutes_commands(sandbox: ModuleType) -> None:
    assert "$(" not in sandbox.SETUP_SCRIPT
    assert "`" not in sandbox.SETUP_SCRIPT


def test_setup_script_does_the_steps_in_order(sandbox: ModuleType) -> None:
    script = sandbox.SETUP_SCRIPT
    steps = [
        "set -eu",
        "for tool in env nice setpriv prlimit timeout; do",
        'mount --bind "$workdir" "$workdir"',
        'mount --rbind "$1" "$1"',
        'mount -o remount,bind,ro "$1"',
        'mount --bind "$harness" "$harness"',
        'mount -o remount,bind,ro "$harness"',
        "mount -o remount,bind,ro,nosuid,nodev /\n",
        "mount --rbind /sys/fs/cgroup /sys/fs/cgroup",
        "mount -o remount,bind,ro,nosuid,nodev,noexec /sys/fs/cgroup",
        f"for dir in {' '.join(sandbox.PRIVATE_DIRS)}; do",
        'mount -t tmpfs -o size=64m,mode=1777 tmpfs "$dir"',
        USERNS_LINE + "\n",
        'cd "$workdir"',
        f"echo {sandbox.READY_MARKER} >&2",
        "exec env -i ",
        "nice -n 19 ",
        "setpriv --no-new-privs",
        "prlimit --cpu=",
        "timeout --kill-after=",
    ]
    positions = [script.find(step) for step in steps]
    assert -1 not in positions, dict(zip(steps, positions, strict=True))
    assert positions == sorted(positions), dict(zip(steps, positions, strict=True))
    for needed in ("size=64m,mode=1777", "/tmp", "/var/tmp", "/dev/shm", "/run"):
        assert needed in script, needed
    tail = script[script.find("exec ") :]
    assert 0 < tail.find(" -- ") < tail.find("timeout --kill-after=") < tail.find('"$@"'), tail


def test_setup_script_ends_by_printing_ready_then_exec_under_env_nice_setpriv_prlimit_timeout(
    sandbox: ModuleType,
) -> None:
    lines = sandbox.SETUP_SCRIPT.rstrip("\n").replace("\\\n", "").split("\n")
    assert lines[-2] == f"echo {sandbox.READY_MARKER} >&2", lines[-2:]
    assert re.sub(" +", " ", lines[-1]) == EXEC_STATEMENT, lines[-1]
    assert sandbox.SETUP_SCRIPT.count(sandbox.READY_MARKER) == 1
    assert sandbox.PRIVATE_DIRS == ("/tmp", "/var/tmp", "/dev/shm", "/run")


def test_setup_script_forbids_nested_user_namespaces_while_it_holds_capabilities(sandbox: ModuleType) -> None:
    # The write needs the capabilities setpriv drops, so it must come after the tmpfs mounts and before the marker.
    lines = sandbox.SETUP_SCRIPT.split("\n")
    assert lines.count(USERNS_LINE) == 1, lines
    assert sandbox.SETUP_SCRIPT.count(USERNS_LIMIT) == 1
    position = lines.index(USERNS_LINE)
    last_tmpfs = max(index for index, line in enumerate(lines) if "mount -t tmpfs" in line)
    ready = lines.index(f"echo {sandbox.READY_MARKER} >&2")
    exec_line = next(index for index, line in enumerate(lines) if line.startswith("exec "))
    assert last_tmpfs < position < ready < exec_line, (last_tmpfs, position, ready, exec_line)


def test_setup_script_under_stubs_mounts_in_order_and_execs_the_program_last(
    sandbox: ModuleType, tmp_path: Path
) -> None:
    layout, value = stub_layout(tmp_path)
    run = run_setup_with_stubs(sandbox, tmp_path, layout)
    assert (run.returncode, run.stderr) == (0, sandbox.READY_MARKER + "\n"), run
    workdir, harness, root_a, root_b = value["workdir"], value["harness"], value["root_a"], value["root_b"]
    assert run.mounts[:8] == [
        ["--bind", workdir, workdir],
        ["--rbind", root_a, root_a],
        ["-o", "remount,bind,ro", root_a],
        ["--rbind", root_b, root_b],
        ["-o", "remount,bind,ro", root_b],
        ["--bind", harness, harness],
        ["-o", "remount,bind,ro", harness],
        ["-o", "remount,bind,ro,nosuid,nodev", "/"],
    ], run.mounts
    rest = run.mounts[8:]
    if rest[:1] == [["--rbind", "/sys/fs/cgroup", "/sys/fs/cgroup"]]:
        assert rest[1] == ["-o", "remount,bind,ro,nosuid,nodev,noexec", "/sys/fs/cgroup"], rest
        rest = rest[2:]
    private = [record[-1] for record in rest]
    assert rest == [["-t", "tmpfs", "-o", "size=64m,mode=1777", "tmpfs", name] for name in private], rest
    assert private == [name for name in sandbox.PRIVATE_DIRS if name in private], rest
    assert run.userns_limit == "0\n", run
    assert run.exec_argv is not None and run.exec_argv[1].startswith("PATH="), run.exec_argv
    assert run.exec_argv[:1] + run.exec_argv[2:] == [
        "-i", f"HOME={workdir}", "LANG=C.UTF-8", "TMPDIR=/tmp",
        "nice", "-n", "19",
        "setpriv", "--no-new-privs", "--inh-caps=-all", "--bounding-set=-all", "--",
        "prlimit", "--cpu=6", "--core=0", "--",
        "timeout", "--kill-after=2", "3", "./main", "one two",
    ]


def test_setup_script_stops_before_the_program_when_a_remount_fails(sandbox: ModuleType, tmp_path: Path) -> None:
    layout, value = stub_layout(tmp_path)
    run = run_setup_with_stubs(sandbox, tmp_path, layout, fail_remount_of=value["root_b"])
    assert run.returncode == 32, run
    assert sandbox.READY_MARKER not in run.stderr, run
    assert (run.exec_argv, run.userns_limit) == (None, None), run
    assert run.mounts[-1] == ["-o", "remount,bind,ro", value["root_b"]], run.mounts
    assert len(run.mounts) == 5, run.mounts


def test_setup_script_stops_before_the_program_when_the_user_namespace_limit_cannot_be_written(
    sandbox: ModuleType, tmp_path: Path
) -> None:
    layout, _value = stub_layout(tmp_path)
    unwritable = tmp_path / "no-such-dir" / "max_user_namespaces"
    run = run_setup_with_stubs(sandbox, tmp_path, layout, userns_target=unwritable)
    assert run.returncode != 0, run
    assert sandbox.READY_MARKER not in run.stderr, run
    assert (run.exec_argv, run.userns_limit) == (None, None), run
    assert ["-o", "remount,bind,ro,nosuid,nodev", "/"] in run.mounts, run.mounts


@pytest.mark.parametrize("missing", ["env", "nice", "setpriv", "prlimit", "timeout"])
def test_setup_script_stops_before_mounting_when_a_tool_is_missing(
    sandbox: ModuleType, tmp_path: Path, missing: str
) -> None:
    layout, _value = stub_layout(tmp_path)
    run = run_setup_with_stubs(sandbox, tmp_path, layout, missing=missing)
    assert run.returncode != 0, run
    assert sandbox.READY_MARKER not in run.stderr, run
    assert (run.mounts, run.exec_argv, run.userns_limit) == ([], None, None), run


def test_setup_script_is_the_same_for_every_spec(sandbox: ModuleType) -> None:
    first = sandbox.sandbox_command(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    other = sample_spec(sandbox, workdir=HOME / "w", readonly_roots=(SCRATCH,), harness=None, tasks_max=8)
    second = sandbox.sandbox_command(other, ["other"], Limits(wall_s=99.0, memory_mb=4096, cpus=8))
    assert script_of(first) == script_of(second) == sandbox.SETUP_SCRIPT


def test_hostile_paths_and_arguments_appear_only_as_separate_argv_elements(sandbox: ModuleType) -> None:
    base = ROOT / "sbx base"
    workdir = base / "work `id` $(reboot)" / "build\nnext line"
    first_root = base / "ro 'one' $HOME;rm -rf x"
    second_root = base / 'ro "two"\n$(touch y)'
    harness = base / "harness; echo pwned > z & $IFS"
    program = ["./main", "QA with space", "'QB single'", '"QC double"', "$(QD)", "`QE`", "QF; rm -rf /"]
    program += ["QG\nQH", "$QI", "QJ&&|<>"]
    spec = sandbox.SandboxSpec(workdir=workdir, readonly_roots=(first_root, second_root), harness=harness)
    command = sandbox.sandbox_command(spec, program, ONE_SECOND)
    assert script_of(command) == sandbox.SETUP_SCRIPT
    start = command.index(sandbox.SETUP_SCRIPT)
    for value in [str(workdir), str(harness), str(first_root), str(second_root), *program[1:]]:
        holders = [i for i, part in enumerate(command) if value in part]
        assert len(holders) == 1 and command[holders[0]] == value, (value, holders)
        assert holders[0] > start + 1, value
    layout = ["sh", str(workdir), str(harness), "2", str(first_root), str(second_root), "1", "1", "2"]
    assert command[start + 1 :] == [*layout, *program]


# ---------------------------------------------------------------------------
# Validation


@pytest.mark.parametrize(
    "overrides",
    [
        {"workdir": Path("runs/build")},
        {"readonly_roots": (SCRATCH, Path("home/user"))},
        {"harness": Path("assets/harness")},
        {"workdir": SCRATCH},
        {"workdir": HOME},
        {"readonly_roots": ()},
        {"readonly_roots": str(SCRATCH)},
        {"harness": WORK},
        {"harness": WORK.parent},
        {"harness": SCRATCH / "runs"},
    ],
    ids=[
        "relative-workdir",
        "relative-root",
        "relative-harness",
        "workdir-is-first-root",
        "workdir-is-second-root",
        "no-root",
        "roots-as-one-string",
        "harness-is-workdir",
        "harness-is-workdir-parent",
        "harness-contains-workdir",
    ],
)
def test_sandbox_spec_rejects_bad_paths(sandbox: ModuleType, overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        sample_spec(sandbox, **overrides)


@pytest.mark.parametrize("tasks_max", [0, -1, True, "8", 1.5, None])
def test_sandbox_spec_rejects_a_bad_tasks_max(sandbox: ModuleType, tasks_max: object) -> None:
    with pytest.raises(ValueError):
        sample_spec(sandbox, tasks_max=tasks_max)


POSIX_WORK = PurePosixPath("/scratch/runs/trial/attempt00/build")


@pytest.mark.parametrize(
    "overrides",
    [
        {"workdir": PurePosixPath("/tmp/trial/build")},
        {"workdir": PurePosixPath("/tmp")},
        {"workdir": PurePosixPath("/var/tmp/b")},
        {"workdir": PurePosixPath("/dev/shm/b")},
        {"workdir": PurePosixPath("/run/user/1025/b")},
        {"workdir": PurePosixPath("/scratch/../tmp/b")},
        {"workdir": PurePosixPath("//tmp/b")},
        {"readonly_roots": (PurePosixPath("/scratch"), PurePosixPath("/run"))},
        {"harness": PurePosixPath("/dev/shm/harness")},
    ],
)
def test_sandbox_spec_rejects_paths_a_private_tmpfs_would_hide(
    sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]
) -> None:
    # POSIX paths on every platform: the rule is about the Linux mount layout the script builds.
    monkeypatch.setattr(sandbox, "Path", PurePosixPath)
    values: dict[str, object] = {"workdir": POSIX_WORK, "readonly_roots": (PurePosixPath("/scratch"),)}
    values.update(overrides)
    with pytest.raises(ValueError, match="private tmpfs"):
        sandbox.SandboxSpec(**values)


@pytest.mark.parametrize("workdir", ["/tmpfoo/b", "/var/tmpx/b", "/scratch/tmp/b", "/runs/b", "/mnt/dev/shm/b"])
def test_sandbox_spec_accepts_lookalikes_of_the_private_dirs(
    sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch, workdir: str
) -> None:
    monkeypatch.setattr(sandbox, "Path", PurePosixPath)
    spec = sandbox.SandboxSpec(workdir=PurePosixPath(workdir), readonly_roots=(PurePosixPath("/scratch/ro"),))
    assert spec.workdir == PurePosixPath(workdir)


@pytest.mark.parametrize(
    "limits",
    [
        Limits(wall_s=0.0, memory_mb=64, cpus=1),
        Limits(wall_s=-1.0, memory_mb=64, cpus=1),
        Limits(wall_s=1.0, memory_mb=0, cpus=1),
        Limits(wall_s=1.0, memory_mb=-64, cpus=1),
        Limits(wall_s=1.0, memory_mb=64, cpus=0),
        Limits(wall_s=1.0, memory_mb=64, cpus=-2),
        Limits(wall_s=math.inf, memory_mb=64, cpus=1),
        Limits(wall_s=math.nan, memory_mb=64, cpus=1),
        Limits(wall_s="3", memory_mb=64, cpus=1),  # type: ignore[arg-type]
        Limits(wall_s=1.0, memory_mb=True, cpus=1),
        Limits(wall_s=1.0, memory_mb=64, cpus=1.5),  # type: ignore[arg-type]
    ],
    ids=[
        "zero-wall",
        "negative-wall",
        "zero-memory",
        "negative-memory",
        "zero-cpus",
        "negative-cpus",
        "infinite-wall",
        "nan-wall",
        "text-wall",
        "bool-memory",
        "fractional-cpus",
    ],
)
def test_sandbox_command_rejects_bad_limits(sandbox: ModuleType, limits: Limits) -> None:
    with pytest.raises(ValueError):
        sandbox.sandbox_command(sample_spec(sandbox), PROGRAM, limits)


def test_sandbox_command_rejects_an_empty_argv(sandbox: ModuleType) -> None:
    with pytest.raises(ValueError):
        sandbox.sandbox_command(sample_spec(sandbox), [], ONE_SECOND)


@pytest.mark.parametrize("argv", ["./main", ["./main", 3], ["./main", None], [b"./main"]], ids=repr)
def test_sandbox_command_rejects_argv_that_is_not_a_list_of_strings(sandbox: ModuleType, argv: object) -> None:
    with pytest.raises(ValueError):
        sandbox.sandbox_command(sample_spec(sandbox), argv, ONE_SECOND)


def test_sandbox_run_validates_before_running_anything(sandbox: ModuleType) -> None:
    runner = FakeRunner()
    box = sandbox.Sandbox(runner=runner)
    with pytest.raises(ValueError):
        box.run(sample_spec(sandbox), PROGRAM, Limits(wall_s=0.0, memory_mb=64, cpus=1))
    with pytest.raises(ValueError):
        box.run(sample_spec(sandbox), [], ONE_SECOND)
    assert runner.calls == []


# ---------------------------------------------------------------------------
# classify: (hang, killed) from the exit status and the measured wall time


@pytest.mark.parametrize(
    ("returncode", "wall_s", "limit_wall_s", "expected"),
    [
        (124, 3.0, 3.0, (True, False)),
        (124, 30.0, 3.0, (True, False)),
        # A program that exits 124 itself before the wall limit must not pass for a hang.
        (124, 0.1, 3.0, (False, False)),
        (124, 2.999, 3.0, (False, False)),
        (137, 3.0, 3.0, (True, False)),
        (137, 30.0, 3.0, (True, False)),
        (137, 2.999, 3.0, (False, True)),
        (137, 0.0, 3.0, (False, True)),
        (143, 3.0, 3.0, (True, False)),
        (143, 30.0, 3.0, (True, False)),
        (143, 2.999, 3.0, (False, False)),
        (-1, 3.0, 3.0, (True, False)),
        (-1, 30.0, 3.0, (True, False)),
        (-1, 0.0, 3.0, (False, False)),
        (-1, 2.999, 3.0, (False, False)),
        (0, 0.5, 3.0, (False, False)),
        (0, 30.0, 3.0, (False, False)),
        (1, 0.5, 3.0, (False, False)),
        (2, 0.5, 3.0, (False, False)),
        (3, 0.5, 3.0, (False, False)),
        (125, 0.5, 3.0, (False, False)),
        (126, 0.5, 3.0, (False, False)),
        (127, 0.5, 3.0, (False, False)),
        (255, 30.0, 3.0, (False, False)),
    ],
)
def test_classify_table(
    sandbox: ModuleType, returncode: int, wall_s: float, limit_wall_s: float, expected: tuple[bool, bool]
) -> None:
    assert sandbox.classify(returncode, wall_s, limit_wall_s) == expected


def test_classify_docstring_documents_the_mapping(sandbox: ModuleType) -> None:
    doc = inspect.getdoc(sandbox.classify) or ""
    for code in ("124", "137", "143", "-1"):
        assert code in doc, code


# ---------------------------------------------------------------------------
# Sandbox.run with a fake runner


def test_run_passes_the_command_workdir_and_runner_timeout(sandbox: ModuleType) -> None:
    runner = FakeRunner(stdout="out\n", stderr="err\n")
    spec = sample_spec(sandbox)
    limits = Limits(wall_s=2.5, memory_mb=512, cpus=2)
    sandbox.Sandbox(runner=runner).run(spec, PROGRAM, limits)
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call.argv == sandbox.sandbox_command(spec, PROGRAM, limits)
    assert call.cwd == WORK
    # wall_i 3 + KILL_AFTER_S 2 + OUTER_MARGIN_S 5 + 10.
    assert call.timeout_s == 20


@pytest.mark.parametrize(
    ("returncode", "elapsed_s", "status", "hang", "killed"),
    [
        (0, 0.5, 0, False, False),
        (3, 0.5, 3, False, False),
        (124, 3.0, 124, True, False),
        # An early 124 is the program's own exit status, not the innermost timeout's.
        (124, 0.5, 124, False, False),
        (124, 2.7, 124, False, False),
        (137, 0.5, 137, False, True),
        # Between wall_s 2.5 and the whole-second limit 3 the timeout has not fired: a 137 is a limit kill.
        (137, 2.7, 137, False, True),
        (137, 30.0, 137, True, False),
        (143, 30.0, 143, True, False),
        (143, 0.5, 143, False, False),
        (143, 2.7, 143, False, False),
        # The runner's timeout is 20 s here (3 + 2 + 5 + 10); only then is -1 the runner's own timeout.
        (-1, 30.0, -1, True, False),
        (-1, 20.0, -1, True, False),
        (-1, 0.5, 129, False, False),
        # Popen reports a death by signal N as -N; the backstop's SIGKILL reaches it as -9.
        (-9, 11.0, 137, True, False),
        (-9, 0.5, 137, False, True),
        (-9, 2.7, 137, False, True),
        (-15, 10.0, 143, True, False),
        (-15, 0.5, 143, False, False),
    ],
)
def test_run_classifies_the_result(
    sandbox: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    elapsed_s: float,
    status: int,
    hang: bool,
    killed: bool,
) -> None:
    clock = install_clock(monkeypatch, sandbox)
    runner = FakeRunner(returncode=returncode, stdout="o\n", stderr="e\n", elapsed_s=elapsed_s, clock=clock)
    limits = Limits(wall_s=2.5, memory_mb=64, cpus=1)
    result = sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, limits)
    assert isinstance(result, sandbox.SandboxResult)
    assert (result.returncode, result.stdout, result.stderr) == (status, "o\n", "e\n")
    assert result.wall_s == pytest.approx(elapsed_s)
    assert (result.hang, result.killed) == (hang, killed)


@pytest.mark.parametrize(
    ("returncode", "stderr"),
    [
        (1, "Failed to connect to bus: No such file or directory\n"),
        (1, "unshare: unshare failed: Operation not permitted\n"),
        (32, "mount: /scratch: permission denied.\n"),
        (2, "sh: 1: cd: can't cd to /scratch/runs/trial/attempt00/build\n"),
        (1, ""),
        (-9, ""),
        (-1, "timed out after 20 s\n"),
    ],
)
def test_run_raises_sandbox_unavailable_when_setup_never_reached_the_program(
    sandbox: ModuleType, returncode: int, stderr: str
) -> None:
    runner = FakeRunner(returncode=returncode, stdout="", stderr=stderr, ready=False)
    with pytest.raises(sandbox.SandboxUnavailableError, match="did not run") as caught:
        sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert f"exit status {returncode}" in str(caught.value)
    assert stderr.strip() in str(caught.value)
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    ("raw", "stderr"),
    [
        ("lassi-sandbox-ready\n", ""),
        ("lassi-sandbox-ready\nboom\n", "boom\n"),
        ("Warning: something from setup\nlassi-sandbox-ready\nboom\n", "Warning: something from setup\nboom\n"),
        ("lassi-sandbox-ready\nlassi-sandbox-ready\n", "lassi-sandbox-ready\n"),
    ],
)
def test_run_removes_only_the_first_ready_line_from_stderr(sandbox: ModuleType, raw: str, stderr: str) -> None:
    assert sandbox.READY_MARKER == "lassi-sandbox-ready"
    runner = FakeRunner(returncode=0, stderr=raw, ready=False)
    result = sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert result.stderr == stderr


@pytest.mark.parametrize("raw", ["xlassi-sandbox-ready\n", "lassi-sandbox-ready", "lassi-sandbox-ready now\n", ""])
def test_run_counts_the_ready_line_only_as_a_whole_line(sandbox: ModuleType, raw: str) -> None:
    runner = FakeRunner(returncode=0, stderr=raw, ready=False)
    with pytest.raises(sandbox.SandboxUnavailableError):
        sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)


def test_run_turns_a_missing_tool_into_sandbox_unavailable(sandbox: ModuleType) -> None:
    runner = FakeRunner(error=FileNotFoundError(2, "No such file or directory", "systemd-run"))
    with pytest.raises(sandbox.SandboxUnavailableError, match="systemd-run"):
        sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert len(runner.calls) == 1
    assert runner.calls[0].argv[0] == "systemd-run"


def test_run_names_a_missing_workdir(sandbox: ModuleType) -> None:
    runner = FakeRunner(error=FileNotFoundError(2, "No such file or directory", str(WORK)))
    with pytest.raises(sandbox.SandboxUnavailableError, match="workdir") as caught:
        sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert str(WORK) in str(caught.value)


def test_default_runner_is_the_toolchains_subprocess_runner(
    sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lassi.toolchains

    assert sandbox.Sandbox().runner is lassi.toolchains.subprocess_runner
    seen = trap_popen(monkeypatch)
    spec = sample_spec(sandbox)
    with pytest.raises(sandbox.SandboxUnavailableError, match="systemd-run"):
        sandbox.Sandbox().run(spec, PROGRAM, ONE_SECOND)
    assert seen == [sandbox.sandbox_command(spec, PROGRAM, ONE_SECOND)]


def test_sandbox_takes_its_runner_by_keyword_only(sandbox: ModuleType) -> None:
    parameters = inspect.signature(sandbox.Sandbox).parameters
    assert list(parameters) == ["runner"]
    assert parameters["runner"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["runner"].default is None


# ---------------------------------------------------------------------------
# Agent Rule 6: in lassi/executors only sandbox.py may start a process


def test_only_the_sandbox_module_starts_processes(sandbox: ModuleType) -> None:
    package_dir = Path(sandbox.__file__).resolve().parent
    names = sorted({path.stem for path in package_dir.glob("*.py")} | set(NAMED_EXECUTOR_MODULES))
    offenders: dict[str, list[str]] = {}
    for name in names:
        spec = importlib.util.find_spec("lassi.executors" if name == "__init__" else f"lassi.executors.{name}")
        assert spec is not None and spec.origin, f"missing module lassi.executors.{name}"
        found = process_references(ast.parse(Path(spec.origin).read_text(encoding="ascii")))
        if name == "sandbox":
            assert "subprocess_runner" in found
        elif found:
            offenders[name] = found
    assert not offenders


def test_in_all_of_lassi_only_the_sandbox_and_the_compiler_runner_start_processes() -> None:
    import lassi

    package_root = Path(lassi.__file__).resolve().parent
    offenders: dict[str, list[str]] = {}
    scanned = 0
    for path in sorted(package_root.rglob("*.py")):
        parts = ("lassi", *path.relative_to(package_root).with_suffix("").parts)
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        found = process_references(ast.parse(path.read_text(encoding="utf-8")))
        scanned += 1
        if found and module not in PROCESS_MODULES:
            offenders[module] = found
    assert scanned > len(PROCESS_MODULES)
    assert not offenders, offenders
