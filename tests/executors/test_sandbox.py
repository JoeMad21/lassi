"""Tests for the sandbox module lassi.executors.sandbox (P0.10, hardened in P0.16).

The sandbox module is the only module that runs generated code (Agent Rule 6;
bible Sandbox, Component Interfaces Executor contract rules). Its mechanism
comes from plans/spikes/p0-sandbox.md and its addenda (P0.10) and from
plans/spikes/p0-sandbox-hardening.md (P0.16). One command runs each program:
prlimit --core=1 around env -i (a constant PATH and the variables systemd-run
needs), around systemd-run --user --scope (memory, task, and backstop
limits), wrapping unshare -rinmpfu --mount-proc --kill-child, which runs a
constant setup script. The script makes every mount read-only with one
recursive mount_setattr call, builds a private /dev, gives the workdir a
size-capped overlay, hides the listed roots ($HOME, the scratch root, the runs
root) behind tmpfs while re-exposing the workdir, the harness, and the
toolchains root, hides /sys device attributes and the entries of /var, checks
/proc/self/mountinfo and fails closed, runs the program in nested pid and IPC
namespaces under env -i, nice, setpriv, prlimit --cpu --core=1 --fsize, the
confinement helper (new session, new session keyring, seccomp filter, and
then the ready line), and an innermost timeout, then unmounts its private
tmpfs mounts, copies the workdir's new files back to the host within the
disk cap, and prints a final line; an EXIT trap makes any failure after the
program print a line of its own last. The sandbox's command runner
(capped_runner) caps stdout and stderr.

The P0.16 acceptance items these local tests pin (the remote tests on the
build host check them for real): R1 private /dev, R2 read-only mounts that
fail closed, R3 default-deny view, R4 output caps (see also
tests/toolchains/test_runner_caps.py), R5 workdir disk cap (the tmpfs size,
the file size limit, and the copy-back's byte budget), R6 core dumps
(--core=1 and the seccomp filter that keeps the program from lowering it),
and R7 runner kill (--kill-child), plus the fixes from the P0.16 review: the
constant PATH and scrubbed environment, the session and keyring, and the
checks on hidden roots and exposures, and from its second round: the ready
line printed by the confinement, the EXIT trap, the done line counted only on
a normal exit, the program's own IPC namespace, the copy-back's rename over
host files (hard links), and the typed, commented embedded programs, and
from its third round: any OSError that keeps the command from starting is
SandboxUnavailableError, the hang test reads the program's own time from the
clock readings in the done line, and the keyring syscalls are refused.

Every test that runs something uses a fake CommandRunner (or a trapped
subprocess.Popen), so no sandbox and no generated code ever starts. The
setup-script tests run SETUP_SCRIPT under sh with stub commands on a PATH
that holds only the stubs, and with the script's writes to
/proc/sys/user/max_user_namespaces and /proc/self/coredump_filter pointed at
files under tmp_path: nothing is mounted, no limit is written, the read-only
and copy-back programs are only recorded, and the program argv is only
recorded, never run. The mount-check tests run MOUNT_CHECK under awk against
synthetic mountinfo lines written for these tests (inputs, not captures from
a host). The copy-back tests run COPY_BACK_PROGRAM on directories under
tmp_path. READONLY_PROGRAM is never run here, and CONFINE_PROGRAM is only
imported under another name (its FILTER is run through a small classic-BPF
interpreter here; nothing is installed). The paths are made up and never touched
unless a test creates them under tmp_path. No value in this module is a
measurement: the limits and caps are sample inputs or design constants, and
the classify table is the P0.10 contract's mapping, which reads the exit
statuses the spike recorded (timeout 124, a limit kill 137, the backstop 143
or 137, the runner's own timeout -1) together with the elapsed wall time.
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
import stat
import struct
import subprocess
import sys
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
TOOLCHAINS = SCRATCH / "toolchains"
PROGRAM = ["./main", "--size", "8"]
ONE_SECOND = Limits(wall_s=1.0, memory_mb=64, cpus=1)
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
PROJECT_NAMES = ("lassi-repro", "lassi-ee", "lassi-df", "hecbench", "qwen", "wizardcoder", "a100", "mi300x")
# The executor modules the P0.10 contract names; any other module found in the package is scanned too.
NAMED_EXECUTOR_MODULES = ("__init__", "native", "none", "sandbox", "workdir")
RUNNER_NAMES = ("CommandResult", "CommandRunner", "subprocess_runner", "capped_runner")
# The only modules in lassi/ that may start a process, and why.
PROCESS_MODULES = {
    "lassi.executors.sandbox": "runs generated code, only inside the sandbox",
    "lassi.toolchains._base": "defines the command runners; only the sandbox runs generated code through one",
    "lassi.toolchains": "re-exports the command runners",
}
# The tools SETUP_SCRIPT checks with command -v before it does anything else, in its order.
CHECKED_TOOLS = ("env", "nice", "setpriv", "prlimit", "timeout", "python3", "awk", "unshare")
# The exact statement that runs the program, with its line continuations joined; @CONFINE_PROGRAM@ stands for
# the embedded program.
PROGRAM_STATEMENT = (
    'unshare --pid --ipc --fork --kill-child -- env -i PATH="$PATH" HOME="$workdir" LANG=C.UTF-8 TMPDIR=/tmp'
    " nice -n 19"
    " setpriv --no-new-privs --inh-caps=-all --bounding-set=-all --"
    ' prlimit --cpu="$cpu" --core=1 --fsize="$disk" --'
    " python3 -I -S -c '@CONFINE_PROGRAM@'"
    ' timeout --kill-after="$kill_after" "$wall" "$@" || status=$?'
)
# The sandbox user namespace's limit, which SETUP_SCRIPT sets to 0 while it still holds capabilities and
# before /proc turns read-only (the host value is kept per user namespace and never changes), and its line.
USERNS_LIMIT = "/proc/sys/user/max_user_namespaces"
USERNS_LINE = f"echo 0 > {USERNS_LIMIT}"
# The setup shell's own core dump filter, which every later process inherits, and its line.
CORE_FILTER = "/proc/self/coredump_filter"
CORE_FILTER_LINE = f"echo 0 > {CORE_FILTER}"
# The disk cap the stub layout passes, in bytes (8 MiB).
STUB_DISK = str(8 << 20)
# The host device nodes the private /dev binds, and the standard symbolic links it holds (name -> target).
DEV_NODES = ("null", "zero", "full", "random", "urandom", "tty")
DEV_LINKS = {
    "fd": "/proc/self/fd",
    "stdin": "/proc/self/fd/0",
    "stdout": "/proc/self/fd/1",
    "stderr": "/proc/self/fd/2",
    "ptmx": "pts/ptmx",
}
# The clock SETUP_SCRIPT reads right before and right after the program's namespace (CLOCK_BOOTTIME in 10 ms
# steps), and the two readings the stub harness gives it: the stub file starts at the first, and the stub
# `unshare` (the program's chain) moves it to the second.
UPTIME = "/proc/uptime"
STUB_STARTED, STUB_ENDED = "1000.25", "1003.50"
# The line CONFINE_PROGRAM prints right before it execs the program's timeout, and the setup's last line once
# copy-back finished, which carries the two clock readings (the constants READY_MARKER and DONE_MARKER, pinned in
# test_constants_match_the_contract); these are the lines of a stub run.
READY_LINE = "lassi-sandbox-ready\n"
DONE_LINE = f"lassi-sandbox-done {STUB_STARTED} {STUB_ENDED}\n"
INCOMPLETE_LINE = f"lassi-sandbox-done {STUB_STARTED} {STUB_ENDED} incomplete\n"
# The EXIT trap the setup sets once the program's namespace has ended, and the line it prints: any failure from
# there on ends stderr with this line, so a done line the program printed itself can never end it.
STOP_LINE = "lassi-sandbox: the setup stopped after the program ended\n"
TRAP_LINE = "trap 'echo \"lassi-sandbox: the setup stopped after the program ended\" >&2' EXIT"
# The stderr of a full stub run: the stubs of python3, awk, and unshare print these tags, so their order
# against the ready and done lines shows. The unshare stub stands in for the program's whole chain, so it
# prints the ready line first, as CONFINE_PROGRAM does.
STUB_ORDER = "stub:python3\nstub:awk\nlassi-sandbox-ready\nstub:unshare\nstub:python3\n" + DONE_LINE


def done_line(program_s: float, *, incomplete: bool = False) -> str:
    """Return a done line whose clock readings are `program_s` apart, as /proc/uptime prints them (10 ms steps)."""
    ended = 100000 + math.floor(round(program_s * 100, 6))
    return f"lassi-sandbox-done 1000.00 {ended // 100}.{ended % 100:02d}" + (" incomplete" if incomplete else "") + "\n"


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
    whenever the sandbox reached the program, and, with `done` True (the
    default), ends with the setup's done line, as it does once copy-back
    finished, marked incomplete with `incomplete`; its clock readings are
    `program_s` apart (the program's own time), or `elapsed_s` apart when
    `program_s` is None. `ready` False stands for a setup that failed, and
    `stderr` is then all there is. The truncation flags are passed to
    CommandResult only when set.
    """

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    elapsed_s: float = 0.0
    clock: Clock | None = None
    error: BaseException | None = None
    ready: bool = True
    done: bool = True
    incomplete: bool = False
    program_s: float | None = None
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    calls: list[Call] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the call, advance the clock, and return the canned CommandResult or raise the canned error."""
        self.calls.append(Call(argv=list(argv), cwd=Path(cwd), timeout_s=timeout_s))
        if self.clock is not None:
            self.clock.now += self.elapsed_s
        if self.error is not None:
            raise self.error
        stderr = self.stderr
        if self.ready:
            program_s = self.elapsed_s if self.program_s is None else self.program_s
            final = done_line(program_s, incomplete=self.incomplete)
            stderr = READY_LINE + stderr + (final if self.done else "")
        flags = {"stdout_truncated": self.stdout_truncated, "stderr_truncated": self.stderr_truncated}
        flags = {name: value for name, value in flags.items() if value}
        return CommandResult(returncode=self.returncode, stdout=self.stdout, stderr=stderr, **flags)


def install_clock(monkeypatch: pytest.MonkeyPatch, sandbox: ModuleType) -> Clock:
    """Replace time.monotonic (and a monotonic imported by name into the module) with a fake clock."""
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    if hasattr(sandbox, "monotonic"):
        monkeypatch.setattr(sandbox, "monotonic", clock)
    return clock


def trap_popen(monkeypatch: pytest.MonkeyPatch, error: OSError | None = None) -> list[list[str]]:
    """Replace subprocess.Popen with a trap that records argv and raises `error`; nothing starts.

    Without `error` it raises FileNotFoundError for the first argv element, as if that tool were missing.
    """
    seen: list[list[str]] = []

    def trap(argv: Sequence[str], *args: object, **kwargs: object) -> None:
        """Record the argv and refuse to start anything."""
        seen.append(list(argv))
        raise error or FileNotFoundError(2, "No such file or directory", str(argv[0]))

    monkeypatch.setattr(subprocess, "Popen", trap)
    return seen


def sample_spec(sandbox: ModuleType, **overrides: object) -> object:
    """Return a SandboxSpec with the sample workdir, the scratch and home roots hidden, the harness, and toolchains."""
    values: dict[str, object] = {
        "workdir": WORK,
        "hidden_roots": (SCRATCH, HOME),
        "harness": HARNESS,
        "toolchains": TOOLCHAINS,
    }
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
    process_names = ("subprocess_runner", "capped_runner", "Popen", "create_subprocess_exec", "create_subprocess_shell")
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


# ---------------------------------------------------------------------------
# The stub harness for SETUP_SCRIPT


@dataclass(frozen=True)
class StubRun:
    """What SETUP_SCRIPT did under stub commands.

    `calls` lists every stub call in order as [command, *arguments]; the
    stubs are mount, umount, mkdir, touch, ln, python3, awk, and unshare
    (the nested unshare that stands in for the program). `userns_limit` is
    what the script wrote in place of /proc/sys/user/max_user_namespaces, or
    None when it wrote nothing there, and `core_filter` the same for
    /proc/self/coredump_filter. `userns_before_python3` holds, for each
    python3 call, whether the user namespace write had already happened.
    """

    returncode: int
    stderr: str
    calls: list[list[str]]
    userns_limit: str | None
    userns_before_python3: list[bool]
    core_filter: str | None = None


def read_calls(path: Path) -> list[list[str]]:
    """Return the records the stubs wrote: fields end with octal 037, records with octal 036."""
    if not path.exists():
        return []
    text = path.read_bytes().decode("utf-8")
    return [record.split("\x1f")[:-1] for record in text.split("\x1e") if record]


def write_stub(directory: Path, name: str, body: str) -> None:
    """Write an executable /bin/sh stub named `name` into `directory`."""
    path = directory / name
    path.write_text("#!/bin/sh\n" + body, encoding="ascii", newline="\n")
    path.chmod(0o755)


def stub_script(sandbox: ModuleType, userns_target: Path, core_target: Path, uptime: Path) -> str:
    """Return SETUP_SCRIPT with its writes to the user namespace limit and the core dump filter pointed at files.

    Its two reads of /proc/uptime read the file `uptime` instead.
    """
    assert sandbox.SETUP_SCRIPT.count(USERNS_LIMIT) == 1
    assert sandbox.SETUP_SCRIPT.count(CORE_FILTER) == 1
    assert sandbox.SETUP_SCRIPT.count(UPTIME) == 2
    assert "'" not in f"{userns_target}{core_target}{uptime}"
    script = sandbox.SETUP_SCRIPT.replace(USERNS_LIMIT, f"'{userns_target.as_posix()}'")
    script = script.replace(UPTIME, f"'{uptime.as_posix()}'")
    return script.replace(CORE_FILTER, f"'{core_target.as_posix()}'")


@dataclass(frozen=True)
class StubProgram:
    """How the stub `unshare`, standing in for the program's whole chain, behaves.

    `ready` prints the ready line first, as CONFINE_PROGRAM does once the
    confinement is in place (False stands for a chain that failed before it);
    `stderr` is printed after the stub's tag, as the program's own last
    output (it holds no single quote); `status` is the exit status.
    """

    status: int = 0
    ready: bool = True
    stderr: str = ""


def stub_bodies(
    log: str,
    target: str,
    uptime: str,
    *,
    fail_target: str,
    check_fails: bool,
    program: StubProgram,
    copy_back_status: int,
) -> dict[str, str]:
    """Return the shell body of each stub: each records its call in `log`, then acts as the options say.

    The stub `unshare`, the program's chain, moves the clock file `uptime` from STUB_STARTED to STUB_ENDED.
    """

    def record(name: str) -> str:
        """Return the lines that append one record, the stub's name and its arguments, to the log."""
        return f"printf '%s\\037' '{name}' \"$@\" >> '{log}'\nprintf '\\036' >> '{log}'\n"

    fail = f"for last; do :; done\n[ \"$last\" = '{fail_target}' ] && exit 32\n" if fail_target else ""
    userns = f"if [ -f '{target}' ]; then set -- yes; else set -- no; fi\n" + record("@userns")
    check = "printf 'writable host mount: /var/tmp ext4 /dev/sdz1\\n' >&2\nexit 1\n" if check_fails else ""
    copy = f'[ "$#" -gt 4 ] && exit {copy_back_status}\n' if copy_back_status else ""
    bodies = {name: record(name) + "exit 0\n" for name in ("umount", "mkdir", "touch", "ln")}
    bodies["mount"] = record("mount") + fail + "exit 0\n"
    # The python3 stub first logs whether the user namespace limit was written, in a subshell so "$@" stays. A
    # copy-back status of 128 or more stands for a death by a signal, which prints nothing, not even the tag.
    tag = "printf 'stub:python3\\n' >&2\n"
    steps = copy + tag if copy_back_status >= 128 else tag + copy
    bodies["python3"] = f"({userns})\n" + record("python3") + steps + "exit 0\n"
    bodies["awk"] = record("awk") + "printf 'stub:awk\\n' >&2\n" + check + "exit 0\n"
    assert "'" not in program.stderr
    ready = "printf 'lassi-sandbox-ready\\n' >&2\n" if program.ready else ""
    output = f"printf '%s' '{program.stderr}' >&2\n" if program.stderr else ""
    bodies["unshare"] = record("unshare") + ready + "printf 'stub:unshare\\n' >&2\n" + output
    bodies["unshare"] += f"printf '{STUB_ENDED} 9.00\\n' > '{uptime}'\nexit {program.status}\n"
    return bodies


# This helper is long on purpose (about 20 of its lines are docstring): every stub setting has to reach one
# shell run, and splitting it would scatter the stub layout the setup-order tests read in one place.
def run_setup_with_stubs(
    sandbox: ModuleType,
    tmp_path: Path,
    layout: list[str],
    *,
    fail_target: str = "",
    check_fails: bool = False,
    program_status: int = 0,
    copy_back_status: int = 0,
    missing: str = "",
    userns_target: Path | None = None,
    program: StubProgram | None = None,
) -> StubRun:
    """Run SETUP_SCRIPT under sh with only stub commands on PATH; nothing is mounted and no program runs.

    Every stub records its call in one shared log and exits 0, except: the
    stub `mount` exits 32 when its last argument is `fail_target`; the stub
    `awk` (the mount check) prints a writable-mount line and exits 1 with
    `check_fails`; the stub `unshare` (the nested one that would run the
    program's chain) prints the ready line and exits `program_status`, or
    behaves as `program` says when it is given; the stub `python3` exits
    `copy_back_status` on the copy-back call (more than four arguments) when
    it is not 0. The stubs env, nice, setpriv, prlimit, and timeout exist
    only for the script's PATH check and exit 99 if called. `missing` names
    a checked tool to leave out. The script's write to
    /proc/sys/user/max_user_namespaces goes to `userns_target` (by default a
    file under tmp_path), and its write to /proc/self/coredump_filter to a
    file under tmp_path, so no real limit is ever written; its reads of
    /proc/uptime read a file under tmp_path that holds STUB_STARTED until
    the stub `unshare` moves it to STUB_ENDED. PATH holds nothing but the
    stubs, so no real mount can ever run; the test skips as root, and when
    sh is absent.
    """
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh is not on PATH")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("never run the setup script as root, even with stubs")
    assert not re.search(r"/s?bin/", sandbox.SETUP_SCRIPT), "the script must find every command through PATH"
    target = tmp_path / "max_user_namespaces" if userns_target is None else userns_target
    core = tmp_path / "coredump_filter"
    uptime = tmp_path / "uptime"
    uptime.write_text(f"{STUB_STARTED} 5.00\n", encoding="ascii", newline="\n")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    log = tmp_path / "calls.log"
    assert "'" not in f"{log}{target}{fail_target}"
    chain = StubProgram(status=program_status) if program is None else program
    bodies = stub_bodies(log.as_posix(), target.as_posix(), uptime.as_posix(), fail_target=fail_target,
                         check_fails=check_fails, program=chain, copy_back_status=copy_back_status)
    for name, body in bodies.items():
        write_stub(stubs, name, body)
    for name in ("env", "nice", "setpriv", "prlimit", "timeout"):
        write_stub(stubs, name, "exit 99\n")
    if missing:
        (stubs / missing).unlink()
    # The script goes in a file, not after -c: MSYS sh on Windows cuts a command-line argument at about 8 KiB,
    # where Linux allows 128 KiB per argument. The positional parameters are the same either way.
    script = tmp_path / "setup.sh"
    script.write_text(stub_script(sandbox, target, core, uptime), encoding="ascii", newline="\n")
    argv = [shell, script.as_posix(), *layout]
    done = subprocess.run(argv, env={"PATH": str(stubs)}, capture_output=True, timeout=120, check=False)
    records = read_calls(log)
    return StubRun(
        returncode=done.returncode,
        stderr=done.stderr.decode("utf-8", errors="replace").replace("\r\n", "\n"),
        calls=[record for record in records if record[0] != "@userns"],
        userns_limit=target.read_text(encoding="ascii") if target.is_file() else None,
        userns_before_python3=[record[1] == "yes" for record in records if record[0] == "@userns"],
        core_filter=core.read_text(encoding="ascii") if core.is_file() else None,
    )


def stub_layout(tmp_path: Path, *, exposures: bool = True) -> tuple[list[str], dict[str, str]]:
    """Return a positional layout for SETUP_SCRIPT and its named values.

    The layout holds the workdir, the harness and the toolchains root (both
    "" when `exposures` is False), a disk cap of 8 MiB in bytes, two hidden roots, and
    the limits cpu 6, wall 3, kill-after 2, then the program argv. Only the
    workdir exists (the script cds into it); the stubs never touch the other
    paths.
    """
    workdir = tmp_path / "work"
    workdir.mkdir()
    values = {
        "workdir": workdir.as_posix(),
        "harness": (tmp_path / "harness").as_posix() if exposures else "",
        "toolchains": (tmp_path / "toolchains").as_posix() if exposures else "",
        "root_a": (tmp_path / "hide-a").as_posix(),
        "root_b": (tmp_path / "hide-b").as_posix(),
    }
    layout = [values["workdir"], values["harness"], values["toolchains"], STUB_DISK, "2", values["root_a"]]
    layout += [values["root_b"], "6", "3", "2"]
    return [*layout, "./main", "one two"], values


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[StubRun, dict[str, str]]:
    """Run SETUP_SCRIPT once under stubs with the full layout and a program that exits 7; the tests below read it."""
    from lassi.executors import sandbox

    tmp_path = tmp_path_factory.mktemp("setup-stubs")
    layout, values = stub_layout(tmp_path)
    return run_setup_with_stubs(sandbox, tmp_path, layout, program_status=7), values


def index_of(calls: list[list[str]], call: list[str]) -> int:
    """Return the index of the one call equal to `call`; fail when there is none or more than one."""
    found = [index for index, seen in enumerate(calls) if seen == call]
    assert len(found) == 1, (call, found, calls)
    return found[0]


def typed_mounts(calls: list[list[str]], fstype: str, source: str | None = None) -> list[tuple[int, list[str]]]:
    """Return (index, arguments) of each `mount -t <fstype> -o <options> <source> <point>` call, maybe by source."""
    found = []
    for index, call in enumerate(calls):
        args = call[1:]
        if call[0] == "mount" and len(args) == 6 and args[0] == "-t" and args[1] == fstype and args[2] == "-o":
            if source is None or args[4] == source:
                found.append((index, args))
    return found


def only(found: list[tuple[int, list[str]]], what: str) -> tuple[int, list[str]]:
    """Return the one (index, arguments) in `found`; fail naming `what` when there is none or more than one."""
    assert len(found) == 1, (what, found)
    return found[0]


def moves(calls: list[list[str]], flag: str) -> list[tuple[int, str, str]]:
    """Return (index, source, target) of each `mount <flag> <source> <target>` call (--bind, --rbind, --move)."""
    return [(i, c[2], c[3]) for i, c in enumerate(calls) if c[0] == "mount" and len(c) == 4 and c[1] == flag]


def remounts(calls: list[list[str]], point: str) -> list[tuple[int, list[str]]]:
    """Return (index, option list) of each `mount -o remount,... <point>` call."""
    found = []
    for index, call in enumerate(calls):
        if call[0] == "mount" and len(call) == 4 and call[1] == "-o" and call[3] == point:
            options = call[2].split(",")
            if "remount" in options:
                found.append((index, options))
    return found


def option_map(text: str) -> dict[str, str]:
    """Return a mount option string as a dict: key=value pairs, and flags mapped to ""."""
    return dict((part.split("=", 1) + [""])[:2] for part in text.split(","))


def the_program_call(calls: list[list[str]]) -> int:
    """Return the index of the one unshare call, the nested one that runs the program."""
    found = [index for index, call in enumerate(calls) if call[0] == "unshare"]
    assert len(found) == 1, calls
    return found[0]


def the_check_call(calls: list[list[str]]) -> int:
    """Return the index of the one awk call, the mountinfo check."""
    found = [index for index, call in enumerate(calls) if call[0] == "awk"]
    assert len(found) == 1, calls
    return found[0]


# ---------------------------------------------------------------------------
# Constants, dataclasses, and the module itself


def test_constants_match_the_contract(sandbox: ModuleType) -> None:
    assert sandbox.KILL_AFTER_S == 2
    assert sandbox.OUTER_MARGIN_S == 5
    assert sandbox.READY_MARKER == "lassi-sandbox-ready"
    assert sandbox.DONE_MARKER == "lassi-sandbox-done"
    assert sandbox.INCOMPLETE_SUFFIX == " incomplete"
    # R5: the documented workdir disk cap, in MiB, that SandboxSpec uses unless told otherwise, and the
    # copy-back's limits on depth and relative path length.
    assert sandbox.WORKDIR_DISK_MB == 256
    assert (sandbox.COPY_DEPTH, sandbox.COPY_PATH) == (32, 1024)
    # The only PATH inside, and the caller's variables the sandbox command keeps (for systemd-run --user).
    assert sandbox.SANDBOX_PATH == "/usr/sbin:/usr/bin:/sbin:/bin"
    assert sandbox.PASSED_VARIABLES == ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
    assert sandbox.SYSTEM_DIRS == ("/var", "/sys")
    for name in ("SETUP_SCRIPT", "MOUNT_CHECK", "READONLY_PROGRAM", "CONFINE_PROGRAM", "COPY_BACK_PROGRAM"):
        value = getattr(sandbox, name)
        assert isinstance(value, str) and value.strip() and value.isascii(), name


def test_sandbox_spec_is_frozen_with_defaults(sandbox: ModuleType) -> None:
    spec = sandbox.SandboxSpec(workdir=WORK, hidden_roots=(SCRATCH,))
    assert dataclasses.is_dataclass(spec)
    assert spec.harness is None
    assert spec.toolchains is None
    assert spec.tasks_max == 256
    assert spec.disk_mb == sandbox.WORKDIR_DISK_MB
    assert spec.hidden_roots == (SCRATCH,)
    # P0.20 adds `environment`, the program's allowlisted environment; None keeps the program defaults.
    assert spec.environment is None
    assert {f.name for f in dataclasses.fields(sandbox.SandboxSpec)} == {
        "workdir", "hidden_roots", "harness", "toolchains", "tasks_max", "disk_mb", "environment"
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.workdir = HOME  # type: ignore[misc]


def test_sandbox_result_fields_and_error_type(sandbox: ModuleType) -> None:
    names = [f.name for f in dataclasses.fields(sandbox.SandboxResult)]
    assert names == [
        "returncode", "stdout", "stderr", "wall_s", "hang", "killed", "stdout_truncated", "stderr_truncated",
        "workdir_incomplete", "program_s",
    ]
    result = sandbox.SandboxResult(returncode=0, stdout="o", stderr="e", wall_s=0.5, hang=False, killed=False)
    assert (result.stdout_truncated, result.stderr_truncated, result.workdir_incomplete) == (False, False, False)
    assert result.program_s is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.hang = True  # type: ignore[misc]
    assert issubclass(sandbox.SandboxUnavailableError, RuntimeError)


def test_module_docstring_states_the_mechanism_and_cites_the_spike(sandbox: ModuleType) -> None:
    doc = sandbox.__doc__ or ""
    assert "plans/spikes/p0-sandbox.md" in doc
    assert "systemd-run" in doc
    assert "unshare" in doc


def test_module_docstring_documents_the_hardening_caps_and_limits(sandbox: ModuleType) -> None:
    doc = sandbox.__doc__ or ""
    assert "plans/spikes/p0-sandbox-hardening.md" in doc
    terms = ("mount_setattr", "--kill-child", "--core=1", "OUTPUT_CAP_BYTES", "WORKDIR_DISK_MB", "Known limits")
    terms += ("CONFINE_PROGRAM", "seccomp", "workdir_cap_bytes", "SANDBOX_PATH", "--fsize", "coredump_filter")
    terms += ("--ipc", "EXIT trap", "READY_MARKER", "/proc/uptime", "program_s", "keyctl")
    for term in terms:
        assert term in doc, term
    # Each step says what the spike measured; nothing claims the composition as written was measured.
    assert "every step above, alone and composed" not in doc
    # Review finding 1: probe L ran a dirty tree, so it is cited as exploratory and never as a measurement.
    assert doc.count("probe L") == doc.count("probe L, exploratory") > 0, doc.count("probe L")
    assert "(probe L)" not in doc and "(probe L measured" not in doc
    # The P0.10 gap list is closed; native runs of generated code no longer wait for P0.16.
    assert "must wait for P0.16" not in doc


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
    assert "capped_runner" in process_references(tree)


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
            [str(WORK), str(HARNESS), str(TOOLCHAINS), str(256 << 20), "2", str(SCRATCH), str(HOME), "5", "3", "2"],
        ),
        (
            {"hidden_roots": (SCRATCH,), "harness": None, "toolchains": None, "tasks_max": 64, "disk_mb": 8},
            Limits(wall_s=0.5, memory_mb=1, cpus=3),
            ["MemoryMax=1M", "MemorySwapMax=0", "TasksMax=64", "RuntimeMaxSec=8", "TimeoutStopSec=1"],
            [str(WORK), "", "", str(1 << 20), "1", str(SCRATCH), "2", "1", "2"],
        ),
        (
            {"hidden_roots": (HOME, SCRATCH)},
            Limits(wall_s=10.0, memory_mb=2048, cpus=4),
            ["MemoryMax=2048M", "MemorySwapMax=0", "TasksMax=256", "RuntimeMaxSec=17", "TimeoutStopSec=1"],
            [str(WORK), str(HARNESS), str(TOOLCHAINS), str(256 << 20), "2", str(HOME), str(SCRATCH), "40", "10", "2"],
        ),
        (
            {"hidden_roots": (SCRATCH, HOME / "sub", HOME, SCRATCH), "disk_mb": 32},
            Limits(wall_s=1.0, memory_mb=64, cpus=1),
            ["MemoryMax=64M", "MemorySwapMax=0", "TasksMax=256", "RuntimeMaxSec=8", "TimeoutStopSec=1"],
            [str(WORK), str(HARNESS), str(TOOLCHAINS), str(32 << 20), "2", str(SCRATCH), str(HOME), "1", "1", "2"],
        ),
    ],
    ids=["ceil-wall-and-cpu", "sub-second-no-harness-no-toolchains", "whole-seconds-roots-in-order", "nested-roots"],
)
def test_sandbox_command_is_exact(
    sandbox: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    limits: Limits,
    properties: list[str],
    layout: list[str],
) -> None:
    for name in sandbox.PASSED_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    command = sandbox.sandbox_command(sample_spec(sandbox, **overrides), PROGRAM, limits)
    expected = ["prlimit", "--core=1", "--", "env", "-i", "PATH=/usr/sbin:/usr/bin:/sbin:/bin"]
    expected += ["systemd-run", "--user", "--scope", "--quiet"]
    expected += [part for p in properties for part in ("-p", p)]
    expected += ["unshare", "-rinmpfu", "--mount-proc", "--kill-child", "sh", "-c", sandbox.SETUP_SCRIPT, "sh"]
    expected += [*layout, *PROGRAM]
    assert isinstance(command, list)
    assert command == expected
    assert all(isinstance(part, str) for part in command)


def test_sandbox_command_keeps_the_program_argv_verbatim(sandbox: ModuleType) -> None:
    program = ("python3", "-c", "print('hi')", "", "--flag=a b")
    command = sandbox.sandbox_command(sample_spec(sandbox), program, ONE_SECOND)
    assert command[-len(program) :] == list(program)


def test_sandbox_command_keeps_only_path_and_the_passed_variables(
    sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Agent Rule 12 and the review's PATH finding: env -i gives the whole sandbox, the setup shell included, a
    # constant PATH and only what systemd-run needs; an API key, LD_PRELOAD, or the caller's PATH never enters.
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1025")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1025/bus")
    monkeypatch.setenv("LASSI_TEST_API_KEY", "sk-lassi-test-secret")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/evil.so")
    monkeypatch.setenv("PATH", "/tmp/evil-bin" + os.pathsep + os.environ.get("PATH", ""))
    command = sandbox.sandbox_command(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    start = command.index("env")
    assert command[start : start + 6] == [
        "env", "-i", "PATH=/usr/sbin:/usr/bin:/sbin:/bin", "XDG_RUNTIME_DIR=/run/user/1025",
        "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1025/bus", "systemd-run",
    ], command[start : start + 6]  # fmt: skip
    for text in ("sk-lassi-test-secret", "evil"):
        assert not [part for part in command if text in part], text
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "")
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    command = sandbox.sandbox_command(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert command[start : start + 4] == ["env", "-i", "PATH=/usr/sbin:/usr/bin:/sbin:/bin", "systemd-run"]


@pytest.mark.parametrize(
    ("disk_mb", "memory_mb", "expected_mb"),
    [(256, 512, 256), (256, 64, 32), (8, 64, 8), (32, 64, 32), (33, 64, 32), (256, 1, 1), (4, 3, 1), (1, 4096, 1)],
)
def test_workdir_cap_is_the_disk_cap_but_at_most_half_the_memory_limit(
    sandbox: ModuleType, disk_mb: int, memory_mb: int, expected_mb: int
) -> None:
    # R5 with probe E: tmpfs pages count toward MemoryMax, so a cap above half the limit would end in a memory
    # kill (137) before ENOSPC, and the copy-back needs the other half.
    spec = sample_spec(sandbox, disk_mb=disk_mb)
    limits = Limits(wall_s=1.0, memory_mb=memory_mb, cpus=1)
    assert sandbox.workdir_cap_bytes(spec, limits) == expected_mb << 20
    command = sandbox.sandbox_command(spec, PROGRAM, limits)
    assert command[command.index(sandbox.SETUP_SCRIPT) + 5] == str(expected_mb << 20)


def test_r6_the_whole_command_runs_at_core_limit_one(sandbox: ModuleType) -> None:
    # The kernel aborts a piped core dump at a soft RLIMIT_CORE of exactly 1; at 0 systemd-coredump still
    # stores the core (plans/spikes/p0-sandbox-hardening.md, probes F and G). The outer prlimit covers unshare.
    command = sandbox.sandbox_command(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert command[:5] == ["prlimit", "--core=1", "--", "env", "-i"], command[:5]
    assert not [part for part in command if "--core=0" in part]
    assert sandbox.SETUP_SCRIPT.count("--core=1") == 1


def test_r7_both_unshare_calls_kill_their_child_when_they_die(sandbox: ModuleType) -> None:
    # The runner's timeout kills the process group, which holds the outer unshare; --kill-child then takes
    # the pid namespace down with its init, setsid children included (spike probes H2 and J3).
    command = sandbox.sandbox_command(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    start = command.index("unshare")
    assert command[start : start + 6] == ["unshare", "-rinmpfu", "--mount-proc", "--kill-child", "sh", "-c"]
    assert "unshare --pid --ipc --fork --kill-child -- " in sandbox.SETUP_SCRIPT


# ---------------------------------------------------------------------------
# SETUP_SCRIPT: constant, positional, and in the contract's order (text)


def test_setup_script_never_substitutes_commands(sandbox: ModuleType) -> None:
    assert "$(" not in sandbox.SETUP_SCRIPT
    assert "`" not in sandbox.SETUP_SCRIPT


def test_setup_script_checks_its_tools_before_anything_else(sandbox: ModuleType) -> None:
    script = sandbox.SETUP_SCRIPT
    lines = script.split("\n")
    assert lines[0] == "set -eu", lines[:2]
    loop = f"for tool in {' '.join(CHECKED_TOOLS)}; do"
    assert script.count(loop) == 1, script
    assert 'command -v "$tool" > /dev/null' in script
    firsts = [script.find(word) for word in ("mount ", "python3 -I", USERNS_LINE)]
    assert -1 not in firsts, firsts
    assert script.find(loop) < min(firsts)


def test_setup_script_limits_user_namespaces_once_and_before_the_program(sandbox: ModuleType) -> None:
    # The write needs the capabilities setpriv drops and a /proc that is still writable, so it must come before
    # the read-only step (checked under stubs below) and before the program's chain, which prints the marker.
    script = sandbox.SETUP_SCRIPT
    assert script.split("\n").count(USERNS_LINE) == 1, script
    assert script.count(USERNS_LIMIT) == 1
    assert script.index(USERNS_LINE) < script.index("unshare --pid")


def test_setup_script_embeds_the_mount_check_for_awk_exactly_once(sandbox: ModuleType) -> None:
    assert "'" not in sandbox.MOUNT_CHECK
    invocation = f"awk '{sandbox.MOUNT_CHECK}' /proc/self/mountinfo"
    assert sandbox.SETUP_SCRIPT.count(invocation) == 1
    assert sandbox.SETUP_SCRIPT.count(sandbox.MOUNT_CHECK) == 1


def test_setup_script_runs_the_program_under_env_nice_setpriv_prlimit_confine_timeout(sandbox: ModuleType) -> None:
    script = sandbox.SETUP_SCRIPT
    assert script.count("unshare --pid") == 1 and script.count("|| status=$?") == 1
    start, end = script.index("unshare --pid"), script.index("|| status=$?") + len("|| status=$?")
    statement = re.sub(r"\\\n *", "", script[start:end])
    assert statement == PROGRAM_STATEMENT.replace("@CONFINE_PROGRAM@", sandbox.CONFINE_PROGRAM), statement
    # The setup stays pid 1 after the program so it can copy the workdir back: it never execs.
    lines = script.split("\n")
    assert not [line for line in lines if re.match(r"\s*exec\s", line)], lines


def test_setup_script_leaves_the_ready_line_to_the_confinement_and_ends_with_done_then_the_program_status(
    sandbox: ModuleType,
) -> None:
    lines = sandbox.SETUP_SCRIPT.rstrip("\n").split("\n")
    final = f'echo "{sandbox.DONE_MARKER} $started $ended$note" >&2'
    assert lines[-3:] == ["trap - EXIT", final, 'exit "$status"'], lines[-4:]
    # The done line gets the incomplete mark only when the copy-back exits 3; any other failure stops the setup.
    assert "0) note= ;;\n  3) note=' incomplete' ;;\n  *) exit \"$copied\" ;;" in sandbox.SETUP_SCRIPT
    # The review's round-2 question: the ready line comes from CONFINE_PROGRAM, the last step before the
    # program's timeout, so a failure of env, nice, setpriv, prlimit, or the confinement comes before it and is
    # reported as a sandbox failure, never as the program's exit status. The setup never prints it itself.
    assert sandbox.SETUP_SCRIPT.count(sandbox.READY_MARKER) == 1
    assert sandbox.CONFINE_PROGRAM.count(sandbox.READY_MARKER) == 1
    assert not [line for line in lines if sandbox.READY_MARKER in line and "echo" in line], lines
    assert sandbox.SETUP_SCRIPT.count(sandbox.DONE_MARKER) == 1
    assert sandbox.PRIVATE_DIRS == ("/tmp", "/var/tmp", "/dev/shm", "/run")


def test_setup_script_ends_stderr_with_a_line_of_its_own_on_any_failure_after_the_program(sandbox: ModuleType) -> None:
    # The trap is set right after the program's namespace has ended and cleared right before the done line, so
    # every failure the shell sees in between (set -e or the copy-back's exit) prints the stop line last.
    script = sandbox.SETUP_SCRIPT
    assert script.split("\n").count(TRAP_LINE) == 1, script
    program_end = script.index("|| status=$?\n") + len("|| status=$?\n")
    assert script.index(TRAP_LINE) == program_end, script[program_end : program_end + 80]
    assert script.index(TRAP_LINE) < script.index('*) exit "$copied" ;;') < script.index("trap - EXIT\n")
    assert script.count("trap ") == 2


def test_setup_script_reads_the_clock_right_before_and_right_after_the_programs_namespace(
    sandbox: ModuleType,
) -> None:
    # Round-3 finding RT1-2: the hang test must use the program's own time, not the setup's or the copy-back's,
    # so the setup reads the kernel's clock (a read builtin: no tool, no substitution) around the program's
    # namespace and prints both readings in its done line, which nothing the program prints can follow.
    script = sandbox.SETUP_SCRIPT
    start_read, end_read = f"read -r started rest < {UPTIME}\n", f"read -r ended rest < {UPTIME}\n"
    assert script.count(start_read) == 1 and script.count(end_read) == 1, script
    assert script.count(UPTIME) == 2
    assert script.index(start_read) + len(start_read) == script.index("unshare --pid"), script
    after_trap = script.index(TRAP_LINE) + len(TRAP_LINE) + 1
    assert script.index(end_read) == after_trap, script[after_trap - 80 : after_trap + 80]
    assert script.index(end_read) < script.index("umount /tmp\n") < script.index("python3 -I -S -c '", after_trap)


def test_setup_script_clears_its_core_filter_once_before_proc_turns_read_only(sandbox: ModuleType) -> None:
    # R6 second layer: coredump_filter is inherited by every later process and kept across exec, and it must be
    # written before the read-only step, which covers /proc.
    script = sandbox.SETUP_SCRIPT
    assert script.split("\n").count(CORE_FILTER_LINE) == 1
    assert script.index(USERNS_LINE) < script.index(CORE_FILTER_LINE) < script.index("python3 -I -S -c '")


def test_setup_script_hides_sys_device_entries_and_var_entries_read_only(sandbox: ModuleType) -> None:
    # The review's /sys and socket findings: device attribute files and daemon sockets under /var go out of
    # view, /var/tmp (a private dir) and /sys/devices/system (cpu topology) stay, and symbolic links are skipped.
    loop = (
        "for dir in /sys/class /sys/bus /sys/devices/* /var/*; do\n"
        "  case $dir in\n"
        "    /sys/devices/system | /var/tmp) ;;\n"
        "    *)\n"
        '      if [ -d "$dir" ] && [ ! -L "$dir" ]; then\n'
        '        mount -t tmpfs -o ro,size=4k,nr_inodes=8,mode=0555 lassi-sys "$dir"\n'
        "      fi\n"
    )
    script = sandbox.SETUP_SCRIPT
    assert script.count(loop) == 1, script
    # After the hidden roots turn read-only and before the private dirs, so /var/tmp gets its private tmpfs.
    last_hide = script.rindex('mount -o remount,bind,ro "$root"')
    assert last_hide < script.index(loop) < script.index("lassi-private") < script.index("awk '")
    assert set(sandbox.SYSTEM_DIRS) == {"/var", "/sys"}


def test_setup_script_unmounts_its_private_mounts_before_the_copy_back_and_never_rehashes(sandbox: ModuleType) -> None:
    script = sandbox.SETUP_SCRIPT
    after = script[script.index("|| status=$?") :]
    order = ["umount /tmp\n", 'umount "$dir"\n', "umount /dev/shm\n", 'umount "$workdir"\n', "python3 -I -S -c '"]
    positions = [after.index(step) for step in order]
    assert positions == sorted(positions), positions
    assert "for dir in /var/tmp /run; do\n  if [ -d \"$dir\" ]; then\n    umount \"$dir\"" in after
    # PATH is the constant from env -i, so nothing needs to be looked up again.
    assert "hash" not in script


def test_setup_script_refuses_numbers_with_a_leading_zero(sandbox: ModuleType) -> None:
    # tmpfs reads size=0 as no limit and a leading 0 as octal, so the disk cap and the root count must be
    # plain positive decimals.
    assert "    '' | 0* | *[!0-9]*) exit 2 ;;\n" in sandbox.SETUP_SCRIPT


def test_readonly_program_is_one_recursive_mount_setattr_call(sandbox: ModuleType) -> None:
    # mount_setattr(AT_FDCWD, "/", AT_RECURSIVE, {attr_set=MOUNT_ATTR_RDONLY}): syscall 442 on x86_64 and in
    # the generic table, AT_RECURSIVE 0x8000 (spike probe B). It exits nonzero on failure so set -e stops setup.
    tree = ast.parse(sandbox.READONLY_PROGRAM)
    constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}
    assert {442, 0x8000, b"/"} <= constants, constants
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert imported <= {"ctypes", "os", "sys"}, imported
    assert "exit" in sandbox.READONLY_PROGRAM or "SystemExit" in sandbox.READONLY_PROGRAM


# ---------------------------------------------------------------------------
# SETUP_SCRIPT under stubs: the full run


def test_setup_under_stubs_runs_each_step_once_and_the_program_once_in_order(
    sandbox: ModuleType, full_run: tuple[StubRun, dict[str, str]]
) -> None:
    run, _values = full_run
    assert run.returncode == 7, run
    assert run.stderr == STUB_ORDER, run
    assert (run.userns_limit, run.core_filter) == ("0\n", "0\n"), run
    check, program = the_check_call(run.calls), the_program_call(run.calls)
    assert run.calls[check] == ["awk", sandbox.MOUNT_CHECK, "/proc/self/mountinfo"]
    assert check < program
    between = [call for call in run.calls[check + 1 : program] if call[0] in ("mount", "umount")]
    assert between == [], between


def test_setup_under_stubs_makes_every_mount_read_only_first(
    sandbox: ModuleType, full_run: tuple[StubRun, dict[str, str]]
) -> None:
    # R2: one recursive mount_setattr before the setup mounts anything, so every host mount turns read-only
    # and the setup's own mounts stay writable; the user namespace limit is already written by then.
    run, _values = full_run
    assert run.calls[0] == ["python3", "-I", "-S", "-c", sandbox.READONLY_PROGRAM], run.calls[:2]
    assert run.userns_before_python3[:1] == [True], run


def test_setup_under_stubs_builds_a_private_dev(full_run: tuple[StubRun, dict[str, str]]) -> None:
    # R1: a tmpfs holding binds of six host nodes, a new devpts instance, a private shm, and the standard
    # links, remounted read-only and moved onto /dev.
    run, _values = full_run
    calls = run.calls
    dev_index, dev_args = only(typed_mounts(calls, "tmpfs", "lassi-dev"), "the lassi-dev tmpfs")
    dev = dev_args[5]
    host_binds = [(i, src, dst) for i, src, dst in moves(calls, "--bind") if src.startswith("/dev/")]
    assert sorted(src for _i, src, _dst in host_binds) == sorted(f"/dev/{name}" for name in DEV_NODES), host_binds
    for index, src, dst in host_binds:
        name = src[len("/dev/") :]
        assert dst == f"{dev}/{name}", (src, dst)
        assert index_of(calls, ["touch", dst]) < index, (src, calls)
    pts_index, pts_args = only(typed_mounts(calls, "devpts"), "the devpts instance")
    assert pts_args[4:] == ["lassi-devpts", f"{dev}/pts"], pts_args
    assert {"newinstance", "ptmxmode"} <= set(option_map(pts_args[3])), pts_args
    assert option_map(pts_args[3])["ptmxmode"] == "0666", pts_args
    shm_index, shm_args = only(typed_mounts(calls, "tmpfs", "lassi-shm"), "the lassi-shm tmpfs")
    assert shm_args[5] == f"{dev}/shm", shm_args
    links = sorted(call[1:] for call in calls if call[0] == "ln")
    assert links == sorted(["-s", target, f"{dev}/{name}"] for name, target in DEV_LINKS.items()), links
    readonly = [(i, options) for i, options in remounts(calls, dev) if "ro" in options and "bind" in options]
    assert len(readonly) == 1, remounts(calls, dev)
    moved = [i for i, src, dst in moves(calls, "--move") if (src, dst) == (dev, "/dev")]
    assert len(moved) == 1, moves(calls, "--move")
    built = [dev_index, pts_index, shm_index, *(i for i, _s, _d in host_binds)]
    built += [i for i, call in enumerate(calls) if call[0] == "ln"]
    assert max(built) < readonly[0][0] < moved[0] < the_check_call(calls)


def test_setup_under_stubs_stages_the_workdir_on_a_size_capped_overlay(
    full_run: tuple[StubRun, dict[str, str]],
) -> None:
    # R5: the overlay's upper layer lives on a tmpfs with size=<disk cap>m and an inode cap; the host workdir
    # is only the read-only lower layer, and the -o strings hold constant staging paths, never an argument.
    run, values = full_run
    calls = run.calls
    lower = [(i, dst) for i, src, dst in moves(calls, "--bind") if src == values["workdir"]]
    assert len(lower) == 1, moves(calls, "--bind")
    lower_index, lower_dir = lower[0]
    wd_index, wd_args = only(typed_mounts(calls, "tmpfs", "lassi-wd"), "the capped lassi-wd tmpfs")
    wd_options = option_map(wd_args[3])
    assert wd_options.get("size") == STUB_DISK and wd_options.get("nr_inodes") == "4096", wd_args
    ov_index, ov_args = only(typed_mounts(calls, "overlay"), "the workdir overlay")
    assert ov_args[4].startswith("lassi-"), ov_args
    assert ov_args[3].split(",")[0] == "userxattr", ov_args
    ov_options = option_map(ov_args[3])
    assert ov_options.get("lowerdir") == lower_dir, ov_args
    for key in ("upperdir", "workdir"):
        assert ov_options.get(key, "").startswith(wd_args[5] + "/"), ov_args
    assert lower_index < ov_index and wd_index < ov_index
    arguments = [value for value in values.values() if value] + ["./main", "one two"]
    option_strings = [call[call.index("-o") + 1] for call in calls if call[0] == "mount" and "-o" in call]
    assert not [(o, a) for o in option_strings for a in arguments if a in o], option_strings
    onto_workdir = [i for i, src, dst in moves(calls, "--move") if (src, dst) == (ov_args[5], values["workdir"])]
    assert len(onto_workdir) == 1, moves(calls, "--move")


def test_setup_under_stubs_hides_the_roots_and_re_exposes_the_workdir_harness_and_toolchains(
    full_run: tuple[StubRun, dict[str, str]],
) -> None:
    # R3: each hidden root gets a tmpfs, later remounted read-only; the harness (bind) and the toolchains root
    # (rbind) are staged before the hiding and moved back after it, like the overlay onto the workdir.
    run, values = full_run
    calls = run.calls
    check = the_check_call(calls)
    hides = typed_mounts(calls, "tmpfs", "lassi-hide")
    assert [args[5] for _i, args in hides] == [values["root_a"], values["root_b"]], hides
    first_hide, last_hide = hides[0][0], hides[-1][0]
    for index, args in hides:
        readonly = [i for i, options in remounts(calls, args[5]) if {"bind", "ro"} <= set(options)]
        assert len(readonly) == 1 and index < readonly[0] < check, (args, remounts(calls, args[5]))
    staged = {}
    for flag, key in (("--bind", "harness"), ("--rbind", "toolchains")):
        found = [(i, dst) for i, src, dst in moves(calls, flag) if src == values[key]]
        assert len(found) == 1 and found[0][0] < first_hide, (key, moves(calls, flag))
        staged[key] = found[0][1]
    for key in ("harness", "toolchains"):
        back = [i for i, src, dst in moves(calls, "--move") if (src, dst) == (staged[key], values[key])]
        assert len(back) == 1 and last_hide < back[0] < check, (key, moves(calls, "--move"))
    workdir_moves = [i for i, _src, dst in moves(calls, "--move") if dst == values["workdir"]]
    assert len(workdir_moves) == 1 and last_hide < workdir_moves[0] < check, moves(calls, "--move")
    for key in ("workdir", "harness", "toolchains"):
        made = [i for i, call in enumerate(calls) if call[:2] == ["mkdir", "-p"] and values[key] in call[2:]]
        assert made and first_hide < made[0] < check, (key, calls)


def test_setup_under_stubs_mounts_the_private_tmp_dirs_last(
    sandbox: ModuleType, full_run: tuple[StubRun, dict[str, str]]
) -> None:
    run, _values = full_run
    calls = run.calls
    check = the_check_call(calls)
    private = [(i, args) for i, args in typed_mounts(calls, "tmpfs") if args[3] == "size=64m,mode=1777"]
    points = [args[5] for _i, args in private]
    assert "/tmp" in points, private
    assert points == [p for p in ("/var/tmp", "/run", "/tmp") if p in points], private
    assert all(args[4].startswith("lassi-") for _i, args in private), private
    assert set(points) <= set(sandbox.PRIVATE_DIRS), points
    last_mount = max(i for i, call in enumerate(calls[:check]) if call[0] == "mount")
    assert calls[last_mount] == ["mount", *private[-1][1]] and private[-1][1][5] == "/tmp", calls[last_mount]


def test_setup_under_stubs_names_every_filesystem_it_mounts_lassi(full_run: tuple[StubRun, dict[str, str]]) -> None:
    # MOUNT_CHECK trusts only writable mounts whose source starts with lassi-, so every mount -t must use one.
    run, _values = full_run
    typed = [call for call in run.calls if call[0] == "mount" and call[1:2] == ["-t"]]
    assert typed, run.calls
    assert all(len(call) == 7 and call[5].startswith("lassi-") for call in typed), typed


def test_setup_under_stubs_runs_the_program_in_nested_pid_and_ipc_namespaces_at_core_limit_one(
    sandbox: ModuleType, full_run: tuple[StubRun, dict[str, str]]
) -> None:
    # R6 and R7: the program is never pid 1 (timeout is), runs at core limit 1 under the seccomp filter that
    # keeps it there, and dies with its namespace. R5: no file it writes may pass the disk cap. The review's
    # round-2 isolation question: System V IPC objects the program leaves behind die with its own IPC
    # namespace, so they do not stay charged to the run's memory while the copy-back runs.
    run, values = full_run
    call = run.calls[the_program_call(run.calls)]
    assert call[8:9] and call[8].startswith("PATH="), call
    assert call[:8] + call[9:] == [
        "unshare", "--pid", "--ipc", "--fork", "--kill-child", "--", "env",
        "-i", f"HOME={values['workdir']}", "LANG=C.UTF-8", "TMPDIR=/tmp",
        "nice", "-n", "19",
        "setpriv", "--no-new-privs", "--inh-caps=-all", "--bounding-set=-all", "--",
        "prlimit", "--cpu=6", "--core=1", f"--fsize={STUB_DISK}", "--",
        "python3", "-I", "-S", "-c", sandbox.CONFINE_PROGRAM,
        "timeout", "--kill-after=2", "3", "./main", "one two",
    ], call  # fmt: skip


def test_setup_under_stubs_gives_the_program_exactly_its_environment_after_the_marker(
    sandbox: ModuleType, tmp_path: Path
) -> None:
    # P0.20: with SandboxSpec.environment set, the marker and one NAME=value element per variable come before
    # the program argv, and the chain ends `timeout ... env -i -- NAME=value... <argv>`: env (from the setup's
    # PATH, like every tool of the chain) gives the program exactly those variables, and no shell parses them.
    layout, _values = stub_layout(tmp_path)
    head, program = layout[:-2], layout[-2:]
    variables = ["LANG=C", "PATH=/opt/a b/bin:$(reboot)", "TMPDIR=/x/t `id`"]
    run = run_setup_with_stubs(sandbox, tmp_path, [*head, sandbox.ENVIRONMENT_MARKER, *variables, *program])
    call = run.calls[the_program_call(run.calls)]
    tail = ["timeout", "--kill-after=2", "3", "env", "-i", "--", *variables, *program]
    assert call[-len(tail) :] == tail, call
    assert sandbox.ENVIRONMENT_MARKER not in call
    assert run.returncode == 0, run.stderr


def test_setup_under_stubs_makes_nothing_writable_again_before_the_program_ends(
    full_run: tuple[StubRun, dict[str, str]],
) -> None:
    run, _values = full_run
    program = the_program_call(run.calls)
    before = [call for call in run.calls[:program] if call[0] == "mount" and "-o" in call]
    reopened = [call for call in before if "rw" in call[call.index("-o") + 1].split(",")]
    assert reopened == [], reopened


def test_setup_under_stubs_copies_back_through_a_bind_made_after_the_program(
    sandbox: ModuleType, full_run: tuple[StubRun, dict[str, str]]
) -> None:
    # R5: after the program's namespace ended, unmount the overlay, bind the staged lower (the host workdir)
    # read-write, and copy the upper layer's regular files into it.
    run, values = full_run
    calls = run.calls
    program = the_program_call(calls)
    lower_dir = next(dst for _i, src, dst in moves(calls, "--bind") if src == values["workdir"])
    _ov_index, ov_args = only(typed_mounts(calls, "overlay"), "the workdir overlay")
    upper = option_map(ov_args[3])["upperdir"]
    unmount = index_of(calls, ["umount", values["workdir"]])
    targets = [(i, dst) for i, src, dst in moves(calls, "--bind") if src == lower_dir and i > program]
    assert len(targets) == 1, moves(calls, "--bind")
    bind_index, target = targets[0]
    writable = [i for i, options in remounts(calls, target) if {"bind", "rw"} <= set(options)]
    assert len(writable) == 1, remounts(calls, target)
    copy = index_of(calls, ["python3", "-I", "-S", "-c", sandbox.COPY_BACK_PROGRAM, upper, target, STUB_DISK])
    assert program < unmount < copy and program < bind_index < writable[0] < copy
    assert copy == len(calls) - 1, calls[copy:]
    # The private tmpfs mounts go first, so their memory is free for the copy-back.
    private = [index_of(calls, ["umount", point]) for point in ("/tmp", "/dev/shm")]
    assert all(program < index < unmount for index in private), calls[program:]


# ---------------------------------------------------------------------------
# SETUP_SCRIPT under stubs: failures stop it before the program


@pytest.mark.parametrize("missing", CHECKED_TOOLS)
def test_setup_script_stops_before_doing_anything_when_a_tool_is_missing(
    sandbox: ModuleType, tmp_path: Path, missing: str
) -> None:
    layout, _values = stub_layout(tmp_path)
    run = run_setup_with_stubs(sandbox, tmp_path, layout, missing=missing)
    assert run.returncode != 0, run
    assert sandbox.READY_MARKER not in run.stderr, run
    assert (run.calls, run.userns_limit) == ([], None), run


def test_setup_script_stops_before_the_program_when_a_mount_fails(sandbox: ModuleType, tmp_path: Path) -> None:
    layout, values = stub_layout(tmp_path)
    run = run_setup_with_stubs(sandbox, tmp_path, layout, fail_target=values["root_b"])
    assert run.returncode == 32, run
    assert sandbox.READY_MARKER not in run.stderr, run
    assert not [call for call in run.calls if call[0] in ("awk", "unshare")], run.calls
    assert run.calls[-1][0] == "mount" and run.calls[-1][-1] == values["root_b"], run.calls


def test_setup_script_fails_closed_when_the_mount_check_fails(sandbox: ModuleType, tmp_path: Path) -> None:
    # R2: a writable mount the setup did not make stops it before the ready marker; the program never runs.
    layout, _values = stub_layout(tmp_path)
    run = run_setup_with_stubs(sandbox, tmp_path, layout, check_fails=True)
    assert run.returncode != 0, run
    assert sandbox.READY_MARKER not in run.stderr, run
    assert "writable host mount: /var/tmp" in run.stderr, run
    assert not [call for call in run.calls if call[0] == "unshare"], run.calls


def test_setup_script_stops_before_the_program_when_the_user_namespace_limit_cannot_be_written(
    sandbox: ModuleType, tmp_path: Path
) -> None:
    layout, _values = stub_layout(tmp_path)
    unwritable = tmp_path / "no-such-dir" / "max_user_namespaces"
    run = run_setup_with_stubs(sandbox, tmp_path, layout, userns_target=unwritable)
    assert run.returncode != 0, run
    assert sandbox.READY_MARKER not in run.stderr, run
    assert (run.calls, run.userns_limit) == ([], None), run


def test_setup_script_prints_no_done_line_when_copy_back_fails(sandbox: ModuleType, tmp_path: Path) -> None:
    layout, _values = stub_layout(tmp_path)
    run = run_setup_with_stubs(sandbox, tmp_path, layout, copy_back_status=1)
    assert run.returncode != 0, run
    assert sandbox.READY_MARKER in run.stderr and sandbox.DONE_MARKER not in run.stderr, run
    assert run.stderr.endswith("stub:python3\n" + STOP_LINE), run
    assert len([call for call in run.calls if call[0] == "unshare"]) == 1, run.calls


@pytest.mark.parametrize("copy_back_status", [1, 137])
def test_setup_script_never_lets_a_done_line_the_program_printed_end_stderr_after_a_failure(
    sandbox: ModuleType, tmp_path: Path, copy_back_status: int
) -> None:
    # The review's round-2 question: the program ends its stderr with the done line itself, and the copy-back
    # then dies without a word (137: a memory kill). The setup's own stop line comes last, so Sandbox.run sees
    # no done line and raises instead of reporting the run as the program's.
    layout, _values = stub_layout(tmp_path)
    forged = StubProgram(status=0, stderr=DONE_LINE)
    run = run_setup_with_stubs(sandbox, tmp_path, layout, copy_back_status=copy_back_status, program=forged)
    assert run.returncode == copy_back_status, run
    copy_back_said = "" if copy_back_status >= 128 else "stub:python3\n"
    assert run.stderr.endswith(DONE_LINE + copy_back_said + STOP_LINE), run
    runner = FakeRunner(returncode=run.returncode, stderr=run.stderr, ready=False)
    with pytest.raises(sandbox.SandboxUnavailableError, match="copy-back"):
        sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)


def test_setup_script_reports_no_ready_line_when_the_programs_chain_fails_before_the_confinement_is_done(
    sandbox: ModuleType, tmp_path: Path
) -> None:
    # The review's round-2 question: env, nice, setpriv, prlimit, or CONFINE_PROGRAM failing (here the stub
    # chain exits 1 without the ready line) is the sandbox's failure: the setup still tidies up and copies back,
    # and Sandbox.run raises SandboxUnavailableError instead of reporting status 1 as the program's.
    layout, _values = stub_layout(tmp_path)
    broken = StubProgram(status=1, ready=False, stderr="lassi-sandbox confine: [Errno 22] prctl failed\n")
    run = run_setup_with_stubs(sandbox, tmp_path, layout, program=broken)
    assert run.returncode == 1, run
    assert sandbox.READY_MARKER not in run.stderr and run.stderr.endswith(DONE_LINE), run
    runner = FakeRunner(returncode=run.returncode, stderr=run.stderr, ready=False)
    with pytest.raises(sandbox.SandboxUnavailableError, match="did not run") as caught:
        sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert "lassi-sandbox confine" in str(caught.value)


def test_setup_script_without_harness_or_toolchains_exposes_only_the_workdir(
    sandbox: ModuleType, tmp_path: Path
) -> None:
    layout, values = stub_layout(tmp_path, exposures=False)
    run = run_setup_with_stubs(sandbox, tmp_path, layout)
    assert (run.returncode, run.stderr) == (0, STUB_ORDER), run
    assert not [call for call in run.calls if "" in call], run.calls
    assert not moves(run.calls, "--rbind"), run.calls
    onto = [dst for _i, _src, dst in moves(run.calls, "--move") if dst != "/dev"]
    assert onto == [values["workdir"]], moves(run.calls, "--move")


def test_setup_script_marks_the_done_line_when_the_copy_back_left_something_out(
    sandbox: ModuleType, tmp_path: Path
) -> None:
    # The copy-back exits 3 when it refused the files past the disk cap or left entries out; the program's run
    # still counts, so the setup prints its done line with the incomplete mark and the program's status.
    layout, _values = stub_layout(tmp_path)
    run = run_setup_with_stubs(sandbox, tmp_path, layout, program_status=5, copy_back_status=3)
    assert run.returncode == 5, run
    assert run.stderr == STUB_ORDER.replace(DONE_LINE, INCOMPLETE_LINE), run


@pytest.mark.parametrize("status", [1, 2, 4, 127])
def test_setup_script_stops_without_a_done_line_on_any_other_copy_back_failure(
    sandbox: ModuleType, tmp_path: Path, status: int
) -> None:
    layout, _values = stub_layout(tmp_path)
    run = run_setup_with_stubs(sandbox, tmp_path, layout, program_status=5, copy_back_status=status)
    assert run.returncode == status, run
    assert sandbox.DONE_MARKER not in run.stderr and run.stderr.endswith(STOP_LINE), run


@pytest.mark.parametrize(
    ("position", "value"),
    [(3, "0"), (3, "0" + STUB_DISK), (3, ""), (3, "8m"), (4, "0"), (4, "02"), (4, "-1")],
    ids=["disk-zero", "disk-padded", "disk-empty", "disk-suffix", "count-zero", "count-padded", "count-negative"],
)
def test_setup_script_stops_before_anything_on_a_bad_number(
    sandbox: ModuleType, tmp_path: Path, position: int, value: str
) -> None:
    layout, _values = stub_layout(tmp_path)
    layout[position] = value
    run = run_setup_with_stubs(sandbox, tmp_path, layout)
    assert run.returncode == 2, run
    assert (run.calls, run.userns_limit, run.core_filter) == ([], None, None), run


def test_setup_under_stubs_mounts_system_hides_read_only_and_never_on_the_kept_entries(
    full_run: tuple[StubRun, dict[str, str]],
) -> None:
    # What the loop meets depends on the host's /sys and /var, so this checks every lassi-sys mount it made:
    # read-only, on a /sys or /var entry, never on /var/tmp or /sys/devices/system, and before the private dirs.
    run, _values = full_run
    calls = run.calls
    hides = typed_mounts(calls, "tmpfs", "lassi-sys")
    private = [i for i, _args in typed_mounts(calls, "tmpfs", "lassi-private")]
    private += [i for i, _args in typed_mounts(calls, "tmpfs", "lassi-tmp")]
    for index, args in hides:
        assert option_map(args[3]).keys() >= {"ro", "size", "nr_inodes", "mode"}, args
        assert args[5].startswith(("/sys/class", "/sys/bus", "/sys/devices/", "/var/")), args
        assert args[5] not in ("/var/tmp", "/sys/devices/system"), args
        assert index < min(private), (index, private)


# ---------------------------------------------------------------------------
# MOUNT_CHECK: the fail-closed check of /proc/self/mountinfo (R2), on synthetic lines

# Mounts a finished setup may leave writable, or that are read-only: none of them may fail the check.
ACCEPTED_MOUNTINFO = [
    "22 1 259:3 / / ro,relatime shared:1 - ext4 /dev/nvme0n1p2 rw,errors=remount-ro",
    "23 22 0:21 / /proc ro,nosuid,nodev,noexec,relatime - proc proc rw",
    "24 22 0:22 / /sys ro,nosuid,nodev,noexec,relatime shared:7 master:2 - sysfs sysfs rw",
    "25 22 0:45 / /scratch ro,relatime - tmpfs lassi-hide rw,size=1024k,nr_inodes=1024,mode=755",
    "26 25 0:46 / /scratch/runs/t/build rw,relatime - overlay lassi-workdir rw,lowerdir=/s/lo,upperdir=/s/u,userxattr",
    "27 22 0:47 / /dev ro,nosuid,noexec,relatime - tmpfs lassi-dev rw,size=64k,nr_inodes=64,mode=755",
    "28 27 0:5 /null /dev/null ro,nosuid,relatime shared:2 - devtmpfs udev rw,size=100k,mode=755",
    "29 27 0:48 / /dev/pts rw,relatime - devpts lassi-devpts rw,mode=620,ptmxmode=666",
    "30 27 0:49 / /dev/shm rw,nosuid,nodev,relatime - tmpfs lassi-shm rw,size=65536k",
    "31 22 0:50 / /tmp rw,relatime - tmpfs lassi-stage rw,size=1024k,nr_inodes=64,mode=700",
    "32 31 0:51 / /tmp/s/wd rw,relatime - tmpfs lassi-wd rw,size=8192k,nr_inodes=4096,mode=755",
    "33 22 0:52 / /tmp rw,relatime - tmpfs lassi-private rw,size=65536k",
    "34 22 0:53 / /mnt/a\\040b ro,relatime - ext4 /dev/sdb1 rw",
    "35 22 0:54 / /run/netns/x ro - nsfs nsfs rw",
]

# One writable mount the setup did not make per case, and the mount point the check must name.
REFUSED_MOUNTINFO = {
    "host-ext4": ("90 22 259:5 /x /var/tmp rw,relatime shared:5 - ext4 /dev/nvme23n1p1 rw", "/var/tmp"),
    "host-tmpfs": (
        "91 22 0:60 / /run/user/1025 rw,nosuid,nodev,relatime shared:9 master:3 - tmpfs tmpfs rw,size=100k",
        "/run/user/1025",
    ),
    "host-overlay": (
        "92 22 0:61 / /var/lib/c/merged rw,relatime - overlay overlay rw,lowerdir=/l,upperdir=/u",
        "/var/lib/c/merged",
    ),
    "host-devpts": ("93 27 0:24 / /dev/pts rw,nosuid,noexec,relatime shared:3 - devpts devpts rw,mode=620", "/dev/pts"),
    "lassi-name-on-a-host-fs": ("94 22 259:6 / /srv rw,relatime - ext4 lassi-disk rw", "/srv"),
    "nsfs": ("95 22 0:4 net:[4026532000] /run/netns/y rw - nsfs nsfs rw", "/run/netns/y"),
    "escaped-point": ("96 22 259:7 / /mnt/a\\040c rw,relatime - xfs /dev/sdc1 rw", "/mnt/a\\040c"),
    "cgroup": ("97 22 0:26 / /sys/fs/cgroup rw,nosuid,nodev - cgroup2 cgroup2 rw", "/sys/fs/cgroup"),
}


def run_mount_check(sandbox: ModuleType, tmp_path: Path, lines: list[str]) -> subprocess.CompletedProcess[str]:
    """Run MOUNT_CHECK under the awk on PATH against `lines` as a mountinfo file; skip when there is no awk."""
    awk = shutil.which("awk")
    if awk is None:
        pytest.skip("awk is not on PATH")
    program = tmp_path / "check.awk"
    program.write_text(sandbox.MOUNT_CHECK, encoding="ascii", newline="\n")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("".join(line + "\n" for line in lines), encoding="ascii", newline="\n")
    argv = [awk, "-f", program.as_posix(), mountinfo.as_posix()]
    return subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)


def test_mount_check_accepts_read_only_mounts_and_the_setups_own_writable_ones(
    sandbox: ModuleType, tmp_path: Path
) -> None:
    done = run_mount_check(sandbox, tmp_path, ACCEPTED_MOUNTINFO)
    assert (done.returncode, done.stdout, done.stderr) == (0, "", ""), done


@pytest.mark.parametrize("case", sorted(REFUSED_MOUNTINFO))
def test_mount_check_refuses_any_other_writable_mount_and_names_it(
    sandbox: ModuleType, tmp_path: Path, case: str
) -> None:
    line, point = REFUSED_MOUNTINFO[case]
    lines = [*ACCEPTED_MOUNTINFO[:5], line, *ACCEPTED_MOUNTINFO[5:]]
    done = run_mount_check(sandbox, tmp_path, lines)
    assert done.returncode != 0, done
    assert done.stdout == "", done
    assert point in done.stderr, done


# ---------------------------------------------------------------------------
# COPY_BACK_PROGRAM: the upper layer's regular files and directories reach the host workdir (R5)


def run_copy_back(
    sandbox: ModuleType,
    upper: Path,
    target: Path,
    cap: int = 1 << 30,
    program: str | None = None,
    prefix: Sequence[str] = (),
) -> subprocess.CompletedProcess[bytes]:
    """Run COPY_BACK_PROGRAM (or `program`) as SETUP_SCRIPT does (python3 -I -S -c) with this interpreter.

    `prefix` goes before the interpreter, for example namespace_root_prefix().
    """
    code = sandbox.COPY_BACK_PROGRAM if program is None else program
    argv = [*prefix, sys.executable, "-I", "-S", "-c", code, str(upper), str(target), str(cap)]
    return subprocess.run(argv, capture_output=True, timeout=60, check=False)


def namespace_root_prefix() -> list[str]:
    """Return [] when this process is root, else ["unshare", "-r"] when that works here; skip the test otherwise.

    The sandbox runs its copy-back as root of the run's user namespace, where it may read a file the program left
    at mode 0; a test that needs that must run the copy-back the same way.
    """
    if os.geteuid() == 0:
        return []
    unshare = shutil.which("unshare")
    if unshare is None or subprocess.run([unshare, "-r", "true"], capture_output=True, check=False).returncode != 0:
        pytest.skip("reading a mode-0 file needs root or an unprivileged user namespace (unshare -r)")
    return [unshare, "-r"]


def tree_of(root: Path) -> dict[str, bytes | None]:
    """Return every entry under `root` as relative POSIX path -> file bytes, or None for a directory."""
    found: dict[str, bytes | None] = {}
    for path in sorted(root.rglob("*")):
        found[path.relative_to(root).as_posix()] = None if path.is_dir() else path.read_bytes()
    return found


def test_copy_back_copies_regular_files_and_directories_and_keeps_the_rest(sandbox: ModuleType, tmp_path: Path) -> None:
    upper, target = tmp_path / "upper", tmp_path / "target"
    (upper / "sub" / "deeper").mkdir(parents=True)
    (upper / "empty").mkdir()
    (upper / "a.txt").write_bytes(b"new a\n")
    (upper / "sub" / "b.txt").write_bytes(b"b\n")
    (upper / "sub" / "deeper" / "c.bin").write_bytes(bytes(range(256)))
    (target / "sub").mkdir(parents=True)
    (target / "a.txt").write_bytes(b"old a\n")
    (target / "keep.txt").write_bytes(b"keep\n")
    (target / "sub" / "old.txt").write_bytes(b"old\n")
    done = run_copy_back(sandbox, upper, target)
    assert (done.returncode, done.stdout, done.stderr) == (0, b"", b""), done
    assert (target / "a.txt").read_bytes() == b"new a\n"
    assert (target / "sub" / "b.txt").read_bytes() == b"b\n"
    assert (target / "sub" / "deeper" / "c.bin").read_bytes() == bytes(range(256))
    assert (target / "empty").is_dir()
    assert (target / "keep.txt").read_bytes() == b"keep\n"
    assert (target / "sub" / "old.txt").read_bytes() == b"old\n"


def test_copy_back_skips_symbolic_links_and_special_files(sandbox: ModuleType, tmp_path: Path) -> None:
    upper, target = tmp_path / "upper", tmp_path / "target"
    (upper / "dir").mkdir(parents=True)
    target.mkdir()
    (upper / "real.txt").write_bytes(b"real\n")
    try:
        os.symlink(tmp_path / "outside.txt", upper / "leak")
        os.symlink(upper / "dir", upper / "dir-link", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"cannot create a symlink here: {exc}")
    if hasattr(os, "mkfifo"):
        os.mkfifo(upper / "fifo")
    done = run_copy_back(sandbox, upper, target)
    assert done.returncode == 0, done
    assert sorted(path.name for path in target.iterdir()) == ["dir", "real.txt"]


def test_copy_back_never_writes_through_a_symbolic_link_in_the_host_workdir(
    sandbox: ModuleType, tmp_path: Path
) -> None:
    # A host symlink in the workdir (the lower layer) that the program replaced must not carry a write outside.
    upper, target, outside = tmp_path / "upper", tmp_path / "target", tmp_path / "outside"
    for directory in (upper / "escape", target, outside):
        directory.mkdir(parents=True)
    (outside / "victim.txt").write_bytes(b"outside\n")
    (upper / "victim").write_bytes(b"evil\n")
    (upper / "escape" / "x.txt").write_bytes(b"x\n")
    try:
        os.symlink(outside / "victim.txt", target / "victim")
        os.symlink(outside, target / "escape", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"cannot create a symlink here: {exc}")
    run_copy_back(sandbox, upper, target)
    assert (outside / "victim.txt").read_bytes() == b"outside\n"
    assert sorted(path.name for path in outside.iterdir()) == ["victim.txt"]


def test_copy_back_never_writes_through_a_hard_link_in_the_host_workdir(sandbox: ModuleType, tmp_path: Path) -> None:
    # The review's round-2 hard-link finding: a host workdir file hard-linked to a file outside (a build step
    # could leave one) is replaced by a new file renamed over it, so the file outside keeps its bytes, and no
    # temporary file is left behind.
    upper, target, outside = tmp_path / "upper", tmp_path / "target", tmp_path / "outside"
    for directory in (upper, target, outside):
        directory.mkdir()
    (outside / "victim.txt").write_bytes(b"outside\n")
    try:
        os.link(outside / "victim.txt", target / "data.txt")
    except OSError as exc:
        pytest.skip(f"cannot create a hard link here: {exc}")
    (upper / "data.txt").write_bytes(b"new\n")
    done = run_copy_back(sandbox, upper, target)
    assert (done.returncode, done.stderr) == (0, b""), done
    assert (outside / "victim.txt").read_bytes() == b"outside\n"
    assert tree_of(target) == {"data.txt": b"new\n"}


def test_copy_back_writes_each_file_under_a_temporary_name_and_renames_it_over_the_target(
    sandbox: ModuleType,
) -> None:
    program = sandbox.COPY_BACK_PROGRAM
    assert "O_TRUNC" not in program and "os.O_EXCL" in program and "O_NOFOLLOW" in program
    assert program.count("os.replace(") == 1 and program.count("os.unlink(") == 1


def test_copy_back_copies_nothing_when_the_new_files_pass_the_cap(sandbox: ModuleType, tmp_path: Path) -> None:
    # R5: the host workdir never grows past the cap; the run still counts, with the workdir marked incomplete.
    upper, target = tmp_path / "upper", tmp_path / "target"
    (upper / "sub").mkdir(parents=True)
    target.mkdir()
    (upper / "a.bin").write_bytes(b"a" * 600)
    (upper / "sub" / "b.bin").write_bytes(b"b" * 600)
    (target / "keep.txt").write_bytes(b"keep\n")
    done = run_copy_back(sandbox, upper, target, cap=1199)
    assert done.returncode == 3, done
    message = done.stderr.decode("ascii")
    assert message.startswith("lassi-sandbox copy-back: ") and "1200 bytes" in message and "1199-byte" in message
    assert message.count("\n") == 1, message
    assert tree_of(target) == {"keep.txt": b"keep\n"}
    done = run_copy_back(sandbox, upper, target, cap=1200)
    assert (done.returncode, done.stderr) == (0, b""), done
    assert tree_of(target) == {"a.bin": b"a" * 600, "keep.txt": b"keep\n", "sub": None, "sub/b.bin": b"b" * 600}


def test_copy_back_counts_every_hard_link_and_a_sparse_file_at_its_full_size(
    sandbox: ModuleType, tmp_path: Path
) -> None:
    # The review's R5 finding: tmpfs charges neither extra links nor holes, but copying writes each in full.
    upper, target = tmp_path / "upper", tmp_path / "target"
    upper.mkdir()
    target.mkdir()
    (upper / "one.bin").write_bytes(b"x" * 1000)
    try:
        for index in range(3):
            os.link(upper / "one.bin", upper / f"link{index}.bin")
    except OSError as exc:
        pytest.skip(f"cannot create a hard link here: {exc}")
    done = run_copy_back(sandbox, upper, target, cap=3999)
    assert done.returncode == 3 and b"4000 bytes" in done.stderr, done
    assert tree_of(target) == {}
    for path in upper.iterdir():
        path.unlink()
    with open(upper / "sparse.bin", "wb") as handle:
        handle.truncate(5000)
    done = run_copy_back(sandbox, upper, target, cap=4999)
    assert done.returncode == 3 and b"5000 bytes" in done.stderr, done
    assert tree_of(target) == {}


def test_copy_back_leaves_out_entries_past_the_depth_limit(sandbox: ModuleType, tmp_path: Path) -> None:
    # The review's recursion finding: a deep tree is walked without recursion and cut at COPY_DEPTH levels (path
    # components), and the program's run is reported with the workdir incomplete, never as a sandbox failure.
    upper, target = tmp_path / "upper", tmp_path / "target"
    target.mkdir()
    depth = sandbox.COPY_DEPTH + 2
    deep = upper.joinpath(*["d"] * depth)
    deep.mkdir(parents=True)
    for level in range(depth + 1):
        upper.joinpath(*["d"] * level, "f.txt").write_bytes(b"%d\n" % level)
    done = run_copy_back(sandbox, upper, target)
    assert done.returncode == 3, done
    assert done.stderr.startswith(b"lassi-sandbox copy-back: left out 2 entries more than 32 levels deep"), done
    copied = tree_of(target)
    kept = sandbox.COPY_DEPTH
    assert "/".join(["d"] * kept) in copied and "/".join(["d"] * (kept - 1) + ["f.txt"]) in copied
    assert not [path for path in copied if path.count("/") >= kept], sorted(copied)[-3:]


def test_copy_back_leaves_out_entries_past_the_path_length_limit(sandbox: ModuleType, tmp_path: Path) -> None:
    # COPY_PATH bytes of relative path is 1024, longer than most local paths can be here, so a copy of the
    # program with a 12-byte limit stands in for it; the limit line is checked to exist exactly once.
    line = "MAX_DEPTH, MAX_PATH, BLOCK = 32, 1024, 1048576\n"
    assert sandbox.COPY_BACK_PROGRAM.count(line) == 1
    program = sandbox.COPY_BACK_PROGRAM.replace(line, "MAX_DEPTH, MAX_PATH, BLOCK = 32, 12, 1048576\n")
    upper, target = tmp_path / "upper", tmp_path / "target"
    (upper / "sub").mkdir(parents=True)
    target.mkdir()
    (upper / "short.txt").write_bytes(b"s")
    (upper / "a-longer-name.txt").write_bytes(b"l")
    (upper / "sub" / "x.txt").write_bytes(b"x")
    (upper / "sub" / "long-x.txt").write_bytes(b"y")
    done = run_copy_back(sandbox, upper, target, program=program)
    assert done.returncode == 3 and b"left out 2 entries" in done.stderr, done
    assert tree_of(target) == {"short.txt": b"s", "sub": None, "sub/x.txt": b"x"}


def test_copy_back_skips_everything_under_an_entry_the_host_holds_as_another_type(
    sandbox: ModuleType, tmp_path: Path
) -> None:
    upper, target = tmp_path / "upper", tmp_path / "target"
    (upper / "a" / "deeper").mkdir(parents=True)
    target.mkdir()
    (upper / "a" / "x.txt").write_bytes(b"x")
    (upper / "a" / "deeper" / "y.txt").write_bytes(b"y")
    (target / "a").write_bytes(b"host file\n")
    done = run_copy_back(sandbox, upper, target)
    assert (done.returncode, done.stderr) == (0, b""), done
    assert tree_of(target) == {"a": b"host file\n"}


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_copy_back_adds_owner_read_write_and_removes_group_and_other_write(sandbox: ModuleType, tmp_path: Path) -> None:
    # The review's mode finding: a program cannot leave a world-writable or unreadable file on a shared host.
    upper, target = tmp_path / "upper", tmp_path / "target"
    upper.mkdir()
    target.mkdir()
    for name, mode in (("open.bin", 0o777), ("locked.bin", 0o000), ("tool", 0o751)):
        (upper / name).write_bytes(b"x")
        (upper / name).chmod(mode)
    (target / "locked.bin").write_bytes(b"old")
    # A read-only host file is replaced too: the new file is renamed over it, never opened for writing.
    (target / "tool").write_bytes(b"old")
    (target / "tool").chmod(0o444)
    # In the sandbox the copy-back runs as root of the run's user namespace, which may read the program's mode-0
    # file; an unprivileged test process may not, so it runs the copy-back the same way (unshare -r) or skips.
    done = run_copy_back(sandbox, upper, target, prefix=namespace_root_prefix())
    assert done.returncode == 0, done
    modes = {path.name: stat.S_IMODE(path.stat().st_mode) for path in target.iterdir()}
    assert modes == {"open.bin": 0o755, "locked.bin": 0o600, "tool": 0o751}
    assert (target / "tool").read_bytes() == b"x"


# ---------------------------------------------------------------------------
# CONFINE_PROGRAM: session, keyring, and the seccomp filter (R6 and the review's findings)

SECCOMP_ALLOW, SECCOMP_EPERM, SECCOMP_KILL = 0x7FFF0000, 0x00050001, 0x80000000
ARCH_X86_64, ARCH_I386, X32_BIT = 0xC000003E, 0x40000003, 0x40000000
# x86_64 syscall numbers and constants the filter names.
NR_READ, NR_WRITE, NR_SOCKET, NR_EXECVE, NR_SETRLIMIT, NR_EXIT_GROUP, NR_PRLIMIT64 = 0, 1, 41, 59, 160, 231, 302
NR_WAITID, NR_ADD_KEY, NR_REQUEST_KEY, NR_KEYCTL, NR_IOPRIO_SET = 247, 248, 249, 250, 251
RLIMIT_CORE, RLIMIT_NOFILE, AF_UNIX, AF_INET, AF_VSOCK = 4, 7, 1, 2, 40
# keyctl commands and special keyring ids (KEYCTL_GET_KEYRING_ID, KEYCTL_READ; @u and @us).
KEYCTL_GET_KEYRING_ID, KEYCTL_READ, KEY_SPEC_USER_KEYRING, KEY_SPEC_USER_SESSION_KEYRING = 0, 11, -4, -5


def confine_filter(sandbox: ModuleType) -> list[tuple[int, int, int, int]]:
    """Return CONFINE_PROGRAM's FILTER, imported under another __name__ so nothing is installed or run."""
    namespace: dict[str, object] = {"__name__": "lassi_confine_under_test"}
    exec(sandbox.CONFINE_PROGRAM, namespace)
    return list(namespace["FILTER"])  # type: ignore[call-overload]


def run_filter(
    rules: Sequence[tuple[int, int, int, int]], nr: int, args: Sequence[int] = (), arch: int = ARCH_X86_64
) -> int:
    """Run a classic-BPF seccomp filter over one struct seccomp_data (little-endian) and return its action."""
    data = struct.pack("<iIQ6Q", nr, arch, 0, *(list(args) + [0] * (6 - len(args))))
    pc, accumulator = 0, 0
    for _step in range(len(rules)):
        code, jt, jf, k = rules[pc]
        if code == 0x20:
            accumulator = struct.unpack_from("<I", data, k)[0]
            pc += 1
        elif code in (0x15, 0x35):
            taken = accumulator == k if code == 0x15 else accumulator >= k
            pc += 1 + (jt if taken else jf)
        elif code == 0x06:
            return k
        else:
            raise AssertionError(f"opcode {code:#x} is not one the filter may use")
    raise AssertionError("the filter ran off its end")


def test_confine_filter_is_a_filter_the_kernel_accepts(sandbox: ModuleType) -> None:
    rules = confine_filter(sandbox)
    assert 0 < len(rules) <= 4096
    assert rules[-1][0] == 0x06
    for index, (code, jt, jf, k) in enumerate(rules):
        assert code in (0x20, 0x15, 0x35, 0x06), (index, code)
        if code == 0x20:
            assert k % 4 == 0 and k < 64, (index, k)
        if code in (0x15, 0x35):
            assert index + 1 + max(jt, jf) < len(rules), (index, jt, jf)
        assert 0 <= jt < 256 and 0 <= jf < 256 and 0 <= k < 1 << 32, index


@pytest.mark.parametrize(
    ("nr", "args", "arch", "action"),
    [
        (NR_READ, (), ARCH_X86_64, SECCOMP_ALLOW),
        (NR_WRITE, (1,), ARCH_X86_64, SECCOMP_ALLOW),
        (NR_EXECVE, (), ARCH_X86_64, SECCOMP_ALLOW),
        (NR_EXIT_GROUP, (), ARCH_X86_64, SECCOMP_ALLOW),
        (NR_SETRLIMIT, (RLIMIT_CORE,), ARCH_X86_64, SECCOMP_EPERM),
        (NR_SETRLIMIT, (RLIMIT_CORE | 1 << 32,), ARCH_X86_64, SECCOMP_EPERM),
        (NR_SETRLIMIT, (RLIMIT_NOFILE,), ARCH_X86_64, SECCOMP_ALLOW),
        (NR_PRLIMIT64, (0, RLIMIT_CORE, 0x7FFE0000), ARCH_X86_64, SECCOMP_EPERM),
        (NR_PRLIMIT64, (0, RLIMIT_CORE, 1 << 32), ARCH_X86_64, SECCOMP_EPERM),
        (NR_PRLIMIT64, (0, RLIMIT_CORE, 0, 0x7FFE0000), ARCH_X86_64, SECCOMP_ALLOW),
        (NR_PRLIMIT64, (0, RLIMIT_NOFILE, 0x7FFE0000), ARCH_X86_64, SECCOMP_ALLOW),
        (NR_PRLIMIT64, (1, RLIMIT_NOFILE, 0x7FFE0000), ARCH_X86_64, SECCOMP_EPERM),
        (NR_PRLIMIT64, (1, RLIMIT_NOFILE, 0, 0x7FFE0000), ARCH_X86_64, SECCOMP_ALLOW),
        (NR_SOCKET, (AF_VSOCK, 1), ARCH_X86_64, SECCOMP_EPERM),
        (NR_SOCKET, (AF_INET, 1), ARCH_X86_64, SECCOMP_ALLOW),
        (NR_SOCKET, (AF_UNIX, 1), ARCH_X86_64, SECCOMP_ALLOW),
        (424, (), ARCH_X86_64, SECCOMP_ALLOW),
        (425, (), ARCH_X86_64, SECCOMP_EPERM),
        (426, (), ARCH_X86_64, SECCOMP_EPERM),
        (427, (), ARCH_X86_64, SECCOMP_EPERM),
        (428, (), ARCH_X86_64, SECCOMP_ALLOW),
        (435, (), ARCH_X86_64, SECCOMP_ALLOW),
        (X32_BIT | NR_SETRLIMIT, (RLIMIT_CORE,), ARCH_X86_64, SECCOMP_EPERM),
        (X32_BIT | NR_READ, (), ARCH_X86_64, SECCOMP_EPERM),
        (75, (RLIMIT_CORE,), ARCH_I386, SECCOMP_KILL),
        (NR_READ, (), ARCH_I386, SECCOMP_KILL),
        # Round-3 nit: a host key would likely stay reachable by its global serial (inferred from the kernel
        # source, not measured), so every keyring call fails; the neighboring syscalls still work.
        (NR_KEYCTL, (KEYCTL_GET_KEYRING_ID, KEY_SPEC_USER_KEYRING & 0xFFFFFFFF), ARCH_X86_64, SECCOMP_EPERM),
        (NR_KEYCTL, (KEYCTL_READ, KEY_SPEC_USER_SESSION_KEYRING & 0xFFFFFFFF), ARCH_X86_64, SECCOMP_EPERM),
        (NR_KEYCTL, (KEYCTL_READ, 123456789), ARCH_X86_64, SECCOMP_EPERM),
        (NR_ADD_KEY, (), ARCH_X86_64, SECCOMP_EPERM),
        (NR_REQUEST_KEY, (), ARCH_X86_64, SECCOMP_EPERM),
        (NR_WAITID, (), ARCH_X86_64, SECCOMP_ALLOW),
        (NR_IOPRIO_SET, (), ARCH_X86_64, SECCOMP_ALLOW),
        (X32_BIT | NR_KEYCTL, (), ARCH_X86_64, SECCOMP_EPERM),
    ],
    ids=[
        "read", "write", "execve", "exit_group", "setrlimit-core", "setrlimit-core-high-bits", "setrlimit-nofile",
        "prlimit-set-core", "prlimit-set-core-high-pointer", "prlimit-read-core", "prlimit-set-nofile",
        "prlimit-set-other-pid", "prlimit-read-other-pid", "socket-vsock", "socket-inet", "socket-unix",
        "syscall-424", "io_uring_setup", "io_uring_enter", "io_uring_register", "syscall-428", "clone3",
        "x32-setrlimit", "x32-read", "i386-setrlimit", "i386-read", "keyctl-user-keyring-id",
        "keyctl-read-user-session-keyring", "keyctl-read-by-serial", "add_key", "request_key", "waitid",
        "ioprio_set", "x32-keyctl",
    ],
)  # fmt: skip
def test_confine_filter_refuses_only_what_it_must(
    sandbox: ModuleType, nr: int, args: tuple[int, ...], arch: int, action: int
) -> None:
    assert run_filter(confine_filter(sandbox), nr, args, arch) == action


def test_confine_filter_names_what_every_row_does(sandbox: ModuleType) -> None:
    # Review finding 3: each row carries a comment with its index, so a reader can follow the jump offsets.
    program = sandbox.CONFINE_PROGRAM
    start = program.index("FILTER = [\n") + len("FILTER = [\n")
    rows = program[start : program.index("\n]\n", start)].split("\n")
    assert len(rows) == len(confine_filter(sandbox)), rows
    for index, row in enumerate(rows):
        assert re.fullmatch(rf"    \(.*\),  # {index}: \S.*", row), row


@pytest.mark.parametrize("name", ["READONLY_PROGRAM", "CONFINE_PROGRAM", "COPY_BACK_PROGRAM"])
def test_the_embedded_programs_are_typed_and_defer_their_annotations(sandbox: ModuleType, name: str) -> None:
    # Review finding 2: the Readability Standards' type hints hold for the embedded programs too; deferred
    # annotations keep them valid on any python3 the sandbox finds, since they are never evaluated.
    tree = ast.parse(getattr(sandbox, name))
    functions = [node for node in ast.walk(tree) if isinstance(node, FUNCTION_NODES)]
    assert [node.name for node in functions if not is_typed(node)] == [], name
    if functions:
        first = tree.body[0]
        assert isinstance(first, ast.ImportFrom) and first.module == "__future__", name
        assert [alias.name for alias in first.names] == ["annotations"], name


def test_confine_program_starts_a_session_joins_a_keyring_filters_then_prints_ready_and_execs(
    sandbox: ModuleType,
) -> None:
    program = sandbox.CONFINE_PROGRAM
    steps = [
        "os.setsid()",
        "libc.syscall(250, 1, None)",
        "libc.prctl(22, 2, program, 0, 0)",
        'os.write(2, b"lassi-sandbox-ready\\n")',
        "os.execvp(sys.argv[1], sys.argv[1:])",
    ]
    positions = [program.index(step) for step in steps]
    assert positions == sorted(positions) and all(program.count(step) == 1 for step in steps), positions
    # The ready line is the last thing before the exec: every step of the chain before it has succeeded.
    between = program[positions[-2] : positions[-1]].split("\n")
    assert len(between) == 2 and between[1].strip() == "", between
    # A failure exits with a message before argv runs; importing the program under another name runs nothing.
    assert 'if __name__ == "__main__":' in program and 'sys.exit("lassi-sandbox confine: %s" % exc)' in program
    assert "'" not in program
    tree = ast.parse(program)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imported == {"ctypes", "errno", "os", "struct", "sys"}, imported
    assert [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)] == ["__future__"]


# ---------------------------------------------------------------------------
# More setup-script properties and hostile inputs


def test_setup_script_is_the_same_for_every_spec(sandbox: ModuleType) -> None:
    first = sandbox.sandbox_command(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    other = sample_spec(
        sandbox, workdir=HOME / "w", hidden_roots=(SCRATCH,), harness=None, toolchains=None, tasks_max=8, disk_mb=4
    )
    second = sandbox.sandbox_command(other, ["other"], Limits(wall_s=99.0, memory_mb=4096, cpus=8))
    assert script_of(first) == script_of(second) == sandbox.SETUP_SCRIPT


def test_hostile_paths_and_arguments_appear_only_as_separate_argv_elements(
    sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in sandbox.PASSED_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    base = ROOT / "sbx base"
    workdir = base / "work `id` $(reboot)" / "build\nnext line"
    first_root = base / "ro 'one' $HOME;rm -rf x"
    second_root = base / 'ro "two"\n$(touch y)'
    harness = base / "harness; echo pwned > z & $IFS"
    toolchains = base / "tc | tee $(x),upperdir=/"
    program = ["./main", "QA with space", "'QB single'", '"QC double"', "$(QD)", "`QE`", "QF; rm -rf /"]
    program += ["QG\nQH", "$QI", "QJ&&|<>"]
    spec = sandbox.SandboxSpec(
        workdir=workdir, hidden_roots=(first_root, second_root), harness=harness, toolchains=toolchains
    )
    command = sandbox.sandbox_command(spec, program, ONE_SECOND)
    assert script_of(command) == sandbox.SETUP_SCRIPT
    start = command.index(sandbox.SETUP_SCRIPT)
    values = [str(workdir), str(harness), str(toolchains), str(first_root), str(second_root), *program[1:]]
    for value in values:
        holders = [i for i, part in enumerate(command) if value in part]
        assert len(holders) == 1 and command[holders[0]] == value, (value, holders)
        assert holders[0] > start + 1, value
    # ONE_SECOND has a 64 MiB memory limit, so the disk cap is half of it, 32 MiB.
    layout = ["sh", str(workdir), str(harness), str(toolchains), str(32 << 20), "2", str(first_root)]
    layout.append(str(second_root))
    assert command[start + 1 :] == [*layout, "1", "1", "2", *program]


# ---------------------------------------------------------------------------
# Validation


@pytest.mark.parametrize(
    "overrides",
    [
        {"workdir": Path("runs/build")},
        {"hidden_roots": (SCRATCH, Path("home/user"))},
        {"harness": Path("assets/harness")},
        {"toolchains": Path("toolchains")},
        {"workdir": SCRATCH},
        {"workdir": HOME},
        {"hidden_roots": ()},
        {"hidden_roots": str(SCRATCH)},
        {"harness": WORK},
        {"harness": WORK.parent},
        {"harness": SCRATCH / "runs"},
        {"toolchains": WORK},
        {"toolchains": WORK.parent},
        {"toolchains": SCRATCH},
        {"toolchains": HOME},
        {"toolchains": HOME.parent},
        {"harness": HOME},
        {"harness": HOME.parent},
        {"hidden_roots": (HOME, WORK / "inner")},
        {"harness": WORK / "harness"},
        {"toolchains": WORK / "sub" / "toolchains"},
        {"hidden_roots": (Path(ROOT),)},
        {"hidden_roots": (SCRATCH, Path(ROOT))},
    ],
    ids=[
        "relative-workdir",
        "relative-root",
        "relative-harness",
        "relative-toolchains",
        "workdir-is-first-root",
        "workdir-is-second-root",
        "no-root",
        "roots-as-one-string",
        "harness-is-workdir",
        "harness-is-workdir-parent",
        "harness-contains-workdir",
        "toolchains-is-workdir",
        "toolchains-is-workdir-parent",
        "toolchains-is-a-hidden-root",
        "toolchains-is-the-other-hidden-root",
        "toolchains-contains-a-hidden-root",
        "harness-is-a-hidden-root",
        "harness-contains-a-hidden-root",
        "workdir-contains-a-hidden-root",
        "harness-under-the-workdir",
        "toolchains-under-the-workdir",
        "hidden-root-is-a-filesystem-root",
        "second-hidden-root-is-a-filesystem-root",
    ],
)
def test_sandbox_spec_rejects_bad_paths(sandbox: ModuleType, overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        sample_spec(sandbox, **overrides)


@pytest.mark.parametrize("tasks_max", [0, -1, True, "8", 1.5, None])
def test_sandbox_spec_rejects_a_bad_tasks_max(sandbox: ModuleType, tasks_max: object) -> None:
    with pytest.raises(ValueError):
        sample_spec(sandbox, tasks_max=tasks_max)


@pytest.mark.parametrize("disk_mb", [0, -1, True, "8", 1.5, None])
def test_sandbox_spec_rejects_a_bad_disk_cap(sandbox: ModuleType, disk_mb: object) -> None:
    with pytest.raises(ValueError):
        sample_spec(sandbox, disk_mb=disk_mb)


@pytest.mark.parametrize(
    ("roots", "expected"),
    [
        ((SCRATCH, HOME), (SCRATCH, HOME)),
        ((HOME, SCRATCH, HOME), (HOME, SCRATCH)),
        ((SCRATCH, SCRATCH / "runs" / "other", HOME), (SCRATCH, HOME)),
        ((HOME / "sub", SCRATCH, HOME), (SCRATCH, HOME)),
        ([str(SCRATCH)], (SCRATCH,)),
    ],
    ids=["distinct", "duplicate", "nested-after", "nested-before", "list-of-strings"],
)
def test_sandbox_spec_keeps_only_the_outermost_hidden_roots_in_order(
    sandbox: ModuleType, roots: Sequence[object], expected: tuple[Path, ...]
) -> None:
    # A tmpfs on a root that an outer root's tmpfs already hid would fail under set -e (spike, design item 3).
    spec = sample_spec(sandbox, hidden_roots=roots)
    assert spec.hidden_roots == expected
    assert all(isinstance(root, Path) for root in spec.hidden_roots)


def test_sandbox_spec_stores_the_toolchains_root_as_a_path(sandbox: ModuleType) -> None:
    spec = sample_spec(sandbox, toolchains=str(TOOLCHAINS), disk_mb=16)
    assert spec.toolchains == TOOLCHAINS and isinstance(spec.toolchains, Path)
    assert spec.disk_mb == 16


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
        {"hidden_roots": (PurePosixPath("/scratch"), PurePosixPath("/run"))},
        {"harness": PurePosixPath("/dev/shm/harness")},
        {"toolchains": PurePosixPath("/tmp/toolchains")},
        {"toolchains": PurePosixPath("/run/toolchains")},
    ],
)
def test_sandbox_spec_rejects_paths_a_private_tmpfs_would_hide(
    sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]
) -> None:
    # POSIX paths on every platform: the rule is about the Linux mount layout the script builds.
    monkeypatch.setattr(sandbox, "Path", PurePosixPath)
    values: dict[str, object] = {"workdir": POSIX_WORK, "hidden_roots": (PurePosixPath("/scratch"),)}
    values.update(overrides)
    with pytest.raises(ValueError, match="private tmpfs"):
        sandbox.SandboxSpec(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"workdir": PurePosixPath("/var/lassi/runs/b")},
        {"workdir": PurePosixPath("/sys/fs/b")},
        {"harness": PurePosixPath("/var/lib/harness")},
        {"toolchains": PurePosixPath("/var/opt/toolchains")},
        {"toolchains": PurePosixPath("/sys/devices/system/x")},
    ],
)
def test_sandbox_spec_rejects_exposures_under_the_hidden_system_dirs(
    sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]
) -> None:
    monkeypatch.setattr(sandbox, "Path", PurePosixPath)
    values: dict[str, object] = {"workdir": POSIX_WORK, "hidden_roots": (PurePosixPath("/scratch"),)}
    values.update(overrides)
    with pytest.raises(ValueError, match="whose entries a tmpfs hides inside"):
        sandbox.SandboxSpec(**values)


def test_sandbox_spec_accepts_a_hidden_root_under_var_and_refuses_the_posix_root(
    sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A hidden root under /var is redundant but harmless; "/" would hide nothing, since a tmpfs on it is not
    # what later lookups of "/" see.
    monkeypatch.setattr(sandbox, "Path", PurePosixPath)
    spec = sandbox.SandboxSpec(workdir=POSIX_WORK, hidden_roots=(PurePosixPath("/var/lib/svc"),))
    assert spec.hidden_roots == (PurePosixPath("/var/lib/svc"),)
    with pytest.raises(ValueError, match="filesystem root"):
        sandbox.SandboxSpec(workdir=POSIX_WORK, hidden_roots=(PurePosixPath("/"),))


@pytest.mark.parametrize(
    "workdir",
    ["/tmpfoo/b", "/srv/var/tmp/b", "/scratch/tmp/b", "/runs/b", "/mnt/dev/shm/b", "/variable/b", "/system/b"],
)
def test_sandbox_spec_accepts_lookalikes_of_the_private_dirs(
    sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch, workdir: str
) -> None:
    monkeypatch.setattr(sandbox, "Path", PurePosixPath)
    spec = sandbox.SandboxSpec(workdir=PurePosixPath(workdir), hidden_roots=(PurePosixPath("/scratch/ro"),))
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
        # A crash (SIGSEGV 139, SIGABRT 134) is the program's own exit status, neither a hang nor a limit kill.
        (139, 0.5, 3.0, (False, False)),
        (134, 0.5, 3.0, (False, False)),
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
    # Round-3 finding RT1-2: the claim that a program cannot fake a hang is bounded by what the time measures.
    assert "program's own time" in doc and "10 ms" in doc, doc


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
    # A signal death of the whole command never carries the setup's done line: the setup died before it.
    finished = returncode >= 0
    runner = FakeRunner(
        returncode=returncode, stdout="o\n", stderr="e\n", elapsed_s=elapsed_s, clock=clock, done=finished
    )
    limits = Limits(wall_s=2.5, memory_mb=64, cpus=1)
    result = sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, limits)
    assert isinstance(result, sandbox.SandboxResult)
    assert (result.returncode, result.stdout, result.stderr) == (status, "o\n", "e\n")
    assert result.wall_s == pytest.approx(elapsed_s)
    assert (result.hang, result.killed) == (hang, killed)
    assert result.workdir_incomplete is not finished


@pytest.mark.parametrize(
    ("returncode", "stderr"),
    [
        (1, "Failed to connect to bus: No such file or directory\n"),
        (1, "unshare: unshare failed: Operation not permitted\n"),
        (32, "mount: /scratch: permission denied.\n"),
        (2, "sh: 1: cd: can't cd to /scratch/runs/trial/attempt00/build\n"),
        (1, "writable host mount: /var/tmp ext4 /dev/nvme23n1p1\n"),
        (1, "mount_setattr: errno 38\n"),
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
        ("lassi-sandbox-ready\n" + DONE_LINE, ""),
        ("lassi-sandbox-ready\nboom\n" + DONE_LINE, "boom\n"),
        (
            "Warning: something from setup\nlassi-sandbox-ready\nboom\n" + DONE_LINE,
            "Warning: something from setup\nboom\n",
        ),
        ("lassi-sandbox-ready\nlassi-sandbox-ready\n" + DONE_LINE, "lassi-sandbox-ready\n"),
    ],
)
def test_run_removes_only_the_first_ready_line_from_stderr(sandbox: ModuleType, raw: str, stderr: str) -> None:
    assert sandbox.READY_MARKER == "lassi-sandbox-ready"
    runner = FakeRunner(returncode=0, stderr=raw, ready=False)
    result = sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert result.stderr == stderr


@pytest.mark.parametrize("raw", ["xlassi-sandbox-ready\n", "lassi-sandbox-ready", "lassi-sandbox-ready now\n", ""])
def test_run_counts_the_ready_line_only_as_a_whole_line(sandbox: ModuleType, raw: str) -> None:
    runner = FakeRunner(returncode=0, stderr=raw + DONE_LINE, ready=False)
    with pytest.raises(sandbox.SandboxUnavailableError):
        sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)


@pytest.mark.parametrize(
    ("program_stderr", "stderr"),
    [
        ("", ""),
        ("a\nb\n", "a\nb\n"),
        # The program's last line may lack a newline; the done line follows it on the same line, and only the
        # done marker itself is removed.
        ("ERR", "ERR"),
        # The program printing the marker itself changes nothing: only the final one is the setup's.
        ("lassi-sandbox-done\nx\n", "lassi-sandbox-done\nx\n"),
        (DONE_LINE + "x\n", DONE_LINE + "x\n"),
        # A done line the program printed without its newline runs into the setup's; only the setup's is removed.
        ("lassi-sandbox-done 1000.00 1009.00", "lassi-sandbox-done 1000.00 1009.00"),
    ],
)
def test_run_removes_the_final_done_line_and_keeps_the_program_stderr_verbatim(
    sandbox: ModuleType, program_stderr: str, stderr: str
) -> None:
    runner = FakeRunner(returncode=0, stderr=program_stderr)
    result = sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert result.stderr == stderr


@pytest.mark.parametrize(
    ("returncode", "program_stderr"),
    [
        (0, ""),
        (1, "Traceback (most recent call last):\nOSError: [Errno 28] No space left on device\n"),
        (3, "umount: /scratch/runs/trial/attempt00/build: target is busy.\n"),
        (139, "lassi-sandbox-done\nmount: /tmp/s/rw: permission denied.\n"),
    ],
)
def test_run_raises_when_the_setup_ended_without_finishing_the_copy_back(
    sandbox: ModuleType, returncode: int, program_stderr: str
) -> None:
    # The program ran, but the setup exited normally without its done line: the status it returned is not the
    # program's, and the workdir did not come back, so the run is not reported as the program's result.
    runner = FakeRunner(returncode=returncode, stderr=program_stderr, done=False)
    with pytest.raises(sandbox.SandboxUnavailableError, match="copy-back") as caught:
        sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert f"exit status {returncode}" in str(caught.value)


@pytest.mark.parametrize(
    ("returncode", "elapsed_s", "status", "hang", "killed"),
    [
        # R7: the runner's own timeout killed the whole sandbox before copy-back; the run is a hang.
        (-1, 20.0, -1, True, False),
        (-1, 30.0, -1, True, False),
        # The scope backstop or a memory kill of the setup itself; the runner sees a signal death.
        (-9, 11.0, 137, True, False),
        (-9, 0.5, 137, False, True),
        (-15, 10.0, 143, True, False),
    ],
)
def test_run_returns_the_result_when_the_whole_sandbox_was_killed_before_copy_back(
    sandbox: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    elapsed_s: float,
    status: int,
    hang: bool,
    killed: bool,
) -> None:
    clock = install_clock(monkeypatch, sandbox)
    partial = "partial\n" + ("timed out after 20 s\n" if returncode == -1 else "")
    runner = FakeRunner(returncode=returncode, stdout="o", stderr=partial, elapsed_s=elapsed_s, clock=clock, done=False)
    limits = Limits(wall_s=2.5, memory_mb=64, cpus=1)
    result = sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, limits)
    assert (result.returncode, result.stdout, result.stderr) == (status, "o", partial)
    assert (result.hang, result.killed) == (hang, killed)
    # The copy-back never finished, so whatever the program wrote may be missing from the host workdir.
    assert result.workdir_incomplete is True


@pytest.mark.parametrize("line", [DONE_LINE, INCOMPLETE_LINE])
@pytest.mark.parametrize("returncode", [-9, -15, -1])
def test_run_counts_no_done_line_after_a_signal_death(sandbox: ModuleType, returncode: int, line: str) -> None:
    # The review's round-2 question: the setup exits normally after its done line, so a signal death with a done
    # line at the end means the program printed it and the setup died before its own; stderr stays verbatim.
    runner = FakeRunner(returncode=returncode, stderr="out\n" + line, done=False)
    result = sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert (result.stderr, result.workdir_incomplete) == ("out\n" + line, True), result


@pytest.mark.parametrize(
    ("returncode", "program_s", "hang", "killed"),
    [
        # Round-3 finding RT1-2: the program stops itself with a timeout-like status 0.1 s before the 3 s limit,
        # and setup and a long copy-back push the whole command to 9 s. Only the program's own time counts.
        (124, 2.9, False, False),
        (137, 2.9, False, True),
        (143, 2.9, False, False),
        # The innermost timeout fired: the program's own time reached the limit.
        (124, 3.0, True, False),
        (137, 5.0, True, False),
        (143, 3.2, True, False),
        # Each reading is cut to 10 ms steps, so readings up to 10 ms short of the limit may hide a timeout that
        # fired; 20 ms short cannot.
        (124, 2.99, True, False),
        (124, 2.98, False, False),
        (137, 2.98, False, True),
        (0, 9.0, False, False),
    ],
)
def test_run_classifies_by_the_programs_own_time_from_the_done_line(
    sandbox: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    program_s: float,
    hang: bool,
    killed: bool,
) -> None:
    clock = install_clock(monkeypatch, sandbox)
    runner = FakeRunner(returncode=returncode, stderr="e\n", elapsed_s=9.0, program_s=program_s, clock=clock)
    result = sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, Limits(3.0, 64, 1))
    assert (result.returncode, result.stderr, result.wall_s) == (returncode, "e\n", pytest.approx(9.0)), result
    assert result.program_s == pytest.approx(program_s), result
    assert (result.hang, result.killed) == (hang, killed), result


@pytest.mark.parametrize("returncode", [-1, -9, -15])
def test_run_has_no_program_time_when_the_whole_sandbox_was_killed(
    sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    # No done line counts after a signal death, so the whole command's time is all there is.
    clock = install_clock(monkeypatch, sandbox)
    runner = FakeRunner(returncode=returncode, stderr="e\n", elapsed_s=30.0, clock=clock, done=False)
    result = sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, Limits(3.0, 64, 1))
    assert (result.program_s, result.hang, result.wall_s) == (None, True, pytest.approx(30.0)), result


@pytest.mark.parametrize(
    "line",
    [
        "lassi-sandbox-done\n",
        "lassi-sandbox-done incomplete\n",
        "lassi-sandbox-done 1000.00\n",
        "lassi-sandbox-done 1000.0 1001.00\n",
        "lassi-sandbox-done 1000.00 1001.000\n",
        "lassi-sandbox-done 1000.00  1001.00\n",
        "lassi-sandbox-done 1000.00 1001.00 \n",
        "lassi-sandbox-done 1000.00 1001.00 partial\n",
        "lassi-sandbox-done 1000.00 1001.00",
        "lassi-sandbox-done 1000.00 999.99\n",
        "lassi-sandbox-done -1.00 1001.00\n",
        "lassi-sandbox-done 1000.00 1\u0661.00\n",
    ],
    ids=[
        "no-readings", "no-readings-incomplete", "one-reading", "one-decimal", "three-decimals", "two-spaces",
        "trailing-space", "other-mark", "no-newline", "clock-backwards", "negative", "non-ascii-digit",
    ],
)  # fmt: skip
def test_run_counts_only_a_well_formed_done_line(sandbox: ModuleType, line: str) -> None:
    # A final line that is not exactly the setup's (two readings of /proc/uptime, the second not before the
    # first, and the optional incomplete mark) is no done line, so a normal exit raises as an unfinished setup.
    runner = FakeRunner(returncode=0, stderr="out\n" + line, done=False)
    with pytest.raises(sandbox.SandboxUnavailableError, match="copy-back"):
        sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)


def test_run_turns_the_incomplete_done_line_into_the_workdir_flag(sandbox: ModuleType) -> None:
    notice = "lassi-sandbox copy-back: the new files total 9 bytes, over the 8-byte workdir cap; none was copied back\n"
    runner = FakeRunner(returncode=4, stderr="err\n" + notice, incomplete=True)
    result = sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert (result.returncode, result.workdir_incomplete) == (4, True)
    assert result.stderr == "err\n" + notice
    complete = sandbox.Sandbox(runner=FakeRunner(stderr="err\n")).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert (complete.workdir_incomplete, complete.stderr) == (False, "err\n")


def test_run_ignores_an_incomplete_mark_the_program_printed_itself(sandbox: ModuleType) -> None:
    # Only the setup's own final line counts; nothing the program prints can follow it.
    # The forged line is well formed, so only its position keeps it from counting.
    forged = f"{sandbox.DONE_MARKER} 1000.25 1001.00{sandbox.INCOMPLETE_SUFFIX}\n"
    runner = FakeRunner(stderr=forged)
    result = sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert (result.workdir_incomplete, result.stderr) == (False, forged)


@pytest.mark.parametrize(
    ("stdout_truncated", "stderr_truncated"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_r4_run_passes_the_runners_truncation_flags_through(
    sandbox: ModuleType, stdout_truncated: bool, stderr_truncated: bool
) -> None:
    runner = FakeRunner(
        returncode=0, stdout="o", stderr="e\n", stdout_truncated=stdout_truncated, stderr_truncated=stderr_truncated
    )
    result = sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert (result.stdout_truncated, result.stderr_truncated) == (stdout_truncated, stderr_truncated)
    assert (result.stdout, result.stderr) == ("o", "e\n")


def test_run_turns_a_missing_tool_into_sandbox_unavailable(sandbox: ModuleType) -> None:
    runner = FakeRunner(error=FileNotFoundError(2, "No such file or directory", "prlimit"))
    with pytest.raises(sandbox.SandboxUnavailableError, match="prlimit"):
        sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert len(runner.calls) == 1
    assert runner.calls[0].argv[0] == "prlimit"


def test_run_names_a_missing_workdir(sandbox: ModuleType) -> None:
    runner = FakeRunner(error=FileNotFoundError(2, "No such file or directory", str(WORK)))
    with pytest.raises(sandbox.SandboxUnavailableError, match="workdir") as caught:
        sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert str(WORK) in str(caught.value)


@pytest.mark.parametrize(
    "error",
    [
        PermissionError(13, "Permission denied", "prlimit"),
        PermissionError(13, "Permission denied", str(WORK)),
        NotADirectoryError(20, "Not a directory", str(WORK)),
        OSError(7, "Argument list too long"),
        OSError(12, "Cannot allocate memory"),
    ],
    ids=["tool-not-executable", "workdir-not-searchable", "workdir-is-a-file", "argv-too-long", "no-memory"],
)
def test_run_turns_any_os_error_starting_the_command_into_sandbox_unavailable(
    sandbox: ModuleType, error: OSError
) -> None:
    # Round-3 finding RT1-1: every OSError the runner raises, not only FileNotFoundError, means the sandbox did
    # not start; the caller (lassi.cli) handles SandboxUnavailableError, never a raw OSError, and nothing else
    # runs in its place.
    runner = FakeRunner(error=error)
    with pytest.raises(sandbox.SandboxUnavailableError, match="cannot start") as caught:
        sandbox.Sandbox(runner=runner).run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert caught.value.__cause__ is error
    assert error.strerror in str(caught.value) and "nothing ran" in str(caught.value)
    if error.filename:
        assert error.filename in str(caught.value)
    assert [call.argv[0] for call in runner.calls] == ["prlimit"]


def test_default_runner_is_the_toolchains_capped_runner(sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    import lassi.toolchains

    # R4: the sandbox's own runner caps stdout and stderr; the compilers' subprocess_runner keeps everything.
    assert sandbox.Sandbox().runner is lassi.toolchains.capped_runner
    seen = trap_popen(monkeypatch)
    spec = sample_spec(sandbox)
    with pytest.raises(sandbox.SandboxUnavailableError, match="prlimit"):
        sandbox.Sandbox().run(spec, PROGRAM, ONE_SECOND)
    assert seen == [sandbox.sandbox_command(spec, PROGRAM, ONE_SECOND)]


def test_default_runner_failing_to_start_on_any_os_error_raises_sandbox_unavailable(
    sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Round-3 finding RT1-1 through the real capped_runner: Popen refusing the workdir as cwd.
    seen = trap_popen(monkeypatch, NotADirectoryError(20, "Not a directory", str(WORK)))
    with pytest.raises(sandbox.SandboxUnavailableError, match="Not a directory"):
        sandbox.Sandbox().run(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    assert len(seen) == 1


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
            assert "capped_runner" in found
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
