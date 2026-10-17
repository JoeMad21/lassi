"""Tests for tools/local/longjob.py, the workstation's long-job runner (task P5.15), with fakes.

Bible: Host Facts (Workstation, Workstation scratch, Workstation approvals),
Agent Rules 1, 7, and 10; AGENTS.md Local Execution (decision 4 (b): one job
at a time, niced, inside a `systemd-run --user` scope with a memory cap) and
Run Reports; plans/p5-ir.md P5.15 ("Long-job runner"); plans/local-plan.md
section 3.3, item 5. It stands in for `rx job start --big`.

The interface these tests fix:

- CLI: `python3 tools/local/longjob.py --name <short name> -- <command...>`,
  and the inner mode `--inner <job dir> --inner-lock-fd <fd> -- <command...>`,
  which the outer process starts inside the scope. `main(argv, environ=None)
  -> int`, argv without the program name, environ defaulting to os.environ.
  A usage error (no --name or --inner, no command, a name that is not one
  safe path component, --inner without --inner-lock-fd or --inner-lock-fd
  without --inner) is exit status 2, possibly as argparse's SystemExit(2); a
  refusal returns 2 with a message on stderr. `parse_args(argv)` returns a
  namespace with `name`, `inner` (None in the outer mode), `inner_lock_fd`
  (an int, None in the outer mode), and `command` (the words after "--",
  options included, without the "--").
- The outer mode needs LASSI_HOST=workstation, LASSI_SCRATCH, LASSI_RUNS_ROOT,
  LASSI_JOB_MEMORY_MAX, LASSI_STOP_GIB, LASSI_HOST_DRIVE, and
  LASSI_DRIVE_FLOOR_GIB from the host profile, each with no default; it
  refuses, naming the variable, when one is missing or empty, and refuses a
  stop line or floor that is not an integer. It refuses while another job
  holds the lock, when the scratch root's size is at or past the stop line,
  and when the drive's free space is below the floor.
- main reaches the host only through these module-level functions, which the
  tests replace: `acquire_lock(path) -> handle | None` (None while another
  job holds it), `scratch_kib(root) -> int` (du -sk), `free_kib(path) -> int`,
  `git_state(cwd) -> (commit | None, dirty | None)`, `launch(argv,
  pass_fds=()) -> int` (runs systemd-run with those descriptors kept open
  and returns its status), and `now() -> datetime` (UTC).
- `acquire_lock(path, fcntl_module=None)` opens the file and takes an
  exclusive non-blocking flock (fcntl_module.LOCK_EX | LOCK_NB), importing
  fcntl only when none is given; a BlockingIOError from flock means busy.
- `scratch_kib(root, run=...)` runs `du -sk <root>` through `run(argv) -> stdout`.
- `wrap(command, which=shutil.which) -> list` puts the command under
  "ionice -c2 -n7 nice -n 10", ionice only when `which("ionice")` finds it.
- `inner_main(job_dir, command, *, lock_fd=None, proc_root="/proc",
  cgroup_root="/sys/fs/cgroup", resource_module=None) -> int` refuses a
  lock_fd that is not an open regular file, sets RLIMIT_CORE to (1, 1) and
  reads it back, reads the "0::" entry of <proc_root>/self/cgroup and that
  cgroup's memory.max, and refuses, with status 2 and without running the
  command, when any of these fails or memory.max is not a number of bytes
  (a scope with no cap). Otherwise it runs wrap(command) through
  `run_logged(command, log_path, pass_fds=...)` with stdout and stderr to
  <job dir>/log.txt, the lock descriptor kept open and handed to the command,
  and writes <job dir>/inner.json: lock_fd, rlimit_core [1, 1], cgroup,
  memory_max (stripped text), memory_peak (int from memory.peak, or null with
  a memory_peak_reason naming memory.peak). It returns the command's status.
  On a refusal inner.json holds `refused` (the reason), memory_peak null, and
  a memory_peak_reason saying the command did not run.
- The outer mode launches exactly: systemd-run --user --scope --quiet
  -p MemoryMax=<LASSI_JOB_MEMORY_MAX> -p MemorySwapMax=0 -- <sys.executable> <this file>
  --inner <job dir> --inner-lock-fd <the lock's fd> -- <command>, with
  pass_fds holding the lock's fd, so the scope holds the lock as long as the
  job runs, with the job dir LASSI_RUNS_ROOT/jobs/<id>
  (id from the UTC start time and the name), and writes <job dir>/job.json,
  plain ASCII, even when the command fails: id, name, command, cwd, commit,
  dirty, start and end (UTC ISO 8601), exit_status, scratch_kib_before and
  _after, drive_free_kib_before and _after, memory_max_requested,
  memory_max_read and memory_peak (from inner.json; null with a
  memory_peak_reason when it is missing), inner_refused (the inner mode's
  reason, when it refused), host (LASSI_HOST), and hostname. It returns the
  command's status.

Every host call is a fake here: no test runs systemd-run, prlimit, ionice,
nice, or a real flock on Windows, and the cgroup tree is a directory of
SYNTHETIC files. git runs for real only for git_state, in a throwaway
repository. No value in this module is a measurement.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import platform
import re
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
LONGJOB = REPO / "tools" / "local" / "longjob.py"
GIB_KIB = 1024 * 1024
REQUIRED = (
    "LASSI_HOST", "LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_JOB_MEMORY_MAX", "LASSI_STOP_GIB",
    "LASSI_HOST_DRIVE", "LASSI_DRIVE_FLOOR_GIB",
)
NAME = "smoke-job"
COMMAND = ["make", "-j", "8", "--name", "inner-option"]
# SYNTHETIC values: a start and an end, a commit, and what a fake scope reads back.
T0 = datetime(2026, 10, 17, 12, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 10, 17, 12, 30, 5, tzinfo=timezone.utc)
COMMIT = "0123456789abcdef0123456789abcdef01234567"
SCOPE = "/user.slice/user-1000.slice/user@1000.service/app.slice/run-u7.scope"
CAP_READ = "21474836480"
PEAK = 7340032
INNER = {"rlimit_core": [1, 1], "cgroup": SCOPE, "memory_max": CAP_READ, "memory_peak": PEAK}
# The descriptor number a fake lock reports (SYNTHETIC; no file is open under it).
FAKE_LOCK_FD = 7


def load_longjob() -> ModuleType:
    """Load tools/local/longjob.py as a module, failing with the task's name while it does not exist."""
    if not LONGJOB.is_file():
        pytest.fail("task P5.15 adds tools/local/longjob.py")
    spec = importlib.util.spec_from_file_location("lassi_longjob_under_test", LONGJOB)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def status_of(call: Callable[[], int]) -> int:
    """Return what `call` returns, or the code of the SystemExit it raises."""
    try:
        return call()
    except SystemExit as stop:
        return stop.code if isinstance(stop.code, int) else 2


def profile_env(tmp_path: Path) -> dict[str, str]:
    """Return the host profile's environment for a scratch root under `tmp_path` (SYNTHETIC values)."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    return {
        "LASSI_HOST": "workstation",
        "LASSI_SCRATCH": str(scratch),
        "LASSI_RUNS_ROOT": str(scratch / "lassi-runs"),
        "LASSI_JOB_MEMORY_MAX": "20G",
        "LASSI_STOP_GIB": "145",
        "LASSI_HOST_DRIVE": "/mnt/c",
        "LASSI_DRIVE_FLOOR_GIB": "100",
    }


def same_path(left: Any, right: Any) -> bool:
    """Return whether two paths name the same place."""
    return os.path.normcase(os.path.abspath(str(left))) == os.path.normcase(os.path.abspath(str(right)))


class FakeLock:
    """A lock handle that records its release."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def fileno(self) -> int:
        return FAKE_LOCK_FD

    def close(self) -> None:
        self.events.append("release")

    def __enter__(self) -> FakeLock:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class FakeHost:
    """Stand-ins for every host call main makes; values before the launch differ from those after it."""

    def __init__(
        self,
        module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        *,
        scratch: tuple[int, int] = (1000, 1500),
        free: tuple[int, int] = (500 * GIB_KIB, 499 * GIB_KIB),
        lock_free: bool = True,
        status: int = 0,
        inner: dict[str, Any] | None = None,
        git: tuple[str | None, bool | None] = (COMMIT, True),
    ) -> None:
        self.events: list[str] = []
        self.launched: list[list[str]] = []
        self.launch_fds: list[list[int]] = []
        self.lock_paths: list[str] = []
        self.scratch_roots: list[str] = []
        self.free_paths: list[str] = []
        self.git_dirs: list[str] = []
        self.after_launch = False
        self.scratch, self.free, self.lock_free = scratch, free, lock_free
        self.status, self.inner, self.git = status, inner, git
        for name in ("acquire_lock", "scratch_kib", "free_kib", "git_state", "launch", "now"):
            monkeypatch.setattr(module, name, getattr(self, name))

    def acquire_lock(self, path: Any, *args: Any, **kwargs: Any) -> FakeLock | None:
        self.events.append("lock")
        self.lock_paths.append(str(path))
        return FakeLock(self.events) if self.lock_free else None

    def scratch_kib(self, root: Any, *args: Any, **kwargs: Any) -> int:
        self.scratch_roots.append(str(root))
        return self.scratch[1] if self.after_launch else self.scratch[0]

    def free_kib(self, path: Any, *args: Any, **kwargs: Any) -> int:
        self.free_paths.append(str(path))
        return self.free[1] if self.after_launch else self.free[0]

    def git_state(self, cwd: Any, *args: Any, **kwargs: Any) -> tuple[str | None, bool | None]:
        self.git_dirs.append(str(cwd))
        return self.git

    def now(self) -> datetime:
        return T1 if self.after_launch else T0

    def launch(self, argv: Sequence[str], *args: Any, pass_fds: Sequence[int] = (), **kwargs: Any) -> int:
        self.events.append("launch")
        argv = [str(word) for word in argv]
        self.launched.append(argv)
        self.launch_fds.append([int(fd) for fd in pass_fds])
        if self.inner is not None and "--inner" in argv:
            job_dir = Path(argv[argv.index("--inner") + 1])
            (job_dir / "inner.json").write_text(json.dumps(self.inner), encoding="ascii")
        self.after_launch = True
        return self.status


def run_outer(module: ModuleType, env: dict[str, str], command: Sequence[str] = COMMAND) -> int:
    """Run the outer mode for NAME and `command` under `env`."""
    return status_of(lambda: module.main(["--name", NAME, "--", *command], environ=env))


def only_job_dir(env: dict[str, str]) -> Path:
    """Return the one job directory under LASSI_RUNS_ROOT/jobs."""
    jobs = Path(env["LASSI_RUNS_ROOT"]) / "jobs"
    found = sorted(path for path in jobs.iterdir() if path.is_dir()) if jobs.is_dir() else []
    assert len(found) == 1, f"expected one job directory under {jobs}, found {found}"
    return found[0]


def read_record(env: dict[str, str]) -> dict[str, Any]:
    """Return the job record the outer mode wrote, checking it is plain ASCII JSON."""
    data = (only_job_dir(env) / "job.json").read_bytes()
    return json.loads(data.decode("ascii"))


def utc(text: str) -> datetime:
    """Parse a UTC ISO 8601 date, with Z or +00:00."""
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    assert parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0, text
    return parsed


# The module itself.


def test_longjob_uses_only_the_standard_library_and_python_310() -> None:
    load_longjob()
    tree = ast.parse(LONGJOB.read_text(encoding="utf-8"), feature_version=(3, 10))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots.add(node.module.split(".")[0])
    outside = sorted(root for root in roots if root not in sys.stdlib_module_names and root != "__future__")
    assert not outside, f"longjob.py imports outside the standard library: {outside}"


def module_level_imports(tree: ast.Module) -> set[str]:
    """Return the modules a file imports outside any function."""
    found: set[str] = set()
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
        pending.extend(ast.iter_child_nodes(node))
    return found


def test_longjob_imports_linux_only_modules_inside_functions() -> None:
    load_longjob()
    eager = module_level_imports(ast.parse(LONGJOB.read_text(encoding="utf-8"))) & {"fcntl", "resource"}
    assert not eager, f"longjob.py imports {sorted(eager)} at module level; the tests run on Windows"


def test_longjob_is_plain_ascii_with_lf_line_endings() -> None:
    load_longjob()
    data = LONGJOB.read_bytes()
    data.decode("ascii")
    assert b"\r" not in data


# Arguments.


def test_parse_args_reads_the_outer_mode() -> None:
    longjob = load_longjob()
    args = longjob.parse_args(["--name", NAME, "--", *COMMAND])
    assert args.name == NAME
    assert args.inner is None
    assert args.inner_lock_fd is None
    assert list(args.command) == COMMAND


def test_parse_args_reads_the_inner_mode(tmp_path: Path) -> None:
    longjob = load_longjob()
    job_dir = str(tmp_path / "job")
    args = longjob.parse_args(["--inner", job_dir, "--inner-lock-fd", "5", "--", *COMMAND])
    assert args.inner == job_dir
    assert args.inner_lock_fd == 5
    assert list(args.command) == COMMAND


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--", "make"],
        ["--name", NAME],
        ["--name", NAME, "--"],
        ["--inner"],
        ["--inner", "job", "--", "make"],
        ["--inner", "job", "--inner-lock-fd", "lock", "--", "make"],
        ["--inner", "job", "--inner-lock-fd", "-1", "--", "make"],
        ["--name", NAME, "--inner-lock-fd", "5", "--", "make"],
    ],
    ids=["nothing", "no-name", "no-command", "empty-command", "inner-without-dir", "inner-without-lock-fd",
         "lock-fd-not-a-number", "lock-fd-negative", "lock-fd-in-the-outer-mode"],
)
def test_usage_errors_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    longjob = load_longjob()
    host = FakeHost(longjob, monkeypatch, inner=INNER)
    env = profile_env(tmp_path)
    assert status_of(lambda: longjob.main(argv, environ=env)) == 2
    assert not host.launched


@pytest.mark.parametrize("name", ["", "a/b", "../up", "..", "with space", "back\\slash"])
def test_a_name_that_is_not_one_safe_path_component_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    longjob = load_longjob()
    host = FakeHost(longjob, monkeypatch, inner=INNER)
    env = profile_env(tmp_path)
    assert status_of(lambda: longjob.main(["--name", name, "--", "true"], environ=env)) == 2
    assert not host.launched


def test_main_dispatches_the_inner_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    longjob = load_longjob()
    calls: list[tuple[str, list[str], Any]] = []

    def fake_inner(job_dir: Any, command: Sequence[str], **kwargs: Any) -> int:
        calls.append((str(job_dir), list(command), kwargs.get("lock_fd")))
        return 4

    monkeypatch.setattr(longjob, "inner_main", fake_inner)
    job_dir = str(tmp_path / "job")
    argv = ["--inner", job_dir, "--inner-lock-fd", "5", "--", *COMMAND]
    status = status_of(lambda: longjob.main(argv, environ=profile_env(tmp_path)))
    assert status == 4
    assert calls == [(job_dir, COMMAND, 5)]


# Refusals before the start.


@pytest.mark.parametrize("missing", REQUIRED)
def test_outer_mode_refuses_each_missing_profile_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    longjob = load_longjob()
    host = FakeHost(longjob, monkeypatch, inner=INNER)
    env = profile_env(tmp_path)
    del env[missing]
    assert run_outer(longjob, env) == 2
    assert missing in capsys.readouterr().err
    assert not host.launched


@pytest.mark.parametrize("empty", REQUIRED)
def test_outer_mode_refuses_each_empty_profile_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    empty: str,
) -> None:
    longjob = load_longjob()
    host = FakeHost(longjob, monkeypatch, inner=INNER)
    env = profile_env(tmp_path)
    env[empty] = ""
    assert run_outer(longjob, env) == 2
    assert empty in capsys.readouterr().err
    assert not host.launched


@pytest.mark.parametrize("host_name", ["alpha01", "Workstation", "local"])
def test_outer_mode_refuses_a_host_other_than_the_workstation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    host_name: str,
) -> None:
    longjob = load_longjob()
    host = FakeHost(longjob, monkeypatch, inner=INNER)
    env = profile_env(tmp_path)
    env["LASSI_HOST"] = host_name
    assert run_outer(longjob, env) == 2
    assert "LASSI_HOST" in capsys.readouterr().err
    assert not host.launched


@pytest.mark.parametrize("variable", ["LASSI_STOP_GIB", "LASSI_DRIVE_FLOOR_GIB"])
@pytest.mark.parametrize("value", ["abc", "1.5", "-1"])
def test_outer_mode_refuses_a_line_that_is_not_a_whole_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    variable: str, value: str,
) -> None:
    longjob = load_longjob()
    host = FakeHost(longjob, monkeypatch, inner=INNER)
    env = profile_env(tmp_path)
    env[variable] = value
    assert run_outer(longjob, env) == 2
    assert variable in capsys.readouterr().err
    assert not host.launched


def test_outer_mode_refuses_while_another_job_holds_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    longjob = load_longjob()
    host = FakeHost(longjob, monkeypatch, lock_free=False, inner=INNER)
    env = profile_env(tmp_path)
    assert run_outer(longjob, env) == 2
    assert capsys.readouterr().err.strip(), "the lock refusal printed no message"
    assert host.lock_paths, "main never asked for the lock"
    assert not host.launched


@pytest.mark.parametrize("used_kib", [145 * GIB_KIB, 145 * GIB_KIB + 1, 160 * GIB_KIB], ids=["at", "past", "far"])
def test_outer_mode_refuses_at_or_past_the_stop_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    used_kib: int,
) -> None:
    longjob = load_longjob()
    host = FakeHost(longjob, monkeypatch, scratch=(used_kib, used_kib), inner=INNER)
    env = profile_env(tmp_path)
    assert run_outer(longjob, env) == 2
    assert capsys.readouterr().err.strip(), "the stop-line refusal printed no message"
    assert host.scratch_roots and all(same_path(root, env["LASSI_SCRATCH"]) for root in host.scratch_roots)
    assert not host.launched


def test_outer_mode_starts_just_below_the_stop_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    host = FakeHost(longjob, monkeypatch, scratch=(145 * GIB_KIB - 1, 145 * GIB_KIB - 1), inner=INNER)
    assert run_outer(longjob, profile_env(tmp_path)) == 0
    assert len(host.launched) == 1


@pytest.mark.parametrize("free_kib", [100 * GIB_KIB - 1, 10 * GIB_KIB, 0], ids=["just-below", "far-below", "full"])
def test_outer_mode_refuses_when_the_drive_is_below_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    free_kib: int,
) -> None:
    longjob = load_longjob()
    host = FakeHost(longjob, monkeypatch, free=(free_kib, free_kib), inner=INNER)
    env = profile_env(tmp_path)
    assert run_outer(longjob, env) == 2
    assert capsys.readouterr().err.strip(), "the drive-floor refusal printed no message"
    assert host.free_paths and all(Path(path) == Path("/mnt/c") for path in host.free_paths)
    assert not host.launched


def test_outer_mode_starts_at_the_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    longjob = load_longjob()
    host = FakeHost(longjob, monkeypatch, free=(100 * GIB_KIB, 100 * GIB_KIB), inner=INNER)
    assert run_outer(longjob, profile_env(tmp_path)) == 0
    assert len(host.launched) == 1


def test_the_entry_point_refuses_without_the_host_profile(tmp_path: Path) -> None:
    if not LONGJOB.is_file():
        pytest.fail("task P5.15 adds tools/local/longjob.py")
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("LASSI_")}
    proc = subprocess.run(
        [sys.executable, str(LONGJOB), "--name", NAME, "--", "true"],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    assert any(name in proc.stderr for name in REQUIRED), proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr


# The launch and the record.


def test_outer_mode_launches_the_inner_mode_in_a_capped_user_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    host = FakeHost(longjob, monkeypatch, inner=INNER)
    env = profile_env(tmp_path)
    assert run_outer(longjob, env) == 0
    assert len(host.launched) == 1
    argv = host.launched[0]
    job_dir = only_job_dir(env)
    head = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=20G", "-p", "MemorySwapMax=0", "--",
            sys.executable]
    assert argv[: len(head)] == head
    assert Path(argv[len(head)]).resolve() == LONGJOB.resolve()
    assert argv[len(head) + 1] == "--inner"
    assert same_path(argv[len(head) + 2], job_dir)
    assert argv[len(head) + 3 :] == ["--inner-lock-fd", str(FAKE_LOCK_FD), "--", *COMMAND]


def test_the_launch_hands_the_lock_descriptor_to_the_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    host = FakeHost(longjob, monkeypatch, inner=INNER)
    assert run_outer(longjob, profile_env(tmp_path)) == 0
    assert host.launch_fds == [[FAKE_LOCK_FD]], "the lock's descriptor did not reach the launched scope"


def test_launch_keeps_the_given_descriptors_open_in_systemd_run(monkeypatch: pytest.MonkeyPatch) -> None:
    longjob = load_longjob()
    seen: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(argv: Sequence[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(([str(word) for word in argv], kwargs))
        return subprocess.CompletedProcess(list(argv), 5)

    monkeypatch.setattr(longjob.subprocess, "run", fake_run)
    assert longjob.launch(["systemd-run", "--user", "--scope"], pass_fds=(FAKE_LOCK_FD,)) == 5
    assert len(seen) == 1
    assert seen[0][0] == ["systemd-run", "--user", "--scope"]
    assert tuple(seen[0][1].get("pass_fds", ())) == (FAKE_LOCK_FD,)


def test_the_job_directory_is_named_by_the_utc_start_and_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    FakeHost(longjob, monkeypatch, inner=INNER)
    env = profile_env(tmp_path)
    assert run_outer(longjob, env) == 0
    job_dir = only_job_dir(env)
    assert same_path(job_dir.parent, Path(env["LASSI_RUNS_ROOT"]) / "jobs")
    assert NAME in job_dir.name
    assert re.sub(r"[^0-9]", "", job_dir.name).startswith("20261017120000"), job_dir.name


def test_the_lock_lies_under_the_scratch_root_and_is_held_through_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    host = FakeHost(longjob, monkeypatch, inner=INNER)
    env = profile_env(tmp_path)
    assert run_outer(longjob, env) == 0
    assert len(host.lock_paths) == 1
    lock = os.path.normcase(os.path.abspath(host.lock_paths[0]))
    scratch = os.path.normcase(os.path.abspath(env["LASSI_SCRATCH"]))
    assert lock.startswith(scratch + os.sep), f"the lock {lock} is not under the scratch root"
    launched_at = host.events.index("launch")
    assert host.events.index("lock") < launched_at
    assert "release" not in host.events[:launched_at], "the lock was released before the job ran"


def test_the_job_record_holds_every_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    host = FakeHost(longjob, monkeypatch, inner=INNER)
    env = profile_env(tmp_path)
    assert run_outer(longjob, env) == 0
    record = read_record(env)
    assert record["id"] == only_job_dir(env).name
    assert record["name"] == NAME
    assert record["command"] == COMMAND
    assert same_path(record["cwd"], work)
    assert host.git_dirs and same_path(host.git_dirs[0], work)
    assert (record["commit"], record["dirty"]) == (COMMIT, True)
    assert utc(record["start"]) == T0
    assert utc(record["end"]) == T1
    assert record["exit_status"] == 0
    assert (record["scratch_kib_before"], record["scratch_kib_after"]) == (1000, 1500)
    assert (record["drive_free_kib_before"], record["drive_free_kib_after"]) == (500 * GIB_KIB, 499 * GIB_KIB)
    assert record["memory_max_requested"] == "20G"
    assert record["memory_max_read"] == CAP_READ
    assert record["memory_peak"] == PEAK
    assert record["host"] == "workstation"
    assert record["hostname"] in {platform.node(), socket.gethostname()}


def test_the_record_names_no_commit_outside_a_git_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    FakeHost(longjob, monkeypatch, inner=INNER, git=(None, None))
    env = profile_env(tmp_path)
    assert run_outer(longjob, env) == 0
    record = read_record(env)
    assert record["commit"] is None
    assert record["dirty"] is None


@pytest.mark.parametrize("status", [1, 3, 137])
def test_the_record_is_written_and_the_status_returned_when_the_command_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    longjob = load_longjob()
    FakeHost(longjob, monkeypatch, inner=INNER, status=status)
    env = profile_env(tmp_path)
    assert run_outer(longjob, env) == status
    record = read_record(env)
    assert record["exit_status"] == status
    assert utc(record["end"]) == T1
    assert record["scratch_kib_after"] == 1500


def test_the_record_marks_the_scope_values_missing_without_an_inner_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    FakeHost(longjob, monkeypatch, inner=None, status=1)
    env = profile_env(tmp_path)
    assert run_outer(longjob, env) == 1
    record = read_record(env)
    assert record["exit_status"] == 1
    assert record["memory_max_requested"] == "20G"
    assert record["memory_max_read"] is None
    assert record["memory_peak"] is None
    assert isinstance(record.get("memory_peak_reason"), str) and record["memory_peak_reason"].strip()


# The lock.


def fake_fcntl(raise_busy: bool = False) -> tuple[SimpleNamespace, list[tuple[Any, int]]]:
    """Return a stand-in fcntl module and the list of its flock calls."""
    calls: list[tuple[Any, int]] = []

    def flock(fd: Any, operation: int) -> None:
        calls.append((fd, operation))
        if raise_busy:
            raise BlockingIOError(11, "Resource temporarily unavailable")

    return SimpleNamespace(LOCK_EX=2, LOCK_NB=4, LOCK_UN=8, LOCK_SH=1, flock=flock), calls


def test_acquire_lock_takes_an_exclusive_nonblocking_flock(tmp_path: Path) -> None:
    longjob = load_longjob()
    module, calls = fake_fcntl()
    handle = longjob.acquire_lock(str(tmp_path / "longjob.lock"), fcntl_module=module)
    try:
        assert handle is not None
        assert [operation for _, operation in calls] == [module.LOCK_EX | module.LOCK_NB]
        assert (tmp_path / "longjob.lock").exists()
    finally:
        if handle is not None:
            handle.close()


def test_acquire_lock_returns_none_while_another_job_holds_it(tmp_path: Path) -> None:
    longjob = load_longjob()
    module, calls = fake_fcntl(raise_busy=True)
    assert longjob.acquire_lock(str(tmp_path / "longjob.lock"), fcntl_module=module) is None
    assert calls, "acquire_lock never called flock"


@pytest.mark.skipif(os.name == "nt", reason="flock exists only on POSIX hosts")
def test_a_real_flock_refuses_a_second_holder_and_frees_on_close(tmp_path: Path) -> None:
    longjob = load_longjob()
    path = str(tmp_path / "longjob.lock")
    first = longjob.acquire_lock(path)
    assert first is not None
    assert longjob.acquire_lock(path) is None
    first.close()
    third = longjob.acquire_lock(path)
    assert third is not None
    third.close()


# du and git.


def test_scratch_kib_reads_du_sk_of_the_root(tmp_path: Path) -> None:
    longjob = load_longjob()
    seen: list[list[str]] = []

    def run(argv: Sequence[str], *args: Any, **kwargs: Any) -> str:
        seen.append([str(word) for word in argv])
        return f"4096\t{tmp_path}\n"

    assert longjob.scratch_kib(str(tmp_path), run=run) == 4096
    assert len(seen) == 1
    assert seen[0][0] == "du" and "-sk" in seen[0] and same_path(seen[0][-1], tmp_path)


def git(repo: Path, *args: str) -> str:
    """Run git in `repo` with a throwaway identity and return its stdout."""
    identity = {
        "GIT_AUTHOR_NAME": "Job Test", "GIT_AUTHOR_EMAIL": "job-test@example.invalid",
        "GIT_COMMITTER_NAME": "Job Test", "GIT_COMMITTER_EMAIL": "job-test@example.invalid",
    }
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], env={**os.environ, **identity},
        capture_output=True, text=True, check=True, timeout=60,
    )
    return proc.stdout.strip()


def test_git_state_reads_the_commit_and_dirty_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "core.hooksPath", os.devnull)
    (repo / "a.txt").write_bytes(b"one\n")
    git(repo, "add", "a.txt")
    git(repo, "commit", "-q", "-m", "one")
    head = git(repo, "rev-parse", "HEAD")
    assert tuple(longjob.git_state(str(repo))) == (head, False)
    (repo / "a.txt").write_bytes(b"two\n")
    assert tuple(longjob.git_state(str(repo))) == (head, True)


def test_git_state_is_none_outside_a_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain = tmp_path / "plain"
    plain.mkdir()
    assert tuple(longjob.git_state(str(plain))) == (None, None)


# The nice and ionice wrapper.


def which_of(*present: str) -> Callable[[str], str | None]:
    """Return a which() that finds only the named programs."""
    return lambda name: f"/usr/bin/{name}" if name in present else None


def test_wrap_puts_the_command_under_ionice_and_nice() -> None:
    longjob = load_longjob()
    wrapped = longjob.wrap(COMMAND, which=which_of("ionice", "nice"))
    assert list(wrapped) == ["ionice", "-c2", "-n7", "nice", "-n", "10", *COMMAND]


def test_wrap_leaves_out_ionice_when_it_is_absent() -> None:
    longjob = load_longjob()
    wrapped = longjob.wrap(COMMAND, which=which_of("nice"))
    assert list(wrapped) == ["nice", "-n", "10", *COMMAND]


# The inner mode.


class FakeResource:
    """A stand-in resource module: setrlimit records, getrlimit reads back what was set or a forced value."""

    RLIMIT_CORE = 4

    def __init__(self, readback: tuple[int, int] | None = None, fail: bool = False) -> None:
        self.calls: list[tuple[int, tuple[int, int]]] = []
        self.limit = (-1, -1)
        self.readback, self.fail = readback, fail

    def setrlimit(self, which: int, limits: Sequence[int]) -> None:
        if self.fail:
            raise ValueError("not allowed to set the limit")
        self.calls.append((which, (int(limits[0]), int(limits[1]))))
        self.limit = (int(limits[0]), int(limits[1]))

    def getrlimit(self, which: int) -> tuple[int, int]:
        return self.readback if self.readback is not None else self.limit


CGROUP_FILE = b"1:name=systemd:/legacy-tree\n0::" + SCOPE.encode("ascii") + b"\n"
CAP_FILE = CAP_READ.encode("ascii") + b"\n"


def cgroup_tree(
    tmp_path: Path, peak: bytes | None = b"7340032\n", *, memory_max: bytes | None = CAP_FILE,
    cgroup_file: bytes = CGROUP_FILE,
) -> tuple[Path, Path]:
    """Write a SYNTHETIC /proc/self/cgroup and cgroup v2 tree; return (proc root, cgroup root).

    memory_max None leaves memory.max out, as in a cgroup whose memory controller is not enabled.
    """
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_bytes(cgroup_file)
    cgroup = tmp_path / "cgroup"
    scope_dir = cgroup.joinpath(*SCOPE.strip("/").split("/"))
    scope_dir.mkdir(parents=True)
    if memory_max is not None:
        (scope_dir / "memory.max").write_bytes(memory_max)
    if peak is not None:
        (scope_dir / "memory.peak").write_bytes(peak)
    return proc, cgroup


def plain_wrap(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Replace wrap with one that records the command and adds nothing; return its record."""
    wrapped: list[list[str]] = []

    def fake_wrap(command: Sequence[str], *args: Any, **kwargs: Any) -> list[str]:
        wrapped.append([str(word) for word in command])
        return [str(word) for word in command]

    monkeypatch.setattr(module, "wrap", fake_wrap)
    return wrapped


def run_inner(
    module: ModuleType, tmp_path: Path, command: Sequence[str], resource: FakeResource,
    peak: bytes | None = b"7340032\n", *, lock_fd: int | None = None, **tree: Any,
) -> tuple[int, Path]:
    """Run inner_main on a SYNTHETIC cgroup tree (cgroup_tree's keywords in `tree`); return its status and job dir."""
    proc, cgroup = cgroup_tree(tmp_path, peak, **tree)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    status = module.inner_main(
        str(job_dir), list(command), lock_fd=lock_fd, proc_root=str(proc), cgroup_root=str(cgroup),
        resource_module=resource,
    )
    return status, job_dir


def marker_command(marker: Path) -> list[str]:
    """Return a command that creates `marker`, so a test can tell whether it ran."""
    return [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]


def read_inner(job_dir: Path) -> dict[str, Any]:
    """Return the inner record, checking it is plain ASCII JSON."""
    return json.loads((job_dir / "inner.json").read_bytes().decode("ascii"))


HARMLESS = [sys.executable, "-c", "import sys; print(1); print('to-stderr', file=sys.stderr)"]


def test_inner_mode_holds_the_core_limit_and_records_the_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    wrapped = plain_wrap(longjob, monkeypatch)
    resource = FakeResource()
    status, job_dir = run_inner(longjob, tmp_path, HARMLESS, resource)
    assert status == 0
    assert resource.calls == [(FakeResource.RLIMIT_CORE, (1, 1))]
    assert wrapped == [HARMLESS], "the inner mode did not run the command through wrap"
    log = (job_dir / "log.txt").read_text(encoding="ascii").splitlines()
    assert "1" in log and "to-stderr" in log, log
    inner = json.loads((job_dir / "inner.json").read_bytes().decode("ascii"))
    assert inner["rlimit_core"] == [1, 1]
    assert inner["cgroup"] == SCOPE
    assert inner["memory_max"] == CAP_READ
    assert inner["memory_peak"] == PEAK


def test_inner_mode_marks_the_peak_missing_without_memory_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    plain_wrap(longjob, monkeypatch)
    status, job_dir = run_inner(longjob, tmp_path, [sys.executable, "-c", "print(1)"], FakeResource(), peak=None)
    assert status == 0
    inner = json.loads((job_dir / "inner.json").read_bytes().decode("ascii"))
    assert inner["memory_peak"] is None
    assert "memory.peak" in inner["memory_peak_reason"]
    assert inner["memory_max"] == CAP_READ


def test_inner_mode_returns_the_command_status_and_still_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    plain_wrap(longjob, monkeypatch)
    status, job_dir = run_inner(longjob, tmp_path, [sys.executable, "-c", "import sys; sys.exit(3)"], FakeResource())
    assert status == 3
    inner = json.loads((job_dir / "inner.json").read_bytes().decode("ascii"))
    assert inner["memory_peak"] == PEAK


@pytest.mark.parametrize(
    "resource",
    [FakeResource(readback=(0, 0)), FakeResource(readback=(1, -1)), FakeResource(fail=True)],
    ids=["reads-zero", "hard-unlimited", "set-fails"],
)
def test_inner_mode_refuses_to_run_without_the_one_byte_core_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    resource: FakeResource,
) -> None:
    longjob = load_longjob()
    wrapped = plain_wrap(longjob, monkeypatch)
    marker = tmp_path / "ran.txt"
    status, _ = run_inner(longjob, tmp_path, marker_command(marker), resource)
    assert status != 0
    assert not marker.exists(), "the command ran without the 1-byte core limit"
    assert not wrapped
    assert "core" in capsys.readouterr().err.lower()


# A scope with no memory cap: (cgroup_tree keywords, a word the refusal names, the cgroup the record holds).
UNCAPPED = {
    "memory-max-reads-max": ({"memory_max": b"max\n"}, "memory.max", SCOPE),
    "memory-max-missing": ({"memory_max": None}, "memory.max", SCOPE),
    "no-cgroup-v2-entry": ({"cgroup_file": b"1:name=systemd:/legacy-tree\n12:memory:/user.slice\n"}, "0::", None),
}


@pytest.mark.parametrize("case", list(UNCAPPED))
def test_inner_mode_refuses_to_run_in_a_scope_with_no_memory_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str,
) -> None:
    tree, named, cgroup = UNCAPPED[case]
    longjob = load_longjob()
    wrapped = plain_wrap(longjob, monkeypatch)
    marker = tmp_path / "ran.txt"
    status, job_dir = run_inner(longjob, tmp_path, marker_command(marker), FakeResource(), **tree)
    assert status == 2
    assert not marker.exists(), "the command ran in a scope with no memory cap"
    assert not wrapped, "the inner mode wrapped the command it refused"
    assert not (job_dir / "log.txt").exists()
    assert named in capsys.readouterr().err
    inner = read_inner(job_dir)
    assert isinstance(inner.get("refused"), str) and named in inner["refused"]
    assert inner["memory_max"] is None
    assert inner["cgroup"] == cgroup
    assert inner["rlimit_core"] == [1, 1]
    assert inner["memory_peak"] is None
    assert "did not run" in inner["memory_peak_reason"]


def test_the_job_record_shows_why_the_inner_mode_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    FakeHost(longjob, monkeypatch)
    wrapped = plain_wrap(longjob, monkeypatch)
    proc, cgroup = cgroup_tree(tmp_path, memory_max=b"max\n")

    def launch_inner(argv: Sequence[str], *args: Any, **kwargs: Any) -> int:
        words = [str(word) for word in argv]
        start = words.index("--inner")
        command = words[words.index("--", start) + 1:]
        return longjob.inner_main(words[start + 1], command, proc_root=str(proc), cgroup_root=str(cgroup),
                                  resource_module=FakeResource())

    monkeypatch.setattr(longjob, "launch", launch_inner)
    env = profile_env(tmp_path)
    marker = tmp_path / "ran.txt"
    assert run_outer(longjob, env, marker_command(marker)) == 2
    assert not marker.exists() and not wrapped
    record = read_record(env)
    assert record["exit_status"] == 2
    assert isinstance(record.get("inner_refused"), str) and "memory.max" in record["inner_refused"]
    assert record["memory_max_read"] is None
    assert record["memory_peak"] is None
    assert "did not run" in record["memory_peak_reason"]


# The lock descriptor inside the scope.


def test_inner_mode_holds_the_lock_descriptor_until_the_command_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    longjob = load_longjob()
    plain_wrap(longjob, monkeypatch)
    lock_fd = os.open(str(tmp_path / "longjob.lock"), os.O_RDWR | os.O_CREAT)
    open_while_running: list[bool] = []
    handed: list[list[int]] = []

    def fake_run_logged(command: Sequence[str], log_path: str, *args: Any, pass_fds: Sequence[int] = (),
                        **kwargs: Any) -> int:
        try:
            os.fstat(lock_fd)
            open_while_running.append(True)
        except OSError:
            open_while_running.append(False)
        handed.append([int(fd) for fd in pass_fds])
        return 0

    monkeypatch.setattr(longjob, "run_logged", fake_run_logged)
    try:
        status, job_dir = run_inner(longjob, tmp_path, ["true"], FakeResource(), lock_fd=lock_fd)
        assert status == 0
        assert open_while_running == [True], "the inner mode closed the lock descriptor before the command ran"
        assert handed == [[lock_fd]], "the command did not inherit the lock descriptor"
        os.fstat(lock_fd)
        assert read_inner(job_dir)["lock_fd"] == lock_fd
    finally:
        try:
            os.close(lock_fd)
        except OSError:
            pass


@pytest.mark.parametrize("kind", ["closed", "pipe"])
def test_inner_mode_refuses_without_an_open_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], kind: str,
) -> None:
    longjob = load_longjob()
    wrapped = plain_wrap(longjob, monkeypatch)
    opened: list[int] = []
    if kind == "closed":
        lock_fd = os.open(str(tmp_path / "longjob.lock"), os.O_RDWR | os.O_CREAT)
        os.close(lock_fd)
    else:
        opened = list(os.pipe())
        lock_fd = opened[0]
    marker = tmp_path / "ran.txt"
    try:
        status, job_dir = run_inner(longjob, tmp_path, marker_command(marker), FakeResource(), lock_fd=lock_fd)
    finally:
        for fd in opened:
            os.close(fd)
    assert status == 2
    assert not marker.exists(), "the command ran without the job lock"
    assert not wrapped
    assert "lock" in capsys.readouterr().err.lower()
    inner = read_inner(job_dir)
    assert "lock" in inner["refused"].lower()
    assert inner["lock_fd"] is None
    assert inner["memory_peak"] is None


@pytest.mark.skipif(os.name == "nt", reason="pass_fds exists only on POSIX hosts")
def test_run_logged_hands_the_lock_descriptor_to_the_command(tmp_path: Path) -> None:
    longjob = load_longjob()
    lock_fd = os.open(str(tmp_path / "longjob.lock"), os.O_RDWR | os.O_CREAT)
    log = tmp_path / "log.txt"
    probe = [sys.executable, "-c", f"import os; os.fstat({lock_fd}); print('held')"]
    try:
        assert longjob.run_logged(probe, str(log), pass_fds=(lock_fd,)) == 0
        assert "held" in log.read_text(encoding="ascii")
        assert longjob.run_logged(probe, str(log)) != 0, "the command saw the descriptor without pass_fds"
    finally:
        os.close(lock_fd)
