#!/usr/bin/env python3
"""Run one long job on the workstation inside a memory-capped systemd user scope (task P5.15).

    python3 tools/local/longjob.py --name <short name> -- <command> [args...]

It stands in for `rx job start --big` in the LASSI distro (AGENTS.md, Local Execution, decision 4 (b); Host Facts,
Workstation approvals): one job at a time, niced, inside a `systemd-run --user` scope with a memory cap, under the
1-byte core limit, each one recorded for the run report.

It needs the host profile's environment (source tools/local/host-profile.sh): LASSI_HOST=workstation,
LASSI_SCRATCH, LASSI_RUNS_ROOT, LASSI_JOB_MEMORY_MAX, LASSI_STOP_GIB, LASSI_HOST_DRIVE, and LASSI_DRIVE_FLOOR_GIB,
each with no default. Before the start it refuses, with status 2: while another job holds the lock (an exclusive
flock on LASSI_SCRATCH/longjob.lock, held for the whole job, so a crashed runner leaves no stale lock); when du -sk
of the scratch root is at or past the stop line; and when the drive that holds the distro's disk has less free
space than the floor.

The job runs as

    systemd-run --user --scope --quiet -p MemoryMax=<cap> -p MemorySwapMax=0 -- <python> <this file> --inner
        <job dir> --inner-lock-fd <fd> -- <command>

MemorySwapMax=0, as the sandbox's scope sets it, makes a job that reaches its cap fail rather than page into
WSL's swap, a file on a Windows drive (inferred), and slow the desktop.

The lock's descriptor goes to systemd-run through subprocess's pass_fds, which leaves it open and inheritable in
the child only. systemd-run --scope registers the scope and then execs the command in place, so the descriptor
reaches the inner mode at the same number (inferred from systemd-run(1); not yet run in the distro). The inner
mode keeps it open while the command runs and hands it to the command, so the scope's processes share the outer
process's open file description, and the flock lasts until the last of them closes it: killing the runner frees no
lock while the job runs on, and a process the job leaves running keeps the lock, so the next job refuses until
that process ends.

The inner mode refuses a lock descriptor that is not an open regular file, holds RLIMIT_CORE at
(1, 1) and reads it back, reads its own cgroup and that cgroup's memory.max (the cap the scope took; it refuses
to run uncapped), runs the command under ionice and nice with its output in <job dir>/log.txt, and writes
<job dir>/inner.json with memory.peak, or with the reason it refused. The outer process writes
<job dir>/job.json under LASSI_RUNS_ROOT/jobs/<id>/, before the launch and again after it, whatever the command's
status, and exits with that status. Standard library only; it runs under the distro's system python3.10.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, NamedTuple

REQUIRED = (
    "LASSI_HOST", "LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_JOB_MEMORY_MAX", "LASSI_STOP_GIB",
    "LASSI_HOST_DRIVE", "LASSI_DRIVE_FLOOR_GIB",
)
# One safe path component, as the gate's job ids allow (tools/server/gate.py, ID_RE).
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
# A bounded systemd size, as tools/local/host-profile.sh accepts: no "infinity", no percentage.
MEMORY_RE = re.compile(r"^[1-9][0-9]*[KMGT]$")
WHOLE_RE = re.compile(r"^[0-9]+$")
GIB_KIB = 1024 * 1024
LOCK_NAME = "longjob.lock"
PROFILE_HINT = "source tools/local/host-profile.sh first (AGENTS.md, Local Execution)"


class Refusal(Exception):
    """A reason not to start, or not to run, the job; main prints it and returns 2."""


class Policy(NamedTuple):
    """The host profile's values the runner enforces."""

    scratch: str
    runs_root: str
    memory_max: str
    stop_gib: int
    drive: str
    floor_gib: int
    host: str


def refuse(message: str) -> int:
    """Print a refusal to stderr and return the refusal status, 2."""
    print(f"longjob: {message}", file=sys.stderr)
    return 2


def exit_code(returncode: int) -> int:
    """Return a process's status as a shell reports it: 128 + N for a death by signal N."""
    return 128 - returncode if returncode < 0 else returncode


def utc_text(moment: datetime) -> str:
    """Return a UTC ISO 8601 date with its offset, to the second."""
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def write_json(path: str, data: Mapping[str, Any]) -> None:
    """Write `data` as plain ASCII JSON, replacing the file in one step."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="ascii", newline="\n") as handle:
        handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


# Host calls. main reaches the host only through these, which the tests replace.


def acquire_lock(path: str, fcntl_module: Any = None) -> Any:
    """Open `path` and take an exclusive non-blocking flock; return the open file, or None while another holds it.

    The lock lasts while the file stays open and ends with the process, so a crashed runner leaves none behind.
    """
    if fcntl_module is None:
        import fcntl as fcntl_module
    handle = open(path, "a+b")
    try:
        fcntl_module.flock(handle.fileno(), fcntl_module.LOCK_EX | fcntl_module.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    except BaseException:
        handle.close()
        raise
    return handle


def run_text(argv: Sequence[str]) -> str:
    """Run a short command and return its stdout; raise CalledProcessError when it fails."""
    return subprocess.run(list(argv), check=True, capture_output=True, text=True, timeout=3600).stdout


def scratch_kib(root: str, run: Callable[[Sequence[str]], str] = run_text) -> int:
    """Return the size of the scratch root in KiB, as `du -sk` reports it."""
    out = run(["du", "-sk", root])
    return int(out.split()[0])


def free_kib(path: str) -> int:
    """Return the free space, in KiB, of the file system that holds `path` (statvfs f_bavail on POSIX)."""
    return shutil.disk_usage(path).free // 1024


def git_state(cwd: str) -> tuple[str | None, bool | None]:
    """Return (commit, dirty) of the git checkout at `cwd`, or (None, None) outside one.

    dirty follows tools/rx.py: any output from `git status --porcelain`, untracked files included.
    """
    def git(*args: str) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError):
            return None

    head = git("rev-parse", "--verify", "HEAD")
    if head is None or head.returncode != 0 or not head.stdout.strip():
        return None, None
    status = git("status", "--porcelain")
    if status is None or status.returncode != 0:
        return head.stdout.strip(), None
    return head.stdout.strip(), bool(status.stdout.strip())


def launch(argv: Sequence[str], pass_fds: Sequence[int] = ()) -> int:
    """Run the systemd-run command line in the foreground and return its status (the job's, under --scope).

    The descriptors in `pass_fds` stay open, at the same numbers, in systemd-run and so in the command it execs.
    """
    try:
        completed = subprocess.run(list(argv), check=False, pass_fds=tuple(pass_fds))
    except OSError as exc:
        print(f"longjob: cannot start {argv[0]}: {exc}", file=sys.stderr)
        return 127
    return exit_code(completed.returncode)


def now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(timezone.utc)


# Arguments.


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse `--name <name> -- <command...>` or `--inner <job dir> --inner-lock-fd <fd> -- <command...>`.

    Everything after the first "--" is the command, options included. Usage errors exit 2. The inner mode, which
    only the outer mode starts, needs the lock's descriptor, so it never runs a job without the lock.
    """
    parser = argparse.ArgumentParser(
        prog="longjob.py", description="Run one long job in a memory-capped systemd user scope.",
        usage="%(prog)s (--name NAME | --inner JOB_DIR --inner-lock-fd FD) -- COMMAND [ARGS...]",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--name", help="a short name for the job (letters, digits, '.', '_', '-'; 40 at most)")
    mode.add_argument("--inner", metavar="JOB_DIR", help=argparse.SUPPRESS)
    parser.add_argument("--inner-lock-fd", metavar="FD", type=int, help=argparse.SUPPRESS)
    words = list(argv)
    split = words.index("--") if "--" in words else len(words)
    args = parser.parse_args(words[:split])
    args.command = words[split + 1:]
    if split == len(words) or not args.command:
        parser.error("give the command after --")
    if args.name is not None and not NAME_RE.match(args.name):
        parser.error(f"--name must be one safe path component (letters, digits, '.', '_', '-'), not {args.name!r}")
    if (args.inner is None) != (args.inner_lock_fd is None):
        parser.error("--inner and --inner-lock-fd go together, and only the outer mode passes them")
    if args.inner_lock_fd is not None and args.inner_lock_fd < 0:
        parser.error(f"--inner-lock-fd must be a descriptor number, not {args.inner_lock_fd}")
    return args


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    """Run the outer mode, or with --inner the inner mode; return the job's status, or 2 on a refusal."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.inner is not None:
        return inner_main(args.inner, args.command, lock_fd=args.inner_lock_fd)
    return outer_main(args.name, args.command, os.environ if environ is None else environ)


# The outer mode.


def read_policy(environ: Mapping[str, str]) -> Policy:
    """Return the host profile's values from `environ`, refusing a missing, empty, or malformed one."""
    missing = [name for name in REQUIRED if not environ.get(name)]
    if missing:
        raise Refusal(f"{', '.join(missing)} not set; {PROFILE_HINT}. None has a default")
    if environ["LASSI_HOST"] != "workstation":
        raise Refusal(f"LASSI_HOST is {environ['LASSI_HOST']!r}; this runner serves only the workstation")
    for name in ("LASSI_STOP_GIB", "LASSI_DRIVE_FLOOR_GIB"):
        if not WHOLE_RE.match(environ[name]):
            raise Refusal(f"{name} must be a whole number of GiB, not {environ[name]!r}")
    if not MEMORY_RE.match(environ["LASSI_JOB_MEMORY_MAX"]):
        raise Refusal(f"LASSI_JOB_MEMORY_MAX must be a bounded systemd size such as 20G, "
                      f"not {environ['LASSI_JOB_MEMORY_MAX']!r}")
    scratch, runs_root = environ["LASSI_SCRATCH"], environ["LASSI_RUNS_ROOT"]
    real_scratch, real_runs = os.path.realpath(scratch), os.path.realpath(runs_root)
    try:
        inside = os.path.commonpath([real_scratch, real_runs]) == real_scratch
    except ValueError:
        inside = False
    if not inside:
        raise Refusal(f"LASSI_RUNS_ROOT {runs_root} lies outside LASSI_SCRATCH {scratch}; job records stay under "
                      "the scratch root (Host Facts, Workstation scratch)")
    return Policy(scratch=scratch, runs_root=runs_root, memory_max=environ["LASSI_JOB_MEMORY_MAX"],
                  stop_gib=int(environ["LASSI_STOP_GIB"]), drive=environ["LASSI_HOST_DRIVE"],
                  floor_gib=int(environ["LASSI_DRIVE_FLOOR_GIB"]), host=environ["LASSI_HOST"])


def take_lock(policy: Policy) -> Any:
    """Return the held job lock under the scratch root, refusing while another job holds it."""
    path = os.path.join(policy.scratch, LOCK_NAME)
    try:
        lock = acquire_lock(path)
    except OSError as exc:
        raise Refusal(f"cannot open the job lock {path}: {exc}") from exc
    if lock is None:
        raise Refusal(f"another long job holds {path}, or a process one left running does; one job at a time "
                      "(AGENTS.md, Local Execution)")
    return lock


def check_space(policy: Policy) -> tuple[int, int]:
    """Return (scratch KiB, drive free KiB), refusing at or past the stop line or below the drive's floor."""
    try:
        used = scratch_kib(policy.scratch)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        raise Refusal(f"cannot read du -sk of the scratch root {policy.scratch}: {exc}") from exc
    if used >= policy.stop_gib * GIB_KIB:
        raise Refusal(f"the scratch root {policy.scratch} holds {used} KiB ({used / GIB_KIB:.1f} GiB), at or past "
                      f"the {policy.stop_gib} GiB stop line (LASSI_STOP_GIB)")
    try:
        free = free_kib(policy.drive)
    except OSError as exc:
        raise Refusal(f"cannot read the free space of LASSI_HOST_DRIVE {policy.drive}: {exc}") from exc
    if free < policy.floor_gib * GIB_KIB:
        raise Refusal(f"the drive {policy.drive} has {free} KiB ({free / GIB_KIB:.1f} GiB) free, below the "
                      f"{policy.floor_gib} GiB floor (LASSI_DRIVE_FLOOR_GIB)")
    return used, free


def scope_argv(memory_max: str, job_dir: str, lock_fd: int, command: Sequence[str]) -> list[str]:
    """Return the systemd-run command line that starts the inner mode, holding lock_fd, in a capped user scope."""
    return ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={memory_max}", "-p",
            "MemorySwapMax=0", "--", sys.executable, os.path.abspath(__file__), "--inner", job_dir,
            "--inner-lock-fd", str(lock_fd), "--", *command]


def scope_values(job_dir: str) -> dict[str, Any]:
    """Return what the inner mode read inside the scope, or nulls with the reason when inner.json is missing."""
    path = os.path.join(job_dir, "inner.json")
    try:
        with open(path, encoding="ascii") as handle:
            inner = json.load(handle)
    except (OSError, ValueError) as exc:
        reason = f"no readable {path} ({exc.__class__.__name__}): the inner mode did not finish"
        return {"memory_max_read": None, "memory_peak": None, "memory_peak_reason": reason,
                "cgroup": None, "rlimit_core": None}
    values = {key: inner.get(key) for key in ("cgroup", "rlimit_core", "memory_peak")}
    values["memory_max_read"] = inner.get("memory_max")
    if values["memory_peak"] is None:
        values["memory_peak_reason"] = (inner.get("memory_peak_reason") or inner.get("refused")
                                        or f"{path} holds no memory_peak")
    if inner.get("refused"):
        values["inner_refused"] = inner["refused"]
    return values


def outer_main(name: str, command: Sequence[str], environ: Mapping[str, str]) -> int:
    """Check the profile and the host, then run the job under the held lock; return its status or 2."""
    try:
        policy = read_policy(environ)
        lock = take_lock(policy)
    except Refusal as exc:
        return refuse(str(exc))
    try:
        return run_job(name, list(command), policy, lock.fileno())
    except Refusal as exc:
        return refuse(str(exc))
    finally:
        lock.close()


def run_job(name: str, command: list[str], policy: Policy, lock_fd: int) -> int:
    """Run one job whose lock is held: check the space, record, launch, and record again; return its status.

    lock_fd, the held lock's descriptor, goes to the scope, so the job holds the lock even if this process dies.
    """
    scratch_before, free_before = check_space(policy)
    start = now()
    job_id = f"{start.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{name}"
    job_dir = os.path.join(policy.runs_root, "jobs", job_id)
    try:
        os.makedirs(os.path.dirname(job_dir), exist_ok=True)
        os.mkdir(job_dir)
    except OSError as exc:
        raise Refusal(f"cannot create the job directory {job_dir}: {exc}") from exc
    cwd = os.getcwd()
    commit, dirty = git_state(cwd)
    record: dict[str, Any] = {
        "id": job_id, "name": name, "command": command, "cmd": shlex.join(command), "cwd": cwd,
        "commit": commit, "dirty": dirty, "start": utc_text(start), "end": None, "exit_status": None, "rc": None,
        "scratch": policy.scratch, "drive": policy.drive, "stop_gib": policy.stop_gib,
        "drive_floor_gib": policy.floor_gib, "scratch_kib_before": scratch_before, "scratch_kib_after": None,
        "drive_free_kib_before": free_before, "drive_free_kib_after": None,
        "memory_max_requested": policy.memory_max, "log": os.path.join(job_dir, "log.txt"),
        "host": policy.host, "hostname": platform.node(),
    }
    record_path = os.path.join(job_dir, "job.json")
    write_json(record_path, record)
    print(f"longjob: job {job_id} started; record and log in {job_dir}", file=sys.stderr)
    try:
        status = launch(scope_argv(policy.memory_max, job_dir, lock_fd, command), pass_fds=(lock_fd,))
    except KeyboardInterrupt:
        status = 130
    record.update({"end": utc_text(now()), "exit_status": status, "rc": status})
    record["scratch_kib_after"] = measured(lambda: scratch_kib(policy.scratch))
    record["drive_free_kib_after"] = measured(lambda: free_kib(policy.drive))
    record.update(scope_values(job_dir))
    write_json(record_path, record)
    print(f"longjob: job {job_id} ended with status {status}; record {record_path}", file=sys.stderr)
    return status


def measured(read: Callable[[], int]) -> int | None:
    """Return read(), or None when the host cannot be read after the job."""
    try:
        return read()
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


# The inner mode, inside the scope.


def wrap(command: Sequence[str], which: Callable[[str], str | None] = shutil.which) -> list[str]:
    """Put the command under "ionice -c2 -n7 nice -n 10", ionice only when present, as the gate's wrap does."""
    prefix = ["nice", "-n", "10"]
    if which("ionice"):
        prefix = ["ionice", "-c2", "-n7", *prefix]
    return [*prefix, *command]


def hold_core_limit(resource_module: Any) -> list[int]:
    """Set RLIMIT_CORE to 1 byte, soft and hard, and return it read back; refuse unless it reads (1, 1).

    WSL pipes core dumps to its crash handler, which stores them under %TEMP%\\wsl-crashes on C:; a piped core
    dump is aborted only when the limit is exactly 1 byte (Host Facts, Workstation scratch;
    plans/spikes/p0-sandbox-hardening.md).
    """
    try:
        resource_module.setrlimit(resource_module.RLIMIT_CORE, (1, 1))
        held = [int(value) for value in resource_module.getrlimit(resource_module.RLIMIT_CORE)]
    except (OSError, ValueError) as exc:
        raise Refusal(f"cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard: {exc}") from exc
    if held != [1, 1]:
        raise Refusal(f"RLIMIT_CORE reads back as {tuple(held)}, not (1, 1); a crashing tool could leave a core dump "
                      "outside the scratch root")
    return held


def read_scope(proc_root: str, cgroup_root: str) -> tuple[str, str]:
    """Return (this process's cgroup v2 path, the directory of that cgroup) from <proc_root>/self/cgroup."""
    path = os.path.join(proc_root, "self", "cgroup")
    try:
        with open(path, encoding="ascii", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError as exc:
        raise Refusal(f"cannot read {path}: {exc}") from exc
    for line in lines:
        if line.startswith("0::"):
            cgroup = line[3:].strip()
            return cgroup, os.path.join(cgroup_root, *[part for part in cgroup.split("/") if part])
    raise Refusal(f"{path} names no cgroup v2 entry (0::); the scope's memory cap cannot be read back")


def read_memory_max(scope_dir: str) -> str:
    """Return the scope's memory.max, stripped; refuse when no cap took effect."""
    path = os.path.join(scope_dir, "memory.max")
    try:
        with open(path, encoding="ascii", errors="replace") as handle:
            text = handle.read().strip()
    except OSError as exc:
        raise Refusal(f"cannot read {path} ({exc}); is the memory controller delegated to the user manager?") from exc
    if not text.isdigit():
        raise Refusal(f"{path} reads {text!r}: no memory cap took effect, and a job never runs uncapped")
    return text


def read_peak(scope_dir: str) -> tuple[int | None, str | None]:
    """Return (memory.peak in bytes, None), or (None, the reason) when it cannot be read."""
    path = os.path.join(scope_dir, "memory.peak")
    try:
        with open(path, encoding="ascii", errors="replace") as handle:
            text = handle.read().strip()
    except OSError as exc:
        return None, f"cannot read {path} ({exc.__class__.__name__}); memory.peak needs Linux 5.19 or later"
    if not text.isdigit():
        return None, f"{path} reads {text!r}, not a number of bytes"
    return int(text), None


def check_lock_fd(fd: int) -> int:
    """Return `fd` when it is an open regular file, as the job lock's inherited descriptor is; refuse otherwise."""
    try:
        mode = os.fstat(fd).st_mode
    except OSError as exc:
        raise Refusal(f"the job lock's descriptor {fd} is not open ({exc}); a job never runs without the "
                      "lock") from exc
    if not stat.S_ISREG(mode):
        raise Refusal(f"descriptor {fd} is not a regular file, so it is not the job lock; a job never runs without "
                      "the lock")
    return fd


def run_logged(command: Sequence[str], log_path: str, pass_fds: Sequence[int] = ()) -> int:
    """Run the command with stdin closed and stdout and stderr appended to `log_path`; return its status.

    The descriptors in `pass_fds` stay open in the command, at the same numbers; every other one is closed.
    """
    with open(log_path, "ab") as log:
        try:
            completed = subprocess.run(list(command), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       pass_fds=tuple(pass_fds))
        except OSError as exc:
            log.write(f"longjob: cannot start {command[0]}: {exc}\n".encode("ascii", errors="replace"))
            return 127
    return exit_code(completed.returncode)


def inner_main(
    job_dir: str, command: Sequence[str], *, lock_fd: int | None = None, proc_root: str = "/proc",
    cgroup_root: str = "/sys/fs/cgroup", resource_module: Any = None,
) -> int:
    """Inside the scope: check the lock, hold the core limit, read the cap back, run the command, write inner.json.

    lock_fd is the job lock's descriptor, inherited from the outer mode (the CLI requires it). It is never closed
    here, so it stays open until this process ends, after the command; the command inherits it too. Returns the
    command's status, or 2 when it refuses to run the command; a refusal is recorded in inner.json as `refused`.
    """
    if resource_module is None:
        import resource as resource_module
    record: dict[str, Any] = {"lock_fd": None, "rlimit_core": None, "cgroup": None, "memory_max": None}
    record_path = os.path.join(job_dir, "inner.json")
    try:
        if not os.path.isdir(job_dir):
            raise Refusal(f"the job directory {job_dir} does not exist")
        if lock_fd is not None:
            record["lock_fd"] = check_lock_fd(lock_fd)
        record["rlimit_core"] = hold_core_limit(resource_module)
        record["cgroup"], scope_dir = read_scope(proc_root, cgroup_root)
        record["memory_max"] = read_memory_max(scope_dir)
    except Refusal as exc:
        if os.path.isdir(job_dir):
            reason = f"the command did not run: {exc}"
            write_json(record_path, {**record, "refused": str(exc), "memory_peak": None, "memory_peak_reason": reason})
        return refuse(str(exc))
    kept = () if lock_fd is None else (lock_fd,)
    status = run_logged(wrap(command), os.path.join(job_dir, "log.txt"), pass_fds=kept)
    record["exit_status"] = status
    record["memory_peak"], reason = read_peak(scope_dir)
    if reason is not None:
        record["memory_peak_reason"] = reason
    write_json(record_path, record)
    return status


if __name__ == "__main__":
    sys.exit(main())
