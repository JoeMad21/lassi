"""The sandbox: the only module that runs generated code (Agent Rule 6; bible Sandbox).

Mechanism: plans/spikes/p0-sandbox.md (probes 1 to 3 and both addenda)
measured the scope, the namespaces, the read-only bind layout, prlimit --cpu,
and the innermost timeout on the build host. The steps marked "not measured"
below close gaps found in review; tests/executors/test_sandbox_remote.py
checks the whole composition there, and none of it counts as proven until
those tests pass. One command runs each program:

    systemd-run --user --scope --quiet -p MemoryMax=... -p MemorySwapMax=0
        -p TasksMax=... -p RuntimeMaxSec=... -p TimeoutStopSec=1
      unshare -rinmpfu --mount-proc
        sh -c SETUP_SCRIPT sh <workdir> <harness> <N> <roots...> <cpu> <wall> <kill-after> <argv...>

- systemd-run --user --scope puts the run in its own cgroup, which enforces
  the memory limit (a hog is killed, exit status 137) and a task limit.
  RuntimeMaxSec is a backstop OUTER_MARGIN_S seconds past the wall limit
  and its kill grace; TimeoutStopSec=1 makes its SIGKILL follow one second
  after its SIGTERM. The backstop is not what enforces wall time, and it
  was not measured in this composition.
- unshare gives the run its own user, IPC, network, mount, pid, and UTS
  namespaces: no network (a connect fails), no view of host processes, no
  SysV IPC or POSIX message queue shared with the host user's processes,
  and its own hostname (IPC and UTS not measured).
- SETUP_SCRIPT runs inside the namespaces with `set -eu`, so a failed step
  stops it before the program starts. In order, it:
  1. checks that env, nice, setpriv, prlimit, and timeout are on PATH;
  2. binds the per-trial workdir onto itself, the one writable host directory;
  3. remounts each read-only root (rbind, then remount) and the harness
     read-only;
  4. remounts / and /sys/fs/cgroup read-only (not measured), so nothing
     outside the workdir is written on the host root filesystem (Agent Rule
     7) and the run cannot rewrite its own cgroup limits, such as memory.max;
  5. mounts a private tmpfs on each of PRIVATE_DIRS that exists (/run not
     measured). The /run mount hides the user and system bus sockets, which
     a network namespace does not isolate, so the program cannot ask a
     service manager to start anything outside the sandbox;
  6. writes 0 to /proc/sys/user/max_user_namespaces while it still holds
     its capabilities, which sets the sandbox user namespace's own limit
     (the kernel keeps it per user namespace; the host value never
     changes), so the program cannot create a nested user namespace and
     regain capabilities there (measured only in an exploratory remote run);
  7. prints READY_MARKER on stderr and execs the program under, in order:
     - env -i with only PATH, HOME (the workdir), LANG, and TMPDIR=/tmp (not
       measured), so no API key or other secret in the caller's environment
       reaches generated code (Agent Rule 12);
     - nice -n 19 (not measured), so the program runs at the lowest CPU
       priority;
     - setpriv --no-new-privs with empty inheritable and bounding sets (not
       measured): the program is uid 0 of its user namespace but holds no
       capability, so it cannot unmount or remount anything, nor raise the
       limit step 6 set;
     - prlimit --cpu (a CPU-time cap; the cgroup cpu controller is not
       delegated, OQ-011) and --core=0 (no core files in the workdir);
     - an innermost `timeout --kill-after`, which is what enforces wall time.
       The spike's first addendum ran a program inside the namespaces under
       an outer timeout and RuntimeMaxSec, and measured that wall time was
       not enforced with both in place (the cause is an inference); the
       second measured the innermost timeout stopping it (exit status 124).

The script is a constant. Every value reaches it as a positional argument,
so no path or argument is ever parsed by a shell. A failure before the ready
marker (a missing tool, a failed mount, or a failed write) stops the script,
and Sandbox.run reports it as SandboxUnavailableError, so a sandbox that
never started is never reported as the program's exit status. A failure of
env, nice, setpriv, prlimit, or timeout after the marker would surface as the
program's exit status (the tools are checked with command -v first).
Sandbox.run reports a death by signal N as the shell does, 128 + N.

Known limits (task P0.16), not yet done: host /dev is visible (there is no
private /dev); only /, /sys/fs/cgroup, the harness, and the listed read-only
roots are read-only, not every host mount, so a filesystem mounted apart
from those keeps the host user's write permission; files the host user can
read stay readable, including $HOME and other trials under the scratch root
(a default-deny view is pending); stdout, stderr, and workdir disk use are
unbounded, and the runner holds stdout and stderr in memory; the runner's
own timeout reaches only the process group, so cleanup of anything left
running relies on the RuntimeMaxSec backstop. Native runs of generated code
must wait for P0.16.
"""

from __future__ import annotations

import math
import os
import posixpath
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath

from lassi.core.interfaces import Limits
from lassi.toolchains import CommandResult, CommandRunner, subprocess_runner

# Grace seconds between the innermost timeout's SIGTERM and its SIGKILL.
KILL_AFTER_S = 2
# Seconds past the wall limit and KILL_AFTER_S before the scope's RuntimeMaxSec backstop fires.
OUTER_MARGIN_S = 5
# Seconds past the backstop before the runner itself gives up on the command (returncode -1).
_RUNNER_MARGIN_S = 10
# The line SETUP_SCRIPT prints on stderr once setup is done, right before it execs the program.
READY_MARKER = "lassi-sandbox-ready"
# The directories SETUP_SCRIPT covers with a private tmpfs, in its order; anything under them is hidden inside.
PRIVATE_DIRS = ("/tmp", "/var/tmp", "/dev/shm", "/run")
# How much of a failed setup's stderr a SandboxUnavailableError quotes.
_STDERR_TAIL = 2000

# Positional layout: workdir, harness or "", N, the N read-only roots, cpu seconds, wall seconds,
# kill-after seconds, then the program argv. The steps and their order are listed in the module
# docstring. The root loop counts with a string's length, since the script holds no command or
# arithmetic substitution.
SETUP_SCRIPT = r"""set -eu
for tool in env nice setpriv prlimit timeout; do
  command -v "$tool" > /dev/null
done
workdir=$1
harness=$2
count=$3
shift 3
mount --bind "$workdir" "$workdir"
done_roots=
while [ "${#done_roots}" -lt "$count" ]; do
  mount --rbind "$1" "$1"
  mount -o remount,bind,ro "$1"
  shift
  done_roots="${done_roots}x"
done
if [ -n "$harness" ]; then
  mount --bind "$harness" "$harness"
  mount -o remount,bind,ro "$harness"
fi
mount -o remount,bind,ro,nosuid,nodev /
if [ -d /sys/fs/cgroup ]; then
  mount --rbind /sys/fs/cgroup /sys/fs/cgroup
  mount -o remount,bind,ro,nosuid,nodev,noexec /sys/fs/cgroup
fi
for dir in /tmp /var/tmp /dev/shm /run; do
  if [ -d "$dir" ]; then
    mount -t tmpfs -o size=64m,mode=1777 tmpfs "$dir"
  fi
done
echo 0 > /proc/sys/user/max_user_namespaces
cd "$workdir"
cpu=$1
wall=$2
kill_after=$3
shift 3
echo lassi-sandbox-ready >&2
exec env -i PATH="$PATH" HOME="$workdir" LANG=C.UTF-8 TMPDIR=/tmp \
  nice -n 19 \
  setpriv --no-new-privs --inh-caps=-all --bounding-set=-all -- \
  prlimit --cpu="$cpu" --core=0 -- \
  timeout --kill-after="$kill_after" "$wall" "$@"
"""


class SandboxUnavailableError(RuntimeError):
    """The sandbox cannot run the program: a tool or the workdir is missing, setup failed, or no root is read-only.

    The program did not run, and nothing is ever run unsandboxed instead.
    """


def _private_dir_holding(path: PurePath) -> str | None:
    """Return the entry of PRIVATE_DIRS that `path` is or lies under, compared as normalized POSIX text, or None.

    The comparison is lexical: `..` segments are resolved, symbolic links are
    not followed.
    """
    text = posixpath.normpath(path.as_posix())
    if text.startswith("//"):
        text = "/" + text.lstrip("/")
    posix = PurePosixPath(text)
    for name in PRIVATE_DIRS:
        private = PurePosixPath(name)
        if posix == private or private in posix.parents:
            return name
    return None


def _check_mount_path(what: str, path: PurePath) -> None:
    """Raise ValueError unless `path` is absolute and outside every directory SETUP_SCRIPT covers with tmpfs."""
    if not path.is_absolute():
        raise ValueError(f"the sandbox {what} must be an absolute path, got {str(path)!r}")
    private = _private_dir_holding(path)
    if private is not None:
        raise ValueError(f"the sandbox {what} {str(path)!r} lies under {private}, which a private tmpfs hides inside")


@dataclass(frozen=True)
class SandboxSpec:
    """Where a sandboxed run may write and what it sees read-only.

    `workdir` is the per-trial build directory, the only writable host
    directory. `readonly_roots` are host directories remounted read-only
    inside (for example the scratch root and $HOME); at least one is
    required. `harness`, when set, is bind-mounted read-only at its own path.
    `tasks_max` caps the number of tasks in the run's cgroup. A ValueError is
    raised when a path is relative or lies under one of PRIVATE_DIRS (the
    private tmpfs would hide it), when the workdir is a read-only root, when
    the harness is or contains the workdir (its read-only mount would cover
    the workdir), or when tasks_max is not an integer >= 1. Paths are stored
    as Path and the roots as a tuple.
    """

    workdir: Path
    readonly_roots: tuple[Path, ...]
    harness: Path | None = None
    tasks_max: int = 256

    def __post_init__(self) -> None:
        """Normalize the paths to Path and check them and tasks_max; raise ValueError on the first problem."""
        if isinstance(self.readonly_roots, (str, os.PathLike)):
            raise ValueError(f"readonly_roots must be a sequence of paths, got {self.readonly_roots!r}")
        roots = tuple(Path(root) for root in self.readonly_roots)
        object.__setattr__(self, "workdir", Path(self.workdir))
        object.__setattr__(self, "readonly_roots", roots)
        if self.harness is not None:
            object.__setattr__(self, "harness", Path(self.harness))
        if not roots:
            raise ValueError("a sandbox needs at least one read-only root")
        named = [("workdir", self.workdir), *(("readonly root", root) for root in roots)]
        if self.harness is not None:
            named.append(("harness", self.harness))
        for what, path in named:
            _check_mount_path(what, path)
        if self.workdir in roots:
            raise ValueError(f"the workdir {str(self.workdir)!r} cannot also be a read-only root")
        if self.harness is not None and (self.harness == self.workdir or self.harness in self.workdir.parents):
            raise ValueError(f"the harness {str(self.harness)!r} cannot be or contain the workdir")
        if isinstance(self.tasks_max, bool) or not isinstance(self.tasks_max, int) or self.tasks_max < 1:
            raise ValueError(f"tasks_max must be an integer >= 1, got {self.tasks_max!r}")


@dataclass(frozen=True)
class SandboxResult:
    """How a sandboxed run ended.

    `returncode` is in the shell's form (a death by signal N is 128 + N; -1
    is the runner's own timeout). `stderr` is the program's, without the
    setup script's ready line. `wall_s` is the elapsed time of the whole
    sandbox command, setup included. `hang` means the run reached the wall
    limit and ended by a timeout or a kill; `killed` means a SIGKILL death
    before the wall limit, which may be a memory or CPU-time limit kill or
    the program killing itself (see classify).
    """

    returncode: int
    stdout: str
    stderr: str
    wall_s: float
    hang: bool
    killed: bool


def _check_request(argv: Sequence[str], limits: Limits) -> None:
    """Raise ValueError unless argv is a non-empty sequence of strings and the limits can be enforced."""
    if isinstance(argv, str) or not argv:
        raise ValueError(f"argv must be a non-empty sequence of strings, got {argv!r}")
    if not all(isinstance(part, str) for part in argv):
        raise ValueError(f"every argv element must be a string, got {list(argv)!r}")
    if not (isinstance(limits.wall_s, (int, float)) and math.isfinite(limits.wall_s) and limits.wall_s > 0):
        raise ValueError(f"wall_s must be a finite number > 0, got {limits.wall_s!r}")
    for name in ("memory_mb", "cpus"):
        value = getattr(limits, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


def _wall_seconds(limits: Limits) -> int:
    """Return the whole-second wall limit the innermost timeout enforces: wall_s rounded up, at least 1."""
    return max(1, math.ceil(limits.wall_s))


def sandbox_command(spec: SandboxSpec, argv: Sequence[str], limits: Limits) -> list[str]:
    """Return the command that runs `argv` in the sandbox described by `spec` under `limits`.

    The wall limit is wall_s rounded up to whole seconds (at least 1). The
    CPU-time budget is wall_s x cpus, rounded up (at least 1): CPU is capped
    as CPU time because the cgroup cpu controller is not delegated (OQ-011).
    The backstop RuntimeMaxSec is the wall limit plus KILL_AFTER_S plus
    OUTER_MARGIN_S. Every path and argument is its own element after
    SETUP_SCRIPT, in the script's positional layout. Raise ValueError when
    argv is empty, or wall_s is not > 0, or memory_mb or cpus is below 1.
    """
    _check_request(argv, limits)
    wall = _wall_seconds(limits)
    cpu = max(1, math.ceil(limits.wall_s * limits.cpus))
    harness = "" if spec.harness is None else str(spec.harness)
    roots = [str(root) for root in spec.readonly_roots]
    scope = ["systemd-run", "--user", "--scope", "--quiet"]
    scope += ["-p", f"MemoryMax={limits.memory_mb}M", "-p", "MemorySwapMax=0", "-p", f"TasksMax={spec.tasks_max}"]
    scope += ["-p", f"RuntimeMaxSec={wall + KILL_AFTER_S + OUTER_MARGIN_S}", "-p", "TimeoutStopSec=1"]
    namespaces = ["unshare", "-rinmpfu", "--mount-proc"]
    setup = ["sh", "-c", SETUP_SCRIPT, "sh", str(spec.workdir), harness, str(len(roots)), *roots]
    setup += [str(cpu), str(wall), str(KILL_AFTER_S)]
    return [*scope, *namespaces, *setup, *argv]


def classify(returncode: int, wall_s: float, limit_wall_s: float) -> tuple[bool, bool]:
    """Return (hang, killed) for a sandboxed run's exit status and elapsed wall time.

    `returncode` is in the shell's form, as the spike recorded it with bash:
    a death by signal N is 128 + N, and -1 is the runner's own timeout.
    Sandbox.run converts the runner's -N (Popen's form) before calling this,
    so the backstop's SIGKILL reaching the runner as -9 arrives here as 137.
    The mapping follows the exit statuses in plans/spikes/p0-sandbox.md and
    its addenda. A hang needs both the status and the elapsed time, so a
    program cannot fake one by exiting 124 early:

    - 124, 137, 143, or -1 with wall_s >= limit_wall_s: hang. The run
      reached the wall limit, so the stop counts as the wall limit's: 124
      is the innermost timeout, 137 its kill after the grace or the scope's
      RuntimeMaxSec backstop (which also gives 143), and -1 the runner's own
      timeout (lassi.toolchains subprocess_runner).
    - 137 with wall_s < limit_wall_s: killed. A SIGKILL death before the
      wall limit: a memory (cgroup MemoryMax) or CPU-time (prlimit --cpu)
      limit kill, or the program killing itself, which the status cannot
      tell apart.
    - Anything else: (False, False), the program's own exit status. That
      includes 124, 143, and -1 before the wall limit.
    """
    if returncode in (124, 137, 143, -1) and wall_s >= limit_wall_s:
        return True, False
    if returncode == 137:
        return False, True
    return False, False


def _shell_status(returncode: int, elapsed_s: float, timeout_s: float) -> int:
    """Return the runner's exit status in the shell's form that classify reads.

    The runner (Popen) reports a death by signal N as -N, where a shell and
    the spike's measurements give 128 + N. -1 stays -1 only when the runner's
    own timeout expired (elapsed_s >= timeout_s); before that it is a death by
    SIGHUP, 129.
    """
    if returncode == -1 and elapsed_s >= timeout_s:
        return -1
    if returncode < 0:
        return 128 - returncode
    return returncode


def _split_ready(stderr: str) -> tuple[bool, str]:
    """Return whether stderr holds SETUP_SCRIPT's ready line, and stderr with that first line removed.

    The line counts only at the start of a line. Only setup output can come
    before it, since the program starts after it, so the program cannot
    forge a sandbox that started.
    """
    line = READY_MARKER + "\n"
    if stderr.startswith(line):
        return True, stderr[len(line) :]
    index = stderr.find("\n" + line)
    if index < 0:
        return False, stderr
    return True, stderr[: index + 1] + stderr[index + 1 + len(line) :]


def _setup_failure(result: CommandResult) -> str:
    """Return the SandboxUnavailableError message for a run whose setup never reached the program."""
    tail = result.stderr.strip()[-_STDERR_TAIL:] or "(no stderr)"
    return f"the sandbox setup did not finish (exit status {result.returncode}); the program did not run: {tail}"


class Sandbox:
    """Runs a program only inside the sandbox; nothing ever runs unsandboxed."""

    def __init__(self, *, runner: CommandRunner | None = None) -> None:
        """Keep the command runner; None means lassi.toolchains subprocess_runner."""
        self.runner: CommandRunner = subprocess_runner if runner is None else runner

    def run(self, spec: SandboxSpec, argv: Sequence[str], limits: Limits) -> SandboxResult:
        """Run `argv` in the sandbox described by `spec` under `limits` and classify how it ended.

        The runner gets sandbox_command(spec, argv, limits), the workdir as
        its cwd, and a timeout 10 s past the backstop. Wall time is measured
        with time.monotonic around the runner call. The runner's status is
        converted to the shell's form (_shell_status), and classify compares
        the elapsed time with the whole-second wall limit the innermost
        timeout enforces. Validation errors raise ValueError before anything
        runs. SandboxUnavailableError is raised when the runner raises
        FileNotFoundError (a missing tool or workdir, named in the message)
        and when stderr lacks the setup script's ready line (setup failed
        before the marker, so the program never ran); the message quotes the
        setup's stderr. A failure of a wrapper that SETUP_SCRIPT execs after
        the marker comes back as the program's exit status.
        """
        command = sandbox_command(spec, argv, limits)
        wall = _wall_seconds(limits)
        timeout_s = wall + KILL_AFTER_S + OUTER_MARGIN_S + _RUNNER_MARGIN_S
        start = time.monotonic()
        try:
            result = self.runner(command, spec.workdir, timeout_s)
        except FileNotFoundError as exc:
            missing = exc.filename or command[0]
            raise SandboxUnavailableError(
                f"the sandbox cannot start: {missing} was not found (a tool it needs, or the workdir); nothing ran"
            ) from exc
        elapsed = time.monotonic() - start
        ready, stderr = _split_ready(result.stderr)
        if not ready:
            raise SandboxUnavailableError(_setup_failure(result))
        returncode = _shell_status(result.returncode, elapsed, timeout_s)
        hang, killed = classify(returncode, elapsed, wall)
        return SandboxResult(
            returncode=returncode,
            stdout=result.stdout,
            stderr=stderr,
            wall_s=elapsed,
            hang=hang,
            killed=killed,
        )
