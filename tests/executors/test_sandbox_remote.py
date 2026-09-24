"""Remote tests for the sandbox and the native executor on the build host (P0.10, hardened in P0.16).

These run real sandboxes, so they are marked `remote` and skip unless the
host has what the mechanism in plans/spikes/p0-sandbox.md and
plans/spikes/p0-sandbox-hardening.md needs: Linux, unshare, systemd-run,
nice, setpriv, prlimit, timeout, python3, and awk on PATH, a python3 in the
sandbox's own PATH (SANDBOX_PATH) and one on PATH outside $HOME and the
scratch root (the sandbox hides both, and the test run's own interpreter
lives under the scratch root on the build host), a reachable user systemd
manager ($XDG_RUNTIME_DIR/bus exists), and $LASSI_SCRATCH. With
LASSI_REQUIRE_SANDBOX=1 a host that cannot run them fails the tests instead
of skipping them, so a silent skip cannot pass for evidence. Run them with
`rx run -- 'LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/executors'`
(P0.10 and P0.16 Remote).

Each test builds a tiny program as a shell or python3 command (no compiler)
in a fresh build directory under $LASSI_SCRATCH/tmp (or, where a test needs
other trials beside it, under $LASSI_RUNS_ROOT), which is removed afterwards,
and runs it only through Sandbox.run or NativeExecutor.run (Agent Rule 6),
with $LASSI_SCRATCH and $HOME hidden, a temp harness directory, and
$LASSI_TOOLCHAINS re-exposed when it is set, unless a test says otherwise.
Nothing is written on the host root filesystem (Agent Rule 7): the root
filesystem and cgroup checks read /proc/self/mountinfo or rewrite a limit
with its own value, the tmp tests remove their marker file if the sandbox
ever lets it through, and the crash test refuses to crash anything unless the
command carries prlimit --core=1, so an unhardened sandbox never stores a
core under /var/lib/systemd/coredump.

The P0.10 tests check: a network connect fails, a memory hog and a CPU-time
hog are killed, wall time ends a run at the limit and not before, writes to
the harness and the hidden roots fail while the workdir stays writable, the
program holds no capability and cannot undo a mount, it cannot create a
nested user namespace, it runs at nice 19, the environment and the bus
sockets stay outside, a failed setup never runs the program, and the exit
status passes through.

The P0.16 tests check each hardening acceptance item as the P0.16 contract
words it:
- R1: /dev holds only null, zero, full, random, urandom, tty (each the host's
  node), a private devpts instance with its ptmx, and a private shm, plus the
  standard links; no accelerator or other host device node.
- R2: every mount of a host filesystem is read-only inside, and the only
  writable mounts are the setup's own (the workdir, the private tmpfs mounts,
  /dev with its devpts and shm, and the hidden-view tmpfs), each a new
  filesystem; a writable host mount injected before the check makes setup
  fail closed with SandboxUnavailableError, and the program never runs.
- R3: $HOME and the scratch root show only the path skeleton to the
  workdir, the harness, and $LASSI_TOOLCHAINS; another trial's file under
  the runs root cannot be read; the harness and toolchains root are
  readable and read-only, the workdir writable.
- R4: stdout and stderr are capped at OUTPUT_CAP_BYTES with truncation
  flags, and a runner draining 768 MiB grows by no more than two capped
  streams plus a fixed allowance, measured against a warm baseline run.
- R5: writes past a 4 MiB workdir cap fail inside with ENOSPC, EFBIG, or
  EDQUOT, the host workdir stays under the cap, and copy-back never writes
  through a host symbolic link; sparse files and hard links whose sizes
  total more than the cap are not copied back at all (the run reports the
  workdir incomplete); and the cap is at most half the memory limit, so a
  large write under a small memory limit ends in ENOSPC, not a memory kill.
- R6: the program cannot lower its core limit (checked first, so a broken
  filter never gets as far as a crash), and a SIGSEGV (self-sent, after a
  refused `ulimit -c 0`, and a real fault) and a SIGABRT leave no entry in
  `coredumpctl list` for the user's own uid (COREDUMP_UID) since the test
  began, polled for 15 s. That is what the user can read: the journal shows
  its own coredumps, and the kernel log is closed to it (dmesg_restrict=1).
- R7: when the runner's own timeout fires, no process of the sandbox
  survives, including a setsid child and an orphaned background child.

The fixes from the P0.16 review have their own checks: no readable
/proc/<pid>/environ inside holds a secret from the caller's environment; the
program runs in its own session with no controlling terminal, and none of
the caller's keys is readable (checked with a key added for the test in a
session keyring of its own), since a seccomp filter refuses every keyring
call (the host user's keyring included; round 3), as well as AF_VSOCK
sockets, io_uring, and limit changes of other processes; /sys
device attributes and every /var entry but tmp are hidden, and every socket
the host shows under /var, /snap, /opt, and /srv is out of view (never
connected to); and a PATH entry the program can write never reaches the
setup, which runs with a constant PATH. From the review's second round: the
copy-back never writes through a host hard link; a confinement that fails
(seccomp refused, injected as a negative control) raises
SandboxUnavailableError and the program never runs; and a done line the
program printed itself never passes for a finished copy-back when the
copy-back then fails (injected), because the setup's EXIT trap prints last.
From its third round: the program's own time from the done line's clock
readings brackets the program (a sleep past the limit, a run within it, and
an early exit 124 that is not a hang).

The limits and caps are test inputs; no value here is a measurement. No file
is written or deleted outside the temp directories these tests create; the
disk-cap tests write at most 8 MiB into a workdir under $LASSI_SCRATCH/tmp
when the sandbox works, and a regression of the sparse-file or hard-link
check would write at most 32 MiB there.
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from lassi.core.interfaces import Limits
from lassi.executors.sandbox import SANDBOX_PATH
from lassi.toolchains import CommandResult, capped_runner

REPO = Path(__file__).resolve().parents[2]
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk")
SHORT = Limits(wall_s=10.0, memory_mb=64, cpus=1)
# Room for a python3 interpreter.
PYTHON = Limits(wall_s=20.0, memory_mb=256, cpus=1)
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
# R1: the device nodes the private /dev may hold (each must be the host's node of that name), its only
# directories, and its standard links with their targets.
DEV_NODES = {"null", "zero", "full", "random", "urandom", "tty", "pts/ptmx"}
DEV_DIRS = {"pts", "shm"}
DEV_LINKS = {
    "fd": "/proc/self/fd",
    "stdin": "/proc/self/fd/0",
    "stdout": "/proc/self/fd/1",
    "stderr": "/proc/self/fd/2",
    "ptmx": "pts/ptmx",
}
# R1: host device nodes that must never appear (the contract's list), as shell-style patterns.
FORBIDDEN_DEV = ("rngd*", "tenstorrent*", "kfd", "dri", "nvidia*", "sd*", "nvme*", "mem")
# R2: filesystem types the setup may leave writable, and the private points it may mount them on.
SETUP_FSTYPES = {"tmpfs", "devpts", "overlay"}
PRIVATE_POINTS = ("/tmp", "/var/tmp", "/run", "/dev", "/dev/pts", "/dev/shm")
# R4: the memory test's allowance, in MiB, past two streams of OUTPUT_CAP_BYTES.
SLACK_MIB = 24


def hidden_default() -> tuple[Path, ...]:
    """Return $LASSI_SCRATCH and, when set and different, $HOME: the roots the sandbox hides."""
    roots = [Path(os.environ["LASSI_SCRATCH"])] if os.environ.get("LASSI_SCRATCH") else []
    home = os.environ.get("HOME", "")
    if home and Path(home) not in roots:
        roots.append(Path(home))
    return tuple(roots)


def is_within(path: Path, roots: Sequence[Path]) -> bool:
    """Return True when `path` is one of `roots` or lies under one."""
    return any(path == root or root in path.parents for root in roots)


def visible_python() -> str:
    """Return a python3 on PATH that lies, also once resolved, outside $HOME and the scratch root; "" if none."""
    hidden = hidden_default()
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        found = shutil.which("python3", path=entry) if entry else None
        if found and not is_within(Path(found), hidden) and not is_within(Path(found).resolve(), hidden):
            return found
    return ""


PY = visible_python() if sys.platform.startswith("linux") else ""


def host_problem() -> str:
    """Return why this host cannot run real sandboxes, or "" when it can."""
    if not sys.platform.startswith("linux"):
        return f"the sandbox needs Linux, not {sys.platform}"
    missing = [tool for tool in TOOLS if shutil.which(tool) is None]
    if missing:
        return f"not on PATH: {', '.join(missing)}"
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if not runtime or not (Path(runtime) / "bus").exists():
        return "no user systemd manager: $XDG_RUNTIME_DIR/bus does not exist"
    if not os.environ.get("LASSI_SCRATCH"):
        return "LASSI_SCRATCH is not set"
    if not PY:
        return "no python3 on PATH outside $HOME and $LASSI_SCRATCH, which the sandbox hides"
    if shutil.which("python3", path=SANDBOX_PATH) is None:
        return f"no python3 in the sandbox's PATH {SANDBOX_PATH}"
    return ""


PROBLEM = host_problem()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can run sandboxes"),
]


@pytest.fixture(autouse=True)
def host_ready() -> None:
    """Fail, rather than skip, when LASSI_REQUIRE_SANDBOX=1 asks for real sandboxes and the host cannot run them."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")


def need(condition: bool, why: str) -> None:
    """Skip the test when `condition` is false, or fail it under LASSI_REQUIRE_SANDBOX=1."""
    if not condition:
        if REQUIRE:
            pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {why}")
        pytest.skip(why)


@pytest.fixture
def sandbox() -> ModuleType:
    """Return the lassi.executors.sandbox module."""
    from lassi.executors import sandbox

    return sandbox


@pytest.fixture
def native() -> ModuleType:
    """Return the lassi.executors.native module."""
    from lassi.executors import native

    return native


@dataclass(frozen=True)
class Layout:
    """One test's directories: all live under `base`, a fresh directory under $LASSI_SCRATCH/tmp."""

    base: Path
    workdir: Path
    harness: Path
    hidden: tuple[Path, ...]


def toolchains_root() -> Path | None:
    """Return $LASSI_TOOLCHAINS as a Path, or None when it is unset."""
    value = os.environ.get("LASSI_TOOLCHAINS", "")
    return Path(value) if value else None


@pytest.fixture
def layout() -> Iterator[Layout]:
    """Create a fresh build directory and a harness directory under $LASSI_SCRATCH/tmp; remove both afterwards."""
    parent = Path(os.environ["LASSI_SCRATCH"]) / "tmp"
    parent.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="lassi-sandbox-test.", dir=parent))
    try:
        workdir = base / "build"
        workdir.mkdir()
        harness = base / "harness"
        harness.mkdir()
        (harness / "lassi_io.h").write_text("// harness placeholder\n", encoding="ascii")
        yield Layout(base=base, workdir=workdir, harness=harness, hidden=hidden_default())
    finally:
        shutil.rmtree(base)


@pytest.fixture
def runs_base() -> Iterator[Path]:
    """Create a fresh directory under $LASSI_RUNS_ROOT, where a test lays out trials side by side; remove it after."""
    runs = os.environ.get("LASSI_RUNS_ROOT", "")
    need(bool(runs) and Path(runs).is_dir(), "LASSI_RUNS_ROOT is not set to a directory")
    base = Path(tempfile.mkdtemp(prefix="lassi-sandbox-test.", dir=runs))
    try:
        yield base
    finally:
        shutil.rmtree(base)


def spec_for(sandbox: ModuleType, layout: Layout, **overrides: object) -> object:
    """Return the SandboxSpec these tests use: the layout's workdir and harness, the hidden roots, and toolchains."""
    values: dict[str, object] = {
        "workdir": layout.workdir,
        "hidden_roots": layout.hidden,
        "harness": layout.harness,
        "toolchains": toolchains_root(),
    }
    values.update(overrides)
    return sandbox.SandboxSpec(**values)


def run_in(sandbox: ModuleType, layout: Layout, argv: Sequence[str], limits: Limits, **overrides: object) -> object:
    """Run `argv` through Sandbox.run with spec_for(layout) and any spec overrides."""
    return sandbox.Sandbox().run(spec_for(sandbox, layout, **overrides), list(argv), limits)


def script_of(command: list[str]) -> str:
    """Return the script element of a sandbox command: the element after "sh", "-c"."""
    start = command.index("sh")
    return command[start + 2]


def host_can_connect() -> bool:
    """Return True when the host itself reaches 1.1.1.1:443, so a blocked connect inside means something."""
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=5):
            return True
    except OSError:
        return False


CONNECT = """
import errno, socket, sys
try:
    socket.create_connection(("1.1.1.1", 443), timeout=5)
except OSError as exc:
    print("BLOCKED" if exc.errno == errno.ENETUNREACH else "OTHER_ERROR", exc)
    sys.exit(0)
print("CONNECTED")
sys.exit(7)
"""

WRITES = """
cat "$1/lassi_io.h" > /dev/null && echo HARNESS_READABLE
if touch "$1/written" 2> /dev/null; then echo HARNESS_WROTE; else echo HARNESS_READONLY; fi
if touch "$2/written" 2> /dev/null; then echo ROOT_WROTE; else echo ROOT_READONLY; fi
echo made > "$3/made.txt" && echo WORKDIR_WROTE
"""

# $1 a hidden root's mount point, $2 the harness, $3 a directory under the root.
PRIVILEGE = """
grep -E '^(CapInh|CapPrm|CapEff|CapBnd|CapAmb|NoNewPrivs):' /proc/self/status | tr -d ' \\t'
if mount -o remount,bind,rw "$1" 2> /dev/null; then echo ROOT_REMOUNTED; else echo ROOT_REMOUNT_REFUSED; fi
if umount "$2" 2> /dev/null; then echo HARNESS_UNMOUNTED; else echo HARNESS_UMOUNT_REFUSED; fi
if umount /tmp 2> /dev/null; then echo TMP_UNMOUNTED; else echo TMP_UMOUNT_REFUSED; fi
if umount -l "$1" 2> /dev/null; then echo ROOT_UNMOUNTED; else echo ROOT_UMOUNT_REFUSED; fi
nested='mount -o remount,bind,rw "$1" && touch "$2/nested"'
if unshare -rm sh -c "$nested" sh "$1" "$3" 2> /dev/null; then echo NESTED_WROTE; else echo NESTED_REFUSED; fi
if touch "$3/written" 2> /dev/null; then echo ROOT_WROTE; else echo ROOT_READONLY; fi
if touch "$2/written" 2> /dev/null; then echo HARNESS_WROTE; else echo HARNESS_READONLY; fi
"""

# Prints, for the top mount on each point named in argv: read-only or not, filesystem type, superblock options.
MOUNTS = """
import sys
top = {}
with open("/proc/self/mountinfo") as mountinfo:
    for line in mountinfo:
        fields = line.split()
        tail = fields[fields.index("-") + 1 :]
        top[fields[4]] = (fields[5].split(","), tail[0], tail[2] if len(tail) > 2 else "")
for point in sys.argv[1:]:
    options, fstype, superblock = top.get(point, ([], "missing", ""))
    print(point, "ro" if "ro" in options else "rw", fstype, superblock)
"""

# Rewrites the run's own memory.max with the value it already has; any successful write would be a hole.
CGROUP = """
path = open("/proc/self/cgroup").read().strip().split("::", 1)[1]
limit = "/sys/fs/cgroup" + path + "/memory.max"
value = open(limit).read()
try:
    with open(limit, "w") as handle:
        handle.write(value)
    print("CGROUP_WROTE")
except OSError as exc:
    print("CGROUP_REFUSED", exc.errno)
"""

# Tries each path named in argv as a unix socket: it must be absent inside, and a connect must fail.
SOCKETS = """
import os, socket, sys
for path in sys.argv[1:]:
    seen = "VISIBLE" if os.path.exists(path) else "HIDDEN"
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(path)
        print(seen, "CONNECTED")
    except OSError:
        print(seen, "REFUSED")
"""

# Burns CPU time on four threads at once (hashlib releases the GIL on large buffers) until a limit stops it.
CPU_BURN = """
import hashlib, threading
data = bytes(1 << 20)
def burn() -> None:
    while True:
        hashlib.sha256(data).digest()
threads = [threading.Thread(target=burn, daemon=True) for _ in range(4)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
"""

PROCESSES = "import os; print(sum(name.isdigit() for name in os.listdir('/proc')))"

# R1: every entry under /dev, without following links, as [kind, ...], plus the device of /dev/pts and
# /dev/shm and a read of /dev/zero after a write to /dev/null.
DEV_REPORT = """
import json, os, stat
entries = {}
for dirpath, dirnames, filenames in os.walk("/dev"):
    for name in dirnames + filenames:
        path = os.path.join(dirpath, name)
        info = os.lstat(path)
        key = os.path.relpath(path, "/dev")
        if stat.S_ISLNK(info.st_mode):
            entries[key] = ["link", os.readlink(path)]
        elif stat.S_ISCHR(info.st_mode) or stat.S_ISBLK(info.st_mode):
            kind = "chr" if stat.S_ISCHR(info.st_mode) else "blk"
            entries[key] = [kind, os.major(info.st_rdev), os.minor(info.st_rdev)]
        elif stat.S_ISDIR(info.st_mode):
            entries[key] = ["dir"]
        else:
            entries[key] = ["other", stat.S_IFMT(info.st_mode)]
with open("/dev/null", "wb") as sink:
    sink.write(b"x")
with open("/dev/zero", "rb") as source:
    zero = source.read(4).hex()
report = {"entries": entries, "zero": zero}
report["pts_dev"] = os.stat("/dev/pts").st_dev
report["shm_dev"] = os.stat("/dev/shm").st_dev
print(json.dumps(report))
"""

MOUNTINFO = "import sys; sys.stdout.write(open('/proc/self/mountinfo').read())"

# Prints {pid: environment text or "unreadable"} for every process the program can see in /proc.
ENVIRONS = """
import json, os
report = {}
for pid in sorted(name for name in os.listdir("/proc") if name.isdigit()):
    try:
        with open("/proc/%s/environ" % pid, "rb") as handle:
            report[pid] = handle.read().decode("utf-8", "replace")
    except OSError:
        report[pid] = "unreadable"
print(json.dumps(report))
"""

# Reports what the confinement leaves the program: its session, terminal, core dump filter, limits, and which
# refused calls fail and how (Python raises ValueError for a refused setrlimit).
CONFINED = """from __future__ import annotations
import ctypes, errno, json, os, resource, socket
from collections.abc import Callable
libc = ctypes.CDLL(None, use_errno=True)
report = {}
def attempt(name: str, action: Callable[[], object]) -> None:
    try:
        action()
        report[name] = "ok"
    except ValueError:
        report[name] = "ValueError"
    except OSError as exc:
        report[name] = errno.errorcode.get(exc.errno, str(exc.errno))
nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
attempt("setrlimit_core", lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
attempt("setrlimit_nofile", lambda: resource.setrlimit(resource.RLIMIT_NOFILE, nofile))
attempt("prlimit_core", lambda: resource.prlimit(0, resource.RLIMIT_CORE, (0, 0)))
attempt("prlimit_other", lambda: resource.prlimit(1, resource.RLIMIT_NOFILE, nofile))
attempt("vsock", lambda: socket.socket(40, socket.SOCK_STREAM).close())
attempt("inet", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).close())
attempt("tty", lambda: open("/dev/tty").close())
params = ctypes.create_string_buffer(120)
report["io_uring"] = "ok" if libc.syscall(425, 1, params) >= 0 else errno.errorcode.get(ctypes.get_errno(), "")
# The keyring calls: the id of the host user's keyring (@u, -4), a key added to the program's own session
# keyring (-3, which dies with it, so a broken filter changes nothing on the host), and a lookup that asks for no
# upcall. Each must fail with EPERM.
calls = {
    "keyctl_user": (250, 0, -4, 0),
    "add_key": (248, b"user", b"lassi-sandbox-probe", b"x", 1, -3),
    "request_key": (249, b"user", b"lassi-sandbox-probe", None, 0),
}
for name, args in calls.items():
    report[name] = "ok" if libc.syscall(*args) >= 0 else errno.errorcode.get(ctypes.get_errno(), "")
report["sid"], report["pid"] = os.getsid(0), os.getpid()
report["core_filter"] = open("/proc/self/coredump_filter").read().strip()
report["rlimit_core"] = list(resource.getrlimit(resource.RLIMIT_CORE))
report["rlimit_fsize"] = list(resource.getrlimit(resource.RLIMIT_FSIZE))
print(json.dumps(report))
"""

# Run by the host-side KEY_PROBE inside the sandbox: tries to read the host key by its serial (argv[1]) and to
# read the keyrings it could reach it through: its session keyring (-3), the host user's keyring (-4), and the
# user session keyring (-5) (keyctl KEYCTL_READ, syscall 250 command 11). Prints the key's bytes or the errno of
# each.
READ_KEY = """
import ctypes, errno, json, sys
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
buffer = ctypes.create_string_buffer(4096)
report = {}
for name, key in (("key", int(sys.argv[1])), ("session", -3), ("user", -4), ("user_session", -5)):
    size = libc.syscall(250, 11, key, buffer, 4096)
    report[name] = buffer.raw[:size].hex() if size >= 0 else errno.errorcode.get(ctypes.get_errno(), "")
print(json.dumps(report))
"""

# Runs in a child process on the host: joins a new session keyring of its own (so nothing outlives it), adds a
# user key with the payload argv[3], and runs READ_KEY in the sandbox with argv[4] as python3.
KEY_PROBE = """
import ctypes, json, sys
from pathlib import Path
from lassi.core.interfaces import Limits
from lassi.executors.sandbox import Sandbox, SandboxSpec
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
if libc.syscall(250, 1, None) < 0:
    sys.exit("keyctl join: errno %d" % ctypes.get_errno())
payload = sys.argv[3].encode("ascii")
serial = libc.syscall(248, b"user", b"lassi-sandbox-test", payload, len(payload), -3)
if serial < 0:
    sys.exit("add_key: errno %d" % ctypes.get_errno())
spec = SandboxSpec(workdir=Path(sys.argv[1]), hidden_roots=tuple(Path(p) for p in json.loads(sys.argv[2])))
result = Sandbox().run(spec, [sys.argv[4], "-c", sys.argv[5], str(serial)], Limits(wall_s=20.0, memory_mb=256, cpus=1))
print(json.dumps({"serial": serial, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}))
"""

# Lists /sys/class, /sys/bus, the entries of /sys/devices that are not empty, and each /var entry's contents.
SYSTEM_VIEW = """
import json, os
devices = sorted(os.listdir("/sys/devices"))
report = {"class": sorted(os.listdir("/sys/class")), "bus": sorted(os.listdir("/sys/bus"))}
report["devices_not_empty"] = [name for name in devices if name != "system" and os.listdir("/sys/devices/" + name)]
report["cpu_online"] = open("/sys/devices/system/cpu/online").read().strip()
report["var"] = {}
for name in sorted(os.listdir("/var")):
    path = "/var/" + name
    if not os.path.islink(path) and os.path.isdir(path):
        report["var"][name] = sorted(os.listdir(path))
print(json.dumps(report))
"""

# argv[1] is a JSON list of host socket paths; prints, for each, whether it exists inside and is writable
# (what connect needs). Nothing connects.
SOCKET_VIEW = """
import json, os, sys
report = {}
for path in json.loads(sys.argv[1]):
    if not os.path.lexists(path):
        report[path] = "hidden"
    else:
        report[path] = "writable" if os.access(path, os.W_OK) else "visible"
print(json.dumps(report))
"""

# R3: argv[1] is JSON {"list": {key: path}, "read": {key: path}, "readonly": {key: path}}; prints what the
# program can list, read, and whether each path's mount is read-only, then writes made.txt in its workdir.
VIEW = """#!{python}
from __future__ import annotations
import errno, json, os, sys
def failure(exc: OSError) -> str:
    return "error:" + errno.errorcode.get(exc.errno, str(exc.errno))
def listing(path: str) -> list[str] | str:
    try:
        return sorted(os.listdir(path))
    except OSError as exc:
        return failure(exc)
def read(path: str) -> str:
    try:
        with open(path, encoding="ascii") as handle:
            return handle.read()
    except OSError as exc:
        return failure(exc)
paths = json.loads(sys.argv[1])
report = {{"list": {{key: listing(path) for key, path in paths["list"].items()}}}}
report["read"] = {{key: read(path) for key, path in paths["read"].items()}}
def readonly(path: str) -> bool:
    return bool(os.statvfs(path).f_flag & os.ST_RDONLY)
report["readonly"] = {{key: readonly(path) for key, path in paths["readonly"].items()}}
with open("made.txt", "w", encoding="ascii") as handle:
    handle.write("made\\n")
print(json.dumps(report))
"""

# R5: writes 256 KiB blocks to big.bin in the workdir until twice argv[1] bytes or an error; prints what happened.
# SIGXFSZ is ignored, so a write past the file size limit fails with EFBIG rather than killing the program. With
# argv[2] "unlink", big.bin is removed afterwards, so nothing large reaches the host.
FILL = """
import errno, json, os, signal, sys
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
cap = int(sys.argv[1])
block = b"x" * 262144
written, error = 0, None
descriptor = os.open("big.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
try:
    while written < 2 * cap:
        written += os.write(descriptor, block)
except OSError as exc:
    error = errno.errorcode.get(exc.errno, str(exc.errno))
finally:
    os.close(descriptor)
if sys.argv[2:] == ["unlink"]:
    os.unlink("big.bin")
print(json.dumps({"written": written, "error": error}))
"""

# R4: runs Sandbox.run in this child process on a program that floods stdout and stderr, and prints the
# child's peak resident set size after a warm run with tiny output and after the flood, with what came back.
MEMORY_PROBE = """
import json, resource, sys
from pathlib import Path
from lassi.core.interfaces import Limits
from lassi.executors.sandbox import Sandbox, SandboxSpec
spec = SandboxSpec(workdir=Path(sys.argv[1]), hidden_roots=tuple(Path(p) for p in json.loads(sys.argv[2])))
Sandbox().run(spec, ["sh", "-c", "echo warm; echo warm >&2"], Limits(wall_s=60.0, memory_mb=64, cpus=1))
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
result = Sandbox().run(spec, ["sh", "-c", sys.argv[3]], Limits(wall_s=60.0, memory_mb=64, cpus=1))
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({
    "before_kib": before, "after_kib": after, "returncode": result.returncode,
    "stdout_bytes": len(result.stdout.encode("utf-8")), "stderr_bytes": len(result.stderr.encode("utf-8")),
    "stdout_truncated": result.stdout_truncated, "stderr_truncated": result.stderr_truncated,
}))
"""


# ---------------------------------------------------------------------------
# P0.10: isolation, limits, and the program's view (kept as they were, with hidden roots and a visible python3)


def test_network_connect_fails_inside(sandbox: ModuleType, layout: Layout) -> None:
    if not host_can_connect():
        pytest.skip("the host itself cannot reach 1.1.1.1:443, so a blocked connect inside would prove nothing")
    result = run_in(sandbox, layout, [PY, "-c", CONNECT], PYTHON)
    assert result.stdout.split()[:1] == ["BLOCKED"], result
    assert result.returncode == 0, result


def test_memory_hog_is_killed(sandbox: ModuleType, layout: Layout) -> None:
    hog = "b = bytearray(256 * 1024 * 1024)\nprint('ALLOCATED', len(b))\n"
    result = run_in(sandbox, layout, [PY, "-c", hog], Limits(wall_s=20.0, memory_mb=64, cpus=1))
    assert "ALLOCATED" not in result.stdout, result
    assert (result.returncode, result.killed, result.hang) == (137, True, False), result


def test_cpu_time_hog_is_killed_before_the_wall_limit(sandbox: ModuleType, layout: Layout) -> None:
    # CPU budget 20 s under a 20 s wall limit; four busy threads spend it in about 5 s of wall time.
    result = run_in(sandbox, layout, [PY, "-c", CPU_BURN], PYTHON)
    assert (result.returncode, result.killed, result.hang) == (137, True, False), result
    assert result.wall_s < 20, result


def test_sleep_past_wall_time_sets_hang(sandbox: ModuleType, layout: Layout) -> None:
    # cpus=2 makes the CPU budget (6 s) differ from the wall limit (3 s), so a swap of the two shows.
    start = time.monotonic()
    result = run_in(sandbox, layout, ["sleep", "30"], Limits(wall_s=3.0, memory_mb=64, cpus=2))
    elapsed = time.monotonic() - start
    assert elapsed < 10, (elapsed, result)
    assert (result.returncode, result.hang, result.killed) == (124, True, False), result
    assert 3.0 <= result.wall_s < 3.0 + sandbox.KILL_AFTER_S, result
    # The program's own time covers the innermost timeout's run, less at most one 10 ms clock step, and lies
    # within the whole command's.
    assert 3.0 - 0.01 <= result.program_s <= result.wall_s, result


def test_a_program_ignoring_sigterm_is_killed_after_the_grace(sandbox: ModuleType, layout: Layout) -> None:
    argv = ["sh", "-c", 'trap "" TERM; sleep 30']
    result = run_in(sandbox, layout, argv, Limits(wall_s=3.0, memory_mb=64, cpus=1))
    assert (result.returncode, result.hang, result.killed) == (124, True, False), result
    assert 3.0 + sandbox.KILL_AFTER_S <= result.wall_s < 9.0, result
    assert 3.0 + sandbox.KILL_AFTER_S - 0.01 <= result.program_s <= result.wall_s, result


def test_a_program_within_the_wall_limit_runs_to_its_end(sandbox: ModuleType, layout: Layout) -> None:
    result = run_in(sandbox, layout, ["sleep", "4"], Limits(wall_s=6.0, memory_mb=64, cpus=1))
    assert (result.returncode, result.hang, result.killed) == (0, False, False), result
    assert 4.0 <= result.wall_s < 6.0, result
    assert 4.0 - 0.01 <= result.program_s <= result.wall_s, result


def test_a_program_exiting_124_itself_before_the_limit_is_no_hang(sandbox: ModuleType, layout: Layout) -> None:
    # Round-3 finding RT1-2: the hang test reads the program's own time, not the whole command's.
    result = run_in(sandbox, layout, ["sh", "-c", "sleep 1; exit 124"], Limits(wall_s=3.0, memory_mb=64, cpus=1))
    assert (result.returncode, result.hang, result.killed) == (124, False, False), result
    assert 1.0 - 0.01 <= result.program_s < 3.0 and result.program_s <= result.wall_s, result


def test_harness_and_hidden_roots_refuse_writes_and_the_workdir_takes_them(
    sandbox: ModuleType, layout: Layout
) -> None:
    argv = ["sh", "-c", WRITES, "sh", str(layout.harness), str(layout.base), str(layout.workdir)]
    result = run_in(sandbox, layout, argv, SHORT)
    assert result.returncode == 0, result
    assert result.stdout.split() == ["HARNESS_READABLE", "HARNESS_READONLY", "ROOT_READONLY", "WORKDIR_WROTE"], result
    assert not (layout.harness / "written").exists()
    assert not (layout.base / "written").exists()
    assert (layout.workdir / "made.txt").read_text(encoding="ascii") == "made\n"


def test_the_harness_mount_alone_is_read_only(sandbox: ModuleType, layout: Layout) -> None:
    other_root = layout.base / "other-root"
    other_root.mkdir()
    script = 'if touch "$1/written" 2> /dev/null; then echo HARNESS_WROTE; else echo HARNESS_READONLY; fi'
    argv = ["sh", "-c", script, "sh", str(layout.harness)]
    result = run_in(sandbox, layout, argv, SHORT, hidden_roots=(other_root,))
    assert result.returncode == 0, result
    assert result.stdout.split() == ["HARNESS_READONLY"], result
    assert not (layout.harness / "written").exists()


def test_every_hidden_root_refuses_writes_not_only_the_first(sandbox: ModuleType, layout: Layout) -> None:
    roots = (layout.base / "ro-a", layout.base / "ro-b")
    for root in roots:
        root.mkdir()
    script = 'for d in "$@"; do if touch "$d/written" 2> /dev/null; then echo WROTE; else echo READONLY; fi; done'
    argv = ["sh", "-c", script, "sh", *map(str, roots)]
    result = run_in(sandbox, layout, argv, SHORT, hidden_roots=roots)
    assert result.returncode == 0, result
    assert result.stdout.split() == ["READONLY", "READONLY"], result
    assert not any((root / "written").exists() for root in roots)


def test_the_program_holds_no_capability_and_cannot_undo_a_mount(sandbox: ModuleType, layout: Layout) -> None:
    root = layout.hidden[0]
    argv = ["sh", "-c", PRIVILEGE, "sh", str(root), str(layout.harness), str(layout.base)]
    result = run_in(sandbox, layout, argv, SHORT)
    lines = result.stdout.split()
    status = dict(line.split(":", 1) for line in lines if ":" in line)
    assert status == {
        "CapInh": "0000000000000000",
        "CapPrm": "0000000000000000",
        "CapEff": "0000000000000000",
        "CapBnd": "0000000000000000",
        "CapAmb": "0000000000000000",
        "NoNewPrivs": "1",
    }, result
    assert [line for line in lines if ":" not in line] == [
        "ROOT_REMOUNT_REFUSED",
        "HARNESS_UMOUNT_REFUSED",
        "TMP_UMOUNT_REFUSED",
        "ROOT_UMOUNT_REFUSED",
        "NESTED_REFUSED",
        "ROOT_READONLY",
        "HARNESS_READONLY",
    ], result
    assert not any((path / name).exists() for path in (layout.base, layout.harness) for name in ("written", "nested"))


def test_the_program_cannot_create_a_nested_user_namespace(sandbox: ModuleType, layout: Layout) -> None:
    limit = Path("/proc/sys/user/max_user_namespaces")
    host_before = limit.read_text(encoding="ascii").strip()
    refused = run_in(sandbox, layout, ["unshare", "-r", "true"], SHORT)
    assert refused.returncode != 0, refused
    assert (refused.hang, refused.killed) == (False, False), refused
    script = "cat /proc/sys/user/max_user_namespaces; echo STILL_RUNS"
    works = run_in(sandbox, layout, ["sh", "-c", script], SHORT)
    assert (works.returncode, works.stdout.split()) == (0, ["0", "STILL_RUNS"]), works
    # The write set only the sandbox user namespace's limit; the host's is untouched.
    host_after = limit.read_text(encoding="ascii").strip()
    assert host_after == host_before != "0", (host_before, host_after)


def test_the_program_runs_at_the_lowest_cpu_priority(sandbox: ModuleType, layout: Layout) -> None:
    # `nice` prints its own niceness; field 19 of /proc/self/stat is the niceness of `cut`, a later child.
    result = run_in(sandbox, layout, ["sh", "-c", "nice; cut -d ' ' -f 19 /proc/self/stat"], SHORT)
    assert (result.returncode, result.stdout.split()) == (0, ["19", "19"]), result


def test_the_root_filesystem_and_cgroups_are_read_only_and_run_is_private(sandbox: ModuleType, layout: Layout) -> None:
    points = ["/", "/sys/fs/cgroup", "/run", "/tmp"]
    result = run_in(sandbox, layout, [PY, "-c", MOUNTS, *points], PYTHON)
    assert result.returncode == 0, result
    seen = {line.split()[0]: line.split()[1:] for line in result.stdout.splitlines()}
    assert seen["/"][0] == "ro", result
    assert seen["/sys/fs/cgroup"][0] == "ro", result
    for private in ("/run", "/tmp"):
        # The host's own /run is a tmpfs too, so the sandbox's private mount is told apart by its 64 MiB size.
        # The kernel's shmem_show_options prints mode only when it differs from the tmpfs default 1777, and
        # prints size=<n>k for a non-default size; seen in exploratory run rx
        # 20260923-071759-desktop-8r113ei-p0-core-e643 (dirty snapshot).
        assert seen[private][:2] == ["rw", "tmpfs"], result
        assert "size=65536k" in seen[private][2].split(","), result


def test_the_run_cannot_rewrite_its_own_cgroup_limits(sandbox: ModuleType, layout: Layout) -> None:
    result = run_in(sandbox, layout, [PY, "-c", CGROUP], PYTHON)
    assert result.returncode == 0, result
    assert result.stdout.split()[:1] == ["CGROUP_REFUSED"], result


def test_the_bus_sockets_are_hidden_inside(sandbox: ModuleType, layout: Layout) -> None:
    runtime = Path(os.environ["XDG_RUNTIME_DIR"])
    sockets = [str(runtime / "bus"), str(runtime / "systemd" / "private"), "/run/dbus/system_bus_socket"]
    argv = [PY, "-c", SOCKETS, *sockets]
    result = run_in(sandbox, layout, argv, PYTHON)
    assert result.returncode == 0, result
    assert result.stdout.splitlines() == ["HIDDEN REFUSED"] * len(sockets), result


def test_the_program_sees_only_the_allowed_environment(
    sandbox: ModuleType, layout: Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = f"lassi-sandbox-test-secret-{uuid.uuid4().hex}"
    monkeypatch.setenv("LASSI_SANDBOX_TEST_SECRET", secret)
    result = run_in(sandbox, layout, ["env"], SHORT)
    assert result.returncode == 0, result
    assert secret not in result.stdout, result
    names = sorted(line.split("=", 1)[0] for line in result.stdout.splitlines())
    assert names == ["HOME", "LANG", "PATH", "TMPDIR"], result
    assert f"HOME={layout.workdir}" in result.stdout.splitlines(), result
    assert "TMPDIR=/tmp" in result.stdout.splitlines(), result
    assert f"PATH={SANDBOX_PATH}" in result.stdout.splitlines(), result
    # Agent Rule 12 for every process inside, the setup shell (pid 1) and the nested unshare included: env -i
    # starts the whole sandbox, so no environment a program can read holds the secret.
    scan = run_in(sandbox, layout, [PY, "-c", ENVIRONS], PYTHON)
    assert scan.returncode == 0, scan
    report = json.loads(scan.stdout)
    assert report and secret not in scan.stdout, report
    assert report["1"] == "unreadable" or secret not in report["1"], report


@pytest.mark.parametrize("private", ["/tmp", "/var/tmp", "/dev/shm"])
def test_tmp_writes_stay_off_the_host(sandbox: ModuleType, layout: Layout, private: str) -> None:
    if not Path(private).is_dir():
        pytest.skip(f"the host has no {private}")
    marker = f"lassi-sandbox-test-{uuid.uuid4().hex}"
    host_file = Path(private) / marker
    script = 'echo inside > "$1/$2" && cat "$1/$2" && echo TMP_WROTE'
    try:
        result = run_in(sandbox, layout, ["sh", "-c", script, "sh", private, marker], SHORT)
        leaked = host_file.exists()
    finally:
        if host_file.exists():
            host_file.unlink()
    assert result.returncode == 0, result
    assert result.stdout.split() == ["inside", "TMP_WROTE"], result
    assert not leaked


def test_a_failed_setup_raises_and_never_runs_the_program(sandbox: ModuleType, layout: Layout) -> None:
    spec = spec_for(sandbox, layout, hidden_roots=(layout.base / "missing",), harness=None)
    argv = ["sh", "-c", 'touch "$1/ran"', "sh", str(layout.workdir)]
    with pytest.raises(sandbox.SandboxUnavailableError, match="did not run"):
        sandbox.Sandbox().run(spec, argv, SHORT)
    assert not (layout.workdir / "ran").exists()


def test_the_pid_namespace_hides_host_processes(sandbox: ModuleType, layout: Layout) -> None:
    result = run_in(sandbox, layout, [PY, "-c", PROCESSES], PYTHON)
    assert result.returncode == 0, result
    assert int(result.stdout) < 10, result


def test_exit_status_passes_through(sandbox: ModuleType, layout: Layout) -> None:
    result = run_in(sandbox, layout, ["sh", "-c", "echo OK; echo ERR >&2; exit 3"], SHORT)
    assert (result.returncode, result.hang, result.killed) == (3, False, False), result
    assert result.stdout == "OK\n", result
    assert "ERR" in result.stderr, result
    assert sandbox.READY_MARKER not in result.stderr, result
    assert sandbox.DONE_MARKER not in result.stderr, result


def test_native_executor_runs_a_shell_script_artifact(
    native: ModuleType, layout: Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(layout.base))
    artifact = layout.workdir / "main"
    artifact.write_text('#!/bin/sh\necho "args:$*"\necho result > out.txt\nexit 5\n', encoding="ascii")
    artifact.chmod(0o755)
    result = native.NativeExecutor(harness=str(layout.harness)).run(artifact, ["a", "b c"], SHORT)
    assert (result.exit_code, result.hang) == (5, False), result
    assert result.stdout == "args:a b c\n", result
    assert dict(result.output_files) == {"out.txt": layout.workdir.resolve() / "out.txt"}
    assert (layout.workdir / "out.txt").read_text(encoding="ascii") == "result\n"
    assert result.wall_s > 0


# ---------------------------------------------------------------------------
# P0.16 R1: private /dev


def test_r1_dev_holds_only_the_allowed_nodes_a_private_devpts_and_a_private_shm(
    sandbox: ModuleType, layout: Layout
) -> None:
    result = run_in(sandbox, layout, [PY, "-c", DEV_REPORT], PYTHON)
    assert result.returncode == 0, result
    report = json.loads(result.stdout)
    entries: dict[str, list[object]] = report["entries"]
    forbidden = [name for name in entries if any(fnmatch.fnmatch(name.split("/")[0], p) for p in FORBIDDEN_DEV)]
    assert forbidden == [], forbidden
    nodes = {name: kind for name, kind in entries.items() if kind[0] in ("chr", "blk")}
    assert set(nodes) == DEV_NODES, sorted(entries)
    for name, kind in nodes.items():
        host = os.stat(f"/dev/{name}")
        assert kind == ["chr", os.major(host.st_rdev), os.minor(host.st_rdev)], (name, kind)
    assert {name for name, kind in entries.items() if kind[0] == "dir"} == DEV_DIRS, sorted(entries)
    links = {name: kind[1] for name, kind in entries.items() if kind[0] == "link"}
    assert set(links) <= set(DEV_LINKS) and all(DEV_LINKS[name] == target for name, target in links.items()), links
    assert set(entries) == set(nodes) | DEV_DIRS | set(links), sorted(entries)
    # A new devpts instance and a new tmpfs, not the host's.
    assert report["pts_dev"] != os.stat("/dev/pts").st_dev, report
    assert report["shm_dev"] != os.stat("/dev/shm").st_dev, report
    assert report["zero"] == "00000000", report


# ---------------------------------------------------------------------------
# P0.16 R2: read-only mounts, fail closed


@dataclass(frozen=True)
class Mount:
    """One /proc/self/mountinfo line: its device (major:minor), mount point, per-mount options, type, source."""

    device: str
    point: str
    options: tuple[str, ...]
    fstype: str
    source: str


def parse_mountinfo(text: str) -> list[Mount]:
    """Return the mounts in a mountinfo text, with octal escapes in mount points decoded."""
    mounts = []
    for line in text.splitlines():
        fields = line.split()
        tail = fields[fields.index("-", 6) + 1 :]
        point = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), fields[4])
        mounts.append(Mount(fields[2], point, tuple(fields[5].split(",")), tail[0], tail[1] if len(tail) > 1 else ""))
    return mounts


def test_r2_every_host_filesystem_is_read_only_and_only_the_setups_own_mounts_are_writable(
    sandbox: ModuleType, layout: Layout
) -> None:
    host_devices = {mount.device for mount in parse_mountinfo(Path("/proc/self/mountinfo").read_text())}
    spec = spec_for(sandbox, layout)
    result = sandbox.Sandbox().run(spec, [PY, "-c", MOUNTINFO], PYTHON)
    assert result.returncode == 0, result
    inside = parse_mountinfo(result.stdout)
    host_inside = [mount for mount in inside if mount.device in host_devices]
    assert host_inside, "no host filesystem is mounted inside at all, so the check would prove nothing"
    assert [m for m in host_inside if "ro" not in m.options] == [], [m for m in host_inside if "rw" in m.options]
    allowed = {str(spec.workdir), *map(str, spec.hidden_roots), *PRIVATE_POINTS}
    writable = [mount for mount in inside if "rw" in mount.options]
    strays = [
        mount
        for mount in writable
        if mount.fstype not in SETUP_FSTYPES
        or not (mount.point in allowed or mount.point.startswith(("/tmp/", "/var/tmp/", "/run/")))
    ]
    assert strays == [], strays
    assert any(mount.point == str(spec.workdir) and mount.fstype == "overlay" for mount in writable), writable
    assert any(mount.point == "/" and "ro" in mount.options for mount in inside), inside


def test_r2_a_writable_host_mount_makes_setup_fail_closed_and_the_program_never_runs(
    sandbox: ModuleType, layout: Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    need(Path("/var/tmp").is_dir(), "the host has no /var/tmp to put the writable host mount on")
    spec = spec_for(sandbox, layout)
    # A unique name, removed if it ever reaches the host's /var/tmp (the root filesystem, Agent Rule 7).
    marker = f"lassi-sandbox-test-{uuid.uuid4().hex}"
    host_file = Path("/var/tmp") / marker
    program = ["sh", "-c", 'touch "/var/tmp/$1"; echo RAN', "sh", marker]
    try:
        control = sandbox.Sandbox().run(spec, program, SHORT)
        leaked = host_file.exists()
    finally:
        if host_file.exists():
            host_file.unlink()
    assert (control.returncode, control.stdout) == (0, "RAN\n"), control
    assert not leaked
    # Negative control: right before the mountinfo check, bind the host harness directory on /var/tmp and make
    # it writable again, as a setup bug would. The check must stop the setup before the ready marker.
    script = sandbox.SETUP_SCRIPT
    anchor = f"awk '{sandbox.MOUNT_CHECK}' /proc/self/mountinfo"
    assert script.count(anchor) == 1 and "'" not in str(layout.harness)
    line_start = script.rfind("\n", 0, script.index(anchor)) + 1
    bait = f"mount --bind '{layout.harness}' /var/tmp\nmount -o remount,bind,rw /var/tmp\n"
    patched = script[:line_start] + bait + script[line_start:]
    monkeypatch.setattr(sandbox, "SETUP_SCRIPT", patched)
    assert script_of(sandbox.sandbox_command(spec, program, SHORT)) == patched
    with pytest.raises(sandbox.SandboxUnavailableError, match="/var/tmp"):
        sandbox.Sandbox().run(spec, program, SHORT)
    # Had the program run, its touch would have gone through the writable bind into the host harness.
    assert not (layout.harness / marker).exists()


# ---------------------------------------------------------------------------
# P0.16 R3: default-deny view


def expected_listing(directory: Path, exposures: Sequence[Path], hidden: Sequence[Path]) -> object:
    """Return what the program should list in `directory`: the skeleton of paths to the exposures under it.

    A hidden root with no exposure under it lists as empty; any other
    directory under a hidden root with no exposure under it does not exist.
    """
    under = [exposure for exposure in exposures if directory in exposure.parents]
    names = sorted({exposure.relative_to(directory).parts[0] for exposure in under})
    if names or directory in hidden:
        return names
    return "error:ENOENT"


def test_r3_home_and_scratch_are_hidden_except_the_workdir_harness_and_toolchains(
    native: ModuleType, sandbox: ModuleType, layout: Layout, runs_base: Path
) -> None:
    toolchains = toolchains_root()
    need(toolchains is not None and toolchains.is_dir(), "LASSI_TOOLCHAINS is not set to a directory")
    assert toolchains is not None
    runs = Path(os.environ["LASSI_RUNS_ROOT"])
    workdir = runs_base / "trial-a" / "attempt00" / "build"
    workdir.mkdir(parents=True)
    other = runs_base / "trial-b"
    other.mkdir()
    (other / "secret.txt").write_text("another trial's file\n", encoding="ascii")
    artifact = workdir / "main"
    artifact.write_text(VIEW.format(python=PY), encoding="ascii")
    artifact.chmod(0o755)
    scratch, home = Path(os.environ["LASSI_SCRATCH"]), Path(os.environ.get("HOME", os.environ["LASSI_SCRATCH"]))
    listed = {"scratch": scratch, "home": home, "runs": runs, "base": runs_base, "other": other}
    paths = {
        "list": {**{key: str(path) for key, path in listed.items()}, "toolchains": str(toolchains)},
        "read": {"secret": str(other / "secret.txt"), "harness": str(layout.harness / "lassi_io.h")},
        "readonly": {"toolchains": str(toolchains), "harness": str(layout.harness), "scratch": str(scratch)},
    }
    paths["readonly"]["workdir"] = str(workdir)
    executor = native.NativeExecutor(harness=str(layout.harness))
    result = executor.run(artifact, [json.dumps(paths)], PYTHON)
    assert (result.exit_code, result.hang) == (0, False), result
    report = json.loads(result.stdout)
    # The native executor hides the runs root too, wherever it lies.
    roots = [*layout.hidden, runs.resolve()]
    hidden = [root for root in roots if not any(other in root.parents for other in roots)]
    exposures = [workdir.resolve(), layout.harness, toolchains]
    for key, path in listed.items():
        assert report["list"][key] == expected_listing(path, exposures, hidden), (key, report["list"])
    assert report["list"]["toolchains"] == sorted(os.listdir(toolchains)), report["list"]
    assert report["read"]["secret"].startswith("error:"), report["read"]
    assert report["read"]["harness"] == "// harness placeholder\n", report["read"]
    assert report["readonly"] == {"toolchains": True, "harness": True, "scratch": True, "workdir": False}, report
    assert (workdir / "made.txt").read_text(encoding="ascii") == "made\n"


# ---------------------------------------------------------------------------
# P0.16 R4: output caps


def test_r4_stdout_and_stderr_are_capped_with_truncation_flags(sandbox: ModuleType, layout: Layout) -> None:
    from lassi.toolchains import _base

    flood = "head -c 67108864 /dev/zero; printf END; head -c 16777216 /dev/zero >&2; printf ERR_END >&2; exit 5"
    result = run_in(sandbox, layout, ["sh", "-c", flood], SHORT)
    assert (result.returncode, result.hang, result.killed) == (5, False, False), result.returncode
    assert (result.stdout_truncated, result.stderr_truncated) == (True, True)
    assert len(result.stdout.encode("utf-8")) <= _base.OUTPUT_CAP_BYTES
    assert len(result.stderr.encode("utf-8")) <= _base.OUTPUT_CAP_BYTES
    # The runner keeps the tail too, so the program's last words and the setup's done line survive.
    assert result.stdout.endswith("END") and result.stderr.endswith("ERR_END"), (result.stdout[-8:], result.stderr[-8:])
    assert sandbox.READY_MARKER not in result.stderr and sandbox.DONE_MARKER not in result.stderr


def test_r4_the_runner_holds_no_more_than_the_cap_in_memory(layout: Layout) -> None:
    from lassi.toolchains import _base

    # 512 MiB on stdout and 256 MiB on stderr: a runner that kept it all would grow by hundreds of MiB. One that
    # keeps the cap holds two streams of OUTPUT_CAP_BYTES plus the copies it makes to decode and split them; the
    # bound allows that plus SLACK_MIB. The bound is a test input.
    flood = "head -c 536870912 /dev/zero; head -c 268435456 /dev/zero >&2"
    roots = json.dumps([str(root) for root in layout.hidden])
    argv = [sys.executable, "-c", MEMORY_PROBE, str(layout.workdir), roots, flood]
    done = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["returncode"] == 0, report
    assert (report["stdout_truncated"], report["stderr_truncated"]) == (True, True), report
    assert report["stdout_bytes"] <= _base.OUTPUT_CAP_BYTES and report["stderr_bytes"] <= _base.OUTPUT_CAP_BYTES
    assert report["after_kib"] - report["before_kib"] < (2 * _base.OUTPUT_CAP_BYTES >> 10) + SLACK_MIB * 1024, report


# ---------------------------------------------------------------------------
# P0.16 R5: workdir disk cap


def host_bytes(directory: Path) -> int:
    """Return the total size of the regular files under `directory` on the host, without following links."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            path = Path(dirpath) / name
            if not path.is_symlink() and path.is_file():
                total += path.stat().st_size
    return total


def test_r5_writes_past_the_workdir_disk_cap_fail_inside_and_the_host_workdir_stays_under_it(
    sandbox: ModuleType, layout: Layout
) -> None:
    cap_mb = 4
    cap = cap_mb << 20
    spec = spec_for(sandbox, layout, disk_mb=cap_mb)
    # The cap must reach the setup script in bytes (the fifth element after it: sh, workdir, harness, toolchains,
    # disk) before anything is written, so an uncapped sandbox never writes the full 8 MiB to the host.
    command = sandbox.sandbox_command(spec, ["true"], PYTHON)
    assert command[command.index(sandbox.SETUP_SCRIPT) + 5] == str(cap), command
    result = sandbox.Sandbox().run(spec, [PY, "-c", FILL, str(cap)], PYTHON)
    assert (result.returncode, result.killed) == (0, False), result
    report = json.loads(result.stdout)
    assert report["error"] in ("ENOSPC", "EFBIG", "EDQUOT"), report
    assert report["written"] <= cap, report
    assert host_bytes(layout.workdir) <= cap, host_bytes(layout.workdir)


def test_r5_copy_back_never_writes_through_a_host_symbolic_link(sandbox: ModuleType, layout: Layout) -> None:
    outside = layout.base / "outside"
    outside.mkdir()
    (outside / "victim.txt").write_text("outside\n", encoding="ascii")
    os.symlink(outside / "victim.txt", layout.workdir / "victim")
    os.symlink(outside, layout.workdir / "escape", target_is_directory=True)
    script = (
        "rm -f victim escape; echo evil > victim; mkdir -p escape; echo x > escape/x.txt; "
        "ln -s /etc/hostname leak; echo PROGRAM_DONE"
    )
    result = run_in(sandbox, layout, ["sh", "-c", script], SHORT)
    assert result.stdout.split() == ["PROGRAM_DONE"], result
    assert (outside / "victim.txt").read_text(encoding="ascii") == "outside\n"
    assert sorted(path.name for path in outside.iterdir()) == ["victim.txt"]
    assert not os.path.lexists(layout.workdir / "leak")


def test_r5_copy_back_never_writes_through_a_host_hard_link(sandbox: ModuleType, layout: Layout) -> None:
    # The review's round-2 hard-link finding: a host workdir file hard-linked to a file outside the workdir is
    # replaced by a new file, so the file outside keeps its bytes; no temporary file stays in the workdir.
    outside = layout.base / "outside"
    outside.mkdir()
    (outside / "victim.txt").write_text("outside\n", encoding="ascii")
    os.link(outside / "victim.txt", layout.workdir / "data.txt")
    result = run_in(sandbox, layout, ["sh", "-c", "echo new > data.txt; echo PROGRAM_DONE"], SHORT)
    assert (result.returncode, result.stdout, result.workdir_incomplete) == (0, "PROGRAM_DONE\n", False), result
    assert (outside / "victim.txt").read_text(encoding="ascii") == "outside\n"
    assert (layout.workdir / "data.txt").read_text(encoding="ascii") == "new\n"
    assert sorted(path.name for path in layout.workdir.iterdir()) == ["data.txt"]


def test_r5_sparse_files_past_the_cap_are_not_copied_back(sandbox: ModuleType, layout: Layout) -> None:
    # The review's R5 finding: holes cost the tmpfs nothing, but copying writes them in full. The file size
    # limit stops one file past the cap (EFBIG), and the copy-back refuses files whose sizes total more.
    cap = 4 << 20
    program = (
        "import json, os, signal\n"
        "signal.signal(signal.SIGXFSZ, signal.SIG_IGN)\n"
        f"for index in range(8):\n    open('sparse%d.bin' % index, 'wb').truncate({cap - 1})\n"
        "try:\n"
        f"    handle = open('past.bin', 'wb'); handle.seek({2 * cap}); handle.write(b'x')\n"
        "    handle.close(); past = 'ok'\n"
        "except OSError as exc:\n    past = exc.errno\n"
        "open('small.txt', 'w').write('small')\n"
        "print(json.dumps({'past': past}))\n"
    )
    result = run_in(sandbox, layout, [PY, "-c", program], PYTHON, disk_mb=4)
    assert (result.returncode, result.killed, result.workdir_incomplete) == (0, False, True), result
    assert json.loads(result.stdout) == {"past": 27}, result.stdout
    assert result.stderr.startswith("lassi-sandbox copy-back: ") and "none was copied back" in result.stderr, result
    assert sorted(path.name for path in layout.workdir.iterdir()) == [], list(layout.workdir.iterdir())
    assert host_bytes(layout.workdir) <= cap


def test_r5_hard_links_count_once_each_toward_the_cap(sandbox: ModuleType, layout: Layout) -> None:
    # One 1 MiB file and seven links to it: 8 MiB to copy back under a 4 MiB cap, so nothing is copied.
    program = (
        "import os\n"
        "open('one.bin', 'wb').write(bytes(1 << 20))\n"
        "for index in range(7):\n    os.link('one.bin', 'link%d.bin' % index)\n"
        "print('LINKED')\n"
    )
    result = run_in(sandbox, layout, [PY, "-c", program], PYTHON, disk_mb=4)
    assert (result.returncode, result.stdout, result.workdir_incomplete) == (0, "LINKED\n", True), result
    assert "8388608 bytes" in result.stderr, result.stderr
    assert list(layout.workdir.iterdir()) == []


def test_r5_the_cap_stays_under_the_memory_limit_so_a_large_write_ends_in_enospc(
    sandbox: ModuleType, layout: Layout
) -> None:
    # The review's memory finding: with the default 256 MiB cap and a 96 MiB memory limit, the cap is 48 MiB,
    # so the write fails with ENOSPC or EFBIG inside instead of a memory kill; the file is removed before exit.
    limits = Limits(wall_s=20.0, memory_mb=96, cpus=1)
    cap = 48 << 20
    spec = spec_for(sandbox, layout)
    command = sandbox.sandbox_command(spec, ["true"], limits)
    assert command[command.index(sandbox.SETUP_SCRIPT) + 5] == str(cap), command
    result = sandbox.Sandbox().run(spec, [PY, "-c", FILL, str(cap), "unlink"], limits)
    assert (result.returncode, result.killed, result.workdir_incomplete) == (0, False, False), result
    report = json.loads(result.stdout)
    assert report["error"] in ("ENOSPC", "EFBIG") and report["written"] <= cap, report
    assert not (layout.workdir / "big.bin").exists()


# ---------------------------------------------------------------------------
# P0.16 R6: core dumps


def test_r6_crashes_leave_no_core_with_the_host_handler(sandbox: ModuleType, layout: Layout) -> None:
    need(shutil.which("coredumpctl") is not None, "coredumpctl is not on PATH, so the host handler cannot be read")
    command = sandbox.sandbox_command(spec_for(sandbox, layout), ["true"], SHORT)
    # Refuse to crash anything without the core limit of 1 around the whole command: at 0 the host handler
    # stores the core on the root filesystem (spike probe G1), which Agent Rule 7 forbids.
    assert command[:3] == ["prlimit", "--core=1", "--"], command[:6]
    assert "--core=1" in sandbox.SETUP_SCRIPT and "--core=0" not in sandbox.SETUP_SCRIPT
    # And refuse to crash anything unless the program provably cannot lower its core limit to 0, which the host
    # handler would honor (the review's R6 finding): a broken filter fails here, before any crash.
    preflight = run_in(sandbox, layout, [PY, "-c", CONFINED], PYTHON)
    assert preflight.returncode == 0, preflight
    report = json.loads(preflight.stdout)
    assert (report["setrlimit_core"], report["prlimit_core"], report["rlimit_core"]) == ("ValueError", "EPERM", [1, 1])
    since = int(time.time()) - 1
    crashes = [
        (["sh", "-c", "kill -SEGV $$"], 139),
        (["sh", "-c", "ulimit -c 0 2> /dev/null; kill -SEGV $$"], 139),
        ([PY, "-c", "import os; os.abort()"], 134),
        ([PY, "-c", "import ctypes; ctypes.string_at(0)"], 139),
    ]
    for argv, status in crashes:
        result = run_in(sandbox, layout, argv, PYTHON)
        assert (result.returncode, result.hang, result.killed) == (status, False, False), (argv, result)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since))
    command = ["coredumpctl", "--no-pager", "list", f"--since={stamp}", f"COREDUMP_UID={os.getuid()}"]
    # systemd-coredump works asynchronously, so the journal is read every second for 15 s, and any entry fails.
    deadline = time.monotonic() + 15
    while True:
        listing = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        assert listing.stdout.strip() == "", listing
        if time.monotonic() >= deadline:
            break
        time.sleep(1)
    # Only this outcome proves the absence: exit status 1 with coredumpctl's own "No coredumps found" notice.
    # A refused journal, a bad timestamp, or any listed entry fails the test.
    assert listing.returncode == 1 and listing.stdout.strip() == "", listing
    assert "No coredumps found" in listing.stderr and "permission" not in listing.stderr.lower(), listing
    assert not list(layout.workdir.glob("core*"))


# ---------------------------------------------------------------------------
# P0.16 R7: runner kill


def processes_with(markers: Sequence[str]) -> list[tuple[int, list[str]]]:
    """Return (pid, argv) of every process whose command line holds one of `markers` as an argument."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        with contextlib.suppress(OSError):
            argv = (entry / "cmdline").read_bytes().decode("utf-8", errors="replace").split("\0")
            if any(marker in part for part in argv for marker in markers):
                found.append((int(entry.name), argv))
    return found


def test_r7_the_runners_own_timeout_leaves_no_process_of_the_sandbox(sandbox: ModuleType, layout: Layout) -> None:
    token = uuid.uuid4().int % 10**9
    markers = [f"{900 + index}.{token:09d}" for index in range(3)]
    # A setsid child (its own session and process group), a plain background child, and an orphaned one.
    program = ["sh", "-c", 'setsid sleep "$1" & sleep "$2" & (sleep "$3" &); wait', "sh", *markers]
    seen: set[str] = set()
    stop = threading.Event()

    def watch() -> None:
        """Record which sleep markers are running, until told to stop."""
        while not stop.is_set():
            for _pid, argv in processes_with(markers):
                seen.update(marker for marker in markers if argv[:2] == ["sleep", marker])
            time.sleep(0.1)

    raw: list[CommandResult] = []

    def short_runner(argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Run the sandbox command with a 4 s runner timeout, far below every limit inside the sandbox."""
        result = capped_runner(argv, cwd, 4.0)
        raw.append(result)
        return result

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        with contextlib.suppress(sandbox.SandboxUnavailableError):
            limits = Limits(wall_s=30.0, memory_mb=64, cpus=1)
            sandbox.Sandbox(runner=short_runner).run(spec_for(sandbox, layout), program, limits)
    finally:
        stop.set()
        watcher.join()
    assert raw and raw[0].returncode == -1, raw
    assert seen == set(markers), (seen, markers)
    deadline = time.monotonic() + 3.0
    while processes_with(markers) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert processes_with(markers) == []


# ---------------------------------------------------------------------------
# Fixes from the P0.16 review


def test_the_program_runs_in_its_own_session_under_the_seccomp_filter(sandbox: ModuleType, layout: Layout) -> None:
    result = run_in(sandbox, layout, [PY, "-c", CONFINED], PYTHON, disk_mb=4)
    assert result.returncode == 0, result
    report = json.loads(result.stdout)
    expected = {
        "setrlimit_core": "ValueError", "setrlimit_nofile": "ok", "prlimit_core": "EPERM", "prlimit_other": "EPERM",
        "vsock": "EPERM", "inet": "ok", "tty": "ENXIO", "io_uring": "EPERM", "core_filter": "00000000",
        "rlimit_core": [1, 1], "rlimit_fsize": [4 << 20, 4 << 20],
        "keyctl_user": "EPERM", "add_key": "EPERM", "request_key": "EPERM",
    }  # fmt: skip
    assert {key: report[key] for key in expected} == expected, report
    # timeout is pid 1 of the program's namespace and leads the session CONFINE_PROGRAM started.
    assert report["sid"] == 1 and report["pid"] != 1, report


def test_the_program_holds_none_of_the_callers_keys(layout: Layout) -> None:
    payload = f"lassi-sandbox-test-key-{uuid.uuid4().hex}"
    roots = json.dumps([str(root) for root in layout.hidden])
    argv = [sys.executable, "-c", KEY_PROBE, str(layout.workdir), roots, payload, PY, READ_KEY]
    done = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["returncode"] == 0, report
    inside = json.loads(report["stdout"])
    # The filter refuses every keyring call: neither the key (by its serial) nor the keyrings @s, @u, and @us can
    # be read. (Inside the sandbox @u and @us resolve per user namespace on this kernel; the serial is global.)
    assert payload.encode("ascii").hex() not in report["stdout"], inside
    assert inside == {"key": "EPERM", "session": "EPERM", "user": "EPERM", "user_session": "EPERM"}, inside


def test_sys_device_attributes_and_var_entries_are_hidden(sandbox: ModuleType, layout: Layout) -> None:
    need(bool(os.listdir("/sys/class")), "the host shows nothing under /sys/class, so hiding it proves nothing")
    result = run_in(sandbox, layout, [PY, "-c", SYSTEM_VIEW], PYTHON)
    assert result.returncode == 0, result
    report = json.loads(result.stdout)
    assert (report["class"], report["bus"], report["devices_not_empty"]) == ([], [], []), report
    assert report["cpu_online"], report
    assert {name: names for name, names in report["var"].items() if names and name != "tmp"} == {}, report["var"]


def host_sockets(tops: Sequence[str], budget_s: float = 20.0) -> list[str]:
    """Return the unix sockets the host shows under `tops`, at most 6 levels down, found within `budget_s`."""
    found: list[str] = []
    began = time.monotonic()
    for top in tops:
        for dirpath, dirnames, filenames in os.walk(top):
            if time.monotonic() - began > budget_s:
                return found
            if dirpath.count("/") >= 6:
                dirnames[:] = []
            for name in filenames:
                path = os.path.join(dirpath, name)
                with contextlib.suppress(OSError):
                    if stat.S_ISSOCK(os.lstat(path).st_mode):
                        found.append(path)
    return found


def test_host_daemon_sockets_under_var_snap_opt_and_srv_are_out_of_view(sandbox: ModuleType, layout: Layout) -> None:
    # The review's socket finding: connect needs no writable mount, so a socket must be out of view. Nothing
    # here ever connects to a host socket.
    sockets = host_sockets(["/var", "/snap", "/opt", "/srv"])
    need(bool(sockets), "the host shows no socket under /var, /snap, /opt, or /srv to check")
    result = run_in(sandbox, layout, [PY, "-c", SOCKET_VIEW, json.dumps(sockets)], PYTHON)
    assert result.returncode == 0, result
    report = json.loads(result.stdout)
    assert {path: seen for path, seen in report.items() if seen != "hidden"} == {}, report


def test_a_confinement_failure_raises_and_never_runs_the_program(
    sandbox: ModuleType, layout: Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The review's round-2 question: a failure of the program's chain before the program starts is the
    # sandbox's, never the program's exit status. Negative control: CONFINE_PROGRAM asks for seccomp mode 99,
    # which the kernel refuses with EINVAL, so it exits before its ready line and before the program.
    spec = spec_for(sandbox, layout)
    program = ["sh", "-c", 'touch "$1/ran"; echo RAN', "sh", str(layout.workdir)]
    control = sandbox.Sandbox().run(spec, program, SHORT)
    assert (control.returncode, control.stdout) == (0, "RAN\n"), control
    (layout.workdir / "ran").unlink()
    call, refused = "libc.prctl(22, 2, program, 0, 0)", "libc.prctl(22, 99, program, 0, 0)"
    assert sandbox.SETUP_SCRIPT.count(call) == 1
    monkeypatch.setattr(sandbox, "SETUP_SCRIPT", sandbox.SETUP_SCRIPT.replace(call, refused))
    with pytest.raises(sandbox.SandboxUnavailableError, match="did not run") as caught:
        sandbox.Sandbox().run(spec, program, SHORT)
    assert "lassi-sandbox confine: " in str(caught.value), caught.value
    assert not (layout.workdir / "ran").exists()


def test_a_done_line_the_program_printed_never_passes_for_a_finished_copy_back(
    sandbox: ModuleType, layout: Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The review's round-2 question: the program ends its stderr with the done line, and the copy-back then
    # fails (here made to exit 9 at once). The setup's EXIT trap prints its own line last, so the run raises
    # instead of reporting the program's status with a workdir that never came back.
    spec = spec_for(sandbox, layout)
    # The forged line is well formed (two clock readings), so only the setup's own final line can reject it.
    forged = sandbox.DONE_MARKER + " 100.00 101.00"
    program = ["sh", "-c", 'echo made > made.txt; printf "%s\\n" "$1" >&2', "sh", forged]
    control = sandbox.Sandbox().run(spec, program, SHORT)
    # Without the break, the setup's own done line follows the program's, and only the setup's is removed.
    done_line = forged + "\n"
    assert (control.returncode, control.stderr, control.workdir_incomplete) == (0, done_line, False), control
    (layout.workdir / "made.txt").unlink()
    call = "status = main(sys.argv[1], sys.argv[2], int(sys.argv[3]))"
    assert sandbox.SETUP_SCRIPT.count(call) == 1
    monkeypatch.setattr(sandbox, "SETUP_SCRIPT", sandbox.SETUP_SCRIPT.replace(call, "status = 9"))
    with pytest.raises(sandbox.SandboxUnavailableError, match="copy-back") as caught:
        sandbox.Sandbox().run(spec, program, SHORT)
    assert "the setup stopped after the program ended" in str(caught.value), caught.value
    assert not (layout.workdir / "made.txt").exists()


def test_a_path_entry_the_program_can_write_never_reaches_the_setup(
    sandbox: ModuleType, layout: Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The review's PATH finding: the program plants umount and python3 on a PATH entry under the private /run;
    # the setup runs with SANDBOX_PATH after env -i, so neither runs as the setup and the copy-back finishes.
    planted = f"/run/lassi-sandbox-test-{uuid.uuid4().hex}"
    monkeypatch.setenv("PATH", planted + os.pathsep + os.environ["PATH"])
    marker = layout.workdir / "planted-ran"
    script = (
        'mkdir -p "$1" && for name in umount python3 mount; do '
        'printf "#!/bin/sh\\ntouch %s\\nexit 0\\n" "$2" > "$1/$name" && chmod 755 "$1/$name"; done && echo PLANTED'
    )
    result = run_in(sandbox, layout, ["sh", "-c", script, "sh", planted, str(marker)], SHORT)
    assert (result.returncode, result.stdout) == (0, "PLANTED\n"), result
    assert not marker.exists()
    assert not Path(planted).exists()
