"""Remote tests for the sandbox and the native executor on the build host (P0.10).

These run real sandboxes, so they are marked `remote` and skip unless the
host has what the mechanism in plans/spikes/p0-sandbox.md needs: Linux,
unshare, systemd-run, nice, setpriv, prlimit, and timeout on PATH, a
reachable user systemd manager ($XDG_RUNTIME_DIR/bus exists), and
$LASSI_SCRATCH. With LASSI_REQUIRE_SANDBOX=1 a host that cannot run them
fails the tests instead of skipping them, so a silent skip cannot pass for
evidence. Run them with
`rx run -- 'LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/executors'`
(P0.10 Remote).

Each test builds a tiny program as a shell or python3 command (no compiler)
in a fresh build directory under $LASSI_SCRATCH/tmp, which is removed
afterwards, and runs it only through Sandbox.run or NativeExecutor.run (Agent
Rule 6), with $LASSI_SCRATCH and $HOME read-only and a temp harness directory
unless a test says otherwise. Nothing is written on the host root filesystem
(Agent Rule 7): the root filesystem and cgroup checks read /proc/self/mountinfo
or rewrite a limit with its own value, and the tmp tests remove their marker
file if the sandbox ever lets it through. They check the P0.10 acceptance
criteria and the review fixes: a network connect fails, a memory hog and a
CPU-time hog are killed, wall time ends a run at the limit and not before,
writes to the harness and read-only roots fail while the workdir stays
writable, the program holds no capability and cannot undo a mount, it cannot
create a nested user namespace, it runs at nice 19, the environment and the
bus sockets stay outside, a failed setup never runs the program, and the exit
status passes through. The limits are test inputs; no value here is a
measurement.
"""

from __future__ import annotations

import os
import shutil
import socket
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from lassi.core.interfaces import Limits

TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout")
SHORT = Limits(wall_s=10.0, memory_mb=64, cpus=1)
# Room for a python3 interpreter.
PYTHON = Limits(wall_s=20.0, memory_mb=256, cpus=1)
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"


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
    roots: tuple[Path, ...]


def default_roots() -> tuple[Path, ...]:
    """Return $LASSI_SCRATCH and, when set and different, $HOME: the roots the sandbox remounts read-only."""
    roots = [Path(os.environ["LASSI_SCRATCH"])]
    home = os.environ.get("HOME", "")
    if home and Path(home) not in roots:
        roots.append(Path(home))
    return tuple(roots)


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
        yield Layout(base=base, workdir=workdir, harness=harness, roots=default_roots())
    finally:
        shutil.rmtree(base)


def run_in(sandbox: ModuleType, layout: Layout, argv: Sequence[str], limits: Limits) -> object:
    """Run `argv` through Sandbox.run with the layout's workdir, read-only roots, and harness."""
    spec = sandbox.SandboxSpec(workdir=layout.workdir, readonly_roots=layout.roots, harness=layout.harness)
    return sandbox.Sandbox().run(spec, list(argv), limits)


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

# $1 a read-only root's mount point, $2 the harness, $3 a directory under the root.
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
def burn():
    while True:
        hashlib.sha256(data).digest()
threads = [threading.Thread(target=burn, daemon=True) for _ in range(4)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
"""

PROCESSES = "import os; print(sum(name.isdigit() for name in os.listdir('/proc')))"


def test_network_connect_fails_inside(sandbox: ModuleType, layout: Layout) -> None:
    if not host_can_connect():
        pytest.skip("the host itself cannot reach 1.1.1.1:443, so a blocked connect inside would prove nothing")
    result = run_in(sandbox, layout, [sys.executable, "-c", CONNECT], PYTHON)
    assert result.stdout.split()[:1] == ["BLOCKED"], result
    assert result.returncode == 0, result


def test_memory_hog_is_killed(sandbox: ModuleType, layout: Layout) -> None:
    hog = "b = bytearray(256 * 1024 * 1024)\nprint('ALLOCATED', len(b))\n"
    result = run_in(sandbox, layout, [sys.executable, "-c", hog], Limits(wall_s=20.0, memory_mb=64, cpus=1))
    assert "ALLOCATED" not in result.stdout, result
    assert (result.returncode, result.killed, result.hang) == (137, True, False), result


def test_cpu_time_hog_is_killed_before_the_wall_limit(sandbox: ModuleType, layout: Layout) -> None:
    # CPU budget 20 s under a 20 s wall limit; four busy threads spend it in about 5 s of wall time.
    result = run_in(sandbox, layout, [sys.executable, "-c", CPU_BURN], PYTHON)
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


def test_a_program_ignoring_sigterm_is_killed_after_the_grace(sandbox: ModuleType, layout: Layout) -> None:
    argv = ["sh", "-c", 'trap "" TERM; sleep 30']
    result = run_in(sandbox, layout, argv, Limits(wall_s=3.0, memory_mb=64, cpus=1))
    assert (result.returncode, result.hang, result.killed) == (124, True, False), result
    assert 3.0 + sandbox.KILL_AFTER_S <= result.wall_s < 9.0, result


def test_a_program_within_the_wall_limit_runs_to_its_end(sandbox: ModuleType, layout: Layout) -> None:
    result = run_in(sandbox, layout, ["sleep", "4"], Limits(wall_s=6.0, memory_mb=64, cpus=1))
    assert (result.returncode, result.hang, result.killed) == (0, False, False), result
    assert 4.0 <= result.wall_s < 6.0, result


def test_harness_and_readonly_roots_refuse_writes_and_the_workdir_takes_them(
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
    spec = sandbox.SandboxSpec(workdir=layout.workdir, readonly_roots=(other_root,), harness=layout.harness)
    script = 'if touch "$1/written" 2> /dev/null; then echo HARNESS_WROTE; else echo HARNESS_READONLY; fi'
    result = sandbox.Sandbox().run(spec, ["sh", "-c", script, "sh", str(layout.harness)], SHORT)
    assert result.returncode == 0, result
    assert result.stdout.split() == ["HARNESS_READONLY"], result
    assert not (layout.harness / "written").exists()


def test_every_readonly_root_is_read_only_not_only_the_first(sandbox: ModuleType, layout: Layout) -> None:
    roots = (layout.base / "ro-a", layout.base / "ro-b")
    for root in roots:
        root.mkdir()
    spec = sandbox.SandboxSpec(workdir=layout.workdir, readonly_roots=roots, harness=layout.harness)
    script = 'for d in "$@"; do if touch "$d/written" 2> /dev/null; then echo WROTE; else echo READONLY; fi; done'
    result = sandbox.Sandbox().run(spec, ["sh", "-c", script, "sh", *map(str, roots)], SHORT)
    assert result.returncode == 0, result
    assert result.stdout.split() == ["READONLY", "READONLY"], result
    assert not any((root / "written").exists() for root in roots)


def test_the_program_holds_no_capability_and_cannot_undo_a_mount(sandbox: ModuleType, layout: Layout) -> None:
    root = layout.roots[0]
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
    result = run_in(sandbox, layout, [sys.executable, "-c", MOUNTS, *points], PYTHON)
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
    result = run_in(sandbox, layout, [sys.executable, "-c", CGROUP], PYTHON)
    assert result.returncode == 0, result
    assert result.stdout.split()[:1] == ["CGROUP_REFUSED"], result


def test_the_bus_sockets_are_hidden_inside(sandbox: ModuleType, layout: Layout) -> None:
    runtime = Path(os.environ["XDG_RUNTIME_DIR"])
    sockets = [str(runtime / "bus"), str(runtime / "systemd" / "private"), "/run/dbus/system_bus_socket"]
    argv = [sys.executable, "-c", SOCKETS, *sockets]
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
    spec = sandbox.SandboxSpec(workdir=layout.workdir, readonly_roots=(layout.base / "missing",), harness=None)
    argv = ["sh", "-c", 'touch "$1/ran"', "sh", str(layout.workdir)]
    with pytest.raises(sandbox.SandboxUnavailableError, match="did not run"):
        sandbox.Sandbox().run(spec, argv, SHORT)
    assert not (layout.workdir / "ran").exists()


def test_the_pid_namespace_hides_host_processes(sandbox: ModuleType, layout: Layout) -> None:
    result = run_in(sandbox, layout, [sys.executable, "-c", PROCESSES], PYTHON)
    assert result.returncode == 0, result
    assert int(result.stdout) < 10, result


def test_exit_status_passes_through(sandbox: ModuleType, layout: Layout) -> None:
    result = run_in(sandbox, layout, ["sh", "-c", "echo OK; echo ERR >&2; exit 3"], SHORT)
    assert (result.returncode, result.hang, result.killed) == (3, False, False), result
    assert result.stdout == "OK\n", result
    assert "ERR" in result.stderr, result
    assert sandbox.READY_MARKER not in result.stderr, result


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
