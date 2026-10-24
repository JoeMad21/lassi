"""Remote tests for the sandbox's Tenstorrent exposure, with stand-ins and no device opened (task P17.15).

Bible: Sandbox, Execution Backends (the tt_silicon row), Agent Rules 6, 7,
and 9; plans/p17-portable.md, task P17.15; plans/spikes/tt-silicon-executor.md
(Findings 1 to 3, and section 6, tests 12 and 13). The interface is the one
tests/executors/test_sandbox_tenstorrent.py describes: a TenstorrentExposure
of numbered nodes, PCI directories, the run's own hugepage directory, and
an explicit reset filter setting, carried in SandboxSpec.devices.

These run real sandboxes through Sandbox.run with hand-made SandboxSpecs
(Agent Rule 6), never through an executor. They are marked `remote` and skip
on a host that cannot run the sandbox (the host checks of
test_sandbox_remote.py, restated here as test_gpu_remote.py restates them);
with LASSI_REQUIRE_SANDBOX=1 such a host fails them instead, so a silent
skip cannot pass for evidence.

No device node is opened, and the host's /dev/tenstorrent is never named in
any file call (Agent Rule 9). The setup binds the stand-in sources only, and
binding opens nothing. Inside, the program only lstat-s the paths under
/dev/tenstorrent, so even a setup that wrongly bound a host card's node
would show its device number without opening it. The one device node any
program opens is /dev/null, for the reset filter's ioctl calls; /dev/null
has no ioctl of its own, so each call fails with an error and does nothing.

The stand-ins:

- the cards' nodes: the host's /dev/zero and /dev/full in turn, bound at
  /dev/tenstorrent/0 to /dev/tenstorrent/7, eight as on alpha01 (the spec
  takes any node under /dev as a source; the P17.16 executor is to bind
  each card's node at its own path);
- the cards' PCI directories: the first PCI devices the host lists under
  /sys/bus/pci/devices that the tenstorrent driver has not bound
  (pci_standins; only each device's driver link is read, so no file of a
  card is opened), each the directory its link resolves to, bound at its
  own /sys/bus/pci/devices/<address>, since the spec takes only a source
  whose last segment is its target's address. Eight of them under the
  /sys/bus cover check that the cover has room for every card's directory;
- /sys/bus/pci/slots, /sys/module/tenstorrent/version, and the 1 GiB
  nr_hugepages file are the host's own, compared with what the host lists
  or reads (a missing file must be missing inside too);
- the hugepage directory. No hugetlbfs can be mounted without root (spike,
  Finding 3), so the tests stand in for the run's directory in two ways:
  - test_a_plain_directory_cannot_stand_in_for_the_hugepage_mount uses a
    plain empty directory under $LASSI_SCRATCH/tmp. The spec takes only a
    directory under /dev/hugepages-1G, so the test names a placeholder there
    that does not exist, and its command runner swaps the placeholder for
    the plain directory in the sandbox command before running it; it
    refuses to run a command that still names any path under
    /dev/hugepages-1G, so the host's hugepage mount is never touched. The
    directory is the user's own, so it passes the setup's owner check; the
    setup must then fail closed: its mount check trusts only a hugetlbfs
    mount there, so a writable bind of another type stops it before the
    program runs;
  - the tests that take the `hugepage_dir` fixture use a real one: an empty
    directory lassi-test-<random hex>, mode 0700, made in the host's
    /dev/hugepages-1G, a hugetlbfs with 1 GiB pages that is mode 777 on
    alpha01 (spike probe 7aeb). Inside, one program creates one empty file
    there and never maps it, so no page is taken. The directory and the file
    are removed afterwards. That is a write outside the scratch root, on a
    mount other users share, so these tests run only with
    LASSI_HUGEPAGE_TESTS=1, set with the owner's approval of the run (Agent
    Rule 17); with it set, a host without such a mount fails them.

Run them on the build host with `PYTHONIOENCODING=utf-8 rx run --
'LASSI_REQUIRE_SANDBOX=1 LASSI_HUGEPAGE_TESTS=1 uv run pytest -q -m remote
tests/executors/test_sandbox_tenstorrent_remote.py'`, or without
LASSI_HUGEPAGE_TESTS for the tests that need no hugepage mount. Every other
file a test writes lies in the temp directory it creates under
$LASSI_SCRATCH/tmp. No value here is a measurement.
"""

from __future__ import annotations

import errno
import json
import os
import posixpath
import re
import shutil
import stat
import sys
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import pytest

from lassi.core.interfaces import Limits
from lassi.executors.sandbox import SANDBOX_PATH
from lassi.toolchains import CommandResult, capped_runner

TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk")
# Room for a python3 interpreter.
PYTHON = Limits(wall_s=20.0, memory_mb=256, cpus=1)
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
HUGEPAGE_TESTS = os.environ.get("LASSI_HUGEPAGE_TESTS") == "1"
# alpha01's eight cards (lspci, rx 20261021-201421-exec-4ea0): the number of stand-in nodes and PCI directories.
CARD_COUNT = 8
# The stand-in nodes, as (host source, path inside).
CARDS = tuple((("/dev/zero", "/dev/full")[card % 2], f"/dev/tenstorrent/{card}") for card in range(CARD_COUNT))
PCI_DEVICES = "/sys/bus/pci/devices"
# A PCI address as the spec takes a target's last segment, and a path segment as it takes every segment.
PCI_ADDRESS = re.compile(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
SLOTS = "/sys/bus/pci/slots"
DRIVER_VERSION = "/sys/module/tenstorrent/version"
PAGE_COUNT = "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages"
HUGEPAGE_POINT = "/dev/hugepages-1G"
# The empty file the program creates in the run's hugepage directory (never mapped, so it holds no page).
PROBE = "lassi-probe"
PROGRAM_ENVIRONMENT = {"PATH": SANDBOX_PATH, "LANG": "C.UTF-8", "TMPDIR": "/tmp"}
# The private /dev's own entries (test_sandbox_remote.py, R1): its nodes, directories, and links.
DEV_NODES = {"null", "zero", "full", "random", "urandom", "tty", "pts/ptmx"}
DEV_DIRS = {"pts", "shm"}
DEV_LINKS = {
    "fd": "/proc/self/fd",
    "stdin": "/proc/self/fd/0",
    "stdout": "/proc/self/fd/1",
    "stderr": "/proc/self/fd/2",
    "ptmx": "pts/ptmx",
}
# The filesystem types the setup's own writable mounts have (each with a lassi-* source).
SETUP_FSTYPES = {"tmpfs", "devpts", "overlay"}


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
    """Return why this host cannot run real sandboxes, or "" when it can (test_sandbox_remote.py's checks)."""
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


def pci_standins(count: int) -> tuple[tuple[str, str], ...]:
    """Return up to `count` of the host's PCI device directories that are no card's, as (source, target) pairs.

    The target is the device's /sys/bus/pci/devices/<address> link and the
    source the directory it resolves to, whose last segment is that
    address. A device the tenstorrent driver has bound is left out; only
    its driver link is read (readlink), so no file of a card is opened.
    """
    if not os.path.isdir(PCI_DEVICES):
        return ()
    found: list[tuple[str, str]] = []
    for address in sorted(os.listdir(PCI_DEVICES)):
        target = f"{PCI_DEVICES}/{address}"
        driver = f"{target}/driver"
        bound = os.path.basename(os.readlink(driver)) if os.path.islink(driver) else ""
        source = os.path.realpath(target)
        parts = source.split("/")[1:]
        usable = PCI_ADDRESS.fullmatch(address) and parts[0] == "sys" and parts[-1] == address
        if bound != "tenstorrent" and usable and all(SEGMENT.fullmatch(part) for part in parts):
            found.append((source, target))
        if len(found) == count:
            break
    return tuple(found)


def need_standins(count: int = CARD_COUNT) -> tuple[tuple[str, str], ...]:
    """Return `count` PCI stand-ins; skip (or fail under LASSI_REQUIRE_SANDBOX=1) without them or the slot list."""
    pci = pci_standins(count)
    need(len(pci) == count, f"the host lists {len(pci)} PCI devices that are not cards, not {count}")
    need(os.path.isdir(SLOTS), f"the host has no {SLOTS} to bind")
    return pci


@pytest.fixture
def sandbox() -> ModuleType:
    """Return the lassi.executors.sandbox module."""
    from lassi.executors import sandbox

    return sandbox


@dataclass(frozen=True)
class Layout:
    """One test's directories: a fresh base under $LASSI_SCRATCH/tmp holding the workdir, and the hidden roots."""

    base: Path
    workdir: Path
    hidden: tuple[Path, ...]


@pytest.fixture
def layout() -> Iterator[Layout]:
    """Create a fresh build directory under $LASSI_SCRATCH/tmp; remove it afterwards."""
    parent = Path(os.environ["LASSI_SCRATCH"]) / "tmp"
    parent.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="lassi-tt-test.", dir=parent))
    try:
        workdir = base / "build"
        workdir.mkdir()
        yield Layout(base=base, workdir=workdir, hidden=hidden_default())
    finally:
        shutil.rmtree(base)


def host_hugepage_mount() -> list[str] | None:
    """Return the host's mountinfo fields of the top hugetlbfs mount at /dev/hugepages-1G with 1 GiB pages, or None."""
    found = None
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        tail = fields[fields.index("-", 6) + 1 :]
        if fields[4] == HUGEPAGE_POINT and tail[0] == "hugetlbfs" and "pagesize=1024M" in tail[2].split(","):
            found = fields
    return found


@pytest.fixture
def hugepage_dir() -> Iterator[tuple[Path, str]]:
    """Make an empty run directory in the host's /dev/hugepages-1G; yield it and the root a bind of it shows.

    Only with LASSI_HUGEPAGE_TESTS=1 (module docstring); then a host without
    a writable 1 GiB hugetlbfs at /dev/hugepages-1G fails the test. The
    directory and whatever the test made in it are removed afterwards.
    """
    if not HUGEPAGE_TESTS:
        pytest.skip("writes one empty directory in the host's /dev/hugepages-1G; set LASSI_HUGEPAGE_TESTS=1")
    mount = host_hugepage_mount() if sys.platform.startswith("linux") else None
    if mount is None or not os.access(HUGEPAGE_POINT, os.W_OK | os.X_OK):
        pytest.fail(f"LASSI_HUGEPAGE_TESTS=1, but no writable 1 GiB hugetlbfs is mounted at {HUGEPAGE_POINT}")
    path = Path(HUGEPAGE_POINT) / f"lassi-test-{uuid.uuid4().hex}"
    path.mkdir(mode=0o700)
    try:
        yield path, posixpath.join(mount[3], path.name)
    finally:
        shutil.rmtree(path)


def tt_exposure(
    sandbox: ModuleType,
    hugepages: str,
    *,
    reset_filter: bool,
    pci: Sequence[tuple[str, str]],
    cards: Sequence[tuple[str, str]] = CARDS,
) -> Any:
    """Return a TenstorrentExposure of the stand-ins and `hugepages`, from (host source, path inside) pairs."""
    bind = sandbox.DeviceBind
    return sandbox.TenstorrentExposure(
        nodes=tuple(bind(PurePosixPath(source), PurePosixPath(target)) for source, target in cards),
        pci_dirs=tuple(bind(PurePosixPath(source), PurePosixPath(target)) for source, target in pci),
        hugepages=PurePosixPath(hugepages),
        reset_filter=reset_filter,
    )


def spec_with(sandbox: ModuleType, layout: Layout, devices: Any) -> Any:
    """Return the SandboxSpec these tests use: the layout's workdir and hidden roots, the program environment."""
    return sandbox.SandboxSpec(
        workdir=layout.workdir, hidden_roots=layout.hidden, environment=dict(PROGRAM_ENVIRONMENT), devices=devices
    )


def run_json(sandbox: ModuleType, spec: Any, program: str, *args: str) -> dict[str, Any]:
    """Run a python3 program through Sandbox.run under `spec` and return the JSON it printed."""
    result = sandbox.Sandbox().run(spec, [PY, "-c", program, *args], PYTHON)
    assert result.returncode == 0, result
    return json.loads(result.stdout)


# Creates PROBE in /dev/hugepages-1G (empty, never mapped), then prints: every entry under /dev (lstat only: no node
# is opened), the /proc/mounts lines at /dev/hugepages-1G, the mountinfo lines, and the listings and texts argv[1]
# names (JSON), each "error:<name>" when it fails.
TT_REPORT = """from __future__ import annotations
import errno, json, os, stat, sys
def failure(exc: OSError) -> str:
    return "error:" + errno.errorcode.get(exc.errno or 0, str(exc.errno))
def kind(path: str) -> list[object]:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        return ["link", os.readlink(path)]
    if stat.S_ISCHR(info.st_mode) or stat.S_ISBLK(info.st_mode):
        return ["chr" if stat.S_ISCHR(info.st_mode) else "blk", os.major(info.st_rdev), os.minor(info.st_rdev)]
    if stat.S_ISDIR(info.st_mode):
        return ["dir"]
    return ["file"] if stat.S_ISREG(info.st_mode) else ["other", stat.S_IFMT(info.st_mode)]
def walk(top: str) -> dict[str, list[object]]:
    entries = {}
    for directory, directories, files in os.walk(top):
        for name in directories + files:
            path = os.path.join(directory, name)
            entries[os.path.relpath(path, top)] = kind(path)
    return entries
def listing(path: str) -> object:
    try:
        return sorted(os.listdir(path))
    except OSError as exc:
        return failure(exc)
def text(path: str) -> object:
    try:
        with open(path, encoding="ascii") as handle:
            return handle.read()
    except OSError as exc:
        return failure(exc)
report = {}
try:
    os.close(os.open("/dev/hugepages-1G/lassi-probe", os.O_RDONLY | os.O_CREAT | os.O_EXCL, 0o600))
    report["probe"] = "created"
except OSError as exc:
    report["probe"] = failure(exc)
report["dev"] = walk("/dev")
with open("/proc/mounts", encoding="ascii") as handle:
    report["proc_mounts"] = [line.split() for line in handle if line.split()[1] == "/dev/hugepages-1G"]
with open("/proc/self/mountinfo", encoding="ascii") as handle:
    report["mountinfo"] = [line.split() for line in handle]
paths = json.loads(sys.argv[1])
report["listings"] = {path: listing(path) for path in paths["listings"]}
report["texts"] = {path: text(path) for path in paths["texts"]}
print(json.dumps(report))
"""

# Calls ioctl(2) (x86_64 syscall 16) on a /dev/null descriptor with the driver's reset request _IO(0xFA, 6), the
# same with a high bit set in the register, and the request below it, and prints each call's errno name. /dev/null
# has no ioctl, so every call the filter lets through fails with the kernel's own error and does nothing.
IOCTLS = """from __future__ import annotations
import ctypes, errno, json, os
libc = ctypes.CDLL(None, use_errno=True)
def ioctl(descriptor: int, request: int) -> str:
    done = libc.syscall(ctypes.c_long(16), ctypes.c_long(descriptor), ctypes.c_ulong(request), ctypes.c_ulong(0))
    return "ok" if done == 0 else errno.errorcode.get(ctypes.get_errno(), str(ctypes.get_errno()))
descriptor = os.open("/dev/null", os.O_RDONLY)
requests = {"reset": 0xFA06, "reset_high_bit": (1 << 32) | 0xFA06, "below": 0xFA05}
print(json.dumps({name: ioctl(descriptor, request) for name, request in requests.items()}))
"""

# Prints whether the private /dev holds a card directory or a hugepage directory, and what /sys/bus lists.
PLAIN_REPORT = """from __future__ import annotations
import json, os
def listing(path: str) -> object:
    try:
        return sorted(os.listdir(path))
    except OSError:
        return "absent"
paths = ("/dev/tenstorrent", "/dev/hugepages-1G")
print(json.dumps({"dev": {path: os.path.lexists(path) for path in paths}, "bus": listing("/sys/bus")}))
"""


def unescape(field: str) -> str:
    """Return a mountinfo path field with its octal escapes decoded."""
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), field)


def top_mount(mountinfo: Sequence[Sequence[str]], path: str) -> Sequence[str]:
    """Return the mountinfo fields of the mount `path` lies on: the longest point that is it or a parent, last seen."""
    best: Sequence[str] | None = None
    length = -1
    for fields in mountinfo:
        point = unescape(fields[4])
        if (path == point or path.startswith(point.rstrip("/") + "/")) and len(point) >= length:
            best, length = fields, len(point)
    assert best is not None, path
    return best


def fstype(fields: Sequence[str]) -> str:
    """Return a mountinfo line's filesystem type."""
    return fields[list(fields).index("-", 6) + 1]


def source(fields: Sequence[str]) -> str:
    """Return a mountinfo line's mount source."""
    return fields[list(fields).index("-", 6) + 2]


def host_node(path: str) -> list[object]:
    """Return ["chr", major, minor] of a host stand-in node, read with stat (nothing is opened)."""
    info = os.stat(path)
    assert stat.S_ISCHR(info.st_mode), path
    return ["chr", os.major(info.st_rdev), os.minor(info.st_rdev)]


def host_text(path: str) -> object:
    """Return a host file's text, or "error:<name>" as the program reports it."""
    try:
        return Path(path).read_text(encoding="ascii")
    except OSError as exc:
        return "error:" + errno.errorcode.get(exc.errno or 0, str(exc.errno))


def report_paths(pci: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    """Return what TT_REPORT lists and reads: the /sys/bus tree, each stand-in target, /sys/class, and the files."""
    listings = ["/sys/bus", "/sys/bus/pci", "/sys/bus/pci/devices", SLOTS, "/sys/class"]
    listings += [target for _source, target in pci]
    return {"listings": listings, "texts": [DRIVER_VERSION, PAGE_COUNT]}


def test_every_stand_in_card_is_visible_read_only_beside_one_writable_hugepage_directory(
    sandbox: ModuleType, layout: Layout, hugepage_dir: tuple[Path, str]
) -> None:
    pci = need_standins()
    host_dir, root = hugepage_dir
    spec = spec_with(sandbox, layout, tt_exposure(sandbox, str(host_dir), reset_filter=False, pci=pci))
    seen = run_json(sandbox, spec, TT_REPORT, json.dumps(report_paths(pci)))
    dev: dict[str, list[object]] = seen["dev"]
    # Every card's node at its own number, each its stand-in, and nothing else under /dev/tenstorrent (no by-id).
    cards = {key: kind for key, kind in dev.items() if key == "tenstorrent" or key.startswith("tenstorrent/")}
    expected = {target[len("/dev/") :]: host_node(node) for node, target in CARDS}
    assert cards == {"tenstorrent": ["dir"], **expected}, cards
    # The run's own hugepage directory only: the program's file, and none of the host directory's other files.
    pages = {key: kind for key, kind in dev.items() if key == "hugepages-1G" or key.startswith("hugepages-1G/")}
    assert seen["probe"] == "created" and pages == {"hugepages-1G": ["dir"], f"hugepages-1G/{PROBE}": ["file"]}, seen
    assert sorted(os.listdir(host_dir)) == [PROBE], "the file reached the host directory"
    # Nothing else under /dev: the private /dev's own entries, as for every run (R1).
    rest = {key: kind for key, kind in dev.items() if key not in cards and key not in pages}
    nodes = {key for key, kind in rest.items() if kind[0] in ("chr", "blk")}
    links = {key: kind[1] for key, kind in rest.items() if kind[0] == "link"}
    assert nodes == DEV_NODES and {key for key, kind in rest.items() if kind[0] == "dir"} == DEV_DIRS, sorted(rest)
    assert set(links) <= set(DEV_LINKS) and all(DEV_LINKS[key] == target for key, target in links.items()), links
    assert set(rest) == nodes | DEV_DIRS | set(links), sorted(rest)
    # /proc/mounts shows the bind as UMD accepts it (umd hugepage.cpp:109-141): source nodev or hugetlbfs, mount point
    # exactly /dev/hugepages-1G, type hugetlbfs, 1 GiB pages.
    lines = seen["proc_mounts"]
    assert len(lines) == 1, lines
    assert lines[0][0] in ("nodev", "hugetlbfs") and lines[0][1:3] == [HUGEPAGE_POINT, "hugetlbfs"], lines
    assert "pagesize=1024M" in lines[0][3].split(","), lines
    # Writable, keeping the host mount's nosuid and nodev, adding noexec, and rooted at the run's own directory.
    mountinfo = seen["mountinfo"]
    huge = top_mount(mountinfo, HUGEPAGE_POINT)
    assert (huge[4], fstype(huge), huge[3]) == (HUGEPAGE_POINT, "hugetlbfs", root), huge
    assert {"rw", "nosuid", "nodev", "noexec"} <= set(huge[5].split(",")), huge
    # Everything else exposed is read-only, and every other writable mount is one of the setup's own.
    readonly = ["/dev", "/sys/bus", "/sys/bus/pci", "/sys/bus/pci/devices", SLOTS, DRIVER_VERSION, PAGE_COUNT]
    readonly += [target for _node, target in CARDS] + [target for _source, target in pci]
    assert [path for path in readonly if "ro" not in top_mount(mountinfo, path)[5].split(",")] == [], readonly
    writable = [fields for fields in mountinfo if "rw" in fields[5].split(",")]
    strays = [
        fields
        for fields in writable
        if fields is not huge and not (fstype(fields) in SETUP_FSTYPES and source(fields).startswith("lassi-"))
    ]
    assert strays == [], strays
    # Nothing else under /sys/bus/pci: the slot list and the cards' directories, each the host's.
    listed = seen["listings"]
    assert (listed["/sys/bus"], listed["/sys/bus/pci"]) == (["pci"], ["devices", "slots"]), listed
    assert listed["/sys/bus/pci/devices"] == sorted(target.rsplit("/", 1)[1] for _source, target in pci), listed
    assert listed[SLOTS] == sorted(os.listdir(SLOTS)), listed[SLOTS]
    for directory, target in pci:
        assert listed[target] == sorted(os.listdir(directory)), target
    assert listed["/sys/class"] == [], "no class directory, the tenstorrent class included"
    # The driver version and the 1 GiB page count read as on the host (or are missing as there).
    assert seen["texts"] == {path: host_text(path) for path in (DRIVER_VERSION, PAGE_COUNT)}, seen["texts"]


def test_the_reset_filter_refuses_the_reset_request_only_when_asked(
    sandbox: ModuleType, layout: Layout, hugepage_dir: tuple[Path, str]
) -> None:
    pci = need_standins(1)
    host_dir, _root = hugepage_dir
    seen = {}
    for reset_filter in (True, False):
        exposure = tt_exposure(sandbox, str(host_dir), reset_filter=reset_filter, cards=CARDS[:1], pci=pci)
        seen[reset_filter] = run_json(sandbox, spec_with(sandbox, layout, exposure), IOCTLS)
    on, off = seen[True], seen[False]
    assert on["reset"] == on["reset_high_bit"] == "EPERM", on
    assert on["below"] != "EPERM", on
    # Off, the calls reach the kernel, which refuses an ioctl /dev/null does not have: the same error for all three.
    assert off["reset"] == off["reset_high_bit"] == off["below"] == on["below"] != "EPERM", (on, off)


def test_a_plain_directory_cannot_stand_in_for_the_hugepage_mount(sandbox: ModuleType, layout: Layout) -> None:
    pci = need_standins(1)
    standin = layout.base / "hugepages"
    standin.mkdir()
    placeholder = f"{HUGEPAGE_POINT}/lassi-standin-{uuid.uuid4().hex}"
    exposure = tt_exposure(sandbox, placeholder, reset_filter=False, cards=CARDS[:1], pci=pci)
    swapped: list[int] = []

    def runner(argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Swap the placeholder for the plain directory, refuse any path left under the host's hugepage mount, run."""
        swapped.append(list(argv).count(placeholder))
        command = [str(standin) if part == placeholder else part for part in argv]
        assert not [part for part in command if part.startswith(HUGEPAGE_POINT + "/")], "the host mount stays out"
        return capped_runner(command, cwd, timeout_s)

    program = ["sh", "-c", 'touch "/dev/hugepages-1G/$1"; touch "$2/ran"; echo RAN', "sh", PROBE, str(layout.workdir)]
    with pytest.raises(sandbox.SandboxUnavailableError, match=f"writable host mount: {HUGEPAGE_POINT} "):
        sandbox.Sandbox(runner=runner).run(spec_with(sandbox, layout, exposure), program, PYTHON)
    assert swapped and swapped[0] >= 1, swapped
    # The setup failed closed: the program never ran, so neither the stand-in nor the workdir holds its files.
    assert list(standin.iterdir()) == [] and not (layout.workdir / "ran").exists()


def test_a_run_without_the_exposure_sees_no_card_and_no_hugepage_directory(sandbox: ModuleType, layout: Layout) -> None:
    # Guard: SETUP_SCRIPT is byte for byte the measured one, so this holds before and after task P17.15.
    for environment in (None, dict(PROGRAM_ENVIRONMENT)):
        spec = sandbox.SandboxSpec(workdir=layout.workdir, hidden_roots=layout.hidden, environment=environment)
        seen = run_json(sandbox, spec, PLAIN_REPORT)
        assert seen == {"dev": {"/dev/tenstorrent": False, HUGEPAGE_POINT: False}, "bus": []}, seen
