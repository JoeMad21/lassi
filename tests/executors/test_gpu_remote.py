"""Remote tests for the sandbox's device access, with harmless stand-in nodes (task P17.5).

Bible: Sandbox, Execution Backends (gpu rows), Agent Rules 6 and 7;
plans/p17-portable.md, task P17.5 (a listed node is visible in the sandbox
and an unlisted one is not; no AMD node is opened) and its Constraints
(OQ-002: nothing opens /dev/kfd, a /dev/dri node, or an NVIDIA node, a test
included; sandbox device tests use a harmless stand-in node).

These run real sandboxes through Sandbox.run with hand-made SandboxSpecs
(Agent Rule 6), never through the gpu executor, whose real exposure would
name the host's GPU nodes. They are marked `remote` and skip on a host that
cannot run the sandbox (the host checks of test_sandbox_remote.py, restated
here, since tests are not a package); with LASSI_REQUIRE_SANDBOX=1 such a
host fails them instead, so a silent skip cannot pass for evidence. Run them
with `PYTHONIOENCODING=utf-8 rx run -- 'LASSI_REQUIRE_SANDBOX=1 uv run pytest
-q -m remote tests/executors/test_gpu_remote.py'`.

The stand-ins:
- nodes: the host's /dev/zero and /dev/full, character devices every Linux
  host has and that drive no hardware, bound at /dev/lassi-standin/a and
  /dev/lassi-standin/b, targets that name no GPU node;
- a /sys directory: the host's /sys/devices/system/cpu bound at
  /sys/devices/virtual/lassi-standin/topology, under an entry the setup
  covers, as the KFD topology lies under /sys/devices/virtual.

Run A lists node a and the /sys stand-in, with ROCR_VISIBLE_DEVICES and
HIP_VISIBLE_DEVICES set; run B lists node b alone. Each sees its listed node,
with the host node's device number, and not the other; A reads 8 zero bytes
from its node; the /sys stand-in shows the host directory's entries in A
only, on a read-only mount, and the cover /sys/devices/virtual, which A's
setup reopened for the bind, is read-only again in every run; /dev/kfd,
/dev/dri, and /dev/nvidiactl are absent in both; the visibility variables
reach A only. A run without an exposure sees no stand-in, an empty
/sys/devices/virtual, and no visibility variable, even when the caller's
environment holds them.

Why no AMD node is opened: the only device node any test opens is the
stand-in for /dev/zero, read inside run A. Inside, /dev/kfd, /dev/dri, and
/dev/nvidiactl are only lstat-ed in the sandbox's private /dev, a tmpfs where
they do not exist, so no host node is reached. On the host the tests stat
/dev/zero and /dev/full and list /sys/devices/system/cpu, and read no KFD,
DRM, or NVIDIA path. No file is written outside the temp directory each test
creates under $LASSI_SCRATCH/tmp. No value here is a measurement.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import pytest

from lassi.core.interfaces import Limits
from lassi.executors.sandbox import SANDBOX_PATH

TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk")
# Room for a python3 interpreter.
PYTHON = Limits(wall_s=20.0, memory_mb=256, cpus=1)
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
# The stand-in nodes (host source, path inside) and the stand-in /sys directory.
NODE_A = ("/dev/zero", "/dev/lassi-standin/a")
NODE_B = ("/dev/full", "/dev/lassi-standin/b")
SYS_SOURCE = "/sys/devices/system/cpu"
SYS_TARGET = "/sys/devices/virtual/lassi-standin/topology"
# The GPU paths the program only lstat-s inside the private /dev, and the names only the gpu executor sets.
GPU_PATHS = ("/dev/kfd", "/dev/dri", "/dev/nvidiactl")
VISIBILITY = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
PROGRAM_ENVIRONMENT = {"PATH": SANDBOX_PATH, "LANG": "C.UTF-8", "TMPDIR": "/tmp"}


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
    base = Path(tempfile.mkdtemp(prefix="lassi-gpu-test.", dir=parent))
    try:
        workdir = base / "build"
        workdir.mkdir()
        yield Layout(base=base, workdir=workdir, hidden=hidden_default())
    finally:
        shutil.rmtree(base)


# Prints what the program sees: each stand-in target's lstat, 8 bytes of the node argv[1] names ("" for none), the
# stand-in directory, the GPU paths (lstat only), /sys/devices/virtual and whether its top mount (the cover) is
# read-only, the /sys target argv[2] and whether its top mount is read-only, and the variables argv[3] names (JSON)
# from its environment.
REPORT = """from __future__ import annotations
import json, os, stat, sys
def node(path: str) -> object:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return "absent"
    if stat.S_ISCHR(info.st_mode):
        return ["chr", os.major(info.st_rdev), os.minor(info.st_rdev)]
    return ["other", stat.S_IFMT(info.st_mode)]
def listing(path: str) -> object:
    try:
        return sorted(os.listdir(path))
    except FileNotFoundError:
        return "absent"
def readonly(point: str) -> object:
    found = None
    with open("/proc/self/mountinfo") as mountinfo:
        for line in mountinfo:
            fields = line.split()
            if fields[4] == point:
                found = "ro" in fields[5].split(",")
    return found
report = {"nodes": {path: node(path) for path in ("/dev/lassi-standin/a", "/dev/lassi-standin/b")}, "read": ""}
if sys.argv[1]:
    with open(sys.argv[1], "rb") as source:
        report["read"] = source.read(8).hex()
report["standin"] = listing("/dev/lassi-standin")
report["gpu"] = {path: os.path.lexists(path) for path in ("/dev/kfd", "/dev/dri", "/dev/nvidiactl")}
report["virtual"] = listing("/sys/devices/virtual")
report["cover_readonly"] = readonly("/sys/devices/virtual")
report["sys"] = listing(sys.argv[2])
report["sys_readonly"] = readonly(sys.argv[2])
report["environ"] = {name: os.environ.get(name) for name in json.loads(sys.argv[3])}
print(json.dumps(report))
"""


def exposure(sandbox: ModuleType, nodes: Sequence[tuple[str, str]], sys_dirs: Sequence[tuple[str, str]] = ()) -> Any:
    """Return a DeviceExposure of DeviceBinds from (host source, path inside) pairs."""
    bind = sandbox.DeviceBind
    return sandbox.DeviceExposure(
        nodes=tuple(bind(PurePosixPath(source), PurePosixPath(target)) for source, target in nodes),
        sys_dirs=tuple(bind(PurePosixPath(source), PurePosixPath(target)) for source, target in sys_dirs),
    )


def report(
    sandbox: ModuleType, layout: Layout, devices: Any, environment: Mapping[str, str] | None, read: str
) -> dict[str, Any]:
    """Run REPORT through Sandbox.run under a spec with `devices` (None: no exposure) and return what it printed."""
    values: dict[str, Any] = {"workdir": layout.workdir, "hidden_roots": layout.hidden, "environment": environment}
    if devices is not None:
        values["devices"] = devices
    spec = sandbox.SandboxSpec(**values)
    argv = [PY, "-c", REPORT, read, SYS_TARGET, json.dumps(VISIBILITY)]
    result = sandbox.Sandbox().run(spec, argv, PYTHON)
    assert result.returncode == 0, result
    return json.loads(result.stdout)


def host_node(path: str) -> list[object]:
    """Return ["chr", major, minor] of a host node, read with stat (nothing is opened)."""
    info = os.stat(path)
    return ["chr", os.major(info.st_rdev), os.minor(info.st_rdev)]


def test_a_listed_stand_in_node_is_visible_and_an_unlisted_one_is_not(sandbox: ModuleType, layout: Layout) -> None:
    need(os.path.isdir(SYS_SOURCE), f"the host has no {SYS_SOURCE} to stand in for a /sys directory")
    visible = {**PROGRAM_ENVIRONMENT, "ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0"}
    first = report(sandbox, layout, exposure(sandbox, [NODE_A], [(SYS_SOURCE, SYS_TARGET)]), visible, NODE_A[1])
    second = report(sandbox, layout, exposure(sandbox, [NODE_B]), dict(PROGRAM_ENVIRONMENT), "")
    a, b = NODE_A[1], NODE_B[1]
    assert first["nodes"] == {a: host_node(NODE_A[0]), b: "absent"}, first
    assert second["nodes"] == {a: "absent", b: host_node(NODE_B[0])}, second
    assert (first["standin"], second["standin"]) == (["a"], ["b"])
    assert first["read"] == "00" * 8, "the listed node is the host's /dev/zero"
    for seen in (first, second):
        assert seen["gpu"] == {path: False for path in GPU_PATHS}, seen["gpu"]
    assert (first["virtual"], second["virtual"]) == (["lassi-standin"], [])
    assert (first["cover_readonly"], second["cover_readonly"]) == (True, True), "the cover is read-only again"
    assert first["sys"] == sorted(os.listdir(SYS_SOURCE)) and first["sys_readonly"] is True, first
    assert (second["sys"], second["sys_readonly"]) == ("absent", None), second
    expected = {name: None for name in VISIBILITY}
    assert first["environ"] == {**expected, "ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0"}
    assert second["environ"] == expected


def test_a_run_without_an_exposure_sees_no_stand_in(
    sandbox: ModuleType, layout: Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The caller's visibility variables never reach the program: env -i starts the sandbox, and only an exposure
    # lets a spec carry them (SETUP_SCRIPT is byte-identical, so P0.16's R1 test still covers /dev for native runs).
    for name in VISIBILITY:
        monkeypatch.setenv(name, "0")
    for environment in (None, dict(PROGRAM_ENVIRONMENT)):
        seen = report(sandbox, layout, None, environment, "")
        assert seen["nodes"] == {NODE_A[1]: "absent", NODE_B[1]: "absent"}, seen
        assert seen["standin"] == "absent" and seen["gpu"] == {path: False for path in GPU_PATHS}, seen
        assert seen["virtual"] == [] and seen["cover_readonly"] is True, seen
        assert (seen["sys"], seen["sys_readonly"]) == ("absent", None), seen
        assert seen["environ"] == {name: None for name in VISIBILITY}, seen
