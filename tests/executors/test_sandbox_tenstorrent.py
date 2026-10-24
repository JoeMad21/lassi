"""Tests for the sandbox's Tenstorrent exposure: every Wormhole card of the cluster in one run (task P17.15).

Bible: Execution Backends (the tt_silicon row), Sandbox, Agent Rules 6, 7,
and 9. The owner's OQ-072 answer gives a silicon run every card in the
cluster; OQ-075 is ANSWERED (its recommendation: (1b) the run's own
hugepage directory, (4a) the reset filter), and the reset setting stays
explicit, with no default. The evidence is
plans/spikes/tt-silicon-executor.md: Findings 1 to 3 say what UMD at the
pin (0450f1be, under tt-metal 5280a9cf) opens and reads, and Finding 5 and
section 6 say what the sandbox must change.

The interface these tests assume, which task P17.15 adds to
lassi.executors.sandbox:

- TenstorrentExposure, a frozen dataclass beside DeviceExposure that
  SandboxSpec.devices carries, with four fields in this order:
  - nodes: DeviceBinds whose targets are numbered nodes /dev/tenstorrent/<n>,
    each bound at that path in the private /dev (a source may be any path
    under /dev: the executor of P17.16 is to bind each host node at its own
    path, and the remote tests bind harmless stand-ins there);
  - pci_dirs: DeviceBinds of each card's PCI sysfs directory (the directory
    /sys/class/tenstorrent/tenstorrent!<n>/device resolves to, a path under
    /sys whose last segment is the card's address), each bound read-only at
    /sys/bus/pci/devices/<address>;
  - hugepages: the run's own hugepage directory, strictly under the host's
    /dev/hugepages-1G, bound writable at /dev/hugepages-1G;
  - reset_filter: a bool with no default, True only when the caller asks the
    seccomp filter to refuse the driver's reset ioctl (OQ-075 (4)).
  A run with one also gets /sys/bus/pci/slots, bound read-only at its own
  path. /sys/module/tenstorrent/version and the 1 GiB nr_hugepages file stay
  in view on the host's /sys, read-only as in every run.
- confine_program(reset_filter), the confinement program a run embeds:
  confine_program(False) is CONFINE_PROGRAM byte for byte, and
  confine_program(True) adds only a rule that fails ioctl(2) (x86_64
  syscall 16) whose request is _IO(0xFA, 6) = 0xFA06, the driver's reset
  ioctl (umd pcie/ioctl.h:24; spike, Finding 5 item 1), with EPERM. The rule
  compares the request's low 32 bits, since the kernel reads the request as
  an unsigned int.

What is checked, on Windows or any host with sh and awk:

- Runs without the exposure keep their scripts: SETUP_SCRIPT and the gpu
  executor's DEVICE_SETUP_SCRIPT are pinned by sha256, and a spec without
  devices, or with a DeviceExposure, builds the same script as before.
- The exposure's fields, and the refusals, each a ValueError that names the
  field: an empty node list, a relative path, a node outside the numbered
  /dev/tenstorrent/<n> nodes, a PCI directory outside /sys, bound outside
  /sys/bus/pci/devices/<address>, or whose last segment is not that
  address, more PCI directories than the /sys/bus cover holds, a hugepage
  directory that is not strictly under /dev/hugepages-1G, and a reset
  setting that is not a bool.
- The setup's calls, from its script run under the stub harness of
  test_sandbox.py (nothing is mounted and no program runs): each node bound
  on a file at its own path in the private /dev before that /dev turns
  read-only and moves onto /dev; the hugepage directory bound on a directory
  there, its owner checked, then made writable, keeping nosuid and nodev and
  adding noexec, before the mount check, and the setup stopped (exit 2)
  right after the bind when the running user does not own the directory;
  the slot list and each PCI directory staged while the host's /sys is in
  view and bound below the /sys/bus cover, which is read-only again before
  the check; that cover's inode cap raised, read-only, before the first
  directory below it, so it holds all eight cards' directories (the 8
  inodes every cover gets hold four); and every other call the same as
  SETUP_SCRIPT's. The stub mkdir makes nothing, so each stub run points the
  owner check at a directory the test made, or at a path the user does not
  own; the shell's own test builtin runs it.
- The run's mount check, its recorded awk call replayed with the awk on
  PATH against SYNTHETIC mountinfo lines: it accepts exactly one writable
  hugetlbfs mount at /dev/hugepages-1G whose root is the run's own directory
  and whose super options hold pagesize=1024M, and refuses a wrong root
  (compared as a string, never as a number), a wrong point, another type,
  another page size, a second such mount, and every writable mount the
  plain check refuses.
- The reset filter, run through test_sandbox.py's classic-BPF interpreter
  (nothing is installed): off, the program is CONFINE_PROGRAM; on, the reset
  request fails with EPERM and nothing else changes, and the run's script
  embeds the program its setting names.

The PCI addresses are the eight lspci listed on alpha01 (rx
20261021-201421-exec-4ea0, OQ-070's evidence); card 0's sysfs path is the
one spike probe 32e2 printed, and the other cards' parent paths are
SYNTHETIC, as are the hugepage directory names and every mountinfo line.
Paths are only strings: no file under /dev or /sys is read or touched. No
value here is a measurement.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import os
import re
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import pytest
from test_sandbox import (
    ACCEPTED_MOUNTINFO,
    ARCH_I386,
    ARCH_X86_64,
    DONE_LINE,
    ONE_SECOND,
    PROGRAM,
    READY_LINE,
    REFUSED_MOUNTINFO,
    SECCOMP_ALLOW,
    SECCOMP_EPERM,
    SETUP_SCRIPT_SHA256,
    WORK,
    X32_BIT,
    StubRun,
    cover_remounts,
    index_of,
    moves,
    only,
    remounts,
    run_filter,
    run_mount_check,
    run_setup_with_stubs,
    sample_spec,
    script_of,
    the_check_call,
    the_program_call,
    tree_paths,
    typed_mounts,
    without_stub_path,
)

# DEVICE_SETUP_SCRIPT at 8342709, the gpu executor's script (task P17.5): runs with a DeviceExposure keep these bytes.
DEVICE_SETUP_SCRIPT_SHA256 = "e0407bc65ed4459ce95b3838d11a8914caa4777d97a4118a65b8d0ae01a11a6b"
DEVICE_SETUP_SCRIPT_LENGTH = 11711
# The eight cards' PCI addresses on alpha01 (lspci, rx 20261021-201421-exec-4ea0).
ADDRESSES = (
    "0000:e8:00.0", "0000:e9:00.0", "0000:ea:00.0", "0000:eb:00.0",
    "0000:f1:00.0", "0000:f2:00.0", "0000:f3:00.0", "0000:f4:00.0",
)  # fmt: skip
# Card 0's PCI directory on alpha01 (spike probe 32e2) without its last segment; the others' parent is SYNTHETIC.
CARD0_PARENT = "/sys/devices/pci0000:e2/0000:e2:01.1/0000:e3:00.0/0000:e4:00.0/0000:e5:00.0/0000:e6:01.0"
SYNTHETIC_PARENT = "/sys/devices/pci0000:e2/0000:e2:01.1"
# Each card's node at its own path, and its PCI directory at /sys/bus/pci/devices/<address>, as (source, target).
NODES = tuple((f"/dev/tenstorrent/{card}", f"/dev/tenstorrent/{card}") for card in range(len(ADDRESSES)))
PCI_DIRS = tuple(
    (f"{CARD0_PARENT if card == 0 else SYNTHETIC_PARENT}/{address}", f"/sys/bus/pci/devices/{address}")
    for card, address in enumerate(ADDRESSES)
)
SLOTS = "/sys/bus/pci/slots"
BUS_COVER = "/sys/bus"
DRIVER_VERSION = "/sys/module/tenstorrent/version"
PAGE_COUNT = "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages"
# UMD's fixed hugepage directory (umd hugepage.cpp:28), the run's own directory under it (SYNTHETIC names), and the
# root each shows in mountinfo once bound there, the host's hugetlbfs being mounted at /dev/hugepages-1G with root /.
HUGEPAGE_POINT = "/dev/hugepages-1G"
HUGEPAGES = "/dev/hugepages-1G/lassi-run-0001"
HUGEPAGE_ROOT = "/lassi-run-0001"
NESTED_HUGEPAGES = "/dev/hugepages-1G/lassi/run-0001"
NESTED_ROOT = "/lassi/run-0001"
# The hugepage step's owner check, between its bind and the remount that makes the bind writable (noexec added).
OWNER_CHECK = "[ -O /tmp/dev/hugepages-1G ] || exit 2\n"
HUGEPAGE_BIND = 'mount --bind "$hugepages" /tmp/dev/hugepages-1G\n'
HUGEPAGE_REMOUNT = "mount -o remount,bind,rw,nosuid,nodev,noexec /tmp/dev/hugepages-1G\n"
# The x86_64 ioctl syscall and the driver's reset request _IO(0xFA, 6) (umd pcie/ioctl.h:24).
NR_IOCTL = 16
RESET_IOCTL = 0xFA06


@pytest.fixture
def sandbox() -> ModuleType:
    """Return the lassi.executors.sandbox module (imported here so each test shows a missing module clearly)."""
    from lassi.executors import sandbox

    return sandbox


def tt_attribute(sandbox: ModuleType, name: str) -> Any:
    """Return sandbox.<name>, failing the test clearly while task P17.15 has not added it."""
    if not hasattr(sandbox, name):
        pytest.fail(f"lassi.executors.sandbox has no {name} yet (task P17.15)")
    return getattr(sandbox, name)


def bind_of(sandbox: ModuleType, item: object) -> object:
    """Return a DeviceBind for a (source, target) pair of strings; any other item is passed through unchanged."""
    if isinstance(item, tuple) and len(item) == 2 and all(isinstance(part, str) for part in item):
        return sandbox.DeviceBind(PurePosixPath(item[0]), PurePosixPath(item[1]))
    return item


def make_tt(
    sandbox: ModuleType,
    *,
    nodes: Sequence[object] = NODES,
    pci_dirs: Sequence[object] = PCI_DIRS,
    hugepages: object = HUGEPAGES,
    reset_filter: object = False,
) -> Any:
    """Return a TenstorrentExposure; strings become PurePosixPaths and (source, target) pairs DeviceBinds."""
    exposure = tt_attribute(sandbox, "TenstorrentExposure")
    return exposure(
        nodes=tuple(bind_of(sandbox, item) for item in nodes),
        pci_dirs=tuple(bind_of(sandbox, item) for item in pci_dirs),
        hugepages=PurePosixPath(hugepages) if isinstance(hugepages, str) else hugepages,
        reset_filter=reset_filter,
    )


def gpu_exposure(sandbox: ModuleType) -> Any:
    """Return a DeviceExposure shaped as the gpu executor's for one AMD GPU (strings only; nothing is touched)."""
    bind = sandbox.DeviceBind
    topology = PurePosixPath("/sys/devices/virtual/kfd/kfd/topology")
    nodes = tuple(bind(PurePosixPath(path), PurePosixPath(path)) for path in ("/dev/kfd", "/dev/dri/renderD128"))
    return sandbox.DeviceExposure(nodes=nodes, sys_dirs=(bind(topology, topology),))


def sha256(text: str) -> str:
    """Return the sha256 of `text` encoded as ASCII."""
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def sys_binds() -> tuple[tuple[str, str], ...]:
    """Return every /sys bind a run with the exposure gets: the slot list, then each card's PCI directory."""
    return ((SLOTS, SLOTS), *PCI_DIRS)


# ---------------------------------------------------------------------------
# The setup script under stubs: one plain run and one run with the exposure, cached per setting


@dataclass(frozen=True)
class Runs:
    """A stub run of SETUP_SCRIPT and one of the run's script with the exposure, on the same layout."""

    plain: StubRun
    exposed: StubRun
    command: list[str]


@dataclass(frozen=True)
class Plain:
    """The plain script's stub run and the workdir every cached run shares, so their calls compare."""

    run: StubRun
    workdir: Path


_RUNS: dict[tuple[bool, str], Runs] = {}
_PLAIN: list[Plain] = []


def stub_layout_of(command: list[str], workdir: Path) -> list[str]:
    """Return the positional layout and argv of a sandbox command, with its workdir replaced by a real one."""
    script = script_of(command)
    start = command.index(script)
    assert command[start + 1 : start + 3] == ["sh", str(WORK)], command[start + 1 : start + 3]
    return [workdir.as_posix(), *command[start + 3 :]]


def stub_run_of(
    sandbox: ModuleType, command: list[str], base: Path, workdir: Path, *, owned: str | None = None
) -> StubRun:
    """Run a sandbox command's script under stubs in the fresh directory `base` (the program exits 7).

    The stub mkdir makes nothing, so a script with the hugepage step's
    OWNER_CHECK has it test `owned` instead: by default a directory made
    here, which the user owns.
    """
    base.mkdir()
    layout = stub_layout_of(command, workdir)
    script = script_of(command)
    if OWNER_CHECK in script:
        if owned is None:
            (base / "owned").mkdir()
            owned = (base / "owned").as_posix()
        assert "'" not in owned and script.count(OWNER_CHECK) == 1, owned
        script = script.replace(OWNER_CHECK, f"[ -O '{owned}' ] || exit 2\n")
    return run_setup_with_stubs(sandbox, base, layout, program_status=7, script=script)


def stub_runs(
    sandbox: ModuleType, factory: pytest.TempPathFactory, *, reset_filter: bool = False, hugepages: str = HUGEPAGES
) -> Runs:
    """Run the plain script and the exposure's script under stubs on one workdir; cached per setting.

    Every run uses the same workdir path, so their calls compare.
    """
    if not _PLAIN:
        base = factory.mktemp("tt-stubs")
        workdir = base / "work"
        workdir.mkdir()
        plain = sandbox.sandbox_command(sample_spec(sandbox), PROGRAM, ONE_SECOND)
        _PLAIN.append(Plain(run=stub_run_of(sandbox, plain, base / "plain", workdir), workdir=workdir))
    key = (reset_filter, hugepages)
    if key not in _RUNS:
        spec = sample_spec(sandbox, devices=make_tt(sandbox, reset_filter=reset_filter, hugepages=hugepages))
        command = sandbox.sandbox_command(spec, PROGRAM, ONE_SECOND)
        exposed = stub_run_of(sandbox, command, factory.mktemp("tt-exposed") / "run", _PLAIN[0].workdir)
        _RUNS[key] = Runs(plain=_PLAIN[0].run, exposed=exposed, command=command)
    return _RUNS[key]


def private_dev(calls: list[list[str]]) -> str:
    """Return the staging point of the private /dev: where the lassi-dev tmpfs is mounted."""
    return only(typed_mounts(calls, "tmpfs", "lassi-dev"), "the lassi-dev tmpfs")[1][5]


def sealing(calls: list[list[str]], dev: str) -> tuple[int, int]:
    """Return the indices of the private /dev's read-only remount and of its move onto /dev."""
    readonly = [index for index, options in remounts(calls, dev) if "ro" in options]
    moved = [index for index, source, target in moves(calls, "--move") if (source, target) == (dev, "/dev")]
    assert len(readonly) == 1 and len(moved) == 1, calls
    return readonly[0], moved[0]


def writable_reopens(calls: list[list[str]], points: Iterable[str]) -> list[tuple[int, list[str]]]:
    """Return (index, call) of each call that makes one of `points` writable.

    That is a `mount ... -o remount,...,rw,... <point>` call, or a python3
    call naming the point (a mount_setattr that clears the read-only flag).
    """
    wanted = set(points)
    found = []
    for index, call in enumerate(calls):
        if call[0] == "mount" and call[-1] in wanted and "-o" in call[:-1]:
            if {"remount", "rw"} <= set(call[call.index("-o") + 1].split(",")):
                found.append((index, call))
        elif call[0] == "python3" and any(point in part for part in call[1:] for point in wanted):
            found.append((index, call))
    return found


def staged(calls: list[list[str]], source: str) -> tuple[int, str]:
    """Return (index, staging point) of the one `mount --bind <source> <stage>` call."""
    found = [(index, target) for index, bound, target in moves(calls, "--bind") if bound == source]
    assert len(found) == 1, (source, found)
    return found[0]


def cover_made(calls: list[list[str]], cover: str) -> int:
    """Return the index of the call that mounts the read-only lassi-sys cover on `cover`, or -1 when none does.

    The setup covers only the /sys directories the host has, and a stub run
    sees the test host's: a Linux host has /sys/bus, Windows has none.
    """
    made = [
        index
        for index, args in typed_mounts(calls, "tmpfs", "lassi-sys")
        if args[5] == cover and "remount" not in args[3].split(",")
    ]
    assert len(made) <= 1, (cover, made)
    return made[0] if made else -1


def tenstorrent_calls(calls: list[list[str]], dev: str) -> set[int]:
    """Return the indices of the calls the exposure adds to a stub run.

    They are each node's bind (with its file and its parent directory in the
    private /dev), the hugepage directory's mkdir, bind, and any call that
    makes it writable, each /sys source's staging bind (and the staging
    directories), each /sys target's mkdir and bind, every remount of the
    /sys/bus cover, and any call naming the driver version or the 1 GiB page
    count.
    """
    nodes = {dev + target[len("/dev") :] for _source, target in NODES}
    huge = {dev + HUGEPAGE_POINT[len("/dev") :], HUGEPAGE_POINT}
    stages = {staged(calls, source)[1] for source, _target in sys_binds()}
    targets = {target for _source, target in sys_binds()}
    made = {dev + "/tenstorrent", *huge}
    made |= set().union(*(tree_paths(stage, "/tmp") for stage in stages))
    made |= set().union(*(tree_paths(target, BUS_COVER) for target in targets))
    named = nodes | huge | stages | targets | made | {HUGEPAGES, DRIVER_VERSION, PAGE_COUNT}
    named |= {source for source, _target in sys_binds()}
    marked = set()
    for index, call in enumerate(calls):
        args = call[1:]
        paths = [part for part in args if not part.startswith("-")]
        if call[0] == "mkdir" and paths and set(paths) <= made:
            marked.add(index)
        elif set(args) & named or cover_remounts([call], BUS_COVER):
            marked.add(index)
        elif call[0] == "python3" and any(point in part for part in args for point in huge):
            marked.add(index)
    return marked


# ---------------------------------------------------------------------------
# Runs without the exposure keep their scripts byte for byte


def test_runs_without_the_tenstorrent_exposure_keep_their_scripts_byte_for_byte(sandbox: ModuleType) -> None:
    # Guard: passes before and after task P17.15. Native, ttsim, compile, and gpu runs keep what was measured.
    assert sha256(sandbox.SETUP_SCRIPT) == SETUP_SCRIPT_SHA256
    assert sha256(sandbox.DEVICE_SETUP_SCRIPT) == DEVICE_SETUP_SCRIPT_SHA256
    assert len(sandbox.DEVICE_SETUP_SCRIPT) == DEVICE_SETUP_SCRIPT_LENGTH
    for spec in (sample_spec(sandbox), sample_spec(sandbox, devices=None)):
        assert script_of(sandbox.sandbox_command(spec, PROGRAM, ONE_SECOND)) == sandbox.SETUP_SCRIPT
    gpu = sample_spec(sandbox, devices=gpu_exposure(sandbox))
    assert script_of(sandbox.sandbox_command(gpu, PROGRAM, ONE_SECOND)) == sandbox.DEVICE_SETUP_SCRIPT


def test_a_tenstorrent_run_has_a_script_of_its_own_and_the_same_command_around_it(
    sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in sandbox.PASSED_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    plain = sandbox.sandbox_command(sample_spec(sandbox), PROGRAM, ONE_SECOND)
    command = sandbox.sandbox_command(sample_spec(sandbox, devices=make_tt(sandbox)), PROGRAM, ONE_SECOND)
    script = script_of(command)
    assert script not in (sandbox.SETUP_SCRIPT, sandbox.DEVICE_SETUP_SCRIPT)
    assert sandbox.SETUP_SCRIPT not in command and sandbox.DEVICE_SETUP_SCRIPT not in command
    start = command.index(script)
    assert command[:start] == plain[: plain.index(sandbox.SETUP_SCRIPT)], "only the script and layout differ"
    assert command[start + 1 : start + 3] == ["sh", str(WORK)] and command[-len(PROGRAM) :] == PROGRAM
    assert all(isinstance(part, str) for part in command)
    assert script.isascii() and "$(" not in script and "`" not in script, "no command or arithmetic substitution"
    assert not re.search(r"/s?bin/", script), "every command comes through PATH"


# ---------------------------------------------------------------------------
# TenstorrentExposure: the fields, and the refusals, each naming its field


def test_the_exposure_is_a_frozen_dataclass_whose_reset_setting_has_no_default(sandbox: ModuleType) -> None:
    exposure = tt_attribute(sandbox, "TenstorrentExposure")
    fields = dataclasses.fields(exposure)
    assert [item.name for item in fields] == ["nodes", "pci_dirs", "hugepages", "reset_filter"]
    reset = next(item for item in fields if item.name == "reset_filter")
    assert reset.default is dataclasses.MISSING and reset.default_factory is dataclasses.MISSING, "explicit setting"
    node = sandbox.DeviceBind(PurePosixPath(NODES[0][0]), PurePosixPath(NODES[0][1]))
    pci = sandbox.DeviceBind(PurePosixPath(PCI_DIRS[0][0]), PurePosixPath(PCI_DIRS[0][1]))
    with pytest.raises(TypeError):
        exposure(nodes=(node,), pci_dirs=(pci,), hugepages=PurePosixPath(HUGEPAGES))
    made = exposure(nodes=[node], pci_dirs=[pci], hugepages=PurePosixPath(HUGEPAGES), reset_filter=False)
    assert (made.nodes, made.pci_dirs) == ((node,), (pci,)), "both kept as tuples"
    with pytest.raises(dataclasses.FrozenInstanceError):
        made.reset_filter = True


def test_a_spec_keeps_an_all_card_exposure(sandbox: ModuleType) -> None:
    for reset_filter in (False, True):
        exposure = make_tt(sandbox, reset_filter=reset_filter)
        spec = sample_spec(sandbox, devices=exposure)
        assert spec.devices == exposure and spec.devices.reset_filter is reset_filter
        assert [bind.target.as_posix() for bind in spec.devices.nodes] == [target for _source, target in NODES]
    # One card in a nested run directory, and the remote tests' stand-ins: harmless nodes bound at card paths, and a
    # host PCI device that is no card, bound at its own address (SYNTHETIC here).
    nested = make_tt(sandbox, nodes=NODES[:1], pci_dirs=PCI_DIRS[:1], hugepages=NESTED_HUGEPAGES)
    standins = make_tt(
        sandbox,
        nodes=[("/dev/zero", "/dev/tenstorrent/0"), ("/dev/full", "/dev/tenstorrent/1")],
        pci_dirs=[("/sys/devices/pci0000:00/0000:00:01.0", "/sys/bus/pci/devices/0000:00:01.0")],
    )
    for exposure in (nested, standins):
        assert sample_spec(sandbox, devices=exposure).devices == exposure


GOOD_PCI = PCI_DIRS[0]
# Exposures the spec refuses: case -> (the overrides of make_tt's arguments, the field the ValueError names).
BAD_FIELDS: dict[str, tuple[dict[str, object], str]] = {
    "no-node": ({"nodes": ()}, "nodes"),
    "relative-node-target": ({"nodes": [("/dev/tenstorrent/0", "dev/tenstorrent/0")]}, "nodes"),
    "relative-node-source": ({"nodes": [("dev/tenstorrent/0", "/dev/tenstorrent/0")]}, "nodes"),
    "node-outside-the-card-directory": ({"nodes": [("/dev/zero", "/dev/zero")]}, "nodes"),
    "node-in-a-lookalike-directory": ({"nodes": [("/dev/tenstorrentx/0", "/dev/tenstorrentx/0")]}, "nodes"),
    "node-is-the-card-directory": ({"nodes": [("/dev/tenstorrent", "/dev/tenstorrent")]}, "nodes"),
    "node-is-a-by-id-link": (
        {"nodes": [("/dev/tenstorrent/by-id/wormhole-3DFB5316BCE44A16",) * 2]},
        "nodes",
    ),
    "node-name-is-not-a-number": ({"nodes": [("/dev/tenstorrent/0a", "/dev/tenstorrent/0a")]}, "nodes"),
    "node-below-a-card": ({"nodes": [("/dev/tenstorrent/0/x", "/dev/tenstorrent/0/x")]}, "nodes"),
    "node-with-dot-dot": ({"nodes": [("/dev/tenstorrent/../zero", "/dev/tenstorrent/0")]}, "nodes"),
    "node-source-outside-dev": ({"nodes": [("/sys/class/lassi-a", "/dev/tenstorrent/0")]}, "nodes"),
    "node-target-twice": (
        {"nodes": [("/dev/tenstorrent/0", "/dev/tenstorrent/0"), ("/dev/tenstorrent/1", "/dev/tenstorrent/0")]},
        "nodes",
    ),
    "node-is-not-a-bind": ({"nodes": ["/dev/tenstorrent/0"]}, "nodes"),
    "relative-pci-target": ({"pci_dirs": [(GOOD_PCI[0], GOOD_PCI[1].lstrip("/"))]}, "pci_dirs"),
    "relative-pci-source": ({"pci_dirs": [(GOOD_PCI[0].lstrip("/"), GOOD_PCI[1])]}, "pci_dirs"),
    "pci-source-outside-sys": ({"pci_dirs": [("/proc/bus/pci/e8", GOOD_PCI[1])]}, "pci_dirs"),
    "pci-target-outside-the-device-list": ({"pci_dirs": [(GOOD_PCI[0], "/sys/class/lassi-a")]}, "pci_dirs"),
    "pci-target-is-the-slot-list": ({"pci_dirs": [(GOOD_PCI[0], SLOTS)]}, "pci_dirs"),
    "pci-target-is-the-device-list": ({"pci_dirs": [(GOOD_PCI[0], "/sys/bus/pci/devices")]}, "pci_dirs"),
    "pci-target-below-a-device": ({"pci_dirs": [(GOOD_PCI[0], GOOD_PCI[1] + "/tenstorrent")]}, "pci_dirs"),
    "pci-target-with-dot-dot": ({"pci_dirs": [(GOOD_PCI[0], "/sys/bus/pci/devices/../slots")]}, "pci_dirs"),
    "pci-target-twice": ({"pci_dirs": [GOOD_PCI, (f"{SYNTHETIC_PARENT}/{ADDRESSES[0]}", GOOD_PCI[1])]}, "pci_dirs"),
    "pci-source-of-another-address": ({"pci_dirs": [(PCI_DIRS[1][0], GOOD_PCI[1])]}, "pci_dirs"),
    "pci-source-not-an-address": ({"pci_dirs": [("/sys/devices/system/cpu/cpu0", GOOD_PCI[1])]}, "pci_dirs"),
    "pci-source-a-lookalike-address": ({"pci_dirs": [(GOOD_PCI[0] + "0", GOOD_PCI[1])]}, "pci_dirs"),
    "pci-source-below-the-address": ({"pci_dirs": [(GOOD_PCI[0] + "/tenstorrent", GOOD_PCI[1])]}, "pci_dirs"),
    "relative-hugepages": ({"hugepages": HUGEPAGES.lstrip("/")}, "hugepages"),
    "hugepages-in-the-private-shm": ({"hugepages": "/dev/shm/lassi-run-0001"}, "hugepages"),
    "hugepages-in-a-plain-directory": ({"hugepages": "/scratch/tmp/lassi-run-0001"}, "hugepages"),
    "hugepages-under-the-2m-mount": ({"hugepages": "/dev/hugepages/lassi-run-0001"}, "hugepages"),
    "hugepages-under-a-lookalike-mount": ({"hugepages": "/dev/hugepages-1GB/lassi-run-0001"}, "hugepages"),
    "hugepages-with-dot-dot": ({"hugepages": "/dev/hugepages-1G/../shm"}, "hugepages"),
    "hugepages-with-a-blank": ({"hugepages": "/dev/hugepages-1G/lassi run"}, "hugepages"),
    "reset-filter-none": ({"reset_filter": None}, "reset_filter"),
    "reset-filter-an-int": ({"reset_filter": 1}, "reset_filter"),
    "reset-filter-a-string": ({"reset_filter": "on"}, "reset_filter"),
}


@pytest.mark.parametrize("case", sorted(BAD_FIELDS))
def test_the_spec_refuses_a_bad_exposure_naming_the_field(sandbox: ModuleType, case: str) -> None:
    overrides, field = BAD_FIELDS[case]
    tt_attribute(sandbox, "TenstorrentExposure")
    with pytest.raises(ValueError) as caught:
        sample_spec(sandbox, devices=make_tt(sandbox, **overrides))
    assert field in str(caught.value), str(caught.value)


def test_the_spec_refuses_the_hosts_whole_hugepage_directory(sandbox: ModuleType) -> None:
    # The run's own directory, never the host's whole /dev/hugepages-1G, which holds other cards' and users' files
    # (spike, Finding 5 item 3 (a)). OQ-075 (1) is ANSWERED (1b): the run's own directory under it.
    tt_attribute(sandbox, "TenstorrentExposure")
    for hugepages in (HUGEPAGE_POINT, HUGEPAGE_POINT + "/"):
        with pytest.raises(ValueError) as caught:
            sample_spec(sandbox, devices=make_tt(sandbox, hugepages=hugepages))
        assert "hugepages" in str(caught.value), str(caught.value)


# ---------------------------------------------------------------------------
# The setup's calls under stubs: binds, the writable hugepage directory, and their order


def test_the_tenstorrent_setup_runs_the_program_once_after_the_mount_check(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> None:
    run = stub_runs(sandbox, tmp_path_factory).exposed
    assert run.returncode == 7, run
    assert run.stderr.count(READY_LINE) == 1 and run.stderr.endswith(DONE_LINE), run.stderr
    assert run.stderr.index("stub:awk\n") < run.stderr.index(READY_LINE), run.stderr
    assert the_check_call(run.calls) < the_program_call(run.calls)


def test_every_card_node_is_bound_on_a_file_at_its_own_path_in_the_private_dev(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> None:
    calls = stub_runs(sandbox, tmp_path_factory).exposed.calls
    dev = private_dev(calls)
    readonly, moved = sealing(calls, dev)
    parent = f"{dev}/tenstorrent"
    made = [index for index, call in enumerate(calls) if call[0] == "mkdir" and parent in call[1:]]
    assert made, "the card directory is made in the private /dev"
    points = set()
    for source, target in NODES:
        point = dev + target[len("/dev") :]
        bound = index_of(calls, ["mount", "--bind", source, point])
        assert made[0] < bound < readonly < moved, (target, made, bound, readonly, moved)
        assert not [call for call in calls if call[0] == "mkdir" and point in call[1:]], "a node is a file bind"
        assert not writable_reopens(calls, (point, target)), "a node's bind stays read-only"
        points.add(point)
    named = {part for call in calls for part in call[1:] if part.startswith(parent + "/")}
    assert named == points, "no other path under /dev/tenstorrent, no by-id link"


def test_the_hugepage_directory_is_a_directory_bind_made_writable_before_the_mount_check(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> None:
    calls = stub_runs(sandbox, tmp_path_factory).exposed.calls
    dev = private_dev(calls)
    readonly, moved = sealing(calls, dev)
    point = dev + HUGEPAGE_POINT[len("/dev") :]
    made = [index for index, call in enumerate(calls) if call[0] == "mkdir" and point in call[1:]]
    assert len(made) == 1, "a directory bind needs a directory, not the file a node bind needs"
    assert not [call for call in calls if call[0] == "touch" and point in call[1:]], calls
    bound = index_of(calls, ["mount", "--bind", HUGEPAGES, point])
    assert made[0] < bound < readonly < moved, (made, bound, readonly, moved)
    assert [call for call in calls if HUGEPAGES in call[1:]] == [["mount", "--bind", HUGEPAGES, point]]
    whole = [call for call in calls if call[0] == "mount" and len(call) == 4 and call[1] in ("--bind", "--rbind")]
    assert not [call for call in whole if call[2] == HUGEPAGE_POINT], "never the host's whole directory"
    reopened = writable_reopens(calls, (point, HUGEPAGE_POINT))
    assert len(reopened) == 1, reopened
    index, call = reopened[0]
    assert bound < index < the_check_call(calls), "writable before the check reads the final mounts"
    if call[0] == "mount":
        # The host mount is nosuid,nodev; a user namespace locks both, and a remount that drops them fails (spike,
        # Finding 5 item 3), so the remount keeps them. It adds noexec: nothing in the directory runs or maps as code.
        assert {"bind", "nosuid", "nodev", "noexec"} <= set(call[call.index("-o") + 1].split(",")), call


def test_the_hugepage_bind_fails_closed_unless_the_running_user_owns_it(sandbox: ModuleType, tmp_path: Path) -> None:
    # Inside, uid 0 is the host user and another host user's directory shows the overflow uid, so `[ -O ]` holds only
    # for the user's own directory. The check sits between the bind and the remount that makes the bind writable.
    for script in sandbox.TENSTORRENT_SETUP_SCRIPTS.values():
        assert script.count(OWNER_CHECK) == 1 and script.count(HUGEPAGE_BIND) == script.count(HUGEPAGE_REMOUNT) == 1
        assert script.index(HUGEPAGE_BIND) < script.index(OWNER_CHECK) < script.index(HUGEPAGE_REMOUNT), script
    assert OWNER_CHECK not in sandbox.SETUP_SCRIPT and OWNER_CHECK not in sandbox.DEVICE_SETUP_SCRIPT
    command = sandbox.sandbox_command(sample_spec(sandbox, devices=make_tt(sandbox)), PROGRAM, ONE_SECOND)
    workdir = tmp_path / "work"
    workdir.mkdir()
    (tmp_path / "owned").mkdir()
    passed = stub_run_of(sandbox, command, tmp_path / "run-owned", workdir, owned=(tmp_path / "owned").as_posix())
    assert passed.returncode == 7 and passed.stderr.count(READY_LINE) == 1, passed
    refused = [(tmp_path / "missing").as_posix()]
    if hasattr(os, "geteuid") and os.stat("/").st_uid != os.geteuid():
        refused.append("/")  # owned by another user (root), as a POSIX host's / is
    for index, path in enumerate(refused):
        run = stub_run_of(sandbox, command, tmp_path / f"run-refused-{index}", workdir, owned=path)
        assert run.returncode == 2 and READY_LINE not in run.stderr, (path, run)
        point = private_dev(run.calls) + HUGEPAGE_POINT[len("/dev") :]
        bound = index_of(run.calls, ["mount", "--bind", HUGEPAGES, point])
        assert bound == len(run.calls) - 1, "the setup stops right after the bind"
        assert not writable_reopens(run.calls, (point, HUGEPAGE_POINT)), "the bind is never made writable"


def test_the_slot_list_and_each_pci_directory_are_staged_then_bound_below_the_bus_cover(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> None:
    calls = stub_runs(sandbox, tmp_path_factory).exposed.calls
    dev = private_dev(calls)
    cover = cover_made(calls, BUS_COVER)
    first_hide = typed_mounts(calls, "tmpfs", "lassi-hide")[0][0]
    covers = [i for i, args in typed_mounts(calls, "tmpfs", "lassi-sys") if "remount" not in args[3].split(",")]
    private = min(i for kind in ("lassi-private", "lassi-tmp") for i, _args in typed_mounts(calls, "tmpfs", kind))
    reopens = cover_remounts(calls, BUS_COVER)
    stages: list[str] = []
    for source, target in sys_binds():
        stage_index, stage = staged(calls, source)
        assert stage.startswith("/tmp/") and not stage.startswith(dev + "/") and stage not in stages, stage
        assert stage_index < min([first_hide, *covers]), "staged while the host's /sys is still in view"
        stages.append(stage)
        bound = index_of(calls, ["mount", "--bind", stage, target])
        assert cover < bound < private, (target, cover, bound, private)
        before = [options for index, options in reopens if index < bound]
        assert before and "rw" in before[-1], f"the /sys/bus cover is open for the bind of {target}"
        assert [i for i, call in enumerate(calls[:bound]) if call[0] == "mkdir" and target in call[1:]], target
        assert not writable_reopens(calls, (target, stage)), f"{target} stays read-only"
    assert reopens and "ro" in reopens[-1][1] and reopens[-1][0] < private, "the cover is read-only again"


def test_the_bus_cover_holds_every_directory_of_an_eight_card_exposure(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # Every cover holds 8 inodes, its root included, and an eight-card exposure needs 12 below /sys/bus: the root,
    # pci, slots, devices, and one directory per card. The run's script raises the /sys/bus cover's cap, keeping it
    # read-only, before the first mkdir below it, with room for twice alpha01's cards; every cover mounts as before.
    runs = stub_runs(sandbox, tmp_path_factory)
    command, calls = runs.command, runs.exposed.calls
    needed = {BUS_COVER}.union(*(tree_paths(target, BUS_COVER) for _source, target in sys_binds()))
    assert len(needed) == 4 + len(ADDRESSES) == 12, sorted(needed)
    below = [i for i, call in enumerate(calls) if call[0] == "mkdir" and call[-1].startswith(BUS_COVER + "/")]
    remounts = cover_remounts(calls, BUS_COVER)
    capped = [(index, options) for index, options in remounts if any(o.startswith("nr_inodes=") for o in options)]
    assert len(capped) == 1 and capped[0][0] == remounts[0][0] < min(below), (capped, remounts, below)
    assert "ro" in capped[0][1] and "rw" not in capped[0][1], "the cap is raised with the cover read-only"
    caps = [int(option[len("nr_inodes=") :]) for option in capped[0][1] if option.startswith("nr_inodes=")]
    assert caps == [tt_attribute(sandbox, "TENSTORRENT_BUS_INODES")] and caps[0] >= 4 + 2 * len(ADDRESSES), caps
    cover_line = "mount -t tmpfs -o ro,size=4k,nr_inodes=8,mode=0555 lassi-sys"
    assert script_of(command).count(cover_line) == sandbox.SETUP_SCRIPT.count(cover_line) == 1


def test_the_spec_refuses_more_pci_directories_than_the_bus_cover_holds(sandbox: ModuleType) -> None:
    most = tt_attribute(sandbox, "TENSTORRENT_CARDS_MAX")
    assert most == tt_attribute(sandbox, "TENSTORRENT_BUS_INODES") - 4 >= 2 * len(ADDRESSES), most
    addresses = [f"0000:{bus:02x}:00.0" for bus in range(most + 1)]
    pci = [(f"{SYNTHETIC_PARENT}/{address}", f"/sys/bus/pci/devices/{address}") for address in addresses]
    assert len(sample_spec(sandbox, devices=make_tt(sandbox, pci_dirs=pci[:most])).devices.pci_dirs) == most
    with pytest.raises(ValueError) as caught:
        sample_spec(sandbox, devices=make_tt(sandbox, pci_dirs=pci))
    assert "pci_dirs" in str(caught.value), str(caught.value)


def test_every_other_setup_call_is_the_plain_scripts(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> None:
    runs = stub_runs(sandbox, tmp_path_factory)
    calls = runs.exposed.calls
    marked = tenstorrent_calls(calls, private_dev(calls))
    rest = [call for index, call in enumerate(calls) if index not in marked and call[0] != "awk"]
    plain = [call for call in runs.plain.calls if call[0] != "awk"]
    # With the reset filter off, the program's chain (the nested unshare) is the plain one too.
    assert without_stub_path(rest) == without_stub_path(plain), "only the exposure's calls are added"


# ---------------------------------------------------------------------------
# The run's mount check: exactly one writable hugetlbfs mount, the run's own directory at /dev/hugepages-1G


def hugepage_line(
    root: str = HUGEPAGE_ROOT,
    point: str = HUGEPAGE_POINT,
    fstype: str = "hugetlbfs",
    source: str = "hugetlbfs",
    number: int = 45,
    supers: str = "rw,mode=777,pagesize=1024M",
) -> str:
    """Return a SYNTHETIC writable mountinfo line of a bind at `point` whose root in its filesystem is `root`.

    `supers` is the last field, the superblock's options.
    """
    return f"{number} 27 0:60 {root} {point} rw,nosuid,nodev,noexec,relatime - {fstype} {source} {supers}"


# What a finished setup with the exposure leaves besides the hugepage mount (SYNTHETIC): a read-only node bind, the
# read-only /sys/bus cover, and the read-only slot list and PCI directory binds below it.
TT_BASE = [
    *ACCEPTED_MOUNTINFO,
    "40 27 0:5 /tenstorrent/0 /dev/tenstorrent/0 ro,nosuid,relatime - devtmpfs udev rw,size=100k,mode=755",
    "41 24 0:56 / /sys/bus ro,relatime - tmpfs lassi-sys rw,size=4k,nr_inodes=8,mode=555",
    "42 41 0:22 /bus/pci/slots /sys/bus/pci/slots ro,nosuid,nodev,noexec,relatime - sysfs sysfs rw",
    f"43 41 0:22 {PCI_DIRS[0][0][len('/sys') :]} {PCI_DIRS[0][1]} ro,nosuid,nodev,noexec,relatime - sysfs sysfs rw",
]
TT_ACCEPTED = [*TT_BASE, hugepage_line()]
# Mountinfo the run's check refuses: case -> (lines, the mount point it must name).
TT_REFUSED: dict[str, tuple[list[str], str]] = {
    "the-hosts-whole-hugepage-directory": ([*TT_BASE, hugepage_line(root="/")], HUGEPAGE_POINT),
    "another-runs-hugepage-directory": ([*TT_BASE, hugepage_line(root="/lassi-run-0002")], HUGEPAGE_POINT),
    "a-lookalike-of-the-runs-directory": ([*TT_BASE, hugepage_line(root=HUGEPAGE_ROOT + "2")], HUGEPAGE_POINT),
    "the-runs-directory-at-the-2m-point": ([*TT_BASE, hugepage_line(point="/dev/hugepages")], "/dev/hugepages"),
    "the-runs-directory-below-the-point": (
        [*TT_BASE, hugepage_line(point=HUGEPAGE_POINT + "/x")],
        HUGEPAGE_POINT + "/x",
    ),
    "the-runs-directory-elsewhere": ([*TT_BASE, hugepage_line(point="/mnt/huge")], "/mnt/huge"),
    "a-plain-directory-at-the-point": (
        [*TT_BASE, hugepage_line(root="/scratch/tmp/x", fstype="xfs", source="/dev/nvme10n1")],
        HUGEPAGE_POINT,
    ),
    "a-host-tmpfs-at-the-point": ([*TT_BASE, hugepage_line(fstype="tmpfs", source="tmpfs")], HUGEPAGE_POINT),
    "a-second-hugepage-mount": ([*TT_ACCEPTED, hugepage_line(number=46)], HUGEPAGE_POINT),
    "2m-pages-at-the-point": ([*TT_BASE, hugepage_line(supers="rw,pagesize=2M")], HUGEPAGE_POINT),
    "no-page-size": ([*TT_BASE, hugepage_line(supers="rw,mode=777")], HUGEPAGE_POINT),
    "a-lookalike-page-size": ([*TT_BASE, hugepage_line(supers="rw,mode=777,pagesize=10240M")], HUGEPAGE_POINT),
    "a-lookalike-option": ([*TT_BASE, hugepage_line(supers="rw,mode=777,xpagesize=1024M")], HUGEPAGE_POINT),
    "the-page-size-only-as-the-source": (
        [*TT_BASE, hugepage_line(source="pagesize=1024M", supers="rw,pagesize=2M")],
        HUGEPAGE_POINT,
    ),
    **{
        f"plain-{case}": ([*TT_ACCEPTED[:5], line, *TT_ACCEPTED[5:]], point)
        for case, (line, point) in REFUSED_MOUNTINFO.items()
    },
}


def replay_check(call: list[str], tmp_path: Path, lines: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a recorded `awk ... <program> ... /proc/self/mountinfo` call with the awk on PATH against `lines`.

    The program (the one argument holding a newline) goes in a file passed
    with -f, and the mountinfo path is replaced by a file of `lines`; every
    other argument is kept as recorded. Skip when there is no awk.
    """
    awk = shutil.which("awk")
    if awk is None:
        pytest.skip("awk is not on PATH")
    args = call[1:]
    assert call[0] == "awk" and args and args[-1] == "/proc/self/mountinfo", call
    programs = [index for index, part in enumerate(args) if "\n" in part]
    assert len(programs) == 1, call
    program = tmp_path / "check.awk"
    program.write_text(args[programs[0]], encoding="ascii", newline="\n")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("".join(line + "\n" for line in lines), encoding="ascii", newline="\n")
    head, tail = args[: programs[0]], args[programs[0] + 1 : -1]
    argv = [awk, *head, "-f", program.as_posix(), *tail, mountinfo.as_posix()]
    return subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)


def check_call(sandbox: ModuleType, factory: pytest.TempPathFactory, hugepages: str = HUGEPAGES) -> list[str]:
    """Return the mount check's awk call from a stub run of the exposure's script."""
    calls = stub_runs(sandbox, factory, hugepages=hugepages).exposed.calls
    return calls[the_check_call(calls)]


def test_the_plain_mount_check_never_trusts_a_hugepage_mount(sandbox: ModuleType, tmp_path: Path) -> None:
    # Guard: passes before and after task P17.15. Runs without the exposure keep MOUNT_CHECK, which refuses it.
    done = run_mount_check(sandbox, tmp_path, TT_ACCEPTED)
    assert done.returncode != 0 and HUGEPAGE_POINT in done.stderr, done


def test_the_runs_mount_check_accepts_exactly_its_own_hugepage_mount(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory, tmp_path: Path
) -> None:
    call = check_call(sandbox, tmp_path_factory)
    assert call != ["awk", sandbox.MOUNT_CHECK, "/proc/self/mountinfo"], "the run's check carries the exception"
    # hugetlbfs prints pagesize for every mount, after uid, gid, mode, and nr_inodes when set, and before size and
    # min_size (Linux v6.6 fs/hugetlbfs/inode.c:1174-1199): the host's mount, a mode-0755 one, and a size-capped one.
    supers = ("rw,pagesize=1024M", "rw,uid=1000,gid=1000,mode=700,pagesize=1024M,size=2147483648")
    for lines in (TT_BASE, TT_ACCEPTED, *([*TT_BASE, hugepage_line(supers=text)] for text in supers)):
        done = replay_check(call, tmp_path, lines)
        assert (done.returncode, done.stdout, done.stderr) == (0, "", ""), done


@pytest.mark.parametrize("case", sorted(TT_REFUSED))
def test_the_runs_mount_check_refuses_any_other_writable_mount_and_names_it(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory, tmp_path: Path, case: str
) -> None:
    lines, point = TT_REFUSED[case]
    done = replay_check(check_call(sandbox, tmp_path_factory), tmp_path, lines)
    assert done.returncode != 0 and done.stdout == "", done
    assert f"writable host mount: {point} " in done.stderr, done


def test_the_expected_root_follows_a_nested_hugepage_directory(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory, tmp_path: Path
) -> None:
    call = check_call(sandbox, tmp_path_factory, hugepages=NESTED_HUGEPAGES)
    accepted = replay_check(call, tmp_path, [*TT_BASE, hugepage_line(root=NESTED_ROOT)])
    assert (accepted.returncode, accepted.stdout, accepted.stderr) == (0, "", ""), accepted
    for root in ("/lassi", HUGEPAGE_ROOT):
        refused = replay_check(call, tmp_path, [*TT_BASE, hugepage_line(root=root)])
        assert refused.returncode != 0 and HUGEPAGE_POINT in refused.stderr, (root, refused)


def test_the_runs_mount_check_compares_the_root_as_a_string(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory, tmp_path: Path
) -> None:
    # awk compares a field and a -v value that both read as numbers numerically, so 1 and 1.0 would match; the
    # check compares ($4 "") as a string. Real roots start with "/", so a SYNTHETIC run_root that reads as a number
    # stands in for one, swapped into the recorded call.
    call = check_call(sandbox, tmp_path_factory)
    assert call.count(f"run_root={HUGEPAGE_ROOT}") == 1, call
    numeric = ["run_root=1.0" if part == f"run_root={HUGEPAGE_ROOT}" else part for part in call]
    same = replay_check(numeric, tmp_path, [*TT_BASE, hugepage_line(root="1.0")])
    assert (same.returncode, same.stdout, same.stderr) == (0, "", ""), same
    for root in ("1", "1.00", "01.0", "1e0"):
        refused = replay_check(numeric, tmp_path, [*TT_BASE, hugepage_line(root=root)])
        assert refused.returncode != 0 and f"writable host mount: {HUGEPAGE_POINT} " in refused.stderr, (root, refused)


# ---------------------------------------------------------------------------
# The reset filter: off unless asked; on, only the reset request fails


def filter_of(program: str) -> list[tuple[int, int, int, int]]:
    """Return a confinement program's FILTER, imported under another __name__ so nothing is installed or run."""
    namespace: dict[str, object] = {"__name__": "lassi_confine_under_test"}
    exec(program, namespace)
    return list(namespace["FILTER"])  # type: ignore[call-overload]


def reset_program(sandbox: ModuleType) -> str:
    """Return confine_program(True), the confinement of a run whose exposure asks for the reset filter."""
    program = tt_attribute(sandbox, "confine_program")(True)
    assert isinstance(program, str)
    return program


def test_without_the_reset_filter_the_confinement_is_the_plain_one(sandbox: ModuleType) -> None:
    # The plain filter lets every ioctl through (guard), and the off setting is CONFINE_PROGRAM byte for byte.
    assert run_filter(filter_of(sandbox.CONFINE_PROGRAM), NR_IOCTL, (3, RESET_IOCTL)) == SECCOMP_ALLOW
    assert tt_attribute(sandbox, "confine_program")(False) == sandbox.CONFINE_PROGRAM


@pytest.mark.parametrize(
    ("args", "action"),
    [
        ((3, RESET_IOCTL), SECCOMP_EPERM),
        ((3, RESET_IOCTL, 0), SECCOMP_EPERM),
        ((3, RESET_IOCTL | 1 << 32), SECCOMP_EPERM),
        ((3, RESET_IOCTL | 0xFFFFFFFF << 32), SECCOMP_EPERM),
        ((3, RESET_IOCTL - 1), SECCOMP_ALLOW),
        ((3, RESET_IOCTL + 1), SECCOMP_ALLOW),
        ((3, 0x5401), SECCOMP_ALLOW),
        ((3, 0), SECCOMP_ALLOW),
        ((RESET_IOCTL, 0x5401), SECCOMP_ALLOW),
        ((3, 0x5401, RESET_IOCTL), SECCOMP_ALLOW),
    ],
    ids=[
        "reset", "reset-with-a-zero-argument", "reset-with-a-high-bit", "reset-with-every-high-bit",
        "the-request-below", "the-request-above", "tcgets", "request-zero", "reset-number-as-the-descriptor",
        "reset-number-as-the-argument",
    ],
)  # fmt: skip
def test_the_reset_filter_refuses_the_reset_request_and_nothing_near_it(
    sandbox: ModuleType, args: tuple[int, ...], action: int
) -> None:
    # The kernel reads ioctl's request as an unsigned int, so high bits in the register still mean the reset.
    assert run_filter(filter_of(reset_program(sandbox)), NR_IOCTL, args) == action


def test_the_reset_filter_changes_no_other_syscall_or_request(sandbox: ModuleType) -> None:
    plain, reset = filter_of(sandbox.CONFINE_PROGRAM), filter_of(reset_program(sandbox))
    samples = [(), (3, RESET_IOCTL), (3, RESET_IOCTL | 1 << 32), (3, 0x5401), (0, 4), (40, 1), (0, 4, 0x7FFE0000)]
    samples += [(1, 7, 0x7FFE0000), (0, 4, 0, 0x7FFE0000)]
    numbers = [*range(460), *(X32_BIT | nr for nr in (0, NR_IOCTL, 160, 302))]
    for arch in (ARCH_X86_64, ARCH_I386):
        for nr in numbers:
            for args in samples:
                expected = run_filter(plain, nr, args, arch)
                if (arch, nr) == (ARCH_X86_64, NR_IOCTL) and len(args) > 1 and args[1] & 0xFFFFFFFF == RESET_IOCTL:
                    expected = SECCOMP_EPERM
                assert run_filter(reset, nr, args, arch) == expected, (hex(arch), nr, args)


def test_the_reset_program_is_a_filter_the_kernel_accepts_with_every_row_named(sandbox: ModuleType) -> None:
    program = reset_program(sandbox)
    assert program.isascii() and "'" not in program, "embedded in single quotes"
    rules = filter_of(program)
    assert 0 < len(rules) <= 4096 and rules[-1][0] == 0x06
    for index, (code, jt, jf, k) in enumerate(rules):
        assert code in (0x20, 0x15, 0x35, 0x06), (index, code)
        if code == 0x20:
            assert k % 4 == 0 and k < 64, (index, k)
        if code in (0x15, 0x35):
            assert index + 1 + max(jt, jf) < len(rules), (index, jt, jf)
        assert 0 <= jt < 256 and 0 <= jf < 256 and 0 <= k < 1 << 32, index
    start = program.index("FILTER = [\n") + len("FILTER = [\n")
    rows = program[start : program.index("\n]\n", start)].split("\n")
    assert len(rows) == len(rules), rows
    for index, row in enumerate(rows):
        assert re.fullmatch(rf"    \(.*\),  # {index}: \S.*", row), row


def test_the_reset_program_differs_from_the_plain_one_only_in_its_constants_and_filter(sandbox: ModuleType) -> None:
    def statements(program: str) -> list[str]:
        """Return every top-level statement but the assignments, as AST dumps (comments are not statements)."""
        return [ast.dump(node) for node in ast.parse(program).body if not isinstance(node, ast.Assign)]

    assert statements(reset_program(sandbox)) == statements(sandbox.CONFINE_PROGRAM)


def test_the_runs_script_embeds_the_confinement_its_reset_setting_names(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> None:
    on_program = reset_program(sandbox)
    off, on = stub_runs(sandbox, tmp_path_factory), stub_runs(sandbox, tmp_path_factory, reset_filter=True)
    off_script, on_script = script_of(off.command), script_of(on.command)
    assert off_script.count(sandbox.CONFINE_PROGRAM) == 1 and on_program not in off_script
    assert on_script.count(on_program) == 1
    assert on_script.replace(on_program, sandbox.CONFINE_PROGRAM, 1) == off_script, "nothing else changes"
    assert on.command[on.command.index(on_script) + 1 :] == off.command[off.command.index(off_script) + 1 :]
    for runs, program in ((off, sandbox.CONFINE_PROGRAM), (on, on_program)):
        chain = runs.exposed.calls[the_program_call(runs.exposed.calls)]
        assert chain[chain.index("-c") + 1] == program, "the program's chain installs that filter"
