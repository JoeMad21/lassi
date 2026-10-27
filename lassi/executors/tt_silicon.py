"""The tt_silicon executor, registered as Executor "tt_silicon" (task P17.16; bible Execution Backends, tt_silicon).

It runs a TT host program (built by the ttmetal-host toolchain) with its
inputs as arguments on the cluster's Wormhole n300 cards, only through
lassi.executors.sandbox (Agent Rule 6), with every card exposed through the
sandbox's TenstorrentExposure (task P17.15): the owner gave a silicon run
every Wormhole card in the cluster (OQ-072) and approved the settings below
(OQ-075). Its capabilities are runs_code, sandboxed, and silicon
(lassi.core.capabilities SILICON). Silicon runs are evaluation runs a human
starts, one program at a time (Agent Rule 9), so the runner refuses it
unless `lassi run --silicon` starts the run, a reward stage or a train
recipe's reward refuses it outright, and both read the capability, never
this name. Until OQ-078 is answered the runner also refuses it, with the
flag too, beside any listed stage that runs model-generated code or an
adversary's proposals (lassi.core.runner _check_silicon): under the cards'
identity IOMMU domain a program can reach host memory through a card's DMA
(Known limits, below), so the executor runs only tracked references a
human starts (OQ-078's recommendation (c)). It declares neither simulator,
since nothing is simulated (Agent Rule 2), nor takes_device, since every
card is exposed and there is no card key.

Settings (a recipe's executor section, `executor: {kind: tt_silicon,
hugepages: private-per-run, reset_filter: true}`), each required with no
default (OQ-075), so nothing picks one silently: `hugepages`, whose one
value is "private-per-run" (OQ-075 (1b): the run's own directory inside the
host's 1 GiB hugetlbfs mount), and `reset_filter`, a bool (OQ-075 (4a):
True makes the sandbox's seccomp filter refuse the driver's reset ioctl).
An absent value or any other is a ValueError naming the key, raised before
any host file is read. `harness` is the ttsim executor's.

Construction reads the host under its root from file metadata (lstat,
access(2), and directory listings) and sysfs or procfs text only: it opens
no device node and starts no process (OQ-002). It takes every all-digit
name in /dev/tenstorrent, in numeric order (the by-id links skipped, as UMD
skips them; plans/spikes/tt-silicon-executor.md, Finding 1), and refuses
with DeviceUnavailable, naming the host path (never the root-joined one),
when: $LASSI_SCRATCH is unset or relative, or a quarantine marker exists
(below); /dev/tenstorrent is missing or holds no numbered node, or more
than the sandbox's TENSTORRENT_CARDS_MAX (28), the most PCI directories one
exposure holds; a node is missing or access(2) refuses this user read and
write; a card's class directory /sys/class/tenstorrent/tenstorrent!<n> is
missing, its tt_card_type is not n300, or its tt_serial, tt_fw_bundle_ver,
or tt_arc_fw_ver is missing or empty (every result records them); its
device/uevent gives no PCI_SLOT_NAME the sandbox can bind; its
device/iommu_group/type starts with DMA (Finding 2: the group's link climbs
out of the card's bind, so UMD inside would read no IOMMU, which would then
differ from the host's); /sys/module/tenstorrent/version or
/sys/bus/pci/slots is missing; the 1 GiB pool's nr_hugepages is unreadable
or below PAGES_PER_CARD per card; /dev/hugepages-1G is no directory; or the
last /proc/mounts line at /dev/hugepages-1G is not hugetlbfs with
pagesize=1024M, the line UMD looks for (Finding 3). A missing iommu_group
or tt_eth_fw_ver is accepted. So under OQ-078's option (a), a translating
(DMA) IOMMU domain for every card, every card is refused until the sandbox
exposes the card's IOMMU group view (a follow-up, not built).

run(artifact, inputs, limits) keeps the ttsim executor's workdir rules (a
relative artifact, a string of inputs, or a workdir outside the runs root
is a ValueError) and install rules for the pinned tt-metal tree alone, no
ttsim library (SandboxUnavailableError naming the tree), all before any
lock or directory. Then, in order:

1. The lock (OQ-075 (2a)): one file for every card, LOCK_FILE directly
   under $LASSI_SCRATCH/locks, opened without truncation and taken with
   flock LOCK_EX | LOCK_NB, never a wait; the holder then writes its pid
   there. Another holder makes the run refuse at once with
   SandboxUnavailableError naming the file and the pid that holder wrote
   (or that it cannot be read; the descriptor is closed either way). The
   lock is held through the cleanup and released on every path (the pid
   cleared, then LOCK_UN, then the descriptor closed; a failure to clear or
   release is passed over, since the close drops the lock, so it hides
   neither the result nor an exception), and the kernel drops a dead
   process's lock; the file stays in place, since removing it would let
   two processes lock two files. A training run started with
   `lassi train --silicon` takes the same lock for its whole length
   (card_lock; task P17.38, OQ-100 (a)), polls with wait_for_cards, and
   honours and writes the same quarantine marker, so a training run and a
   silicon run never overlap. The scratch root is always one of the
   hidden roots, so no program sees the lock or a marker. The default
   flock is fcntl.flock, imported only when a lock is taken, so this module
   imports on Windows; LOCK_EX, LOCK_NB, and LOCK_UN are Linux's
   <sys/file.h> values, which fcntl gives on Linux.
2. The quarantine check, under the lock: while any entry of
   $LASSI_SCRATCH/locks whose name holds "quarantine" exists, the run
   refuses with SandboxUnavailableError naming it.
3. The poll (OQ-075 (5); the owner: "sometimes the previous user takes a
   few minutes to relieve the cards. Be sure to poll and check first."):
   the cards are busy while another account owns a regular file directly
   in /dev/hugepages-1G or a TT_UMD_LOCK* file in /dev/shm, or while
   free_hugepages reads fewer than PAGES_PER_CARD pages per card;
   directories (libvirt's) and this account's own files do not count. It
   checks, sleeps at most POLL_INTERVAL_S, and checks again until the cards
   are free or POLL_LIMIT_S has passed, then refuses with
   SandboxUnavailableError naming what stayed busy. The sandbox never runs
   while a card is busy.
4. The run's own hugepage directory (hugepages "private-per-run"): a new
   directory one segment below /dev/hugepages-1G, mode 0700, so it is
   empty when the program starts and no other account sees its files.
   Afterwards, on every path this process lives through (a kill of it
   leaves the directory beside item 6's marker), it is removed with every
   entry in it, links not followed (shutil.rmtree; _remove_tree): an entry
   the program made read-only gets its owner's write permission back and
   is tried once more, and the whole removal runs a second time when the
   first fails. A directory that still cannot be removed is named in the
   quarantine marker (item 6), and the run's result or exception stands.
   Nothing else there changes. Each run thus maps fresh zeroed pages, and
   no bytes pass to the next run.
5. One sandboxed run, never retried and never rerun with Watcher, each of
   which would be another placement on cards in an unknown state (Agent
   Rule 9). Its exposure holds every card's node at its own path, each
   card's PCI directory at /sys/bus/pci/devices/<address> (that bus path is
   the source too, which the bind follows to the device directory on the
   host), the run's hugepage directory, and reset_filter. The environment
   is the ttsim row's without TT_METAL_SIMULATOR (OQ-075 (8a): slow
   dispatch and TT_METAL_DISABLE_SFPLOADMACRO=1, as the smoke ran), with
   the JIT cache and logs under <workdir>/@tt-silicon (RESERVED_DIR; no
   model file path may start with "@"), created before the run, no HOME,
   and nothing from the caller's environment. TT_VISIBLE_DEVICES stays
   unset: with every node bound, UMD enumerates every numbered node whether
   it is set or unset (Finding 1), setting it costs more opens of every
   node, and the sandbox's ENVIRONMENT_NAMES does not hold it. The spec
   and the sandbox command (sandbox_command) are built before item 6's
   marker, so a ValueError from either writes none.
6. The quarantine (OQ-075 (3a)): right before the sandbox is called, the
   marker QUARANTINE_FILE goes under $LASSI_SCRATCH/locks, naming every
   card's serial and the ending UNRECORDED, so it is in place while the
   program may run, and a kill of this process (SIGKILL included) or a
   host crash leaves it. It is removed only after a clean ending, status 0
   to 127 with neither hang nor killed (124 too when the program stopped
   itself before the limit), or a SandboxUnavailableError other than
   SandboxCopyBackError: the program never started. Every other ending
   keeps it, rewritten to name the ending: a hang; a kill of the program
   before the wall limit (killed: a memory or CPU-time limit, or the
   program itself); -1 (the runner's own timeout); any status outside 0 to
   127 (a crash is a death by signal, which the sandbox reports as
   128 + N, and the shell's form cannot tell it from a program's own
   status of 128 or more); the copy-back failure (SandboxCopyBackError,
   raised after the program ran); and any other exception (an interrupt),
   which then propagates. A hugepage directory that cannot be removed
   (item 4) is named in it too.
   While it exists every later run and construction is refused, until a
   human checks the cards and removes it. A kernel JIT failure aborts the
   host program with status 134, as on ttsim (plans/spikes/p4-ttsim-runtime.md,
   the jit-error case), so it quarantines every card until a human removes
   the marker.
7. The RunResult is the ttsim executor's without simulator readings: the
   sandbox's status, streams, flags, and wall time, the regular files new
   in the workdir outside RESERVED_DIR, sim_ub and sim_gap left None, and
   the kernel JIT's diagnostics (lassi.executors.ttsim
   read_jit_diagnostics). No line is read as a ttsim finding.

device() names "Tenstorrent Wormhole n300 x <N> (silicon)", then each
card's number, serial, firmware bundle, and ARC firmware in card order, the
tenstorrent driver's version, and the pinned tt-metal, from what
construction read, as one line of printable ASCII (any other character
written "?"); never a simulator. `pins` holds toolchains/tt-metal.pin's
pairs alone, which the runner records (Agent Rule 10). Wall times are never
performance: wall_s is the whole sandbox command's time, setup and
copy-back included.

Known limits (readings, not runs): the sandbox does not isolate the host
from a silicon run's program, since under the cards' identity IOMMU domain
a program can reach host memory through a card's DMA (plans/spikes/
tt-card-dma.md; OQ-078, open); hugetlb pages count toward no limit the
sandbox sets, so the program may map every free 1 GiB page while it runs;
and the poll sees only hugepage files and UMD locks, so a process that
holds a card without either, or starts after the poll, is not seen.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import stat
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from lassi.core.capabilities import SILICON
from lassi.core.devices import DeviceUnavailable
from lassi.core.interfaces import Limits, RunResult
from lassi.core.registry import register
from lassi.executors import devices
from lassi.executors.native import (
    _TOOLCHAINS_VARIABLE,
    _absolute_variable,
    _checked_workdir,
    _environment_roots,
    _regular_files,
    _resolved,
)
from lassi.executors.sandbox import (
    HUGEPAGE_OPTION,
    HUGEPAGE_POINT,
    PCI_SLOTS,
    SANDBOX_PATH,
    TENSTORRENT_CARDS_MAX,
    DeviceBind,
    Sandbox,
    SandboxCopyBackError,
    SandboxResult,
    SandboxSpec,
    SandboxUnavailableError,
    TenstorrentExposure,
    sandbox_command,
)
from lassi.executors.ttsim import TT_METAL_PIN, printable, read_jit_diagnostics
from lassi.executors.workdir import runs_root
from lassi.toolchains import pins as pin_files
from lassi.toolchains.ttmetal_build import check_cpm_sources, check_install_record

# The one value of the hugepages setting (OQ-075 (1b)): the run's own directory inside the host's 1 GiB mount.
PRIVATE_PER_RUN = "private-per-run"
# The card type the executor runs on: card 0's tt_card_type on alpha01 (results/tt-silicon-smoke-20261023).
CARD_TYPE = "n300"
# The 1 GiB pages one n300 card's run maps: tt-metal asks for two host-memory channels per n300 (Finding 3).
PAGES_PER_CARD = 2
# The poll (OQ-075 (5)): how long it waits between checks, and how long the previous user may take, in seconds.
POLL_INTERVAL_S = 5
POLL_LIMIT_S = 300
# Under $LASSI_SCRATCH: the directory of the lock and the marker, the lock, the marker, and the text every marker's
# name holds (a human may add a marker of another name to stop silicon runs).
LOCKS_DIR = "locks"
LOCK_FILE = "tt-silicon.lock"
QUARANTINE_FILE = "tt-silicon.quarantine"
QUARANTINE_MARK = "quarantine"
# The ending the marker names from just before the sandbox is called until the run's ending replaces it or a clean
# ending removes the marker, so it stays only when this process dies first (a kill of it, or a host crash).
UNRECORDED = (
    "not recorded: this marker was written before the program could start on these cards, and the process that "
    "wrote it ended before it recorded how the run ended (a kill of that process, or a host crash)"
)
# flock(2) operations, Linux's <sys/file.h> values, which fcntl gives on Linux.
LOCK_EX, LOCK_NB, LOCK_UN = 2, 4, 8
# The executor's own directory in a workdir, and the JIT cache and logs under it (no model path may start with "@").
RESERVED_DIR = "@tt-silicon"
CACHE_DIR = "cache"
LOGS_DIR = "logs"
# The name of a run's hugepage directory starts with this; its pid and random hex follow, so every segment matches
# the sandbox's device-segment pattern.
RUN_DIR_PREFIX = "lassi-tt-silicon-"
# The host paths construction and the poll read, as POSIX text under the host root.
NODE_DIR = "/dev/tenstorrent"
CLASS_DIR = "/sys/class/tenstorrent"
DRIVER_VERSION = "/sys/module/tenstorrent/version"
POOL_DIR = "/sys/kernel/mm/hugepages/hugepages-1048576kB"
MOUNTS = "/proc/mounts"
SHM = "/dev/shm"
PCI_DEVICES = "/sys/bus/pci/devices"
# UMD's cross-process lock files in /dev/shm start with this (umd utils/robust_mutex.cpp:51; Finding 4).
UMD_LOCK_PREFIX = "TT_UMD_LOCK"
# A PCI address the sandbox can bind at /sys/bus/pci/devices/<address> (its exposure check's pattern).
_PCI_ADDRESS = re.compile(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
# The sysfs attributes every card must give, which every result records.
_RECORDED = ("tt_serial", "tt_fw_bundle_ver", "tt_arc_fw_ver")


@dataclass(frozen=True)
class Card:
    """One card as construction read it: its /dev/tenstorrent name, PCI address, serial, and firmware versions."""

    name: str
    address: str
    serial: str
    bundle: str
    arc: str


@dataclass(frozen=True)
class _Mounts:
    """What a run's sandbox sees: the hidden roots, the harness, the resolved toolchains root, and the tt-metal tree."""

    hidden_roots: tuple[Path, ...]
    harness: Path | None
    root: Path
    tree: Path


def _host(root: Path, host_path: str) -> Path:
    """Return the root-joined path of a host path such as /dev/tenstorrent."""
    return root.joinpath(*PurePosixPath(host_path).parts[1:])


def _value(path: Path) -> str:
    """Return a sysfs or procfs file's text, whitespace collapsed, in printable ASCII; "" when it cannot be read."""
    return printable(" ".join((devices._read_text(path) or "").split()))


def _count(path: Path) -> int | None:
    """Return a sysfs page count as an int, or None when the file cannot be read or holds no whole number."""
    text = _value(path)
    return int(text) if text.isdigit() else None


def _owner(path: Path) -> int:
    """Return the uid that owns `path`, from lstat, so a link is not followed."""
    return os.lstat(path).st_uid


def _scratch_root() -> Path:
    """Return $LASSI_SCRATCH, which holds the lock and the quarantine marker; DeviceUnavailable unless absolute."""
    value = os.environ.get("LASSI_SCRATCH", "")
    if not value or not Path(value).is_absolute():
        found = "unset" if not value else f"the relative path {value!r}"
        raise DeviceUnavailable(
            f"$LASSI_SCRATCH is {found}; the tt_silicon executor keeps its lock and its quarantine marker under "
            "$LASSI_SCRATCH/locks, so set it to an absolute path"
        )
    return Path(value)


def quarantine_markers(locks: Path) -> list[Path]:
    """Return the entries of `locks` whose name holds "quarantine", sorted; none when `locks` cannot be listed."""
    return [path for path in devices._entries(locks) if QUARANTINE_MARK in path.name]


def _refuse_quarantine(locks: Path, error: type[Exception]) -> None:
    """Raise `error` naming every quarantine marker under `locks` when there is one (OQ-075 (3a))."""
    markers = quarantine_markers(locks)
    if markers:
        named = ", ".join(str(path) for path in markers)
        raise error(
            f"the Tenstorrent cards are quarantined: {named} exists (a silicon run holds the cards, or one ended in a "
            "way that leaves them in an unknown state, as the file says); a human checks every card and removes it "
            "before another silicon run (OQ-075 (3a)); nothing ran"
        )


def _card_names(root: Path) -> list[str]:
    """Return the all-digit names in /dev/tenstorrent under `root`, in numeric order.

    DeviceUnavailable when there is none, or more than the sandbox's
    TENSTORRENT_CARDS_MAX, which a run's exposure cannot hold.
    """
    directory = _host(root, NODE_DIR)
    if not directory.is_dir():
        raise DeviceUnavailable(f"{NODE_DIR} does not exist on this host, so there is no Tenstorrent card to run on")
    names = [path.name for path in devices._entries(directory) if path.name.isascii() and path.name.isdigit()]
    if not names:
        raise DeviceUnavailable(f"{NODE_DIR} holds no numbered card node on this host")
    if len(names) > TENSTORRENT_CARDS_MAX:
        raise DeviceUnavailable(
            f"{NODE_DIR} holds {len(names)} numbered card nodes, more than the {TENSTORRENT_CARDS_MAX} whose PCI "
            "directories a sandboxed run can expose (lassi.executors.sandbox TENSTORRENT_CARDS_MAX); every card is "
            "exposed, so none can run here"
        )
    return sorted(names, key=int)


def _check_nodes(root: Path, names: Sequence[str], access: Callable[[Path], bool]) -> None:
    """Refuse a card node that does not exist (lstat) or that `access` does not let this user read and write.

    `access` is asked once per node, and only about a node that exists; nothing is opened.
    """
    for name in names:
        path = _host(root, NODE_DIR) / name
        if not os.path.lexists(path):
            raise DeviceUnavailable(f"{NODE_DIR}/{name} does not exist on this host")
        if not access(path):
            raise DeviceUnavailable(f"{NODE_DIR}/{name} exists, but this user may not read and write it")


def _read_card(root: Path, name: str) -> Card:
    """Return card `name` from its class directory under `root`, or raise DeviceUnavailable naming what is wrong."""
    where = f"{CLASS_DIR}/tenstorrent!{name}"
    directory = _host(root, where)
    if not directory.is_dir():
        raise DeviceUnavailable(f"{where} does not exist on this host, so card {name} cannot be recorded")
    card_type = _value(directory / "tt_card_type")
    if card_type != CARD_TYPE:
        raise DeviceUnavailable(
            f"{where}/tt_card_type reads {card_type or 'nothing'!r}; the tt_silicon executor runs on {CARD_TYPE} "
            "cards only"
        )
    values: dict[str, str] = {}
    for attribute in _RECORDED:
        values[attribute] = _value(directory / attribute)
        if not values[attribute]:
            raise DeviceUnavailable(f"{where}/{attribute} is missing or empty, and every silicon result records it")
    address = _pci_address(directory / "device" / "uevent", where)
    iommu = _value(directory / "device" / "iommu_group" / "type")
    if iommu.startswith("DMA"):
        raise DeviceUnavailable(
            f"{where}/device/iommu_group/type reads {iommu!r}: the group's link climbs out of the card's bind, so "
            "UMD in the sandbox would read no IOMMU where the host has a DMA one (plans/spikes/"
            "tt-silicon-executor.md, Finding 2)"
        )
    return Card(name, address, values["tt_serial"], values["tt_fw_bundle_ver"], values["tt_arc_fw_ver"])


def _pci_address(uevent: Path, where: str) -> str:
    """Return the PCI_SLOT_NAME of a card's PCI uevent, or raise DeviceUnavailable naming the class directory."""
    text = devices._read_text(uevent)
    if text is None:
        raise DeviceUnavailable(f"{where}/device/uevent cannot be read, so the card's PCI address is unknown")
    for line in text.splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key == "PCI_SLOT_NAME" and _PCI_ADDRESS.fullmatch(value.strip()):
            return value.strip()
    raise DeviceUnavailable(f"{where}/device/uevent gives no PCI_SLOT_NAME of the form {_PCI_ADDRESS.pattern}")


def _driver_version(root: Path) -> str:
    """Return the tenstorrent driver's version under `root`; DeviceUnavailable when it is missing or empty."""
    version = _value(_host(root, DRIVER_VERSION))
    if not version:
        raise DeviceUnavailable(f"{DRIVER_VERSION} is missing or empty; UMD reads the driver's version there")
    return version


def _check_host(root: Path, cards: int) -> None:
    """Refuse a host whose slot list, 1 GiB pool, hugepage directory, or hugetlbfs mount cannot serve `cards` cards."""
    if not _host(root, PCI_SLOTS).is_dir():
        raise DeviceUnavailable(f"{PCI_SLOTS} does not exist on this host; UMD lists it for every card")
    pages, needed = _count(_host(root, f"{POOL_DIR}/nr_hugepages")), PAGES_PER_CARD * cards
    if pages is None:
        raise DeviceUnavailable(f"{POOL_DIR}/nr_hugepages holds no page count; UMD reads the 1 GiB pool there")
    if pages < needed:
        raise DeviceUnavailable(
            f"{POOL_DIR}/nr_hugepages reads {pages}, fewer than the {needed} 1 GiB pages {cards} cards need "
            f"({PAGES_PER_CARD} each)"
        )
    if not _host(root, HUGEPAGE_POINT).is_dir():
        raise DeviceUnavailable(f"{HUGEPAGE_POINT} is not a directory on this host, so no run can have its own there")
    text = devices._read_text(_host(root, MOUNTS))
    lines = [line.split() for line in (text or "").splitlines()]
    at_point = [fields for fields in lines if len(fields) >= 4 and fields[1] == HUGEPAGE_POINT]
    if not at_point or at_point[-1][2] != "hugetlbfs" or HUGEPAGE_OPTION not in at_point[-1][3].split(","):
        raise DeviceUnavailable(
            f"{MOUNTS} shows no hugetlbfs mount at {HUGEPAGE_POINT} with {HUGEPAGE_OPTION}, the mount UMD maps its "
            "1 GiB host-memory channels from (plans/spikes/tt-silicon-executor.md, Finding 3)"
        )


def _metal_pin() -> dict[str, str]:
    """Return toolchains/tt-metal.pin's pairs; ValueError naming the file when a key the executor reads is missing."""
    pin = pin_files.read_pin(TT_METAL_PIN)
    missing = [key for key in ("NAME", "VERSION", "COMMIT", "PREFIX_NAME") if not pin.get(key)]
    if missing:
        raise ValueError(f"toolchains/{TT_METAL_PIN}.pin has no {', '.join(missing)}, which this executor reads")
    return pin


def _check_settings(hugepages: object, reset_filter: object) -> None:
    """Raise ValueError naming the key unless hugepages is "private-per-run" and reset_filter a bool (OQ-075)."""
    if not (isinstance(hugepages, str) and hugepages == PRIVATE_PER_RUN):
        raise ValueError(
            f"hugepages must be {PRIVATE_PER_RUN!r}, the run's own directory inside the host's 1 GiB hugetlbfs mount "
            f"(OQ-075 (1b)); it has no default; got {hugepages!r}"
        )
    if not isinstance(reset_filter, bool):
        raise ValueError(
            "reset_filter must be true or false, whether the sandbox's seccomp filter refuses the driver's reset "
            f"ioctl (OQ-075 (4a)); it has no default; got {reset_filter!r}"
        )


def program_environment(tree: Path, workdir: Path, limits: Limits) -> dict[str, str]:
    """Return a silicon program's environment in `workdir`: the ttsim row's settings without TT_METAL_SIMULATOR.

    TT_METAL_RUNTIME_ROOT is the pinned tree; the JIT cache and the logs are
    <workdir>/@tt-silicon/cache and <workdir>/@tt-silicon/logs; and
    TT_METAL_THREADCOUNT is limits.cpus. No HOME and no TT_VISIBLE_DEVICES
    (see the module docstring); the caller's environment is never read.
    """
    reserved = workdir / RESERVED_DIR
    return {
        "PATH": SANDBOX_PATH,
        "TMPDIR": "/tmp",
        "LANG": "C",
        "LC_ALL": "C",
        "TT_METAL_SLOW_DISPATCH_MODE": "1",
        "TT_METAL_DISABLE_SFPLOADMACRO": "1",
        "TT_METAL_RUNTIME_ROOT": str(tree),
        "TT_METAL_CACHE": str(reserved / CACHE_DIR),
        "TT_METAL_LOGS_PATH": str(reserved / LOGS_DIR),
        "TT_METAL_INSPECTOR_RPC": "0",
        "TT_METAL_THREADCOUNT": str(limits.cpus),
    }


def _ending(result: SandboxResult) -> str | None:
    """Return why a run's ending quarantines the cards (a hang, a kill, or a crash), or None for a clean exit.

    A clean exit has status 0 to 127 with neither hang nor killed; anything
    else counts (the module docstring, item 6).
    """
    status = f"exit status {result.returncode}"
    if result.hang:
        return f"a hang at the wall limit ({status})"
    if result.killed:
        return f"a kill before the wall limit, by a memory or CPU-time limit or the program itself ({status})"
    if result.returncode == -1:
        return f"the runner's own timeout ({status})"
    if not 0 <= result.returncode <= 127:
        return f"a death by signal, or a status the shell's form cannot tell from one ({status})"
    return None


def _raised(error: BaseException) -> str | None:
    """Return why an exception from the sandbox quarantines the cards, or None when the program never started.

    Every SandboxUnavailableError but SandboxCopyBackError comes before the
    program starts; the copy-back failure comes after it ran, and any other
    exception (an interrupt) may come while it holds the cards.
    """
    if isinstance(error, SandboxCopyBackError):
        return (
            "the sandbox ran the program but did not finish the copy-back of its workdir (SandboxCopyBackError), "
            "so how the program ended is unknown"
        )
    if isinstance(error, SandboxUnavailableError):
        return None
    return f"the run ended by {type(error).__name__} while the program held the cards"


def _writable_again(function: Callable[..., object], path: str, excinfo: object) -> None:
    """shutil.rmtree's error handler: give `path` and its directory their owner's rwx back, links skipped, and retry.

    Only an unlink or an rmdir is retried, once, and a second failure raises,
    which ends that pass; a directory that could not be opened or listed is
    left to _remove_tree's second pass, which can read it now.
    """
    for target in (os.path.dirname(path), path):
        try:
            mode = os.lstat(target).st_mode
            if not stat.S_ISLNK(mode):
                os.chmod(target, stat.S_IMODE(mode) | stat.S_IRWXU)
        except OSError:
            continue
    if function in (os.unlink, os.rmdir):
        function(path)


def _remove_tree(path: Path) -> str | None:
    """Remove `path` with everything in it, links not followed; return why it could not be, or None once it is gone.

    shutil.rmtree runs with _writable_again as its error handler, so entries
    the program made read-only go too, and runs once more when that pass
    fails. The last error is returned, never raised, so a directory left
    behind cannot hide the run's result.
    """
    try:
        shutil.rmtree(path, onerror=_writable_again)
    except OSError:
        if not os.path.lexists(path):
            return None
        try:
            shutil.rmtree(path, onerror=_writable_again)
        except OSError as error:
            return f"{type(error).__name__}: {error}"
    return None


def _kernel_flock() -> Callable[[int, int], object]:
    """Return fcntl.flock, imported now; SandboxUnavailableError on a host without it (Windows)."""
    try:
        import fcntl
    except ImportError:
        raise SandboxUnavailableError(
            "fcntl.flock, which the tt_silicon lock takes, does not exist on this host; nothing ran"
        ) from None
    return fcntl.flock


def _recorded_pid(fd: int) -> str | None:
    """Return the pid a lock holder wrote at the start of the lock file open at `fd`, as printable text, or "".

    None when the file cannot be read.
    """
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        data = os.read(fd, 64)
    except OSError:
        return None
    return printable(data.decode("ascii", errors="replace").strip())


def _release(fd: int, flock: Callable[[int, int], object]) -> None:
    """Clear the pid in the lock file open at `fd`, release the lock, and close `fd`.

    A failure to clear the pid or to release is passed over, since closing
    the descriptor drops the lock anyway (the next holder writes its own
    pid), so it never hides the run's result or its own exception.
    """
    try:
        with suppress(OSError):
            os.ftruncate(fd, 0)
        with suppress(OSError):
            flock(fd, LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def card_lock(locks: Path, flock: Callable[[int, int], object] | None = None) -> Iterator[None]:
    """Hold the one lock of every card, `locks`/LOCK_FILE, for the block (OQ-075 (2a)).

    The executor's runs and `lassi train --silicon` (task P17.38) share it.
    The file is opened without truncation and taken with LOCK_EX | LOCK_NB,
    never a wait: another holder is SandboxUnavailableError naming the file
    and the pid that holder wrote there (or that it cannot be read), and the
    descriptor is closed whether or not that read works. Once taken, this
    process's pid is written there; on the way out the pid is cleared and the
    lock released, on every path (_release). `flock` None means fcntl.flock,
    imported only now.
    """
    flock = _kernel_flock() if flock is None else flock
    locks.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = locks / LOCK_FILE
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
    try:
        flock(fd, LOCK_EX | LOCK_NB)
    except BlockingIOError:
        try:
            holder = _recorded_pid(fd)
        finally:
            os.close(fd)
        if holder is None:
            pid = "the pid it wrote there cannot be read"
        else:
            pid = f"pid {holder}, as it wrote there" if holder else "it wrote no pid there"
        raise SandboxUnavailableError(
            f"the tt_silicon lock {path} is held by another process ({pid}): one silicon program runs at a time "
            "(OQ-075 (2a); Agent Rule 9); nothing ran"
        ) from None
    except BaseException:
        os.close(fd)
        raise
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        yield
    finally:
        _release(fd, flock)


def busy_reasons(root: Path, uid: int, needed: int, owner: Callable[[Path], int]) -> list[str]:
    """Return why the cards under `root` are busy now: another account's hugepage file or UMD lock, or few pages.

    `owner` gives a path's uid (lstat's); a directory, a link, and this
    account's own files do not count, and fewer than `needed` free 1 GiB
    pages does.
    """
    found: list[str] = []
    for directory, prefix in ((HUGEPAGE_POINT, ""), (SHM, UMD_LOCK_PREFIX)):
        for path in devices._entries(_host(root, directory)):
            if not path.name.startswith(prefix) or path.is_symlink() or not path.is_file():
                continue
            holder = owner(path)
            if holder != uid:
                found.append(f"{directory}/{path.name} belongs to another account (uid {holder})")
    free = _count(_host(root, f"{POOL_DIR}/free_hugepages"))
    if free is None or free < needed:
        read = "nothing" if free is None else str(free)
        found.append(f"{POOL_DIR}/free_hugepages reads {read}, fewer than the {needed} 1 GiB pages the cards need")
    return found


def wait_for_cards(
    root: Path, uid: int, needed: int, owner: Callable[[Path], int], clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    """Wait until no other account holds the cards, checking every POLL_INTERVAL_S for at most POLL_LIMIT_S.

    Cards that stay busy are SandboxUnavailableError naming each reason
    (busy_reasons); nothing is claimed or marked (OQ-075 (5)).
    """
    start = clock()
    while True:
        busy = busy_reasons(root, uid, needed, owner)
        if not busy:
            return
        left = POLL_LIMIT_S - (clock() - start)
        if left <= 0:
            raise SandboxUnavailableError(
                f"the Tenstorrent cards stayed busy for {POLL_LIMIT_S} s: {'; '.join(busy)}; nothing ran"
            )
        sleep(min(POLL_INTERVAL_S, left))


@register("Executor", "tt_silicon")
class TtSiliconExecutor:
    """Runs a TT host program on every Wormhole card of the host, only inside the sandbox (see the module docstring).

    The config keys are `hugepages` and `reset_filter`, each required with
    no default, and `harness`. The keyword settings `hidden_roots`,
    `toolchains`, and `sandbox` are the ttsim executor's (None means
    $LASSI_SCRATCH and $HOME, $LASSI_TOOLCHAINS, and a Sandbox with the
    default runner). The rest are seams for the tests: `root`, the host
    root every host path is read under; `access`, the read-and-write check
    asked about each card node (access(2), which opens nothing); `owner`, a
    path's uid (lstat's st_uid); `uid`, this account's (None means
    os.getuid(), read when the cards are polled); `clock` and `sleep` of the
    poll; and `flock` (None means fcntl.flock, imported only when a lock is
    taken). `cards` holds what construction read, `driver` the driver's
    version, and `pins` toolchains/tt-metal.pin's pairs.
    """

    name = "tt_silicon"
    capabilities = frozenset({"runs_code", "sandboxed", SILICON})
    config_keys = frozenset({"hugepages", "reset_filter", "harness"})

    def __init__(
        self,
        *,
        hugepages: str | None = None,
        reset_filter: bool | None = None,
        harness: str | None = None,
        hidden_roots: Sequence[str] | None = None,
        toolchains: str | None = None,
        sandbox: Sandbox | None = None,
        root: Path = devices.HOST_ROOT,
        access: Callable[[Path], bool] = devices._accessible,
        owner: Callable[[Path], int] = _owner,
        uid: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        flock: Callable[[int, int], object] | None = None,
    ) -> None:
        """Check the settings, then read and check every card of the host; nothing is opened and no process starts.

        The settings are refused with a ValueError naming the key before any
        host file is read; the host with DeviceUnavailable, naming the host
        path (see the module docstring).
        """
        _check_settings(hugepages, reset_filter)
        if isinstance(hidden_roots, (str, os.PathLike)):
            raise ValueError(f"hidden_roots must be a sequence of paths, got {hidden_roots!r}")
        self.reset_filter: bool = bool(reset_filter)
        self.harness: Path | None = None if harness is None else Path(harness)
        self.hidden_roots: tuple[Path, ...] | None = (
            None if hidden_roots is None else tuple(Path(item) for item in hidden_roots)
        )
        self.toolchains: Path | None = None if toolchains is None else Path(toolchains)
        self.sandbox: Sandbox = Sandbox() if sandbox is None else sandbox
        self.root, self.owner, self.uid = Path(root), owner, uid
        self.clock, self.sleep, self.flock = clock, sleep, flock
        self.pins: dict[str, dict[str, str]] = {TT_METAL_PIN: _metal_pin()}
        self.scratch: Path = _scratch_root()
        self.locks: Path = self.scratch / LOCKS_DIR
        _refuse_quarantine(self.locks, DeviceUnavailable)
        names = _card_names(self.root)
        _check_nodes(self.root, names, access)
        self.cards: tuple[Card, ...] = tuple(_read_card(self.root, name) for name in names)
        self.driver: str = _driver_version(self.root)
        _check_host(self.root, len(self.cards))

    def device(self) -> str:
        """Return the cards, their firmware, the driver, and the pinned tt-metal, as construction read them.

        For example "Tenstorrent Wormhole n300 x 8 (silicon): card 0 serial
        <s>, firmware bundle <b>, ARC firmware <a>; card 1 ...; tenstorrent
        driver <v>; tt-metal 5280a9cf": one line of printable ASCII. It
        reads no file and starts no process.
        """
        cards = "; ".join(
            f"card {card.name} serial {card.serial}, firmware bundle {card.bundle}, ARC firmware {card.arc}"
            for card in self.cards
        )
        text = (
            f"Tenstorrent Wormhole {CARD_TYPE} x {len(self.cards)} (silicon): {cards}; tenstorrent driver "
            f"{self.driver}; tt-metal {self.pins[TT_METAL_PIN]['VERSION']}"
        )
        return printable(" ".join(text.split()))

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Run `artifact` with `inputs` once on every card, in the sandbox under `limits`; return its RunResult.

        The request and the install are checked first (ValueError, or
        SandboxUnavailableError); then the lock, the quarantine check, the
        poll, the run's hugepage directory, and the one run under the
        quarantine marker, kept unless the run ends cleanly or never
        started, as the module docstring says.
        """
        artifact = Path(artifact)
        if not artifact.is_absolute():
            raise ValueError(f"the artifact must be an absolute path, got {str(artifact)!r}")
        if isinstance(inputs, str):
            raise ValueError(f"inputs must be a sequence of strings, got {inputs!r}")
        workdir = _checked_workdir(artifact)
        mounts = self._mounts()
        with self._lock():
            _refuse_quarantine(self.locks, SandboxUnavailableError)
            self._poll()
            before = _regular_files(workdir)
            result = self._placed(mounts, workdir, [str(artifact), *inputs], limits)
        after = _regular_files(workdir)
        reserved = RESERVED_DIR + "/"
        outputs = {path: after[path] for path in sorted(after) if path not in before and not path.startswith(reserved)}
        return RunResult(
            exit_code=result.returncode,
            hang=result.hang,
            stdout=result.stdout,
            stderr=result.stderr,
            output_files=outputs,
            wall_s=result.wall_s,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            workdir_incomplete=result.workdir_incomplete,
            diagnostics=read_jit_diagnostics(result.stdout, result.stderr, workdir),
        )

    def _mounts(self) -> _Mounts:
        """Return the run's mounts after checking the pinned tt-metal tree as the ttsim executor checks it.

        The toolchains root is the configured one, else $LASSI_TOOLCHAINS,
        resolved; none, or a tree that is not the pinned install, is
        SandboxUnavailableError. The hidden roots are the configured or
        environment ones, the scratch root (so no program sees the lock or a
        marker), and the runs root, resolved.
        """
        root = self.toolchains
        if root is None:
            root = _absolute_variable(_TOOLCHAINS_VARIABLE, "the root of the pinned tt-metal install")
        if root is None:
            raise SandboxUnavailableError(
                f"no toolchains root: set ${_TOOLCHAINS_VARIABLE} or pass toolchains; the tt_silicon executor runs "
                "programs against the pinned tt-metal install there; nothing ran"
            )
        root = _resolved(root)
        metal = self.pins[TT_METAL_PIN]
        tree = root / metal["PREFIX_NAME"]
        try:
            check_install_record(tree, metal)
            check_cpm_sources(tree)
        except ValueError as error:
            message = f"the tt-metal tree {tree} is not the pinned install: {error}; nothing ran"
            raise SandboxUnavailableError(message) from None
        roots = _environment_roots() if self.hidden_roots is None else self.hidden_roots
        hidden = (*(_resolved(item) for item in roots), _resolved(self.scratch), runs_root().resolve())
        harness = None if self.harness is None else _resolved(self.harness)
        return _Mounts(hidden_roots=hidden, harness=harness, root=root, tree=tree)

    def _lock(self) -> AbstractContextManager[None]:
        """Return the one lock of every card (card_lock under $LASSI_SCRATCH/locks, with this executor's flock)."""
        return card_lock(self.locks, self.flock)

    def _poll(self) -> None:
        """Wait until no other account holds every card of the host (wait_for_cards, with this executor's seams)."""
        uid = os.getuid() if self.uid is None else self.uid
        wait_for_cards(self.root, uid, PAGES_PER_CARD * len(self.cards), self.owner, self.clock, self.sleep)

    def _placed(self, mounts: _Mounts, workdir: Path, argv: list[str], limits: Limits) -> SandboxResult:
        """Make the run's hugepage directory, run `argv` once in the sandbox under the marker, and clean up.

        The spec and the sandbox command are built first, so a ValueError
        there writes no marker; then the run goes through _sandboxed. On
        every path the directory is removed with everything in it
        (_remove_tree); when it cannot be, the marker names it beside any
        ending already recorded, and the run's result or exception stands.
        """
        for name in (CACHE_DIR, LOGS_DIR):
            (workdir / RESERVED_DIR / name).mkdir(parents=True, exist_ok=True)
        run_dir = f"{RUN_DIR_PREFIX}{os.getpid()}-{secrets.token_hex(4)}"
        local = _host(self.root, HUGEPAGE_POINT) / run_dir
        local.mkdir(mode=0o700)
        endings: list[str] = []
        try:
            os.chmod(local, 0o700)
            spec = self._spec(mounts, workdir, run_dir, limits)
            sandbox_command(spec, argv, limits)
            return self._sandboxed(spec, argv, limits, endings)
        finally:
            left = _remove_tree(local)
            if left is not None:
                endings.append(f"the run's hugepage directory {HUGEPAGE_POINT}/{run_dir} could not be removed: {left}")
                self._quarantine(endings)

    def _sandboxed(self, spec: SandboxSpec, argv: list[str], limits: Limits, endings: list[str]) -> SandboxResult:
        """Run once in the sandbox with the quarantine marker in place, and settle the marker by the ending.

        The marker is written before the call, naming the ending UNRECORDED,
        so a kill of this process leaves it. A clean ending (_ending) or a
        SandboxUnavailableError that is not SandboxCopyBackError (_raised:
        the program never started) removes it; any other ending is appended
        to `endings` and the marker rewritten to name them. After an
        exception a failure to settle the marker is passed over, since the
        marker written first stays and the exception propagates.
        """
        self._quarantine([UNRECORDED])
        try:
            result = self.sandbox.run(spec, argv, limits)
        except BaseException as error:
            with suppress(OSError):
                self._settle(_raised(error), endings)
            raise
        self._settle(_ending(result), endings)
        return result

    def _settle(self, ending: str | None, endings: list[str]) -> None:
        """Remove the marker when `ending` is None; otherwise append it to `endings` and write the marker with them."""
        if ending is None:
            (self.locks / QUARANTINE_FILE).unlink(missing_ok=True)
            return
        endings.append(ending)
        self._quarantine(endings)

    def _spec(self, mounts: _Mounts, workdir: Path, run_dir: str, limits: Limits) -> SandboxSpec:
        """Return the run's SandboxSpec: the ttsim executor's mounts, the environment, and every card's exposure."""
        nodes = [PurePosixPath(f"{NODE_DIR}/{card.name}") for card in self.cards]
        pci_dirs = [PurePosixPath(f"{PCI_DEVICES}/{card.address}") for card in self.cards]
        exposure = TenstorrentExposure(
            nodes=tuple(DeviceBind(node, node) for node in nodes),
            pci_dirs=tuple(DeviceBind(directory, directory) for directory in pci_dirs),
            hugepages=PurePosixPath(HUGEPAGE_POINT) / run_dir,
            reset_filter=self.reset_filter,
        )
        return SandboxSpec(
            workdir=workdir,
            hidden_roots=mounts.hidden_roots,
            harness=mounts.harness,
            toolchains=mounts.root,
            environment=program_environment(mounts.tree, workdir, limits),
            devices=exposure,
        )

    def _quarantine(self, endings: Sequence[str]) -> None:
        """Write the quarantine marker under $LASSI_SCRATCH/locks, naming each ending and every card (OQ-075 (3a))."""
        written = datetime.now(timezone.utc).isoformat(timespec="seconds")
        lines = [
            f"tt_silicon quarantine (OQ-075 (3a)), written {written} by pid {os.getpid()}",
            *(f"ending: {ending}" for ending in endings),
            *(f"card {card.name}: serial {card.serial}, PCI {card.address}" for card in self.cards),
            "Every later tt_silicon run is refused until a human checks these cards and removes this file.",
        ]
        self.locks.mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.locks / QUARANTINE_FILE).write_bytes("".join(f"{printable(line)}\n" for line in lines).encode("ascii"))
