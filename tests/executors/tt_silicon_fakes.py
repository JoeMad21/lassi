"""Shared fakes for the tests of the tt_silicon executor (task P17.16); not a test module.

- write_host(root, cards) writes a SYNTHETIC host tree under `root` that
  stands in for the paths the executor reads on a host with Wormhole cards
  (plans/spikes/tt-silicon-executor.md, Findings 1 to 4):
  - dev/tenstorrent/<n>, one empty regular file per card, and
    dev/tenstorrent/by-id/wormhole-SYNTHETIC<n>, a file standing in for
    udev's link (the executor skips every name that is not all digits);
  - dev/hugepages-1G/ with a libvirt/ directory in it (root's, as on
    alpha01, spike probe 7aeb) and dev/shm/, both otherwise empty;
  - proc/mounts, whose two hugetlbfs lines are the ones spike probe 7aeb
    printed on alpha01;
  - sys/class/tenstorrent/tenstorrent!<n>/ with tt_card_type, tt_serial,
    tt_fw_bundle_ver, tt_arc_fw_ver, and tt_eth_fw_ver, and a plain device/
    directory holding uevent (with PCI_SLOT_NAME=<address>) and
    iommu_group/type. On the host, device and iommu_group are links; Windows
    cannot make links without a privilege this workstation lacks, so here
    they are directories, and the card's PCI address is read from the
    uevent text, which is what the kernel's PCI uevent carries;
  - sys/bus/pci/slots/ with two SYNTHETIC slot entries,
    sys/module/tenstorrent/version, and the 1 GiB pool's nr_hugepages and
    free_hugepages under sys/kernel/mm/hugepages/hugepages-1048576kB/.
  No PCI address is a file or directory name, since Windows refuses ':' in
  names.
- Access stands in for access(2) on the card nodes; Owners for the owner of
  a path (the uid lstat gives on Linux; Windows has no uids); Clock for
  time.monotonic and time.sleep; FakeFlock for fcntl.flock, with the
  operation values of Linux's <sys/file.h> and an exclusive lock per open
  file, keyed by the file's (st_dev, st_ino) as os.fstat gives them.
- build(module, host, ...) returns a TtSiliconExecutor on a host tree with
  every seam faked.

The card type "n300" is the value tt_card_type read for card 0 on alpha01
(results/tt-silicon-smoke-20261023/loopback/record.json); the PCI addresses
are the eight lspci listed on alpha01 (rx 20261021-201421-exec-4ea0), used
only as text; the two hugetlbfs mount lines are spike probe 7aeb's output.
Every other value (serials, firmware versions, the driver version, uids,
page counts, uevent fields) is SYNTHETIC. No value here is a measurement.
"""

from __future__ import annotations

import errno
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

# The card type the executor accepts: card 0's tt_card_type on alpha01 (the smoke record).
CARD_TYPE = "n300"
# The eight cards' PCI addresses on alpha01 (rx 20261021-201421-exec-4ea0), as text only.
ADDRESSES = (
    "0000:e8:00.0", "0000:e9:00.0", "0000:ea:00.0", "0000:eb:00.0",
    "0000:f1:00.0", "0000:f2:00.0", "0000:f3:00.0", "0000:f4:00.0",
)  # fmt: skip
CARDS = tuple(range(len(ADDRESSES)))
# SYNTHETIC driver version and uids: this account's, another account's, and root's.
DRIVER = "SYNTHETIC-2.8.0"
OWN_UID = 4100
OTHER_UID = 4200
ROOT_UID = 0
# The pool's SYNTHETIC page counts (alpha01's pool held 32 pages, spike probe 7aeb), and the pages a card's run maps.
POOL_PAGES = 32
PAGES_PER_CARD = 2
# The hugetlbfs lines spike probe 7aeb printed from alpha01's /proc/mounts, and SYNTHETIC lines around them.
HUGEPAGE_LINE = "hugetlbfs /dev/hugepages-1G hugetlbfs rw,nosuid,nodev,relatime,mode=777,pagesize=1024M 0 0"
SMALL_PAGE_LINE = "hugetlbfs /dev/hugepages hugetlbfs rw,relatime,pagesize=2M 0 0"
OTHER_LINES = (
    "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
    "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0",
    "tmpfs /dev/shm tmpfs rw,nosuid,nodev 0 0",
)
# The host paths the executor reads, as POSIX text.
NODE_DIR = "/dev/tenstorrent"
HUGEPAGE_POINT = "/dev/hugepages-1G"
SHM = "/dev/shm"
CLASS_DIR = "/sys/class/tenstorrent"
SLOTS = "/sys/bus/pci/slots"
DRIVER_VERSION = "/sys/module/tenstorrent/version"
POOL_DIR = "/sys/kernel/mm/hugepages/hugepages-1048576kB"
MOUNTS = "/proc/mounts"
# The sysfs attributes the executor records for each card (smoke.py's card_facts reads the same files).
CARD_FILES = ("tt_card_type", "tt_serial", "tt_fw_bundle_ver", "tt_arc_fw_ver", "tt_eth_fw_ver")
# flock(2) operation values from Linux's <sys/file.h>, which fcntl gives on Linux.
LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 4, 8
# The one value the hugepages setting takes (OQ-075 (1b)).
PRIVATE_PER_RUN = "private-per-run"


def address(card: int) -> str:
    """Return card `card`'s PCI address: alpha01's for the first eight, a SYNTHETIC one past them."""
    return ADDRESSES[card] if card < len(ADDRESSES) else f"0000:{0x10 + card:02x}:00.0"


def serial(card: int) -> str:
    """Return card `card`'s SYNTHETIC serial."""
    return f"SYNTHETIC{card:08d}"


def bundle(card: int) -> str:
    """Return card `card`'s SYNTHETIC firmware bundle version."""
    return f"SYNTHETIC-bundle-19.5.{card}"


def arc(card: int) -> str:
    """Return card `card`'s SYNTHETIC ARC firmware version."""
    return f"SYNTHETIC-arc-2.37.{card}"


def uevent(card: int) -> str:
    """Return a SYNTHETIC PCI uevent for card `card`, in the kernel's KEY=value layout, with its PCI_SLOT_NAME."""
    lines = (
        "DRIVER=tenstorrent",
        "PCI_CLASS=SYNTHETIC",
        "PCI_ID=SYNTHETIC",
        "PCI_SUBSYS_ID=SYNTHETIC",
        f"PCI_SLOT_NAME={address(card)}",
        "MODALIAS=pci:SYNTHETIC",
    )
    return "".join(f"{line}\n" for line in lines)


def mounts_text(*lines: str) -> str:
    """Return /proc/mounts text: OTHER_LINES, then `lines` (default: the 2M and the 1G hugetlbfs lines)."""
    chosen = lines or (SMALL_PAGE_LINE, HUGEPAGE_LINE)
    return "".join(f"{line}\n" for line in (*OTHER_LINES, *chosen))


@dataclass(frozen=True)
class Host:
    """A SYNTHETIC host tree: its root and its cards' numbers."""

    root: Path
    cards: tuple[int, ...]

    def at(self, host_path: str) -> Path:
        """Return the root-joined path of a host path such as /dev/hugepages-1G."""
        return self.root.joinpath(*PurePosixPath(host_path).relative_to("/").parts)

    def class_dir(self, card: int) -> Path:
        """Return card `card`'s class directory, sys/class/tenstorrent/tenstorrent!<card>, under the root."""
        return self.at(f"{CLASS_DIR}/tenstorrent!{card}")

    def class_path(self, card: int) -> str:
        """Return card `card`'s class directory as the host path a message names."""
        return f"{CLASS_DIR}/tenstorrent!{card}"

    def pool(self, name: str) -> Path:
        """Return the 1 GiB pool's file `name` (nr_hugepages or free_hugepages) under the root."""
        return self.at(f"{POOL_DIR}/{name}")

    def needed_pages(self) -> int:
        """Return the free 1 GiB pages a run of every card needs: PAGES_PER_CARD per card."""
        return PAGES_PER_CARD * len(self.cards)


def write_text(path: Path, text: str) -> Path:
    """Write `text` at `path` as ASCII with LF, making its parent directories; return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("ascii"))
    return path


def write_host(root: Path, cards: Iterable[int] = CARDS, *, pages: int = POOL_PAGES, free: int | None = None) -> Host:
    """Write the SYNTHETIC host tree at `root` for `cards`; return it. `free` defaults to `pages`."""
    numbers = tuple(cards)
    host = Host(root=Path(root), cards=numbers)
    for card in numbers:
        write_text(host.at(f"{NODE_DIR}/{card}"), "")
        write_text(host.at(f"{NODE_DIR}/by-id/wormhole-SYNTHETIC{card}"), "")
        files = {
            "tt_card_type": CARD_TYPE,
            "tt_serial": serial(card),
            "tt_fw_bundle_ver": bundle(card),
            "tt_arc_fw_ver": arc(card),
            "tt_eth_fw_ver": "SYNTHETIC-eth-7.5.0",
        }
        for name, value in files.items():
            write_text(host.class_dir(card) / name, f"{value}\n")
        write_text(host.class_dir(card) / "device" / "uevent", uevent(card))
        write_text(host.class_dir(card) / "device" / "iommu_group" / "type", "identity\n")
    host.at(f"{HUGEPAGE_POINT}/libvirt").mkdir(parents=True)
    host.at(SHM).mkdir(parents=True)
    write_text(host.at(MOUNTS), mounts_text())
    for slot in ("1", "2"):
        write_text(host.at(f"{SLOTS}/{slot}/address"), "SYNTHETIC\n")
    write_text(host.at(DRIVER_VERSION), f"{DRIVER}\n")
    write_text(host.pool("nr_hugepages"), f"{pages}\n")
    write_text(host.pool("free_hugepages"), f"{pages if free is None else free}\n")
    return host


class Access:
    """A stand-in access check: it records each path asked about, relative to `root`, and denies those in `deny`."""

    def __init__(self, root: Path, deny: Iterable[str] = ()) -> None:
        """Answer True except for the relative POSIX paths in `deny`."""
        self.root = Path(root)
        self.deny = set(deny)
        self.asked: list[str] = []

    def __call__(self, path: Path) -> bool:
        """Record the path (relative to the root, which it must lie under) and answer."""
        relative = Path(path).relative_to(self.root).as_posix()
        self.asked.append(relative)
        return relative not in self.deny


class Owners:
    """A stand-in for a path's owner: the uid in `uids` by relative POSIX path, else OWN_UID; root's for libvirt/."""

    def __init__(self, root: Path, uids: Mapping[str, int] | None = None) -> None:
        """Keep the root and the uids; dev/hugepages-1G/libvirt is root's unless `uids` says otherwise."""
        self.root = Path(root)
        self.uids: dict[str, int] = {"dev/hugepages-1G/libvirt": ROOT_UID, **dict(uids or {})}
        self.asked: list[str] = []

    def __call__(self, path: Path) -> int:
        """Record a path under the root (relative) and return its uid; a path outside the root is OWN_UID's."""
        try:
            relative = Path(path).relative_to(self.root).as_posix()
        except ValueError:
            return OWN_UID
        self.asked.append(relative)
        return self.uids.get(relative, OWN_UID)


@dataclass
class Clock:
    """A stand-in for time.monotonic and time.sleep: sleep advances `now` and calls `on_sleep(<sleeps so far>)`."""

    now: float = 1000.0
    sleeps: list[float] = field(default_factory=list)
    on_sleep: Callable[[int], None] | None = None

    def __call__(self) -> float:
        """Return the fake time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Record the sleep, advance the fake time, and run the hook."""
        assert seconds >= 0, f"a negative sleep: {seconds}"
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep(len(self.sleeps))


def identity(fd: int) -> tuple[int, int] | None:
    """Return (st_dev, st_ino) of an open descriptor, or None when it is not open."""
    try:
        info = os.fstat(fd)
    except OSError:
        return None
    return info.st_dev, info.st_ino


def path_identity(path: Path) -> tuple[int, int]:
    """Return (st_dev, st_ino) of a file."""
    info = os.stat(path)
    return info.st_dev, info.st_ino


class FakeFlock:
    """flock(2) within this process: one exclusive holder per file, kept until LOCK_UN or until its fd is closed.

    LOCK_EX | LOCK_NB on a file another open descriptor holds raises
    BlockingIOError (EWOULDBLOCK), as flock does; a blocking LOCK_EX there
    fails the test, since it would wait. Every call is recorded as
    (fd, operation).
    """

    def __init__(self) -> None:
        """Start with no holder."""
        self.holders: dict[tuple[int, int], int] = {}
        self.calls: list[tuple[int, int]] = []

    def _prune(self) -> None:
        """Drop each holder whose descriptor is closed or now names another file (flock's release on close)."""
        for key, fd in list(self.holders.items()):
            if identity(fd) != key:
                del self.holders[key]

    def __call__(self, fd: int, operation: int) -> None:
        """Apply `operation` to the file open at `fd`."""
        self.calls.append((fd, operation))
        self._prune()
        key = identity(fd)
        assert key is not None, f"flock on a closed descriptor {fd}"
        if operation & LOCK_UN:
            if self.holders.get(key) == fd:
                del self.holders[key]
            return
        assert operation & LOCK_EX, f"only exclusive locks are expected, got {operation}"
        holder = self.holders.get(key)
        if holder is not None and holder != fd:
            if operation & LOCK_NB:
                raise BlockingIOError(errno.EWOULDBLOCK, "SYNTHETIC: another open file holds the lock")
            raise AssertionError("a blocking flock on a held file would wait")
        self.holders[key] = fd

    def held(self, path: Path) -> bool:
        """Return True when some open descriptor holds the lock on the file at `path`."""
        self._prune()
        return path_identity(path) in self.holders

    def held_files(self, directory: Path) -> list[Path]:
        """Return the files directly in `directory` that a descriptor holds."""
        self._prune()
        if not directory.is_dir():
            return []
        return [path for path in sorted(directory.iterdir()) if path.is_file() and path_identity(path) in self.holders]


def build(
    module: ModuleType,
    host: Host,
    *,
    sandbox: Any,
    toolchains: Path | None = None,
    clock: Clock | None = None,
    flock: FakeFlock | None = None,
    access: Access | None = None,
    owners: Owners | None = None,
    **settings: Any,
) -> Any:
    """Return module.TtSiliconExecutor on `host` with every seam faked; `settings` add or replace keywords.

    The defaults are hugepages "private-per-run", reset_filter True, device_profiler "off" (task P17.42), the
    host root, Access and Owners on it, uid OWN_UID, a fresh Clock (also as
    sleep), and a fresh FakeFlock.
    """
    clock = Clock() if clock is None else clock
    values: dict[str, Any] = {
        "hugepages": PRIVATE_PER_RUN,
        "reset_filter": True,
        "device_profiler": "off",
        "root": host.root,
        "access": Access(host.root) if access is None else access,
        "owner": Owners(host.root) if owners is None else owners,
        "uid": OWN_UID,
        "clock": clock,
        "sleep": clock.sleep,
        "flock": FakeFlock() if flock is None else flock,
        "sandbox": sandbox,
    }
    if toolchains is not None:
        values["toolchains"] = str(toolchains)
    values.update(settings)
    return module.TtSiliconExecutor(**values)
