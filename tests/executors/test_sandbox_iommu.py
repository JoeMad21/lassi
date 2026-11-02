"""Tests for UMD's view of each card's IOMMU group inside the sandbox (task P17.54).

Bible: Execution Backends (the tt_silicon row), Sandbox, Agent Rule 6;
plans/p17-portable.md, task P17.54; plans/spikes/tt-silicon-executor.md,
Finding 2.

UMD at the pin (0450f1be, the UMD of tt-metal 5280a9cf) decides each card's
IOMMU mode from one file. PCIDevice's constructor calls detect_iommu (umd
device/pcie/pci_device.cpp:361), which reads the attribute iommu_group/type
through try_read_sysfs (:102-108, the attribute at :103) and counts a value
that starts with DMA as an IOMMU (:105). try_read_sysfs opens
/sys/bus/pci/devices/<domain>:<bus>:<device>.<function>/<attribute> (:47-77,
the path at :49-55, the open at :56), and a file it cannot open reads as no
IOMMU. On the host /sys/bus/pci/devices/<address> is a link to the card's
device directory below /sys/devices, and that directory's iommu_group is a
relative link that climbs as many levels as the directory lies below /sys,
to /sys/kernel/iommu_groups/<n>: on alpha01 eight levels, every card in a
group of its own (the read-only probe rx 20261029-191025-exec-99dd). Bound
at /sys/bus/pci/devices/<address>, four levels below /sys, the link climbed
past the root, so UMD inside read no IOMMU whatever the host's domain
(Finding 2), and under a DMA domain it would map its buffers as under
pass-through.

The contract these tests fix (lassi.executors.sandbox): a Tenstorrent
exposure's PCI directory is a DeviceBind whose source is the card's device
directory below /sys/devices (what the host's bus link resolves to) and
whose target is /sys/bus/pci/devices/<address>; the setup binds the source
read-only at its own path below its /sys/devices cover, whose inode cap it
first raises to what the binds need, and makes the target a link with the
host's text (pci_link_text), so the walk climbs to /sys/kernel, which the
setup never covers. The run's hugepage directory may be None, for a run
whose every card is in a DMA domain (UMD then maps no hugepage file); the
script then has neither the hugepage step nor the hugepage exception of the
mount check.

The tests build a SYNTHETIC host sysfs tree in alpha01's shape (the probe's
paths and groups, with each mode's group type), run the exposure's setup
script under the stub harness of test_sandbox.py (nothing is mounted),
rebuild the namespace the recorded calls make, and walk UMD's path in it as
the kernel walks a path (links followed, ".." taken from the path walked so
far, so a bind's root climbs to its mount point's parent), outside and
inside, under identity and DMA domains alike. Paths are only strings: no
file under /sys is read. No value here is a measurement.
"""

from __future__ import annotations

import posixpath
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import pytest
from test_sandbox import ONE_SECOND, PROGRAM, cover_remounts, moves, sample_spec, script_of, the_check_call
from test_sandbox_tenstorrent import stub_run_of

PCI_DEVICES = "/sys/bus/pci/devices"
SYS_DEVICES = "/sys/devices"
GROUPS = "/sys/kernel/iommu_groups"
HUGEPAGES = "/dev/hugepages-1G/lassi-run-0001"
# The two switch paths above the cards on alpha01, and each card's address, its parent below /sys/devices, and its
# IOMMU group (the read-only probe rx 20261029-191025-exec-99dd, 2026-10-29).
SWITCH_A = "pci0000:e2/0000:e2:01.1/0000:e3:00.0/0000:e4:00.0/0000:e5:00.0"
SWITCH_B = "pci0000:e2/0000:e2:01.1/0000:e3:00.0/0000:e4:01.0/0000:ee:00.0"
CARDS = (
    ("0000:e8:00.0", f"{SWITCH_A}/0000:e6:01.0", 135),
    ("0000:e9:00.0", f"{SWITCH_A}/0000:e6:02.0", 136),
    ("0000:ea:00.0", f"{SWITCH_A}/0000:e6:03.0", 137),
    ("0000:eb:00.0", f"{SWITCH_A}/0000:e6:04.0", 138),
    ("0000:f1:00.0", f"{SWITCH_B}/0000:ef:01.0", 150),
    ("0000:f2:00.0", f"{SWITCH_B}/0000:ef:02.0", 151),
    ("0000:f3:00.0", f"{SWITCH_B}/0000:ef:03.0", 152),
    ("0000:f4:00.0", f"{SWITCH_B}/0000:ef:04.0", 153),
)
# Card 0's two links as the probe printed them.
CARD0_BUS_LINK = (
    "../../../devices/pci0000:e2/0000:e2:01.1/0000:e3:00.0/0000:e4:00.0/0000:e5:00.0/0000:e6:01.0/0000:e8:00.0"
)
CARD0_GROUP_LINK = "../../../../../../../../kernel/iommu_groups/135"
# The group types: identity is what every card's group read on alpha01 (the probe); DMA and DMA-FQ are the
# translating types UMD counts as an IOMMU (pci_device.cpp:105), SYNTHETIC here.
MODES = ("identity", "DMA", "DMA-FQ")
# The directories alpha01's eight cards need below the /sys/devices/pci0000:e2 cover: e2:01.1, e3:00.0, e4:00.0 and
# e4:01.0, e5:00.0 and ee:00.0, the eight downstream ports, and the eight cards; with the cover's root, 23 inodes.
CARD_COVER = "/sys/devices/pci0000:e2"
CARD_COVER_INODES = 23


@pytest.fixture
def sandbox() -> ModuleType:
    """Return the lassi.executors.sandbox module."""
    from lassi.executors import sandbox

    return sandbox


# ---------------------------------------------------------------------------
# A SYNTHETIC sysfs tree, and the kernel's walk of a path through it


@dataclass(frozen=True)
class Tree:
    """A SYNTHETIC host tree by absolute POSIX path: directories, files with their text, and links with theirs."""

    dirs: frozenset[str]
    files: Mapping[str, str]
    links: Mapping[str, str]


Node = tuple[str | None, str]


def host_tree(mode: str) -> Tree:
    """Return alpha01's card sysfs in miniature, every card's group of type `mode`; every parent is a directory."""
    dirs: set[str] = {"/sys/bus/pci/slots", PCI_DEVICES, "/sys/class", "/sys/devices/system/cpu"}
    files: dict[str, str] = {}
    links: dict[str, str] = {}
    for address, parent, group in CARDS:
        device = f"{SYS_DEVICES}/{parent}/{address}"
        below_sys = len(device.split("/")) - 2
        dirs.update({device, f"{GROUPS}/{group}/devices"})
        links[f"{PCI_DEVICES}/{address}"] = f"../../../devices/{parent}/{address}"
        links[f"{device}/iommu_group"] = "../" * below_sys + f"kernel/iommu_groups/{group}"
        links[f"{GROUPS}/{group}/devices/{address}"] = f"../../../../devices/{parent}/{address}"
        files[f"{device}/revision"] = "0x01\n"
        files[f"{GROUPS}/{group}/type"] = f"{mode}\n"
    for path in list(dirs) + list(files) + list(links):
        parent = posixpath.dirname(path)
        while parent not in ("/", ""):
            dirs.add(parent)
            parent = posixpath.dirname(parent)
    return Tree(dirs=frozenset(dirs), files=files, links=links)


def host_node(tree: Tree, path: str) -> Node:
    """Return (kind, text) of `path` in the host tree: ("link", its text), ("file", its text), ("dir", ""), or None."""
    if path in tree.links:
        return "link", tree.links[path]
    if path in tree.files:
        return "file", tree.files[path]
    return ("dir", "") if path == "/" or path in tree.dirs else (None, "")


def walk(node_at: Callable[[str], Node], path: str, depth: int = 0) -> str | None:
    """Return where `path` leads as the kernel walks it, links followed, or None when a step is missing.

    ".." drops the last step walked, never past the root, so from a bind's
    root it climbs to the mount point's parent, as the kernel's does.
    """
    if depth > 40:
        return None
    walked: list[str] = []
    parts = [part for part in path.split("/") if part and part != "."]
    for index, part in enumerate(parts):
        if part == "..":
            walked = walked[:-1]
            continue
        here = "/" + "/".join([*walked, part])
        kind, text = node_at(here)
        if kind is None or (kind == "file" and index != len(parts) - 1):
            return None
        if kind == "link":
            base = "/" + "/".join(walked)
            found = walk(node_at, text if text.startswith("/") else posixpath.join(base, text), depth + 1)
            if found is None:
                return None
            walked = [part for part in found.split("/") if part]
            continue
        walked.append(part)
    return "/" + "/".join(walked)


def read(node_at: Callable[[str], Node], path: str) -> str | None:
    """Return the text of the file `path` leads to, or None when it leads to no file (UMD reads no value)."""
    found = walk(node_at, path)
    if found is None:
        return None
    kind, text = node_at(found)
    return text if kind == "file" else None


def cover_of(path: str) -> str | None:
    """Return the entry the setup covers that `path` is or lies under: /sys/class, /sys/bus, a /sys/devices entry."""
    parts = path.split("/")
    if parts[1:2] == ["sys"] and len(parts) >= 3 and parts[2] in ("class", "bus"):
        return f"/sys/{parts[2]}"
    if parts[1:3] == ["sys", "devices"] and len(parts) >= 4 and parts[3] != "system":
        return f"{SYS_DEVICES}/{parts[3]}"
    return None


@dataclass(frozen=True)
class Inside:
    """The namespace a stub run's calls make over the host tree: its binds, its directories, and its links."""

    tree: Tree
    binds: Mapping[str, str]
    made: frozenset[str]
    links: Mapping[str, str] = field(default_factory=dict)

    def __call__(self, path: str) -> Node:
        """Return (kind, text) of `path` inside: a bind shows its host source, a cover only what the setup made."""
        bound = [target for target in self.binds if path == target or path.startswith(target + "/")]
        if bound:
            target = max(bound, key=len)
            return host_node(self.tree, self.binds[target] + path[len(target) :])
        cover = cover_of(path)
        if cover is None:
            return host_node(self.tree, path)
        if path in self.links:
            return "link", self.links[path]
        return ("dir", "") if path == cover or path in self.made else (None, "")


def inside_of(tree: Tree, calls: list[list[str]]) -> Inside:
    """Return the namespace the recorded calls make: staged /sys binds, `mkdir -p` under /sys, and `ln -s` there."""
    staged = {target: source for _index, source, target in moves(calls, "--bind") if target.startswith("/tmp/expose/")}
    binds = {target: staged[source] for _index, source, target in moves(calls, "--bind") if source in staged}
    made: set[str] = set()
    for call in calls:
        if call[0] == "mkdir" and call[-1].startswith("/sys/"):
            path = call[-1]
            while cover_of(path) is not None and path != cover_of(path):
                made.add(path)
                path = posixpath.dirname(path)
    links = {call[3]: call[2] for call in calls if call[:2] == ["ln", "-s"] and call[3].startswith("/sys/")}
    return Inside(tree=tree, binds=binds, made=frozenset(made), links=links)


# ---------------------------------------------------------------------------
# The exposure, and its setup under stubs


def exposure(sandbox: ModuleType, tree: Tree, *, hugepages: bool) -> Any:
    """Return every card's exposure: its node, its device directory as its bus link resolves it, and hugepages."""
    bind = sandbox.DeviceBind
    pci = []
    for address, _parent, _group in CARDS:
        device = walk(lambda path: host_node(tree, path), f"{PCI_DEVICES}/{address}")
        assert device is not None, address
        pci.append(bind(PurePosixPath(device), PurePosixPath(f"{PCI_DEVICES}/{address}")))
    paths = [PurePosixPath(f"/dev/tenstorrent/{card}") for card in range(len(CARDS))]
    nodes = tuple(bind(path, path) for path in paths)
    return sandbox.TenstorrentExposure(
        nodes=nodes, pci_dirs=tuple(pci), hugepages=PurePosixPath(HUGEPAGES) if hugepages else None, reset_filter=True
    )


_CALLS: dict[bool, tuple[list[str], list[list[str]]]] = {}


def setup_calls(sandbox: ModuleType, factory: pytest.TempPathFactory, *, hugepages: bool) -> tuple[list[str], Any]:
    """Return (command, calls) of a stub run of the eight-card exposure's script; cached per hugepage setting.

    The exposure's paths are the same in every mode (only a group's type
    differs), so one run serves every mode.
    """
    if hugepages not in _CALLS:
        spec = sample_spec(sandbox, devices=exposure(sandbox, host_tree("identity"), hugepages=hugepages))
        command = sandbox.sandbox_command(spec, PROGRAM, ONE_SECOND)
        base = factory.mktemp("iommu-stubs")
        (base / "work").mkdir()
        run = stub_run_of(sandbox, command, base / "run", base / "work")
        assert run.returncode == 7, run
        _CALLS[hugepages] = (command, run.calls)
    return _CALLS[hugepages]


UMD_TYPE = "/sys/bus/pci/devices/{address}/iommu_group/type"


# ---------------------------------------------------------------------------
# The SYNTHETIC tree stands in for alpha01's


def test_the_synthetic_tree_holds_the_links_the_probe_printed() -> None:
    tree = host_tree("identity")
    assert tree.links[f"{PCI_DEVICES}/0000:e8:00.0"] == CARD0_BUS_LINK
    assert tree.links[f"{SYS_DEVICES}/{CARDS[0][1]}/0000:e8:00.0/iommu_group"] == CARD0_GROUP_LINK
    outside = lambda path: host_node(tree, path)  # noqa: E731
    for address, _parent, group in CARDS:
        assert walk(outside, f"{PCI_DEVICES}/{address}/iommu_group") == f"{GROUPS}/{group}"
        assert read(outside, UMD_TYPE.format(address=address)) == "identity\n"


# ---------------------------------------------------------------------------
# Inside, UMD reads the host's IOMMU mode, under identity and DMA domains alike


@pytest.mark.parametrize("hugepages", [True, False], ids=["with-hugepages", "without-hugepages"])
@pytest.mark.parametrize("mode", MODES)
def test_umd_inside_reads_every_cards_group_type_as_the_host_does(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory, mode: str, hugepages: bool
) -> None:
    tree = host_tree(mode)
    _command, calls = setup_calls(sandbox, tmp_path_factory, hugepages=hugepages)
    inside = inside_of(tree, calls)
    outside = lambda path: host_node(tree, path)  # noqa: E731
    for address, parent, group in CARDS:
        path = UMD_TYPE.format(address=address)
        assert read(inside, path) == read(outside, path) == f"{mode}\n", (mode, address)
        assert walk(inside, f"{PCI_DEVICES}/{address}") == f"{SYS_DEVICES}/{parent}/{address}", address
        assert walk(inside, f"{PCI_DEVICES}/{address}/iommu_group") == f"{GROUPS}/{group}", address
        assert read(inside, f"{PCI_DEVICES}/{address}/revision") == "0x01\n", "the card's attributes, as before"
    # UMD's decision (pci_device.cpp:105): an IOMMU exactly when the type starts with DMA, inside as outside.
    seen = read(inside, UMD_TYPE.format(address=CARDS[0][0]))
    assert seen is not None and seen.startswith("DMA") == mode.startswith("DMA")


def test_a_bind_at_the_bus_path_reads_no_iommu_which_is_why_the_directory_is_mirrored(sandbox: ModuleType) -> None:
    # The layout before task P17.54: the card's directory bound at /sys/bus/pci/devices/<address>. Its iommu_group link
    # climbs eight levels from four below /sys, past the root, so UMD reads no type and takes it as no IOMMU.
    tree = host_tree("DMA")
    address, parent, _group = CARDS[0]
    earlier = Inside(
        tree=tree,
        binds={f"{PCI_DEVICES}/{address}": f"{SYS_DEVICES}/{parent}/{address}"},
        made=frozenset({"/sys/bus/pci", PCI_DEVICES}),
    )
    assert read(earlier, f"{PCI_DEVICES}/{address}/revision") == "0x01\n"
    assert walk(earlier, f"{PCI_DEVICES}/{address}/iommu_group") is None
    assert read(earlier, UMD_TYPE.format(address=address)) is None


def test_the_bus_entries_are_links_with_the_hosts_text_and_the_directories_are_bound_at_their_own_paths(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> None:
    tree = host_tree("identity")
    _command, calls = setup_calls(sandbox, tmp_path_factory, hugepages=False)
    inside = inside_of(tree, calls)
    expected = {f"{PCI_DEVICES}/{address}": tree.links[f"{PCI_DEVICES}/{address}"] for address, _p, _g in CARDS}
    assert inside.links == expected
    devices = {f"{SYS_DEVICES}/{parent}/{address}" for address, parent, _group in CARDS}
    assert {target for target in inside.binds if target.startswith(SYS_DEVICES + "/")} == devices
    assert all(inside.binds[target] == target for target in devices), "each directory at its own path"
    assert inside.binds["/sys/bus/pci/slots"] == "/sys/bus/pci/slots"
    for address, parent, _group in CARDS:
        text = sandbox.pci_link_text(PurePosixPath(f"{SYS_DEVICES}/{parent}/{address}"))
        assert text == tree.links[f"{PCI_DEVICES}/{address}"], address


def test_each_card_cover_is_raised_read_only_before_the_first_directory_below_it(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # Every cover holds 8 inodes; alpha01's eight cards need 22 directories below /sys/devices/pci0000:e2.
    _command, calls = setup_calls(sandbox, tmp_path_factory, hugepages=False)
    remounts = cover_remounts(calls, CARD_COVER)
    capped = [(index, options) for index, options in remounts if any(o.startswith("nr_inodes=") for o in options)]
    assert len(capped) == 1, remounts
    index, options = capped[0]
    assert "ro" in options and "rw" not in options and f"nr_inodes={CARD_COVER_INODES}" in options, options
    below = [i for i, call in enumerate(calls) if call[0] == "mkdir" and call[-1].startswith(CARD_COVER + "/")]
    assert below and index < min(below), (index, below)
    assert remounts[-1][0] > max(below) and "ro" in remounts[-1][1], "the cover is read-only again after the binds"


def test_a_bus_link_is_made_with_the_bus_cover_open_and_the_cover_read_only_after(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> None:
    _command, calls = setup_calls(sandbox, tmp_path_factory, hugepages=False)
    remounts = cover_remounts(calls, "/sys/bus")
    made = [i for i, call in enumerate(calls) if call[:2] == ["ln", "-s"] and call[3].startswith(PCI_DEVICES + "/")]
    assert len(made) == len(CARDS), made
    for index in made:
        before = [options for at, options in remounts if at < index]
        assert before and "rw" in before[-1], index
    assert remounts[-1][0] > max(made) and "ro" in remounts[-1][1], remounts
    assert remounts[-1][0] < the_check_call(calls), "read-only before the mount check"


# ---------------------------------------------------------------------------
# Without hugepages: no hugepage step and the plain mount check


def test_without_hugepages_the_script_drops_the_hugepage_step_and_keeps_the_plain_check(
    sandbox: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> None:
    command, calls = setup_calls(sandbox, tmp_path_factory, hugepages=False)
    script = script_of(command)
    for reset_filter in (False, True):
        assert sandbox.TENSTORRENT_SETUP_SCRIPTS[(reset_filter, False)] != sandbox.TENSTORRENT_SETUP_SCRIPTS[
            (reset_filter, True)
        ]
    assert script == sandbox.TENSTORRENT_SETUP_SCRIPTS[(True, False)]
    assert "hugepages-1G" not in script and "run_root" not in script, "no hugepage bind and no hugepage exception"
    assert calls[the_check_call(calls)] == ["awk", sandbox.MOUNT_CHECK, "/proc/self/mountinfo"]
    assert not [call for call in calls if any("hugepages-1G" in part for part in call)], "nothing at the point"
    with_pages, _ = setup_calls(sandbox, tmp_path_factory, hugepages=True)
    assert script_of(with_pages) == sandbox.TENSTORRENT_SETUP_SCRIPTS[(True, True)]
    assert HUGEPAGES in with_pages and HUGEPAGES not in command


# ---------------------------------------------------------------------------
# The spec refuses a PCI directory it cannot bind at its own path


def _bind(sandbox: ModuleType, source: str, target: str) -> Any:
    """Return a DeviceBind of two POSIX paths."""
    return sandbox.DeviceBind(PurePosixPath(source), PurePosixPath(target))


CARD0 = f"{SYS_DEVICES}/{CARDS[0][1]}/0000:e8:00.0"
CARD0_BUS = f"{PCI_DEVICES}/0000:e8:00.0"
# PCI directory pairs the spec refuses: case -> (the binds, a text the ValueError holds).
BAD_SOURCES: dict[str, tuple[list[tuple[str, str]], str]] = {
    "the-bus-link-itself": ([(CARD0_BUS, CARD0_BUS)], "/sys/devices"),
    "under-the-uncovered-system-entry": ([("/sys/devices/system/0000:e8:00.0", CARD0_BUS)], "/sys/devices"),
    "a-cover-root": ([("/sys/devices/0000:e8:00.0", CARD0_BUS)], "/sys/devices"),
    "under-the-class-cover": ([("/sys/class/tenstorrent/0000:e8:00.0", CARD0_BUS)], "/sys/devices"),
    "one-below-another": (
        [(CARD0, CARD0_BUS), (f"{CARD0}/0000:e9:00.0", f"{PCI_DEVICES}/0000:e9:00.0")],
        "under",
    ),
}


@pytest.mark.parametrize("case", sorted(BAD_SOURCES))
def test_the_spec_refuses_a_pci_directory_it_cannot_bind_at_its_own_path(sandbox: ModuleType, case: str) -> None:
    pairs, text = BAD_SOURCES[case]
    node = _bind(sandbox, "/dev/tenstorrent/0", "/dev/tenstorrent/0")
    devices = sandbox.TenstorrentExposure(
        nodes=(node,), pci_dirs=tuple(_bind(sandbox, *pair) for pair in pairs), hugepages=None, reset_filter=False
    )
    with pytest.raises(ValueError) as caught:
        sample_spec(sandbox, devices=devices)
    assert "pci_dirs" in str(caught.value) and text in str(caught.value), str(caught.value)


def test_the_spec_takes_no_hugepage_directory_and_keeps_it_none(sandbox: ModuleType, tmp_path: Path) -> None:
    devices = exposure(sandbox, host_tree("DMA"), hugepages=False)
    assert sample_spec(sandbox, devices=devices).devices.hugepages is None
