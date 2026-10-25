"""Tests for the tt_silicon executor, registered as Executor "tt_silicon" (task P17.16).

Bible: Execution Backends (the tt_silicon row), Component Interfaces
(Executor, device()), Sandbox, Agent Rules 1, 2, 6, 7, 9, and 10;
plans/p17-portable.md, task P17.16; OQ-072's answer (a silicon run gets every
Wormhole card in the cluster); OQ-075's approved recommendations (1b), (2a),
(3a), (4a), (6a), and (8a), and part (5)'s direction, "Be sure to poll and
check first" (plans/LESSONS.md, Owner directions); the design in
plans/spikes/tt-silicon-executor.md, section 6, with every card in place of
one; and the sandbox's TenstorrentExposure (task P17.15).

The interface these tests assume, which task P17.16 adds:

- lassi.executors.tt_silicon registers TtSiliconExecutor as Executor
  "tt_silicon" with exactly the capabilities runs_code, sandboxed, and
  silicon, and the config keys hugepages, reset_filter, and harness. The
  capability `silicon` is proposed here as lassi.core.capabilities SILICON:
  an executor that runs programs on accelerator silicon, which only a human
  starts (Agent Rule 9), so the runner refuses it without `lassi run
  --silicon` (tests/core/test_runner_silicon.py). It declares neither
  simulator nor takes_device: every card is exposed, so there is no card key.
- Settings, each required with no default (OQ-075): hugepages, whose one
  value is "private-per-run" (1b), and reset_filter, a bool (4a). An absent
  value or any other is a ValueError naming the key, raised before any host
  file is read.
- Keyword settings for the runner and tests, as the gpu executor has
  them: harness, hidden_roots, toolchains, and sandbox (the native
  executor's); root, the host root every host path is read under (default
  "/"); access, the read-and-write check asked about each card node
  (default access(2), which opens nothing); owner, a path's uid (default
  lstat's st_uid); uid, this account's (default os.getuid()); clock and
  sleep (default time.monotonic and time.sleep); and flock (default None,
  meaning fcntl.flock, imported only when a lock is taken, so the module
  imports on Windows). The fakes pass flock Linux's <sys/file.h> values
  (LOCK_EX 2, LOCK_NB 4, LOCK_UN 8), which fcntl gives on Linux.
- Construction reads the host under root from file metadata and sysfs or
  procfs text only, opens no device node, and starts no process. It takes
  every all-digit name in /dev/tenstorrent, in numeric order (by-id and
  any other name skipped), and refuses with DeviceUnavailable, naming the
  host path and never the root-joined one: no card; a node that is missing
  or that access denies; a card whose class directory
  /sys/class/tenstorrent/tenstorrent!<n> is missing, whose tt_card_type is
  not n300, whose tt_serial, tt_fw_bundle_ver, or tt_arc_fw_ver is missing
  (every result records them), whose device/uevent gives no PCI_SLOT_NAME,
  or whose device/iommu_group/type starts with DMA (spike, Finding 2: inside
  the bind UMD would read no IOMMU); no /sys/module/tenstorrent/version, no
  /sys/bus/pci/slots, no 1 GiB nr_hugepages or fewer than 2 pages per card
  in it; no /dev/hugepages-1G directory, or no /proc/mounts line of type
  hugetlbfs at /dev/hugepages-1G with pagesize=1024M; more numbered nodes
  than the sandbox's TENSTORRENT_CARDS_MAX; and a quarantine marker.
  $LASSI_SCRATCH unset or relative is refused too, naming it.
- run(artifact, inputs, limits), with the ttsim executor's workdir and
  install rules (the pinned tt-metal tree only; no ttsim library):
  1. it refuses while a quarantine marker exists (SandboxUnavailableError
     naming it);
  2. the lock (OQ-075 (2a)): one file directly under $LASSI_SCRATCH/locks,
     which the sandbox hides, taken with flock LOCK_EX | LOCK_NB and held
     through the cleanup; another holder makes the run refuse with
     SandboxUnavailableError naming the file and the pid the holder wrote
     in it, so the file is opened without truncation before the lock is
     taken; it is released on every path, and the file stays in place
     between runs (removing it would let two processes lock two files); a
     holder whose pid cannot be read refuses the run too, its descriptor
     closed, and a failure to clear the pid never hides the run's own
     exception;
  3. the poll (OQ-075 (5)): the cards are busy while another account owns a
     regular file directly in /dev/hugepages-1G or a TT_UMD_LOCK* file in
     /dev/shm, or while fewer than 2 x <cards> 1 GiB pages are free
     (free_hugepages); directories (libvirt's) and this account's own files
     do not count. It checks, then sleeps at most POLL_INTERVAL_S, until the
     cards are free or POLL_LIMIT_S has passed, and then refuses with
     SandboxUnavailableError naming what stayed busy. The sandbox never runs
     while a card is busy;
  4. the run's own hugepage directory (OQ-075 (1b)): new, one segment below
     /dev/hugepages-1G, mode 0700, empty when the program starts; afterwards
     every entry in it is removed, links not followed, and then the
     directory, on every path, entries the program made read-only included;
     nothing else in /dev/hugepages-1G changes. When the directory still
     cannot be removed, the quarantine marker names it, and the run's
     result or exception stands;
  5. one sandboxed run, never retried and never rerun with Watcher (Agent
     Rule 9), whose SandboxSpec holds a TenstorrentExposure: every card's
     node at its own path; each card's PCI directory at
     /sys/bus/pci/devices/<address>, from a source under /sys that ends in
     that address (the bus path itself, which a bind follows on the host, or
     the directory it resolves to); the run's hugepage directory; and the
     reset_filter setting. The environment is the ttsim row's without
     TT_METAL_SIMULATOR (OQ-075 (8a)), the JIT cache and logs under a
     reserved directory of the workdir whose name starts with "@", no HOME,
     and no TT_VISIBLE_DEVICES: with every node bound, UMD enumerates every
     numbered node set or unset (spike, Finding 1), setting it costs more
     opens of every node, and the sandbox's ENVIRONMENT_NAMES does not hold
     it;
  6. the quarantine (OQ-075 (3a)): a marker whose name holds "quarantine"
     goes under $LASSI_SCRATCH/locks, naming every card's serial, once the
     spec is built and before the sandbox is called, so it is in place
     while the program may run and a kill of this process leaves it. It is
     removed only after a clean ending (status 0 to 127, neither hang nor
     killed) or a SandboxUnavailableError other than the sandbox's
     SandboxCopyBackError (the program never started). Every other ending
     keeps it, rewritten to name the ending: a hang, a kill, a status
     outside 0 to 127 (a crash is a death by signal, which the sandbox
     reports as 128 + N, and the shell's form cannot tell it from a
     program's own status of 128 or more; -1 is the runner's own timeout),
     the copy-back failure (SandboxCopyBackError, raised after the program
     ran), and any other exception (an interrupt), which then propagates.
     A ValueError from building the spec writes no marker;
  7. its RunResult is the ttsim executor's without simulator readings:
     sim_ub and sim_gap stay None, and the diagnostics are the kernel JIT's
     (lassi.executors.ttsim read_jit_diagnostics).
- device() names "Tenstorrent Wormhole n300 x <N> (silicon)", each card's
  serial, firmware bundle, and ARC firmware in card order, the tenstorrent
  driver's version, and the pinned tt-metal, from what construction read,
  as one line of printable ASCII; never simulator. pins are tt-metal's
  alone.

Every host tree is SYNTHETIC (tt_silicon_fakes), under tmp_path; no test
reads the real /dev or /sys. No sandbox or process starts (fake Sandboxes
record each run; one Linux-only test starts a Python helper that holds a
real flock). Wall times are PLACEHOLDER values, never performance. No value
in this module is a measurement.
"""

from __future__ import annotations

import builtins
import importlib
import inspect
import io
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import pytest
from test_gpu import module_problems
from tt_silicon_fakes import (
    DRIVER_VERSION,
    HUGEPAGE_POINT,
    LOCK_EX,
    LOCK_NB,
    MOUNTS,
    NODE_DIR,
    OTHER_UID,
    PRIVATE_PER_RUN,
    SHM,
    SLOTS,
    Access,
    Clock,
    FakeFlock,
    Host,
    Owners,
    address,
    arc,
    build,
    bundle,
    identity,
    mounts_text,
    path_identity,
    serial,
    uevent,
    write_host,
    write_text,
)
from ttsim_fakes import FakeSandbox, canned, capture, make_artifact, make_install, names_path, same_path

import lassi.executors  # noqa: F401  (importing the package registers every executor)
from lassi.core.devices import DeviceUnavailable
from lassi.core.interfaces import Limits, RunResult
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.executors import sandbox as sandbox_module
from lassi.executors import ttsim as ttsim_module
from lassi.executors.sandbox import SANDBOX_PATH, SandboxUnavailableError
from lassi.toolchains import pins as pins_module

LIMITS = Limits(wall_s=2.5, memory_mb=256, cpus=2)
INPUTS = ["--size", "8"]
CAPABILITIES = frozenset({"runs_code", "sandboxed", "silicon"})
CONFIG_KEYS = frozenset({"hugepages", "reset_filter", "harness"})
KEYWORDS = frozenset(
    {
        "hugepages", "reset_filter", "harness", "hidden_roots", "toolchains", "sandbox", "root", "access", "owner",
        "uid", "clock", "sleep", "flock",
    }
)  # fmt: skip
MISSING = object()
# The environment names of a silicon run: the ttsim row's without TT_METAL_SIMULATOR (OQ-075 (8a)).
SILICON_NAMES = frozenset(
    {
        "PATH", "TMPDIR", "LANG", "LC_ALL", "TT_METAL_SLOW_DISPATCH_MODE", "TT_METAL_DISABLE_SFPLOADMACRO",
        "TT_METAL_RUNTIME_ROOT", "TT_METAL_CACHE", "TT_METAL_LOGS_PATH", "TT_METAL_INSPECTOR_RPC",
        "TT_METAL_THREADCOUNT",
    }
)  # fmt: skip
# A ttsim finding line (the shape of the P4.9 fixtures), which a silicon run never reads as a finding.
FINDING_LINE = "[4500] ERROR: NonContractualBehavior: rv32_mem_rd: unaligned addr=0x16dfe2 size=4\n"
# A Python helper for the Linux-only test: it holds a real exclusive flock on argv[1], writes its pid there, says so.
HOLDER = (
    "import fcntl, os, sys, time\n"
    "fd = os.open(sys.argv[1], os.O_RDWR)\n"
    "fcntl.flock(fd, fcntl.LOCK_EX)\n"
    "os.ftruncate(fd, 0)\n"
    "os.write(fd, str(os.getpid()).encode('ascii'))\n"
    "print('held', flush=True)\n"
    "time.sleep(120)\n"
)


# ---------------------------------------------------------------------------
# Modules, roots, and helpers


@pytest.fixture(scope="module")
def module() -> ModuleType:
    """Import lassi.executors.tt_silicon, failing each test clearly while it does not exist."""
    try:
        return importlib.import_module("lassi.executors.tt_silicon")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.executors.tt_silicon does not exist yet (task P17.16): {error}")


@dataclass(frozen=True)
class Roots:
    """The test's runs root, scratch root (with its locks directory), home, and TMPDIR."""

    runs: Path
    scratch: Path
    home: Path
    tmpdir: Path

    @property
    def locks(self) -> Path:
        """Return $LASSI_SCRATCH/locks, where the lock and the quarantine marker go."""
        return self.scratch / "locks"


@pytest.fixture(autouse=True)
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Roots:
    """Set the runs, scratch, and home roots and TMPDIR under tmp_path, and unset $LASSI_TOOLCHAINS."""
    found = Roots(tmp_path / "runs", tmp_path / "scratch", tmp_path / "home", tmp_path / "tmpdir")
    for path in (found.runs, found.scratch, found.home, found.tmpdir):
        path.mkdir()
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(found.runs))
    monkeypatch.setenv("LASSI_SCRATCH", str(found.scratch))
    monkeypatch.setenv("HOME", str(found.home))
    monkeypatch.setenv("TMPDIR", str(found.tmpdir))
    monkeypatch.delenv("LASSI_TOOLCHAINS", raising=False)
    return found


@pytest.fixture
def host(tmp_path: Path) -> Host:
    """Write the SYNTHETIC host tree of eight cards under tmp_path/host."""
    return write_host(tmp_path / "host")


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build the PLACEHOLDER toolchains root whose tt-metal tree the install check accepts (ttsim_fakes)."""
    return make_install(tmp_path, monkeypatch)


def attribute(module: ModuleType, name: str) -> Any:
    """Return module.<name>, failing the test clearly while task P17.16 has not added it."""
    if not hasattr(module, name):
        pytest.fail(f"{module.__name__} has no {name} yet (task P17.16)")
    return getattr(module, name)


def copy_back_error() -> type[SandboxUnavailableError]:
    """Return the sandbox's SandboxCopyBackError, raised after the program ran, failing clearly while it is missing."""
    found = attribute(sandbox_module, "SandboxCopyBackError")
    assert isinstance(found, type) and issubclass(found, SandboxUnavailableError), found
    return found


def posix(path: object) -> str:
    """Return a path as POSIX text: its as_posix() when it has one, else str()."""
    method = getattr(path, "as_posix", None)
    return method() if callable(method) else str(path)


def pairs(binds: Iterable[Any]) -> list[tuple[str, str]]:
    """Return each DeviceBind as (source, target) POSIX text."""
    return [(posix(bind.source), posix(bind.target)) for bind in binds]


def script_of(command: list[str]) -> str:
    """Return the script element of a sandbox command: the element after "sh", "-c"."""
    start = command.index("sh")
    assert command[start + 1] == "-c", command
    return command[start + 2]


def remove(path: Path) -> None:
    """Remove a file or a directory tree."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def run_dir(host: Host, spec: Any) -> Path:
    """Return the root-joined path of the run's hugepage directory that a spec's exposure names."""
    return host.at(posix(spec.devices.hugepages))


def tree(directory: Path) -> dict[str, bytes | None]:
    """Return every entry under `directory` as relative POSIX path -> bytes (None for a directory)."""
    found: dict[str, bytes | None] = {}
    for dirpath, dirnames, filenames in os.walk(directory):
        base = Path(dirpath)
        for name in dirnames:
            found[(base / name).relative_to(directory).as_posix()] = None
        for name in filenames:
            found[(base / name).relative_to(directory).as_posix()] = (base / name).read_bytes()
    return found


def markers(roots: Roots) -> list[Path]:
    """Return the quarantine markers under $LASSI_SCRATCH/locks: the files whose name holds "quarantine"."""
    if not roots.locks.is_dir():
        return []
    return sorted(path for path in roots.locks.iterdir() if "quarantine" in path.name)


def not_root_joined(message: str, host: Host) -> None:
    """Assert that a message names host paths, never the root-joined ones."""
    assert str(host.root) not in message and host.root.as_posix() not in message, message


def guard_node_opens(monkeypatch: pytest.MonkeyPatch, host: Host) -> list[str]:
    """Fail any open of a path under the host tree's /dev/tenstorrent; return the opens recorded there.

    builtins.open, io.open, os.open, and pathlib.Path.open are wrapped (Python
    3.10's pathlib opens through its own accessor). The open is recorded
    before AssertionError is raised, so code that catches it still fails.
    """
    nodes = os.path.normcase(os.path.abspath(host.at(NODE_DIR)))
    opened: list[str] = []

    def note(target: Any) -> None:
        if isinstance(target, int):
            return
        path = os.path.normcase(os.path.abspath(os.fspath(target)))
        if path == nodes or path.startswith(nodes + os.sep):
            opened.append(path)
            raise AssertionError(f"a card node was opened: {path}")

    real_open, real_os_open, real_path_open = builtins.open, os.open, pathlib.Path.open

    def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        note(file)
        return real_open(file, *args, **kwargs)

    def guarded_os_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        note(path)
        return real_os_open(path, *args, **kwargs)

    def guarded_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        note(self)
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(pathlib.Path, "open", guarded_path_open)
    return opened


def trap_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any process start fail the test: subprocess.Popen and os.system raise AssertionError."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"a process was started: {args!r}")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(os, "system", refuse)


class ExplodingSandbox:
    """A stand-in sandbox: any use of it fails the test."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the sandbox was used ({name})")


class RaisingSandbox:
    """A stand-in sandbox whose run applies `effect` (when set) and then raises `error`; it records each call."""

    def __init__(self, error: BaseException, effect: Callable[[Any, list[str]], None] | None = None) -> None:
        """Keep the error and the effect."""
        self.error = error
        self.effect = effect
        self.calls: list[Any] = []

    def run(self, spec: Any, argv: Sequence[str], limits: Limits) -> Any:
        """Record the call, apply the effect, and raise the error."""
        self.calls.append(spec)
        if self.effect is not None:
            self.effect(spec, list(argv))
        raise self.error


def umd_files(host: Host) -> Callable[[Any, list[str]], None]:
    """Return an effect that checks the run's hugepage directory is empty, then writes into it as UMD and a program.

    It writes UMD's two sysmem file names for card 0 and a nested
    directory, as a program inside could.
    """

    def effect(spec: Any, argv: list[str]) -> None:
        directory = run_dir(host, spec)
        assert directory.is_dir(), f"the run's hugepage directory {directory} does not exist when the program starts"
        assert list(directory.iterdir()) == [], "the run's hugepage directory is not empty when the program starts"
        (directory / "tenstorrent").write_bytes(b"SYNTHETIC sysmem bytes\n")
        (directory / "device_0_channel_1_tenstorrent").write_bytes(b"SYNTHETIC sysmem bytes\n")
        (directory / "stray" / "deeper").mkdir(parents=True)
        (directory / "stray" / "deeper" / "left").write_bytes(b"SYNTHETIC\n")

    return effect


def silicon(
    module: ModuleType,
    host: Host,
    install: Any,
    results: Sequence[Any] = (),
    effects: Sequence[Any] = (),
    **settings: Any,
) -> tuple[Any, FakeSandbox]:
    """Return an executor on `host` with a FakeSandbox of `results` and `effects`, and that sandbox.

    The toolchains root is the install's unless `settings` names another
    (None: none, so $LASSI_TOOLCHAINS, which the tests unset, applies).
    """
    fake = FakeSandbox(results=list(results) or [canned()], effects=list(effects))
    toolchains = settings.pop("toolchains", install.root)
    return build(module, host, sandbox=fake, toolchains=toolchains, **settings), fake


# ---------------------------------------------------------------------------
# Registration, the capability, the constructor, and the module


def test_registered_as_executor_tt_silicon_with_its_capabilities(module: ModuleType) -> None:
    entry = DEFAULT_REGISTRY.get("Executor", "tt_silicon")
    assert entry.factory is module.TtSiliconExecutor
    assert entry.capabilities == CAPABILITIES, "runs_code, sandboxed, and silicon; never simulator or takes_device"
    assert entry.config_keys == CONFIG_KEYS, "no card key: every card is exposed (OQ-072)"
    assert module.TtSiliconExecutor.name == "tt_silicon"
    assert not callable(getattr(module.TtSiliconExecutor, "framework", None)), "it runs a built program, no framework"


def test_the_silicon_capability_is_named_once_in_lassi_core_capabilities() -> None:
    from lassi.core import capabilities

    assert getattr(capabilities, "SILICON", None) == "silicon", "lassi.core.capabilities needs SILICON (task P17.16)"
    assert capabilities.SILICON != capabilities.SIMULATOR


def test_package_imports_and_re_exports_tt_silicon(module: ModuleType) -> None:
    import lassi.executors as package

    assert package.tt_silicon is module, "importing lassi.executors alone registers the executor"
    assert package.TtSiliconExecutor is module.TtSiliconExecutor
    assert {"TtSiliconExecutor", "tt_silicon"} <= set(package.__all__)
    assert '"tt_silicon"' in (package.__doc__ or ""), "the package docstring lists every executor it registers"


def test_the_constructor_takes_the_two_settings_and_keyword_seams(module: ModuleType) -> None:
    params = inspect.signature(module.TtSiliconExecutor).parameters
    assert set(params) == KEYWORDS
    assert all(param.kind is inspect.Parameter.KEYWORD_ONLY for param in params.values())
    assert Path(params["root"].default) == Path("/")
    assert params["access"].default is importlib.import_module("lassi.executors.devices")._accessible
    assert params["clock"].default is time.monotonic and params["sleep"].default is time.sleep
    assert params["flock"].default is None, "None means fcntl.flock, imported only when a lock is taken"
    assert all(params[name].default is None for name in ("harness", "hidden_roots", "toolchains", "sandbox"))
    assert list(inspect.signature(module.TtSiliconExecutor.run).parameters) == ["self", "artifact", "inputs", "limits"]


def test_the_module_is_documented_typed_ascii_and_writes_down_its_choices(module: ModuleType) -> None:
    doc, missing, untyped = module_problems(module)
    assert not missing, missing
    assert not untyped, untyped
    for term in ("OQ-072", "OQ-075", "Agent Rule 9", "private-per-run", "reset_filter", "flock", "quarantine"):
        assert term in doc, term
    for term in ("OQ-078", "SandboxCopyBackError", "TENSTORRENT_CARDS_MAX"):
        assert term in doc, f"the docstring names {term}: the runner's refusal, the kept marker, the card limit"
    assert "TT_VISIBLE_DEVICES" in doc and "unset" in doc, "the docstring says TT_VISIBLE_DEVICES stays unset, and why"
    assert "128" in doc, "the docstring says how a crash differs from a clean nonzero exit"
    assert "POLL_INTERVAL_S" in doc and "POLL_LIMIT_S" in doc


def test_the_poll_runs_every_few_seconds_for_a_few_minutes(module: ModuleType) -> None:
    # The owner: "the previous user takes a few minutes to relieve the cards. Be sure to poll and check first."
    interval, limit = attribute(module, "POLL_INTERVAL_S"), attribute(module, "POLL_LIMIT_S")
    assert isinstance(interval, (int, float)) and 0 < interval <= 10, "every few seconds"
    assert isinstance(limit, (int, float)) and limit >= 120, "a few minutes"
    assert limit > interval


# ---------------------------------------------------------------------------
# The two settings: required, no default, one value each


@pytest.mark.parametrize(
    "value",
    [MISSING, None, "", "private-per-card", "host", "PRIVATE-PER-RUN", " private-per-run", 1, True],
    ids=["absent", "none", "empty", "per-card", "host", "upper-case", "padded", "int", "bool"],
)
def test_hugepages_takes_only_private_per_run_before_any_host_file_is_read(
    module: ModuleType, tmp_path: Path, value: Any
) -> None:
    empty = tmp_path / "empty"
    access, owners = Access(empty), Owners(empty)
    settings: dict[str, Any] = {"reset_filter": True}
    if value is not MISSING:
        settings["hugepages"] = value
    with pytest.raises(ValueError) as caught:
        module.TtSiliconExecutor(
            **settings, root=empty, access=access, owner=owners, uid=4100, sandbox=ExplodingSandbox()
        )
    message = str(caught.value)
    assert "hugepages" in message and PRIVATE_PER_RUN in message, message
    assert access.asked == [] and owners.asked == [], "the setting is refused before any host file is read"


@pytest.mark.parametrize(
    "value", [MISSING, None, "true", 1, 0, "yes"], ids=["absent", "none", "text", "one", "zero", "yes"]
)
def test_reset_filter_must_be_a_bool_before_any_host_file_is_read(
    module: ModuleType, tmp_path: Path, value: Any
) -> None:
    empty = tmp_path / "empty"
    access = Access(empty)
    settings: dict[str, Any] = {"hugepages": PRIVATE_PER_RUN}
    if value is not MISSING:
        settings["reset_filter"] = value
    with pytest.raises(ValueError) as caught:
        module.TtSiliconExecutor(**settings, root=empty, access=access, uid=4100, sandbox=ExplodingSandbox())
    assert "reset_filter" in str(caught.value), str(caught.value)
    assert access.asked == []


# ---------------------------------------------------------------------------
# Construction: every card, from file metadata and sysfs or procfs text


def test_every_numbered_node_is_asked_about_and_nothing_else(module: ModuleType, host: Host) -> None:
    access = Access(host.root)
    build(module, host, sandbox=ExplodingSandbox(), access=access)
    assert sorted(set(access.asked)) == sorted(f"dev/tenstorrent/{card}" for card in host.cards), access.asked


def test_construction_and_device_open_no_card_node_and_start_no_process(
    module: ModuleType, host: Host, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = guard_node_opens(monkeypatch, host)
    trap_processes(monkeypatch)
    build(module, host, sandbox=ExplodingSandbox()).device()
    with pytest.raises(DeviceUnavailable):
        build(module, host, sandbox=ExplodingSandbox(), access=Access(host.root, deny=["dev/tenstorrent/3"]))
    assert opened == [], f"a card node was opened: {opened}"


def _uevent_without_address(host: Host) -> list[str]:
    write_text(host.class_dir(2) / "device" / "uevent", uevent(2).replace(f"PCI_SLOT_NAME={address(2)}\n", ""))
    return [host.class_path(2)]


def _mounts(*lines: str) -> Callable[[Host], list[str]]:
    def change(host: Host) -> list[str]:
        write_text(host.at(MOUNTS), mounts_text(*lines))
        return [HUGEPAGE_POINT]

    return change


def _removed(path: Callable[[Host], Path], named: Callable[[Host], list[str]]) -> Callable[[Host], list[str]]:
    def change(host: Host) -> list[str]:
        remove(path(host))
        return named(host)

    return change


def _written(
    path: Callable[[Host], Path], text: str, named: Callable[[Host], list[str]]
) -> Callable[[Host], list[str]]:
    def change(host: Host) -> list[str]:
        write_text(path(host), text)
        return named(host)

    return change


def _no_numbered_node(host: Host) -> list[str]:
    for card in host.cards:
        remove(host.at(f"{NODE_DIR}/{card}"))
    return [NODE_DIR]


# Each case changes the SYNTHETIC tree and returns the texts the refusal must name (host paths, never root-joined).
REFUSALS: dict[str, Callable[[Host], list[str]]] = {
    "no-node-directory": _removed(lambda h: h.at(NODE_DIR), lambda h: [NODE_DIR]),
    "no-numbered-node": _no_numbered_node,
    "no-class-directory": _removed(lambda h: h.class_dir(5), lambda h: [h.class_path(5)]),
    "no-uevent": _removed(lambda h: h.class_dir(3) / "device" / "uevent", lambda h: [h.class_path(3)]),
    "no-pci-address": _uevent_without_address,
    "not-an-n300": _written(
        lambda h: h.class_dir(1) / "tt_card_type", "SYNTHETIC-p150\n", lambda h: [h.class_path(1), "SYNTHETIC-p150"]
    ),
    "dma-iommu": _written(
        lambda h: h.class_dir(4) / "device" / "iommu_group" / "type", "DMA-FQ\n", lambda h: [h.class_path(4), "DMA"]
    ),
    "no-serial": _removed(lambda h: h.class_dir(6) / "tt_serial", lambda h: [h.class_path(6), "tt_serial"]),
    "no-firmware-bundle": _removed(
        lambda h: h.class_dir(0) / "tt_fw_bundle_ver", lambda h: [h.class_path(0), "tt_fw_bundle_ver"]
    ),
    "no-arc-firmware": _removed(
        lambda h: h.class_dir(7) / "tt_arc_fw_ver", lambda h: [h.class_path(7), "tt_arc_fw_ver"]
    ),
    "no-driver-version": _removed(lambda h: h.at(DRIVER_VERSION), lambda h: [DRIVER_VERSION]),
    "no-slot-list": _removed(lambda h: h.at(SLOTS), lambda h: [SLOTS]),
    "no-page-count": _removed(lambda h: h.pool("nr_hugepages"), lambda h: ["nr_hugepages"]),
    "pool-smaller-than-two-pages-a-card": _written(
        lambda h: h.pool("nr_hugepages"), "15\n", lambda h: ["nr_hugepages", str(h.needed_pages())]
    ),
    "no-hugepage-directory": _removed(lambda h: h.at(HUGEPAGE_POINT), lambda h: [HUGEPAGE_POINT]),
    "no-hugetlbfs-mount": _mounts("tmpfs /run tmpfs rw 0 0"),
    "small-pages-at-the-point": _mounts("hugetlbfs /dev/hugepages-1G hugetlbfs rw,relatime,pagesize=2M 0 0"),
    "tmpfs-at-the-point": _mounts("tmpfs /dev/hugepages-1G tmpfs rw,nosuid,nodev 0 0"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_host_that_cannot_give_every_card_is_refused_at_construction(
    module: ModuleType, host: Host, case: str
) -> None:
    named = REFUSALS[case](host)
    with pytest.raises(DeviceUnavailable) as caught:
        build(module, host, sandbox=ExplodingSandbox())
    message = str(caught.value)
    for text in named:
        assert text in message, f"{case}: {text!r} not in {message!r}"
    not_root_joined(message, host)


def test_a_denied_node_is_refused_and_access_is_asked_only_about_nodes_that_exist(
    module: ModuleType, host: Host
) -> None:
    remove(host.at(f"{NODE_DIR}/6"))
    remove(host.class_dir(6))
    access = Access(host.root, deny=["dev/tenstorrent/3"])
    with pytest.raises(DeviceUnavailable) as caught:
        build(module, host, sandbox=ExplodingSandbox(), access=access)
    assert "/dev/tenstorrent/3" in str(caught.value), str(caught.value)
    not_root_joined(str(caught.value), host)
    assert all(host.root.joinpath(relative).exists() for relative in access.asked)


def test_a_card_without_an_iommu_group_or_an_eth_version_is_accepted(module: ModuleType, host: Host) -> None:
    remove(host.class_dir(0) / "device" / "iommu_group")
    remove(host.class_dir(1) / "tt_eth_fw_ver")
    assert "(silicon)" in build(module, host, sandbox=ExplodingSandbox()).device()


def test_more_cards_than_the_sandbox_can_expose_are_refused_at_construction(
    module: ModuleType, tmp_path: Path
) -> None:
    most = sandbox_module.TENSTORRENT_CARDS_MAX
    fits = write_host(tmp_path / "most", cards=range(most), pages=2 * most)
    assert f"x {most} (silicon)" in build(module, fits, sandbox=ExplodingSandbox()).device()
    access = Access(tmp_path / "more")
    more = write_host(tmp_path / "more", cards=range(most + 1), pages=2 * (most + 1))
    with pytest.raises(DeviceUnavailable) as caught:
        build(module, more, sandbox=ExplodingSandbox(), access=access)
    message = str(caught.value)
    assert NODE_DIR in message and str(most + 1) in message and str(most) in message, message
    assert access.asked == [], "refused before any node is asked about"
    not_root_joined(message, more)


@pytest.mark.parametrize("value", [None, "relative/scratch"], ids=["unset", "relative"])
def test_a_scratch_root_is_required_for_the_lock_and_the_quarantine(
    module: ModuleType, host: Host, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("LASSI_SCRATCH")
    else:
        monkeypatch.setenv("LASSI_SCRATCH", value)
    with pytest.raises((DeviceUnavailable, SandboxUnavailableError)) as caught:
        build(module, host, sandbox=ExplodingSandbox())
    assert "LASSI_SCRATCH" in str(caught.value), str(caught.value)


# ---------------------------------------------------------------------------
# device() and pins


def test_device_names_every_card_its_firmware_the_driver_and_the_pin(module: ModuleType, host: Host) -> None:
    executor = build(module, host, sandbox=ExplodingSandbox())
    line = executor.device()
    assert line.startswith(f"Tenstorrent Wormhole n300 x {len(host.cards)} (silicon)"), line
    places = [line.index(serial(card)) for card in host.cards]
    assert places == sorted(places), "the cards in card order"
    for card in host.cards:
        assert bundle(card) in line and arc(card) in line, card
    assert "SYNTHETIC-2.8.0" in line and "driver" in line
    assert pins_module.read_pin("tt-metal")["VERSION"] in line
    lowered = line.lower()
    assert "simulator" not in lowered and "ttsim" not in lowered and "virtual" not in lowered, "never a simulator"
    assert line.isascii() and line.isprintable() and line == line.strip()


def test_device_keeps_printable_ascii_and_counts_one_card(module: ModuleType, tmp_path: Path) -> None:
    host = write_host(tmp_path / "one", cards=[0])
    (host.class_dir(0) / "tt_serial").write_bytes("SYN\tTHETIC \N{MICRO SIGN}\x07\n".encode("utf-8"))
    line = build(module, host, sandbox=ExplodingSandbox()).device()
    assert line.startswith("Tenstorrent Wormhole n300 x 1 (silicon)"), line
    assert line.isascii() and line.isprintable() and line == line.strip()


def test_device_formats_what_construction_read(module: ModuleType, host: Host, monkeypatch: pytest.MonkeyPatch) -> None:
    executor = build(module, host, sandbox=ExplodingSandbox())
    expected = executor.device()
    for card in host.cards:
        write_text(host.class_dir(card) / "tt_serial", "SYNTHETIC-changed-after-construction\n")
    trap_processes(monkeypatch)
    assert executor.device() == expected


def test_the_pins_are_tt_metal_alone(module: ModuleType, host: Host) -> None:
    executor = build(module, host, sandbox=ExplodingSandbox())
    assert executor.pins == {"tt-metal": pins_module.read_pin("tt-metal")}, "no ttsim pin: nothing is simulated"


# ---------------------------------------------------------------------------
# run(): the workdir, the install, the exposure, the environment


def test_a_bad_request_is_refused_before_any_lock_or_directory(
    module: ModuleType, host: Host, install: Any, roots: Roots, tmp_path: Path
) -> None:
    artifact = make_artifact(roots.runs)
    outside = tmp_path / "elsewhere" / "main"
    outside.parent.mkdir()
    outside.write_bytes(b"PLACEHOLDER artifact, never executed\n")
    flock = FakeFlock()
    executor, fake = silicon(module, host, install, flock=flock)
    before = tree(host.at(HUGEPAGE_POINT))
    for path, inputs in ((Path("relative") / "main", INPUTS), (outside, INPUTS), (artifact, "--size 8")):
        with pytest.raises(ValueError):
            executor.run(path, inputs, LIMITS)
    assert fake.calls == [] and flock.calls == []
    assert tree(host.at(HUGEPAGE_POINT)) == before


def test_an_install_that_is_not_the_pin_is_refused_before_the_program(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    artifact = make_artifact(roots.runs)
    flock = FakeFlock()
    executor, fake = silicon(module, host, install, flock=flock)
    install.record.write_bytes(b"tt-metal SYNTHETIC-other-commit https://example.invalid\n")
    with pytest.raises(SandboxUnavailableError) as caught:
        executor.run(artifact, INPUTS, LIMITS)
    assert names_path(str(caught.value), install.tree), str(caught.value)
    no_toolchains, other = silicon(module, host, install, flock=flock, toolchains=None)
    with pytest.raises(SandboxUnavailableError):
        no_toolchains.run(artifact, INPUTS, LIMITS)
    assert fake.calls == [] and other.calls == []
    assert flock.held_files(roots.locks) == [] and markers(roots) == []
    assert sorted(path.name for path in host.at(HUGEPAGE_POINT).iterdir()) == ["libvirt"]


@pytest.mark.parametrize("reset_filter", [True, False], ids=["reset-filter-on", "reset-filter-off"])
def test_the_exposure_gives_every_card_its_pci_directory_and_the_runs_hugepages(
    module: ModuleType, host: Host, install: Any, roots: Roots, reset_filter: bool
) -> None:
    executor, fake = silicon(module, host, install, reset_filter=reset_filter)
    executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    ((spec, argv, limits),) = [(call.spec, call.argv, call.limits) for call in fake.calls]
    exposure = spec.devices
    assert isinstance(exposure, sandbox_module.TenstorrentExposure)
    nodes = [f"{NODE_DIR}/{card}" for card in host.cards]
    assert pairs(exposure.nodes) == [(node, node) for node in nodes], "every card, each at its own path; no by-id"
    targets = [posix(bind.target) for bind in exposure.pci_dirs]
    assert targets == [f"/sys/bus/pci/devices/{address(card)}" for card in host.cards]
    for bind, card in zip(exposure.pci_dirs, host.cards, strict=True):
        source = posix(bind.source)
        assert source.startswith("/sys/") and source.rsplit("/", 1)[1] == address(card), source
    hugepages = PurePosixPath(posix(exposure.hugepages))
    assert hugepages.parent == PurePosixPath(HUGEPAGE_POINT) and len(hugepages.parts) == 4, hugepages
    assert exposure.reset_filter is reset_filter
    command = sandbox_module.sandbox_command(spec, argv, limits)
    assert script_of(command) == sandbox_module.TENSTORRENT_SETUP_SCRIPTS[reset_filter]


def test_nodes_are_taken_in_numeric_order(module: ModuleType, tmp_path: Path, install: Any, roots: Roots) -> None:
    host = write_host(tmp_path / "eleven", cards=range(11))
    executor, fake = silicon(module, host, install)
    executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    targets = [posix(bind.target) for bind in fake.calls[0].spec.devices.nodes]
    assert targets == [f"{NODE_DIR}/{card}" for card in range(11)], "10 comes after 9"


def test_the_environment_is_the_ttsim_rows_without_the_simulator(
    module: ModuleType, host: Host, install: Any, roots: Roots, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("TT_VISIBLE_DEVICES", "TT_METAL_SIMULATOR", "TT_METAL_WATCHER", "TT_METAL_HOME"):
        monkeypatch.setenv(name, "SYNTHETIC-callers-value")
    artifact = make_artifact(roots.runs)
    workdir = Path(os.path.realpath(artifact.parent))
    seen: dict[str, bool] = {}

    def made(spec: Any, argv: list[str]) -> None:
        environment = dict(spec.environment)
        seen["made"] = Path(environment["TT_METAL_CACHE"]).is_dir() and Path(environment["TT_METAL_LOGS_PATH"]).is_dir()

    executor, fake = silicon(module, host, install, effects=[made])
    executor.run(artifact, INPUTS, LIMITS)
    environment = fake.calls[0].environment
    assert set(environment) == SILICON_NAMES, "no TT_METAL_SIMULATOR, no TT_VISIBLE_DEVICES, no HOME, no Watcher"
    assert environment["PATH"] == SANDBOX_PATH and environment["TMPDIR"] == "/tmp"
    assert environment["LANG"] == "C" and environment["LC_ALL"] == "C"
    assert environment["TT_METAL_SLOW_DISPATCH_MODE"] == "1" and environment["TT_METAL_DISABLE_SFPLOADMACRO"] == "1"
    assert environment["TT_METAL_INSPECTOR_RPC"] == "0"
    assert environment["TT_METAL_THREADCOUNT"] == str(LIMITS.cpus)
    assert same_path(environment["TT_METAL_RUNTIME_ROOT"], install.tree)
    for name in ("TT_METAL_CACHE", "TT_METAL_LOGS_PATH"):
        relative = Path(os.path.realpath(environment[name])).relative_to(workdir)
        assert relative.parts[0].startswith("@") and len(relative.parts) >= 2, (name, relative)
    assert seen == {"made": True}, "the cache and log directories exist before the program starts"


def test_the_spec_keeps_the_native_workdir_roots_and_argv(
    module: ModuleType, host: Host, install: Any, roots: Roots, tmp_path: Path
) -> None:
    harness = tmp_path / "assets" / "harness"
    harness.mkdir(parents=True)
    artifact = make_artifact(roots.runs)
    executor, fake = silicon(module, host, install, harness=str(harness))
    executor.run(artifact, INPUTS, LIMITS)
    (call,) = fake.calls
    assert same_path(call.spec.workdir, artifact.parent) and same_path(call.argv[0], artifact)
    assert call.argv[1:] == INPUTS and call.limits == LIMITS
    assert same_path(call.spec.harness, harness) and same_path(call.spec.toolchains, install.root)
    expected = [os.path.realpath(path) for path in (roots.scratch, roots.home, roots.runs)]
    assert [os.path.realpath(root) for root in call.spec.hidden_roots] == expected, "the locks directory is hidden"


# ---------------------------------------------------------------------------
# The lock: one for every card, under $LASSI_SCRATCH/locks, released on every path


def test_the_lock_is_held_through_the_run_and_released_after(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    flock = FakeFlock()
    held: list[list[Path]] = []

    def check(spec: Any, argv: list[str]) -> None:
        held.append(flock.held_files(roots.locks))

    executor, _ = silicon(module, host, install, flock=flock, effects=[check])
    executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    ((lock,),) = held
    assert lock.parent == roots.locks, f"the lock {lock} lies directly under $LASSI_SCRATCH/locks"
    taken = [operation for _fd, operation in flock.calls if operation & LOCK_EX]
    assert taken and all(operation & LOCK_NB for operation in taken), "LOCK_EX | LOCK_NB: never a wait on the lock"
    assert flock.held_files(roots.locks) == [], "released after the run"
    assert list(roots.tmpdir.iterdir()) == [], "nothing under TMPDIR"


def _interrupted(host: Host) -> RaisingSandbox:
    return RaisingSandbox(KeyboardInterrupt(), effect=umd_files(host))


def _unavailable(host: Host) -> RaisingSandbox:
    return RaisingSandbox(SandboxUnavailableError("SYNTHETIC: the setup failed; nothing ran"))


# The copy-back failure's SYNTHETIC message, in the shape of the sandbox's own (_copy_back_failure).
COPY_BACK_MESSAGE = (
    "SYNTHETIC: the sandbox ran the program but did not finish the copy-back of its workdir (exit status 1); the "
    "result is not reported: (no stderr)"
)


def _copy_back_failed(host: Host) -> RaisingSandbox:
    return RaisingSandbox(copy_back_error()(COPY_BACK_MESSAGE), effect=umd_files(host))


# Each case: the sandbox stand-in, and the exception the run raises (None: it returns a RunResult).
ENDINGS: dict[str, tuple[Callable[[Host], Any], type[BaseException] | None]] = {
    "clean": (lambda h: FakeSandbox(results=[canned()], effects=[umd_files(h)]), None),
    "nonzero": (lambda h: FakeSandbox(results=[canned(returncode=3)], effects=[umd_files(h)]), None),
    "hang": (lambda h: FakeSandbox(results=[canned(returncode=124, hang=True)], effects=[umd_files(h)]), None),
    "crash": (lambda h: FakeSandbox(results=[canned(returncode=139)], effects=[umd_files(h)]), None),
    "sandbox-unavailable": (_unavailable, SandboxUnavailableError),
    "copy-back-failure": (_copy_back_failed, SandboxUnavailableError),
    "interrupted": (_interrupted, KeyboardInterrupt),
}


@pytest.mark.parametrize("case", sorted(ENDINGS))
def test_every_ending_releases_the_lock_and_removes_the_runs_hugepage_directory(
    module: ModuleType, host: Host, install: Any, roots: Roots, case: str
) -> None:
    make_sandbox, raised = ENDINGS[case]
    flock = FakeFlock()
    before = tree(host.at(HUGEPAGE_POINT))
    sandbox = make_sandbox(host)
    executor = build(module, host, sandbox=sandbox, toolchains=install.root, flock=flock)
    if raised is None:
        assert isinstance(executor.run(make_artifact(roots.runs), INPUTS, LIMITS), RunResult)
    else:
        with pytest.raises(raised):
            executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert flock.held_files(roots.locks) == [], f"{case}: the lock is still held"
    assert tree(host.at(HUGEPAGE_POINT)) == before, f"{case}: the hugepage directory is not as it was"


def test_a_lock_another_process_holds_refuses_the_run_naming_the_file_and_its_pid(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    flock = FakeFlock()
    files: list[Path] = []
    effects = [lambda spec, argv: files.extend(flock.held_files(roots.locks)), None]
    executor, fake = silicon(module, host, install, flock=flock, results=[canned(), canned()], effects=effects)
    artifact = make_artifact(roots.runs)
    executor.run(artifact, INPUTS, LIMITS)
    (lock,) = files
    holder = os.open(lock, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        flock(holder, LOCK_EX | LOCK_NB)
        os.ftruncate(holder, 0)
        os.write(holder, b"424242\n")
        with pytest.raises(SandboxUnavailableError) as caught:
            executor.run(artifact, INPUTS, LIMITS)
        message = str(caught.value)
        assert names_path(message, lock) and "424242" in message, message
        assert len(fake.calls) == 1, "no program runs while another process holds the lock"
    finally:
        os.close(holder)
    executor.run(artifact, INPUTS, LIMITS)
    assert len(fake.calls) == 2, "once the holder is gone, the next run goes ahead"


def _lock_fds(monkeypatch: pytest.MonkeyPatch, lock: Path) -> list[int]:
    """Record every descriptor os.open returns for the lock file from now on; return the list it fills."""
    opened: list[int] = []
    real_open = os.open

    def recording_open(path: Any, *args: Any, **kwargs: Any) -> int:
        fd = real_open(path, *args, **kwargs)
        if not isinstance(path, int) and os.path.basename(os.fspath(path)) == lock.name:
            opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", recording_open)
    return opened


def _held_lock(
    module: ModuleType, host: Host, install: Any, roots: Roots, flock: FakeFlock
) -> tuple[Path, Path, int]:
    """Run once so the lock file exists, then hold its lock on another descriptor; return the artifact, file, and fd."""
    artifact = make_artifact(roots.runs)
    executor, _ = silicon(module, host, install, flock=flock)
    executor.run(artifact, INPUTS, LIMITS)
    lock = roots.locks / module.LOCK_FILE
    holder = os.open(lock, os.O_RDWR | getattr(os, "O_BINARY", 0))
    flock(holder, LOCK_EX | LOCK_NB)
    return artifact, lock, holder


def test_a_holder_whose_pid_cannot_be_read_still_refuses_and_leaks_no_descriptor(
    module: ModuleType, host: Host, install: Any, roots: Roots, monkeypatch: pytest.MonkeyPatch
) -> None:
    flock = FakeFlock()
    artifact, lock, holder = _held_lock(module, host, install, roots, flock)
    try:
        executor, fake = silicon(module, host, install, flock=flock)
        opened = _lock_fds(monkeypatch, lock)
        real_lseek = os.lseek

        def failing_lseek(fd: int, *args: Any) -> int:
            if fd != holder and identity(fd) == path_identity(lock):
                raise OSError(5, "SYNTHETIC: the lock file cannot be read")
            return real_lseek(fd, *args)

        monkeypatch.setattr(os, "lseek", failing_lseek)
        with pytest.raises(SandboxUnavailableError) as caught:
            executor.run(artifact, INPUTS, LIMITS)
        assert names_path(str(caught.value), lock), str(caught.value)
        assert fake.calls == [], "no program runs while another process holds the lock"
        assert opened, "the run opened the lock file"
        assert all(identity(fd) != path_identity(lock) for fd in opened), "the run's descriptor of the lock is closed"
    finally:
        os.close(holder)


@pytest.mark.parametrize("case", ["clean", "interrupted"])
def test_a_failure_to_clear_the_pid_hides_neither_the_result_nor_the_exception(
    module: ModuleType, host: Host, install: Any, roots: Roots, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    flock = FakeFlock()
    sandbox = FakeSandbox(results=[canned()]) if case == "clean" else _interrupted(host)
    executor = build(module, host, sandbox=sandbox, toolchains=install.root, flock=flock)
    lock = roots.locks / module.LOCK_FILE
    real_ftruncate, calls = os.ftruncate, []

    def failing_ftruncate(fd: int, length: int) -> None:
        if lock.exists() and identity(fd) == path_identity(lock):
            calls.append(fd)
            if len(calls) > 1:
                raise OSError(28, "SYNTHETIC: no space left to clear the pid")
        real_ftruncate(fd, length)

    monkeypatch.setattr(os, "ftruncate", failing_ftruncate)
    if case == "clean":
        assert isinstance(executor.run(make_artifact(roots.runs), INPUTS, LIMITS), RunResult)
    else:
        with pytest.raises(KeyboardInterrupt):
            executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert len(calls) == 2, "the pid is written once the lock is taken, and cleared on the way out"
    assert flock.held_files(roots.locks) == [], f"{case}: the lock is still held"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the real fcntl.flock exists only on Linux")
def test_the_default_flock_is_the_kernels_and_a_real_holder_refuses_the_run(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    executor, fake = silicon(module, host, install, flock=None, results=[canned(), canned()])
    artifact = make_artifact(roots.runs)
    executor.run(artifact, INPUTS, LIMITS)
    (lock,) = [path for path in roots.locks.iterdir() if "quarantine" not in path.name]
    child = subprocess.Popen([sys.executable, "-I", "-c", HOLDER, str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "held"
        with pytest.raises(SandboxUnavailableError) as caught:
            executor.run(artifact, INPUTS, LIMITS)
        assert str(child.pid) in str(caught.value), str(caught.value)
        assert len(fake.calls) == 1
    finally:
        child.kill()
        child.wait()


# ---------------------------------------------------------------------------
# The poll: never claim a card that is busy


@dataclass(frozen=True)
class Busy:
    """A way the cards are busy: how to make it, how the previous user releases it, and what a refusal names."""

    make: Callable[[Host, Owners], None]
    release: Callable[[Host, Owners], None]
    named: Callable[[Host], str]


def _foreign_file(host_path: str) -> Busy:
    relative = host_path.lstrip("/")

    def make(host: Host, owners: Owners) -> None:
        write_text(host.at(host_path), "")
        owners.uids[relative] = OTHER_UID

    return Busy(make=make, release=lambda host, owners: host.at(host_path).unlink(), named=lambda host: host_path)


def _few_pages() -> Busy:
    def make(host: Host, owners: Owners) -> None:
        write_text(host.pool("free_hugepages"), f"{host.needed_pages() - 1}\n")

    def release(host: Host, owners: Owners) -> None:
        write_text(host.pool("free_hugepages"), f"{host.needed_pages()}\n")

    return Busy(make=make, release=release, named=lambda host: str(host.needed_pages()))


BUSY: dict[str, Busy] = {
    "another-accounts-hugepage-file": _foreign_file(f"{HUGEPAGE_POINT}/tenstorrent"),
    "another-accounts-umd-lock": _foreign_file(f"{SHM}/TT_UMD_LOCK.CHIP_IN_USE_0_PCIe"),
    "fewer-free-pages-than-two-a-card": _few_pages(),
}


def _own_file(host_path: str) -> Callable[[Host, Owners], None]:
    return lambda host, owners: write_text(host.at(host_path), "")


def _other_owned(host_path: str) -> Callable[[Host, Owners], None]:
    def make(host: Host, owners: Owners) -> None:
        write_text(host.at(host_path), "")
        owners.uids[host_path.lstrip("/")] = OTHER_UID

    return make


def _exact_pages(host: Host, owners: Owners) -> None:
    write_text(host.pool("free_hugepages"), f"{host.needed_pages()}\n")


FREE: dict[str, Callable[[Host, Owners], None]] = {
    "own-leftover-hugepage-file": _own_file(f"{HUGEPAGE_POINT}/tenstorrent"),
    "roots-libvirt-directory-only": lambda host, owners: None,
    "own-umd-lock": _own_file(f"{SHM}/TT_UMD_LOCK.ARC_MSG"),
    "another-accounts-shm-file-that-is-no-umd-lock": _other_owned(f"{SHM}/SYNTHETIC-other-shm-object"),
    "exactly-two-free-pages-a-card": _exact_pages,
}


@pytest.mark.parametrize("case", sorted(FREE))
def test_free_cards_run_at_once(module: ModuleType, host: Host, install: Any, roots: Roots, case: str) -> None:
    owners, clock = Owners(host.root), Clock()
    FREE[case](host, owners)
    executor, fake = silicon(module, host, install, owners=owners, clock=clock)
    executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert len(fake.calls) == 1 and clock.sleeps == [], f"{case}: the cards count as free"


@pytest.mark.parametrize("case", sorted(BUSY))
def test_busy_cards_are_polled_until_the_previous_user_releases_them(
    module: ModuleType, host: Host, install: Any, roots: Roots, case: str
) -> None:
    busy, owners, clock = BUSY[case], Owners(host.root), Clock()
    busy.make(host, owners)

    def released_on_third(count: int) -> None:
        if count == 3:
            busy.release(host, owners)

    clock.on_sleep = released_on_third
    starts: list[int] = []
    effects = [lambda spec, argv: starts.append(len(clock.sleeps))]
    executor, fake = silicon(module, host, install, owners=owners, clock=clock, effects=effects)
    executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    interval = module.POLL_INTERVAL_S
    assert starts == [3], f"{case}: the program started only once the cards were free, after three polls"
    assert all(0 < seconds <= interval for seconds in clock.sleeps), clock.sleeps
    assert len(fake.calls) == 1


@pytest.mark.parametrize("case", sorted(BUSY))
def test_cards_that_stay_busy_are_refused_after_the_poll_limit(
    module: ModuleType, host: Host, install: Any, roots: Roots, case: str
) -> None:
    busy, owners, clock, flock = BUSY[case], Owners(host.root), Clock(), FakeFlock()
    busy.make(host, owners)
    before = tree(host.at(HUGEPAGE_POINT))
    executor = build(module, host, sandbox=ExplodingSandbox(), toolchains=install.root, owners=owners, clock=clock,
                     flock=flock)  # fmt: skip
    with pytest.raises(SandboxUnavailableError) as caught:
        executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    message = str(caught.value)
    assert busy.named(host) in message, message
    not_root_joined(message, host)
    interval, limit = module.POLL_INTERVAL_S, module.POLL_LIMIT_S
    assert limit - interval <= sum(clock.sleeps) <= limit + interval, clock.sleeps
    assert flock.held_files(roots.locks) == [] and markers(roots) == [], "a refusal claims nothing and marks nothing"
    assert tree(host.at(HUGEPAGE_POINT)) == before


def test_the_cards_are_checked_before_every_run(module: ModuleType, host: Host, install: Any, roots: Roots) -> None:
    owners, clock = Owners(host.root), Clock()
    executor, fake = silicon(module, host, install, owners=owners, clock=clock, results=[canned()])
    artifact = make_artifact(roots.runs)
    executor.run(artifact, INPUTS, LIMITS)
    BUSY["another-accounts-umd-lock"].make(host, owners)
    with pytest.raises(SandboxUnavailableError):
        executor.run(artifact, INPUTS, LIMITS)
    assert len(fake.calls) == 1, "the second run found a busy card and never started"


# ---------------------------------------------------------------------------
# The run's own hugepage directory


def test_the_runs_hugepage_directory_is_new_private_and_gone_after(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    write_text(host.at(f"{HUGEPAGE_POINT}/tenstorrent"), "SYNTHETIC own leftover\n")
    write_text(host.at(f"{HUGEPAGE_POINT}/libvirt/SYNTHETIC-guest"), "SYNTHETIC\n")
    before = tree(host.at(HUGEPAGE_POINT))
    modes: list[int] = []

    def check(spec: Any, argv: list[str]) -> None:
        umd_files(host)(spec, argv)
        directory = run_dir(host, spec)
        assert directory.parent == host.at(HUGEPAGE_POINT)
        assert directory.name not in before, "a directory made for this run, not one that was there"
        modes.append(stat.S_IMODE(os.stat(directory).st_mode))

    executor, _ = silicon(module, host, install, effects=[check])
    executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert len(modes) == 1
    if os.name == "posix":
        assert modes == [0o700], "mode 0700: no other account sees the run's files"
    assert tree(host.at(HUGEPAGE_POINT)) == before, "every entry is gone and nothing else was touched"


def _can_link(directory: Path) -> bool:
    """Return True when this host lets the tests make a symbolic link."""
    probe = directory / "link-probe"
    try:
        os.symlink(directory, probe, target_is_directory=True)
    except OSError:
        return False
    probe.unlink()
    return True


def test_a_link_in_the_runs_directory_is_removed_without_being_followed(
    module: ModuleType, host: Host, install: Any, roots: Roots, tmp_path: Path
) -> None:
    if not _can_link(tmp_path):
        pytest.skip("this host cannot make symbolic links without a privilege (Windows)")
    outside = tmp_path / "outside"
    write_text(outside / "keep", "SYNTHETIC: never removed\n")

    def link(spec: Any, argv: list[str]) -> None:
        os.symlink(outside, run_dir(host, spec) / "link", target_is_directory=True)

    executor, _ = silicon(module, host, install, effects=[link])
    executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert (outside / "keep").read_bytes() == b"SYNTHETIC: never removed\n"
    assert sorted(path.name for path in host.at(HUGEPAGE_POINT).iterdir()) == ["libvirt"]


def read_only_entries(host: Host) -> Callable[[Any, list[str]], None]:
    """Return an effect that leaves read-only entries in the run's hugepage directory, as a program inside could.

    A read-only file, and a read-only directory holding a read-only file:
    on Linux the directory's mode stops the unlink of its file, and on
    Windows the read-only attribute stops each removal.
    """

    def effect(spec: Any, argv: list[str]) -> None:
        directory = run_dir(host, spec)
        locked = directory / "locked"
        write_text(locked / "inside", "SYNTHETIC\n")
        write_text(directory / "read-only", "SYNTHETIC\n")
        for path in (locked / "inside", directory / "read-only"):
            os.chmod(path, stat.S_IRUSR)
        os.chmod(locked, stat.S_IRUSR | stat.S_IXUSR)

    return effect


def test_read_only_entries_in_the_runs_directory_are_removed_too(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    before = tree(host.at(HUGEPAGE_POINT))
    executor, _ = silicon(module, host, install, effects=[read_only_entries(host)])
    ran = executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert ran.exit_code == 0
    assert tree(host.at(HUGEPAGE_POINT)) == before, "write permission is restored, and every entry is gone"
    assert markers(roots) == []


@pytest.mark.parametrize("returncode", [0, 139], ids=["clean", "crash"])
def test_a_directory_that_cannot_be_removed_quarantines_the_cards_and_the_result_stands(
    module: ModuleType, host: Host, install: Any, roots: Roots, monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    attempts: list[str] = []

    def refused(path: Any, *args: Any, **kwargs: Any) -> None:
        attempts.append(os.fspath(path))
        raise PermissionError(13, "SYNTHETIC: the directory cannot be removed", os.fspath(path))

    flock = FakeFlock()
    executor, fake = silicon(module, host, install, results=[canned(returncode=returncode)], flock=flock)
    artifact = make_artifact(roots.runs)
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "rmtree", refused)
        ran = executor.run(artifact, INPUTS, LIMITS)
    assert ran.exit_code == returncode, "the run's result is still returned"
    assert attempts, "the removal was tried"
    (marker,) = markers(roots)
    text = marker.read_bytes().decode("ascii")
    hugepages = posix(fake.calls[0].spec.devices.hugepages)
    assert hugepages in text, f"the marker names the directory {hugepages}: {text}"
    if returncode:
        assert str(returncode) in text, "the run's own ending is named too"
    assert all(serial(card) in text for card in host.cards)
    assert flock.held_files(roots.locks) == []


# ---------------------------------------------------------------------------
# The quarantine: after a hang, a kill, or a crash, until a human removes the marker

# (returncode, hang, killed) -> whether the run writes the quarantine marker.
STATUSES: dict[str, tuple[int, bool, bool, bool]] = {
    "clean": (0, False, False, False),
    "nonzero-exit-1": (1, False, False, False),
    "nonzero-exit-3": (3, False, False, False),
    "own-exit-124-before-the-limit": (124, False, False, False),
    "env-127": (127, False, False, False),
    "hang-124": (124, True, False, True),
    "hang-137": (137, True, False, True),
    "runner-timeout": (-1, True, False, True),
    "memory-kill-137": (137, False, True, True),
    "sighup-129": (129, False, False, True),
    "sigabrt-134": (134, False, False, True),
    "sigbus-135": (135, False, False, True),
    "sigsegv-139": (139, False, False, True),
    "sigterm-143": (143, False, False, True),
    "status-255": (255, False, False, True),
}


@pytest.mark.parametrize("case", sorted(STATUSES))
def test_a_hang_a_kill_or_a_death_by_signal_quarantines_every_card(
    module: ModuleType, host: Host, install: Any, roots: Roots, case: str
) -> None:
    returncode, hang, killed, quarantined = STATUSES[case]
    result = canned(returncode=returncode, hang=hang, killed=killed)
    during: list[int] = []
    effects = [lambda spec, argv: during.append(len(markers(roots)))]
    executor, fake = silicon(module, host, install, results=[result], effects=effects)
    ran = executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert (ran.exit_code, ran.hang) == (returncode, hang), "the run's own result is still returned"
    assert len(fake.calls) == 1, "never retried, never rerun with Watcher"
    assert during == [1], f"{case}: the marker is in place while the program runs"
    assert len(markers(roots)) == (1 if quarantined else 0), case
    if quarantined:
        assert f"exit status {returncode}" in markers(roots)[0].read_bytes().decode("ascii"), "it names the ending"


def test_the_marker_is_in_place_before_the_program_starts_so_a_killed_parent_leaves_it(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    # A SIGKILL of this process while the program runs ends nothing in Python, so what the sandbox's call sees is
    # what a kill would leave behind: the marker, naming every card.
    seen: list[str] = []

    def check(spec: Any, argv: list[str]) -> None:
        (marker,) = markers(roots)
        seen.append(marker.read_bytes().decode("ascii"))

    executor, _ = silicon(module, host, install, effects=[check])
    executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    (text,) = seen
    assert all(serial(card) in text for card in host.cards), text
    assert "ending" in text and text.isascii(), text
    assert markers(roots) == [], "a clean ending removes it"


def test_the_marker_names_the_status_and_every_card_and_refuses_every_later_run(
    module: ModuleType, host: Host, install: Any, roots: Roots, tmp_path: Path
) -> None:
    executor, fake = silicon(module, host, install, results=[canned(returncode=124, hang=True)])
    artifact = make_artifact(roots.runs)
    executor.run(artifact, INPUTS, LIMITS)
    assert "TT_METAL_WATCHER" not in fake.calls[0].environment
    (marker,) = markers(roots)
    assert marker.parent == roots.locks
    text = marker.read_bytes().decode("ascii")
    assert "124" in text and "hang" in text.lower(), text
    assert all(serial(card) in text for card in host.cards), "the marker names every card the run held"
    with pytest.raises(SandboxUnavailableError) as caught:
        executor.run(artifact, INPUTS, LIMITS)
    assert names_path(str(caught.value), marker), str(caught.value)
    assert len(fake.calls) == 1, "a quarantined run never reaches the sandbox"
    with pytest.raises(DeviceUnavailable) as refused:
        build(module, host, sandbox=ExplodingSandbox())
    assert names_path(str(refused.value), marker), str(refused.value)
    marker.unlink()  # what a human does after checking the cards
    again, again_fake = silicon(module, host, install)
    again.run(artifact, INPUTS, LIMITS)
    assert len(again_fake.calls) == 1


def test_a_sandbox_that_never_started_the_program_removes_the_marker(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    # Every SandboxUnavailableError but the copy-back failure comes before the program starts (no ready line, or
    # the command did not start), so the marker written before the call is removed again.
    during: list[int] = []
    error = SandboxUnavailableError("SYNTHETIC: the sandbox setup failed; the program did not run")
    sandbox = RaisingSandbox(error, effect=lambda spec, argv: during.append(len(markers(roots))))
    executor = build(module, host, sandbox=sandbox, toolchains=install.root)
    with pytest.raises(SandboxUnavailableError) as caught:
        executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert caught.value is error
    assert during == [1], "the marker was written before the sandbox was called"
    assert markers(roots) == []


def test_a_copy_back_failure_keeps_the_marker_naming_it(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    # The sandbox raises SandboxCopyBackError after the program ran on the cards (OQ-075 (3a)): its ending is
    # unknown, so the cards stay quarantined, and the error still reaches the caller as a SandboxUnavailableError.
    flock = FakeFlock()
    before = tree(host.at(HUGEPAGE_POINT))
    executor = build(module, host, sandbox=_copy_back_failed(host), toolchains=install.root, flock=flock)
    with pytest.raises(SandboxUnavailableError) as caught:
        executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert isinstance(caught.value, copy_back_error())
    (marker,) = markers(roots)
    text = marker.read_bytes().decode("ascii")
    assert "copy-back" in text and all(serial(card) in text for card in host.cards), text
    assert flock.held_files(roots.locks) == [] and tree(host.at(HUGEPAGE_POINT)) == before


def test_a_spec_that_cannot_be_built_writes_no_marker(
    module: ModuleType, host: Host, install: Any, roots: Roots, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(**fields: Any) -> Any:
        raise ValueError("SYNTHETIC: the exposure is refused")

    flock = FakeFlock()
    before = tree(host.at(HUGEPAGE_POINT))
    executor = build(module, host, sandbox=ExplodingSandbox(), toolchains=install.root, flock=flock)
    monkeypatch.setattr(module, "TenstorrentExposure", refused)
    with pytest.raises(ValueError, match="SYNTHETIC"):
        executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert markers(roots) == [], "nothing could start, so nothing is quarantined"
    assert flock.held_files(roots.locks) == [] and tree(host.at(HUGEPAGE_POINT)) == before


def test_an_interrupted_run_quarantines_every_card_then_propagates(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    executor = build(module, host, sandbox=_interrupted(host), toolchains=install.root)
    with pytest.raises(KeyboardInterrupt):
        executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert len(markers(roots)) == 1


# ---------------------------------------------------------------------------
# The RunResult: no simulator readings; the kernel JIT's diagnostics


def test_the_result_carries_the_sandboxs_fields_and_no_simulator_reading(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    def writes(spec: Any, argv: list[str]) -> None:
        (Path(spec.workdir) / "out.bin").write_bytes(b"SYNTHETIC output\n")
        (Path(dict(spec.environment)["TT_METAL_CACHE"]) / "kernel.o").write_bytes(b"SYNTHETIC JIT output\n")

    result = canned(returncode=3, stdout=FINDING_LINE, stderr="e", stdout_truncated=True, workdir_incomplete=True)
    executor, _ = silicon(module, host, install, results=[result], effects=[writes])
    ran = executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert isinstance(ran, RunResult)
    assert (ran.exit_code, ran.hang, ran.stdout, ran.stderr) == (3, False, FINDING_LINE, "e")
    assert (ran.stdout_truncated, ran.stderr_truncated, ran.workdir_incomplete) == (True, False, True)
    assert list(ran.output_files) == ["out.bin"], "nothing under the reserved directory is an output"
    assert ran.sim_ub is None and ran.sim_gap is None, "silicon has no simulator readings"
    assert [entry for entry in ran.diagnostics if entry.code == "NonContractualBehavior"] == []


def test_kernel_jit_errors_are_read_as_the_ttsim_executor_reads_them(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    artifact = make_artifact(roots.runs)
    result = capture("jit-error")
    executor, _ = silicon(module, host, install, results=[result])
    ran = executor.run(artifact, INPUTS, LIMITS)
    workdir = Path(os.path.realpath(artifact.parent))
    expected = ttsim_module.read_jit_diagnostics(result.stdout, result.stderr, workdir)
    assert expected and ran.diagnostics == expected
