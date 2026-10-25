"""Remote tests of the tt_silicon executor on the cluster's Wormhole cards (task P17.16; task P17.17 runs them).

ONLY THE OWNER'S GO STARTS THESE TESTS. They put a program on Tenstorrent
silicon, which only a human starts as an evaluation run (Agent Rule 9), on
alpha01, which needs the owner's approval in the working session and an
approved window (Agent Rule 17; plans/LESSONS.md, Owner directions). No
unattended session, CI job, or test sweep ever sets LASSI_SILICON_TESTS.

Bible: Execution Backends (the tt_silicon row), Sandbox, Oracles
(binary_io), Agent Rules 2, 6, 7, 9, and 17; plans/p17-portable.md, tasks
P17.16 and P17.17; OQ-072 and OQ-075; plans/spikes/tt-silicon-executor.md,
section 6, test 14, with every card in place of one. The contract is in
tests/executors/test_tt_silicon.py; this module runs it once for real.

What runs, in test_the_loopback_reference_runs_once_on_the_cards:

1. tt-pairs-v0's pinned files are fetched (tools/fetch_bench.py, as P4.G
   does); the loopback TT reference builds with ttmetal-host and its C++
   counterpart with gcc-native, as tests/bench/test_tt_pairs_remote.py
   builds them, and the item's held-out inputs are staged.
2. The C++ side runs through the native executor.
3. The TT side runs once through TtSiliconExecutor(hugepages=
   "private-per-run", reset_filter=True): it polls until no other account
   holds the cards, takes the lock, and runs the program in the sandbox with
   every card exposed. It is the item's tracked upstream reference, never
   model output (Agent Rule 6 binds model-generated code; the sandbox runs
   this one anyway).
4. The checks: exit status 0 and no hang; every output passes binary_io
   under the item's declared tolerance; device() names "(silicon)" and no
   simulator; afterwards no quarantine marker exists, /dev/hugepages-1G
   lists what it listed before, free_hugepages reads what it read before,
   and the lock is free (a real flock LOCK_EX | LOCK_NB on it succeeds).

One program, one run, one placement: no loop of device opens, no reset,
no flash (Agent Rule 9). A hang or a crash quarantines the cards, and a human
checks them before removing the marker (OQ-075 (3a)). Wall time is never
asserted on and is never performance (Agent Rule 2).

Gates: the tests are marked `remote` and skip unless LASSI_SILICON_TESTS=1
is set; LASSI_REQUIRE_SANDBOX=1 alone never starts them. With
LASSI_SILICON_TESTS=1 a host that cannot run them fails them instead (Linux,
the sandbox tools on PATH, a reachable user systemd manager,
$LASSI_SCRATCH, $LASSI_RUNS_ROOT, and $LASSI_TOOLCHAINS set, TMPDIR inside
$LASSI_SCRATCH, both pinned host compilers, the tt-metal tree installed,
the card directory and the 1 GiB hugepage directory present), so a silent
skip never passes for evidence. Run them, after the owner's go and from a
clean commit, with

    PYTHONIOENCODING=utf-8 uv run tools/rx.py run -- 'LASSI_REQUIRE_SANDBOX=1 LASSI_SILICON_TESTS=1
    uv run pytest -q -m remote tests/executors/test_tt_silicon_remote.py'

on one line; the command names no device path, which the gate refuses
(plans/LESSONS.md). Every build lies in a temp directory under
$LASSI_RUNS_ROOT, removed afterwards; the lock and any marker lie under
$LASSI_SCRATCH/locks (Agent Rule 7). No value here is a measurement until
its run is logged with provenance (AGENTS.md, Results).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType

import pytest

from lassi import bench
from lassi.bench import registry
from lassi.core.interfaces import Limits, RunResult
from lassi.core.runner import BuiltToolchain, build_toolchain
from lassi.harness.lassi_io import read_array
from lassi.oracles.binary_io import BinaryIOOracle
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
SUITE = "tt-pairs-v0"
MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "loopback"
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk", "env", "sh", "git")
SILICON_TESTS = os.environ.get("LASSI_SILICON_TESTS") == "1"
TO_TT, TO_CPP = bench.Direction("cpp", "tt"), bench.Direction("tt", "cpp")
# The smoke's limits (plans/runs/tt-silicon/smoke.py), under which the loopback reference passed on card 0.
LIMITS = Limits(wall_s=120.0, memory_mb=8192, cpus=16)
CARD_DIR = Path("/dev/tenstorrent")
HUGEPAGE_POINT = Path("/dev/hugepages-1G")
FREE_PAGES = Path("/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages")


def installed(pin_name: str) -> Path:
    """Return the install prefix of a pin under the resolved toolchains root."""
    return Path(os.environ["LASSI_TOOLCHAINS"]).resolve() / pins_module.read_pin(pin_name)["PREFIX_NAME"]


def host_problem() -> str:
    """Return why this host cannot run the tests, or "" when it can; it only lists and stats, never opens a node."""
    if not sys.platform.startswith("linux"):
        return f"silicon needs the Linux build host, not {sys.platform}"
    missing = [tool for tool in TOOLS if shutil.which(tool) is None]
    if missing:
        return f"not on PATH: {', '.join(missing)}"
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if not runtime or not (Path(runtime) / "bus").exists():
        return "no user systemd manager: $XDG_RUNTIME_DIR/bus does not exist"
    for name in ("LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS", "TMPDIR"):
        if not os.environ.get(name):
            return f"{name} is not set"
    scratch, tmpdir = Path(os.environ["LASSI_SCRATCH"]).resolve(), Path(os.environ["TMPDIR"]).resolve()
    if scratch not in tmpdir.parents:
        return "TMPDIR does not lie inside $LASSI_SCRATCH"
    for pin_name in ("tt-metal", "gcc"):
        executable = pins_module.read_pin(pin_name).get("EXECUTABLE", "")
        if not executable or not Path(executable).is_file():
            return f"the pinned host compiler {executable!r} (toolchains/{pin_name}.pin) is not on this host"
    if not (installed("tt-metal") / "lassi-install.txt").is_file():
        return f"{installed('tt-metal')} is not an install of the pinned tt-metal"
    if not CARD_DIR.is_dir() or not HUGEPAGE_POINT.is_dir():
        return "no Tenstorrent card directory or no 1 GiB hugepage directory on this host"
    return ""


PROBLEM = host_problem() if SILICON_TESTS else ""
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(
        not SILICON_TESTS,
        reason="only the owner's go starts a silicon run (Agent Rules 9 and 17); LASSI_SILICON_TESTS=1 is not set",
    ),
]


@pytest.fixture(autouse=True)
def host_ready() -> None:
    """Fail, rather than skip, when LASSI_SILICON_TESTS=1 asks for silicon and the host cannot run it."""
    if PROBLEM:
        pytest.fail(f"LASSI_SILICON_TESTS=1, but {PROBLEM}")


@pytest.fixture(scope="module")
def toolchains() -> dict[str, BuiltToolchain]:
    """Build ttmetal-host and gcc-native as a run builds them: trees checked, pinned compilers checked by --version."""
    if PROBLEM:
        pytest.fail(f"LASSI_SILICON_TESTS=1, but {PROBLEM}")
    root = Path(os.environ["LASSI_TOOLCHAINS"])
    return {"tt": build_toolchain("ttmetal-host", root), "cpp": build_toolchain("gcc-native", root)}


@pytest.fixture
def scratch_dirs() -> Iterator[Path]:
    """Create a temp directory under $LASSI_RUNS_ROOT for the two builds; remove it afterwards."""
    base = Path(tempfile.mkdtemp(prefix="lassi-tt-silicon.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        yield base
    finally:
        shutil.rmtree(base)


def fetch_tool() -> ModuleType:
    """Import tools/fetch_bench.py by path and return the module."""
    spec = importlib.util.spec_from_file_location("fetch_bench", REPO / "tools" / "fetch_bench.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_in(
    built: BuiltToolchain, item: registry.SuiteItem, files: Mapping[str, str], harness: Mapping[str, str], workdir: Path
) -> tuple[Path, list[str]]:
    """Build `files` with `harness` in `workdir` and stage the item's inputs there; return the artifact and args."""
    workdir.mkdir(parents=True)
    result = built.toolchain.build(files, workdir, harness=harness)
    stderr = (workdir / "compile.stderr").read_bytes().decode("utf-8", "replace")
    errors = [entry for entry in result.diagnostics if entry.severity == "error"]
    assert result.artifact is not None and not errors, f"{item.name} did not build:\n{stderr[-4000:]}"
    return Path(result.artifact), registry.stage_inputs(item, workdir)


def clean(run: RunResult, side: str) -> None:
    """Assert a clean run: exit status 0, no hang, and no kernel JIT error."""
    tail = f"{run.stdout[-3000:]}\n{run.stderr[-3000:]}"
    assert (run.exit_code, run.hang) == (0, False), f"{side}: exit {run.exit_code}, hang {run.hang}\n{tail}"
    jit = [entry for entry in run.diagnostics if entry.stage == "jit" and entry.severity == "error"]
    assert jit == [], f"{side}: {run.diagnostics}\n{tail}"


def lock_is_free(locks: Path) -> bool:
    """Return True when every lock file under `locks` can be taken now with LOCK_EX | LOCK_NB (then released)."""
    import fcntl

    for path in sorted(locks.iterdir()) if locks.is_dir() else []:
        if "quarantine" in path.name or not path.is_file():
            continue
        fd = os.open(path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            return False
        finally:
            os.close(fd)
    return True


def test_the_loopback_reference_runs_once_on_the_cards(
    toolchains: dict[str, BuiltToolchain], scratch_dirs: Path
) -> None:
    from lassi.executors.native import NativeExecutor

    silicon = importlib.import_module("lassi.executors.tt_silicon")
    assert fetch_tool().main([str(MANIFEST)]) == 0, "the pinned files were fetched and every sha256 matched"
    suite = bench.load_suite(MANIFEST)
    root = registry.sources_dir(Path(os.environ["LASSI_SCRATCH"]), suite)
    spec = suite.item(ITEM, purpose="eval")
    tt_files = suite.reference_target(ITEM, TO_TT, root, purpose="eval")
    tt_harness = suite.support_files(ITEM, root, purpose="eval", language="tt")
    cpp_files = suite.reference_target(ITEM, TO_CPP, root, purpose="eval")
    cpp_harness = suite.support_files(ITEM, root, purpose="eval", language="cpp")
    tt_artifact, tt_args = build_in(toolchains["tt"], spec, tt_files, tt_harness, scratch_dirs / "tt" / "build")
    cpp_artifact, cpp_args = build_in(toolchains["cpp"], spec, cpp_files, cpp_harness, scratch_dirs / "cpp" / "build")
    cpp_run = NativeExecutor().run(cpp_artifact, cpp_args, LIMITS)
    clean(cpp_run, f"{ITEM} C++ counterpart, native")

    executor = silicon.TtSiliconExecutor(hugepages="private-per-run", reset_filter=True)
    device = executor.device()
    assert "(silicon)" in device and "simulator" not in device.lower(), device
    locks = Path(os.environ["LASSI_SCRATCH"]) / "locks"
    pages_before = FREE_PAGES.read_text().strip()
    listing_before = sorted(path.name for path in HUGEPAGE_POINT.iterdir())

    tt_run = executor.run(tt_artifact, tt_args, LIMITS)  # the one placement on silicon

    clean(tt_run, f"{ITEM} TT reference on {device}")
    assert tt_run.sim_ub is None and tt_run.sim_gap is None, "silicon has no simulator readings"
    assert spec.tolerance is not None
    oracle = BinaryIOOracle(metric=spec.tolerance.metric, threshold=spec.tolerance.threshold)
    stats = oracle.compare(cpp_run.output_files, tt_run.output_files, sides=("cpp reference", "tt on silicon"))
    assert stats and all(entry.passed for entry in stats), f"{ITEM}: {stats}"
    declared = sorted(spec.outputs)
    assert sorted(read_array(path).name for path in tt_run.output_files.values()) == declared
    assert [path for path in locks.iterdir() if "quarantine" in path.name] == [], "a clean run quarantines nothing"
    assert sorted(path.name for path in HUGEPAGE_POINT.iterdir()) == listing_before, "the run's directory is gone"
    assert FREE_PAGES.read_text().strip() == pages_before, "every page the run mapped is back in the pool"
    assert lock_is_free(locks), "the lock was released"
