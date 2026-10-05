"""Remote tests of the Tier A suite tt-pairs-v0 on the build host (task P4.13).

Bible: Benchmark Suites (Tier A), Harness Contract (lassi_io, held-out
inputs, the CPU -> TT guard), Oracles (binary_io), Execution Backends (the
native and ttsim rows), ttsim Facts, Agent Rules 2, 6, 7, and 9. Plan:
plans/p4-ttsim.md, P4.13, the remote acceptance: each TT reference builds,
runs clean on ttsim (no UB, no gap), and passes the guard; each C++
counterpart builds and runs clean natively; their binary_io agreement meets
the item's tolerance.

Each test takes one item (parametrized, so `-k <item>` runs one) and:

1. fetches the suite's pinned files with tools/fetch_bench.py on the
   manifest, as P4.G does (the copy from the installed tt-metal tree,
   idempotent), and expects status 0: the commit and every kernel's sha256
   verified;
2. builds the TT reference (the tracked host program, with the item's TT
   support files: the fetched kernels at their relative paths and
   lassi_io.h) with ttmetal-host as a run builds it, writes the item's
   held-out inputs with its seed (lassi.bench.registry stage_inputs, as the
   stages do), and runs it through the ttsim executor: exit status 0, no
   hang, sim_ub False, sim_gap None, no jit-stage error;
3. builds the C++ counterpart (with lassi_io.h) with gcc-native and runs it
   the same way through the native executor: exit status 0, no hang;
4. compares every output with the binary_io oracle's statistics under the
   item's declared tolerance: every output passes, and the outputs are
   exactly the item's declared ones, on both sides;
5. reads the TT host program with the CPU -> TT guard
   (TtMetalHost.host_compute_guard with the TT support files): host_compute
   False; loopback carries exactly the data-movement-only tag, the compute
   items no guard diagnostic.

Every program runs only through its executor, in the sandbox; the TT one on
ttsim, a simulator, so nothing opens a device (Agent Rule 9). Simulator wall
times are not asserted on and are never performance (Agent Rule 2).

The tests are marked `remote` and skip unless the host can run them (Linux,
the sandbox tools on PATH, a reachable user systemd manager, $LASSI_SCRATCH,
$LASSI_RUNS_ROOT, and $LASSI_TOOLCHAINS set, TMPDIR inside $LASSI_SCRATCH,
both pinned host compilers, the tt-metal tree installed, and the ttsim
library in place). With LASSI_REQUIRE_SANDBOX=1 such a host fails them
instead, so a silent skip never passes for evidence. They load the
manifest as it is, so a PLACEHOLDER in it (load_suite refuses it) fails
them, never skips them. Run
one item from the task's clean commit with `uv run tools/rx.py run -- '<command>'`,
the command being

    LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/bench/test_tt_pairs_remote.py -k loopback

and all five by leaving out `-k loopback`. Every build and run lies in a
temp directory under $LASSI_RUNS_ROOT, removed afterwards; the fetched
kernels stay in $LASSI_SCRATCH/bench, where a run reads them (Agent Rule 7).
The pinned trees are only read. No value here is a measurement.
"""

from __future__ import annotations

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
from lassi.toolchains.ttmetal_build import TtMetalHost

REPO = Path(__file__).resolve().parents[2]
SUITE = "tt-pairs-v0"
MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEMS = ("eltwise_binary", "eltwise_sfpu", "loopback", "matmul_multi_core", "matmul_single_core")
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk", "env", "sh", "git")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
TAG = "guard-data-movement-only"
DATA_MOVEMENT_ONLY = frozenset({"loopback"})
TO_TT, TO_CPP = bench.Direction("cpp", "tt"), bench.Direction("tt", "cpp")
# A reference run's limits: 16 CPUs and 8192 MiB, under which P4.9's sandboxed runs passed, and a 120 s wall limit:
# P4.9's clean runs of the upstream examples took at most 3.73 s of simulator wall time (exploratory, sizing only).
LIMITS = Limits(wall_s=120.0, memory_mb=8192, cpus=16)


def installed(pin_name: str) -> Path:
    """Return the install prefix of a pin under the resolved toolchains root."""
    return Path(os.environ["LASSI_TOOLCHAINS"]).resolve() / pins_module.read_pin(pin_name)["PREFIX_NAME"]


def host_problem() -> str:
    """Return why this host cannot run the tests, or "" when it can."""
    if not sys.platform.startswith("linux"):
        return f"the sandbox needs Linux, not {sys.platform}; run it through `uv run tools/rx.py run`"
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
    library = installed("ttsim") / pins_module.read_pin("ttsim")["LIBRARY"]
    if not library.is_file():
        return f"the pinned ttsim library {library} is not installed"
    return ""


PROBLEM = host_problem()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can run Tier A programs"),
]


@pytest.fixture(autouse=True)
def host_ready() -> None:
    """Fail, rather than skip, when LASSI_REQUIRE_SANDBOX=1 asks for a real sandbox and the host cannot run one."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")


@pytest.fixture(scope="module")
def toolchains() -> dict[str, BuiltToolchain]:
    """Build ttmetal-host and gcc-native as a run builds them: trees checked, pinned compilers checked by --version."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")
    root = Path(os.environ["LASSI_TOOLCHAINS"])
    return {"tt": build_toolchain("ttmetal-host", root), "cpp": build_toolchain("gcc-native", root)}


@pytest.fixture
def scratch_dirs() -> Iterator[Path]:
    """Create a temp directory under $LASSI_RUNS_ROOT for the two builds; remove it afterwards."""
    base = Path(tempfile.mkdtemp(prefix="lassi-tt-pairs.", dir=os.environ["LASSI_RUNS_ROOT"]))
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


def build_and_run(
    built: BuiltToolchain, executor: object, item: registry.SuiteItem, files: Mapping[str, str],
    harness: Mapping[str, str], workdir: Path,
) -> RunResult:
    """Build `files` with `harness` in `workdir`, write the item's inputs there, and run the artifact."""
    workdir.mkdir(parents=True)
    result = built.toolchain.build(files, workdir, harness=harness)
    stderr = (workdir / "compile.stderr").read_bytes().decode("utf-8", "replace")
    errors = [entry for entry in result.diagnostics if entry.severity == "error"]
    assert result.artifact is not None and not errors, f"{item.name} did not build:\n{stderr[-4000:]}"
    args = registry.stage_inputs(item, workdir)
    return executor.run(result.artifact, args, LIMITS)  # type: ignore[attr-defined]


def clean(run: RunResult, side: str) -> None:
    """Assert a clean run: exit status 0, no hang, and no simulator finding or kernel JIT error."""
    tail = f"{run.stdout[-3000:]}\n{run.stderr[-3000:]}"
    assert (run.exit_code, run.hang) == (0, False), f"{side}: exit {run.exit_code}, hang {run.hang}\n{tail}"
    jit = [entry for entry in run.diagnostics if entry.stage == "jit" and entry.severity == "error"]
    assert run.sim_gap is None and run.sim_ub is not True and jit == [], f"{side}: {run.diagnostics}\n{tail}"


def output_names(run: RunResult) -> list[str]:
    """Return the array names of a run's output files, sorted."""
    return sorted(read_array(path).name for path in run.output_files.values())


@pytest.mark.parametrize("item", ITEMS)
def test_an_items_references_run_clean_agree_and_pass_the_guard(
    item: str, toolchains: dict[str, BuiltToolchain], scratch_dirs: Path
) -> None:
    from lassi.executors.native import NativeExecutor
    from lassi.executors.ttsim import TtsimExecutor

    assert fetch_tool().main([str(MANIFEST)]) == 0, "the pinned files were fetched and every sha256 matched"
    suite = bench.load_suite(MANIFEST)
    root = registry.sources_dir(Path(os.environ["LASSI_SCRATCH"]), suite)
    spec = suite.item(item, purpose="eval")
    tt_files = suite.reference_target(item, TO_TT, root, purpose="eval")
    tt_harness = suite.support_files(item, root, purpose="eval", language="tt")
    cpp_files = suite.reference_target(item, TO_CPP, root, purpose="eval")
    cpp_harness = suite.support_files(item, root, purpose="eval", language="cpp")

    tt_run = build_and_run(toolchains["tt"], TtsimExecutor(), spec, tt_files, tt_harness, scratch_dirs / "tt" / "build")
    clean(tt_run, f"{item} tt reference on ttsim")
    assert tt_run.sim_ub is False, "the ttsim executor always checks for undefined behavior"
    cpp_run = build_and_run(
        toolchains["cpp"], NativeExecutor(), spec, cpp_files, cpp_harness, scratch_dirs / "cpp" / "build"
    )
    clean(cpp_run, f"{item} C++ counterpart, native")

    declared = sorted(spec.outputs)
    assert output_names(tt_run) == declared and output_names(cpp_run) == declared
    assert spec.tolerance is not None
    oracle = BinaryIOOracle(metric=spec.tolerance.metric, threshold=spec.tolerance.threshold)
    stats = oracle.compare(tt_run.output_files, cpp_run.output_files, sides=("tt reference", "cpp reference"))
    assert stats and all(entry.passed for entry in stats), f"{item}: {stats}"

    reading = TtMetalHost.host_compute_guard(tt_files, tt_harness)
    codes = [entry.code for entry in reading.diagnostics]
    assert reading.host_compute is False, f"{item}: {reading.diagnostics}"
    assert codes == ([TAG] if item in DATA_MOVEMENT_ONLY else []), f"{item}: {reading.diagnostics}"
