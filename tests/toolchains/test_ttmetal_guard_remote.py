"""Remote tests of the CPU -> TT host-compute guard against the pinned tt-metal (task P4.12).

Bible: Harness Contract (the CPU -> TT guard; data-movement-only programs
tagged, not failed), Toolchain Pins (the joint pin; the guard's identifier
sets are facts of tt-metal 5280a9cf, so a pin change re-checks them), Sandbox
(compiles), Agent Rules 1, 6, 7, 9, and 10. The guard's contract is in
test_ttmetal_guard.py and lassi.toolchains.ttmetal_guard's module
docstring; this file ties it to the pin. The guard reads no run, so the plan
needs no remote test for it (plans/p4-ttsim.md, P4.12); the task adds this
one anyway, so that the identifier sets and the fixtures are checked
against the pinned tree:

- The host sources of the six pinned examples the smoke driver knows
  (tools/ttsim_smoke.py example_sources, which reads them from the tree, so
  no tt-metal source text is in this file) go through read_host_compute with
  the harness files example_sources gives. add_2_integers_in_riscv and
  loopback create only data-movement kernels (plans/spikes/p4-ttsim-runtime.md,
  the unpack_to_dest table), so each carries the data-movement-only tag
  exactly once; eltwise_binary, eltwise_sfpu, matmul_single_core, and
  matmul_multi_core name a compute config, so none carries it. None reads
  host_compute True, and none gets a note that its host text could not be
  read or that the reader failed. The examples print their checks rather
  than write lassi_io outputs, so a "no output write" not-checked note is
  allowed beside the tag (ttmetal_guard's module docstring, The reading,
  step 2).
- The four hand-written acceptance programs under fixtures/host_compute/
  build with ttmetal-host as a run builds them, in the compile sandbox, with
  assets/harness/c/lassi_io.h as the harness file: no error diagnostic and
  the artifact `main` in place. Their kernel placeholders are placed, never
  compiled.

Nothing here runs a built program: no program starts the kernel JIT or
ttsim, and no command names a device (Agent Rules 6 and 9).

The tests are marked `remote` and skip unless the host can run them (Linux,
the sandbox tools on PATH, a reachable user systemd manager, $LASSI_SCRATCH,
$LASSI_RUNS_ROOT, and $LASSI_TOOLCHAINS set, TMPDIR inside $LASSI_SCRATCH,
the pin's EXECUTABLE present, and the tree installed); with
LASSI_REQUIRE_SANDBOX=1 they fail instead, so a silent skip never passes for
evidence. Run them from the task's clean commit with
`uv run tools/rx.py run -- '<command>'`, the command holding
`LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/toolchains/test_ttmetal_guard_remote.py`.
Every file they write lies in a temp directory under $LASSI_RUNS_ROOT or,
for build_toolchain's --version check, under TMPDIR on the scratch disk,
and is removed afterwards (Agent Rule 7); the pinned tree is only read. No
value here is a measurement.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from lassi.core.runner import BuiltToolchain, build_toolchain
from lassi.toolchains import pins as pins_module

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from ttsim_smoke import EXAMPLES, example_sources  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "host_compute"
HARNESS_HEADER = REPO / "assets" / "harness" / "c" / "lassi_io.h"
ACCEPTANCE = ("clean_offload", "host_loop_output", "data_movement_only", "compute_beside_host_loop")
NAME = "ttmetal-host"
PIN_NAME = "tt-metal"
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
OUTPUT = "main"
ATTACHMENT = "compile.stderr"
TAG = "guard-data-movement-only"
# The examples with no compute kernel (plans/spikes/p4-ttsim-runtime.md, the unpack_to_dest table).
DATA_MOVEMENT_ONLY = frozenset({"add_2_integers_in_riscv", "loopback"})
UNREADABLE = "could not be read"


def installed_tree() -> Path:
    """Return the pinned tree under the resolved toolchains root."""
    return Path(os.environ["LASSI_TOOLCHAINS"]).resolve() / pins_module.read_pin(PIN_NAME)["PREFIX_NAME"]


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
    executable = pins_module.read_pin(PIN_NAME).get("EXECUTABLE", "")
    if not executable or not Path(executable).is_file():
        return f"the pinned host compiler {executable!r} (EXECUTABLE in toolchains/tt-metal.pin) is not on this host"
    if not (installed_tree() / "lassi-install.txt").is_file():
        return f"{installed_tree()} is not an install of the pinned tt-metal"
    return ""


PROBLEM = host_problem()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can build tt-metal programs"),
]


@pytest.fixture(autouse=True)
def host_ready() -> None:
    """Fail, rather than skip, when LASSI_REQUIRE_SANDBOX=1 asks for a real sandbox and the host cannot run one."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")


@pytest.fixture(scope="module")
def built() -> BuiltToolchain:
    """Build ttmetal-host as a run builds it: the tree checked, the pinned clang checked by --version."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")
    return build_toolchain(NAME, Path(os.environ["LASSI_TOOLCHAINS"]))


@pytest.fixture
def workdir() -> Iterator[Path]:
    """Create a fresh attempt build directory under $LASSI_RUNS_ROOT; remove it afterwards."""
    base = Path(tempfile.mkdtemp(prefix="lassi-ttmetal-guard.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        build = base / "attempt00" / "build"
        build.mkdir(parents=True)
        yield build
    finally:
        shutil.rmtree(base)


def guard() -> ModuleType:
    """Return lassi.toolchains.ttmetal_guard; fail the test clearly while it is missing."""
    try:
        return importlib.import_module("lassi.toolchains.ttmetal_guard")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.toolchains.ttmetal_guard does not exist ({error}); task P4.12 adds it")


def fixture_files(case: str) -> dict[str, str]:
    """Return the files of one acceptance program under fixtures/host_compute/."""
    root = FIXTURES / case
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    return {path.relative_to(root).as_posix(): path.read_bytes().decode("utf-8") for path in paths}


@pytest.mark.parametrize("example", sorted(EXAMPLES))
def test_an_upstream_example_reads_as_the_pinned_api_says(example: str) -> None:
    sources = example_sources(installed_tree(), example)
    result = guard().read_host_compute(dict(sources.files), dict(sources.harness))
    codes = [item.code for item in result.diagnostics]
    assert result.host_compute is not True, f"{example} reads as host compute: {result.diagnostics}"
    assert codes.count(TAG) == (1 if example in DATA_MOVEMENT_ONLY else 0), f"{example}: {codes}"
    for item in result.diagnostics:
        assert UNREADABLE not in item.message, f"{example}: {item.message}"
        assert item.code in (TAG, "guard-not-checked"), f"{example}: {item}"


@pytest.mark.parametrize("case", ACCEPTANCE)
def test_an_acceptance_program_builds_at_the_pin(built: BuiltToolchain, workdir: Path, case: str) -> None:
    files = fixture_files(case)
    harness = {"lassi_io.h": HARNESS_HEADER.read_bytes().decode("utf-8")}
    result = built.toolchain.build(files, workdir, harness=harness)
    stderr = (workdir / ATTACHMENT).read_bytes().decode("utf-8", "replace")
    errors = [item for item in result.diagnostics if item.severity == "error"]
    assert errors == [], f"{case} did not build:\n{stderr}"
    assert result.artifact == workdir / OUTPUT and (workdir / OUTPUT).is_file(), f"{case} left no program:\n{stderr}"
    for path in files:
        if path.startswith("kernels/"):
            assert (workdir / path).read_bytes() == files[path].encode("utf-8"), f"{path} is placed, not compiled"
