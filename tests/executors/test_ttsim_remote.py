"""Remote tests of the ttsim executor on the build host (task P4.11).

Bible: Execution Backends (the ttsim row), ttsim Facts, Sandbox, Harness
Contract (the hang diagnostic with the Watcher dump), Agent Rules 2, 6, 7,
and 9. The contract is in test_ttsim.py and test_ttsim_findings.py; these
tests run it for real, in the P0.16 sandbox, on ttsim's virtual Wormhole:

1. A sandboxed program the executor runs (a small shell script, the
   artifact) sees no Tenstorrent device node: it only tests whether the path
   exists and prints the answer; it opens nothing.
2. The gate's example, add_2_integers_in_riscv, built from the pinned tree
   with the ttmetal-host toolchain exactly as a run builds it, runs clean:
   exit status 0, no hang, sim_ub False, sim_gap None, no jit error, and its
   own check's line "Success: Result is 21".
3. A copy of the example whose kernel holds P4.9's seeded 4-byte load from
   an address 2 mod 4 gives sim_ub True with a NonContractualBehavior
   diagnostic.
4. A copy whose kernel holds P4.9's seeded wait on a word nothing writes,
   run with a 30 s wall limit, hangs, and the Watcher rerun adds a note
   naming the hung core's semaphore wait and the kernel's path.

The seeded lines are P4.9's hand-written reference edits, read at test time
from the tracked captures' status.json (seed_line); each goes in as its own
line right after the line that opens the kernel's entry function, as P4.9's
batch placed it. The example's host program and kernel are read from the
pinned tree at test time with the smoke driver's helpers
(tools/ttsim_smoke.py example_sources and seeded, which hold the example
layout once), so no tt-metal source text is in this file. Every
program runs only through the executor, in the sandbox, on the simulator;
nothing opens a device (Agent Rule 9).

The tests are marked `remote` and skip unless the host can run them (Linux,
the sandbox tools on PATH, a reachable user systemd manager,
$LASSI_SCRATCH, $LASSI_RUNS_ROOT, and $LASSI_TOOLCHAINS set, TMPDIR inside
$LASSI_SCRATCH, the pinned host compiler, the tt-metal tree installed, and
the ttsim library in place); with LASSI_REQUIRE_SANDBOX=1 they fail instead,
so a silent skip never passes for evidence. Run them with
`uv run tools/rx.py run -- '<command>'`, the command holding
`LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/executors/test_ttsim_remote.py`.
They take a few minutes (PROJECTED, not measured), most of it the hang's two runs. Every file
they write lies in a temp directory under $LASSI_RUNS_ROOT, removed
afterwards (Agent Rule 7); the pinned trees are only read. Simulator wall
times are not asserted on and are never performance. No value here is a
measurement.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from lassi.core.interfaces import Limits, RunResult
from lassi.core.runner import BuiltToolchain, build_toolchain
from lassi.toolchains import pins as pins_module

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from ttsim_smoke import SUCCESS, example_sources  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ttsim"
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk", "env", "sh")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
EXAMPLE = "add_2_integers_in_riscv"
SUCCESS_LINE = SUCCESS[EXAMPLE]
# The device node directory a Tenstorrent driver creates; the probe only tests whether it exists.
DEVICE_NODE = "/dev/tenstorrent"
CLEAN_LIMITS = Limits(wall_s=120.0, memory_mb=8192, cpus=16)
HANG_LIMITS = Limits(wall_s=30.0, memory_mb=8192, cpus=16)


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
    executable = pins_module.read_pin("tt-metal").get("EXECUTABLE", "")
    if not executable or not Path(executable).is_file():
        return f"the pinned host compiler {executable!r} (EXECUTABLE in toolchains/tt-metal.pin) is not on this host"
    if not (installed("tt-metal") / "lassi-install.txt").is_file():
        return f"{installed('tt-metal')} is not an install of the pinned tt-metal"
    library = installed("ttsim") / pins_module.read_pin("ttsim")["LIBRARY"]
    if not library.is_file():
        return f"the pinned ttsim library {library} is not installed"
    return ""


PROBLEM = host_problem()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can run ttsim programs"),
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
    return build_toolchain("ttmetal-host", Path(os.environ["LASSI_TOOLCHAINS"]))


@pytest.fixture
def workdir() -> Iterator[Path]:
    """Create a fresh attempt build directory under $LASSI_RUNS_ROOT; remove it afterwards."""
    base = Path(tempfile.mkdtemp(prefix="lassi-ttsim.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        build = base / "attempt00" / "build"
        build.mkdir(parents=True)
        yield build
    finally:
        shutil.rmtree(base)


def executor() -> Any:
    """Return the ttsim executor with its defaults: the pinned install under $LASSI_TOOLCHAINS, the real sandbox."""
    from lassi.executors.ttsim import TtsimExecutor

    return TtsimExecutor()


def example_files(seed: str | None = None) -> tuple[dict[str, str], str]:
    """Return the gate example's files read from the pinned tree, with `seed` placed in its kernel, and the kernel.

    The host program keeps its own name; its one named kernel lies at the
    path it names, where the kernel JIT looks first in the working directory
    (tools/ttsim_smoke.py example_sources, which puts the seed in with seeded).
    """
    sources = example_sources(installed("tt-metal"), EXAMPLE, seed)
    assert len(sources.kernels) == 1 and not sources.harness, f"the gate example names one kernel: {sources}"
    return dict(sources.files), sources.kernels[0]


def seed_line(capture: str) -> str:
    """Return the seeded line P4.9 recorded for a capture (status.json seed_line)."""
    status = json.loads((FIXTURES / capture / "status.json").read_bytes().decode("utf-8"))
    return str(status["seed_line"])


def build_and_run(built: BuiltToolchain, workdir: Path, files: dict[str, str], limits: Limits) -> RunResult:
    """Build `files` with ttmetal-host in `workdir` and run the artifact through the ttsim executor."""
    result = built.toolchain.build(files, workdir)
    stderr = (workdir / "compile.stderr").read_text(encoding="utf-8")
    assert result.artifact is not None, f"the example did not build:\n{stderr}"
    return executor().run(result.artifact, [], limits)


def jit_errors(run: RunResult) -> list[Any]:
    """Return the run's jit-stage errors."""
    return [item for item in run.diagnostics if item.stage == "jit" and item.severity == "error"]


def test_a_sandboxed_program_sees_no_tenstorrent_device_node(workdir: Path) -> None:
    probe = workdir / "probe.sh"
    probe.write_bytes(
        (
            "#!/bin/sh\n"
            "# Tests whether a Tenstorrent device node is visible and prints the answer; it opens nothing.\n"
            f"if [ -e {DEVICE_NODE} ]; then echo 'tenstorrent-node: present'\n"
            "else echo 'tenstorrent-node: absent'; fi\n"
            "for entry in /dev/*; do\n"
            f'  case "$entry" in {DEVICE_NODE}*) echo "tenstorrent-entry: $entry" ;; esac\n'
            "done\n"
        ).encode("ascii")
    )
    probe.chmod(0o755)
    run = executor().run(probe, [], Limits(wall_s=30.0, memory_mb=1024, cpus=1))
    assert (run.exit_code, run.hang) == (0, False), run.stderr
    assert run.stdout == "tenstorrent-node: absent\n", run.stdout


def test_the_gate_example_runs_clean_on_ttsim(built: BuiltToolchain, workdir: Path) -> None:
    files, _ = example_files()
    run = build_and_run(built, workdir, files, CLEAN_LIMITS)
    assert (run.exit_code, run.hang) == (0, False), f"{run.stdout[-2000:]}\n{run.stderr[-2000:]}"
    assert (run.sim_ub, run.sim_gap, jit_errors(run)) == (False, None, [])
    assert SUCCESS_LINE in run.stdout, "the example's own check passed"


def test_a_seeded_unaligned_load_gives_undefined_behavior(built: BuiltToolchain, workdir: Path) -> None:
    files, _ = example_files(seed_line("non-contractual-behavior"))
    run = build_and_run(built, workdir, files, CLEAN_LIMITS)
    assert run.sim_ub is True and run.sim_gap is None, run.stdout[-2000:]
    codes = [item.code for item in run.diagnostics if item.stage == "run" and item.severity == "error"]
    assert "NonContractualBehavior" in codes, codes


def test_a_seeded_hang_is_stopped_and_gets_the_watcher_note(built: BuiltToolchain, workdir: Path) -> None:
    files, kernel = example_files(seed_line("hang"))
    run = build_and_run(built, workdir, files, HANG_LIMITS)
    assert run.hang is True, f"exit {run.exit_code}; {run.stderr[-2000:]}"
    notes = [item for item in run.diagnostics if item.code == "watcher"]
    assert len(notes) == 1, run.diagnostics
    (note,) = notes
    assert (note.stage, note.severity) == ("run", "note")
    assert kernel in note.message and "NSW" in note.message, note.message
