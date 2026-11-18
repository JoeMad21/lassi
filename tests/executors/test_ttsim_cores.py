"""Tests for the ttsim executor's core count (task P17.70; plans/spikes/ttsim-cores.md).

Bible: Execution Backends (ttsim), Result Record (Diagnostic), Agent Rules 2
and 6. The spike's choice: the Watcher's last complete dump names, for each
core, the kernel ids of the last launch message it received; a worker core
whose k_ids hold a nonzero id ran a kernel. Under ttsim the device profiler
fails, so this is the way to count cores there; the count is a count of
cores, never a time (Agent Rule 2).

The contract these tests fix:
- count_worker_cores(log) reads the log's last complete dump and returns
  (its number, the worker cores with a nonzero kernel id, the worker core
  lines), or None when the log holds no complete dump. Only worker lines
  count (an idle ethernet core is not a worker), and a worker line without
  k_ids counts as a core that ran no kernel.
- cores_note(log) is Diagnostic(stage "run", severity "note", code
  lassi.core.capabilities CORES_CODE, message "cores used: <n> of <m>
  worker cores ran a kernel (Watcher dump #<k>); a count, not a time"), or
  None when count_worker_cores gives None or no worker line.
- TtsimExecutor(count_cores=True): after a scored run the stages read as
  clean (exit 0, no hang, no undefined behavior, no gap, no error), the
  program runs once more with TT_METAL_WATCHER=1 in a
  fresh copy of the workdir's pre-run files (as the hang rerun does), and
  the cores note of that copy's Watcher log is appended to the scored
  run's diagnostics; the scored run's status, streams, and outputs stay
  its own, and a rerun that did not exit 0 or hung gives no note.
  count_cores is False by default: one run, as before. A run that failed,
  hung, or exited 0 with a finding or a kernel JIT error is not counted.
- count_cores must be a bool (ValueError naming it), and it is one of the
  executor's config keys.
- The stages never put a cores note into a run's error text: run_findings
  carries only Watcher dumps of a hung run.

Every test uses a FakeSandbox (tests/executors/ttsim_fakes.py) and a
PLACEHOLDER install; nothing is executed. The Watcher logs here are
SYNTHETIC, in the format of the dumps the spike read; no value in this
module is a measurement.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from ttsim_fakes import (
    RESERVED,
    WATCHER_NAME,
    FakeInstall,
    FakeSandbox,
    canned,
    make_artifact,
    make_install,
    same_path,
    writes_watcher_log,
)

from lassi.core import stages
from lassi.core.capabilities import CORES_CODE
from lassi.core.interfaces import Limits, RunResult
from lassi.core.record import Diagnostic

LIMITS = Limits(wall_s=30.0, memory_mb=4096, cpus=4)
INPUTS = ["in0.lassiio", "out0.lassiio"]

BUSY = "Device 0 worker core(x= {x},y= 0) virtual(x=18,y=18):   GW,   W,   W,   W,   W  rmsg:H0D|BNT h_id:  0 " \
    "smsg:DDDD k_ids:  1|  2|  3|  3|  3"
IDLE = "Device 0 worker core(x= {x},y= 1) virtual(x=18,y=19):   GW,   W,   W,   W,   W  rmsg:H0D|BNT h_id:  0 " \
    "smsg:DDDD k_ids:  0|  0|  0|  0|  0"
NO_IDS = "Device 0 worker core(x= {x},y= 2) virtual(x=18,y=20):   GW,   W,   W,   W,   W  rmsg:H0D|BNT h_id:  0"
ETH = "Device 0 idleth core(x= {x},y= 0) virtual(x=25,y=16):   GW,   W,   W,   W,   W  rmsg:H0D|BNT h_id:  0 " \
    "smsg:DDDD k_ids:  4|  4|  4|  4|  4"
# SYNTHETIC: dump #1 complete with one busy worker; dump #2 complete with 3 busy, 2 idle, 1 without k_ids, and an idle
# ethernet line whose ids are nonzero; dump #3 started but never completed.
LOG = "\n".join([
    "Dump #1 at 0.500s",
    BUSY.format(x=0),
    "Dump #1 completed at 0.510s",
    "Dump #2 at 1.000s",
    BUSY.format(x=0), BUSY.format(x=1), BUSY.format(x=2),
    IDLE.format(x=0), IDLE.format(x=1),
    NO_IDS.format(x=0),
    ETH.format(x=0),
    "k_id[  1]: kernels/dataflow/reader.cpp",
    "Dump #2 completed at 1.010s",
    "Dump #3 at 1.500s",
    BUSY.format(x=3), BUSY.format(x=4),
]) + "\n"


@pytest.fixture
def ttsim() -> ModuleType:
    """Return the lassi.executors.ttsim module."""
    from lassi.executors import ttsim

    return ttsim


@pytest.fixture(autouse=True)
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make tmp_path/runs the runs root, set the scratch and home roots beside it, and unset $LASSI_TOOLCHAINS."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(runs))
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LASSI_TOOLCHAINS", raising=False)
    return runs


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeInstall:
    """Return the PLACEHOLDER install the executor's checks accept."""
    return make_install(tmp_path, monkeypatch)


def run_counting(
    ttsim: ModuleType, install: FakeInstall, artifact: Path, results: list[Any], effects: list[Any], **settings: Any
) -> tuple[RunResult, FakeSandbox]:
    """Run `artifact` on a FakeSandbox with the given executor settings; return the result and the fake."""
    fake = FakeSandbox(results=results, effects=effects)
    executor = ttsim.TtsimExecutor(toolchains=str(install.root), sandbox=fake, **settings)
    return executor.run(artifact, INPUTS, LIMITS), fake


def cores_notes(result: RunResult) -> list[Diagnostic]:
    """Return the result's cores notes."""
    return [item for item in result.diagnostics if item.code == CORES_CODE]


# ---------------------------------------------------------------------------
# Reading the dump


def test_the_count_reads_the_last_complete_dump_and_only_its_worker_lines(ttsim: ModuleType) -> None:
    assert ttsim.count_worker_cores(LOG) == ("2", 3, 6)


def test_a_log_with_no_complete_dump_gives_no_count(ttsim: ModuleType) -> None:
    assert ttsim.count_worker_cores("Dump #1 at 0.5s\n" + BUSY.format(x=0) + "\n") is None
    assert ttsim.count_worker_cores("") is None


def test_the_cores_note_names_the_count_the_workers_and_the_dump(ttsim: ModuleType) -> None:
    note = ttsim.cores_note(LOG)
    assert note == Diagnostic(
        stage="run",
        severity="note",
        code=CORES_CODE,
        message="cores used: 3 of 6 worker cores ran a kernel (Watcher dump #2); a count, not a time",
    )


def test_no_complete_dump_or_no_worker_line_gives_no_note(ttsim: ModuleType) -> None:
    assert ttsim.cores_note("Dump #1 at 0.5s\n") is None
    only_eth = "Dump #1 at 0.5s\n" + ETH.format(x=0) + "\nDump #1 completed at 0.6s\n"
    assert ttsim.cores_note(only_eth) is None


def test_the_cores_code_is_its_own() -> None:
    assert CORES_CODE == "cores" and CORES_CODE != stages.WATCHER_CODE


# ---------------------------------------------------------------------------
# The executor setting


def test_count_cores_is_off_by_default_and_one_run_happens(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    result, fake = run_counting(ttsim, install, make_artifact(runs_root), [canned(), canned()], [None, None])
    assert len(fake.calls) == 1 and WATCHER_NAME not in fake.calls[0].environment
    assert cores_notes(result) == []


def test_a_clean_run_is_counted_in_a_fresh_copy_with_watcher(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    artifact = make_artifact(runs_root)
    workdir = Path(os.path.realpath(artifact.parent))

    def scored_run_writes(spec: Any, argv: list[str]) -> None:
        """Write the scored run's output file."""
        (workdir / "out0.lassiio").write_bytes(b"\x01")

    scored = canned(stdout="SYNTHETIC scored stdout\n", stderr="SYNTHETIC scored stderr\n")
    rerun = canned(stdout="SYNTHETIC rerun stdout\n")
    result, fake = run_counting(
        ttsim, install, artifact, [scored, rerun], [scored_run_writes, writes_watcher_log(LOG)], count_cores=True
    )
    assert len(fake.calls) == 2
    first, second = fake.calls
    assert WATCHER_NAME not in first.environment and second.environment[WATCHER_NAME] == "1"
    assert same_path(second.spec.workdir, workdir / RESERVED / "watcher"), "the rerun runs in the fresh copy"
    assert second.limits == first.limits == LIMITS and second.argv[1:] == first.argv[1:] == INPUTS
    assert (result.exit_code, result.stdout, result.stderr) == (0, scored.stdout, scored.stderr)
    assert sorted(result.output_files) == ["out0.lassiio"], "nothing of the rerun's copy is an output file"
    assert cores_notes(result) == [ttsim.cores_note(LOG)]


@pytest.mark.parametrize("outcome", ["failed", "hung"])
def test_a_failed_or_hung_run_is_not_counted(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path, outcome: str
) -> None:
    scored = canned(returncode=1) if outcome == "failed" else canned(returncode=137, hang=True)
    effects = [None, writes_watcher_log(LOG)]
    result, fake = run_counting(
        ttsim, install, make_artifact(runs_root), [scored, canned()], effects, count_cores=True
    )
    assert cores_notes(result) == []
    # A hang still gets its one Watcher rerun (task P4.11); a failed run gets none.
    assert len(fake.calls) == (2 if outcome == "hung" else 1)


@pytest.mark.parametrize("rerun", [canned(returncode=5), canned(returncode=137, hang=True)], ids=["failed", "hung"])
def test_a_rerun_that_did_not_finish_cleanly_adds_no_note(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path, rerun: Any
) -> None:
    # Its last complete dump may predate the kernel launches, so its count is not taken.
    effects = [None, writes_watcher_log(LOG)]
    result, fake = run_counting(
        ttsim, install, make_artifact(runs_root), [canned(), rerun], effects, count_cores=True
    )
    assert len(fake.calls) == 2 and cores_notes(result) == []


JIT_LOG = (
    "brisc build failed. Log: In file included from SYNTHETIC.hpp:1,\n"
    "/SYNTHETIC/kernels/k.cpp:4:5: error: SYNTHETIC kernel error\n\n"
)


@pytest.mark.parametrize(
    "stdout",
    [JIT_LOG, "[1234] ERROR: UnimplementedFunctionality: SYNTHETIC_function: SYNTHETIC message\n",
     "[1234] ERROR: UndefinedBehavior: SYNTHETIC_function: SYNTHETIC message\n"],
    ids=["jit-error", "gap", "undefined-behavior"],
)
def test_a_run_that_exited_0_but_is_not_clean_is_not_counted(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path, stdout: str
) -> None:
    # The stages read such a run as S1 or as a failed run, so it is never counted (and never prompts with a count).
    effects = [None, writes_watcher_log(LOG)]
    result, fake = run_counting(
        ttsim, install, make_artifact(runs_root), [canned(stdout=stdout), canned()], effects, count_cores=True
    )
    assert len(fake.calls) == 1 and cores_notes(result) == []


def test_a_rerun_with_no_complete_dump_adds_no_note(ttsim: ModuleType, install: FakeInstall, runs_root: Path) -> None:
    effects = [None, writes_watcher_log("Dump #1 at 0.5s\n")]
    result, _ = run_counting(
        ttsim, install, make_artifact(runs_root), [canned(), canned()], effects, count_cores=True
    )
    assert cores_notes(result) == []


@pytest.mark.parametrize("value", [1, "yes", None])
def test_count_cores_must_be_a_bool(ttsim: ModuleType, value: Any) -> None:
    with pytest.raises(ValueError, match="count_cores"):
        ttsim.TtsimExecutor(count_cores=value)


def test_count_cores_is_a_config_key(ttsim: ModuleType) -> None:
    assert "count_cores" in ttsim.TtsimExecutor.config_keys


# ---------------------------------------------------------------------------
# The stages


def test_the_stages_never_put_a_cores_note_into_a_runs_error_text(ttsim: ModuleType) -> None:
    note = ttsim.cores_note(LOG)
    assert note is not None
    for hang, exit_code in ((False, 1), (True, 137)):
        run = RunResult(
            exit_code=exit_code, hang=hang, stdout="", stderr="", output_files={}, wall_s=1.0, diagnostics=[note]
        )
        assert note.message not in stages.run_findings(run, simulator=True)
        assert note.message not in stages.run_error_text(run, LIMITS, {}, simulator=True)


def test_the_prompt_filters_drop_a_cores_note(ttsim: ModuleType) -> None:
    note = ttsim.cores_note(LOG)
    error = Diagnostic(stage="jit", severity="error", code=None, message="SYNTHETIC kernel error")
    assert stages.prompt_diagnostics([note, error]) == [error]
    # A note whose code equals the run's gap class is not a finding (a program may print a finding of any class).
    run = RunResult(exit_code=1, hang=False, stdout="", stderr="", output_files={}, wall_s=1.0,
                    diagnostics=[note], sim_gap=CORES_CODE)
    assert stages.gap_findings(run) == []
