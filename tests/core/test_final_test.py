"""Tests for the loop's final test on a second executor, silicon after ttsim (task P17.55).

Bible: Component Interfaces (the Stage row: run_loop and the oracle),
Project Recipes (the loop opt-ins), Result Record (Attempt.final_test,
RunInfo.device_timing), Execution Backends (tt_silicon; Tenstorrent
performance telemetry), Agent Rules 2, 6, and 14. The owner's answer to
OQ-099 on 2026-10-29: "It should be like TT-SIM. SIlicon [sic] is there as a final
test gate, no human's required." plans/p17-portable.md, task P17.55.

The contract these tests fix (lassi.core.recipe, lassi.core.stages,
lassi.core.runner, lassi.core.record, lassi.core.trial_md, lassi.scoring,
lassi.analysis.tt_rates, plans/runs/rngd-sweep/tabulate.py):

- A recipe's final_executor section binds a second Executor at
  final_executor.kind. The loader refuses it under faithful: true, without
  loop.correct_on_mismatch true, and when its executor does not declare
  runs_code and sandboxed (Agent Rule 6); the runner refuses it, before any
  directory exists, when no listed stage declares final_test (run_loop
  does). Its binding is never one of the trial's own executors.
- An attempt whose clean run passed the oracle on the trial's executor is
  built again in attemptNN/final and run once on the final executor on the
  same held-out inputs, with a wall limit of at least FINAL_WALL_FLOOR_S;
  its outputs go through the same oracle. Attempt.final_test records that
  run whole: the executor, its device, the stage, the run with its output
  files and device timing (self_reported, a model's program), the
  alignment, and passed. A pass ends the loop; a failure (a mismatch, a
  failed run) gives the attempt a final-test error and is fed back as a run
  correction, after words naming the simulator and the hardware, within the
  one correction cap; left at the cap it ends the trial at correction-cap.
  An attempt that fails the oracle on the trial's executor never reaches the
  final test, and an error the final executor raises stops the run.
- provenance.json names the final executor and its device, run.md shows
  them, and its pins join the run's. trial.md shows a Final test section.
- The lassi profile's correct reads a failed final test as 0.0; df-v0
  scores an attempt with a final test by that run's stage and alignment;
  tt_rates gives each final test's rates labelled with its device; the
  sweep tabulator writes final_tests.tsv.

The suite, programs, replies, arrays, device lines, and device timings are
SYNTHETIC; the fakes compile nothing, start no process, and open no device.
No value here is a measurement.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from test_correct_on_mismatch import (
    CPP,
    INPUTS,
    ITEM,
    MODEL_ID,
    SUITE,
    TT,
    Behavior,
    Script,
    candidate,
    doubled,
    f32_values,
    make_registry,
    recipe_data,
    write_recipe,
    write_suite,
    wrong,
)

from lassi.analysis import tt_rates
from lassi.bench.registry import Work
from lassi.core.interfaces import Limits, RunResult
from lassi.core.recipe import RecipeError, load_recipe, run_executor_bindings
from lassi.core.record import (
    Alignment,
    Attempt,
    DeviceProgram,
    DeviceTiming,
    FinalTest,
    RunInfo,
    Trial,
    from_dict,
    make_trial_id,
    to_dict,
)
from lassi.core.registry import Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import FINAL_TEST_FAILED, FINAL_WALL_FLOOR_S, RunLoopStage
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.executors.sandbox import SandboxUnavailableError
from lassi.harness.lassi_io import write_array
from lassi.scoring.df_v0 import DfV0Profile
from lassi.scoring.lassi_profile import correct_value

REPO = Path(__file__).resolve().parents[2]
FINAL_KIND = "final_fake"
FINAL_DEVICE = "SYNTHETIC Tenstorrent Wormhole n300 x 8 (silicon)"
ON = {"correct_on_mismatch": True}
FINAL_SECTION = {"kind": FINAL_KIND}
# A SYNTHETIC tt-metal pin pair set the fake final executor declares (not the repository's pin).
FINAL_PINS = {"tt_metal": {"VERSION": "SYNTHETIC-tt-metal"}}
# A SYNTHETIC device time: one program of 1000 cycles, 1000 ns at 1000 MHz.
TIMING = DeviceTiming(
    source="tt-metal device profiler", build="SYNTHETIC-tt-metal", arch="wormhole_b0", aiclk_mhz=1000,
    programs=[DeviceProgram(device_id=0, op_name="", call_count=1, cores=1, kernel_cycles=1000, kernel_ns=1000)],
    self_reported=False,
)


# ---------------------------------------------------------------------------
# The SYNTHETIC final executor


@dataclass
class FinalScript:
    """What the final fake runs, in order: each final test's behavior or failure, and what it saw."""

    runs: list[Behavior | RunResult | Exception]
    limits: list[Limits] = field(default_factory=list)
    workdirs: list[Path] = field(default_factory=list)
    inputs: list[list[float]] = field(default_factory=list)


def final_executor(final: FinalScript) -> type:
    """Return an Executor class that plays the Wormhole cards: it runs the scripted final tests and opens nothing."""

    class FinalFake:
        """A SYNTHETIC silicon executor: it writes the scripted outputs with a SYNTHETIC device time."""

        name = FINAL_KIND
        capabilities = frozenset({"runs_code", "sandboxed", "silicon"})
        config_keys: frozenset[str] = frozenset()
        pins = FINAL_PINS

        def device(self) -> str:
            """Return the SYNTHETIC device line."""
            return FINAL_DEVICE

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Run the next scripted final test: write its outputs, or return or raise its scripted failure."""
            workdir = Path(artifact).parent
            final.limits.append(limits)
            final.workdirs.append(workdir)
            step = final.runs.pop(0)
            if isinstance(step, Exception):
                raise step
            if isinstance(step, RunResult):
                return step
            values = f32_values(workdir / inputs[0])
            final.inputs.append(values)
            files = {}
            for file, (name, result) in step(values).items():
                write_array(workdir / file, name, "f32", (len(result),), struct.pack(f"<{len(result)}f", *result))
                files[file] = workdir / file
            return RunResult(exit_code=0, hang=False, stdout="SYNTHETIC\n", stderr="", output_files=files,
                             wall_s=2.0, device_timing=TIMING)

    return FinalFake


def registry_with_final(script: Script, final: FinalScript) -> Registry:
    """Return test_correct_on_mismatch's registry with the final fake registered."""
    registry = make_registry(script)
    registry.register("Executor", FINAL_KIND, final_executor(final))
    return registry


def run_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: Script, final: FinalScript, loop: Mapping[str, Any],
    **changes: Any,
) -> tuple[Path, Trial]:
    """Run the recipe with a final_executor section; return the run directory and its one trial."""
    bench = write_suite(tmp_path, monkeypatch, INPUTS)
    data = recipe_data(loop, final_executor=dict(FINAL_SECTION), **changes)
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench,
                         registry=registry_with_final(script, final))
    run_dir = run_recipe(write_recipe(tmp_path, data, "final"), options)
    trial_id = make_trial_id("final", MODEL_ID, SUITE, f"{CPP}-{TT}", ITEM, 1)
    return run_dir, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))


def refused_load(tmp_path: Path, script: Script, final: FinalScript, data: Mapping[str, Any]) -> str:
    """Load `data` and return the RecipeError's message."""
    with pytest.raises(RecipeError) as caught:
        load_recipe(write_recipe(tmp_path, data, "final"), registry=registry_with_final(script, final))
    return str(caught.value)


def failed_run(exit_code: int) -> RunResult:
    """Return a SYNTHETIC failed run on the final executor."""
    return RunResult(exit_code=exit_code, hang=False, stdout="", stderr="SYNTHETIC device error\n", wall_s=2.0)


def final_errors(attempt: Attempt) -> list[str]:
    """Return the messages of the attempt's final-test errors."""
    return [item.message for item in attempt.diagnostics if item.code == FINAL_TEST_FAILED]


# ---------------------------------------------------------------------------
# Load and start


def test_the_section_is_refused_under_faithful_and_without_correct_on_mismatch(tmp_path: Path) -> None:
    script, final = Script(replies=[], attempts=[]), FinalScript(runs=[])
    message = refused_load(tmp_path, script, final, recipe_data(None, final_executor=dict(FINAL_SECTION)))
    assert "final_executor" in message and "correct_on_mismatch" in message
    faithful = recipe_data(None, final_executor=dict(FINAL_SECTION), faithful=True)
    message = refused_load(tmp_path, script, final, faithful)
    assert "final_executor" in message and "faithful" in message


def test_the_section_needs_an_executor_that_runs_programs_in_the_sandbox(tmp_path: Path) -> None:
    script, final = Script(replies=[], attempts=[]), FinalScript(runs=[])
    registry = registry_with_final(script, final)

    class Unsandboxed:
        """A fake executor that runs programs outside the sandbox."""

        name = "unsandboxed"
        capabilities = frozenset({"runs_code"})
        config_keys: frozenset[str] = frozenset()

    registry.register("Executor", "unsandboxed", Unsandboxed)
    data = recipe_data(ON, final_executor={"kind": "unsandboxed"})
    with pytest.raises(RecipeError) as caught:
        load_recipe(write_recipe(tmp_path, data, "final"), registry=registry)
    assert "final_executor.kind" in str(caught.value) and "sandboxed" in str(caught.value)


def test_the_binding_is_not_one_of_the_trials_executors(tmp_path: Path) -> None:
    script, final = Script(replies=[], attempts=[]), FinalScript(runs=[])
    data = recipe_data(ON, final_executor=dict(FINAL_SECTION))
    recipe = load_recipe(write_recipe(tmp_path, data, "final"), registry=registry_with_final(script, final))
    assert [binding.where for binding in run_executor_bindings(recipe.bindings)] == ["executor.kind"]
    assert any(binding.where == "final_executor.kind" for binding in recipe.bindings)


def test_the_runner_refuses_a_section_no_listed_stage_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = write_suite(tmp_path, monkeypatch, INPUTS)
    script, final = Script(replies=[], attempts=[]), FinalScript(runs=[])
    registry = registry_with_final(script, final)

    class LoopWithoutFinal(RunLoopStage):
        """run_loop as it was before task P17.55: it honors correct_on_mismatch but runs no final test."""

        final_test = False

    registry.register("Stage", "loop_without_final", LoopWithoutFinal)
    stages = ["baseline", "generate", "compile_loop", "loop_without_final", "oracle"]
    data = recipe_data(ON, final_executor=dict(FINAL_SECTION), stages=stages)
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench, registry=registry)
    with pytest.raises(RunError) as caught:
        run_recipe(write_recipe(tmp_path, data, "final"), options)
    assert "final_executor.kind" in str(caught.value) and "run_loop" in str(caught.value)
    assert not (tmp_path / "runs-root").exists(), "a refused run creates no directory"


# ---------------------------------------------------------------------------
# The loop


def test_a_pass_on_ttsim_then_on_silicon_passes_the_trial_and_records_both_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, final = Script(replies=[candidate(0)], attempts=[doubled]), FinalScript(runs=[doubled])
    run_dir, trial = run_final(tmp_path, monkeypatch, script, final, ON)
    (attempt,) = trial.attempts
    assert attempt.stage_reached == "S5" and attempt.alignment.mean == 1.0, "the ttsim run passed"
    test = attempt.final_test
    assert test is not None and test.passed and test.stage_reached == "S5"
    assert (test.executor, test.device) == (FINAL_KIND, FINAL_DEVICE)
    assert test.alignment.mean == 1.0 and test.run.outputs and test.run.wall_s == 2.0
    assert test.run.device_timing is not None and test.run.device_timing.self_reported is True
    assert attempt.run.wall_s == 1.0 and attempt.run.device_timing is None, "the ttsim run stays the attempt's"
    assert final.inputs == [script.inputs[-1]], "the final test runs on the same held-out inputs"
    assert final.workdirs[0].name == "final" and final.workdirs[0].parent.name == "attempt00"
    assert final.limits[0].wall_s >= FINAL_WALL_FLOOR_S
    assert [request.stage for request in trial.requests or []] == ["generate"]
    assert trial.final.end_reason is None and final_errors(attempt) == []
    manifest = json.loads((run_dir / "provenance.json").read_text(encoding="ascii"))
    assert manifest["final_executor"] == {"executor": FINAL_KIND, "device": FINAL_DEVICE}
    assert manifest["pins"].get("tt_metal") == "SYNTHETIC-tt-metal", "the final executor's pins are recorded"
    page = (run_dir / "run.md").read_text(encoding="ascii")
    assert "Final executor" in page and FINAL_DEVICE in page
    trial_page = (trial_dir(run_dir, trial.trial_id) / "trial.md").read_text(encoding="ascii")
    assert "### Final test" in trial_page and FINAL_DEVICE in trial_page


def test_a_mismatch_on_silicon_is_fed_back_and_the_next_attempt_runs_on_ttsim_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script(replies=[candidate(0), candidate(1)], attempts=[doubled, doubled])
    final = FinalScript(runs=[wrong, doubled])
    _, trial = run_final(tmp_path, monkeypatch, script, final, ON)
    first, second = trial.attempts
    assert first.final_test is not None and not first.final_test.passed
    assert first.final_test.stage_reached == "S5" and first.final_test.alignment.mean == 0.0
    assert any(item.code == "oracle-mismatch" for item in first.final_test.diagnostics)
    (error,) = final_errors(first)
    assert FINAL_KIND in error and "oracle" in error
    assert first.alignment.mean == 1.0, "the ttsim alignment stays the attempt's"
    assert second.final_test is not None and second.final_test.passed
    assert [request.stage for request in trial.requests or []] == ["generate", "run_loop"]
    prompt = script.requests[1][-1].content
    assert "passed the oracle on the trial's executor" in prompt or "on the simulator" in prompt
    assert "on the hardware" in prompt, "the correction says the hardware run failed"
    assert trial.final.end_reason is None and trial.final.corrections == 1
    assert script.attempts == [] and final.runs == []


def test_a_failed_run_on_silicon_is_fed_back_as_a_run_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Script(replies=[candidate(0), candidate(1)], attempts=[doubled, doubled])
    final = FinalScript(runs=[failed_run(3), doubled])
    _, trial = run_final(tmp_path, monkeypatch, script, final, ON)
    first = trial.attempts[0]
    assert first.final_test is not None and first.final_test.stage_reached == "S4"
    assert [item.code for item in first.final_test.diagnostics] == ["run-error"]
    assert "status 3" in final_errors(first)[0]
    prompt = script.requests[1][-1].content
    assert "exited with status 3" in prompt and "SYNTHETIC device error" in prompt
    assert trial.attempts[1].final_test is not None and trial.attempts[1].final_test.passed


def test_a_failed_final_test_at_the_cap_ends_the_trial_at_correction_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script(replies=[candidate(0), candidate(1)], attempts=[doubled, doubled])
    final = FinalScript(runs=[wrong, wrong])
    _, trial = run_final(tmp_path, monkeypatch, script, final, {"max_corrections": 1, **ON})
    reason = trial.final.end_reason
    assert reason is not None and reason.code == "correction-cap"
    assert "final test" in reason.message and FINAL_KIND in reason.message
    assert len(script.requests) == 2 and final.runs == []


def test_a_mismatch_on_ttsim_never_reaches_the_final_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Script(replies=[candidate(0), candidate(1)], attempts=[wrong, doubled])
    final = FinalScript(runs=[doubled])
    _, trial = run_final(tmp_path, monkeypatch, script, final, ON)
    first, second = trial.attempts
    assert first.final_test is None, "only a pass on the trial's executor reaches the final test"
    assert second.final_test is not None and second.final_test.passed
    assert len(final.workdirs) == 1


def test_an_error_the_final_executor_raises_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    refusal = SandboxUnavailableError("SYNTHETIC: a quarantine marker exists")
    script = Script(replies=[candidate(0)], attempts=[doubled])
    final = FinalScript(runs=[refusal])
    with pytest.raises(SandboxUnavailableError):
        run_final(tmp_path, monkeypatch, script, final, ON)
    manifests = list((tmp_path / "runs-root").rglob("provenance.json"))
    assert [json.loads(path.read_text(encoding="ascii"))["status"] for path in manifests] == ["failed"]


# ---------------------------------------------------------------------------
# The record, the scores, the rates, and the table


def final_record(passed: bool, timing: DeviceTiming | None = TIMING) -> FinalTest:
    """Return a SYNTHETIC FinalTest."""
    return FinalTest(
        executor=FINAL_KIND, device=FINAL_DEVICE, stage_reached="S5" if passed else "S4",
        run=RunInfo(exit_code=0 if passed else 1, hang=False, wall_s=2.0, device_timing=timing),
        alignment=Alignment(per_input=[1.0 if passed else 0.0], mean=1.0 if passed else 0.0), passed=passed,
    )


def test_the_record_round_trips_and_an_older_attempt_loads_without_a_final_test() -> None:
    attempt = Attempt(index=0, stage_reached="S5", final_test=final_record(True))
    assert from_dict(Attempt, to_dict(attempt)) == attempt
    older = to_dict(Attempt(index=0, stage_reached="S5"))
    del older["final_test"]
    assert from_dict(Attempt, older).final_test is None
    with pytest.raises(ValueError):
        FinalTest(executor="", device=None, stage_reached="S5", passed=True)


def scored_trial(trial: Trial, final_test: FinalTest | None) -> Trial:
    """Return `trial` with its last attempt's final test replaced."""
    last = dataclasses.replace(trial.attempts[-1], final_test=final_test)
    return dataclasses.replace(trial, attempts=[*trial.attempts[:-1], last])


def test_the_lassi_profile_reads_a_failed_final_test_as_not_correct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, final = Script(replies=[candidate(0)], attempts=[doubled]), FinalScript(runs=[doubled])
    _, trial = run_final(tmp_path, monkeypatch, script, final, ON)
    notes = {key: f"SYNTHETIC {key}" for key in ("baseline", "gap", "compile_only", "not_aligned")}
    assert correct_value(trial, notes) == (1.0, None)
    assert correct_value(scored_trial(trial, final_record(False)), notes) == (0.0, None)


def test_df_v0_scores_an_attempt_with_a_final_test_by_that_run() -> None:
    profile = DfV0Profile()
    plain = Attempt(index=0, stage_reached="S5", alignment=Alignment(per_input=[1.0], mean=1.0))
    passed = dataclasses.replace(plain, final_test=final_record(True))
    failed = dataclasses.replace(plain, final_test=final_record(False))
    assert profile.score_attempt(passed).scalar == profile.score_attempt(plain).scalar
    assert profile.score_attempt(failed).scalar < profile.score_attempt(plain).scalar
    assert profile.score_attempt(failed).components["stage_base"] == profile.weights.stage_base["S4"]


def test_tt_rates_label_each_final_test_with_its_device_and_refuse_a_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, final = Script(replies=[candidate(0)], attempts=[doubled]), FinalScript(runs=[doubled])
    _, trial = run_final(tmp_path, monkeypatch, script, final, ON)
    work = Work(flop=2000, ops={"mac": 1000}, bytes=4000, basis="SYNTHETIC")
    rates = tt_rates.final_test_rates(trial, work)
    assert rates and all(rate.run == f"attempt 0 final test on {FINAL_DEVICE}" for rate in rates)
    assert any(rate.value is not None for rate in rates)
    ttsim = dataclasses_replace_device(trial, "ttsim v1.3.4 (a simulator, not silicon)")
    with pytest.raises(tt_rates.RateRefused):
        tt_rates.final_test_rates(ttsim, work)


def dataclasses_replace_device(trial: Trial, device: str) -> Trial:
    """Return `trial` with its last attempt's final test naming `device`."""
    test = trial.attempts[-1].final_test
    assert test is not None
    return scored_trial(trial, dataclasses.replace(test, device=device))


def load_tabulate() -> Any:
    """Import plans/runs/rngd-sweep/tabulate.py, a script outside the package."""
    spec = importlib.util.spec_from_file_location("tabulate", REPO / "plans" / "runs" / "rngd-sweep" / "tabulate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sweep_tabulator_writes_one_final_test_row_per_final_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script(replies=[candidate(0), candidate(1)], attempts=[doubled, doubled])
    final = FinalScript(runs=[wrong, doubled])
    run_dir, trial = run_final(tmp_path, monkeypatch, script, final, ON)
    runs = tmp_path / "table-runs" / "runs" / "sweep-20261101-0400-syn-maxabs"
    record = runs / "t" / "trial.json"
    record.parent.mkdir(parents=True)
    record.write_bytes((trial_dir(run_dir, trial.trial_id) / "trial.json").read_bytes())
    out = tmp_path / "tables"
    assert load_tabulate().main([str(tmp_path / "table-runs"), str(out)]) == 0
    lines = (out / "final_tests.tsv").read_text(encoding="ascii").splitlines()
    header, *rows = (line.split("\t") for line in lines)
    assert header[6:10] == ["executor", "passed", "stage", "kernel_ns"]
    assert [(row[5], row[7], row[9], row[10]) for row in rows] == [("0", "false", "1000", "1000"),
                                                                  ("1", "true", "1000", "1000")]
    assert all(row[11] == FINAL_DEVICE for row in rows)

