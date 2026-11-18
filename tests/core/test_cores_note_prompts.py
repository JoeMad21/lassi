"""The ttsim executor's core count never reaches a prompt (task P17.70).

An executor under count_cores adds a run-stage note coded
lassi.core.capabilities CORES_CODE to a clean run. These run_loop tests, on
the scripted stages of tests/core/test_sim_readings.py, fix that the note
lands in the attempt's diagnostics (and so in the trial record) and in no
request's prompt: after a clean run that mismatched, and after a run that
exited 0 with a kernel JIT error (stage S1), whose diagnostics feed the
compile correction. Nothing is run; every value is SYNTHETIC.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from test_sim_readings import (
    FIXED_REPLY,
    JIT_ERROR,
    REPLY,
    TARGET,
    Script,
    clean_sim_reference,
    clean_sim_run,
    recipe_data,
    run_one,
    sim_run,
)

from lassi.core.capabilities import CORES_CODE
from lassi.core.record import Diagnostic

NOTE = Diagnostic(stage="run", severity="note", code=CORES_CODE,
                  message="cores used: 1 of 80 worker cores ran a kernel (Watcher dump #2); a count, not a time")


def test_a_note_on_a_run_with_a_jit_error_stays_out_of_the_compile_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jit_exit0 = sim_run(exit_code=0, stdout="", sim_ub=False, diagnostics=[JIT_ERROR, NOTE])
    script = Script(replies=[REPLY, FIXED_REPLY], references={TARGET: clean_sim_reference()},
                    attempt_runs=[jit_exit0, clean_sim_run()])
    outcome = run_one(tmp_path, monkeypatch, script, recipe_data())
    assert [attempt.stage_reached for attempt in outcome.trial.attempts][0] == "S1"
    assert NOTE in outcome.trial.attempts[0].diagnostics, "the note stays in the trial record"
    assert NOTE.message not in outcome.prompt(1), "the compile correction never names the cores"


def test_the_target_references_count_is_kept_in_the_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = clean_sim_reference()
    counted = dataclasses.replace(reference, diagnostics=[*reference.diagnostics, NOTE])
    script = Script(replies=[REPLY], references={TARGET: counted}, attempt_runs=[clean_sim_run()])
    outcome = run_one(tmp_path, monkeypatch, script, recipe_data())
    assert NOTE in outcome.trial.baseline_diagnostics, "the reference's count, beside each translation's"
    assert all(NOTE.message not in outcome.prompt(index) for index in range(len(outcome.trial.attempts)))


def test_a_note_on_a_clean_run_is_recorded_and_prompts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = clean_sim_run()
    counted = sim_run(exit_code=0, stdout="", sim_ub=False, diagnostics=[*clean.diagnostics, NOTE])
    script = Script(replies=[REPLY], references={TARGET: clean_sim_reference()}, attempt_runs=[counted])
    outcome = run_one(tmp_path, monkeypatch, script, recipe_data())
    assert NOTE in outcome.trial.attempts[-1].diagnostics
    assert all(NOTE.message not in outcome.prompt(index) for index in range(len(outcome.trial.attempts)))
