"""Tests for how the lassi and df-v0 profiles read a trial that ended at sim-gap (task P4.6).

Bible: ttsim Facts (UnimplementedFunctionality and UnsupportedFunctionality
are simulator gaps, not model errors; the trial stops with the end reason
sim-gap), Risks And Questions (simulator gaps cause false negatives in the
loop and in rewards; the sim-gap end reason stops the trial), Evaluation Protocol >
LASSI Score Profile (correct and compiled are null for a trial that ended at
the baseline), Training Module > Reward Function (stage table and scoring
decisions), Component Interfaces (ScoreProfile: a component or the scalar is
null when not measured or not computed). Plan: plans/p4-ttsim.md, P4.6, and
the P4 constraint that each new end reason ending a trial at the baseline
joins the lassi profile's baseline-end set and that df-v0's reading of a
sim-gap attempt is stated.

The contract these tests fix:

- lassi.scoring.lassi_profile BASELINE_ENDS holds `sim-gap`, so a trial that
  ended there before any attempt (its reference run hit a gap) scores
  correct, first_try, compiled, compiled_first_try, sim_t, sim_t_c, and
  sim_l null, each noted with the profile's `baseline` note and the code.
- df-v0 reads a sim-gap attempt, the last attempt of a trial whose
  final.end_reason is `sim-gap` (the record's tag of the gap; the trial
  stops right after that attempt's run), as holding no evidence either way:
  the simulator could not run the program, so its R is not computed (null),
  neither the failed run of a stage-S4 attempt (a false negative) nor a
  clean run. Every earlier attempt is scored by its own record as before.
  The trial's multi_turn value and scalar, which read R_final, are null, and
  single_turn is attempt 0's R (null when attempt 0 is the gap attempt).
  apply() leaves final.score null. The same attempts in a trial that did not
  end at sim-gap score as before.
- A kernel JIT failure (S1 with jit-stage errors) and a run with UB (S4 with
  a run-stage error) need no new reading: each scores by its stage.

This reading is the most conservative one: it computes no reward from a run
the simulator could not complete. The records are hand-made SYNTHETIC
Trial and Attempt records; no model is called and no program is run.
Expected values come from the shipped weights file by the bible's formula.
Wall times are PLACEHOLDER fixture values. No value here is a measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.core import record as record_module
from lassi.core.interfaces import Sampling
from lassi.core.record import (
    Alignment,
    Attempt,
    BenchItem,
    Diagnostic,
    EndReason,
    Final,
    Guards,
    ModelInfo,
    Provenance,
    RunInfo,
    TextRef,
    Trial,
    make_trial_id,
)
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.scoring.df_v0 import WEIGHTS_FILE, load_weights

REPO = Path(__file__).resolve().parents[2]
LASSI_PROFILE_FILE = REPO / "assets" / "scoring" / "lassi.yaml"
SIM_GAP = "sim-gap"
SUITE, ITEM, DIRECTION, MODEL_ID = "simfix", "vadd", "cpu-tt", "synthetic-model"
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
# A commit id for synthetic provenance; not a commit of this repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
PLACEHOLDER_WALL_S = 0.5
GAP_EXIT = 1
TOLERANCE = 1e-12
_STDOUT_SHA = hashlib.sha256(b"SYNTHETIC stdout\n").hexdigest()
STDOUT_REF = TextRef(sha256=_STDOUT_SHA, path=f"texts/{_STDOUT_SHA[:2]}/{_STDOUT_SHA}.txt")
TARGET_FILE = "host.cpp"
UNSCORED = ("correct", "first_try", "compiled", "compiled_first_try", "sim_t", "sim_t_c", "sim_l")

UB_ERROR = Diagnostic(
    stage="run", severity="error", code="run-error", message="SYNTHETIC: the simulator reported undefined behavior"
)
JIT_ERROR = Diagnostic(
    stage="jit", severity="error", code="SYNTHETIC-jit-error", file="kernels/compute.cpp", line=12, column=5,
    message="SYNTHETIC kernel JIT error",
)


def require_sim_gap() -> None:
    """Fail the test clearly while END_REASONS lacks sim-gap."""
    if SIM_GAP not in record_module.END_REASONS:
        pytest.fail(f"lassi.core.record.END_REASONS has no {SIM_GAP!r}; task P4.6 adds it")


def df_v0() -> Any:
    """Return df-v0 as a recipe's `score: df-v0` builds it, from the shipped weights file."""
    importlib.import_module("lassi.scoring")
    return DEFAULT_REGISTRY.get("ScoreProfile", "df-v0").factory()


def base(stage: str) -> float:
    """Return b(stage) from the shipped weights file."""
    return load_weights(WEIGHTS_FILE).stage_base[stage]


# ---------------------------------------------------------------------------
# SYNTHETIC records


def ran(exit_code: int, sim_ub: bool = False) -> RunInfo:
    """Return a SYNTHETIC RunInfo of a simulator run that happened and ended with `exit_code`."""
    return RunInfo(
        exit_code=exit_code, hang=False, sim_ub=sim_ub, wall_s=PLACEHOLDER_WALL_S, stdout_ref=STDOUT_REF,
        stdout_truncated=False, stderr_truncated=False, workdir_incomplete=False,
    )


def attempt(index: int, stage: str, diagnostics: Sequence[Diagnostic] = (), run: RunInfo | None = None) -> Attempt:
    """Return a SYNTHETIC attempt at `stage` with one target file."""
    return Attempt(
        index=index, response_text="SYNTHETIC reply", files={TARGET_FILE: "SYNTHETIC code\n"}, stage_reached=stage,
        diagnostics=list(diagnostics), run=run or RunInfo(),
    )


def ub_attempt(index: int) -> Attempt:
    """Return a SYNTHETIC attempt whose simulator run reported UB: a failed run, so S4 with a run error."""
    return attempt(index, "S4", [UB_ERROR], ran(0, sim_ub=True))


def gap_attempt(index: int) -> Attempt:
    """Return a SYNTHETIC attempt whose simulator run stopped at a gap: it compiled, so S4, and did not run clean."""
    return attempt(index, "S4", run=ran(GAP_EXIT))


def trial_of(attempts: Sequence[Attempt], end: str | None, reference: RunInfo | None = None) -> Trial:
    """Return a SYNTHETIC trial of `attempts` ended with the code `end` (None: ended normally)."""
    reason = None if end is None else EndReason(code=end, message=f"SYNTHETIC: the trial ended at {end}")
    return Trial(
        trial_id=make_trial_id("sim-scoring", MODEL_ID, SUITE, DIRECTION, ITEM, 1),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=FAKE_COMMIT, dirty=False, device="SYNTHETIC device", sdk=None,
                              date="2026-09-25T00:00:00+00:00"),
        bench_item=BenchItem(suite=SUITE, item=ITEM, split="eval", direction=DIRECTION),
        model=ModelInfo(backend="scripted", id=MODEL_ID, sampling=SAMPLING),
        reference_run=reference or ran(0),
        requests=[],
        attempts=list(attempts),
        final=Final(
            stage_reached=attempts[-1].stage_reached if attempts else None,
            corrections=max(len(attempts) - 1, 0),
            end_reason=reason,
        ),
    )


# ---------------------------------------------------------------------------
# The lassi profile: sim-gap at the baseline


def test_sim_gap_joins_the_lassi_profiles_baseline_ends() -> None:
    from lassi.scoring.lassi_profile import BASELINE_ENDS

    assert SIM_GAP in BASELINE_ENDS, "a reference run that hit a gap ends the trial at the baseline"


def test_the_lassi_profile_leaves_correct_and_compiled_null_when_the_baseline_hit_a_gap(tmp_path: Path) -> None:
    from lassi.scoring.lassi_profile import LassiProfile

    require_sim_gap()
    trial = trial_of([], SIM_GAP, reference=ran(GAP_EXIT))
    (tmp_path / "bench").mkdir()
    score = LassiProfile(bench_root=tmp_path / "bench").score(trial)
    found = {name: score.components[name] for name in UNSCORED}
    assert found == dict.fromkeys(UNSCORED), f"a trial that ended at the baseline scores these null: {found}"
    assert score.scalar is None
    baseline_note = yaml.safe_load(LASSI_PROFILE_FILE.read_text(encoding="ascii"))["notes"]["baseline"]
    for name in UNSCORED:
        assert score.notes[name].startswith(baseline_note) and SIM_GAP in score.notes[name], (name, score.notes)


# ---------------------------------------------------------------------------
# df-v0: the reading of a sim-gap attempt


def test_df_v0_gives_a_sim_gap_attempt_no_reward_and_the_trial_no_score() -> None:
    require_sim_gap()
    trial = trial_of([gap_attempt(0)], SIM_GAP)
    profile = df_v0()
    (only,) = profile.score_attempts(trial)
    assert only.scalar is None, f"a simulator gap is not a model error, so its R is not computed; got {only.scalar!r}"
    score = profile.score(trial)
    assert (score.components["single_turn"], score.components["multi_turn"], score.scalar) == (None, None, None)


def test_df_v0_scores_the_attempts_before_the_gap_by_their_own_record() -> None:
    require_sim_gap()
    trial = trial_of([ub_attempt(0), gap_attempt(1)], SIM_GAP)
    profile = df_v0()
    first, gap = profile.score_attempts(trial)
    assert first.scalar == pytest.approx(base("S4"), abs=TOLERANCE), "the UB attempt is a failed run at S4"
    assert gap.scalar is None, "the gap attempt's R is not computed"
    score = profile.score(trial)
    assert score.components["single_turn"] == pytest.approx(first.scalar, abs=TOLERANCE), "attempt 0's R"
    assert score.components["multi_turn"] is None and score.scalar is None, "R_final is not computed"


def test_df_v0_apply_writes_no_final_score_at_sim_gap() -> None:
    require_sim_gap()
    applied = df_v0().apply(trial_of([ub_attempt(0), gap_attempt(1)], SIM_GAP))
    assert applied.final.score is None, f"final.score is the multi-turn value, not computed: {applied.final.score!r}"
    assert applied.attempts[1].score.scalar is None, "the gap attempt's Attempt.score holds no R"
    assert applied.attempts[0].score.scalar == pytest.approx(base("S4"), abs=TOLERANCE)


def test_df_v0_scores_a_sim_gap_at_the_baseline_as_no_score() -> None:
    require_sim_gap()
    score = df_v0().score(trial_of([], SIM_GAP, reference=ran(GAP_EXIT)))
    assert (score.components["single_turn"], score.components["multi_turn"], score.scalar) == (None, None, None)


# ---------------------------------------------------------------------------
# Readings that do not change (they pass before the task, and must after it)


def test_without_the_sim_gap_end_the_same_attempts_score_as_before() -> None:
    trial = trial_of([ub_attempt(0), gap_attempt(1)], None)
    profile = df_v0()
    scores = profile.score_attempts(trial)
    assert [score.scalar for score in scores] == pytest.approx([base("S4"), base("S4")], abs=TOLERANCE)
    penalty = load_weights(WEIGHTS_FILE).correction_penalty
    assert profile.score(trial).scalar == pytest.approx(base("S4") - penalty * 1, abs=TOLERANCE)


def test_df_v0_reads_a_kernel_jit_failure_at_s1_and_undefined_behavior_at_s4() -> None:
    trial = trial_of([attempt(0, "S1", [JIT_ERROR]), ub_attempt(1)], None)
    scores = df_v0().score_attempts(trial)
    assert [score.scalar for score in scores] == pytest.approx([base("S1"), base("S4")], abs=TOLERANCE)


# ---------------------------------------------------------------------------
# The lassi profile's clean run, and its reading of a trial that ended at sim-gap after an attempt (the P4.6 audit)


def lassi_outcome(trial: Trial) -> tuple[dict[str, float | None], dict[str, str], dict[str, str]]:
    """Return the lassi profile's outcome values and notes for `trial`, and the profile file's note texts."""
    from lassi.scoring.lassi_profile import PROFILE_FILE, load_profile, outcome_components

    notes = dict(load_profile(PROFILE_FILE).notes)
    values, written = outcome_components(trial, notes)
    return values, written, notes


def test_the_lassi_profile_reads_a_standing_run_with_undefined_behavior_at_exit_zero_as_not_clean() -> None:
    values, written, _ = lassi_outcome(trial_of([ub_attempt(0)], "correction-cap"))
    assert values["correct"] == 0.0 and "correct" not in written, (values, written)


def test_the_lassi_profile_reads_a_standing_kernel_jit_failure_as_not_clean_whatever_its_exit_status() -> None:
    """A jit error with exit 0 leaves the attempt at S1; its standing run is never read as clean (P4.6 recheck)."""
    values, written, _ = lassi_outcome(trial_of([attempt(0, "S1", [JIT_ERROR], run=ran(0))], "correction-cap"))
    assert values["correct"] == 0.0 and "correct" not in written, (values, written)


def test_the_lassi_profile_reads_stale_output_with_undefined_behavior_aligned_at_one_as_not_correct() -> None:
    aligned = dataclasses.replace(ub_attempt(0), alignment=Alignment(per_input=[1.0], mean=1.0))
    stale = Diagnostic(stage="run", severity="warning", code="stale-output", message="SYNTHETIC: stale output stands")
    trial = trial_of([aligned, attempt(1, "S4", [stale])], None)
    values, _, _ = lassi_outcome(trial)
    assert values["correct"] == 0.0, "past the execution gate, a UB run's output stands but was not a clean run"


@pytest.mark.parametrize("sim_ub", [False, True], ids=["no-ub", "ub"])
@pytest.mark.parametrize("exit_code", [0, GAP_EXIT], ids=["exit-0", "exit-1"])
def test_the_lassi_profile_leaves_correct_null_with_the_gap_note_when_an_attempt_hit_a_gap(
    exit_code: int, sim_ub: bool
) -> None:
    trial = trial_of([ub_attempt(0), attempt(1, "S4", run=ran(exit_code, sim_ub=sim_ub))], SIM_GAP)
    values, written, notes = lassi_outcome(trial)
    assert (values["correct"], values["first_try"]) == (None, None), values
    assert written["correct"] == written["first_try"] == notes["gap"], written
    assert values["compiled"] == 1.0, "the gap attempt compiled"


# ---------------------------------------------------------------------------
# df-v0: a guard violation on the gap attempt, the notes, and undefined behavior in the gap run (the P4.6 audit)


def test_df_v0_a_guard_violation_recorded_on_the_gap_attempt_still_sets_r() -> None:
    violated = dataclasses.replace(gap_attempt(1), guards=Guards(host_compute=True))
    trial = trial_of([ub_attempt(0), violated], SIM_GAP)
    weights = load_weights(WEIGHTS_FILE)
    profile = df_v0()
    assert profile.score_attempts(trial)[1].scalar == pytest.approx(weights.guard_violation, abs=TOLERANCE)
    score = profile.score(trial)
    expected = weights.guard_violation - weights.correction_penalty * 1
    assert score.scalar == pytest.approx(expected, abs=TOLERANCE) and score.notes == {}, score


def test_df_v0_notes_each_trial_component_the_gap_leaves_null() -> None:
    from lassi.scoring.df_v0 import GAP_NOTES

    alone = df_v0().score(trial_of([gap_attempt(0)], SIM_GAP))
    assert dict(alone.notes) == {"single_turn": GAP_NOTES["single_turn"], "multi_turn": GAP_NOTES["multi_turn"]}
    after = df_v0().score(trial_of([ub_attempt(0), gap_attempt(1)], SIM_GAP))
    assert dict(after.notes) == {"multi_turn": GAP_NOTES["multi_turn"]}


def test_df_v0_a_gap_attempt_whose_run_also_reported_undefined_behavior_has_no_r() -> None:
    (only,) = df_v0().score_attempts(trial_of([attempt(0, "S4", run=ran(GAP_EXIT, sim_ub=True))], SIM_GAP))
    assert only.scalar is None, "the gap reading wins over undefined behavior"


def test_df_v0_score_attempt_reads_the_attempt_alone() -> None:
    assert df_v0().score_attempt(gap_attempt(0)).scalar == pytest.approx(base("S4"), abs=TOLERANCE), (
        "score_attempt does not see the trial's end reason; score_attempts applies the sim-gap reading"
    )
