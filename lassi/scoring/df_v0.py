"""The df-v0 score profile: the Reward Function's score of each attempt and of the trial.

Bible: Training Module, Reward Function (stage table, formula, scoring
decisions) and Algorithms (Episodes); Component Interfaces (ScoreProfile);
Result Record (Attempt.score, Attempt.guards, final.score, Diagnostic).

Every weight is read from a YAML weights file, by default
assets/scoring/df-v0.yaml. The code holds none: a file missing a weight is
refused with a ValueError naming the key, since a fallback here would be a
weight held in code. With b(s) the base score of the stage reached, W the
warning count, and A the alignment mean, an attempt scores

    R = b(s) - warning_weight x min(W, warning_cap) + alignment_weight x A x 1[s = S5]

and R = guard_violation when a guard is violated.

Readings of the Result Record (task P2.6):

- W counts an attempt's compile- and jit-stage warnings, each unique by
  (code, file, line, column), when the attempt reached S4 or S5 (a
  successful compile). Parse- and run-stage warnings (fence-quirk,
  stale-output, the truncation flags) are pipeline notes, not code warnings,
  and an attempt below S4 has W = 0.
- A is Attempt.alignment.mean, counted only at S5. An S5 attempt that was
  never aligned (mean None, as when the reference stdout was cut) holds no
  evidence of correct output: its alignment term is 0.0 and its component
  alignment_missing is 1.0 (0.0 on every other attempt).
- The guard component is 1.0 when any Guards field is True, 0.0 when every
  field is False, and None otherwise: a None field was not checked.
- The trial's single_turn component is attempt 0's R. Its multi_turn
  component, which is also its scalar and final.score, is R_final minus
  correction_penalty x final.corrections, where R_final is the R of the
  last attempt, a guard violation or a stale-output attempt included: each
  attempt is scored by its own record (a sim-gap attempt's R is None,
  below). A trial with no attempts has both None.

The reading of a sim-gap attempt (task P4.6): the gap attempt is the last
attempt of a trial whose final.end_reason is sim-gap (ended_at_gap), the
attempt whose run stopped at a simulator gap. A gap is not a model error,
and the simulator could not complete the run, so the run holds no evidence
either way: its R is None (not computed), neither a failed S4 run nor a
clean one, even when its run also reported undefined behavior (the gap
reading wins, as run_loop ends such a trial at sim-gap). Its components are
those of its own record, as for any attempt, and a guard violation recorded
on it (Attempt.guards; df-v0 reads the recorded guards, not the run) still
sets R = guard_violation. Every earlier attempt is scored by its own record.
single_turn is attempt 0's R, and multi_turn, the scalar, and final.score
are None when R_final is; a trial Score component left None this way
carries a note (GAP_NOTES). A trial that ended at sim-gap with no attempt
has no score, as any trial with no attempts. A kernel JIT failure (S1) and
a run with undefined behavior (S4 with a run-error) need no reading of
their own: each scores by its stage.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from lassi.core.capabilities import SCORES_ATTEMPTS
from lassi.core.interfaces import Score
from lassi.core.record import STAGES, Attempt, Guards, ScoreBreakdown, Trial
from lassi.core.registry import register

# The weights file a recipe's `score: df-v0` reads.
WEIGHTS_FILE = Path(__file__).resolve().parents[2] / "assets" / "scoring" / "df-v0.yaml"
# Warnings count from this stage on (a successful compile); alignment counts only at CLEAN_RUN (Reward Function).
COMPILED = "S4"
CLEAN_RUN = "S5"
# The Diagnostic stages whose warnings are code warnings: the host build and the kernel JIT.
WARNING_STAGES = frozenset({"compile", "jit"})
# The weight keys besides stage_base, in file order; each holds one number.
NUMBER_KEYS = ("warning_weight", "warning_cap", "alignment_weight", "guard_violation", "correction_penalty")

# Component names of an attempt's Score.
STAGE_BASE = "stage_base"
WARNING_COUNT = "warning_count"
ALIGNMENT_TERM = "alignment_term"
GUARD = "guard"
ALIGNMENT_MISSING = "alignment_missing"
# Component names of the trial's Score.
SINGLE_TURN = "single_turn"
MULTI_TURN = "multi_turn"
# The end reason of a trial stopped by a simulator gap (lassi.core.record END_REASONS; task P4.6). The last attempt of
# such a trial is its gap attempt, the one whose run stopped at the gap.
SIM_GAP_END = "sim-gap"
# The notes on the trial Score's components that the gap attempt leaves null.
GAP_NOTES = {
    SINGLE_TURN: "not computed: attempt 0 is the sim-gap attempt, whose run stopped at a simulator gap, not a model "
    "error, so its R is not computed",
    MULTI_TURN: "not computed: the last attempt is the sim-gap attempt, whose run stopped at a simulator gap, not a "
    "model error, so R_final is not computed",
}


@dataclass(frozen=True)
class Weights:
    """The df-v0 weights as one weights file gives them (see load_weights)."""

    stage_base: Mapping[str, float]
    warning_weight: float
    warning_cap: int
    alignment_weight: float
    guard_violation: float
    correction_penalty: float


def _check_keys(data: Mapping[Any, Any], expected: Sequence[str], where: str) -> None:
    """Raise ValueError naming each expected key `data` lacks, or each key it holds that is not expected."""
    missing = [key for key in expected if key not in data]
    if missing:
        raise ValueError(f"{where}: missing weight {', '.join(missing)}; df-v0 has no default for any weight")
    unknown = sorted(str(key) for key in data if key not in expected)
    if unknown:
        raise ValueError(f"{where}: unknown key {', '.join(unknown)}; expected {', '.join(expected)}")


def _number(value: Any, where: str) -> float:
    """Return `value` as a float; raise ValueError naming `where` unless it is a finite int or float (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where}: expected a finite number, got {value!r}")
    return float(value)


def load_weights(path: str | Path) -> Weights:
    """Return the weights in the YAML file at `path`.

    Raises ValueError naming the file and the key for a missing file, a
    missing or unknown key, a value that is not a finite number, or a
    warning_cap that is not a whole number of at least 0.
    """
    where = Path(path).as_posix()
    if not Path(path).is_file():
        raise ValueError(f"the df-v0 weights file {where} does not exist")
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping of weights, got {data!r}")
    _check_keys(data, (STAGE_BASE, *NUMBER_KEYS), where)
    table = data[STAGE_BASE]
    if not isinstance(table, dict):
        raise ValueError(f"{where}: stage_base must map each stage to its base score, got {table!r}")
    _check_keys(table, STAGES, f"{where} stage_base")
    base = {stage: _number(table[stage], f"{where} stage_base.{stage}") for stage in STAGES}
    numbers = {key: _number(data[key], f"{where} {key}") for key in NUMBER_KEYS}
    cap = data["warning_cap"]
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise ValueError(f"{where} warning_cap: expected a whole number of warnings, at least 0, got {cap!r}")
    numbers["warning_cap"] = cap
    return Weights(stage_base=MappingProxyType(base), **numbers)


def warning_count(attempt: Attempt) -> int:
    """Return W: the attempt's compile- and jit-stage warnings unique by (code, file, line, column); 0 below S4."""
    if STAGES.index(attempt.stage_reached) < STAGES.index(COMPILED):
        return 0
    places = {
        (item.code, item.file, item.line, item.column)
        for item in attempt.diagnostics
        if item.severity == "warning" and item.stage in WARNING_STAGES
    }
    return len(places)


def guard_state(guards: Guards) -> float | None:
    """Return 1.0 when any guard is True, 0.0 when every guard is False, and None otherwise (not all checked)."""
    outcomes = [getattr(guards, item.name) for item in dataclasses.fields(guards)]
    if any(outcome is True for outcome in outcomes):
        return 1.0
    if all(outcome is False for outcome in outcomes):
        return 0.0
    return None


def ended_at_gap(trial: Trial) -> bool:
    """Return True when `trial` ended at sim-gap after an attempt: its last attempt is the gap attempt."""
    end = trial.final.end_reason
    return bool(trial.attempts) and end is not None and end.code == SIM_GAP_END


@register("ScoreProfile", "df-v0")
class DfV0Profile:
    """The df-v0 ScoreProfile: R per attempt, and the trial's single-turn and multi-turn values.

    Built with no arguments (a recipe's `score: df-v0`) it reads WEIGHTS_FILE;
    built as `DfV0Profile(weights_path=<path>)` it reads that file. The file is
    read and checked when the profile is built.
    """

    name = "df-v0"
    capabilities = frozenset({SCORES_ATTEMPTS})
    # The components of the trial's Score, in order; `lassi run` offers them as metrics (lassi.scoring.run_scoring).
    trial_components = (SINGLE_TURN, MULTI_TURN)

    def __init__(self, *, weights_path: str | Path | None = None) -> None:
        """Read and check the weights file: `weights_path`, or WEIGHTS_FILE when it is None."""
        self.weights_path = WEIGHTS_FILE if weights_path is None else Path(weights_path)
        self.weights = load_weights(self.weights_path)

    def score_attempt(self, attempt: Attempt) -> Score:
        """Return one attempt's Score from its record alone: its components and, as the scalar, its R.

        It reads the attempt without its trial, so it does not apply the
        sim-gap reading, which needs the trial's end reason: score_attempts
        does, and a caller scoring the attempts of a trial uses it.
        """
        weights = self.weights
        stage = attempt.stage_reached
        count = warning_count(attempt)
        mean = attempt.alignment.mean
        clean_run = stage == CLEAN_RUN
        term = weights.alignment_weight * mean if clean_run and mean is not None else 0.0
        guard = guard_state(attempt.guards)
        if guard == 1.0:
            r = weights.guard_violation
        else:
            r = weights.stage_base[stage] - weights.warning_weight * min(count, weights.warning_cap) + term
        components = {
            STAGE_BASE: weights.stage_base[stage],
            WARNING_COUNT: float(count),
            ALIGNMENT_TERM: term,
            GUARD: guard,
            ALIGNMENT_MISSING: 1.0 if clean_run and mean is None else 0.0,
        }
        return Score(components=components, scalar=r)

    def score_attempts(self, trial: Trial) -> list[Score]:
        """Return one Score per attempt of `trial`, in attempt order.

        Each attempt is scored by its own record (score_attempt), except
        the gap attempt of a trial that ended at sim-gap (ended_at_gap): it
        keeps its components, but its R, the scalar, is None (not computed),
        undefined behavior in its run included, unless a guard violation is
        recorded on it (component guard 1.0), which still sets
        R = guard_violation.
        """
        scores = [self.score_attempt(attempt) for attempt in trial.attempts]
        if scores and ended_at_gap(trial) and scores[-1].components[GUARD] != 1.0:
            scores[-1] = Score(components=scores[-1].components, scalar=None)
        return scores

    def score(self, trial: Trial) -> Score:
        """Return the trial's Score: single_turn and multi_turn, with multi_turn as the scalar."""
        return self._trial_score(trial, self.score_attempts(trial))

    def apply(self, trial: Trial) -> Trial:
        """Return a new Trial whose Attempt.score fields and final.score hold this profile's scores.

        Each Attempt.score holds its attempt's components and R; final.score
        is the multi-turn value. Nothing else changes, and `trial` itself is
        left as it was.
        """
        scores = self.score_attempts(trial)
        attempts = [
            dataclasses.replace(attempt, score=ScoreBreakdown(components=dict(score.components), scalar=score.scalar))
            for attempt, score in zip(trial.attempts, scores, strict=True)
        ]
        final = dataclasses.replace(trial.final, score=self._trial_score(trial, scores).scalar)
        return dataclasses.replace(trial, attempts=attempts, final=final)

    def _trial_score(self, trial: Trial, scores: Sequence[Score]) -> Score:
        """Return the trial's Score from its attempts' Scores; every value is None when there is no attempt.

        single_turn is attempt 0's R and multi_turn is R_final minus the
        correction penalty, each None when the R it reads is None (the gap
        attempt's), with that component's GAP_NOTES note.
        """
        if not scores:
            return Score(components={SINGLE_TURN: None, MULTI_TURN: None}, scalar=None)
        last = scores[-1].scalar
        multi = None if last is None else last - self.weights.correction_penalty * trial.final.corrections
        components = {SINGLE_TURN: scores[0].scalar, MULTI_TURN: multi}
        notes = {name: GAP_NOTES[name] for name, value in components.items() if value is None}
        return Score(components=components, scalar=multi, notes=notes)
