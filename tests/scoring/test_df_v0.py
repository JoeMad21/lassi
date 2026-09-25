"""Tests for the df-v0 score profile (task P2.6).

Bible: Training Module, Reward Function (the stage table, the formula, and
the scoring decisions); Training Module, Algorithms (Episodes: single_turn
rewards the first attempt, multi_turn makes the correction loop the
episode); Component Interfaces (ScoreProfile: Trial -> components + scalar);
Result Record (Attempt.score, Attempt.guards, final.score, Diagnostic).

The Reward Function's formula, with b(s) the base score of stage s from its
stage table, W the unique warning count, and A the alignment:

    R = b(s) - 0.02 min(W, 10) + 0.8 A 1[s = S5]

A guard violation (host compute, harness tamper, oracle access) sets R = -1,
and multi-turn episodes use R_final - 0.05 x corrections.

The contract these tests fix, from the P2.6 acceptance criteria and the P2
the planning decisions on W and on final.score (plans/p2-scoring.md):

- `assets/scoring/df-v0.yaml` holds every weight the Reward Function states,
  under the keys of BIBLE_WEIGHTS below: `stage_base` (S0 to S5),
  `warning_weight`, `warning_cap`, `alignment_weight`, `guard_violation` (the
  R a violation sets), and `correction_penalty`. The code holds none: a
  changed weight in the file changes the score by the formula, and a file
  missing a weight is refused with a ValueError naming the key, since a
  fallback in code would be a weight held in code.
- Importing `lassi.scoring` registers the ScoreProfile `df-v0`. Built as
  `factory()` (a recipe's `score: df-v0`) it reads the shipped file; built
  as `factory(weights_path=<path>)` it reads the file at that path.
- `score_attempts(trial)` returns one Score per attempt, in order. Its scalar
  is the attempt's R, and its components are `stage_base` (b(s)),
  `warning_count` (W before the cap), `alignment_term` (0.8 A at S5, else
  0.0), and `guard` (1.0 when any guard is True, 0.0 when all three were
  checked and are False, and None otherwise, since a guard that is None was
  not checked), and `alignment_missing` (1.0 at an S5 attempt never aligned,
  else 0.0).
- W counts the compile- and jit-stage warnings of an attempt at S4 or S5,
  each unique by (code, file, line, column) (the planning decision on W). Parse- and
  run-stage warnings (fence-quirk, stale-output, stdout-truncated), notes,
  and errors are not counted, and an attempt below S4 has W = 0.
- `score(trial)` returns the trial's Score: component `single_turn` is
  attempt 0's R, component `multi_turn` is the last attempt's R minus
  0.05 x final.corrections, and the scalar is the multi-turn value
  (the planning decision on final.score). The last attempt is the final one, as final.stage_reached
  reads it; a stale-output last attempt did not run, so its own record
  scores it, never the output that stands.
- `apply(trial)` returns the trial with each Attempt.score holding its
  attempt's components and R, and final.score the multi-turn value; nothing
  else changes.

Every expected value below is hand-computed from the bible's formula and
stage table, as its comment shows. The records are hand-made SYNTHETIC
Trial and Attempt records; no model is called and no program is run. No
value in this module is a measurement.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.core.interfaces import Sampling
from lassi.core.record import (
    STAGES,
    Alignment,
    Attempt,
    BenchItem,
    Diagnostic,
    Final,
    Guards,
    ModelInfo,
    Provenance,
    RunInfo,
    ScoreBreakdown,
    TextRef,
    Trial,
    make_trial_id,
)
from lassi.core.registry import DEFAULT_REGISTRY, RegistryError

REPO = Path(__file__).resolve().parents[2]
WEIGHTS_FILE = REPO / "assets" / "scoring" / "df-v0.yaml"
PACKAGE = "lassi.scoring"
PROFILE = "df-v0"
TOLERANCE = 1e-12

# The weights the bible's Reward Function states (stage table and formula), under the YAML keys this task fixes.
BIBLE_WEIGHTS: dict[str, Any] = {
    "stage_base": {"S0": -1.0, "S1": -0.8, "S2": -0.6, "S3": -0.4, "S4": 0.0, "S5": 0.2},
    "warning_weight": 0.02,
    "warning_cap": 10,
    "alignment_weight": 0.8,
    "guard_violation": -1.0,
    "correction_penalty": 0.05,
}
TOP_KEYS = tuple(BIBLE_WEIGHTS)
# Every weight as a key path into the YAML mapping.
WEIGHT_PATHS: tuple[tuple[str, ...], ...] = (
    *(("stage_base", stage) for stage in STAGES),
    *((key,) for key in TOP_KEYS if key != "stage_base"),
)

# Component names.
STAGE_BASE = "stage_base"
WARNING_COUNT = "warning_count"
ALIGNMENT_TERM = "alignment_term"
GUARD = "guard"
SINGLE_TURN = "single_turn"
MULTI_TURN = "multi_turn"
GUARD_NAMES = ("host_compute", "harness_tamper", "oracle_access")

# SYNTHETIC record values. The wall time is a PLACEHOLDER fixture value, never a measurement.
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
SUITE, ITEM, DIRECTION, MODEL_ID = "synthetic-suite", "probe", "omp-cuda", "synthetic-model"
PLACEHOLDER_WALL_S = 0.5
RUN_ERROR_EXIT = 3
_STDOUT_SHA = hashlib.sha256(b"SYNTHETIC stdout\n").hexdigest()
STDOUT_REF = TextRef(sha256=_STDOUT_SHA, path=f"text/{_STDOUT_SHA[:2]}/{_STDOUT_SHA}")
SOURCE_FILE = "main.cpp"

COMPILE_ERROR = Diagnostic(
    stage="compile", severity="error", code="synthetic-error", file=SOURCE_FILE, line=1, column=1,
    message="SYNTHETIC compile error",
)
FENCE_QUIRK = Diagnostic(stage="parse", severity="warning", code="fence-quirk", message="SYNTHETIC fence quirk")
STDOUT_TRUNCATED = Diagnostic(stage="run", severity="warning", code="stdout-truncated", message="SYNTHETIC stdout cut")
RUN_ERROR = Diagnostic(stage="run", severity="error", code="run-error", message="SYNTHETIC run error")
STALE_OUTPUT = Diagnostic(
    stage="run", severity="warning", code="stale-output",
    message="SYNTHETIC: the stdout of attempt 0 stands as the trial's output",
)


# ---------------------------------------------------------------------------
# The profile and its weights file, looked up so that a missing one fails each test with a clear message


def profile_class() -> type:
    """Return the class registered as ScoreProfile df-v0; fail the test clearly while there is none."""
    importlib.import_module(PACKAGE)
    try:
        return DEFAULT_REGISTRY.get("ScoreProfile", PROFILE).factory
    except RegistryError as error:
        pytest.fail(f"importing {PACKAGE} registers no ScoreProfile {PROFILE!r} (task P2.6 adds it): {error}")


def default_profile() -> Any:
    """Return df-v0 built as a recipe's `score: df-v0` builds it: with no arguments, from the shipped file."""
    return profile_class()()


def shipped_weights() -> dict[str, Any]:
    """Return the shipped weights file as a mapping; fail the test clearly while it does not exist."""
    if not WEIGHTS_FILE.is_file():
        pytest.fail(f"{WEIGHTS_FILE.relative_to(REPO).as_posix()} does not exist; task P2.6 adds it")
    data = yaml.safe_load(WEIGHTS_FILE.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"the weights file must hold a mapping, got {data!r}"
    return data


def profile_from(tmp_path: Path, weights: Mapping[str, Any]) -> Any:
    """Write `weights` as a YAML file under `tmp_path` and return df-v0 built to read it."""
    path = tmp_path / "df-v0.yaml"
    path.write_text(yaml.safe_dump(dict(weights), sort_keys=False), encoding="utf-8")
    return profile_class()(weights_path=path)


def attempt_scores(profile: Any, trial: Trial) -> list[Any]:
    """Return the profile's per-attempt Scores for `trial`, one per attempt."""
    scores = list(profile.score_attempts(trial))
    assert len(scores) == len(trial.attempts), f"one Score per attempt, got {len(scores)} for {len(trial.attempts)}"
    return scores


def assert_components(actual: Mapping[str, Any], expected: Mapping[str, float | None]) -> None:
    """Assert that `actual` holds every expected component with its value (None where None is expected)."""
    missing = sorted(set(expected) - set(actual))
    assert not missing, f"components {missing} are missing; got {dict(actual)!r}"
    for name, value in expected.items():
        if value is None:
            assert actual[name] is None, f"component {name}: expected None, got {actual[name]!r}"
        else:
            assert actual[name] == pytest.approx(value, abs=TOLERANCE), f"component {name}: expected {value}"


def assert_attempt(
    score: Any, r: float, stage_base: float, warning_count: float, alignment_term: float, guard: float | None
) -> None:
    """Assert one attempt's R (the scalar) and its four components."""
    assert score.scalar == pytest.approx(r, abs=TOLERANCE), f"R: expected {r}, got {score.scalar!r}"
    expected = {STAGE_BASE: stage_base, WARNING_COUNT: warning_count, ALIGNMENT_TERM: alignment_term, GUARD: guard}
    assert_components(score.components, expected)


# ---------------------------------------------------------------------------
# Hand-made SYNTHETIC records


def warning(
    line: int, column: int = 1, *, code: str = "synthetic-warning", stage: str = "compile",
    file: str = SOURCE_FILE, message: str = "SYNTHETIC warning",
) -> Diagnostic:
    """Return a SYNTHETIC warning Diagnostic at (file, line, column)."""
    return Diagnostic(
        stage=stage, severity="warning", code=code, file=file, line=line, column=column, message=message
    )


def unique_warnings(count: int) -> list[Diagnostic]:
    """Return `count` compile warnings, each at its own line, so each is unique."""
    return [warning(line) for line in range(1, count + 1)]


def twice(diagnostics: Sequence[Diagnostic]) -> list[Diagnostic]:
    """Return each diagnostic followed by a copy at the same place with another message (a duplicate)."""
    out: list[Diagnostic] = []
    for diagnostic in diagnostics:
        out += [diagnostic, dataclasses.replace(diagnostic, message=f"{diagnostic.message} (again)")]
    return out


def ran(exit_code: int) -> RunInfo:
    """Return a SYNTHETIC RunInfo of a run that happened and ended with `exit_code`."""
    return RunInfo(
        exit_code=exit_code, hang=False, wall_s=PLACEHOLDER_WALL_S, stdout_ref=STDOUT_REF,
        stdout_truncated=False, stderr_truncated=False, workdir_incomplete=False,
    )


def aligned(value: float | None) -> Alignment:
    """Return the Alignment of one input with `value` (unset for None)."""
    return Alignment() if value is None else Alignment(per_input=[value], mean=value)


def attempt(
    index: int, stage: str, *, diagnostics: Sequence[Diagnostic] = (), run: RunInfo | None = None,
    alignment: float | None = None, guards: Guards | None = None,
) -> Attempt:
    """Return a SYNTHETIC attempt at `stage`; at S0 no file was extracted."""
    files = {} if stage == "S0" else {SOURCE_FILE: "SYNTHETIC code\n"}
    return Attempt(
        index=index, response_text="SYNTHETIC reply", files=files, stage_reached=stage,
        diagnostics=list(diagnostics), run=run or RunInfo(), alignment=aligned(alignment), guards=guards or Guards(),
    )


def clean(index: int, alignment: float, **fields: Any) -> Attempt:
    """Return a SYNTHETIC attempt that compiled and ran clean (S5), aligned at `alignment`."""
    return attempt(index, "S5", run=ran(0), alignment=alignment, **fields)


def run_failed(index: int, alignment: float, *, diagnostics: Sequence[Diagnostic] = (), **fields: Any) -> Attempt:
    """Return a SYNTHETIC attempt that compiled and whose run failed (S4 with a run error), aligned at `alignment`."""
    run = ran(RUN_ERROR_EXIT)
    return attempt(index, "S4", run=run, alignment=alignment, diagnostics=[*diagnostics, RUN_ERROR], **fields)


def compile_failed(index: int, *, diagnostics: Sequence[Diagnostic] = ()) -> Attempt:
    """Return a SYNTHETIC attempt that parsed (S1) and did not compile."""
    return attempt(index, "S1", diagnostics=[*diagnostics, COMPILE_ERROR])


def hand_trial(attempts: Sequence[Attempt], *, final_alignment: float | None = None) -> Trial:
    """Return a SYNTHETIC trial holding `attempts`, its final fields as the runner sets them."""
    last = attempts[-1].stage_reached if attempts else None
    final = Final(stage_reached=last, alignment=final_alignment, corrections=max(len(attempts) - 1, 0))
    return Trial(
        trial_id=make_trial_id("df-v0-test", MODEL_ID, SUITE, DIRECTION, ITEM, 1),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=None, dirty=None, device=None, sdk=None, date="2026-09-24T00:00:00+00:00"),
        bench_item=BenchItem(suite=SUITE, item=ITEM, split="synthetic", direction=DIRECTION),
        model=ModelInfo(backend="scripted", id=MODEL_ID, sampling=SAMPLING),
        attempts=list(attempts),
        final=final,
    )


def one_attempt(item: Attempt) -> Trial:
    """Return a SYNTHETIC trial whose one attempt is `item`."""
    return hand_trial([item])


# ---------------------------------------------------------------------------
# The weights file


def test_the_weights_file_holds_every_weight_of_the_reward_function() -> None:
    data = shipped_weights()
    missing = [key for key in TOP_KEYS if key not in data]
    assert not missing, f"{WEIGHTS_FILE.name} lacks the weight keys {missing}"
    assert set(data["stage_base"]) == set(STAGES), "stage_base holds one base score per stage, S0 to S5"
    for key in TOP_KEYS:
        assert data[key] == BIBLE_WEIGHTS[key], f"{key}: the Reward Function states {BIBLE_WEIGHTS[key]!r}"


def test_df_v0_is_a_registered_score_profile() -> None:
    profile = default_profile()
    for method in ("score", "score_attempts", "apply"):
        assert callable(getattr(profile, method, None)), f"the df-v0 profile needs a {method} method"


# The probe: one attempt at each stage, one past the cap, one guarded, and a last one with W and A both set.
# (stage, unique counted warnings, alignment, guard violated) per attempt, as the formula reads them.
PROBE_FACTS: tuple[tuple[str, int, float | None, bool], ...] = (
    ("S0", 0, None, False),
    ("S1", 0, None, False),
    ("S2", 0, None, False),
    ("S3", 0, None, False),
    ("S4", 12, 1.0, False),
    ("S5", 0, 1.0, True),
    ("S5", 3, 0.5, False),
)


def probe_trial() -> Trial:
    """Return the SYNTHETIC probe trial PROBE_FACTS describes, with uncounted diagnostics mixed in."""
    return hand_trial([
        attempt(0, "S0", diagnostics=[FENCE_QUIRK]),
        compile_failed(1, diagnostics=unique_warnings(1)),
        attempt(2, "S2"),
        attempt(3, "S3"),
        run_failed(4, 1.0, diagnostics=twice(unique_warnings(12))),
        clean(5, 1.0, guards=Guards(host_compute=True)),
        clean(6, 0.5, diagnostics=[*twice(unique_warnings(3)), STDOUT_TRUNCATED]),
    ])


def formula(weights: Mapping[str, Any]) -> tuple[list[float], float, float]:
    """Return the probe's R per attempt, its single-turn value, and its multi-turn value, by the bible's formula.

    R = b(s) - warning_weight x min(W, warning_cap) + alignment_weight x A x 1[s = S5], or guard_violation
    when a guard is violated; single turn = R of attempt 0; multi turn = R of the last attempt minus
    correction_penalty x corrections.
    """
    rs = []
    for stage, count, alignment, violated in PROBE_FACTS:
        if violated:
            rs.append(weights["guard_violation"])
            continue
        term = weights["alignment_weight"] * alignment if stage == "S5" else 0.0
        rs.append(weights["stage_base"][stage] - weights["warning_weight"] * min(count, weights["warning_cap"]) + term)
    corrections = len(PROBE_FACTS) - 1
    return rs, rs[0], rs[-1] - weights["correction_penalty"] * corrections


def profile_values(profile: Any, trial: Trial) -> tuple[list[float], float, float]:
    """Return what `profile` gives the trial: R per attempt, the single-turn value, and the multi-turn value."""
    rs = [score.scalar for score in attempt_scores(profile, trial)]
    components = profile.score(trial).components
    return rs, components[SINGLE_TURN], components[MULTI_TURN]


def test_the_shipped_weights_score_the_probe_by_the_bible() -> None:
    rs, single, multi = profile_values(default_profile(), probe_trial())
    # Hand-computed with the bible's weights: S0 -1.0; S1 -0.8 (its warning uncounted); S2 -0.6; S3 -0.4;
    # S4 with W 12: 0.0 - 0.02 x 10 = -0.2 (A uncounted below S5); S5 guarded: -1.0;
    # S5 with W 3, A 0.5: 0.2 - 0.06 + 0.4 = 0.54. Single turn -1.0; multi turn 0.54 - 0.05 x 6 = 0.24.
    assert rs == pytest.approx([-1.0, -0.8, -0.6, -0.4, -0.2, -1.0, 0.54], abs=TOLERANCE)
    assert single == pytest.approx(-1.0, abs=TOLERANCE)
    assert multi == pytest.approx(0.24, abs=TOLERANCE)


# A changed value for each weight, each far enough from the bible's to move the probe's score.
CHANGED: dict[tuple[str, ...], float] = {
    ("stage_base", "S0"): -0.9,
    ("stage_base", "S1"): -0.7,
    ("stage_base", "S2"): -0.5,
    ("stage_base", "S3"): -0.3,
    ("stage_base", "S4"): 0.1,
    ("stage_base", "S5"): 0.3,
    ("warning_weight",): 0.03,
    ("warning_cap",): 5,
    ("alignment_weight",): 0.6,
    ("guard_violation",): -2.0,
    ("correction_penalty",): 0.1,
}


def with_value(data: Mapping[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    """Return a deep copy of `data` with the value at key path `path` set to `value`."""
    changed = copy.deepcopy(dict(data))
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return changed


def without(data: Mapping[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    """Return a deep copy of `data` with the key at key path `path` removed."""
    changed = copy.deepcopy(dict(data))
    target = changed
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    return changed


@pytest.mark.parametrize("path", WEIGHT_PATHS, ids=".".join)
def test_a_changed_weight_in_the_file_changes_the_score(tmp_path: Path, path: tuple[str, ...]) -> None:
    shipped = shipped_weights()
    changed = with_value(shipped, path, CHANGED[path])
    expected = formula(changed)
    assert expected != formula(BIBLE_WEIGHTS), "the probe must be sensitive to this weight"
    rs, single, multi = profile_values(profile_from(tmp_path, changed), probe_trial())
    expected_rs, expected_single, expected_multi = expected
    assert rs == pytest.approx(expected_rs, abs=TOLERANCE), f"R per attempt with {'.'.join(path)} changed"
    assert single == pytest.approx(expected_single, abs=TOLERANCE)
    assert multi == pytest.approx(expected_multi, abs=TOLERANCE)


@pytest.mark.parametrize("path", WEIGHT_PATHS, ids=".".join)
def test_a_weights_file_missing_a_weight_is_refused(tmp_path: Path, path: tuple[str, ...]) -> None:
    incomplete = without(shipped_weights(), path)
    with pytest.raises(ValueError, match=re.escape(path[-1])):
        profile = profile_from(tmp_path, incomplete)
        profile.score_attempts(probe_trial())
        profile.score(probe_trial())


# ---------------------------------------------------------------------------
# Attempt scores


@pytest.mark.parametrize(
    ("stage", "r"),
    [
        # Hand-computed: R = b(s) with W = 0, A term 0, no guard.
        ("S0", -1.0),
        ("S1", -0.8),
        ("S2", -0.6),
        ("S3", -0.4),
        ("S4", 0.0),
    ],
)
def test_each_stage_below_s5_scores_its_base(stage: str, r: float) -> None:
    run = ran(RUN_ERROR_EXIT) if stage == "S4" else None
    diagnostics = [RUN_ERROR] if stage == "S4" else []
    (score,) = attempt_scores(default_profile(), one_attempt(attempt(0, stage, run=run, diagnostics=diagnostics)))
    assert_attempt(score, r, stage_base=r, warning_count=0.0, alignment_term=0.0, guard=None)


def test_s5_scores_its_base_when_the_output_does_not_align() -> None:
    (score,) = attempt_scores(default_profile(), one_attempt(clean(0, 0.0)))
    # Hand-computed: R = 0.2 - 0.02 x 0 + 0.8 x 0.0 x 1 = 0.2.
    assert_attempt(score, 0.2, stage_base=0.2, warning_count=0.0, alignment_term=0.0, guard=None)


@pytest.mark.parametrize(
    ("alignment", "term", "r"),
    [
        (0.5, 0.4, 0.6),  # hand-computed: term 0.8 x 0.5 = 0.4; R = 0.2 + 0.4 = 0.6
        (1.0, 0.8, 1.0),  # hand-computed: term 0.8 x 1.0 = 0.8; R = 0.2 + 0.8 = 1.0
    ],
)
def test_the_alignment_term_counts_at_s5(alignment: float, term: float, r: float) -> None:
    (score,) = attempt_scores(default_profile(), one_attempt(clean(0, alignment)))
    assert_attempt(score, r, stage_base=0.2, warning_count=0.0, alignment_term=term, guard=None)


def test_the_alignment_term_counts_only_at_s5() -> None:
    # A run that failed after printing the reference's output: the oracle aligned it at 1.0, but it is S4.
    (score,) = attempt_scores(default_profile(), one_attempt(run_failed(0, 1.0)))
    # Hand-computed: R = 0.0 - 0.02 x 0 + 0.8 x 1.0 x 0 = 0.0.
    assert_attempt(score, 0.0, stage_base=0.0, warning_count=0.0, alignment_term=0.0, guard=None)


def test_w_counts_compile_and_jit_warnings_unique_by_code_file_line_and_column() -> None:
    note = Diagnostic(
        stage="compile", severity="note", code="n-a", file=SOURCE_FILE, line=20, column=1, message="SYNTHETIC note"
    )
    diagnostics = [
        # Counted: 1.
        warning(10, 5, code="w-a"),
        # The same (code, file, line, column) with another message: not counted again.
        warning(10, 5, code="w-a", message="other text"),
        # Another column: 2.
        warning(10, 9, code="w-a"),
        # Another code: 3.
        warning(10, 5, code="w-b"),
        # A jit-stage warning: 4.
        warning(3, 1, code="w-jit", stage="jit", file="kernel.cpp"),
        # A note, a parse-stage warning, and a run-stage warning: not counted.
        note,
        FENCE_QUIRK,
        STDOUT_TRUNCATED,
    ]
    (score,) = attempt_scores(default_profile(), one_attempt(run_failed(0, 0.0, diagnostics=diagnostics)))
    # Hand-computed: W = 4 at S4; R = 0.0 - 0.02 x 4 = -0.08.
    assert_attempt(score, -0.08, stage_base=0.0, warning_count=4.0, alignment_term=0.0, guard=None)


@pytest.mark.parametrize(
    ("count", "r"),
    [
        (9, 0.82),   # hand-computed: 0.2 - 0.02 x 9 + 0.8 x 1.0 = 0.82
        (10, 0.8),   # hand-computed: 0.2 - 0.02 x 10 + 0.8 x 1.0 = 0.8
        (12, 0.8),   # hand-computed: 0.2 - 0.02 x min(12, 10) + 0.8 x 1.0 = 0.8
    ],
)
def test_w_is_deduplicated_and_capped_at_10(count: int, r: float) -> None:
    item = clean(0, 1.0, diagnostics=twice(unique_warnings(count)))
    (score,) = attempt_scores(default_profile(), one_attempt(item))
    assert_attempt(score, r, stage_base=0.2, warning_count=float(count), alignment_term=0.8, guard=None)


@pytest.mark.parametrize(("stage", "r"), [("S1", -0.8), ("S3", -0.4)])
def test_warnings_of_an_attempt_that_did_not_compile_are_not_counted(stage: str, r: float) -> None:
    item = attempt(0, stage, diagnostics=[*unique_warnings(3), warning(4, stage="jit"), COMPILE_ERROR])
    (score,) = attempt_scores(default_profile(), one_attempt(item))
    # Hand-computed: below S4 no warning counts, so W = 0 and R = b(s).
    assert_attempt(score, r, stage_base=r, warning_count=0.0, alignment_term=0.0, guard=None)


@pytest.mark.parametrize("name", GUARD_NAMES)
def test_any_guard_true_sets_r_to_minus_one(name: str) -> None:
    item = clean(0, 1.0, diagnostics=unique_warnings(2), guards=Guards(**{name: True}))
    (score,) = attempt_scores(default_profile(), one_attempt(item))
    # Hand-computed: without the guard R would be 0.2 - 0.04 + 0.8 = 0.96; a violation sets R = -1.
    assert_attempt(score, -1.0, stage_base=0.2, warning_count=2.0, alignment_term=0.8, guard=1.0)


def test_every_guard_true_sets_r_to_minus_one() -> None:
    item = clean(0, 1.0, guards=Guards(host_compute=True, harness_tamper=True, oracle_access=True))
    (score,) = attempt_scores(default_profile(), one_attempt(item))
    assert_attempt(score, -1.0, stage_base=0.2, warning_count=0.0, alignment_term=0.8, guard=1.0)


@pytest.mark.parametrize(
    ("guards", "guard"),
    [
        # None checked; one checked and two not checked; all three checked and clean.
        (Guards(), None),
        (Guards(host_compute=False), None),
        (Guards(host_compute=False, harness_tamper=False, oracle_access=False), 0.0),
    ],
    ids=["none-checked", "one-checked", "all-checked"],
)
def test_a_guard_that_is_none_counts_as_not_checked(guards: Guards, guard: float | None) -> None:
    item = clean(0, 1.0, diagnostics=unique_warnings(2), guards=guards)
    (score,) = attempt_scores(default_profile(), one_attempt(item))
    # Hand-computed: no violation, so R = 0.2 - 0.02 x 2 + 0.8 x 1.0 = 0.96.
    assert_attempt(score, 0.96, stage_base=0.2, warning_count=2.0, alignment_term=0.8, guard=guard)


def stale_trial() -> Trial:
    """Return the SYNTHETIC faithful-loop trial whose last attempt compiled past the execution gate, unrun.

    Attempt 0 compiled and ran, its run failed after printing output the oracle aligned at 1.0; attempts 1 to 7
    did not compile; attempt 8 compiled past the gate and was not run, so it carries the stale-output warning
    and the output of attempt 0 stands as the trial's (final.alignment 1.0, as P2.3 sets it).
    """
    return hand_trial(
        [
            run_failed(0, 1.0, diagnostics=unique_warnings(1)),
            *(compile_failed(index) for index in range(1, 8)),
            attempt(8, "S4", diagnostics=[*twice(unique_warnings(2)), STALE_OUTPUT]),
        ],
        final_alignment=1.0,
    )


def test_a_stale_output_last_attempt_scores_its_own_record() -> None:
    scores = attempt_scores(default_profile(), stale_trial())
    # Hand-computed: attempt 0 is S4 with W 1: R = 0.0 - 0.02 = -0.02, its alignment uncounted below S5.
    assert_attempt(scores[0], -0.02, stage_base=0.0, warning_count=1.0, alignment_term=0.0, guard=None)
    # Hand-computed: attempts 1 to 7 are S1: R = -0.8.
    for score in scores[1:8]:
        assert_attempt(score, -0.8, stage_base=-0.8, warning_count=0.0, alignment_term=0.0, guard=None)
    # Hand-computed: attempt 8 is S4 with W 2 (the stale-output warning is run-stage, not counted) and no
    # alignment term (S4; the output that stands is attempt 0's): R = 0.0 - 0.02 x 2 = -0.04.
    assert_attempt(scores[8], -0.04, stage_base=0.0, warning_count=2.0, alignment_term=0.0, guard=None)


# ---------------------------------------------------------------------------
# Trial scores


def three_attempt_trial() -> Trial:
    """Return a SYNTHETIC trial: a compile error, then a failed run, then a clean aligned run."""
    return hand_trial(
        [
            compile_failed(0, diagnostics=unique_warnings(1)),
            run_failed(1, 0.0, diagnostics=unique_warnings(2)),
            clean(2, 1.0, diagnostics=unique_warnings(1)),
        ],
        final_alignment=1.0,
    )


# Hand-computed for three_attempt_trial: attempt 0 is S1: R = -0.8 (its warning uncounted); attempt 1 is S4 with
# W 2: R = 0.0 - 0.04 = -0.04; attempt 2 is S5 with W 1 and A 1.0: R = 0.2 - 0.02 + 0.8 = 0.98.
# Single turn = R of attempt 0 = -0.8; multi turn = 0.98 - 0.05 x 2 corrections = 0.88.
THREE_RS = (-0.8, -0.04, 0.98)
THREE_COMPONENTS = (
    {STAGE_BASE: -0.8, WARNING_COUNT: 0.0, ALIGNMENT_TERM: 0.0, GUARD: None},
    {STAGE_BASE: 0.0, WARNING_COUNT: 2.0, ALIGNMENT_TERM: 0.0, GUARD: None},
    {STAGE_BASE: 0.2, WARNING_COUNT: 1.0, ALIGNMENT_TERM: 0.8, GUARD: None},
)
THREE_SINGLE, THREE_MULTI = -0.8, 0.88


def test_the_trial_components_hold_the_single_turn_and_multi_turn_values() -> None:
    trial = three_attempt_trial()
    score = default_profile().score(trial)
    assert_components(score.components, {SINGLE_TURN: THREE_SINGLE, MULTI_TURN: THREE_MULTI})
    assert score.scalar == pytest.approx(THREE_MULTI, abs=TOLERANCE), "the trial's scalar is the multi-turn value"


def test_a_one_attempt_trial_has_equal_single_turn_and_multi_turn_values() -> None:
    score = default_profile().score(one_attempt(clean(0, 1.0, diagnostics=unique_warnings(3))))
    # Hand-computed: R = 0.2 - 0.06 + 0.8 = 0.94; no correction, so multi turn = 0.94 - 0.05 x 0 = 0.94.
    assert_components(score.components, {SINGLE_TURN: 0.94, MULTI_TURN: 0.94})
    assert score.scalar == pytest.approx(0.94, abs=TOLERANCE)


def test_a_stale_output_trial_takes_the_multi_turn_value_from_its_last_attempt() -> None:
    score = default_profile().score(stale_trial())
    # Hand-computed: single turn = R of attempt 0 = -0.02; multi turn = R of attempt 8 minus 0.05 x 8 corrections
    # = -0.04 - 0.40 = -0.44.
    assert_components(score.components, {SINGLE_TURN: -0.02, MULTI_TURN: -0.44})
    assert score.scalar == pytest.approx(-0.44, abs=TOLERANCE)


def test_apply_fills_each_attempt_score_and_final_score_with_the_multi_turn_value() -> None:
    trial = three_attempt_trial()
    filled = default_profile().apply(trial)
    assert isinstance(filled, Trial), f"apply returns a Trial, got {type(filled).__name__}"
    assert filled.final.score == pytest.approx(THREE_MULTI, abs=TOLERANCE), "final.score is the multi-turn value"
    assert len(filled.attempts) == len(trial.attempts)
    for index, (r, components) in enumerate(zip(THREE_RS, THREE_COMPONENTS, strict=True)):
        breakdown = filled.attempts[index].score
        assert isinstance(breakdown, ScoreBreakdown)
        assert breakdown.scalar == pytest.approx(r, abs=TOLERANCE), f"attempt {index}'s score scalar is its R"
        assert_components(breakdown.components, components)
    # Nothing else changes: with the scores cleared, the filled trial is the trial it was given.
    cleared = dataclasses.replace(
        filled,
        attempts=[dataclasses.replace(item, score=ScoreBreakdown()) for item in filled.attempts],
        final=dataclasses.replace(filled.final, score=None),
    )
    assert cleared == trial
    assert all(item.score == ScoreBreakdown() for item in trial.attempts), "the trial given is not changed"


# ---------------------------------------------------------------------------
# Cases the acceptance criteria leave open, as task P2.6 decides them: an S5 attempt never aligned scores no
# alignment term and is marked alignment_missing; R_final is the last attempt's R, a guard violation included;
# a trial with no attempts has no score.


# The component that marks an S5 attempt whose alignment mean is None.
ALIGNMENT_MISSING = "alignment_missing"


def test_an_s5_attempt_never_aligned_gets_no_alignment_term_and_is_marked() -> None:
    never_aligned = attempt(0, "S5", run=ran(0), diagnostics=unique_warnings(1))
    (score,) = attempt_scores(default_profile(), one_attempt(never_aligned))
    # Hand-computed: no alignment mean is no evidence of correct output, so R = 0.2 - 0.02 x 1 + 0.0 = 0.18.
    assert_attempt(score, 0.18, stage_base=0.2, warning_count=1.0, alignment_term=0.0, guard=None)
    assert_components(score.components, {ALIGNMENT_MISSING: 1.0})


@pytest.mark.parametrize(
    "item", [clean(0, 0.0), run_failed(0, 1.0), attempt(0, "S4")], ids=["s5-aligned", "s4-aligned", "s4-unaligned"]
)
def test_alignment_missing_is_zero_unless_an_s5_attempt_lacks_its_mean(item: Attempt) -> None:
    (score,) = attempt_scores(default_profile(), one_attempt(item))
    assert_components(score.components, {ALIGNMENT_MISSING: 0.0})


def test_a_guard_violation_on_the_last_attempt_is_r_final() -> None:
    trial = hand_trial([compile_failed(0), clean(1, 1.0, guards=Guards(oracle_access=True))], final_alignment=1.0)
    score = default_profile().score(trial)
    # Hand-computed: single turn = -0.8 (S1); R_final = -1.0 (guard); multi turn = -1.0 - 0.05 x 1 = -1.05.
    assert_components(score.components, {SINGLE_TURN: -0.8, MULTI_TURN: -1.05})
    assert score.scalar == pytest.approx(-1.05, abs=TOLERANCE)


def test_a_trial_with_no_attempts_has_no_score() -> None:
    trial = hand_trial([])
    profile = default_profile()
    assert profile.score_attempts(trial) == []
    score = profile.score(trial)
    assert_components(score.components, {SINGLE_TURN: None, MULTI_TURN: None})
    assert score.scalar is None
    filled = profile.apply(trial)
    assert filled.final.score is None
    assert filled == trial


@pytest.mark.parametrize(
    ("change", "named"),
    [
        (lambda data: {**data, "warning_weigth": 0.02}, "warning_weigth"),
        (lambda data: {**data, "stage_base": {**data["stage_base"], "S6": 0.4}}, "S6"),
        (lambda data: {**data, "alignment_weight": True}, "alignment_weight"),
        (lambda data: {**data, "correction_penalty": "0.05"}, "correction_penalty"),
        (lambda data: {**data, "warning_cap": 10.5}, "warning_cap"),
        (lambda data: {**data, "warning_cap": -1}, "warning_cap"),
    ],
    ids=["unknown-key", "unknown-stage", "bool", "text", "fractional-cap", "negative-cap"],
)
def test_a_weights_file_with_a_bad_key_or_value_is_refused(tmp_path: Path, change: Any, named: str) -> None:
    with pytest.raises(ValueError, match=re.escape(named)):
        profile_from(tmp_path, change(shipped_weights()))


def test_a_missing_weights_file_is_refused(tmp_path: Path) -> None:
    missing = tmp_path / "absent.yaml"
    with pytest.raises(ValueError, match="does not exist"):
        profile_class()(weights_path=missing)
