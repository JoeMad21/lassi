"""Tests for the adversary stage: which trials it takes, what it runs, and what it records (task P16.5).

Bible: Component Interfaces (Stage; Executor; Oracle; the adversary
bullet), Result Record (Trial; Output files and statistics; Storage),
Project Recipes (the bindings block's adversary line), Oracles (binary_io
rules), Sandbox (attempt run limits; model-generated code), Adversarial
Setups (item 1; the adversary rule; Input contract), Agent Rules 1, 6, 10,
and 14. Plan: plans/p16-adversarial.md, P16.5, and its planning decision
"Confirmed divergence (P16.5)".

The contract these tests fix:

- Importing lassi.core.runner registers the Stage `adversary` and, since
  the runner imports lassi.agents, the Agent `fuzzer`. A recipe lists the
  stage after oracle and binds the adversary at adversary.kind; the stage
  uses the Agent the run's registry holds under that name, built from its
  section's keys other than kind (the resolved config).
- The stage takes a trial only when its standing attempt
  (lassi.core.record standing_attempt) ran clean (S5) and passed binary_io
  on the held-out inputs (its alignment mean is 1.0). A trial it does not
  take keeps Trial.adversary null: the adversary is not asked and nothing
  runs.
- For a trial it takes, it asks the adversary within the section's
  budget.inputs: each AdversaryView holds the item's InputContract, the
  source program's files (path -> text), the standing attempt's files, the
  budget left (budget.inputs less the proposals made so far, refused ones
  included), and the earlier proposals' outcomes in order. It asks again
  while budget is left and the last reply proposed something, and never
  with no budget left. A reply of more proposals than the budget left is
  refused with a ValueError naming the Agent and its budget, and a reply
  item that is not a Proposal with a TypeError naming the Agent and the
  item's position, before any proposal of that reply runs.
- Each proposal is checked against the contract (lassi.harness
  check_inputs) before any run. A refused one runs nothing; its outcome
  code is the check_inputs reason (an array lassi_io cannot write, such as
  data shorter than its shape, is refused as unreadable, and one whose
  shape holds more than MAX_INPUT_ELEMENTS elements as too-many-elements,
  before its data is read), and Trial.adversary.refusals counts refusals
  by reason.
- Each accepted set runs the target reference program and the candidate
  (the standing attempt's program, rebuilt by the stage from the attempt's
  files with the target toolchain, as the target reference is, never
  copied from the attempt's build directory, where run_loop ran it and it
  could have changed its own file), each once, on the target language's
  bound executor (executors per language included) with run_loop's limits
  (lassi.core.stages attempt_limits). Each run uses a fresh workdir under
  the trial's directory, holding only a copy of the built program, the
  item's support files, and its @inputs written from the proposal
  (lassi.harness write_inputs; the program's sources may be there too),
  never an earlier run's outputs, so every run lists its own outputs, as
  the native executor lists only new files.
- A confirmed divergence: the target reference's run is clean with
  readable outputs and binary_io fails an output of the candidate's run (a
  missing output included, as from a candidate that crashed). The
  threshold is the recipe's metric and number, and under from_baseline the
  item's declared tolerance, never the references' agreement, which was
  measured on the held-out inputs only. Its record (Divergence) holds the
  proposal's index, its inputs as name -> sha256 of the lassi_io file in
  the run's binary store, binary_io's OutputStats, and candidate_failure:
  how the candidate's run failed (an exit status, a simulator gap, and the
  like), null when it ran clean, so a crash or a gap is never read as an
  output that merely differs. A reference that
  fails on an in-contract input is a contract finding (ContractFinding:
  the index, the inputs, and a message naming how the reference run
  ended), never a divergence. Both lists keep proposal order.
- The outcome codes of a proposal that ran: agreed, diverged, and
  contract-finding. Trial.adversary.runs counts every program run the
  stage made, proposals every proposal, refused ones included, and wall_s
  the stage's own wall time in the trial, within final.wall_s, which
  includes it.
- The stage changes no attempt's stage, alignment, or score (df-v0 bound,
  so the scores are values), and no part of the final block but the wall
  time a second run of the recipe measures anyway.

Every program, array, reply, and wall time here is SYNTHETIC (see
tests/core/adversary_support.py); no value in this module is a
measurement. Expected statistics come from binary_io itself over arrays
written here.
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from adversary_support import (
    AGREED,
    CANDIDATE_ROLE,
    CONTRACT_FINDING,
    CONTRACT_HI,
    CONTRACT_LO,
    DIVERGED,
    GAP,
    HELD_OUT,
    INPUT_FILE,
    OUTPUT_FILE,
    REFERENCE_ROLE,
    REWRITES,
    SOURCE_ROLE,
    SPLIT,
    SUITE,
    SUPPORT,
    SUPPORT_TEXT,
    WALL_S,
    Outcome,
    RunSeen,
    World,
    adversary_runs,
    adversary_stage,
    clean_environment,
    codes,
    constant,
    earlier_runs,
    program,
    proposal,
    recipe_data,
    run,
    stored_array,
    views,
    without_wall_time,
    write_suite,
)

from lassi.core.interfaces import Limits
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.core.store import trial_dir
from lassi.harness import LassiArray, input_contract, write_array
from lassi.harness.contract import MAX_INPUT_ELEMENTS

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no gate variable and a TMPDIR under the test's directory."""
    clean_environment(tmp_path, monkeypatch)


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the SYNTHETIC suite and return its bench root."""
    return write_suite(tmp_path, monkeypatch)


def run_taken(tmp_path: Path, bench_root: Path, world: World, *, item: str = "scale", **changes: Any) -> Outcome:
    """Run the recipe over `item` with `changes` after checking the stage is registered; return the outcome."""
    adversary_stage()
    changes.setdefault("bench", {"suite": SUITE, "split": SPLIT, "items": [item]})
    return run(tmp_path, bench_root, world, recipe_data(**changes), item=item)


def record_of(outcome: Outcome) -> Any:
    """Return the trial's adversary record, failing when the trial has none."""
    found = getattr(outcome.trial, "adversary", None)
    assert found is not None, "the trial records no adversary (Trial.adversary, task P16.5)"
    return found


def pairs(runs: list[RunSeen]) -> dict[bytes, list[RunSeen]]:
    """Group the adversary's runs by the input bytes they read."""
    grouped: dict[bytes, list[RunSeen]] = {}
    for seen in runs:
        assert seen.inputs is not None, f"a run read no input: {seen}"
        grouped.setdefault(seen.inputs.data, []).append(seen)
    return grouped


def expected_stats(tmp_path: Path, reference: tuple[float, ...] | None, candidate: tuple[float, ...] | None,
                   threshold: float) -> list[tuple[Any, ...]]:
    """Return binary_io's (name, pcc, max_abs, max_ulp, passed) for y arrays written here; None writes no file."""
    oracle = DEFAULT_REGISTRY.get("Oracle", "binary_io").factory(metric="max_abs", threshold=threshold)
    sides: dict[str, dict[str, Path]] = {}
    for side, ys in (("reference", reference), ("candidate", candidate)):
        directory = tmp_path / "expected" / side
        directory.mkdir(parents=True)
        sides[side] = {}
        if ys is not None:
            write_array(directory / OUTPUT_FILE, "y", "f32", (len(ys),), struct.pack(f"<{len(ys)}f", *ys))
            sides[side] = {OUTPUT_FILE: directory / OUTPUT_FILE}
    stats = oracle.compare(sides["reference"], sides["candidate"])
    return [(entry.name, entry.pcc, entry.max_abs, entry.max_ulp, entry.passed) for entry in stats]


def stat_tuples(stats: list[Any]) -> list[tuple[Any, ...]]:
    """Return each OutputStats as (name, pcc, max_abs, max_ulp, passed)."""
    return [(entry.name, entry.pcc, entry.max_abs, entry.max_ulp, entry.passed) for entry in stats]


# ---------------------------------------------------------------------------
# Registration


def test_importing_the_runner_registers_the_stage_and_the_fuzzer() -> None:
    code = (
        "import json\n"
        "import lassi.core.runner\n"
        "from lassi.core.registry import DEFAULT_REGISTRY as registry\n"
        "print(json.dumps({name: registry.names(name) for name in ('Stage', 'Agent')}))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    names = json.loads(done.stdout)
    assert "adversary" in names["Stage"], "lassi.core.runner registers no Stage 'adversary' (task P16.5)"
    assert "fuzzer" in names["Agent"], "lassi.core.runner does not import lassi.agents, so no fuzzer is registered"


# ---------------------------------------------------------------------------
# Which trials the stage takes


def test_a_trial_whose_standing_attempt_fails_binary_io_is_not_taken(tmp_path: Path, bench: Path) -> None:
    world = World(candidate="triple", batches=[[proposal(constant(500.0))]])
    outcome = run_taken(tmp_path, bench, world)
    standing = outcome.trial.attempts[-1]
    assert standing.stage_reached == "S5" and standing.alignment.mean == 0.0, "the candidate must run clean and fail"
    assert ("stage", "adversary") in world.events, "the runner runs the listed stage"
    assert views(world) == [], "the adversary is asked nothing about a trial the stage does not take"
    assert adversary_runs(world) == []
    assert outcome.trial.adversary is None


def test_a_taken_trial_records_the_agent_and_its_resolved_config(tmp_path: Path, bench: Path) -> None:
    world = World(candidate="double", batches=[[proposal(constant(1.0))]])
    outcome = run_taken(tmp_path, bench, world, adversary={"kind": "scripted-adversary", "budget": {"inputs": 3}})
    record = record_of(outcome)
    assert record.agent == "scripted-adversary"
    assert record.config == {"budget": {"inputs": 3}}
    assert record.proposals == 1 and record.refusals == {} and record.runs == 2
    assert record.divergences == [] and record.contract_findings == []
    final_wall = outcome.trial.final.wall_s
    assert final_wall is not None and 0.0 <= record.wall_s <= final_wall, "the stage's own time, inside the trial's"


def test_the_adversary_sees_the_contract_the_source_and_the_candidate(tmp_path: Path, bench: Path) -> None:
    world = World(candidate="large-zero", batches=[[proposal(constant(1.0))]])
    outcome = run_taken(tmp_path, bench, world)
    first = views(world)[0]
    declared = [{"name": "x", "dtype": "f32", "lo": CONTRACT_LO, "hi": CONTRACT_HI}]
    assert first.contract == input_contract([HELD_OUT], declared=declared)
    assert dict(first.source) == {"main.cpp": program(SOURCE_ROLE, "double")}
    assert dict(first.candidate) == outcome.trial.attempts[-1].files
    assert dict(first.candidate) == {"main.cpp": program(CANDIDATE_ROLE, "large-zero")}
    assert first.budget_left == 4 and tuple(first.outcomes) == ()


# ---------------------------------------------------------------------------
# The budget


def test_the_adversary_is_asked_within_its_budget(tmp_path: Path, bench: Path) -> None:
    world = World(candidate="double", propose=lambda view: [proposal(constant(float(len(view.outcomes) + 1)))])
    outcome = run_taken(tmp_path, bench, world, adversary={"kind": "scripted-adversary", "budget": {"inputs": 3}})
    assert [view.budget_left for view in views(world)] == [3, 2, 1], "never asked with no budget left"
    assert [len(view.outcomes) for view in views(world)] == [0, 1, 2]
    record = record_of(outcome)
    assert record.proposals == 3 and record.runs == 6 == len(adversary_runs(world))


def test_the_stage_stops_when_a_reply_proposes_nothing(tmp_path: Path, bench: Path) -> None:
    batch = [proposal(constant(1.0)), proposal(constant(2.0))]
    world = World(candidate="double", batches=[batch])
    outcome = run_taken(tmp_path, bench, world)
    assert [view.budget_left for view in views(world)] == [4, 2]
    assert record_of(outcome).proposals == 2 and len(adversary_runs(world)) == 4


def test_a_reply_past_the_budget_is_refused_before_any_of_it_runs(tmp_path: Path, bench: Path) -> None:
    greedy = [proposal(constant(float(value))) for value in range(1, 6)]
    world = World(candidate="double", batches=[greedy])
    with pytest.raises(ValueError) as caught:
        run_taken(tmp_path, bench, world, adversary={"kind": "scripted-adversary", "budget": {"inputs": 3}})
    message = str(caught.value)
    assert "scripted-adversary" in message and "budget" in message, message
    assert adversary_runs(world) == [], "no proposal of a reply past the budget runs"


def test_a_reply_item_that_is_not_a_proposal_is_refused_before_any_of_it_runs(tmp_path: Path, bench: Path) -> None:
    world = World(candidate="double", batches=[[proposal(constant(1.0)), "SYNTHETIC int main() {}"]])
    with pytest.raises(TypeError) as caught:
        run_taken(tmp_path, bench, world)
    message = str(caught.value)
    assert "scripted-adversary" in message and "position 1" in message and "str" in message, message
    assert adversary_runs(world) == [], "no proposal of a reply holding a non-Proposal runs"


# ---------------------------------------------------------------------------
# Refused proposals


def test_a_refused_proposal_runs_nothing_and_is_counted_by_reason(tmp_path: Path, bench: Path) -> None:
    short = LassiArray("x", "f32", (4,), b"\x00" * 12)  # data for three elements under a shape of four
    batches = [
        [proposal(constant(1.0))],
        [proposal(constant(5000.0))],
        [proposal(LassiArray("x", "f64", (4,), b"\x00" * 32))],
        [proposal()],
        [proposal(short)],
        [proposal(constant(2.0))],
    ]
    world = World(candidate="double", batches=batches)
    outcome = run_taken(tmp_path, bench, world, adversary={"kind": "scripted-adversary", "budget": {"inputs": 6}})
    seen = views(world)
    assert len(seen) == 6
    assert codes(seen[-1]) == [AGREED, "out-of-range", "dtype", "missing-input", "unreadable"]
    runs = adversary_runs(world)
    assert sorted({seen_run.inputs.data for seen_run in runs if seen_run.inputs is not None}) == sorted(
        {constant(1.0).data, constant(2.0).data}
    ), "only the accepted sets run"
    assert len(runs) == 4
    record = record_of(outcome)
    assert record.proposals == 6 and record.runs == 4
    assert record.refusals == {"out-of-range": 1, "dtype": 1, "missing-input": 1, "unreadable": 1}


def test_too_many_elements_and_a_repeated_input_are_refused_before_any_run(tmp_path: Path, bench: Path) -> None:
    huge = LassiArray("x", "f32", (MAX_INPUT_ELEMENTS + 1,), b"")  # refused on its shape, before its data is read
    batches = [
        [proposal(huge)],
        [proposal(constant(1.0), constant(2.0))],
        [proposal(constant(3.0))],
    ]
    world = World(candidate="double", batches=batches)
    outcome = run_taken(tmp_path, bench, world, adversary={"kind": "scripted-adversary", "budget": {"inputs": 3}})
    assert codes(views(world)[-1]) == ["too-many-elements", "repeated-input"]
    runs = adversary_runs(world)
    assert len(runs) == 2 and {seen.inputs.data for seen in runs if seen.inputs is not None} == {constant(3.0).data}
    record = record_of(outcome)
    assert record.proposals == 3 and record.runs == 2
    assert record.refusals == {"too-many-elements": 1, "repeated-input": 1}


# ---------------------------------------------------------------------------
# What each accepted set runs, where, and with which limits


@pytest.mark.parametrize(
    ("wall_s", "expected"),
    [(45, Limits(wall_s=45.0, memory_mb=2048, cpus=16)),
     ("baseline_x10", Limits(wall_s=max(10 * WALL_S, 30.0), memory_mb=2048, cpus=16))],
    ids=["seconds", "baseline-x10"],
)
def test_each_accepted_set_runs_the_reference_and_the_candidate_with_run_loops_limits(
    tmp_path: Path, bench: Path, wall_s: Any, expected: Limits
) -> None:
    sets = [constant(1.0), constant(500.0), constant(-3.0)]
    world = World(candidate="large-zero", batches=[[proposal(array)] for array in sets])
    outcome = run_taken(tmp_path, bench, world, sandbox={"wall_s": wall_s, "mem_gb": 2})
    attempt_runs = [seen for seen in earlier_runs(world) if seen.role == CANDIDATE_ROLE]
    assert attempt_runs and {seen.limits for seen in attempt_runs} == {expected}, "run_loop's own limits"
    grouped = pairs(adversary_runs(world))
    assert set(grouped) == {array.data for array in sets}
    for runs in grouped.values():
        assert sorted(seen.role for seen in runs) == sorted([REFERENCE_ROLE, CANDIDATE_ROLE])
        assert {seen.limits for seen in runs} == {expected}
    reference = next(seen for seen in adversary_runs(world) if seen.role == REFERENCE_ROLE)
    assert reference.behavior == "double" and reference.artifact.is_absolute()
    assert record_of(outcome).runs == 6


def test_the_runs_go_to_the_target_languages_executor(tmp_path: Path, bench: Path) -> None:
    world = World(candidate="double", batches=[[proposal(constant(1.0))], [proposal(constant(7.0))]])
    executor = {"omp": {"kind": "scripted-omp"}, "cpp": {"kind": "scripted-cpp"}}
    run_taken(tmp_path, bench, world, executor=executor)
    assert {seen.executor for seen in earlier_runs(world)} == {"scripted-omp", "scripted-cpp"}
    runs = adversary_runs(world)
    assert len(runs) == 4 and {seen.executor for seen in runs} == {"scripted-cpp"}


def test_every_run_gets_a_fresh_workdir_with_its_own_inputs_and_no_earlier_outputs(
    tmp_path: Path, bench: Path
) -> None:
    sets = [constant(1.0), constant(2.0), constant(3.0)]
    world = World(candidate="double", batches=[[proposal(array)] for array in sets])
    outcome = run_taken(tmp_path, bench, world)
    runs = adversary_runs(world)
    assert len(runs) == 6
    home = trial_dir(outcome.run_dir, outcome.trial.trial_id).resolve()
    workdirs = [seen.workdir.resolve() for seen in runs]
    assert len(set(workdirs)) == 6, "each run, reference and candidate alike, gets its own workdir"
    earlier = {seen.workdir.resolve() for seen in earlier_runs(world)}
    for seen, workdir in zip(runs, workdirs, strict=True):
        assert home in workdir.parents, f"{workdir} is not under the trial's directory {home}"
        assert workdir not in earlier, "an adversary run reuses the baseline's or an attempt's workdir"
        allowed = {seen.artifact.name, "main.cpp", SUPPORT, INPUT_FILE}
        assert {seen.artifact.name, SUPPORT, INPUT_FILE} <= seen.before <= allowed, sorted(seen.before)
        assert OUTPUT_FILE not in seen.before
        assert (workdir / SUPPORT).read_text(encoding="ascii") == SUPPORT_TEXT
        assert seen.outputs == {OUTPUT_FILE}, "every run lists its own output"
    assert {seen.inputs.data for seen in runs if seen.inputs is not None} == {array.data for array in sets}


def test_a_control_finds_no_divergence_and_no_missing_output(tmp_path: Path, bench: Path) -> None:
    sets = [constant(1.0), constant(-250.0), constant(999.0), constant(0.0)]
    world = World(candidate="double", batches=[[proposal(array)] for array in sets])
    outcome = run_taken(tmp_path, bench, world)
    record = record_of(outcome)
    assert record.divergences == [] and record.contract_findings == []
    assert len(adversary_runs(world)) == 8
    assert all(seen.outputs == {OUTPUT_FILE} and seen.exit_code == 0 for seen in adversary_runs(world))
    assert codes(views(world)[-1]) == [AGREED, AGREED, AGREED]


# ---------------------------------------------------------------------------
# Confirmed divergences and contract findings


def test_a_confirmed_divergence_records_its_inputs_in_the_binary_store_and_its_stats(
    tmp_path: Path, bench: Path
) -> None:
    world = World(candidate="large-zero", batches=[[proposal(constant(1.0))], [proposal(constant(500.0))], []])
    outcome = run_taken(tmp_path, bench, world)
    assert codes(views(world)[-1]) == [AGREED, DIVERGED]
    record = record_of(outcome)
    (divergence,) = record.divergences
    assert divergence.proposal == 1
    assert set(divergence.inputs) == {"x"}
    assert stored_array(outcome.run_dir, divergence.inputs["x"], tmp_path / "blobs-read") == constant(500.0)
    expected = expected_stats(tmp_path, (1000.0,) * 4, (0.0,) * 4, 0.25)
    assert stat_tuples(divergence.stats) == expected
    assert expected[0][4] is False
    assert divergence.candidate_failure is None, "the candidate's run was clean; only its outputs differ"
    assert record.contract_findings == []


@pytest.mark.parametrize(("threshold", "diverges"), [(0.25, True), (1.0, False)], ids=["below", "above"])
def test_the_threshold_is_the_recipes_number(tmp_path: Path, bench: Path, threshold: float, diverges: bool) -> None:
    world = World(candidate="large-half", batches=[[proposal(constant(500.0))]])
    oracle = {"kind": "binary_io", "metric": "max_abs", "threshold": threshold}
    outcome = run_taken(tmp_path, bench, world, oracle=oracle)
    record = record_of(outcome)
    assert bool(record.divergences) is diverges
    if diverges:
        assert stat_tuples(record.divergences[0].stats) == expected_stats(
            tmp_path, (1000.0,) * 4, (1000.5,) * 4, threshold
        )


@pytest.mark.parametrize(("candidate", "diverges"), [("large-half", False), ("large-two", True)])
def test_under_from_baseline_the_threshold_is_the_items_declared_tolerance(
    tmp_path: Path, bench: Path, candidate: str, diverges: bool
) -> None:
    world = World(candidate=candidate, batches=[[proposal(constant(500.0))]])
    oracle = {"kind": "binary_io", "metric": "max_abs", "threshold": "from_baseline"}
    outcome = run_taken(tmp_path, bench, world, oracle=oracle)
    agreement = outcome.trial.reference_agreement
    assert agreement is not None and [entry.max_abs for entry in agreement] == [0.0], "agreement measured exactly"
    record = record_of(outcome)
    assert bool(record.divergences) is diverges, "the item's tolerance (max_abs 1.0) is the threshold, not 0.0"


def test_a_reference_that_fails_on_an_in_contract_input_is_a_contract_finding(tmp_path: Path, bench: Path) -> None:
    world = World(candidate="double", batches=[[proposal(constant(-3.0))], [proposal(constant(1.0))]])
    outcome = run_taken(tmp_path, bench, world, item="fragile")
    assert codes(views(world)[1]) == [CONTRACT_FINDING]
    assert codes(views(world)[-1]) == [CONTRACT_FINDING, AGREED]
    record = record_of(outcome)
    assert record.divergences == [], "a reference failure is never a divergence"
    (finding,) = record.contract_findings
    assert finding.proposal == 0 and set(finding.inputs) == {"x"}
    assert stored_array(outcome.run_dir, finding.inputs["x"], tmp_path / "blobs-read") == constant(-3.0)
    assert "139" in finding.message, finding.message
    assert record.runs == len(adversary_runs(world))


def test_a_candidate_that_crashes_where_the_reference_runs_clean_diverges(tmp_path: Path, bench: Path) -> None:
    world = World(candidate="crash-negative", batches=[[proposal(constant(-3.0))]])
    outcome = run_taken(tmp_path, bench, world)
    candidate = next(seen for seen in adversary_runs(world) if seen.role == CANDIDATE_ROLE)
    assert candidate.exit_code == 139 and candidate.outputs == frozenset()
    record = record_of(outcome)
    (divergence,) = record.divergences
    assert stat_tuples(divergence.stats) == expected_stats(tmp_path, (-6.0,) * 4, None, 0.25)
    assert divergence.candidate_failure is not None and "139" in divergence.candidate_failure
    assert record.contract_findings == []


def test_a_candidate_that_stops_at_a_simulator_gap_diverges_with_the_gap_named(tmp_path: Path, bench: Path) -> None:
    world = World(candidate="gap-large", batches=[[proposal(constant(500.0))]])
    outcome = run_taken(tmp_path, bench, world)
    candidate = next(seen for seen in adversary_runs(world) if seen.role == CANDIDATE_ROLE)
    assert candidate.outputs == frozenset(), "the gap run writes no output"
    (divergence,) = record_of(outcome).divergences
    assert stat_tuples(divergence.stats) == expected_stats(tmp_path, (1000.0,) * 4, None, 0.25)
    failure = divergence.candidate_failure
    assert failure is not None and "simulator gap" in failure and GAP in failure, failure


def test_the_candidate_is_rebuilt_from_its_files_not_copied_from_its_attempts_build(
    tmp_path: Path, bench: Path
) -> None:
    world = World(candidate=REWRITES, batches=[[proposal(constant(500.0))], [proposal(constant(600.0))]])
    outcome = run_taken(tmp_path, bench, world)
    (attempt_run,) = [seen for seen in earlier_runs(world) if seen.role == CANDIDATE_ROLE]
    assert b"SYNTHETIC-PROGRAM double" in attempt_run.artifact.read_bytes(), "run_loop's run rewrote its artifact"
    runs = [seen for seen in adversary_runs(world) if seen.role == CANDIDATE_ROLE]
    assert [seen.behavior for seen in runs] == [REWRITES, REWRITES], "each run gets the program as its files build"
    assert all(attempt_run.workdir.resolve() not in seen.workdir.resolve().parents for seen in runs)
    assert codes(views(world)[-1]) == [DIVERGED, DIVERGED]
    assert [entry.proposal for entry in record_of(outcome).divergences] == [0, 1]


# ---------------------------------------------------------------------------
# Measured beats judged: nothing else in the trial changes


def test_the_stage_changes_no_attempt_stage_alignment_or_score(tmp_path: Path, bench: Path) -> None:
    world = World(candidate="large-zero", batches=[[proposal(constant(500.0))]])
    taken = run_taken(tmp_path, bench, world, score="df-v0")
    assert record_of(taken).divergences, "the adversary found the fault"
    plain_stages = ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
    plain = recipe_data(stages=plain_stages, adversary=None, score="df-v0")
    untouched = run(tmp_path, bench, World(candidate="large-zero"), plain, run_id="no-adversary")
    assert untouched.trial.adversary is None
    scores = [attempt.score for attempt in taken.trial.attempts]
    assert scores and all(score is not None for score in scores), "df-v0 scores every attempt, so scores compare"
    assert taken.trial.final.score is not None
    assert scores == [attempt.score for attempt in untouched.trial.attempts]
    assert taken.trial.attempts == untouched.trial.attempts
    assert taken.trial.reference_run == untouched.trial.reference_run
    assert taken.trial.reference_agreement == untouched.trial.reference_agreement
    assert without_wall_time(taken.trial).final == without_wall_time(untouched.trial).final
    assert taken.trial.attempts[-1].alignment.mean == 1.0


# ---------------------------------------------------------------------------
# The fuzzer through the stage


@pytest.mark.parametrize(("candidate", "finds"), [("large-zero", True), ("double", False)], ids=["fault", "control"])
def test_the_fuzzer_finds_the_seeded_fault_and_nothing_in_a_control(
    tmp_path: Path, bench: Path, candidate: str, finds: bool
) -> None:
    world = World(candidate=candidate)
    section = {"kind": "fuzzer", "budget": {"inputs": 6}, "seed": 1}
    outcome = run_taken(tmp_path, bench, world, adversary=section)
    record = record_of(outcome)
    assert record.agent == "fuzzer" and record.config == {"budget": {"inputs": 6}, "seed": 1}
    assert record.proposals == 6 and record.refusals == {}, "every fuzzer proposal is inside the contract"
    assert record.runs == 12 == len(adversary_runs(world))
    assert bool(record.divergences) is finds
    if finds:
        first = record.divergences[0]
        assert first.proposal == 0, "boundary proposal 0 fills x with the contract's lowest value"
        assert stored_array(outcome.run_dir, first.inputs["x"], tmp_path / "blobs-read") == constant(CONTRACT_LO)


def test_trial_json_holds_the_record(tmp_path: Path, bench: Path) -> None:
    world = World(candidate="large-zero", batches=[[proposal(constant(500.0))]])
    outcome = run_taken(tmp_path, bench, world)
    data = json.loads((trial_dir(outcome.run_dir, outcome.trial.trial_id) / "trial.json").read_text(encoding="ascii"))
    stored = data["adversary"]
    assert stored["agent"] == "scripted-adversary" and stored["proposals"] == 1 and stored["runs"] == 2
    array = constant(500.0)
    written = tmp_path / "expected-input" / "x.lassiio"
    written.parent.mkdir()
    write_array(written, array.name, array.dtype, array.shape, array.data)
    assert stored["divergences"][0]["inputs"] == {"x": hashlib.sha256(written.read_bytes()).hexdigest()}
    assert stored["divergences"][0]["candidate_failure"] is None
    assert isinstance(stored["wall_s"], float)
