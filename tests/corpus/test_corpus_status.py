"""Tests for the per-entry gate status, the exit-code taxonomy moved from v0 (task P5.11).

Bible: Review Of v0 (the Keep row: exit-code taxonomy, moved into
lassi/corpus/), Corpus Pipeline (Gate; Record, gate status). Plan:
plans/p5-ir.md, task P5.11, and the planning decision "Gate reading
(P5.12)" (the steps a round trip passes: raise, verify, normalize, lower,
build, run, compare); plans/spikes/p5-v0-corpus.md, Results 5, the
migration map's exit-code row (a status per corpus entry: the furthest step
it passed and the failing step's code, over the gate reading's steps plus
timeout and killed, which v0's lower_all.sh adds; P5.12's per-kernel gate
rate reads the same field; test: each step's failure maps to one code, an
unknown code is refused).

The contract these tests fix (lassi.corpus):

- GATE_STEPS is ("raise", "verify", "normalize", "lower", "build", "run",
  "compare"), in that order.
- failure_code(step) returns the one code of a failure at that step; a
  name outside GATE_STEPS raises ValueError. FAILURE_CODES holds the seven
  step codes, all distinct, then "timeout" and "killed": nine codes.
- GateStatus(passed=<step or None>, code=<code or None>) is frozen. passed
  is the furthest step the entry passed (None when it passed none) and code
  is None exactly when passed is the last step (a gated entry). Otherwise
  code is a failure at the step right after `passed` (the failing step's
  code) or timeout or killed, which also stop that step. Anything else,
  an unknown code or step among it, raises ValueError naming the value.

No value here is a measurement.
"""

from __future__ import annotations

import dataclasses

import pytest
from corpus_fixtures import corpus

STEPS = ("raise", "verify", "normalize", "lower", "build", "run", "compare")


def test_the_gate_steps_are_the_gate_readings_in_order() -> None:
    assert tuple(corpus().GATE_STEPS) == STEPS


def test_each_steps_failure_maps_to_one_code_and_the_codes_add_timeout_and_killed() -> None:
    module = corpus()
    codes = [module.failure_code(step) for step in STEPS]
    assert len(set(codes)) == len(STEPS), codes
    assert not {"timeout", "killed"} & set(codes)
    assert tuple(module.FAILURE_CODES) == (*codes, "timeout", "killed")


def test_a_failure_at_an_unknown_step_is_refused() -> None:
    with pytest.raises(ValueError, match="no-such-step"):
        corpus().failure_code("no-such-step")


def test_the_status_is_a_frozen_record_of_the_furthest_step_and_the_code() -> None:
    module = corpus()
    assert dataclasses.is_dataclass(module.GateStatus) and module.GateStatus.__dataclass_params__.frozen
    assert {item.name for item in dataclasses.fields(module.GateStatus)} == {"passed", "code"}


def valid_statuses() -> list[tuple[str | None, str | None]]:
    """Return every valid (passed, code) pair: a gated entry and each step's failure, timeout, and killed."""
    module = corpus()
    pairs: list[tuple[str | None, str | None]] = [(STEPS[-1], None)]
    for index, step in enumerate(STEPS):
        before = STEPS[index - 1] if index else None
        pairs += [(before, module.failure_code(step)), (before, "timeout"), (before, "killed")]
    return pairs


def test_every_valid_status_builds() -> None:
    module = corpus()
    for passed, code in valid_statuses():
        status = module.GateStatus(passed=passed, code=code)
        assert (status.passed, status.code) == (passed, code)


def test_an_unknown_code_or_step_is_refused_naming_it() -> None:
    with pytest.raises(ValueError, match="no-such-code"):
        corpus().GateStatus(passed="build", code="no-such-code")
    with pytest.raises(ValueError, match="no-such-step"):
        corpus().GateStatus(passed="no-such-step", code="timeout")


@pytest.mark.parametrize(
    ("passed", "code"),
    [("run", None), (None, None), ("compare", "timeout"), ("compare", "killed")],
    ids=["unfinished-without-a-code", "nothing-passed-without-a-code", "timeout-after-all", "killed-after-all"],
)
def test_a_status_whose_code_does_not_fit_its_step_is_refused(passed: str | None, code: str | None) -> None:
    with pytest.raises(ValueError):
        corpus().GateStatus(passed=passed, code=code)


@pytest.mark.parametrize(
    ("passed", "failing"),
    [("raise", "raise"), ("raise", "normalize"), (None, "verify"), ("build", "build")],
    ids=["the-step-it-passed", "a-step-skipped", "nothing-passed-later-step", "same-step"],
)
def test_a_code_must_name_the_step_after_the_furthest_one_passed(passed: str | None, failing: str) -> None:
    module = corpus()
    with pytest.raises(ValueError):
        module.GateStatus(passed=passed, code=module.failure_code(failing))
