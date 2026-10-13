"""Tests for the guard coverage counts of a run (task P17.14).

Bible: Result Record (Guards: Attempt.guards.host_compute and
host_compute_not_checked), Evaluation Protocol (Run Metrics, Guard
coverage), Readability Standards (Run row), Agent Rule 1; OQ-028 (the
owner's condition of 2026-10-05: the reason is logged whenever the guard
does not run, and the not-checked rate is reported per run and per
target); OQ-018 (names and counts only).

The contract these tests fix, from the P17.14 acceptance and its design:

- lassi.analysis.guard_coverage holds NOT_RECORDED ("not recorded") and
  REASON_KEYS, the codes of lassi.core.record.HOST_COMPUTE_NOT_CHECKED in
  order and then NOT_RECORDED.
- GuardCoverage is a frozen record of `attempts`, `checked`, and
  `reasons` (a mapping holding every REASON_KEYS key in order, a count of 0
  included), with the properties `not_checked` (the sum of the reason
  counts) and `rate` (not_checked / attempts, None when there is no
  attempt). The rate has no interval.
- count_attempts(attempts) counts Attempt records. The unit is the
  attempt, whatever its stage. Checked is host_compute true or false; an
  attempt whose host_compute is null counts under its code, or under
  NOT_RECORDED when it records none (a record written before the task),
  never under a code read from anything else (Agent Rule 1).
- combined(coverages) sums coverages.
- target_language(direction) is the part of a direction's name after its
  only hyphen, and None when the name holds more than one hyphen.
- summary_rows(groups), for (direction, coverage) groups, gives ("run",
  every group summed), then ("target <language>", that target's groups
  summed) for each target in sorted order, then ("direction <name>",
  coverage) for each direction whose target cannot be read.
- guard_coverage(trials) groups every attempt of the trials by the
  trial_id's direction and gives summary_rows; a trial with no attempt
  adds nothing.
- coverage_cells(coverage) gives the attempts, checked, not checked, the
  rate with three decimals or "-", and each REASON_KEYS count, as text.
  coverage_markdown(rows) gives a plain ASCII section: the heading
  "## Guard coverage", a legend, and one table whose columns are Scope,
  Attempts, Checked, Not checked, Not-checked rate, and one per
  REASON_KEYS entry, one line per row.

Every trial here is SYNTHETIC and hand-built from the Result Record
classes; no model is called and no program is run. Every expected count is
counted by hand, as the comments show. No value in this module is a
measurement.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import pytest

from lassi.core.interfaces import Sampling
from lassi.core.record import (
    Attempt,
    BenchItem,
    Final,
    Guards,
    ModelInfo,
    Provenance,
    Trial,
    from_dict,
    make_trial_id,
    to_dict,
)

MODULE = "lassi.analysis.guard_coverage"
CODES = ("not-built", "no-program", "no-guard", "guard-not-checked")
NOT_RECORDED = "not recorded"
REASON_KEYS = (*CODES, NOT_RECORDED)
COLUMNS = ("Scope", "Attempts", "Checked", "Not checked", "Not-checked rate", *REASON_KEYS)

PROJECT, SUITE, ITEM = "guard-coverage", "synthetic-suite", "probe"
ARM_A, ARM_B = "fixture-arm-a", "fixture-arm-b"
CPU_TT, TT_CPU, OMP_CUDA = "cpu-tt", "tt-cpu", "omp-cuda"
# A direction whose name holds two hyphens: a language name may hold one, so its target cannot be read.
TWO_HYPHENS = "c-plus-cuda"
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
# Each attempt kind: True or False is a guard reading; a code is a null reading with that code; None records none.
CHECKED_TRUE, CHECKED_FALSE = True, False


# ---------------------------------------------------------------------------
# The module under test, looked up so that a missing one fails each test with a clear message


def module() -> ModuleType:
    """Import lassi.analysis.guard_coverage; fail the test clearly while task P17.14 has not added it."""
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as error:
        if error.name != MODULE:
            raise
        pytest.fail(f"task P17.14 adds {MODULE}: {error}")


# ---------------------------------------------------------------------------
# SYNTHETIC attempts and trials


def attempt(index: int, kind: bool | str | None) -> Attempt:
    """Return a SYNTHETIC attempt: a reading of `kind` when it is a bool, else a null reading with code `kind`."""
    if isinstance(kind, bool):
        return Attempt(index=index, stage_reached="S4", guards=Guards(host_compute=kind))
    stage = "S1" if kind in (None, "not-built", "no-program") else "S4"
    return Attempt(index=index, stage_reached=stage, guards=Guards(host_compute_not_checked=kind))


def trial(direction: str, kinds: Sequence[bool | str | None], *, arm: str = ARM_A, run: int = 1) -> Trial:
    """Return a SYNTHETIC trial in `direction` whose attempts are of `kinds`, in order (none: ended at the baseline)."""
    attempts = [attempt(index, kind) for index, kind in enumerate(kinds)]
    return Trial(
        trial_id=make_trial_id(PROJECT, arm, SUITE, direction, ITEM, run),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=None, dirty=None, device=None, sdk=None, date="2026-10-06T00:00:00+00:00"),
        bench_item=BenchItem(suite=SUITE, item=ITEM, split="synthetic", direction=direction),
        model=ModelInfo(backend="scripted", id=arm, sampling=SAMPLING),
        attempts=attempts,
        final=Final(stage_reached=attempts[-1].stage_reached if attempts else None,
                    corrections=max(len(attempts) - 1, 0)),
    )


def counts(coverage: Any) -> dict[str, Any]:
    """Return a coverage's values as plain data: attempts, checked, not_checked, rate, and every reason count."""
    return {
        "attempts": coverage.attempts, "checked": coverage.checked, "not_checked": coverage.not_checked,
        "rate": coverage.rate, "reasons": dict(coverage.reasons),
    }


def expected(attempts: int, checked: int, **reasons: int) -> dict[str, Any]:
    """Return the plain data of a coverage; `reasons` are keyed by code with underscores, a missing one 0."""
    by_key = {key: reasons.get(key.replace("-", "_").replace(" ", "_"), 0) for key in REASON_KEYS}
    not_checked = attempts - checked
    assert sum(by_key.values()) == not_checked, "the hand counts sum to not checked"
    return {
        "attempts": attempts, "checked": checked, "not_checked": not_checked,
        "rate": None if attempts == 0 else not_checked / attempts, "reasons": by_key,
    }


def make(attempts: int, checked: int, **reasons: int) -> Any:
    """Return a GuardCoverage built from hand counts, keyed as `expected` keys them."""
    data = expected(attempts, checked, **reasons)
    return module().GuardCoverage(attempts=attempts, checked=checked, reasons=data["reasons"])


# The cpu -> tt run: arm A's two trials and one that ended at the baseline, and arm B's one trial.
CPU_TT_TRIALS = (
    ("not-built", "no-program", "guard-not-checked", CHECKED_FALSE),  # 4 attempts, 1 checked
    (),  # ended at the baseline: adds nothing
    (CHECKED_TRUE,),  # 1 attempt, checked
)
CPU_TT_ARM_B = ("no-program", CHECKED_FALSE)  # 2 attempts, 1 checked
# The tt -> cpu run: the cpu toolchain declares no guard, so every attempt is no-guard.
TT_CPU_TRIALS = (("no-guard", "no-guard", "no-guard"), ("no-guard",))


def run_trials() -> list[Trial]:
    """Return every SYNTHETIC trial of the run: both arms of cpu -> tt, and tt -> cpu."""
    trials = [trial(CPU_TT, kinds, run=run) for run, kinds in enumerate(CPU_TT_TRIALS, 1)]
    trials.append(trial(CPU_TT, CPU_TT_ARM_B, arm=ARM_B))
    trials += [trial(TT_CPU, kinds, run=run) for run, kinds in enumerate(TT_CPU_TRIALS, 1)]
    return trials


# ---------------------------------------------------------------------------
# The constants


def test_the_reason_keys_are_the_codes_then_not_recorded() -> None:
    record = importlib.import_module("lassi.core.record")
    assert module().NOT_RECORDED == NOT_RECORDED
    assert tuple(module().REASON_KEYS) == (*record.HOST_COMPUTE_NOT_CHECKED, NOT_RECORDED) == REASON_KEYS
    assert NOT_RECORDED not in record.HOST_COMPUTE_NOT_CHECKED, "not recorded is a count, never a stored code"


# ---------------------------------------------------------------------------
# Counting attempts


def test_counts_every_attempt_and_no_trial_without_one() -> None:
    cpu_tt = [item for kinds in CPU_TT_TRIALS for item in trial(CPU_TT, kinds).attempts]
    # Hand counts: 4 + 0 + 1 attempts; checked: the False and the True; one of each of three codes.
    assert counts(module().count_attempts(cpu_tt)) == expected(5, 2, not_built=1, no_program=1, guard_not_checked=1)
    rows = dict(module().guard_coverage([trial(CPU_TT, kinds, run=run) for run, kinds in enumerate(CPU_TT_TRIALS, 1)]))
    assert counts(rows["run"]) == expected(5, 2, not_built=1, no_program=1, guard_not_checked=1)


def test_checked_and_the_reason_counts_sum_to_attempts() -> None:
    attempts = [attempt(index, kind) for index, kind in enumerate((*CODES, None, True, False, *CODES))]
    coverage = module().count_attempts(attempts)
    assert list(coverage.reasons) == list(REASON_KEYS), "every key, in order"
    assert coverage.checked + coverage.not_checked == coverage.attempts == 11
    assert sum(coverage.reasons.values()) == coverage.not_checked == 9
    assert dict(coverage.reasons) == {"not-built": 2, "no-program": 2, "no-guard": 2, "guard-not-checked": 2,
                                      NOT_RECORDED: 1}


def test_every_reason_key_is_kept_with_a_zero_count() -> None:
    coverage = module().count_attempts([attempt(0, True)])
    assert dict(coverage.reasons) == dict.fromkeys(REASON_KEYS, 0)
    assert (coverage.attempts, coverage.checked, coverage.not_checked, coverage.rate) == (1, 1, 0, 0.0)


def test_an_attempt_without_a_recorded_reason_counts_as_not_recorded() -> None:
    # An attempt from a trial.json written before the field: host_compute null and no code key at all.
    data = to_dict(Attempt(index=0, stage_reached="S1"))
    data["guards"].pop("host_compute_not_checked", None)
    older = from_dict(Attempt, data)
    coverage = module().count_attempts([older, attempt(1, None)])
    assert counts(coverage) == expected(2, 0, not_recorded=2), "never read as not-built or any other code"


def test_the_rate_is_null_without_attempts() -> None:
    empty = module().count_attempts([])
    assert counts(empty) == expected(0, 0)
    assert empty.rate is None and empty.not_checked == 0
    rows = module().guard_coverage([trial(CPU_TT, ())])
    assert rows[0][0] == "run" and counts(rows[0][1]) == expected(0, 0), "a trial with no attempt adds nothing"
    assert counts(dict(module().guard_coverage([]))["run"]) == expected(0, 0), "an empty run still has a run row"


def test_the_rate_is_not_checked_over_attempts() -> None:
    kinds = ("no-program", None, True, "no-guard")
    coverage = module().count_attempts([attempt(index, kind) for index, kind in enumerate(kinds)])
    # Hand counts: 4 attempts, 1 checked, 3 not checked: 3/4.
    assert coverage.rate == 0.75 and type(coverage.rate) is float
    assert not [name for name in ("wilson_low", "wilson_high", "interval") if hasattr(coverage, name)], (
        "the rate describes the record and has no interval")


def test_combined_sums_coverages() -> None:
    first = make(4, 1, not_built=1, no_program=1, guard_not_checked=1)
    second = make(3, 0, no_guard=2, not_recorded=1)
    assert counts(module().combined([first, second])) == expected(
        7, 1, not_built=1, no_program=1, no_guard=2, guard_not_checked=1, not_recorded=1)
    assert counts(module().combined([])) == expected(0, 0)


# ---------------------------------------------------------------------------
# Targets and summary rows


@pytest.mark.parametrize(
    ("direction", "target"),
    [(CPU_TT, "tt"), (TT_CPU, "cpu"), (OMP_CUDA, "cuda"), ("cuda-omp", "omp"), (TWO_HYPHENS, None)],
)
def test_target_language_is_the_part_after_the_only_hyphen(direction: str, target: str | None) -> None:
    assert module().target_language(direction) == target


def test_summary_rows_give_the_run_then_each_target_then_unreadable_directions() -> None:
    omp_cuda = make(2, 2)
    cpu_cuda = make(3, 1, no_program=2)
    tt_cpu = make(4, 0, no_guard=4)
    odd = make(1, 0, not_recorded=1)
    rows = module().summary_rows([(TT_CPU, tt_cpu), (TWO_HYPHENS, odd), (OMP_CUDA, omp_cuda), ("cpu-cuda", cpu_cuda)])
    assert [scope for scope, _ in rows] == ["run", "target cpu", "target cuda", f"direction {TWO_HYPHENS}"]
    found = {scope: counts(coverage) for scope, coverage in rows}
    # Hand sums: run 2 + 3 + 4 + 1 = 10 attempts, 3 checked; cuda 2 + 3 = 5 attempts, 3 checked.
    assert found["run"] == expected(10, 3, no_program=2, no_guard=4, not_recorded=1)
    assert found["target cuda"] == expected(5, 3, no_program=2)
    assert found["target cpu"] == expected(4, 0, no_guard=4)
    assert found[f"direction {TWO_HYPHENS}"] == expected(1, 0, not_recorded=1)


def test_guard_coverage_reports_the_run_and_each_target_over_every_arm() -> None:
    rows = module().guard_coverage(run_trials())
    assert [scope for scope, _ in rows] == ["run", "target cpu", "target tt"]
    found = {scope: counts(coverage) for scope, coverage in rows}
    # Hand counts, cpu -> tt: arm A 5 attempts (2 checked), arm B 2 (1 checked); tt -> cpu: 4 attempts, all no-guard.
    assert found["target tt"] == expected(7, 3, not_built=1, no_program=2, guard_not_checked=1)
    assert found["target cpu"] == expected(4, 0, no_guard=4)
    assert found["run"] == expected(11, 3, not_built=1, no_program=2, no_guard=4, guard_not_checked=1)


def test_a_target_without_a_guard_reports_every_attempt_under_no_guard() -> None:
    rows = dict(module().guard_coverage([trial(TT_CPU, kinds, run=run) for run, kinds in enumerate(TT_CPU_TRIALS, 1)]))
    cpu = rows["target cpu"]
    assert cpu.checked == 0 and cpu.not_checked == cpu.attempts == cpu.reasons["no-guard"] == 4
    assert cpu.rate == 1.0


def test_guard_coverage_does_not_depend_on_the_trial_order() -> None:
    trials = run_trials()
    forward = [(scope, counts(coverage)) for scope, coverage in module().guard_coverage(trials)]
    backward = [(scope, counts(coverage)) for scope, coverage in module().guard_coverage(trials[::-1])]
    assert forward == backward


# ---------------------------------------------------------------------------
# Markdown


def table_lines(markdown: str) -> list[list[str]]:
    """Return the cells of every Markdown table line, stripped."""
    return [[cell.strip() for cell in line.strip()[1:-1].split("|")] for line in markdown.splitlines()
            if line.strip().startswith("|")]


def test_coverage_cells_give_the_counts_and_the_rate_with_three_decimals() -> None:
    cells = module().coverage_cells(make(4, 1, not_built=1, no_program=1, guard_not_checked=1))
    assert cells == ["4", "1", "3", "0.750", "1", "1", "0", "1", "0"]
    assert module().coverage_cells(make(0, 0)) == ["0", "0", "0", "-", "0", "0", "0", "0", "0"]
    assert module().coverage_cells(make(3, 0, no_guard=3))[3] == "1.000"


def test_coverage_markdown_is_plain_ascii_with_one_column_per_code() -> None:
    rows = module().guard_coverage(run_trials())
    markdown = module().coverage_markdown(rows)
    assert markdown.isascii() and "\r" not in markdown and markdown.endswith("\n")
    assert markdown.startswith("## Guard coverage\n")
    legend = markdown.split("\n|", 1)[0]
    assert "attempt" in legend and NOT_RECORDED in legend, "the legend names the unit and the not-recorded count"
    header, rule, *body = table_lines(markdown)
    assert tuple(header) == COLUMNS
    assert set(rule) == {"---"} and len(rule) == len(COLUMNS)
    assert body == [[scope, *module().coverage_cells(coverage)] for scope, coverage in rows]
    assert [line[0] for line in body] == ["run", "target cpu", "target tt"]


def test_the_counts_ignore_every_other_attempt_field() -> None:
    plain = [attempt(index, kind) for index, kind in enumerate(("no-program", True, None))]
    busy = [dataclasses.replace(item, stage_reached="S0", response_text="SYNTHETIC reply", files={"main.cpp": "x\n"})
            for item in plain]
    assert counts(module().count_attempts(plain)) == counts(module().count_attempts(busy))
