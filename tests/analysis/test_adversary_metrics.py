"""Tests for the adversary metrics per arm, suite, adversary, and direction (task P16.6).

Bible: Evaluation Protocol (the Judges and adversaries row: adversary
confirmed-divergence rate, invalid-input rejection rate; Run Metrics: Wilson
95% score intervals, a rate over an empty population is null with a note and
has no counts and no interval), Result Record (Trial.adversary,
AdversaryRecord), Readability Standards (Run row; machine formats mirror
human formats), Agent Rule 14 (measured beats judged). Plan:
plans/p16-adversarial.md, P16.6.

The contract these tests fix:

- `lassi.analysis.adversary` names the three metrics: DIVERGENCE_RATE
  ("confirmed_divergence_rate"), REJECTION_RATE
  ("invalid_input_rejection_rate"), and FINDING_COUNT
  ("contract_finding_count"), in that order in ADVERSARY_METRICS. None of
  them is a row name of the run metric tables (lassi.analysis.metrics).
- `adversary_tables(trials) -> list[AdversaryTable]` reads only trial
  records and groups the trials the stage took (Trial.adversary set) by arm
  (the trial_id's arm segment; Evaluation Protocol preamble: every result is
  reported per arm and direction), suite (bench_item.suite), adversary
  (AdversaryRecord.agent), and direction (bench_item.direction), one
  AdversaryTable per group, sorted by those four; the result does not
  depend on the order of `trials`. A table has `arm`, `suite`, `adversary`,
  `direction`, `trials` (the trials the stage took), and `rows`, one per
  ADVERSARY_METRICS name in that order, each with `name`, `value`,
  `numerator`, `denominator`, `wilson_low`, `wilson_high`, and `note`:
  - confirmed_divergence_rate: the trials with at least one Divergence over
    the trials the stage took, with the Wilson 95% interval of that count;
  - invalid_input_rejection_rate: the refused proposals (the sum of
    AdversaryRecord.refusals) over the proposals (AdversaryRecord.proposals),
    pooled over the group's trials, with its Wilson 95% interval; with no
    proposal (0 of 0) it is null with a note and has no counts and no
    interval;
  - contract_finding_count: the number of ContractFindings over the group's
    trials, its value the count (0 included), with no interval.
- A trial the stage did not take (Trial.adversary null) counts in neither
  denominator: it changes no value, count, or interval of any table, and
  trials the stage did not take make no table of their own (the record of
  such a trial names no adversary).
- No judged value enters them (Agent Rule 14): only an AdversaryRecord's
  proposals, refusals, divergences, and contract findings move a value;
  scores, the adversary's config, and every other field of the trial do not.
- `adversary_markdown(tables) -> str` is plain ASCII with one heading line
  per table naming its arm, suite, adversary, and direction, and one Markdown
  table line per row whose first cell is the row name, then the value, the
  count, and the Wilson interval, as the run metric tables show them
  (lassi.analysis.tables): a rate as three decimals, `<numerator>/<denominator>`,
  and `<low> to <high>` with three decimals; a null rate as "-" in all three;
  the finding count as the integer, with "-" for its interval.
- `adversary_rows(tables)` gives one Parquet row per table and metric with
  arm, suite, adversary, direction, trials, and the row's name, value (a
  double, the count's included), value_int (the count's value as an int, so
  the machine copy shows it as run.md does; null on a rate row), numerator,
  denominator, wilson_low, wilson_high, and note;
  `write_adversary_parquet(tables, out_dir)` writes them under `out_dir`, and
  `read_adversary_parquet(out_dir)` reads back exactly adversary_rows(tables),
  [] when nothing was written. Writing keeps every other file in `out_dir`,
  the trials' own Parquet table `adversary` (lassi.core.parquet) included.

Oracles: every count is worked by hand from the hand-built records, as the
comments show. Boundary intervals use the Wilson closed forms (at 0 of n the
upper bound is z^2 / (n + z^2); at n of n the lower bound is n / (n + z^2)),
each bound to 1e-12, as tests/analysis/test_analysis_stats.py checks them
(wilson_interval's clamped bound is not always exactly 0.0 or 1.0: 0 of 3
gives a lower bound of about 5.6e-17); the others use
lassi.analysis.stats.wilson_interval, which
tests/analysis/test_analysis_stats.py checks against Newcombe 1998.

Every trial and record here is SYNTHETIC and hand-built; no value in this
module is a measurement.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.analysis.metrics import METRIC_NAMES as RUN_METRIC_NAMES
from lassi.analysis.stats import wilson_interval
from lassi.core.interfaces import Sampling
from lassi.core.record import (
    AdversaryRecord,
    Attempt,
    BenchItem,
    ContractFinding,
    Divergence,
    Final,
    ModelInfo,
    OutputStats,
    Provenance,
    ScoreBreakdown,
    Trial,
    make_trial_id,
)

MODULE = "lassi.analysis.adversary"
DIVERGENCE_RATE = "confirmed_divergence_rate"
REJECTION_RATE = "invalid_input_rejection_rate"
FINDING_COUNT = "contract_finding_count"
NAMES = (DIVERGENCE_RATE, REJECTION_RATE, FINDING_COUNT)
Z95 = 1.959963984540054
Z2 = Z95 * Z95
EXACT = 1e-12

PROJECT, MODEL, OTHER_ARM = "adv-metrics", "scripted-fixture", "b0-fixture"
SUITE_A, SUITE_B = "adv-suite-a", "adv-suite-b"
FUZZER, LLM = "fuzzer", "llm"
OMP_CPP, CPP_CPP = "omp-cpp", "cpp-cpp"
# SYNTHETIC digest of a proposed input file; no file holds it.
SHA = "c" * 64
FAILED = OutputStats(name="y", pcc=0.0, max_abs=1000.0, max_ulp=7, passed=False, note="SYNTHETIC: y differs")
FINDING = "SYNTHETIC: the cpp target reference run ended with exit status 139"
ROW_KEYS = ("arm", "suite", "adversary", "direction", "trials", "name", "value", "value_int", "numerator",
            "denominator", "wilson_low", "wilson_high", "note")
GROUP_KEYS = ("arm", "suite", "adversary", "direction")


def adversary() -> ModuleType:
    """Import lassi.analysis.adversary; fail clearly while task P16.6 has not added it."""
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as error:
        if error.name != MODULE:
            raise
        pytest.fail(f"task P16.6 adds {MODULE}: {error}")


def need(name: str) -> Any:
    """Return the module's attribute `name`, failing with the task's name when it does not exist yet."""
    module = adversary()
    assert hasattr(module, name), f"{MODULE} has no {name} (task P16.6)"
    return getattr(module, name)


# ---------------------------------------------------------------------------
# Hand-built records


def record(
    *,
    agent: str = FUZZER,
    proposals: int = 0,
    refusals: Mapping[str, int] | None = None,
    divergences: int = 0,
    findings: int = 0,
) -> AdversaryRecord:
    """Return a SYNTHETIC AdversaryRecord: refused proposals first, then the divergences, then the findings."""
    refused = dict(refusals or {})
    first = sum(refused.values())
    assert proposals >= first + divergences + findings, "precondition: every entry names its own proposal"
    found = [
        Divergence(proposal=index, inputs={"x": SHA}, stats=[FAILED], candidate_failure=None)
        for index in range(first, first + divergences)
    ]
    start = first + divergences
    finds = [
        ContractFinding(proposal=index, inputs={"x": SHA}, message=FINDING) for index in range(start, start + findings)
    ]
    return AdversaryRecord(
        agent=agent,
        config={"budget": {"inputs": max(proposals, 1)}, "seed": 7},
        proposals=proposals,
        refusals=refused,
        runs=2 * (proposals - first),
        wall_s=1.5,
        divergences=found,
        contract_findings=finds,
    )


def trial(
    item: str, found: AdversaryRecord | None, *, suite: str = SUITE_A, direction: str = OMP_CPP, arm: str = MODEL
) -> Trial:
    """Return a SYNTHETIC trial of `item` in `arm` with `found` as its adversary record (None: not taken)."""
    return Trial(
        trial_id=make_trial_id(PROJECT, arm, suite, direction, item, 1),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=None, dirty=None, device=None, sdk=None, date="2026-10-19T00:00:00+00:00"),
        bench_item=BenchItem(suite=suite, item=item, split="unassigned", direction=direction),
        model=ModelInfo(backend="scripted", id=MODEL, sampling=Sampling(0.2, 0.9, 8)),
        adversary=found,
    )


def taken(*records: AdversaryRecord, suite: str = SUITE_A, direction: str = OMP_CPP) -> list[Trial]:
    """Return one trial per record, items item0, item1, ... of one suite and direction."""
    return [trial(f"item{index}", found, suite=suite, direction=direction) for index, found in enumerate(records)]


def only_table(trials: Sequence[Trial]) -> Any:
    """Return the one table adversary_tables gives for `trials`."""
    tables = need("adversary_tables")(trials)
    assert len(tables) == 1, f"expected one table, got {len(tables)}"
    return tables[0]


def row(table: Any, name: str) -> Any:
    """Return the table's one row named `name`."""
    found = [entry for entry in table.rows if entry.name == name]
    assert len(found) == 1, f"the table has {len(found)} rows named {name!r}"
    return found[0]


def numbers(entry: Any) -> tuple[Any, ...]:
    """Return a row's value, numerator, denominator, and interval: everything but its note."""
    return (entry.value, entry.numerator, entry.denominator, entry.wilson_low, entry.wilson_high)


def assert_rate(entry: Any, count: int, total: int, interval: tuple[float, float]) -> None:
    """Assert that `entry` is the rate count / total with the given Wilson interval."""
    assert (entry.numerator, entry.denominator) == (count, total), f"{entry.name}: counts"
    assert entry.value == pytest.approx(count / total, abs=EXACT), f"{entry.name}: value"
    assert entry.wilson_low == pytest.approx(interval[0], abs=EXACT), f"{entry.name}: Wilson low"
    assert entry.wilson_high == pytest.approx(interval[1], abs=EXACT), f"{entry.name}: Wilson high"


# ---------------------------------------------------------------------------
# Names and groups


def test_the_module_names_its_three_metrics_in_order() -> None:
    assert need("ADVERSARY_METRICS") == NAMES
    assert (need("DIVERGENCE_RATE"), need("REJECTION_RATE"), need("FINDING_COUNT")) == NAMES
    assert not set(NAMES) & set(RUN_METRIC_NAMES), "an adversary metric shares a name with a run metric table row"


def test_one_table_per_suite_adversary_and_direction_sorted() -> None:
    trials = [
        trial("i1", record(agent=LLM, proposals=1), suite=SUITE_B, direction=OMP_CPP),
        trial("i2", record(agent=FUZZER, proposals=1), suite=SUITE_A, direction=OMP_CPP),
        trial("i3", record(agent=FUZZER, proposals=1), suite=SUITE_A, direction=CPP_CPP),
        trial("i4", record(agent=LLM, proposals=1), suite=SUITE_A, direction=OMP_CPP),
        trial("i5", record(agent=FUZZER, proposals=1), suite=SUITE_A, direction=OMP_CPP),
    ]
    tables = need("adversary_tables")(trials)
    keys = [(table.arm, table.suite, table.adversary, table.direction, table.trials) for table in tables]
    assert keys == [
        (MODEL, SUITE_A, FUZZER, CPP_CPP, 1),
        (MODEL, SUITE_A, FUZZER, OMP_CPP, 2),
        (MODEL, SUITE_A, LLM, OMP_CPP, 1),
        (MODEL, SUITE_B, LLM, OMP_CPP, 1),
    ]
    for table in tables:
        assert tuple(entry.name for entry in table.rows) == NAMES, "one row per metric, in ADVERSARY_METRICS order"


def test_each_arm_gets_its_own_tables_sorted_first_by_arm() -> None:
    # Same suite, adversary, and direction in two arms: the scripted-fixture arm diverged on 1 of 1 trial, the
    # b0-fixture arm on 0 of 2; pooled they would read 1 of 3.
    trials = [
        trial("i1", record(proposals=2, divergences=1), arm=MODEL),
        trial("i2", record(proposals=2), arm=OTHER_ARM),
        trial("i3", record(proposals=2), arm=OTHER_ARM),
    ]
    tables = need("adversary_tables")(trials)
    assert [(table.arm, table.trials) for table in tables] == [(OTHER_ARM, 2), (MODEL, 1)]
    assert [row(table, DIVERGENCE_RATE).numerator for table in tables] == [0, 1], "an arm's trials stay in its table"


def test_the_tables_do_not_depend_on_the_order_of_the_trials() -> None:
    trials = [
        trial("i1", record(proposals=3, refusals={"dtype": 1}, divergences=1)),
        trial("i2", record(agent=LLM, proposals=2, findings=1)),
        trial("i3", record(proposals=2), direction=CPP_CPP),
        trial("i4", None),
    ]
    build = need("adversary_tables")
    assert build(trials) == build(list(reversed(trials)))


# ---------------------------------------------------------------------------
# confirmed_divergence_rate


def test_the_divergence_rate_counts_each_trial_with_a_divergence_once() -> None:
    # item0 has two divergences and counts once, item1 none, item2 one: 2 of 3 trials.
    records = (record(proposals=4, divergences=2), record(proposals=4), record(proposals=4, divergences=1))
    table = only_table(taken(*records))
    assert table.trials == 3
    assert_rate(row(table, DIVERGENCE_RATE), 2, 3, wilson_interval(2, 3))


@pytest.mark.parametrize(
    ("diverged", "total"),
    [(0, 3), (3, 3), (1, 1), (0, 1)],
    ids=["0-of-n", "n-of-n", "one-trial-diverged", "one-trial-agreed"],
)
def test_the_divergence_rate_boundaries_are_exact(diverged: int, total: int) -> None:
    records = [record(proposals=2, divergences=1 if index < diverged else 0) for index in range(total)]
    entry = row(only_table(taken(*records)), DIVERGENCE_RATE)
    if diverged == 0:
        interval = (0.0, Z2 / (total + Z2))
    else:
        interval = (total / (total + Z2), 1.0)
    assert_rate(entry, diverged, total, interval)
    assert entry.value == diverged / total, "a boundary rate is exactly 0 or 1"


def test_a_taken_trial_whose_adversary_proposed_nothing_counts_in_the_divergence_denominator() -> None:
    # item0 proposed nothing; item1 diverged: 1 of 2 trials the stage took.
    entry = row(only_table(taken(record(proposals=0), record(proposals=1, divergences=1))), DIVERGENCE_RATE)
    assert_rate(entry, 1, 2, wilson_interval(1, 2))


def test_a_proposal_that_found_only_a_contract_finding_is_no_divergence() -> None:
    entry = row(only_table(taken(record(proposals=3, findings=2))), DIVERGENCE_RATE)
    assert_rate(entry, 0, 1, (0.0, Z2 / (1 + Z2)))


# ---------------------------------------------------------------------------
# invalid_input_rejection_rate


def test_the_rejection_rate_pools_refused_proposals_over_proposals() -> None:
    # item0: 7 proposals, 2 + 1 refused; item1: 5 proposals, none refused; item2: 3 proposals, 3 refused.
    records = (
        record(proposals=7, refusals={"out-of-range": 2, "dtype": 1}, divergences=1),
        record(proposals=5),
        record(proposals=3, refusals={"nan": 2, "shape": 1}),
    )
    entry = row(only_table(taken(*records)), REJECTION_RATE)
    assert_rate(entry, 6, 15, wilson_interval(6, 15))


@pytest.mark.parametrize(
    ("proposals", "refusals"),
    [
        ((5,), ({},)),
        ((3, 1), ({"out-of-range": 2, "dtype": 1}, {"unreadable": 1})),
        ((1,), ({"missing-input": 1},)),
        ((1,), ({},)),
    ],
    ids=["0-of-n", "n-of-n", "one-proposal-refused", "one-proposal-accepted"],
)
def test_the_rejection_rate_boundaries_are_exact(
    proposals: tuple[int, ...], refusals: tuple[dict[str, int], ...]
) -> None:
    records = [record(proposals=count, refusals=refused) for count, refused in zip(proposals, refusals, strict=True)]
    entry = row(only_table(taken(*records)), REJECTION_RATE)
    refused, total = sum(sum(item.values()) for item in refusals), sum(proposals)
    interval = (0.0, Z2 / (total + Z2)) if refused == 0 else (total / (total + Z2), 1.0)
    assert refused in (0, total), "precondition: a boundary case"
    assert_rate(entry, refused, total, interval)
    assert entry.value == refused / total, "a boundary rate is exactly 0 or 1"


def test_the_rejection_rate_over_no_proposal_is_null_with_a_note() -> None:
    # Two trials the stage took whose adversary proposed nothing: 0 of 0.
    entry = row(only_table(taken(record(proposals=0), record(proposals=0))), REJECTION_RATE)
    assert numbers(entry) == (None, None, None, None, None), "0 of 0 has no value, no counts, and no interval"
    assert isinstance(entry.note, str) and entry.note.strip(), "a null rate says why in its note"


# ---------------------------------------------------------------------------
# contract_finding_count


def test_the_contract_finding_count_sums_the_findings_of_the_group() -> None:
    # item0 one finding, item1 two, item2 none: 3.
    records = (record(proposals=2, findings=1), record(proposals=3, findings=2, divergences=1), record(proposals=1))
    entry = row(only_table(taken(*records)), FINDING_COUNT)
    assert entry.value == 3
    assert (entry.wilson_low, entry.wilson_high) == (None, None), "a count has no interval"


def test_a_group_with_no_finding_counts_zero_not_null() -> None:
    entry = row(only_table(taken(record(proposals=2, divergences=1))), FINDING_COUNT)
    assert entry.value == 0 and entry.value is not None


# ---------------------------------------------------------------------------
# Trials the stage did not take


def test_a_trial_the_stage_did_not_take_counts_in_neither_denominator() -> None:
    took = taken(record(proposals=4, refusals={"dtype": 1}, divergences=1), record(proposals=2, findings=1))
    skipped = [trial("skipped0", None), trial("skipped1", None)]
    alone, mixed = only_table(took), only_table([*skipped[:1], *took, *skipped[1:]])
    assert mixed.trials == alone.trials == 2, "only the trials the stage took are counted"
    for name in NAMES:
        assert numbers(row(mixed, name)) == numbers(row(alone, name)), f"{name}: a trial not taken changed it"


def test_trials_the_stage_did_not_take_make_no_table() -> None:
    assert need("adversary_tables")([trial("skipped0", None), trial("skipped1", None)]) == []
    assert need("adversary_tables")([]) == []


# ---------------------------------------------------------------------------
# Agent Rule 14: no judged value enters them


def judged_looking(source: Trial) -> Trial:
    """Return `source` with every field but the record's four counted ones changed, scores and claims included."""
    found = source.adversary
    assert found is not None
    attempt = Attempt(index=0, stage_reached="S5", score=ScoreBreakdown(components={"judge": 1.0}, scalar=1.0))
    config = {**found.config, "claimed_divergences": 9, "verdict": "diverges"}
    changed = dataclasses.replace(found, config=config, runs=found.runs + 5, wall_s=99.0)
    final = Final(stage_reached="S5", alignment=1.0, score=0.0, corrections=2, wall_s=120.0)
    return dataclasses.replace(source, attempts=[attempt], final=final, adversary=changed)


def test_no_judged_value_enters_the_metrics() -> None:
    plain = taken(record(proposals=5, refusals={"dtype": 2}, divergences=1, findings=1), record(proposals=3))
    judged = [judged_looking(item) for item in plain]
    base, other = only_table(plain), only_table(judged)
    for name in NAMES:
        assert numbers(row(other, name)) == numbers(row(base, name)), f"{name} moved with a field it must not read"


# ---------------------------------------------------------------------------
# Markdown


def table_line(text: str, name: str) -> list[str]:
    """Return the cells of the one Markdown table line of `text` whose first cell is `name`."""
    found = [line for line in text.split("\n") if line.startswith(f"| {name} |")]
    assert len(found) == 1, f"the Markdown holds {len(found)} lines for {name!r}"
    return [cell.strip() for cell in found[0].strip()[1:-1].split("|")]


def test_the_markdown_shows_each_table_and_row_in_the_run_tables_form() -> None:
    # 1 of 2 trials diverged; 3 of 9 proposals refused; 2 contract findings.
    records = (record(proposals=5, refusals={"dtype": 3}, divergences=1, findings=1), record(proposals=4, findings=1))
    table = only_table(taken(*records, suite=SUITE_B, direction=CPP_CPP))
    text = need("adversary_markdown")([table])
    assert text.isascii(), "the Markdown is plain ASCII"
    headings = [line for line in text.split("\n") if line.startswith("#")]
    named = [line for line in headings if all(part in line for part in (MODEL, SUITE_B, FUZZER, CPP_CPP))]
    assert len(named) == 1, f"expected one heading naming the arm, suite, adversary, and direction: {headings}"
    for name, count, total in ((DIVERGENCE_RATE, 1, 2), (REJECTION_RATE, 3, 9)):
        low, high = wilson_interval(count, total)
        assert table_line(text, name)[1:4] == [f"{count / total:.3f}", f"{count}/{total}", f"{low:.3f} to {high:.3f}"]
    cells = table_line(text, FINDING_COUNT)
    assert (cells[1], cells[3]) == ("2", "-"), "the count as an integer, with no interval"


def test_the_markdown_shows_a_null_rate_as_dashes() -> None:
    text = need("adversary_markdown")([only_table(taken(record(proposals=0)))])
    assert table_line(text, REJECTION_RATE)[1:4] == ["-", "-", "-"]


# ---------------------------------------------------------------------------
# Parquet


def mixed_tables() -> list[Any]:
    """Return the tables of SYNTHETIC trials over two suites, both adversaries, and both directions.

    The suite-a fuzzer cpp-cpp table's rejection rate is null (0 of 0).
    """
    trials = [
        *taken(record(proposals=3, refusals={"dtype": 1}, divergences=1), record(proposals=0)),
        trial("c0", record(proposals=0), direction=CPP_CPP),
        trial("b0", record(agent=LLM, proposals=2, findings=1), suite=SUITE_B, direction=CPP_CPP),
        trial("b1", None, suite=SUITE_B),
    ]
    tables = need("adversary_tables")(trials)
    assert any(row(table, REJECTION_RATE).value is None for table in tables), "precondition: a null rate"
    return tables


def test_the_parquet_rows_hold_every_row_of_every_table() -> None:
    tables = mixed_tables()
    rows = need("adversary_rows")(tables)
    assert len(rows) == len(tables) * len(NAMES)
    for item in rows:
        missing = [key for key in ROW_KEYS if key not in item]
        assert not missing, f"a Parquet row lacks {missing}"
    for table in tables:
        for entry in table.rows:
            key = (table.arm, table.suite, table.adversary, table.direction, entry.name)
            found = [item for item in rows if tuple(item[name] for name in GROUP_KEYS) + (item["name"],) == key]
            assert len(found) == 1, f"{key}: {len(found)} Parquet rows"
            stored = found[0]
            assert stored["trials"] == table.trials
            assert (stored["value"], stored["numerator"], stored["denominator"]) == numbers(entry)[:3], key
            assert (stored["wilson_low"], stored["wilson_high"]) == numbers(entry)[3:], key
            assert stored["note"] == entry.note, key


def test_the_parquet_holds_the_count_as_an_int_and_no_rate_as_one(tmp_path: Path) -> None:
    tables = mixed_tables()
    need("write_adversary_parquet")(tables, tmp_path)
    for stored in need("read_adversary_parquet")(tmp_path):
        if stored["name"] == FINDING_COUNT:
            assert type(stored["value_int"]) is int, "run.md shows the count as an integer; so does the Parquet"
            assert stored["value_int"] == stored["value"], "value_int is the count itself"
        else:
            assert stored["value_int"] is None, f"{stored['name']}: a rate has no int value"
    rows = [item for item in need("adversary_rows")(tables) if item["name"] == FINDING_COUNT]
    counts = {(item["arm"], item["suite"], item["direction"]): item["value_int"] for item in rows}
    assert counts == {(MODEL, SUITE_A, OMP_CPP): 0, (MODEL, SUITE_A, CPP_CPP): 0, (MODEL, SUITE_B, CPP_CPP): 1}


def test_the_parquet_round_trips_and_keeps_the_other_tables(tmp_path: Path) -> None:
    out = tmp_path / "parquet"
    other = out / "adversary" / "part-0.parquet"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"SYNTHETIC stand-in for the trials' own adversary table")
    tables = mixed_tables()
    need("write_adversary_parquet")(tables, out)
    assert need("read_adversary_parquet")(out) == need("adversary_rows")(tables)
    assert other.read_bytes() == b"SYNTHETIC stand-in for the trials' own adversary table", "another table changed"


def test_reading_a_tree_without_the_table_gives_no_rows(tmp_path: Path) -> None:
    (tmp_path / "parquet").mkdir()
    assert need("read_adversary_parquet")(tmp_path / "parquet") == []
