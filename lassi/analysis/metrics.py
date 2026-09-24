"""Run metrics per arm and direction from the lassi profile's scores (task P2.8).

Bible: Evaluation Protocol (preamble; LASSI reproduction row; LASSI Paper
Metrics; Acceptance Criteria, the compile-only label), Reporting Rules,
Readability Standards (Run row), Repository Layout (`analysis/`).

metric_tables takes (Trial, Score) pairs, each Score the lassi profile's
(lassi.scoring.lassi_profile), groups them by the trial_id's arm and
direction segments, and gives one MetricTable per group, sorted by arm and
direction. It never rescores: every value comes from a Score's components
and from the trial's own fields (the last attempt's stage_reached,
final.corrections, provenance.device), so it needs no bench root.

Rows and the population each covers; every interval is Wilson 95% over the
row's own numerator and denominator:

- compile_rate (compiled = 1), run_rate (the last attempt reached S5),
  correct_rate (correct = 1, the automated oracle), cap_hit_rate
  (cap_hit = 1), fence_quirk_rate (fence_quirk > 0; the note gives the total
  hit count): all trials of the arm and direction.
- first_try_rate (first_try = 1) and sim_t_ge_0.6_rate (sim_t >= 0.6): the
  correct trials, the paper's denominator. The Sim-T row's note says that it
  reads the faithful, unrounded sim_t while the paper names no tokenizer
  (OQ-022).
- within_10pct_rate: always None and marked PLACEHOLDER; no timing profiler
  exists until P10.
- pass@1 and pass@3: the unbiased pass@k of each scenario (bench item) over
  its trials whose correct is not None, averaged over the scenarios. The
  numerator is the sum of the per-scenario values and the denominator the
  scenario count, so the interval treats the scenario as the unit. None
  when any scenario has fewer than k such trials.

A trial whose needed component is None is left out of the numerator and the
denominator, and the row's note counts it as excluded. A value that cannot
be computed is None, with no counts or interval, and a note saying why.

A group is a compile-stage reproduction when every trial's correct is None
with a note naming one (the lassi profile's compile-only note). Its
correctness rows are None with that label, and the table carries no paper
value (Acceptance Criteria). Every other table shows the paper's published
value and recount next to each row the paper reports, from
lassi.analysis.paper.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from lassi.analysis.paper import PaperCount, PaperMetric, PaperValues, load_paper, paper_interval
from lassi.analysis.stats import pass_at_k, wilson_interval
from lassi.core.interfaces import Score
from lassi.core.record import STAGES, Trial, parse_trial_id

# The label of a run whose correctness was never checked (bible Evaluation Protocol, Acceptance Criteria).
COMPILE_STAGE = "compile-stage reproduction"
NO_STAGE = "none"
STAGE_KEYS = (*STAGES, NO_STAGE)
CLEAN_RUN = "S5"
SIM_T_THRESHOLD = 0.6
PASS_AT = (1, 3)

# The lassi profile components the rows read.
CORRECT, FIRST_TRY, SIM_T = "correct", "first_try", "sim_t"
COMPILED, CAP_HIT, FENCE_QUIRK = "compiled", "cap_hit", "fence_quirk"
NEEDED = (CORRECT, FIRST_TRY, SIM_T, COMPILED, CAP_HIT, FENCE_QUIRK)

WITHIN_10PCT = "within_10pct_rate"
METRIC_NAMES = (
    "compile_rate", "run_rate", "correct_rate", "cap_hit_rate", "fence_quirk_rate",
    "first_try_rate", "sim_t_ge_0.6_rate", WITHIN_10PCT, "pass@1", "pass@3",
)
# Rows a compile-stage reproduction never computes or compares with the paper.
CORRECTNESS_ROWS = frozenset({"correct_rate", "first_try_rate", "sim_t_ge_0.6_rate", "pass@1", "pass@3"})

ALL_TRIALS = "over all trials of the arm and direction"
CORRECT_TRIALS = "over the correct trials, the paper's denominator"
WITHIN_NOTE = (
    "PLACEHOLDER: not measured; no timing profiler exists until P10, and the within-10% rule is open "
    "(Evaluation Protocol, LASSI Paper Metrics)"
)
COMPILE_STAGE_NOTE = (
    f"{COMPILE_STAGE}: no program ran, so correctness is not computed and is never compared with the paper"
)
# The paper names no tokenizer for its Sim-T (bible Evaluation Protocol, LASSI Paper Metrics, [OPEN]; OQ-022).
SIM_T_OPEN_NOTE = (
    "reads the faithful sim_t (Python tokenize), unrounded; the paper does not state its Sim-T tokenizer "
    "([OPEN], OQ-022) and prints Sim-T to two decimals"
)

Pair = tuple[Trial, Score]


@dataclass(frozen=True)
class MetricRow:
    """One metric of a table: its value, counts, Wilson 95% interval, the paper's values, and a note.

    `numerator` is an int for a rate and a float (the sum of per-scenario
    values) for pass@k. A None value has no counts and no interval.
    `paper_published` and `paper_recount` are `<count>/<denominator>` texts
    with each reading's rate and Wilson 95% interval, None where the paper
    reports no value or the table is a compile-stage reproduction.
    `placeholder` marks a value not yet measured.
    """

    name: str
    value: float | None
    numerator: int | float | None
    denominator: int | None
    wilson_low: float | None
    wilson_high: float | None
    paper_published: str | None = None
    paper_recount: str | None = None
    paper_cite: str | None = None
    note: str = ""
    placeholder: bool = False


@dataclass(frozen=True)
class MetricTable:
    """The metrics of one arm and direction.

    `stage_reached` maps S0 to S5 and `none` (no attempt), in that order,
    to the count of trials whose last attempt reached it. `corrections`
    maps each final.corrections value that occurs, ascending, to its count
    of trials. `trials_per_scenario` maps each scenario, `<bench>/<item>`,
    to its trial count. `devices` lists the trials' provenance devices.
    """

    arm: str
    direction: str
    compile_only: bool
    trials: int
    trials_per_scenario: Mapping[str, int]
    devices: tuple[str, ...]
    rows: tuple[MetricRow, ...]
    stage_reached: Mapping[str, int]
    corrections: Mapping[int, int]


def _none_row(name: str, note: str, *, placeholder: bool = False) -> MetricRow:
    """Return a row whose value cannot be computed or was not measured."""
    return MetricRow(name=name, value=None, numerator=None, denominator=None, wilson_low=None, wilson_high=None,
                     note=note, placeholder=placeholder)


def _flag(score: Score, component: str, test: Callable[[float], bool]) -> bool | None:
    """Return test(component value), or None when the component is None."""
    value = score.components[component]
    return None if value is None else test(value)


def _is_one(value: float) -> bool:
    """Return True when a 0/1 component is 1."""
    return value == 1.0


def _rate(name: str, flags: Sequence[bool | None], what: str, component: str) -> MetricRow:
    """Return the rate of True among `flags`; a None flag is excluded and counted in the note."""
    scored = [flag for flag in flags if flag is not None]
    excluded = len(flags) - len(scored)
    notes = [what, *([f"{excluded} excluded ({component} is None)"] if excluded else [])]
    if not scored:
        return _none_row(name, "; ".join(["not computed: no trial in the population", *notes]))
    count, total = sum(scored), len(scored)
    low, high = wilson_interval(count, total)
    return MetricRow(name=name, value=count / total, numerator=count, denominator=total, wilson_low=low,
                     wilson_high=high, note="; ".join(notes))


def _trial_rows(pairs: Sequence[Pair]) -> list[MetricRow]:
    """Return the rates over every trial: compile, run, correct, cap hit, and fence quirk."""
    ran = [bool(trial.attempts) and trial.attempts[-1].stage_reached == CLEAN_RUN for trial, _ in pairs]
    hits = sum(score.components[FENCE_QUIRK] or 0.0 for _, score in pairs)
    return [
        _rate("compile_rate", [_flag(s, COMPILED, _is_one) for _, s in pairs], f"compiled = 1, {ALL_TRIALS}", COMPILED),
        _rate("run_rate", ran, f"last attempt at {CLEAN_RUN}, {ALL_TRIALS}", "stage_reached"),
        _rate("correct_rate", [_flag(s, CORRECT, _is_one) for _, s in pairs],
              f"correct = 1 (automated oracle), {ALL_TRIALS}", CORRECT),
        _rate("cap_hit_rate", [_flag(s, CAP_HIT, _is_one) for _, s in pairs], f"cap_hit = 1, {ALL_TRIALS}", CAP_HIT),
        _rate("fence_quirk_rate", [_flag(s, FENCE_QUIRK, lambda value: value > 0) for _, s in pairs],
              f"fence_quirk > 0, {ALL_TRIALS}; {hits:g} fence-quirk hits in total", FENCE_QUIRK),
    ]


def _correct_trial_rows(pairs: Sequence[Pair]) -> list[MetricRow]:
    """Return first try and Sim-T >= 0.6, each a share of the correct trials."""
    correct = [score for _, score in pairs if _flag(score, CORRECT, _is_one)]
    threshold = f"{SIM_T} >= {SIM_T_THRESHOLD:g}"
    return [
        _rate("first_try_rate", [_flag(s, FIRST_TRY, _is_one) for s in correct], f"first_try = 1, {CORRECT_TRIALS}",
              FIRST_TRY),
        _rate("sim_t_ge_0.6_rate", [_flag(s, SIM_T, lambda value: value >= SIM_T_THRESHOLD) for s in correct],
              f"{threshold}, {CORRECT_TRIALS}; {SIM_T_OPEN_NOTE}", SIM_T),
    ]


def scenario_key(trial: Trial) -> str:
    """Return the scenario of a trial within its arm and direction: `<bench>/<item>` of its trial_id."""
    parsed = parse_trial_id(trial.trial_id)
    return f"{parsed.bench}/{parsed.item}"


def _pass_at_k_row(k: int, pairs: Sequence[Pair]) -> MetricRow:
    """Return pass@k: the mean over scenarios of each scenario's pass@k over its trials whose correct is set."""
    name = f"pass@{k}"
    outcomes: dict[str, list[bool]] = {}
    excluded = 0
    for trial, score in pairs:
        found = outcomes.setdefault(scenario_key(trial), [])
        flag = _flag(score, CORRECT, _is_one)
        if flag is None:
            excluded += 1
        else:
            found.append(flag)
    notes = [f"{excluded} excluded ({CORRECT} is None)"] if excluded else []
    short = sorted(len(found) for found in outcomes.values() if len(found) < k)
    if short:
        why = f"not computed: {len(short)} scenario(s) have fewer than k = {k} scored trials (smallest n = {short[0]})"
        return _none_row(name, "; ".join([why, *notes]))
    values = [pass_at_k(len(found), sum(found), k) for found in outcomes.values()]
    total, scenarios = float(sum(value for value in values if value is not None)), len(values)
    low, high = wilson_interval(total, scenarios)
    what = f"unbiased pass@{k} per scenario over its scored trials, mean over {scenarios} scenario(s)"
    notes = [what, "Wilson interval over the scenario count", *notes]
    return MetricRow(name=name, value=total / scenarios, numerator=total, denominator=scenarios, wilson_low=low,
                     wilson_high=high, note="; ".join(notes))


def paper_text(count: PaperCount) -> str:
    """Return a paper reading as `<count>/<denominator> = <rate> (Wilson 95% <low> to <high>)`."""
    low, high = paper_interval(count)
    return f"{count.fraction()} = {count.count / count.denominator:.3f} (Wilson 95% {low:.3f} to {high:.3f})"


def _with_paper(row: MetricRow, metric: PaperMetric | None) -> MetricRow:
    """Return `row` with the paper's published value, recount (then its alternate), and cite, when there is one."""
    if metric is None:
        return row
    recount = paper_text(metric.recount)
    if metric.recount_alternate is not None:
        recount = f"{recount}; alternate {paper_text(metric.recount_alternate)}"
    return dataclasses.replace(row, paper_published=paper_text(metric.published), paper_recount=recount,
                               paper_cite=metric.cite)


def is_compile_only(scores: Sequence[Score]) -> bool:
    """Return True when every score's correct is None with a note naming a compile-stage reproduction."""
    return bool(scores) and all(
        score.components[CORRECT] is None and COMPILE_STAGE in score.notes.get(CORRECT, "").lower()
        for score in scores
    )


def _rows(pairs: Sequence[Pair], direction: str, compile_only: bool, paper: PaperValues) -> tuple[MetricRow, ...]:
    """Return the table's rows in METRIC_NAMES order, labeled or with the paper's values."""
    rows = [
        *_trial_rows(pairs),
        *_correct_trial_rows(pairs),
        _none_row(WITHIN_10PCT, WITHIN_NOTE, placeholder=True),
        *(_pass_at_k_row(k, pairs) for k in PASS_AT),
    ]
    if compile_only:
        rows = [_none_row(row.name, COMPILE_STAGE_NOTE) if row.name in CORRECTNESS_ROWS else row for row in rows]
    else:
        rows = [_with_paper(row, paper.metric(direction, row.name)) for row in rows]
    return tuple(rows)


def _table(arm: str, direction: str, pairs: Sequence[Pair], paper: PaperValues) -> MetricTable:
    """Return the MetricTable of one arm and direction; `pairs` are sorted by trial_id."""
    compile_only = is_compile_only([score for _, score in pairs])
    stages = Counter(trial.attempts[-1].stage_reached if trial.attempts else NO_STAGE for trial, _ in pairs)
    corrections = Counter(trial.final.corrections for trial, _ in pairs)
    scenarios = Counter(scenario_key(trial) for trial, _ in pairs)
    return MetricTable(
        arm=arm,
        direction=direction,
        compile_only=compile_only,
        trials=len(pairs),
        trials_per_scenario=MappingProxyType(dict(sorted(scenarios.items()))),
        devices=tuple(sorted({trial.provenance.device for trial, _ in pairs})),
        rows=_rows(pairs, direction, compile_only, paper),
        stage_reached=MappingProxyType({key: stages.get(key, 0) for key in STAGE_KEYS}),
        corrections=MappingProxyType(dict(sorted(corrections.items()))),
    )


def _check_pairs(pairs: Sequence[Pair]) -> None:
    """Raise ValueError when a trial_id repeats or a Score lacks a component the rows read."""
    repeated = sorted(trial_id for trial_id, count in Counter(t.trial_id for t, _ in pairs).items() if count > 1)
    if repeated:
        raise ValueError(f"trial_id appears more than once: {repeated}")
    for trial, score in pairs:
        missing = [name for name in NEEDED if name not in score.components]
        if missing:
            raise ValueError(f"the score of {trial.trial_id} lacks the lassi component(s) {', '.join(missing)}")


def metric_tables(pairs: Sequence[Pair], paper: PaperValues | None = None) -> list[MetricTable]:
    """Return one MetricTable per arm and direction of `pairs`, sorted by arm and direction.

    `pairs` holds (Trial, Score) with the lassi profile's Score of each
    trial; `paper` defaults to load_paper(). The result does not depend on
    the order of `pairs`. Raises ValueError when a trial_id repeats or a
    Score lacks a component the rows read.
    """
    values = load_paper() if paper is None else paper
    _check_pairs(pairs)
    groups: dict[tuple[str, str], list[Pair]] = {}
    for trial, score in pairs:
        parsed = parse_trial_id(trial.trial_id)
        groups.setdefault((parsed.arm, parsed.direction), []).append((trial, score))
    return [
        _table(arm, direction, sorted(members, key=lambda pair: pair[0].trial_id), values)
        for (arm, direction), members in sorted(groups.items())
    ]
