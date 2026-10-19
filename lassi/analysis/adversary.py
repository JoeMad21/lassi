"""Adversary metrics per arm, suite, adversary, and direction from the trial records (task P16.6).

Bible: Evaluation Protocol (preamble: every result per arm and direction;
the Judges and adversaries row; Run Metrics), Result Record (Trial.adversary,
AdversaryRecord), Readability Standards (Run row; machine formats mirror
human formats), Design Principle 7, Agent Rule 14. Plan:
plans/p16-adversarial.md, P16.6. [DESIGN]

adversary_tables reads only the trials' records. It takes the trials the
adversary stage took (Trial.adversary set; lassi.core.adversary_stage),
groups them by arm (the trial_id's arm segment, as the run metric tables
group; one `lassi run` has one), suite (BenchItem.suite), adversary
(AdversaryRecord.agent, the adversary.kind it ran), and direction
(BenchItem.direction), and gives one AdversaryTable per group, sorted by
those four, so trials of several runs or models passed together never
merge across arms. A trial the stage did not take holds no record, so it
counts in no table and makes none of its own. The result does not depend
on the order of the trials.

Rows (ADVERSARY_METRICS, in this order), each interval the Wilson 95% score
interval (lassi.analysis.stats) over the row's own count:

- confirmed_divergence_rate: the trials with at least one confirmed
  divergence (binary_io failed an output of the candidate's run against the
  target reference's) over the trials the stage took, those whose adversary
  proposed nothing included; a trial counts once however many it has.
- invalid_input_rejection_rate: the proposals refused against the input
  contract before any run (the sum of AdversaryRecord.refusals, whose
  reasons are lassi.harness check_inputs's) over the proposals made,
  refused ones included, pooled over the group's trials. With no proposal
  it is null with a note, no counts, and no interval (Run Metrics, rules).
  A reply an adversary could not turn into proposals (an llm adversary's
  malformed reply, which PHASE-NOTES P16 hands to P16.4 for that adversary
  to count itself) is no proposal, so it counts in neither part. Known
  case: the interval treats each proposal as an independent draw, though
  the proposals of one trial come from one adversary on one item (the
  fuzzer's follow from its seed), so it overstates the precision.
- contract_finding_count: the contract findings of the group's trials (an
  accepted proposal on which the target reference's run failed or gave
  outputs the oracle cannot compare against; never a divergence), an int,
  0 when there are none; a count, so no fraction and no interval.

No judged value enters them (Agent Rule 14): only an AdversaryRecord's
proposals, refusals, divergences, and contract findings move a value, and
a divergence is one binary_io confirmed; the adversary's config, its runs
and wall time, the trial's scores, and every other field do not.

adversary_markdown renders the tables as the run metric tables show their
rows (lassi.analysis.tables): a rate with three decimals, the count as
`<numerator>/<denominator>`, the interval as `<low> to <high>`, and "-"
for each of the three when null; plain ASCII, names, counts, rates, and
notes only (OQ-018). write_adversary_parquet writes adversary_rows, one row
per table and metric, to `<out_dir>/adversary_metrics/part-0.parquet`, a
directory of its own, so the trials' own Parquet table `adversary`
(lassi.core.parquet) stays as written; read_adversary_parquet reads them
back, a tree without the table as no rows. Every row's `value` is a double,
the count's included; `value_int` holds the count's value as an int, as
run.md shows it, and is null on a rate row, so it also marks the count.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from lassi.analysis.guard_coverage import table_lines
from lassi.analysis.stats import wilson_interval
from lassi.core.record import AdversaryRecord, Trial, parse_trial_id

DIVERGENCE_RATE = "confirmed_divergence_rate"
REJECTION_RATE = "invalid_input_rejection_rate"
FINDING_COUNT = "contract_finding_count"
ADVERSARY_METRICS = (DIVERGENCE_RATE, REJECTION_RATE, FINDING_COUNT)
# The Parquet table of the adversary metrics: a directory of its own beside the trials' `adversary` table.
TABLE = "adversary_metrics"
PART = "part-0.parquet"
HEADING = "Adversary metrics"
HEADER = ("Metric", "Value", "Count", "Wilson 95%", "Note")
LEGEND = (
    "One table per arm, suite, adversary (adversary.kind), and direction, from the trials the adversary stage "
    "took; a trial it did not take holds no adversary record and counts in no table. The confirmed-divergence rate "
    "is the trials with at least one divergence binary_io confirmed over the trials the stage took. The "
    "invalid-input rejection rate is the proposals refused against the input contract (check_inputs's reasons) "
    "before any run over the proposals made, pooled over the table's trials; it is null, with no counts and no "
    "interval, when no input set was proposed. A reply an adversary could not turn into proposals, such as an llm "
    "adversary's malformed reply, is no proposal and counts in neither part. Each interval is Wilson 95% over the "
    "row's own count. Known case: the rejection rate's interval treats each proposal as an independent draw, "
    "though one trial's proposals come from one adversary on one item, so it overstates the precision. The "
    "contract-finding count is the accepted proposals on which the target reference's run failed or gave outputs "
    "the oracle cannot compare against, never a divergence; a count has no interval. No judged value enters them "
    "(Agent Rule 14): an adversary's claims are never recorded, and only the stage's checks and binary_io's "
    "verdicts are counted."
)
NO_TABLES = "The adversary stage took no trial, so no adversary metric was computed."

_SCHEMA = pa.schema(
    [
        ("arm", pa.string()),
        ("suite", pa.string()),
        ("adversary", pa.string()),
        ("direction", pa.string()),
        ("trials", pa.int64()),
        ("ordinal", pa.int64()),
        ("name", pa.string()),
        ("value", pa.float64()),
        ("value_int", pa.int64()),
        ("numerator", pa.int64()),
        ("denominator", pa.int64()),
        ("wilson_low", pa.float64()),
        ("wilson_high", pa.float64()),
        ("note", pa.string()),
    ]
)
_SORT_KEYS = ("arm", "suite", "adversary", "direction", "ordinal")


@dataclass(frozen=True)
class AdversaryRow:
    """One adversary metric: its value, counts, Wilson 95% interval, and a note.

    A rate's value is a float with its int numerator and denominator; a null
    rate has no counts and no interval. The contract-finding count's value
    is an int, with no counts and no interval.
    """

    name: str
    value: float | int | None
    numerator: int | None
    denominator: int | None
    wilson_low: float | None
    wilson_high: float | None
    note: str


@dataclass(frozen=True)
class AdversaryTable:
    """The adversary metrics of one arm, suite, adversary, and direction.

    `arm` is the trial_id's arm segment. `trials` counts the trials the
    stage took in the group; `rows` holds one AdversaryRow per
    ADVERSARY_METRICS name, in that order.
    """

    arm: str
    suite: str
    adversary: str
    direction: str
    trials: int
    rows: tuple[AdversaryRow, ...]


# ---------------------------------------------------------------------------
# Tables


def _rate(name: str, count: int, total: int, what: str, empty: str) -> AdversaryRow:
    """Return the rate count / total with its Wilson 95% interval; null with `empty` as its note when total is 0."""
    if total == 0:
        return AdversaryRow(name, None, None, None, None, None, f"not computed: {empty}")
    low, high = wilson_interval(count, total)
    return AdversaryRow(name, count / total, count, total, low, high, what)


def _divergence_row(records: Sequence[AdversaryRecord]) -> AdversaryRow:
    """Return the confirmed-divergence rate: the trials with at least one divergence over the trials taken."""
    diverged = sum(1 for record in records if record.divergences)
    total = sum(len(record.divergences) for record in records)
    what = (
        f"trials with at least one divergence binary_io confirmed, over the {len(records)} trial(s) the adversary "
        f"stage took; {total} divergence(s) in all"
    )
    return _rate(DIVERGENCE_RATE, diverged, len(records), what, "the adversary stage took no trial")


def _rejection_row(records: Sequence[AdversaryRecord]) -> AdversaryRow:
    """Return the invalid-input rejection rate: the refused proposals over the proposals, pooled over the trials."""
    reasons: dict[str, int] = {}
    for record in records:
        for reason, count in record.refusals.items():
            reasons[reason] = reasons.get(reason, 0) + count
    refused, proposals = sum(reasons.values()), sum(record.proposals for record in records)
    by_reason = ", ".join(f"{reason} {reasons[reason]}" for reason in sorted(reasons)) or "none refused"
    what = (
        "proposals the input contract check refused before any run, over the proposals made, refused ones "
        f"included, pooled over {len(records)} trial(s); by reason: {by_reason}"
    )
    empty = f"the adversary proposed no input set in the {len(records)} trial(s) the stage took"
    return _rate(REJECTION_RATE, refused, proposals, what, empty)


def _finding_row(records: Sequence[AdversaryRecord]) -> AdversaryRow:
    """Return the contract-finding count of the group: an int, 0 included, with no counts and no interval."""
    count = sum(len(record.contract_findings) for record in records)
    note = (
        f"contract findings over {len(records)} trial(s): accepted proposals on which the target reference's run "
        "failed or gave outputs the oracle cannot compare against, never a divergence; a count, with no interval"
    )
    return AdversaryRow(FINDING_COUNT, count, None, None, None, None, note)


def adversary_tables(trials: Sequence[Trial]) -> list[AdversaryTable]:
    """Return one AdversaryTable per arm, suite, adversary, and direction of the trials the stage took, sorted.

    The arm is the trial_id's arm segment (lassi.core.record
    parse_trial_id). Trials whose Trial.adversary is None count in no table;
    [] when the stage took none. The result does not depend on the order of
    `trials`.
    """
    groups: dict[tuple[str, str, str, str], list[AdversaryRecord]] = {}
    for trial in trials:
        if trial.adversary is None:
            continue
        arm = parse_trial_id(trial.trial_id).arm
        key = (arm, trial.bench_item.suite, trial.adversary.agent, trial.bench_item.direction)
        groups.setdefault(key, []).append(trial.adversary)
    return [
        AdversaryTable(
            arm=arm,
            suite=suite,
            adversary=adversary,
            direction=direction,
            trials=len(records),
            rows=(_divergence_row(records), _rejection_row(records), _finding_row(records)),
        )
        for (arm, suite, adversary, direction), records in sorted(groups.items())
    ]


# ---------------------------------------------------------------------------
# Markdown


def _cell(text: str) -> str:
    """Return `text` as one plain ASCII Markdown table cell: non-ASCII escaped, pipes and newlines guarded."""
    safe = text.encode("ascii", "backslashreplace").decode("ascii")
    return safe.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _cells(row: AdversaryRow) -> list[str]:
    """Return a row's cells: name, value, count, Wilson interval, and note, "-" for each one that is null."""
    if row.value is None:
        value = "-"
    else:
        value = str(row.value) if isinstance(row.value, int) else f"{row.value:.3f}"
    count = "-" if row.numerator is None or row.denominator is None else f"{row.numerator}/{row.denominator}"
    interval = "-" if row.wilson_low is None else f"{row.wilson_low:.3f} to {row.wilson_high:.3f}"
    return [_cell(cell) for cell in (row.name, value, count, interval, row.note or "-")]


def _table_lines(table: AdversaryTable, level: int) -> list[str]:
    """Return one table's lines: its heading one level below `level`, its trial count, and its rows."""
    heading = f"Arm {table.arm}, suite {table.suite}, adversary {table.adversary}, direction {table.direction}"
    lines = [f"{'#' * (level + 1)} {_cell(heading)}", "", f"Trials the adversary stage took: {table.trials}.", ""]
    return lines + table_lines(HEADER, (_cells(row) for row in table.rows))


def adversary_markdown(tables: Sequence[AdversaryTable], level: int = 1) -> str:
    """Return the adversary metrics as plain ASCII Markdown: a heading at `level`, the legend, and every table.

    The tables come sorted by arm, suite, adversary, and direction, each
    under a heading one level down; NO_TABLES stands in their place when
    there is none. Raises ValueError when `level` is not an int from 1 to 5.
    """
    if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 5:
        raise ValueError(f"adversary_markdown: level must be an int from 1 to 5, got {level!r}")
    ordered = sorted(tables, key=lambda table: (table.arm, table.suite, table.adversary, table.direction))
    lines = [f"{'#' * level} {HEADING}", "", LEGEND, ""]
    for table in ordered:
        lines += [*_table_lines(table, level), ""]
    if not ordered:
        lines += [NO_TABLES, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parquet


def adversary_rows(tables: Sequence[AdversaryTable]) -> list[dict[str, Any]]:
    """Return one Parquet row per table and metric, in column order, sorted by _SORT_KEYS.

    Each row holds its table's arm, suite, adversary, direction, and trials,
    the metric's ordinal in ADVERSARY_METRICS order, and the row's name,
    value (a float, the count included), value_int (the count's value as an
    int; None on a rate row), numerator, denominator, Wilson bounds, and
    note.
    """
    rows = []
    for table in tables:
        for ordinal, row in enumerate(table.rows):
            rows.append(
                {
                    "arm": table.arm,
                    "suite": table.suite,
                    "adversary": table.adversary,
                    "direction": table.direction,
                    "trials": table.trials,
                    "ordinal": ordinal,
                    "name": row.name,
                    "value": None if row.value is None else float(row.value),
                    "value_int": row.value if isinstance(row.value, int) else None,
                    "numerator": row.numerator,
                    "denominator": row.denominator,
                    "wilson_low": row.wilson_low,
                    "wilson_high": row.wilson_high,
                    "note": row.note,
                }
            )
    return sorted(rows, key=lambda row: tuple(row[key] for key in _SORT_KEYS))


def write_adversary_parquet(tables: Sequence[AdversaryTable], out_dir: Path) -> None:
    """Write adversary_rows(tables) to `<out_dir>/adversary_metrics/part-0.parquet`.

    The table is built before its directory is removed, so a value its
    column cannot hold leaves `out_dir` unchanged. The directory then holds
    only these rows, and none is written when there are no rows; every other
    file in `out_dir` is kept.
    """
    arrow = pa.Table.from_pylist(adversary_rows(tables), schema=_SCHEMA)
    table_dir = Path(out_dir) / TABLE
    if table_dir.exists():
        shutil.rmtree(table_dir)
    if not arrow.num_rows:
        return
    table_dir.mkdir(parents=True)
    pq.write_table(arrow, str(table_dir / PART))


def read_adversary_parquet(out_dir: Path) -> list[dict[str, Any]]:
    """Read the rows write_adversary_parquet wrote under `out_dir`, in its order; [] when it wrote none."""
    path = Path(out_dir) / TABLE / PART
    if not path.is_file():
        return []
    rows = pq.read_table(str(path), schema=_SCHEMA).to_pylist()
    return sorted(rows, key=lambda row: tuple(row[key] for key in _SORT_KEYS))
