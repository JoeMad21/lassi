"""Markdown and Parquet forms of the run metric tables (task P2.8).

Bible: Evaluation Protocol (preamble; Acceptance Criteria, the compile-only
label), Reporting Rules (device labels; PLACEHOLDER for unmeasured values),
Readability Standards (Run row: per-arm tables; machine formats mirror
human formats), Design Principle 7 (Parquet mirrors).

table_markdown renders one lassi.analysis.metrics MetricTable: a heading
with the arm and direction, the trial and scenario counts, the devices,
the compile-stage label where it applies, one Markdown line per metric row
(first cell the row name), and the stage-reached and corrections
distributions, one line per key. metrics_markdown joins every table under a
legend. The Markdown is plain ASCII and holds only names, counts, rates,
and notes, never prompt, source, context, or model text (OQ-018).

write_metrics_parquet writes three tables, metrics, stage_reached, and
corrections, as Hive-partitioned Parquet under
`<out_dir>/<table>/arm=.../direction=.../part-0.parquet`, as
lassi.core.parquet does for trials; read_metrics_parquet reads them back.
It is metrics_arrow, which builds the three Arrow tables in memory and
raises for any value a column cannot hold, then write_metrics_arrow, which
only writes them; a caller that must refuse before creating a directory
builds first and writes later.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds

from lassi.analysis.metrics import COMPILE_STAGE, MetricRow, MetricTable

TABLES = ("metrics", "stage_reached", "corrections")
PARTITION_COLUMNS = ("arm", "direction")
ROW_FIELDS = (
    "name", "value", "numerator", "denominator", "wilson_low", "wilson_high",
    "paper_published", "paper_recount", "paper_cite", "note", "placeholder",
)

_STRING = pa.string()
_INT = pa.int64()
_DOUBLE = pa.float64()
_BOOL = pa.bool_()
_KEYS = [(name, _STRING) for name in PARTITION_COLUMNS]

SCHEMAS = {
    "metrics": pa.schema(
        [
            *_KEYS,
            ("ordinal", _INT),
            ("compile_only", _BOOL),
            ("trials", _INT),
            ("scenarios", _INT),
            ("trials_per_scenario_min", _INT),
            ("trials_per_scenario_max", _INT),
            ("devices", pa.list_(_STRING)),
            ("name", _STRING),
            ("value", _DOUBLE),
            ("numerator", _DOUBLE),
            ("denominator", _INT),
            ("wilson_low", _DOUBLE),
            ("wilson_high", _DOUBLE),
            ("paper_published", _STRING),
            ("paper_recount", _STRING),
            ("paper_cite", _STRING),
            ("note", _STRING),
            ("placeholder", _BOOL),
        ]
    ),
    "stage_reached": pa.schema([*_KEYS, ("ordinal", _INT), ("stage", _STRING), ("count", _INT)]),
    "corrections": pa.schema([*_KEYS, ("corrections", _INT), ("count", _INT)]),
}
_SORT_KEYS = {
    "metrics": ("arm", "direction", "ordinal"),
    "stage_reached": ("arm", "direction", "ordinal"),
    "corrections": ("arm", "direction", "corrections"),
}

LEGEND = (
    "One table per arm and direction (bible Evaluation Protocol). Every value comes from the lassi score "
    "profile's components and the trial records; nothing is rescored. Intervals are Wilson 95% over each row's "
    "own count: compile, run, correct, cap-hit, and fence-quirk rates cover every trial of the arm and "
    "direction; first try and Sim-T >= 0.6 cover the correct trials, the paper's denominator; pass@k is the "
    "mean over scenarios of each scenario's unbiased pass@k, with its interval over the scenario count. A trial "
    "whose component is None is excluded and counted in the row's note. PLACEHOLDER marks a value not yet "
    "measured. The paper columns give the paper's published value and the recount read from its tables, each "
    "with the Wilson 95% interval of the paper's own count and denominator; they are paper values, not "
    "measurements, and a compile-stage reproduction shows none. The paper does not state which tokenizer its "
    "Sim-T used ([OPEN], OQ-022); the Sim-T row compares the faithful Python-tokenize sim_t, unrounded, while "
    "the paper prints Sim-T to two decimals."
)


# ---------------------------------------------------------------------------
# Markdown


def _cell(text: str) -> str:
    """Return `text` as one plain ASCII Markdown table cell: non-ASCII escaped, pipes and newlines guarded."""
    safe = text.encode("ascii", "backslashreplace").decode("ascii")
    return safe.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _number(value: float | None) -> str:
    """Return a rate or bound with three decimals, or '-' for None."""
    return "-" if value is None else f"{value:.3f}"


def _count(row: MetricRow) -> str:
    """Return a row's `<numerator>/<denominator>`, the numerator with three decimals when it is a float."""
    if row.numerator is None or row.denominator is None:
        return "-"
    numerator = str(row.numerator) if isinstance(row.numerator, int) else f"{row.numerator:.3f}"
    return f"{numerator}/{row.denominator}"


def _metric_line(row: MetricRow) -> str:
    """Return the Markdown table line of one metric row."""
    value = "PLACEHOLDER" if row.placeholder and row.value is None else _number(row.value)
    interval = "-" if row.wilson_low is None else f"{_number(row.wilson_low)} to {_number(row.wilson_high)}"
    cells = [row.name, value, _count(row), interval, row.paper_published or "-", row.paper_recount or "-",
             row.paper_cite or "-", row.note or "-"]
    return "| " + " | ".join(_cell(cell) for cell in cells) + " |"


def _summary(table: MetricTable) -> list[str]:
    """Return the lines under a table's heading: trials, scenarios, devices, and the compile-stage label."""
    sizes = sorted(set(table.trials_per_scenario.values()))
    per = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]} to {sizes[-1]}"
    lines = [
        f"Trials: {table.trials} in {len(table.trials_per_scenario)} scenario(s), n = {per} per scenario.",
        f"Device: {', '.join(table.devices) or 'not recorded'}.",
    ]
    if table.compile_only:
        lines.append(f"Label: {COMPILE_STAGE}. No program ran, so correctness is not computed, and no value "
                     "here is compared with the paper's correctness values.")
    return [_cell(line) for line in lines]


def table_markdown(table: MetricTable) -> str:
    """Return one metric table as plain ASCII Markdown (see the module docstring)."""
    lines = [f"## {_cell(table.arm)} {_cell(table.direction)}", "", *_summary(table), ""]
    lines += [
        "| Metric | Value | Count | Wilson 95% | Paper published | Paper recount | Paper cite | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *(_metric_line(row) for row in table.rows),
        "",
        "Stage reached by the last attempt of each trial (none: the trial holds no attempt):",
        "",
        "| Stage | Trials |",
        "| --- | --- |",
        *(f"| {_cell(stage)} | {count} |" for stage, count in table.stage_reached.items()),
        "",
        "Corrections per trial (final.corrections):",
        "",
        "| Corrections | Trials |",
        "| --- | --- |",
        *(f"| {value} | {count} |" for value, count in table.corrections.items()),
    ]
    return "\n".join(lines) + "\n"


def metrics_markdown(tables: Sequence[MetricTable]) -> str:
    """Return a plain ASCII Markdown document with the legend and every table, sorted by arm and direction."""
    ordered = sorted(tables, key=lambda table: (table.arm, table.direction))
    sections = [table_markdown(table) for table in ordered] or ["No trials were scored.\n"]
    return "\n".join(["# Run metrics\n", LEGEND + "\n", *sections])


# ---------------------------------------------------------------------------
# Parquet


def metrics_rows(tables: Sequence[MetricTable]) -> dict[str, list[dict[str, Any]]]:
    """Return the rows of the three Parquet tables, in column order and sort order."""
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLES}
    for table in tables:
        key = {"arm": table.arm, "direction": table.direction}
        sizes = list(table.trials_per_scenario.values())
        head = {
            "compile_only": table.compile_only,
            "trials": table.trials,
            "scenarios": len(sizes),
            "trials_per_scenario_min": min(sizes, default=None),
            "trials_per_scenario_max": max(sizes, default=None),
            "devices": list(table.devices),
        }
        for ordinal, row in enumerate(table.rows):
            fields = {name: getattr(row, name) for name in ROW_FIELDS}
            fields["numerator"] = None if row.numerator is None else float(row.numerator)
            rows["metrics"].append({**key, "ordinal": ordinal, **head, **fields})
        for ordinal, (stage, count) in enumerate(table.stage_reached.items()):
            rows["stage_reached"].append({**key, "ordinal": ordinal, "stage": stage, "count": count})
        for value, count in table.corrections.items():
            rows["corrections"].append({**key, "corrections": value, "count": count})
    return {name: _sorted_rows(name, found) for name, found in rows.items()}


def _sorted_rows(table: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows in the table's sort order."""
    keys = _SORT_KEYS[table]
    return sorted(rows, key=lambda row: tuple(row[key] for key in keys))


def _partitioning() -> ds.Partitioning:
    """Return the Hive partitioning by arm and direction, both typed as strings."""
    return ds.HivePartitioning(pa.schema(_KEYS))


def check_partition_dirs(tables: Sequence[MetricTable]) -> None:
    """Raise ValueError when two tables share (arm, direction) or their values would not map to distinct directories.

    Windows and macOS compare directory names without letter case, and
    Windows drops a trailing '.', as lassi.core.parquet guards for trials.
    """
    keys: set[tuple[str, str]] = set()
    folded: dict[tuple[str, ...], tuple[str, ...]] = {}
    for table in tables:
        key = (table.arm, table.direction)
        for name, value in zip(PARTITION_COLUMNS, key, strict=True):
            if value.endswith("."):
                raise ValueError(f"partition value {name}={value!r} ends with '.', which Windows drops")
        if key in keys:
            raise ValueError(f"two metric tables share arm and direction {key}")
        keys.add(key)
        for depth in (1, 2):
            prefix = key[:depth]
            other = folded.setdefault(tuple(value.casefold() for value in prefix), prefix)
            if other != prefix:
                raise ValueError(f"partition values {other} and {prefix} differ only by letter case")


def metrics_arrow(tables: Sequence[MetricTable]) -> dict[str, pa.Table]:
    """Return the metrics, stage_reached, and corrections tables of `tables` as Arrow tables, keyed by name.

    Nothing is written. Raises ValueError from check_partition_dirs, and
    whatever pyarrow raises for a value its column cannot hold, such as
    OverflowError for a corrections count beyond int64 or UnicodeEncodeError
    for a string holding a lone surrogate.
    """
    check_partition_dirs(tables)
    rows = metrics_rows(tables)
    return {name: pa.Table.from_pylist(rows[name], schema=SCHEMAS[name]) for name in TABLES}


def write_metrics_arrow(arrow: Mapping[str, pa.Table], out_dir: Path) -> None:
    """Write the tables metrics_arrow built as Hive-partitioned Parquet under `out_dir`.

    Each table directory is removed first, so the result holds only these
    tables; other files in `out_dir` are kept, and a table with no rows gets
    no directory. Only I/O happens here: no value is checked, and pyarrow's
    default cap of 1024 partitions per write is lifted, so the number of arms
    and directions cannot make a write fail.
    """
    for name in TABLES:
        table_dir = Path(out_dir) / name
        if table_dir.exists():
            shutil.rmtree(table_dir)
        if not arrow[name].num_rows:
            continue
        ds.write_dataset(
            arrow[name],
            str(table_dir),
            format="parquet",
            partitioning=_partitioning(),
            basename_template="part-{i}.parquet",
            existing_data_behavior="error",
            max_partitions=arrow[name].num_rows,
        )


def write_metrics_parquet(tables: Sequence[MetricTable], out_dir: Path) -> None:
    """Write the metrics, stage_reached, and corrections tables of `tables` as Parquet under `out_dir`.

    Each table directory is removed first, so the result holds only these
    tables; other files in `out_dir` are kept, and a table with no rows gets
    no directory. Every table is built (metrics_arrow) before any directory
    is removed (write_metrics_arrow), so a ValueError (tables sharing an arm
    and direction, or partition values that differ only by letter case)
    leaves `out_dir` unchanged.
    """
    write_metrics_arrow(metrics_arrow(tables), out_dir)


def read_metrics_parquet(out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Read the tables written by write_metrics_parquet back into rows, keyed metrics, stage_reached, corrections.

    A missing table directory reads as no rows.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for name in TABLES:
        table_dir = Path(out_dir) / name
        if not table_dir.is_dir():
            result[name] = []
            continue
        dataset = ds.dataset(str(table_dir), schema=SCHEMAS[name], format="parquet", partitioning=_partitioning())
        result[name] = _sorted_rows(name, dataset.to_table().to_pylist())
    return result
