"""The Parquet mirror of a run's Trial records.

Parquet mirrors the JSON records and never replaces them (bible Design
Principle 7). A run's trials flatten into four tables, trials, attempts,
diagnostics, and requests, each written as Hive-partitioned Parquet under
`<out_dir>/<table>/project=.../arm=.../bench=.../direction=.../part-0.parquet`.
Partition values come from the parsed trial_id and are URI-encoded in
directory names as pyarrow does by default, so a '+' is written as `%2B` and
read back as '+'. Response texts stay in the text store; the attempts table
keeps only their sha256. Nested record fields become `<field>_<key>` columns,
so the trials table carries each trial's provenance (its copy of the run
manifest) as provenance_commit, provenance_dirty, provenance_device,
provenance_sdk, and provenance_date, the target reference's baseline run as
reference_run_<key> columns, and the end reason as final_end_reason_code
and final_end_reason_message (null when the trial ended normally).

The requests table has one row per recorded model request (Trial.requests),
in index order: the stage that sent it, the attempt its reply became
(attempt_index, null for a context request), the role and the sha256 of each
message as lists in the order sent, the reply's text reference, and the
count of its diagnostics. Message and reply texts stay in the text store. A
trial whose requests were not recorded (None) has no rows, as does one that
asked no model.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds

from lassi.core.record import TOOLCHAIN_PIN_NAMES, Attempt, Diagnostic, Request, TextRef, Trial, parse_trial_id
from lassi.core.store import sha256_text

TABLES = ("trials", "attempts", "diagnostics", "requests")
PARTITION_COLUMNS = ("project", "arm", "bench", "direction")

_STRING = pa.string()
_INT = pa.int64()
_DOUBLE = pa.float64()
_BOOL = pa.bool_()
_DOUBLE_LIST = pa.list_(pa.float64())
_STRING_LIST = pa.list_(pa.string())
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

_KEY_FIELDS = [(name, _STRING) for name in (*PARTITION_COLUMNS, "trial_id")]

SCHEMAS = {
    "trials": pa.schema(
        [
            *_KEY_FIELDS,
            ("item", _STRING),
            ("run", _INT),
            ("recipe_hash", _STRING),
            *[(f"toolchain_pins_{name}", _STRING) for name in TOOLCHAIN_PIN_NAMES],
            ("provenance_commit", _STRING),
            ("provenance_dirty", _BOOL),
            ("provenance_device", _STRING),
            ("provenance_sdk", _STRING),
            ("provenance_date", _STRING),
            ("bench_item_suite", _STRING),
            ("bench_item_item", _STRING),
            ("bench_item_split", _STRING),
            ("bench_item_direction", _STRING),
            ("model_backend", _STRING),
            ("model_id", _STRING),
            ("model_sampling_temperature", _DOUBLE),
            ("model_sampling_top_p", _DOUBLE),
            ("model_sampling_max_tokens", _INT),
            ("reference_run_exit_code", _INT),
            ("reference_run_hang", _BOOL),
            ("reference_run_sim_ub", _BOOL),
            ("reference_run_wall_s", _DOUBLE),
            ("reference_run_stdout_ref_sha256", _STRING),
            ("reference_run_stdout_ref_path", _STRING),
            ("reference_run_outputs_ref_sha256", _STRING),
            ("reference_run_outputs_ref_path", _STRING),
            ("context_knowledge_summary", _STRING),
            ("context_source_description", _STRING),
            ("final_stage_reached", _STRING),
            ("final_alignment", _DOUBLE),
            ("final_score", _DOUBLE),
            ("final_corrections", _INT),
            ("final_wall_s", _DOUBLE),
            ("final_end_reason_code", _STRING),
            ("final_end_reason_message", _STRING),
            ("attempt_count", _INT),
        ]
    ),
    "attempts": pa.schema(
        [
            *_KEY_FIELDS,
            ("index", _INT),
            ("prompt_ref_sha256", _STRING),
            ("prompt_ref_path", _STRING),
            ("response_sha256", _STRING),
            ("files", _STRING),
            ("diff_from_previous", _STRING),
            ("stage_reached", _STRING),
            ("diagnostic_count", _INT),
            ("run_exit_code", _INT),
            ("run_hang", _BOOL),
            ("run_sim_ub", _BOOL),
            ("run_wall_s", _DOUBLE),
            ("run_stdout_ref_sha256", _STRING),
            ("run_stdout_ref_path", _STRING),
            ("run_outputs_ref_sha256", _STRING),
            ("run_outputs_ref_path", _STRING),
            ("alignment_per_input", _DOUBLE_LIST),
            ("alignment_mean", _DOUBLE),
            ("profile_runtime_s", _DOUBLE),
            ("profile_avg_power_w", _DOUBLE),
            ("profile_energy_j", _DOUBLE),
            ("guards_host_compute", _BOOL),
            ("guards_harness_tamper", _BOOL),
            ("guards_oracle_access", _BOOL),
            ("score_components", _STRING),
            ("score_scalar", _DOUBLE),
        ]
    ),
    "diagnostics": pa.schema(
        [
            *_KEY_FIELDS,
            ("attempt_index", _INT),
            ("ordinal", _INT),
            ("stage", _STRING),
            ("severity", _STRING),
            ("code", _STRING),
            ("file", _STRING),
            ("line", _INT),
            ("column", _INT),
            ("message", _STRING),
        ]
    ),
    "requests": pa.schema(
        [
            *_KEY_FIELDS,
            ("index", _INT),
            ("stage", _STRING),
            ("attempt_index", _INT),
            ("message_roles", _STRING_LIST),
            ("message_sha256", _STRING_LIST),
            ("reply_ref_sha256", _STRING),
            ("reply_ref_path", _STRING),
            ("diagnostic_count", _INT),
        ]
    ),
}

_SORT_KEYS = {
    "trials": ("trial_id",),
    "attempts": ("trial_id", "index"),
    "diagnostics": ("trial_id", "attempt_index", "ordinal"),
    "requests": ("trial_id", "index"),
}


# ---------------------------------------------------------------------------
# Rows


def _typed_row(table: str, values: dict[str, Any]) -> dict[str, Any]:
    """Return values in the table's column order; an int that does not fit int64 raises ValueError.

    The record classes already store every float field as a float, so no value
    needs converting. Record ints are unbounded, but their columns are int64.
    """
    schema = SCHEMAS[table]
    if set(values) != set(schema.names):
        raise ValueError(f"{table} row columns differ from the schema: {sorted(set(values) ^ set(schema.names))}")
    for spec in schema:
        value = values[spec.name]
        if spec.type == _INT and value is not None and not _INT64_MIN <= value <= _INT64_MAX:
            where = f"{table}.{spec.name} of trial {values['trial_id']!r}"
            raise ValueError(f"{where} must fit in a 64-bit signed int, got {value!r}")
    return {name: values[name] for name in schema.names}


def _sorted_rows(table: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows in the table's sort order."""
    keys = _SORT_KEYS[table]
    return sorted(rows, key=lambda row: tuple(row[key] for key in keys))


def _ref_columns(prefix: str, ref: TextRef | None) -> dict[str, str | None]:
    """Return the sha256 and path columns of an optional text reference."""
    return {f"{prefix}_sha256": ref.sha256 if ref else None, f"{prefix}_path": ref.path if ref else None}


def _trial_row(trial: Trial, key: dict[str, str]) -> dict[str, Any]:
    """Return the trials row of one trial."""
    parsed = parse_trial_id(trial.trial_id)
    bench, model, final, provenance = trial.bench_item, trial.model, trial.final, trial.provenance
    pins = {f"toolchain_pins_{name}": getattr(trial.toolchain_pins, name) for name in TOOLCHAIN_PIN_NAMES}
    reference, reason = trial.reference_run, final.end_reason
    return {
        **key,
        "item": parsed.item,
        "run": parsed.run,
        "recipe_hash": trial.recipe_hash,
        **pins,
        "provenance_commit": provenance.commit,
        "provenance_dirty": provenance.dirty,
        "provenance_device": provenance.device,
        "provenance_sdk": provenance.sdk,
        "provenance_date": provenance.date,
        "bench_item_suite": bench.suite,
        "bench_item_item": bench.item,
        "bench_item_split": bench.split,
        "bench_item_direction": bench.direction,
        "model_backend": model.backend,
        "model_id": model.id,
        "model_sampling_temperature": model.sampling.temperature,
        "model_sampling_top_p": model.sampling.top_p,
        "model_sampling_max_tokens": model.sampling.max_tokens,
        "reference_run_exit_code": reference.exit_code,
        "reference_run_hang": reference.hang,
        "reference_run_sim_ub": reference.sim_ub,
        "reference_run_wall_s": reference.wall_s,
        **_ref_columns("reference_run_stdout_ref", reference.stdout_ref),
        **_ref_columns("reference_run_outputs_ref", reference.outputs_ref),
        "context_knowledge_summary": trial.context.knowledge_summary,
        "context_source_description": trial.context.source_description,
        "final_stage_reached": final.stage_reached,
        "final_alignment": final.alignment,
        "final_score": final.score,
        "final_corrections": final.corrections,
        "final_wall_s": final.wall_s,
        "final_end_reason_code": reason.code if reason else None,
        "final_end_reason_message": reason.message if reason else None,
        "attempt_count": len(trial.attempts),
    }


def _attempt_row(attempt: Attempt, key: dict[str, str]) -> dict[str, Any]:
    """Return the attempts row of one attempt; files and score components are JSON text."""
    run, profile, guards = attempt.run, attempt.profile, attempt.guards
    return {
        **key,
        "index": attempt.index,
        **_ref_columns("prompt_ref", attempt.prompt_ref),
        "response_sha256": sha256_text(attempt.response_text),
        "files": json.dumps(attempt.files, sort_keys=True, ensure_ascii=True),
        "diff_from_previous": attempt.diff_from_previous,
        "stage_reached": attempt.stage_reached,
        "diagnostic_count": len(attempt.diagnostics),
        "run_exit_code": run.exit_code,
        "run_hang": run.hang,
        "run_sim_ub": run.sim_ub,
        "run_wall_s": run.wall_s,
        **_ref_columns("run_stdout_ref", run.stdout_ref),
        **_ref_columns("run_outputs_ref", run.outputs_ref),
        "alignment_per_input": list(attempt.alignment.per_input),
        "alignment_mean": attempt.alignment.mean,
        "profile_runtime_s": profile.runtime_s,
        "profile_avg_power_w": profile.avg_power_w,
        "profile_energy_j": profile.energy_j,
        "guards_host_compute": guards.host_compute,
        "guards_harness_tamper": guards.harness_tamper,
        "guards_oracle_access": guards.oracle_access,
        "score_components": json.dumps(attempt.score.components, sort_keys=True, ensure_ascii=True),
        "score_scalar": attempt.score.scalar,
    }


def _diagnostic_row(diagnostic: Diagnostic, key: dict[str, str], attempt_index: int, ordinal: int) -> dict[str, Any]:
    """Return the diagnostics row of one diagnostic; ordinal is its position in the attempt."""
    fields = ("stage", "severity", "code", "file", "line", "column", "message")
    return {
        **key,
        "attempt_index": attempt_index,
        "ordinal": ordinal,
        **{name: getattr(diagnostic, name) for name in fields},
    }


def _request_row(request: Request, key: dict[str, str]) -> dict[str, Any]:
    """Return the requests row of one request; its messages become role and sha256 lists in the order sent."""
    return {
        **key,
        "index": request.index,
        "stage": request.stage,
        "attempt_index": request.attempt_index,
        "message_roles": [message.role for message in request.messages],
        "message_sha256": [message.ref.sha256 for message in request.messages],
        **_ref_columns("reply_ref", request.reply_ref),
        "diagnostic_count": len(request.diagnostics),
    }


def trial_rows(trials: Sequence[Trial]) -> dict[str, list[dict[str, Any]]]:
    """Flatten trials into rows for each table, in column order and sorted by trial_id, index, and ordinal.

    Raises ValueError when two trials share a trial_id or an int value does
    not fit its int64 column.
    """
    counts = Counter(trial.trial_id for trial in trials)
    duplicates = sorted(trial_id for trial_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"trial_id appears more than once: {duplicates}")
    rows: dict[str, list[dict[str, Any]]] = {table: [] for table in TABLES}
    for trial in trials:
        parsed = parse_trial_id(trial.trial_id)
        key = {name: getattr(parsed, name) for name in PARTITION_COLUMNS}
        key["trial_id"] = trial.trial_id
        rows["trials"].append(_trial_row(trial, key))
        for attempt in trial.attempts:
            rows["attempts"].append(_attempt_row(attempt, key))
            for ordinal, diagnostic in enumerate(attempt.diagnostics):
                rows["diagnostics"].append(_diagnostic_row(diagnostic, key, attempt.index, ordinal))
        for request in trial.requests or []:
            rows["requests"].append(_request_row(request, key))
    return {table: _sorted_rows(table, [_typed_row(table, row) for row in rows[table]]) for table in TABLES}


# ---------------------------------------------------------------------------
# Files


def _partitioning() -> ds.Partitioning:
    """Return the Hive partitioning by project, arm, bench, and direction, all typed as strings."""
    return ds.HivePartitioning(pa.schema([(name, _STRING) for name in PARTITION_COLUMNS]))


def _check_partition_dirs(rows: Sequence[dict[str, Any]]) -> None:
    """Raise ValueError when partition values would not map to distinct, faithful directory names.

    Windows and macOS compare directory names without letter case, and Windows
    drops a trailing '.', so such values would merge partitions or come back
    changed from read_run_parquet. Every directory level is checked: two
    partitions whose arms differ only by case clash even when their directions
    differ, because both land under one arm directory.
    """
    seen: dict[tuple[str, ...], tuple[str, ...]] = {}
    for row in rows:
        key = tuple(row[name] for name in PARTITION_COLUMNS)
        for name, value in zip(PARTITION_COLUMNS, key, strict=True):
            if value.endswith("."):
                raise ValueError(f"partition value {name}={value!r} ends with '.', which Windows drops")
        for depth in range(1, len(key) + 1):
            prefix = key[:depth]
            other = seen.setdefault(tuple(value.casefold() for value in prefix), prefix)
            if other != prefix:
                columns = "/".join(PARTITION_COLUMNS[:depth])
                raise ValueError(
                    f"partition {columns} values {other} and {prefix} differ only by letter case "
                    "and would share a directory"
                )


def _arrow_tables(rows: dict[str, list[dict[str, Any]]]) -> dict[str, pa.Table]:
    """Return each table's rows as an Arrow table with its schema; values Arrow cannot hold raise ValueError."""
    tables = {}
    for table in TABLES:
        try:
            tables[table] = pa.Table.from_pylist(rows[table], schema=SCHEMAS[table])
        except (OverflowError, UnicodeError, pa.ArrowException) as error:
            raise ValueError(f"{table} rows do not fit the Parquet schema: {error}") from error
    return tables


def write_run_parquet(trials: Sequence[Trial], out_dir: Path) -> None:
    """Write the four tables of `trials` as Hive-partitioned Parquet under `out_dir`.

    Each table directory is removed first, so the result holds only these
    trials; other files in `out_dir` are kept. A table with no rows gets no
    directory. Every table is built before any directory is removed, so a
    value no table can hold (an int outside int64, or partition values that
    cannot be stored as distinct directory names on every OS) raises
    ValueError and leaves `out_dir` unchanged.
    """
    rows = trial_rows(trials)
    _check_partition_dirs(rows["trials"])
    tables = _arrow_tables(rows)
    for table in TABLES:
        table_dir = Path(out_dir) / table
        if table_dir.exists():
            shutil.rmtree(table_dir)
        if not tables[table].num_rows:
            continue
        ds.write_dataset(
            tables[table],
            str(table_dir),
            format="parquet",
            partitioning=_partitioning(),
            basename_template="part-{i}.parquet",
            existing_data_behavior="error",
        )


def read_run_parquet(out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Read the tables written by write_run_parquet back into the rows trial_rows gives.

    A missing table directory reads as no rows. Every column is nullable,
    so a column a file lacks (one written before the column existed) reads
    as null in each of that file's rows.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for table in TABLES:
        table_dir = Path(out_dir) / table
        if not table_dir.is_dir():
            result[table] = []
            continue
        dataset = ds.dataset(str(table_dir), schema=SCHEMAS[table], format="parquet", partitioning=_partitioning())
        result[table] = _sorted_rows(table, dataset.to_table().to_pylist())
    return result
