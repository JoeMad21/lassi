"""Tests for the Parquet mirror of Result records (P0.2), with the requests table (P2.1).

lassi/core/parquet.py flattens Trials into four tables (trials, attempts,
diagnostics, requests), writes them as Hive-partitioned Parquet by project, arm, bench,
and direction, and reads them back to exactly the rows trial_rows produces.
Parquet mirrors the JSON records and never replaces them (bible Design
Principle 7). The fixtures span two arms and two directions, with None values,
an empty per_input list, a trial without attempts, a numeric-looking bench
name, and diagnostics; one more trial sets every nullable field, and its rows
are written out by hand so each column must hold its own field. Every fixture
value is synthetic; none is a measurement.

The trials table carries each Trial's provenance (P0.18) as the columns
provenance_commit, provenance_dirty, provenance_device, provenance_sdk, and
provenance_date, named like the other nested fields (<field>_<key>), placed
after the toolchain pins as the bible's Trial block orders them, and typed
string, bool, string, string, string. Like every column they are nullable
in the Arrow schema; a value is null where the trial's provenance is unknown,
and date is never null because the record always sets it.

The requests table (P2.1) has one row per Trial.requests entry, sorted by
trial_id and index: the key columns, index, stage, attempt_index (null for
a context request), message_roles and message_sha256 (string lists in the
order sent), reply_ref_sha256, reply_ref_path, and diagnostic_count. A
trial whose requests were not recorded (None) or that asked no model ([])
has no rows.

The run flags of Trial.reference_run and Attempt.run (P2.2) are bool
columns right after the run's outputs_ref columns: reference_run_<flag> in
the trials table and run_<flag> in the attempts table, for stdout_truncated,
stderr_truncated, and workdir_incomplete, null where not recorded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lassi.core import interfaces, parquet, record

PROJECT = "lassi-repro"
SUITE = "lassi-hecbench-10"
RECIPE_HASH = "0123456789abcdef" * 4
SHA_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
BACKSLASH = "\\"
E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"

PIN_COLUMNS = tuple(f"toolchain_pins_{name}" for name in record.TOOLCHAIN_PIN_NAMES)
PROVENANCE_COLUMNS = (
    "provenance_commit",
    "provenance_dirty",
    "provenance_device",
    "provenance_sdk",
    "provenance_date",
)
# The target reference's baseline run (Trial.reference_run) and the end reason (Final.end_reason), task P1.5.
REFERENCE_RUN_COLUMNS = (
    "reference_run_exit_code",
    "reference_run_hang",
    "reference_run_sim_ub",
    "reference_run_wall_s",
    "reference_run_stdout_ref_sha256",
    "reference_run_stdout_ref_path",
    "reference_run_outputs_ref_sha256",
    "reference_run_outputs_ref_path",
    "reference_run_stdout_truncated",
    "reference_run_stderr_truncated",
    "reference_run_workdir_incomplete",
)
END_REASON_COLUMNS = ("final_end_reason_code", "final_end_reason_message")
TRIAL_COLUMNS = (
    "project",
    "arm",
    "bench",
    "direction",
    "trial_id",
    "item",
    "run",
    "recipe_hash",
    *PIN_COLUMNS,
    *PROVENANCE_COLUMNS,
    "bench_item_suite",
    "bench_item_item",
    "bench_item_split",
    "bench_item_direction",
    "model_backend",
    "model_id",
    "model_sampling_temperature",
    "model_sampling_top_p",
    "model_sampling_max_tokens",
    *REFERENCE_RUN_COLUMNS,
    "context_knowledge_summary",
    "context_source_description",
    "final_stage_reached",
    "final_alignment",
    "final_score",
    "final_corrections",
    "final_wall_s",
    *END_REASON_COLUMNS,
    "attempt_count",
)
ATTEMPT_COLUMNS = (
    "project",
    "arm",
    "bench",
    "direction",
    "trial_id",
    "index",
    "prompt_ref_sha256",
    "prompt_ref_path",
    "response_sha256",
    "files",
    "diff_from_previous",
    "stage_reached",
    "diagnostic_count",
    "run_exit_code",
    "run_hang",
    "run_sim_ub",
    "run_wall_s",
    "run_stdout_ref_sha256",
    "run_stdout_ref_path",
    "run_outputs_ref_sha256",
    "run_outputs_ref_path",
    "run_stdout_truncated",
    "run_stderr_truncated",
    "run_workdir_incomplete",
    "alignment_per_input",
    "alignment_mean",
    "profile_runtime_s",
    "profile_avg_power_w",
    "profile_energy_j",
    "guards_host_compute",
    "guards_harness_tamper",
    "guards_oracle_access",
    "score_components",
    "score_scalar",
)
DIAGNOSTIC_COLUMNS = (
    "project",
    "arm",
    "bench",
    "direction",
    "trial_id",
    "attempt_index",
    "ordinal",
    "stage",
    "severity",
    "code",
    "file",
    "line",
    "column",
    "message",
)
REQUEST_COLUMNS = (
    "project",
    "arm",
    "bench",
    "direction",
    "trial_id",
    "index",
    "stage",
    "attempt_index",
    "message_roles",
    "message_sha256",
    "reply_ref_sha256",
    "reply_ref_path",
    "diagnostic_count",
)
COLUMNS = {
    "trials": TRIAL_COLUMNS,
    "attempts": ATTEMPT_COLUMNS,
    "diagnostics": DIAGNOSTIC_COLUMNS,
    "requests": REQUEST_COLUMNS,
}

INT64_COLUMNS = frozenset(
    {
        "run",
        "model_sampling_max_tokens",
        "reference_run_exit_code",
        "final_corrections",
        "attempt_count",
        "index",
        "diagnostic_count",
        "run_exit_code",
        "attempt_index",
        "ordinal",
        "line",
        "column",
    }
)
DOUBLE_COLUMNS = frozenset(
    {
        "model_sampling_temperature",
        "model_sampling_top_p",
        "reference_run_wall_s",
        "final_alignment",
        "final_score",
        "final_wall_s",
        "run_wall_s",
        "alignment_mean",
        "profile_runtime_s",
        "profile_avg_power_w",
        "profile_energy_j",
        "score_scalar",
    }
)
BOOL_COLUMNS = frozenset(
    {
        "provenance_dirty",
        "reference_run_hang",
        "reference_run_sim_ub",
        "reference_run_stdout_truncated",
        "reference_run_stderr_truncated",
        "reference_run_workdir_incomplete",
        "run_hang",
        "run_sim_ub",
        "run_stdout_truncated",
        "run_stderr_truncated",
        "run_workdir_incomplete",
        "guards_host_compute",
        "guards_harness_tamper",
        "guards_oracle_access",
    }
)
LIST_DOUBLE_COLUMNS = frozenset({"alignment_per_input"})
LIST_STRING_COLUMNS = frozenset({"message_roles", "message_sha256"})

ID_A_OMP_ENTROPY = f"{PROJECT}/arm-a/{SUITE}/omp-cuda/entropy/run01"
ID_A_OMP_LAYOUT = f"{PROJECT}/arm-a/{SUITE}/omp-cuda/layout/run01"
ID_B_OMP_ENTROPY = f"{PROJECT}/arm-b/{SUITE}/omp-cuda/entropy/run01"
ID_A_CUDA_ENTROPY = f"{PROJECT}/arm-a/{SUITE}/cuda-omp/entropy/run02"
ID_B_CUDA_STENCIL = f"{PROJECT}/arm-b/2024/cuda-omp/stencil/run01"
ID_B_OMP_FILLED = f"{PROJECT}/arm-b/{SUITE}/omp-cuda/layout/run03"
INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)

PARTITION_A_OMP = f"project={PROJECT}/arm=arm-a/bench={SUITE}/direction=omp-cuda"
PARTITION_B_OMP = f"project={PROJECT}/arm=arm-b/bench={SUITE}/direction=omp-cuda"
PARTITION_A_CUDA = f"project={PROJECT}/arm=arm-a/bench={SUITE}/direction=cuda-omp"
PARTITION_B_CUDA = f"project={PROJECT}/arm=arm-b/bench=2024/direction=cuda-omp"

# Synthetic provenance values: neither commit is a commit of this repository; dates are in started_utc format.
FIXTURE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
FILLED_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
FIXTURE_DATE = "2026-09-23T12:34:56+00:00"
FILLED_DATE = "2026-09-24T01:02:03+00:00"


# ---------------------------------------------------------------------------
# Fixture builders


def sha(text: str) -> str:
    """Return the sha256 hex of the UTF-8 bytes of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_ref(text: str) -> record.TextRef:
    """Return the text store reference for text, computed with hashlib."""
    digest = sha(text)
    return record.TextRef(sha256=digest, path=f"texts/{digest[:2]}/{digest}.txt")


def message(role: str, text: str) -> record.RequestMessage:
    """Return a request message whose text reference is computed with hashlib."""
    return record.RequestMessage(role=role, ref=text_ref(text))


def fixture_provenance(**changes: Any) -> record.Provenance:
    """Return the provenance of a compile-only run with git available and no sdk reported, with the given changes."""
    fields: dict[str, Any] = {
        "commit": FIXTURE_COMMIT,
        "dirty": False,
        "device": "none (compile only)",
        "sdk": None,
        "date": FIXTURE_DATE,
    }
    fields.update(changes)
    return record.Provenance(**fields)


def make_trial(trial_id: str, attempts: list[record.Attempt], **changes: Any) -> record.Trial:
    """Return a Trial whose bench item is derived from trial_id, with the given attempts and fields."""
    _, _, bench, direction, item, _ = trial_id.split("/")
    sampling = interfaces.Sampling(temperature=0.2, top_p=0.95, max_tokens=4096)
    fields: dict[str, Any] = {
        "trial_id": trial_id,
        "recipe_hash": RECIPE_HASH,
        "provenance": fixture_provenance(),
        "bench_item": record.BenchItem(suite=bench, item=item, split="eval", direction=direction),
        "model": record.ModelInfo(backend="mock", id="mock-fixture", sampling=sampling),
        "attempts": attempts,
    }
    fields.update(changes)
    return record.Trial(**fields)


def trial_a_omp_entropy() -> record.Trial:
    """Return a two-attempt trial with diagnostics, None values, an empty per_input list, and two requests."""
    first = record.Attempt(
        index=0,
        prompt_ref=text_ref("prompt 0\n"),
        response_text="response 0\n",
        files={"main.cu": "int x;\n", "a.h": f"// caf{E_ACUTE}\n"},
        stage_reached="S1",
        diagnostics=[
            record.Diagnostic(
                stage="compile", severity="error", code="20", file="main.cu", line=1, column=5, message="m0"
            ),
            record.Diagnostic(stage="compile", severity="note", message="m1 | x"),
        ],
    )
    second = record.Attempt(
        index=1,
        diff_from_previous="--- a/main.cu\n+++ /dev/null\n@@ -1 +0,0 @@\n-int x;\n",
        stage_reached="S5",
        run=record.RunInfo(exit_code=0, hang=False, wall_s=0.5, stdout_ref=text_ref("PASS\n")),
        alignment=record.Alignment(per_input=[1.0, 0.5], mean=0.75),
        guards=record.Guards(host_compute=False),
        score=record.ScoreBreakdown(components={"energy": None, "alignment": 0.75}),
    )
    requests = [
        record.Request(
            index=0,
            stage="summarize_context",
            attempt_index=None,
            messages=[message("system", "system general\n"), message("user", "summary request\n")],
            reply_ref=text_ref("summary\n"),
        ),
        record.Request(
            index=1,
            stage="generate",
            attempt_index=0,
            messages=[message("system", "system direction\n"), message("user", "prompt 0\n")],
            reply_ref=text_ref("response 0\n"),
        ),
    ]
    return make_trial(
        ID_A_OMP_ENTROPY,
        [first, second],
        toolchain_pins=record.ToolchainPins(cuda="fixture-cuda"),
        context=record.Context(knowledge_summary="summary\n"),
        requests=requests,
        final=record.Final(stage_reached="S5", alignment=0.75, corrections=1),
    )


def trial_a_omp_layout() -> record.Trial:
    """Return a one-attempt trial sharing a partition, every attempt field at its default, requests not recorded."""
    return make_trial(ID_A_OMP_LAYOUT, [record.Attempt(index=0, stage_reached="S0")])


def trial_b_omp_entropy() -> record.Trial:
    """Return a one-attempt trial in the second arm with a partly set profile and a scalar score."""
    attempt = record.Attempt(
        index=0,
        response_text="r\n",
        files={"main.cu": "int y;\n"},
        stage_reached="S4",
        run=record.RunInfo(exit_code=1, hang=True, sim_ub=False, outputs_ref=text_ref("out\n")),
        profile=record.Profile(runtime_s=0.25),
        guards=record.Guards(host_compute=True, harness_tamper=False, oracle_access=False),
        score=record.ScoreBreakdown(components={"stage": 0.0}, scalar=0.0),
    )
    return make_trial(ID_B_OMP_ENTROPY, [attempt], final=record.Final(stage_reached="S4", score=0.0, wall_s=3.5))


def trial_a_cuda_entropy() -> record.Trial:
    """Return a trial that asked no model in the second direction, from a run where git was unavailable."""
    return make_trial(ID_A_CUDA_ENTROPY, [], provenance=fixture_provenance(commit=None, dirty=None), requests=[])


def trial_b_cuda_stencil() -> record.Trial:
    """Return a trial whose bench name is all digits, with one run-stage warning and one request."""
    warning = record.Diagnostic(stage="run", severity="warning", file="k.cu", line=7, message="w")
    request = record.Request(
        index=0, stage="generate", attempt_index=0, messages=[message("user", "prompt s\n")], reply_ref=text_ref("")
    )
    attempt = record.Attempt(index=0, stage_reached="S5", diagnostics=[warning])
    return make_trial(ID_B_CUDA_STENCIL, [attempt], requests=[request])


def filled_attempts() -> list[record.Attempt]:
    """Return two attempts that set every nullable field, each to a value unlike its sibling columns."""
    first = record.Attempt(
        index=0,
        prompt_ref=text_ref("prompt b0\n"),
        response_text="response b0\n",
        files={"k.cu": "int z;\n"},
        stage_reached="S3",
        diagnostics=[
            record.Diagnostic(stage="jit", severity="warning", code="W7", file="k.cu", line=4, column=9, message="w0")
        ],
        run=record.RunInfo(
            exit_code=3,
            hang=False,
            sim_ub=True,
            wall_s=1.5,
            stdout_ref=text_ref("stdout b0\n"),
            outputs_ref=text_ref("outputs b0\n"),
            stdout_truncated=False,
            stderr_truncated=True,
            workdir_incomplete=False,
        ),
        alignment=record.Alignment(per_input=[0.5], mean=0.625),
        profile=record.Profile(runtime_s=0.25, avg_power_w=150.0, energy_j=37.5),
        guards=record.Guards(host_compute=True, harness_tamper=False, oracle_access=None),
        score=record.ScoreBreakdown(components={"stage": 0.6}, scalar=0.125),
    )
    second = record.Attempt(
        index=1,
        prompt_ref=text_ref("prompt b1\n"),
        response_text="response b1\n",
        files={"k.cu": "int w;\n"},
        diff_from_previous="--- a/k.cu\n+++ b/k.cu\n@@ -1 +1 @@\n-int z;\n+int w;\n",
        stage_reached="S4",
        run=record.RunInfo(
            exit_code=-2,
            hang=True,
            sim_ub=False,
            wall_s=2.5,
            stdout_ref=text_ref("stdout b1\n"),
            outputs_ref=text_ref("outputs b1\n"),
            stdout_truncated=False,
            stderr_truncated=False,
            workdir_incomplete=True,
        ),
        alignment=record.Alignment(per_input=[0.75, 0.25], mean=0.5),
        profile=record.Profile(runtime_s=0.5, avg_power_w=75.0, energy_j=37.25),
        guards=record.Guards(host_compute=False, harness_tamper=None, oracle_access=True),
        score=record.ScoreBreakdown(components={"stage": 0.8, "alignment": 0.5}, scalar=0.375),
    )
    return [first, second]


def filled_reference_run() -> record.RunInfo:
    """Return a reference run that sets every field, each to a value unlike its sibling columns."""
    return record.RunInfo(
        exit_code=7,
        hang=False,
        sim_ub=True,
        wall_s=4.5,
        stdout_ref=text_ref("reference stdout b\n"),
        outputs_ref=text_ref("reference outputs b\n"),
        stdout_truncated=True,
        stderr_truncated=False,
        workdir_incomplete=False,
    )


FILLED_END_REASON = record.EndReason(code="correction-cap", message="synthetic: the cap stopped the loop")


def filled_requests() -> list[record.Request]:
    """Return a context request with a warning, then one request per attempt, each index unlike its attempt index.

    attempt_index is null only on the context request, where the record
    requires it.
    """
    warning = record.Diagnostic(stage="parse", severity="warning", code="invalid-text", message="w-r0")
    notes = [
        record.Diagnostic(stage="parse", severity="warning", code="fence-quirk", message="w-r2a"),
        record.Diagnostic(stage="parse", severity="note", message="w-r2b"),
    ]
    return [
        record.Request(
            index=0,
            stage="describe_source",
            attempt_index=None,
            messages=[message("system", "system b\n"), message("user", "describe b\n")],
            reply_ref=text_ref("source b\n"),
            diagnostics=[warning],
        ),
        record.Request(
            index=1,
            stage="generate",
            attempt_index=0,
            messages=[message("system", "direction b\n"), message("user", "prompt b0\n")],
            reply_ref=text_ref("response b0\n"),
        ),
        record.Request(
            index=2,
            stage="run_loop",
            attempt_index=1,
            messages=[message("user", "prompt b1\n")],
            reply_ref=text_ref("response b1\n"),
            diagnostics=notes,
        ),
    ]


def trial_b_omp_filled() -> record.Trial:
    """Return a trial in which every nullable field is set, so each column must hold its own field."""
    sampling = interfaces.Sampling(temperature=0.7, top_p=0.9, max_tokens=512)
    return make_trial(
        ID_B_OMP_FILLED,
        filled_attempts(),
        toolchain_pins=record.ToolchainPins(**{name: f"pin-{name}" for name in record.TOOLCHAIN_PIN_NAMES}),
        provenance=record.Provenance(
            commit=FILLED_COMMIT, dirty=True, device="fixture-device", sdk="fixture-sdk", date=FILLED_DATE
        ),
        model=record.ModelInfo(backend="mock", id="mock-filled", sampling=sampling),
        reference_run=filled_reference_run(),
        context=record.Context(knowledge_summary="knowledge b\n", source_description="source b\n"),
        requests=filled_requests(),
        final=record.Final(
            stage_reached="S4", alignment=0.25, score=0.5, corrections=1, wall_s=3.5, end_reason=FILLED_END_REASON
        ),
    )


def run_trials() -> list[record.Trial]:
    """Return the five fixture trials: two arms, two directions."""
    return [
        trial_a_omp_entropy(),
        trial_a_omp_layout(),
        trial_b_omp_entropy(),
        trial_a_cuda_entropy(),
        trial_b_cuda_stencil(),
    ]


def partition(trial_id: str) -> dict[str, str]:
    """Return the four partition columns for a fixture trial_id."""
    project, arm, bench, direction, _, _ = trial_id.split("/")
    return {"project": project, "arm": arm, "bench": bench, "direction": direction, "trial_id": trial_id}


def files_under(root: Path) -> set[str]:
    """Return every file below root as a relative POSIX path."""
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def snapshot(root: Path) -> dict[str, bytes]:
    """Return the bytes of every file below root, keyed by relative POSIX path."""
    return {path: (root / path).read_bytes() for path in files_under(root)}


def is_string_type(value_type: pa.DataType) -> bool:
    """Return True for an Arrow string or large string type."""
    return pa.types.is_string(value_type) or pa.types.is_large_string(value_type)


def column_check(name: str) -> Callable[[pa.DataType], bool]:
    """Return a predicate for the Arrow type the contract gives a column."""
    if name in INT64_COLUMNS:
        return lambda t: t == pa.int64()
    if name in DOUBLE_COLUMNS:
        return lambda t: t == pa.float64()
    if name in BOOL_COLUMNS:
        return lambda t: t == pa.bool_()
    if name in LIST_DOUBLE_COLUMNS:
        return lambda t: pa.types.is_list(t) and t.value_type == pa.float64()
    if name in LIST_STRING_COLUMNS:
        return lambda t: pa.types.is_list(t) and is_string_type(t.value_type)
    return is_string_type


def assert_rows_identical(got: dict[str, list[dict[str, Any]]], want: dict[str, list[dict[str, Any]]]) -> None:
    """Assert equal rows with the same key order and the same Python type for every value."""
    assert set(got) == set(parquet.TABLES)
    assert got == want
    for table in parquet.TABLES:
        for got_row, want_row in zip(got[table], want[table], strict=True):
            assert list(got_row) == list(want_row), table
            for key, value in want_row.items():
                where = (table, key, got_row[key], value)
                assert type(got_row[key]) is type(value), where
                if isinstance(value, list):
                    assert [type(v) for v in got_row[key]] == [type(v) for v in value], where


# ---------------------------------------------------------------------------
# Expected rows, written out by hand from the contract


def expected_trial_row_a_omp_entropy() -> dict[str, Any]:
    """Return the trials row for trial_a_omp_entropy."""
    pins = {column: None for column in PIN_COLUMNS}
    pins["toolchain_pins_cuda"] = "fixture-cuda"
    return {
        **partition(ID_A_OMP_ENTROPY),
        "item": "entropy",
        "run": 1,
        "recipe_hash": RECIPE_HASH,
        **pins,
        "provenance_commit": FIXTURE_COMMIT,
        "provenance_dirty": False,
        "provenance_device": "none (compile only)",
        "provenance_sdk": None,
        "provenance_date": FIXTURE_DATE,
        "bench_item_suite": SUITE,
        "bench_item_item": "entropy",
        "bench_item_split": "eval",
        "bench_item_direction": "omp-cuda",
        "model_backend": "mock",
        "model_id": "mock-fixture",
        "model_sampling_temperature": 0.2,
        "model_sampling_top_p": 0.95,
        "model_sampling_max_tokens": 4096,
        **dict.fromkeys(REFERENCE_RUN_COLUMNS),
        "context_knowledge_summary": "summary\n",
        "context_source_description": "",
        "final_stage_reached": "S5",
        "final_alignment": 0.75,
        "final_score": None,
        "final_corrections": 1,
        "final_wall_s": None,
        **dict.fromkeys(END_REASON_COLUMNS),
        "attempt_count": 2,
    }


def expected_attempt_rows_a_omp_entropy() -> list[dict[str, Any]]:
    """Return the two attempts rows for trial_a_omp_entropy."""
    prompt, stdout = text_ref("prompt 0\n"), text_ref("PASS\n")
    unset_run = {
        "run_exit_code": None,
        "run_hang": None,
        "run_sim_ub": None,
        "run_wall_s": None,
        "run_stdout_ref_sha256": None,
        "run_stdout_ref_path": None,
        "run_outputs_ref_sha256": None,
        "run_outputs_ref_path": None,
        "run_stdout_truncated": None,
        "run_stderr_truncated": None,
        "run_workdir_incomplete": None,
    }
    unset_profile = {"profile_runtime_s": None, "profile_avg_power_w": None, "profile_energy_j": None}
    first = {
        **partition(ID_A_OMP_ENTROPY),
        "index": 0,
        "prompt_ref_sha256": prompt.sha256,
        "prompt_ref_path": prompt.path,
        "response_sha256": sha("response 0\n"),
        "files": '{"a.h": "// caf' + BACKSLASH + 'u00e9\\n", "main.cu": "int x;\\n"}',
        "diff_from_previous": "",
        "stage_reached": "S1",
        "diagnostic_count": 2,
        **unset_run,
        "alignment_per_input": [],
        "alignment_mean": None,
        **unset_profile,
        "guards_host_compute": None,
        "guards_harness_tamper": None,
        "guards_oracle_access": None,
        "score_components": "{}",
        "score_scalar": None,
    }
    second = {
        **partition(ID_A_OMP_ENTROPY),
        "index": 1,
        "prompt_ref_sha256": None,
        "prompt_ref_path": None,
        "response_sha256": SHA_EMPTY,
        "files": "{}",
        "diff_from_previous": "--- a/main.cu\n+++ /dev/null\n@@ -1 +0,0 @@\n-int x;\n",
        "stage_reached": "S5",
        "diagnostic_count": 0,
        **unset_run,
        "run_exit_code": 0,
        "run_hang": False,
        "run_wall_s": 0.5,
        "run_stdout_ref_sha256": stdout.sha256,
        "run_stdout_ref_path": stdout.path,
        "alignment_per_input": [1.0, 0.5],
        "alignment_mean": 0.75,
        **unset_profile,
        "guards_host_compute": False,
        "guards_harness_tamper": None,
        "guards_oracle_access": None,
        "score_components": '{"alignment": 0.75, "energy": null}',
        "score_scalar": None,
    }
    return [first, second]


def expected_diagnostic_rows_a_omp_entropy() -> list[dict[str, Any]]:
    """Return the diagnostics rows for trial_a_omp_entropy."""
    base = {**partition(ID_A_OMP_ENTROPY), "attempt_index": 0}
    return [
        {
            **base,
            "ordinal": 0,
            "stage": "compile",
            "severity": "error",
            "code": "20",
            "file": "main.cu",
            "line": 1,
            "column": 5,
            "message": "m0",
        },
        {
            **base,
            "ordinal": 1,
            "stage": "compile",
            "severity": "note",
            "code": None,
            "file": None,
            "line": None,
            "column": None,
            "message": "m1 | x",
        },
    ]


def expected_request_rows_a_omp_entropy() -> list[dict[str, Any]]:
    """Return the two requests rows for trial_a_omp_entropy."""
    summary, response = text_ref("summary\n"), text_ref("response 0\n")
    return [
        {
            **partition(ID_A_OMP_ENTROPY),
            "index": 0,
            "stage": "summarize_context",
            "attempt_index": None,
            "message_roles": ["system", "user"],
            "message_sha256": [sha("system general\n"), sha("summary request\n")],
            "reply_ref_sha256": summary.sha256,
            "reply_ref_path": summary.path,
            "diagnostic_count": 0,
        },
        {
            **partition(ID_A_OMP_ENTROPY),
            "index": 1,
            "stage": "generate",
            "attempt_index": 0,
            "message_roles": ["system", "user"],
            "message_sha256": [sha("system direction\n"), sha("prompt 0\n")],
            "reply_ref_sha256": response.sha256,
            "reply_ref_path": response.path,
            "diagnostic_count": 0,
        },
    ]


def expected_request_rows_b_omp_filled() -> list[dict[str, Any]]:
    """Return the three requests rows for trial_b_omp_filled."""
    requests = [
        ("describe_source", None, ["system", "user"], ["system b\n", "describe b\n"], "source b\n", 1),
        ("generate", 0, ["system", "user"], ["direction b\n", "prompt b0\n"], "response b0\n", 0),
        ("run_loop", 1, ["user"], ["prompt b1\n"], "response b1\n", 2),
    ]
    return [
        {
            **partition(ID_B_OMP_FILLED),
            "index": index,
            "stage": stage,
            "attempt_index": attempt_index,
            "message_roles": roles,
            "message_sha256": [sha(text) for text in texts],
            "reply_ref_sha256": text_ref(reply).sha256,
            "reply_ref_path": text_ref(reply).path,
            "diagnostic_count": count,
        }
        for index, (stage, attempt_index, roles, texts, reply, count) in enumerate(requests)
    ]


def expected_trial_row_b_omp_filled() -> dict[str, Any]:
    """Return the trials row for trial_b_omp_filled."""
    return {
        **partition(ID_B_OMP_FILLED),
        "item": "layout",
        "run": 3,
        "recipe_hash": RECIPE_HASH,
        **{f"toolchain_pins_{name}": f"pin-{name}" for name in record.TOOLCHAIN_PIN_NAMES},
        "provenance_commit": FILLED_COMMIT,
        "provenance_dirty": True,
        "provenance_device": "fixture-device",
        "provenance_sdk": "fixture-sdk",
        "provenance_date": FILLED_DATE,
        "bench_item_suite": SUITE,
        "bench_item_item": "layout",
        "bench_item_split": "eval",
        "bench_item_direction": "omp-cuda",
        "model_backend": "mock",
        "model_id": "mock-filled",
        "model_sampling_temperature": 0.7,
        "model_sampling_top_p": 0.9,
        "model_sampling_max_tokens": 512,
        "reference_run_exit_code": 7,
        "reference_run_hang": False,
        "reference_run_sim_ub": True,
        "reference_run_wall_s": 4.5,
        "reference_run_stdout_ref_sha256": text_ref("reference stdout b\n").sha256,
        "reference_run_stdout_ref_path": text_ref("reference stdout b\n").path,
        "reference_run_outputs_ref_sha256": text_ref("reference outputs b\n").sha256,
        "reference_run_outputs_ref_path": text_ref("reference outputs b\n").path,
        "reference_run_stdout_truncated": True,
        "reference_run_stderr_truncated": False,
        "reference_run_workdir_incomplete": False,
        "context_knowledge_summary": "knowledge b\n",
        "context_source_description": "source b\n",
        "final_stage_reached": "S4",
        "final_alignment": 0.25,
        "final_score": 0.5,
        "final_corrections": 1,
        "final_wall_s": 3.5,
        "final_end_reason_code": "correction-cap",
        "final_end_reason_message": "synthetic: the cap stopped the loop",
        "attempt_count": 2,
    }


def expected_attempt_row_b_omp_filled_0() -> dict[str, Any]:
    """Return the attempts row for attempt 0 of trial_b_omp_filled."""
    prompt, stdout, outputs = text_ref("prompt b0\n"), text_ref("stdout b0\n"), text_ref("outputs b0\n")
    return {
        **partition(ID_B_OMP_FILLED),
        "index": 0,
        "prompt_ref_sha256": prompt.sha256,
        "prompt_ref_path": prompt.path,
        "response_sha256": sha("response b0\n"),
        "files": '{"k.cu": "int z;\\n"}',
        "diff_from_previous": "",
        "stage_reached": "S3",
        "diagnostic_count": 1,
        "run_exit_code": 3,
        "run_hang": False,
        "run_sim_ub": True,
        "run_wall_s": 1.5,
        "run_stdout_ref_sha256": stdout.sha256,
        "run_stdout_ref_path": stdout.path,
        "run_outputs_ref_sha256": outputs.sha256,
        "run_outputs_ref_path": outputs.path,
        "run_stdout_truncated": False,
        "run_stderr_truncated": True,
        "run_workdir_incomplete": False,
        "alignment_per_input": [0.5],
        "alignment_mean": 0.625,
        "profile_runtime_s": 0.25,
        "profile_avg_power_w": 150.0,
        "profile_energy_j": 37.5,
        "guards_host_compute": True,
        "guards_harness_tamper": False,
        "guards_oracle_access": None,
        "score_components": '{"stage": 0.6}',
        "score_scalar": 0.125,
    }


def expected_attempt_row_b_omp_filled_1() -> dict[str, Any]:
    """Return the attempts row for attempt 1 of trial_b_omp_filled."""
    prompt, stdout, outputs = text_ref("prompt b1\n"), text_ref("stdout b1\n"), text_ref("outputs b1\n")
    return {
        **partition(ID_B_OMP_FILLED),
        "index": 1,
        "prompt_ref_sha256": prompt.sha256,
        "prompt_ref_path": prompt.path,
        "response_sha256": sha("response b1\n"),
        "files": '{"k.cu": "int w;\\n"}',
        "diff_from_previous": "--- a/k.cu\n+++ b/k.cu\n@@ -1 +1 @@\n-int z;\n+int w;\n",
        "stage_reached": "S4",
        "diagnostic_count": 0,
        "run_exit_code": -2,
        "run_hang": True,
        "run_sim_ub": False,
        "run_wall_s": 2.5,
        "run_stdout_ref_sha256": stdout.sha256,
        "run_stdout_ref_path": stdout.path,
        "run_outputs_ref_sha256": outputs.sha256,
        "run_outputs_ref_path": outputs.path,
        "run_stdout_truncated": False,
        "run_stderr_truncated": False,
        "run_workdir_incomplete": True,
        "alignment_per_input": [0.75, 0.25],
        "alignment_mean": 0.5,
        "profile_runtime_s": 0.5,
        "profile_avg_power_w": 75.0,
        "profile_energy_j": 37.25,
        "guards_host_compute": False,
        "guards_harness_tamper": None,
        "guards_oracle_access": True,
        "score_components": '{"alignment": 0.5, "stage": 0.8}',
        "score_scalar": 0.375,
    }


def expected_rows_b_omp_filled() -> dict[str, list[dict[str, Any]]]:
    """Return every table's rows for trial_b_omp_filled."""
    diagnostic = {
        **partition(ID_B_OMP_FILLED),
        "attempt_index": 0,
        "ordinal": 0,
        "stage": "jit",
        "severity": "warning",
        "code": "W7",
        "file": "k.cu",
        "line": 4,
        "column": 9,
        "message": "w0",
    }
    return {
        "trials": [expected_trial_row_b_omp_filled()],
        "attempts": [expected_attempt_row_b_omp_filled_0(), expected_attempt_row_b_omp_filled_1()],
        "diagnostics": [diagnostic],
        "requests": expected_request_rows_b_omp_filled(),
    }


# ---------------------------------------------------------------------------
# trial_rows


def test_table_and_partition_constants() -> None:
    assert parquet.TABLES == ("trials", "attempts", "diagnostics", "requests")
    assert parquet.PARTITION_COLUMNS == ("project", "arm", "bench", "direction")


def test_trial_rows_columns_in_order() -> None:
    rows = parquet.trial_rows(run_trials())
    assert set(rows) == set(parquet.TABLES)
    for table, columns in COLUMNS.items():
        assert rows[table], table
        for row in rows[table]:
            assert tuple(row) == columns, table


def test_trial_rows_values() -> None:
    rows = parquet.trial_rows([trial_a_omp_entropy()])
    assert rows["trials"] == [expected_trial_row_a_omp_entropy()]
    assert rows["attempts"] == expected_attempt_rows_a_omp_entropy()
    assert rows["diagnostics"] == expected_diagnostic_rows_a_omp_entropy()
    assert rows["requests"] == expected_request_rows_a_omp_entropy()


def test_trial_rows_values_with_every_field_set(tmp_path: Path) -> None:
    rows = parquet.trial_rows([trial_b_omp_filled()])
    assert_rows_identical(rows, expected_rows_b_omp_filled())
    out = tmp_path / "parquet"
    parquet.write_run_parquet([trial_b_omp_filled()], out)
    assert_rows_identical(parquet.read_run_parquet(out), expected_rows_b_omp_filled())


def test_trial_rows_for_trial_without_attempts() -> None:
    rows = parquet.trial_rows([trial_a_cuda_entropy()])
    assert rows["attempts"] == []
    assert rows["diagnostics"] == []
    assert rows["requests"] == [], "a trial that asked no model has no request rows"
    (row,) = rows["trials"]
    assert row["run"] == 2
    assert row["direction"] == "cuda-omp"
    assert row["attempt_count"] == 0
    assert row["final_corrections"] == 0
    unset = [*PIN_COLUMNS, "final_stage_reached", "final_alignment", "final_score", "final_wall_s"]
    assert all(row[column] is None for column in unset)
    provenance = [row[column] for column in PROVENANCE_COLUMNS]
    assert provenance == [None, None, "none (compile only)", None, FIXTURE_DATE]


def test_trials_schema_types_the_provenance_columns_explicitly() -> None:
    schema = parquet.SCHEMAS["trials"]
    assert set(PROVENANCE_COLUMNS) <= set(schema.names), schema.names
    start = schema.names.index("provenance_commit")
    end = start + len(PROVENANCE_COLUMNS)
    assert tuple(schema.names[start:end]) == PROVENANCE_COLUMNS
    assert (schema.names[start - 1], schema.names[end]) == (PIN_COLUMNS[-1], "bench_item_suite")
    types = [schema.field(name).type for name in PROVENANCE_COLUMNS]
    assert types == [pa.string(), pa.bool_(), pa.string(), pa.string(), pa.string()]


def test_provenance_columns_hold_each_trials_provenance_and_read_back(tmp_path: Path) -> None:
    trials = [*run_trials(), trial_b_omp_filled()]
    rows = parquet.trial_rows(trials)
    by_id = {trial.trial_id: trial.provenance for trial in trials}
    for row in rows["trials"]:
        provenance = by_id[row["trial_id"]]
        expected = [provenance.commit, provenance.dirty, provenance.device, provenance.sdk, provenance.date]
        assert [row[column] for column in PROVENANCE_COLUMNS] == expected, row["trial_id"]
    out = tmp_path / "parquet"
    parquet.write_run_parquet(trials, out)
    back = parquet.read_run_parquet(out)
    assert_rows_identical(back, rows)
    dirty = {row["trial_id"]: row["provenance_dirty"] for row in back["trials"]}
    assert (dirty[ID_A_CUDA_ENTROPY], dirty[ID_A_OMP_ENTROPY], dirty[ID_B_OMP_FILLED]) == (None, False, True)


def test_trial_rows_sorted() -> None:
    forward = parquet.trial_rows(run_trials())
    backward = parquet.trial_rows(list(reversed(run_trials())))
    assert backward == forward
    ids = [row["trial_id"] for row in forward["trials"]]
    assert ids == sorted(ids)
    assert len(ids) == 5
    attempt_keys = [(row["trial_id"], row["index"]) for row in forward["attempts"]]
    assert attempt_keys == sorted(attempt_keys)
    assert len(attempt_keys) == 5
    diagnostic_keys = [(row["trial_id"], row["attempt_index"], row["ordinal"]) for row in forward["diagnostics"]]
    assert diagnostic_keys == sorted(diagnostic_keys)
    assert len(diagnostic_keys) == 3
    request_keys = [(row["trial_id"], row["index"]) for row in forward["requests"]]
    assert request_keys == sorted(request_keys)
    assert len(request_keys) == 3


def test_a_trial_whose_requests_were_not_recorded_has_no_request_rows() -> None:
    assert trial_a_omp_layout().requests is None
    assert parquet.trial_rows([trial_a_omp_layout()])["requests"] == []
    rows = parquet.trial_rows(run_trials())["requests"]
    assert {row["trial_id"] for row in rows} == {ID_A_OMP_ENTROPY, ID_B_CUDA_STENCIL}


def test_trial_rows_of_nothing() -> None:
    assert parquet.trial_rows([]) == {table: [] for table in parquet.TABLES}


# ---------------------------------------------------------------------------
# write_run_parquet and read_run_parquet


def test_hive_layout(tmp_path: Path) -> None:
    out = tmp_path / "parquet"
    parquet.write_run_parquet(run_trials(), out)
    assert files_under(out) == {
        f"trials/{PARTITION_A_OMP}/part-0.parquet",
        f"trials/{PARTITION_B_OMP}/part-0.parquet",
        f"trials/{PARTITION_A_CUDA}/part-0.parquet",
        f"trials/{PARTITION_B_CUDA}/part-0.parquet",
        f"attempts/{PARTITION_A_OMP}/part-0.parquet",
        f"attempts/{PARTITION_B_OMP}/part-0.parquet",
        f"attempts/{PARTITION_B_CUDA}/part-0.parquet",
        f"diagnostics/{PARTITION_A_OMP}/part-0.parquet",
        f"diagnostics/{PARTITION_B_CUDA}/part-0.parquet",
        f"requests/{PARTITION_A_OMP}/part-0.parquet",
        f"requests/{PARTITION_B_CUDA}/part-0.parquet",
    }


def test_round_trip_is_exact(tmp_path: Path) -> None:
    trials = run_trials()
    out = tmp_path / "parquet"
    parquet.write_run_parquet(trials, out)
    assert_rows_identical(parquet.read_run_parquet(out), parquet.trial_rows(trials))


def test_round_trip_keeps_partition_values_as_strings(tmp_path: Path) -> None:
    out = tmp_path / "parquet"
    parquet.write_run_parquet([trial_b_cuda_stencil()], out)
    back = parquet.read_run_parquet(out)
    for table in parquet.TABLES:
        (row,) = back[table]
        assert row["bench"] == "2024", table
        assert row["arm"] == "arm-b", table


def test_parquet_files_use_contract_types(tmp_path: Path) -> None:
    out = tmp_path / "parquet"
    parquet.write_run_parquet(run_trials(), out)
    for table, columns in COLUMNS.items():
        paths = sorted((out / table).rglob("*.parquet"))
        assert paths, table
        for path in paths:
            schema = pq.read_schema(path)
            missing = set(columns) - set(parquet.PARTITION_COLUMNS) - set(schema.names)
            assert not missing, (path, missing)
            for field in schema:
                assert field.name in columns, (path, field.name)
                assert column_check(field.name)(field.type), (path, field.name, field.type)


def test_table_without_rows_gets_no_directory(tmp_path: Path) -> None:
    out = tmp_path / "parquet"
    parquet.write_run_parquet([trial_a_cuda_entropy()], out)
    assert (out / "trials").is_dir()
    assert not (out / "attempts").exists()
    assert not (out / "diagnostics").exists()
    assert not (out / "requests").exists()
    back = parquet.read_run_parquet(out)
    assert back == parquet.trial_rows([trial_a_cuda_entropy()])
    assert back["attempts"] == []
    assert back["diagnostics"] == []
    assert back["requests"] == []


def test_empty_run(tmp_path: Path) -> None:
    out = tmp_path / "parquet"
    parquet.write_run_parquet([], out)
    assert not out.exists() or files_under(out) == set()
    assert parquet.read_run_parquet(out) == {table: [] for table in parquet.TABLES}


def test_read_missing_directory(tmp_path: Path) -> None:
    assert parquet.read_run_parquet(tmp_path / "absent") == {table: [] for table in parquet.TABLES}


def test_rewrite_replaces_tables(tmp_path: Path) -> None:
    out = tmp_path / "parquet"
    parquet.write_run_parquet(run_trials(), out)
    (out / "trials" / "stale.txt").write_text("stale\n", encoding="ascii")
    (out / "notes.txt").write_text("keep\n", encoding="ascii")
    parquet.write_run_parquet([trial_b_omp_entropy()], out)
    assert files_under(out) == {
        "notes.txt",
        f"trials/{PARTITION_B_OMP}/part-0.parquet",
        f"attempts/{PARTITION_B_OMP}/part-0.parquet",
    }
    assert_rows_identical(parquet.read_run_parquet(out), parquet.trial_rows([trial_b_omp_entropy()]))


@pytest.mark.parametrize("builder", [trial_a_omp_entropy, trial_b_omp_entropy, trial_b_cuda_stencil])
def test_single_trial_round_trip(tmp_path: Path, builder: Callable[[], record.Trial]) -> None:
    out = tmp_path / "parquet"
    parquet.write_run_parquet([builder()], out)
    assert_rows_identical(parquet.read_run_parquet(out), parquet.trial_rows([builder()]))


def test_plus_in_a_partition_value_is_uri_encoded_and_read_back(tmp_path: Path) -> None:
    trial = make_trial(f"{PROJECT}/Model_v1.2+q4/{SUITE}/omp-cuda/entropy/run01", [])
    out = tmp_path / "parquet"
    parquet.write_run_parquet([trial], out)
    assert files_under(out) == {
        f"trials/project={PROJECT}/arm=Model_v1.2%2Bq4/bench={SUITE}/direction=omp-cuda/part-0.parquet"
    }
    back = parquet.read_run_parquet(out)
    assert back["trials"][0]["arm"] == "Model_v1.2+q4"
    assert_rows_identical(back, parquet.trial_rows([trial]))


def test_int_values_in_float_columns_round_trip_as_floats(tmp_path: Path) -> None:
    attempt = record.Attempt(
        index=0,
        stage_reached="S5",
        run=record.RunInfo(exit_code=0, wall_s=1),
        alignment=record.Alignment(per_input=[1, 0], mean=1),
        profile=record.Profile(runtime_s=2, avg_power_w=0, energy_j=3),
        score=record.ScoreBreakdown(components={"stage": 1}, scalar=1),
    )
    trial = make_trial(
        ID_A_OMP_LAYOUT,
        [attempt],
        model=record.ModelInfo(
            backend="mock", id="m", sampling=interfaces.Sampling(temperature=0, top_p=1, max_tokens=16)
        ),
        reference_run=record.RunInfo(exit_code=0, wall_s=3),
        final=record.Final(stage_reached="S5", alignment=1, score=1, wall_s=2),
    )
    rows = parquet.trial_rows([trial])
    for table in ("trials", "attempts"):
        for column, value in rows[table][0].items():
            if column in DOUBLE_COLUMNS:
                assert type(value) is float, (table, column, value)
    assert rows["attempts"][0]["alignment_per_input"] == [1.0, 0.0]
    assert rows["attempts"][0]["score_components"] == '{"stage": 1.0}'
    out = tmp_path / "parquet"
    parquet.write_run_parquet([trial], out)
    assert_rows_identical(parquet.read_run_parquet(out), rows)


@pytest.mark.parametrize(
    "arms",
    [
        pytest.param(("Qwen", "qwen"), id="letter-case"),
        pytest.param(("arm-a", "ARM-A"), id="letter-case-upper"),
        pytest.param(("arm.",), id="trailing-dot"),
        pytest.param(("arm", "arm."), id="trailing-dot-beside-plain"),
    ],
)
def test_partition_values_that_would_share_a_directory_are_refused(tmp_path: Path, arms: tuple[str, ...]) -> None:
    out = tmp_path / "parquet"
    parquet.write_run_parquet(run_trials(), out)
    before = snapshot(out)
    trials = [make_trial(f"{PROJECT}/{arm}/{SUITE}/omp-cuda/entropy/run01", []) for arm in arms]
    with pytest.raises(ValueError) as info:
        parquet.write_run_parquet(trials, out)
    assert arms[-1] in str(info.value)
    assert snapshot(out) == before


@pytest.mark.parametrize(
    "trial_ids",
    [
        pytest.param(
            (f"{PROJECT}/Qwen/{SUITE}/omp-cuda/entropy/run01", f"{PROJECT}/qwen/{SUITE}/cuda-omp/entropy/run01"),
            id="arm-other-direction",
        ),
        pytest.param(
            (f"{PROJECT}/Qwen/b1/omp-cuda/entropy/run01", f"{PROJECT}/qwen/b2/omp-cuda/entropy/run01"),
            id="arm-other-bench",
        ),
        pytest.param(
            (f"{PROJECT}/arm-a/Bench/omp-cuda/entropy/run01", f"{PROJECT}/arm-a/bench/cuda-omp/entropy/run01"),
            id="bench-other-direction",
        ),
        pytest.param(
            (f"Proj/arm-a/{SUITE}/omp-cuda/entropy/run01", f"proj/arm-b/{SUITE}/cuda-omp/entropy/run01"),
            id="project-other-arm",
        ),
    ],
)
def test_partition_prefixes_that_differ_only_by_case_are_refused(tmp_path: Path, trial_ids: tuple[str, str]) -> None:
    out = tmp_path / "parquet"
    parquet.write_run_parquet(run_trials(), out)
    before = snapshot(out)
    with pytest.raises(ValueError) as info:
        parquet.write_run_parquet([make_trial(trial_id, []) for trial_id in trial_ids], out)
    first, second = (trial_id.split("/") for trial_id in trial_ids)
    level = next(i for i in range(len(parquet.PARTITION_COLUMNS)) if first[i] != second[i])
    assert repr(first[level]) in str(info.value) and repr(second[level]) in str(info.value)
    assert snapshot(out) == before


def test_case_differences_under_different_parents_round_trip(tmp_path: Path) -> None:
    trials = [
        make_trial(f"{PROJECT}/arm-a/Bench/omp-cuda/entropy/run01", []),
        make_trial(f"{PROJECT}/arm-b/bench/omp-cuda/entropy/run01", []),
        make_trial(f"{PROJECT}/arm-b/Stencil/cuda-omp/entropy/run01", []),
        make_trial(f"{PROJECT}/arm-a/stencil/omp-cuda/entropy/run01", []),
    ]
    out = tmp_path / "parquet"
    parquet.write_run_parquet(trials, out)
    assert_rows_identical(parquet.read_run_parquet(out), parquet.trial_rows(trials))


def attempt_trial(**attempt_fields: Any) -> record.Trial:
    """Return a one-attempt trial for ID_A_OMP_LAYOUT whose attempt has the given fields."""
    attempt_fields.setdefault("stage_reached", "S5")
    return make_trial(ID_A_OMP_LAYOUT, [record.Attempt(index=0, **attempt_fields)])


def diagnostic_trial(**diagnostic_fields: Any) -> record.Trial:
    """Return a one-attempt trial whose attempt holds one diagnostic with the given fields."""
    diagnostic = record.Diagnostic(stage="compile", severity="error", message="m", **diagnostic_fields)
    return attempt_trial(diagnostics=[diagnostic])


def sampling_trial(max_tokens: int) -> record.Trial:
    """Return a trial without attempts whose sampling has the given max_tokens."""
    sampling = interfaces.Sampling(temperature=0.2, top_p=0.95, max_tokens=max_tokens)
    return make_trial(ID_A_OMP_LAYOUT, [], model=record.ModelInfo(backend="mock", id="m", sampling=sampling))


OUT_OF_INT64 = [
    pytest.param(lambda: attempt_trial(run=record.RunInfo(exit_code=2**64)), "run_exit_code", id="exit-code"),
    pytest.param(lambda: attempt_trial(run=record.RunInfo(exit_code=INT64_MIN - 1)), "run_exit_code", id="exit-low"),
    pytest.param(lambda: sampling_trial(INT64_MAX + 1), "model_sampling_max_tokens", id="max-tokens"),
    pytest.param(
        lambda: make_trial(ID_A_OMP_LAYOUT, [], final=record.Final(corrections=INT64_MAX + 1)),
        "final_corrections",
        id="corrections",
    ),
    pytest.param(lambda: diagnostic_trial(file="k.cu", line=INT64_MAX + 1), "line", id="diagnostic-line"),
    pytest.param(
        lambda: make_trial(f"{PROJECT}/arm-a/{SUITE}/omp-cuda/layout/run{INT64_MAX + 1}", []), "run", id="run-number"
    ),
]


@pytest.mark.parametrize(("build", "column"), OUT_OF_INT64)
def test_ints_outside_int64_are_refused_before_anything_changes(
    tmp_path: Path, build: Callable[[], record.Trial], column: str
) -> None:
    out = tmp_path / "parquet"
    parquet.write_run_parquet(run_trials(), out)
    before = snapshot(out)
    with pytest.raises(ValueError) as info:
        parquet.write_run_parquet([trial_b_omp_entropy(), build()], out)
    assert f".{column} of trial" in str(info.value)
    assert snapshot(out) == before


def test_int64_bounds_round_trip(tmp_path: Path) -> None:
    trials = [
        attempt_trial(
            run=record.RunInfo(exit_code=INT64_MIN),
            diagnostics=[
                record.Diagnostic(stage="run", severity="note", file="k.cu", line=INT64_MAX, column=0, message="m")
            ],
        ),
        make_trial(ID_B_OMP_ENTROPY, [], final=record.Final(corrections=INT64_MAX)),
    ]
    out = tmp_path / "parquet"
    parquet.write_run_parquet(trials, out)
    back = parquet.read_run_parquet(out)
    assert_rows_identical(back, parquet.trial_rows(trials))
    assert back["attempts"][0]["run_exit_code"] == INT64_MIN
    assert back["diagnostics"][0]["line"] == INT64_MAX


def test_a_value_arrow_cannot_encode_leaves_every_table_unchanged(tmp_path: Path) -> None:
    out = tmp_path / "parquet"
    parquet.write_run_parquet(run_trials(), out)
    before = snapshot(out)
    # A lone surrogate passes the record checks but has no UTF-8 form. It sits in
    # the diagnostics table, which is written after the trials and attempts
    # tables, so those must not be replaced before the failure.
    bad = diagnostic_trial(file="k.cu", line=1, code="a\ud800b")
    with pytest.raises(ValueError) as info:
        parquet.write_run_parquet([trial_b_omp_entropy(), bad], out)
    assert "diagnostics" in str(info.value)
    assert snapshot(out) == before
