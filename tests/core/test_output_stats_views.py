"""Tests for how trial.md and the Parquet mirror show the per-output statistics (task P4.4).

Bible: Result Record (Storage; the trial.md and Parquet views of the
record), Readability Standards (Trial row), Oracles (binary_io row), Design
Principle 7. Plan: plans/p4-ttsim.md, P4.4.

The contract these tests fix:

- trial.md shows the baseline's reference agreement (Trial.reference_agreement)
  in the trial-level part of the page, before the first attempt section, and
  each attempt's statistics (Attempt.alignment.outputs) in that attempt's
  section: one Markdown table row per output whose cells hold the output
  name, pcc, max_abs, max_ulp, and passed as trial.md formats values (floats
  by repr, ints as digits, bools as true or false), and the output's note.
- lassi.core.parquet gains the table `output_stats` (in TABLES): one row per
  output statistic, Hive-partitioned like the other tables, with at least
  the columns trial_id, attempt_index (null for a row of the baseline's
  reference agreement), name, pcc, max_abs, max_ulp, passed, and note. A
  trial with no statistics has no rows. The rows round-trip through
  write_run_parquet and read_run_parquet.

Every value here is SYNTHETIC and written for these tests; none is a
measurement.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from lassi.core import parquet
from lassi.core import record as record_module
from lassi.core.interfaces import Sampling
from lassi.core.record import (
    Alignment,
    Attempt,
    BenchItem,
    Final,
    ModelInfo,
    Provenance,
    RunInfo,
    Trial,
    make_trial_id,
)
from lassi.core.store import TextStore
from lassi.core.trial_md import render_trial_md

SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
# A commit id for synthetic provenance; not a commit of this repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SUITE = "binio-fixture"
TRIAL_ID = make_trial_id("binio-views", "scripted-fixture", SUITE, "cpu-tt", "vadd", 1)
SHA_C = "a" * 64
SHA_CAND = "b" * 64
MISSING_NOTE = "SYNTHETIC note: d is missing from the candidate's outputs"
PAST_NOTE = "SYNTHETIC note: c is past its threshold"
# The SYNTHETIC statistics, as (name, pcc, max_abs, max_ulp, passed, note).
AGREEMENT = [("c", 0.96875, 0.001953125, 17, True, None)]
ATTEMPT_STATS = [("c", 0.75, 0.125, 4096, False, PAST_NOTE), ("d", None, None, None, False, MISSING_NOTE)]


def require_field(cls: type, name: str) -> None:
    """Fail the test clearly while the record class `cls` has no field `name`."""
    if name not in {spec.name for spec in dataclasses.fields(cls)}:
        pytest.fail(f"{cls.__name__} has no {name} field; task P4.4 adds it to the Result Record")


def stats_of(rows: list[tuple[Any, ...]]) -> list[Any]:
    """Return OutputStats records for (name, pcc, max_abs, max_ulp, passed, note) rows."""
    cls = getattr(record_module, "OutputStats", None)
    if cls is None:
        pytest.fail("lassi.core.record has no OutputStats; task P4.4 adds the per-output statistics record")
    keys = ("name", "pcc", "max_abs", "max_ulp", "passed", "note")
    return [cls(**dict(zip(keys, row, strict=True))) for row in rows]


def views_trial(store: TextStore, *, with_stats: bool = True) -> Trial:
    """Return a SYNTHETIC trial with one attempt; `with_stats` sets the agreement and the attempt's statistics."""
    require_field(RunInfo, "outputs")
    require_field(Alignment, "outputs")
    require_field(Trial, "reference_agreement")

    def run(outputs: dict[str, str]) -> RunInfo:
        return RunInfo(
            exit_code=0, hang=False, wall_s=1.25, stdout_ref=store.put("SYNTHETIC stdout\n"), outputs=outputs,
            stdout_truncated=False, stderr_truncated=False, workdir_incomplete=False,
        )

    alignment = Alignment(per_input=[0.0], mean=0.0, outputs=stats_of(ATTEMPT_STATS) if with_stats else None)
    attempt = Attempt(
        index=0,
        prompt_ref=store.put("SYNTHETIC prompt\n"),
        response_text="SYNTHETIC reply",
        files={"host.cpp": "int main() { return 0; }\n"},
        stage_reached="S5",
        run=run({"c.lassiio": SHA_CAND}),
        alignment=alignment,
    )
    return Trial(
        trial_id=TRIAL_ID,
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(
            commit=FAKE_COMMIT, dirty=False, device="SYNTHETIC device", sdk=None, date="2026-09-25T00:00:00+00:00"
        ),
        bench_item=BenchItem(suite=SUITE, item="vadd", split="eval", direction="cpu-tt"),
        model=ModelInfo(backend="scripted", id="scripted-fixture", sampling=SAMPLING),
        reference_run=run({"c.lassiio": SHA_C}),
        reference_agreement=stats_of(AGREEMENT) if with_stats else None,
        requests=[],
        attempts=[attempt],
        final=Final(stage_reached="S5", alignment=0.0, corrections=0),
    )


def cells_of(line: str) -> list[str]:
    """Return the stripped cells of one Markdown table row."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def row_offsets(page: str, expected: list[str]) -> list[int]:
    """Return the character offset of every table row of `page` whose cells include each of `expected`."""
    offsets, position = [], 0
    for line in page.splitlines(keepends=True):
        if line.startswith("|") and set(expected) <= set(cells_of(line)):
            offsets.append(position)
        position += len(line)
    return offsets


def shown(value: Any) -> str:
    """Return a statistic as trial.md formats it: floats by repr, ints as digits, bools as true or false."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value) if isinstance(value, float) else str(value)


# ---------------------------------------------------------------------------
# trial.md


def test_trial_md_shows_the_reference_agreement_before_the_attempts(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "store")
    page = render_trial_md(views_trial(store), store)
    first_attempt = page.index("## Attempt 0")
    name, pcc, max_abs, max_ulp, passed, _ = AGREEMENT[0]
    rows = row_offsets(page, [name, shown(pcc), shown(max_abs), shown(max_ulp), shown(passed)])
    assert rows, "trial.md shows no row of the reference agreement"
    assert min(rows) < first_attempt, "the reference agreement belongs to the trial-level part of the page"


def test_trial_md_shows_each_attempts_statistics_in_its_section(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "store")
    page = render_trial_md(views_trial(store), store)
    first_attempt = page.index("## Attempt 0")
    name, pcc, max_abs, max_ulp, passed, note = ATTEMPT_STATS[0]
    rows = row_offsets(page, [name, shown(pcc), shown(max_abs), shown(max_ulp), shown(passed)])
    assert rows and max(rows) > first_attempt, "trial.md shows no row of attempt 0's statistics for c"
    assert row_offsets(page, ["d", "false"]), "trial.md shows no row for the missing output d"
    for text in (note, MISSING_NOTE):
        assert text in page[first_attempt:], f"trial.md does not show the note {text!r} in attempt 0's section"


def test_trial_md_renders_a_trial_without_statistics(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "store")
    page = render_trial_md(views_trial(store, with_stats=False), store)
    assert "## Attempt 0" in page
    assert not row_offsets(page, ["c", shown(0.125)]), "a trial without statistics shows none"


# ---------------------------------------------------------------------------
# Parquet


def stats_rows(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    """Return output_stats rows as (attempt_index, name, pcc, max_abs, max_ulp, passed, note), agreement first."""
    keys = ("attempt_index", "name", "pcc", "max_abs", "max_ulp", "passed", "note")
    picked = [tuple(row[key] for key in keys) for row in rows]
    return sorted(picked, key=lambda row: (row[0] is not None, row[0] or 0, row[1]))


def test_the_parquet_mirror_has_an_output_stats_table() -> None:
    assert "output_stats" in parquet.TABLES


def test_output_stats_rows_hold_one_row_per_output_with_no_attempt_for_the_agreement(tmp_path: Path) -> None:
    trial = views_trial(TextStore(tmp_path / "store"))
    rows = parquet.trial_rows([trial]).get("output_stats")
    assert rows is not None, "trial_rows gives no output_stats table"
    assert all(row["trial_id"] == TRIAL_ID for row in rows)
    for column in ("project", "arm", "bench", "direction"):
        assert all(column in row for row in rows), f"output_stats rows lack the partition column {column}"
    expected = [(None, *AGREEMENT[0]), *((0, *entry) for entry in ATTEMPT_STATS)]
    assert stats_rows(rows) == expected


def test_output_stats_rows_round_trip_through_parquet(tmp_path: Path) -> None:
    trial = views_trial(TextStore(tmp_path / "store"))
    out = tmp_path / "parquet"
    parquet.write_run_parquet([trial], out)
    written = parquet.trial_rows([trial])["output_stats"]
    read = parquet.read_run_parquet(out)["output_stats"]
    assert stats_rows(read) == stats_rows(written)
    assert len(read) == 3


def test_a_trial_without_statistics_has_no_output_stats_rows(tmp_path: Path) -> None:
    trial = views_trial(TextStore(tmp_path / "store"), with_stats=False)
    assert parquet.trial_rows([trial]).get("output_stats") == []
