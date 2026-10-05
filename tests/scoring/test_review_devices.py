"""review.md's source-run rows for executors per language (task P4.13; PHASE-NOTES P4, the P4.5 commit audit).

A run whose recipe binds one executor records `executor` (a name) and
`device` in provenance.json; a run with executors per language records
`executor` as a language -> name mapping and `devices` per language (bible
Project Recipes, Notes; Result Record). review.md's Source run table shows
the first as the Executor and Device rows, as before, and the second as
run.md does: one Executor row naming each language's executor, then one
Device row per language, `device (<language>)`, never a Python mapping and
never a Device row of `-`. The manifests are SYNTHETIC; no value here is a
measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lassi.core.store import TextStore
from lassi.scoring.score_run import review_markdown


def manifest(**fields: Any) -> dict[str, Any]:
    """Return a SYNTHETIC run manifest with the fields review.md shows, updated with `fields`."""
    data: dict[str, Any] = {
        "run_id": "fixture-run", "status": "complete", "recipe": "fixture", "recipe_hash": "a" * 64,
        "commit": "b" * 40, "dirty": False, "driver": None,
    }
    data.update(fields)
    return data


def source_rows(page: str) -> list[str]:
    """Return the table rows of review.md's Source run section."""
    section = page.split("## Source run\n", 1)[1].split("## Profiles\n", 1)[0]
    return [line for line in section.splitlines() if line.startswith("| ") and "---" not in line]


def test_a_single_executor_keeps_its_executor_and_device_rows(tmp_path: Path) -> None:
    page = review_markdown([], [], manifest(executor="native", device="host CPU (native): SYNTHETIC"), [],
                           TextStore(tmp_path))
    rows = source_rows(page)
    assert "| executor | native |" in rows
    assert "| device | host CPU (native): SYNTHETIC |" in rows


def test_executors_per_language_show_each_executor_and_each_device(tmp_path: Path) -> None:
    data = manifest(
        executor={"tt": "ttsim", "cpp": "native"},
        devices={"tt": "ttsim v1.3.4 (SYNTHETIC)", "cpp": "host CPU (native): SYNTHETIC"},
    )
    rows = source_rows(review_markdown([], [], data, [], TextStore(tmp_path)))
    assert "| executor | cpp: native; tt: ttsim |" in rows
    assert "| device (cpp) | host CPU (native): SYNTHETIC |" in rows
    assert "| device (tt) | ttsim v1.3.4 (SYNTHETIC) |" in rows
    assert not any(row.startswith("| device |") for row in rows), "no single Device row for a per-language run"
    assert not any("{" in row for row in rows), "no Python mapping in a cell"
