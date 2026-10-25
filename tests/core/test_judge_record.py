"""Tests for judge records: Trial.judgements, its JSON, trial.md, and its Parquet table (task P15.1).

Bible: Result Record (Trial; Storage), Agents And Judges (the Judge
interface; Judge rules: a judged value carries the [JUDGED] tag and is
never presented as a measurement), Readability Standards (Trial row),
Design Principle 7 (Parquet mirrors the JSON records), Agent Rule 14. Plan:
plans/p15-judges.md, P15.1.

The contract these tests fix:

- lassi.core.record holds JudgeRecord, a frozen, keyword-only record with
  exactly the fields kind (the Judge's registry name), rubric and
  rubric_sha256, model and family, mode (None for a judge that declares no
  modes), verdict, score, values (name -> finite number), and rationale,
  in that order, every one required. It refuses, with a ValueError naming
  the field: an empty kind, rubric, model, family, or verdict, a mode that
  is an empty string, a rubric_sha256 that is not 64 lowercase hex
  characters, a score outside [0, 1] or not finite, a value name that is
  empty or a value that is not finite, and a rationale that is not plain
  ASCII (printable characters, line feeds, and tabs).
- Trial.judgements is the Trial's last field, after adversary: a list of
  JudgeRecord or None, None by default. None means not recorded, so a
  trial.json written before the field loads with it None.
- A trial with judgements round-trips through to_json and from_json and
  through write_trial and read_trial.
- trial.md shows a "## Judgements" section for a trial whose judgements
  are recorded (after every other section), stating that each value there
  is [JUDGED] and never a measurement: one "### <kind> [JUDGED]" part per
  record, with a table of its rubric, rubric_sha256, model, family, mode
  ("none" when the judge declares no modes), verdict, and score, the
  judged cells (verdict, score, and each value) tagged [JUDGED], a table
  of its values when it has any, and its rationale in a fenced block. An
  empty list shows "None."; None shows no section, so a page without
  judgements is unchanged.
- The Parquet mirror gains the table judgements: one row per JudgeRecord,
  the key columns, ordinal (its position in Trial.judgements), kind,
  rubric, rubric_sha256, model, family, mode, verdict, score (a double),
  values (JSON text with sorted keys), rationale, and tag, which holds
  "[JUDGED]" on every row. A trial without judgements has no row.

Every value here is a SYNTHETIC fixture value, never a measurement.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pytest

from lassi.core import parquet, record
from lassi.core.interfaces import Sampling
from lassi.core.record import BenchItem, ModelInfo, Provenance, Trial, from_dict, from_json, to_json
from lassi.core.store import TextStore, read_trial, write_trial
from lassi.core.trial_md import render_trial_md

TRIAL_ID = record.make_trial_id("judge-record", "scripted-fixture", "judge-suite", "omp-cpp", "scale", 1)
OTHER_ID = record.make_trial_id("judge-record", "scripted-fixture", "judge-suite", "omp-cpp", "plainish", 1)
RUBRIC_SHA = "c" * 64
RATIONALE = "SYNTHETIC: both stdouts print the same sums.\nThe loop bounds match."
FIELDS = (
    "kind",
    "rubric",
    "rubric_sha256",
    "model",
    "family",
    "mode",
    "verdict",
    "score",
    "values",
    "rationale",
)
TABLE_COLUMNS = (
    *parquet.PARTITION_COLUMNS,
    "trial_id",
    "ordinal",
    "kind",
    "rubric",
    "rubric_sha256",
    "model",
    "family",
    "mode",
    "verdict",
    "score",
    "values",
    "rationale",
    "tag",
)


def need(name: str) -> Any:
    """Return lassi.core.record's attribute `name`, failing with the task's name when it does not exist yet."""
    assert hasattr(record, name), f"lassi.core.record has no {name} (task P15.1)"
    return getattr(record, name)


def base_trial(trial_id: str = TRIAL_ID) -> Trial:
    """Return a SYNTHETIC trial with no attempt and no judgements."""
    item = record.parse_trial_id(trial_id).item
    return Trial(
        trial_id=trial_id,
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=None, dirty=None, device=None, sdk=None, date="2026-10-25T00:00:00+00:00"),
        bench_item=BenchItem(suite="judge-suite", item=item, split="unassigned", direction="omp-cpp"),
        model=ModelInfo(backend="scripted", id="scripted-fixture", sampling=Sampling(0.2, 0.9, 8)),
    )


def equivalence(**changes: Any) -> Any:
    """Return a SYNTHETIC JudgeRecord of a judge that declares no modes."""
    values: dict[str, Any] = {
        "kind": "equivalence",
        "rubric": "synthetic-v1",
        "rubric_sha256": RUBRIC_SHA,
        "model": "SYNTHETIC-judge",
        "family": "synthetic-b",
        "mode": None,
        "verdict": "AGREE",
        "score": 0.875,
        "values": {},
        "rationale": RATIONALE,
    }
    return need("JudgeRecord")(**{**values, **changes})


def efficiency() -> Any:
    """Return a SYNTHETIC JudgeRecord of a judge in a mode, with named judged values."""
    return equivalence(kind="efficiency", mode="predict", verdict="DISAGREE", score=0.25, values={"delta": -0.5})


def judged_trial(trial_id: str = TRIAL_ID) -> Trial:
    """Return base_trial with both SYNTHETIC judge records."""
    return dataclasses.replace(base_trial(trial_id), judgements=[equivalence(), efficiency()])


def section(page: str) -> str:
    """Return the page's "## Judgements" section, up to the next "## " heading or the end."""
    assert "## Judgements\n" in page, "trial.md has no Judgements section"
    start = page.index("## Judgements\n")
    end = page.find("\n## ", start + 1)
    return page[start:] if end == -1 else page[start:end]


def table_rows(text: str) -> list[list[str]]:
    """Return every Markdown table row of `text` as its cells, stripped of blanks and backticks."""
    return [
        [cell.strip().strip("`") for cell in line[1:-1].split("|")]
        for line in text.splitlines()
        if line.startswith("|") and line.endswith("|")
    ]


def has_row(text: str, *cells: str) -> bool:
    """Return True when some table row of `text` holds every one of `cells`."""
    return any(all(cell in row for cell in cells) for row in table_rows(text))


# ---------------------------------------------------------------------------
# The record class and Trial.judgements


def test_the_record_is_frozen_keyword_only_with_its_fields() -> None:
    cls = need("JudgeRecord")
    assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen
    specs = dataclasses.fields(cls)
    assert tuple(spec.name for spec in specs) == FIELDS
    assert all(spec.kw_only for spec in specs)
    required = [spec.name for spec in specs if spec.default is dataclasses.MISSING]
    assert tuple(required) == FIELDS, "every field is required"


def test_trial_judgements_is_the_last_field_and_null_by_default() -> None:
    names = [spec.name for spec in dataclasses.fields(Trial)]
    assert names[-2:] == ["adversary", "judgements"], "Trial.judgements follows Trial.adversary (task P15.1)"
    assert base_trial().judgements is None
    assert judged_trial().judgements == [equivalence(), efficiency()]


def test_a_record_keeps_a_float_score_and_values() -> None:
    found = equivalence(score=1, values={"delta": 2})
    assert found.score == 1.0 and type(found.score) is float
    assert found.values == {"delta": 2.0} and type(found.values["delta"]) is float


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"kind": ""}, "kind"),
        ({"rubric": ""}, "rubric"),
        ({"rubric_sha256": "C" * 64}, "rubric_sha256"),
        ({"rubric_sha256": "c" * 63}, "rubric_sha256"),
        ({"model": ""}, "model"),
        ({"family": ""}, "family"),
        ({"mode": ""}, "mode"),
        ({"verdict": ""}, "verdict"),
        ({"score": -0.01}, "score"),
        ({"score": 1.5}, "score"),
        ({"score": math.nan}, "score"),
        ({"score": True}, "score"),
        ({"values": {"delta": math.inf}}, "values"),
        ({"values": {"": 1.0}}, "values"),
        ({"values": {"delta": "1"}}, "values"),
        ({"rationale": "caf\N{LATIN SMALL LETTER E WITH ACUTE}"}, "rationale"),
        ({"rationale": "a\rb"}, "rationale"),
        ({"rationale": 3}, "rationale"),
    ],
)
def test_the_record_refuses_values_outside_its_rules(changes: dict[str, Any], field: str) -> None:
    with pytest.raises(ValueError) as caught:
        equivalence(**changes)
    assert field in str(caught.value), str(caught.value)


# ---------------------------------------------------------------------------
# JSON and the trial files


def test_judgements_round_trip_through_json_and_the_trial_files(tmp_path: Path) -> None:
    trial = judged_trial()
    assert from_json(Trial, to_json(trial)) == trial
    store = TextStore(tmp_path / "run")
    directory = write_trial(trial, tmp_path / "run", store)
    assert read_trial(directory, store) == trial
    data = json.loads((directory / "trial.json").read_text(encoding="ascii"))
    assert [list(entry) for entry in data["judgements"]] == [list(FIELDS)] * 2
    assert data["judgements"][1]["values"] == {"delta": -0.5} and data["judgements"][0]["mode"] is None


def test_a_trial_json_written_before_the_field_loads_with_it_null(tmp_path: Path) -> None:
    data = json.loads(to_json(base_trial()))
    assert "judgements" in data and data["judgements"] is None, "to_dict writes the key, null when not recorded"
    del data["judgements"]
    assert from_dict(Trial, data).judgements is None
    store = TextStore(tmp_path / "run")
    directory = write_trial(base_trial(), tmp_path / "run", store)
    stored = json.loads((directory / "trial.json").read_text(encoding="ascii"))
    del stored["judgements"]
    (directory / "trial.json").write_bytes(json.dumps(stored, indent=2).encode("ascii") + b"\n")
    assert read_trial(directory, store).judgements is None


def test_a_trial_json_with_a_bad_judgement_is_refused() -> None:
    data = json.loads(to_json(judged_trial()))
    data["judgements"][0]["score"] = 2.0
    with pytest.raises(ValueError) as caught:
        from_dict(Trial, data)
    assert "score" in str(caught.value)
    data = json.loads(to_json(judged_trial()))
    data["judgements"][0]["extra"] = 1
    with pytest.raises(ValueError):
        from_dict(Trial, data)


# ---------------------------------------------------------------------------
# trial.md


def test_trial_md_shows_each_judgement_tagged_judged(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "store")
    page = render_trial_md(judged_trial(), store)
    text = section(page)
    assert page.index("## Judgements") > page.index("## Requests"), "the section comes after the others"
    lead = text.split("###", 1)[0]
    assert "[JUDGED]" in lead and "never a measurement" in lead, lead
    assert "### equivalence [JUDGED]\n" in text and "### efficiency [JUDGED]\n" in text
    assert has_row(text, "rubric", "synthetic-v1") and has_row(text, "rubric_sha256", RUBRIC_SHA)
    assert has_row(text, "model", "SYNTHETIC-judge") and has_row(text, "family", "synthetic-b")
    assert has_row(text, "verdict", "AGREE [JUDGED]") and has_row(text, "score", "0.875 [JUDGED]")
    assert has_row(text, "verdict", "DISAGREE [JUDGED]") and has_row(text, "score", "0.25 [JUDGED]")
    assert has_row(text, "mode", "none (the judge declares no modes)") and has_row(text, "mode", "predict")
    assert has_row(text, "delta", "-0.5 [JUDGED]"), "each judged value is tagged"
    assert "SYNTHETIC: both stdouts print the same sums.\nThe loop bounds match.\n" in text, "the rationale is fenced"
    assert "PLACEHOLDER" not in text, "every value here is recorded"
    page.encode("ascii")


def test_trial_md_without_judgements_is_unchanged_and_an_empty_list_says_none(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "store")
    plain = render_trial_md(base_trial(), store)
    assert "Judgements" not in plain and "[JUDGED]" not in plain
    empty = render_trial_md(dataclasses.replace(base_trial(), judgements=[]), store)
    assert empty.startswith(plain.rstrip("\n")), "the section is added after the page as it was"
    assert "None." in section(empty)


# ---------------------------------------------------------------------------
# The Parquet mirror


def test_the_parquet_mirror_has_a_judgements_table_with_one_row_per_record(tmp_path: Path) -> None:
    assert "judgements" in parquet.TABLES and tuple(parquet.SCHEMAS["judgements"].names) == TABLE_COLUMNS
    trials = [judged_trial(), base_trial(OTHER_ID)]
    rows = parquet.trial_rows(trials)["judgements"]
    assert [(row["trial_id"], row["ordinal"]) for row in rows] == [(TRIAL_ID, 0), (TRIAL_ID, 1)]
    first, second = rows
    assert (first["kind"], first["mode"], first["verdict"], first["score"]) == ("equivalence", None, "AGREE", 0.875)
    assert (second["kind"], second["mode"], second["values"]) == ("efficiency", "predict", '{"delta": -0.5}')
    assert first["values"] == "{}" and first["rationale"] == RATIONALE and first["rubric_sha256"] == RUBRIC_SHA
    assert {row["tag"] for row in rows} == {"[JUDGED]"}
    parquet.write_run_parquet(trials, tmp_path / "parquet")
    assert parquet.read_run_parquet(tmp_path / "parquet")["judgements"] == rows


def test_a_run_without_judgements_has_no_judgements_rows(tmp_path: Path) -> None:
    third = record.make_trial_id("judge-record", "scripted-fixture", "judge-suite", "omp-cpp", "third", 1)
    trials = [base_trial(), base_trial(OTHER_ID), dataclasses.replace(base_trial(third), judgements=[])]
    assert parquet.trial_rows(trials)["judgements"] == []
    parquet.write_run_parquet(trials, tmp_path / "parquet")
    assert not (tmp_path / "parquet" / "judgements").exists()
