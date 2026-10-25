"""Tests for the adversary stage's record: Trial.adversary, its JSON, trial.md, and its Parquet table (task P16.5).

Bible: Result Record (Trial; Output files and statistics; Storage),
Readability Standards (Trial row), Design Principle 7 (Parquet mirrors the
JSON records), Agent Rules 1 and 14. Plan: plans/p16-adversarial.md, P16.5
(the third acceptance line).

The contract these tests fix:

- lassi.core.record holds three new frozen, keyword-only record classes:
  AdversaryRecord (agent: the adversary's registry name, adversary.kind;
  config: the adversary section's keys other than kind, as resolved;
  proposals: the proposals the adversary made, refused ones included;
  refusals: check_inputs reason -> count of refused proposals; runs: the
  program runs the stage made; wall_s: the stage's own wall time in the
  trial, in seconds; divergences: [Divergence]; contract_findings:
  [ContractFinding]), Divergence (proposal: the proposal's index in the
  trial; inputs: input name -> sha256 of its lassi_io file in the run's
  binary store; stats: binary_io's [OutputStats]; candidate_failure: how
  the candidate's run failed, null when it ran clean), and ContractFinding
  (proposal; inputs; message: how the reference failed). Each checks its
  values as every record does: a count, index, or wall time below 0, a
  wall time that is not finite, a refusal reason outside CONTRACT_REASONS,
  an input digest that is not 64 lowercase hex, and an empty
  candidate_failure are refused with a ValueError naming the field.
- Trial.adversary is its last field but judgements (task P15.1), an
  AdversaryRecord or null, null by default: null means the stage did not take the trial (or the record
  predates the field), so a trial.json written before the field loads with
  it null. The bible's Result Record block names it last in Trial, as
  AdversaryRecord, and defines the AdversaryRecord, Divergence, and
  ContractFinding blocks after DeviceRecord (tests/core/test_record.py).
- A record round-trips through to_json and from_json, and through
  write_trial and read_trial.
- trial.md shows a "## Adversary" section for a trial with a record: a
  table of its agent, proposals, runs, and wall_s by field name (as the
  page's other record tables name their rows), its config, its refusals as
  rows of reason and count, and each divergence and contract finding with
  a row of each input's name and sha256 (each divergence with its
  candidate_failure and its statistics per output, as the page shows an
  attempt's, and each finding with its message). A trial with no record shows no such section, so the committed
  golden page is unchanged.
- The Parquet mirror gains the table `adversary`, one row per trial with a
  record: the key columns, agent, config, proposals, refusals, runs,
  wall_s, divergences, and contract_findings, the counts as int64 columns,
  wall_s as a double, and the rest as JSON text with sorted keys (as the
  attempts table holds files), the lists as the records' dicts. A trial with no record has no row.

Every value here is a SYNTHETIC fixture value, never a measurement.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from lassi.core import parquet, record
from lassi.core.interfaces import Sampling
from lassi.core.record import BenchItem, ModelInfo, OutputStats, Provenance, Trial, from_dict, from_json, to_json
from lassi.core.store import TextStore, read_trial, write_trial
from lassi.core.trial_md import render_trial_md

TRIAL_ID = record.make_trial_id("adv-record", "scripted-fixture", "adv-stage", "omp-cpp", "scale", 1)
UNTAKEN_ID = record.make_trial_id("adv-record", "scripted-fixture", "adv-stage", "omp-cpp", "plainish", 1)
# SYNTHETIC digests of input files in a binary store; no file holds them.
SHA_X = "a" * 64
SHA_FINDING = "b" * 64
NOTE = "SYNTHETIC: y differs past max_abs 0.25"
FINDING = "SYNTHETIC: the cpp reference run ended with exit status 139"
CANDIDATE_FAILURE = "SYNTHETIC: the cpp candidate run ended with exit status 139"
# A SYNTHETIC wall time in seconds, not a measurement.
WALL_S = 12.5
# agent_log (task P16.4) is the last field, optional, so a record written before it loads.
RECORD_FIELDS = (
    "agent",
    "config",
    "proposals",
    "refusals",
    "runs",
    "wall_s",
    "divergences",
    "contract_findings",
    "agent_log",
)
DIVERGENCE_FIELDS = ("proposal", "inputs", "stats", "candidate_failure")
FINDING_FIELDS = ("proposal", "inputs", "message")
TABLE_COLUMNS = (
    *parquet.PARTITION_COLUMNS,
    "trial_id",
    "agent",
    "config",
    "proposals",
    "refusals",
    "runs",
    "wall_s",
    "divergences",
    "contract_findings",
    "agent_log",
)


def need(name: str) -> Any:
    """Return lassi.core.record's attribute `name`, failing with the task's name when it does not exist yet."""
    assert hasattr(record, name), f"lassi.core.record has no {name} (task P16.5)"
    return getattr(record, name)


def base_trial(trial_id: str = TRIAL_ID) -> Trial:
    """Return a SYNTHETIC trial with no attempt and no adversary record."""
    item = record.parse_trial_id(trial_id).item
    return Trial(
        trial_id=trial_id,
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=None, dirty=None, device=None, sdk=None, date="2026-10-19T00:00:00+00:00"),
        bench_item=BenchItem(suite="adv-stage", item=item, split="unassigned", direction="omp-cpp"),
        model=ModelInfo(backend="scripted", id="scripted-fixture", sampling=Sampling(0.2, 0.9, 8)),
    )


def example_record() -> Any:
    """Return a SYNTHETIC AdversaryRecord with one divergence and one contract finding."""
    stats = OutputStats(name="y", pcc=0.0, max_abs=1000.0, max_ulp=12345, passed=False, note=NOTE)
    divergence = need("Divergence")(
        proposal=4, inputs={"x": SHA_X}, stats=[stats], candidate_failure=CANDIDATE_FAILURE
    )
    finding = need("ContractFinding")(proposal=5, inputs={"x": SHA_FINDING}, message=FINDING)
    return need("AdversaryRecord")(
        agent="fuzzer",
        config={"budget": {"inputs": 7}, "seed": 3},
        proposals=7,
        refusals={"out-of-range": 2, "dtype": 1},
        runs=6,
        wall_s=WALL_S,
        divergences=[divergence],
        contract_findings=[finding],
    )


def taken_trial() -> Trial:
    """Return base_trial with example_record as its adversary record."""
    return dataclasses.replace(base_trial(), adversary=example_record())


def section(page: str) -> str:
    """Return the page's "## Adversary" section, up to the next "## " heading or the end."""
    assert "## Adversary\n" in page, "trial.md has no Adversary section"
    start = page.index("## Adversary\n")
    end = page.find("\n## ", start + 1)
    return page[start:] if end == -1 else page[start:end]


def table_rows(text: str) -> list[list[str]]:
    """Return every Markdown table row of `text` as its cells, stripped of blanks and backticks."""
    rows = []
    for line in text.splitlines():
        if line.startswith("|") and line.endswith("|"):
            rows.append([cell.strip().strip("`") for cell in line[1:-1].split("|")])
    return rows


def has_row(text: str, *cells: str) -> bool:
    """Return True when some table row of `text` holds every one of `cells`."""
    return any(all(cell in row for cell in cells) for row in table_rows(text))


# ---------------------------------------------------------------------------
# The record classes and Trial.adversary


@pytest.mark.parametrize(
    ("name", "fields"),
    [("AdversaryRecord", RECORD_FIELDS), ("Divergence", DIVERGENCE_FIELDS), ("ContractFinding", FINDING_FIELDS)],
)
def test_the_record_classes_are_frozen_keyword_only_with_their_fields(name: str, fields: tuple[str, ...]) -> None:
    cls = need(name)
    assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen, f"{name} is not a frozen dataclass"
    specs = dataclasses.fields(cls)
    assert tuple(spec.name for spec in specs) == fields
    assert all(spec.kw_only for spec in specs), f"{name} has positional fields"


def test_trial_adversary_is_the_last_field_but_judgements_and_null_by_default() -> None:
    names = [spec.name for spec in dataclasses.fields(Trial)]
    assert names[-2:] == ["adversary", "judgements"], "Trial.adversary (task P16.5) precedes judgements (task P15.1)"
    assert base_trial().adversary is None
    assert taken_trial().adversary == example_record()


def test_a_record_round_trips_through_json_and_the_trial_files(tmp_path: Path) -> None:
    trial = taken_trial()
    assert from_json(Trial, to_json(trial)) == trial
    store = TextStore(tmp_path / "run")
    directory = write_trial(trial, tmp_path / "run", store)
    assert read_trial(directory, store) == trial
    data = json.loads((directory / "trial.json").read_text(encoding="ascii"))
    assert data["adversary"]["divergences"][0]["inputs"] == {"x": SHA_X}
    assert data["adversary"]["refusals"] == {"out-of-range": 2, "dtype": 1}


def test_a_trial_json_written_before_the_field_loads_with_it_null(tmp_path: Path) -> None:
    data = json.loads(to_json(base_trial()))
    assert "adversary" in data, "to_dict writes the key, null when the stage did not take the trial"
    assert data["adversary"] is None
    del data["adversary"]
    assert from_dict(Trial, data).adversary is None
    store = TextStore(tmp_path / "run")
    directory = write_trial(base_trial(), tmp_path / "run", store)
    stored = json.loads((directory / "trial.json").read_text(encoding="ascii"))
    del stored["adversary"]
    (directory / "trial.json").write_bytes(json.dumps(stored, indent=2).encode("ascii") + b"\n")
    assert read_trial(directory, store).adversary is None


@pytest.mark.parametrize(
    ("name", "changes", "field"),
    [
        ("AdversaryRecord", {"proposals": -1}, "proposals"),
        ("AdversaryRecord", {"runs": -1}, "runs"),
        ("AdversaryRecord", {"wall_s": -0.5}, "wall_s"),
        ("AdversaryRecord", {"wall_s": float("nan")}, "wall_s"),
        ("Divergence", {"candidate_failure": ""}, "candidate_failure"),
        ("AdversaryRecord", {"refusals": {"too-big": 1}}, "refusals"),
        ("AdversaryRecord", {"refusals": {"dtype": -1}}, "refusals"),
        ("Divergence", {"inputs": {"x": "A" * 64}}, "inputs"),
        ("Divergence", {"proposal": -1}, "proposal"),
        ("ContractFinding", {"inputs": {"x": "not a digest"}}, "inputs"),
    ],
)
def test_the_record_refuses_values_outside_its_rules(name: str, changes: dict[str, Any], field: str) -> None:
    good = example_record()
    base = {"AdversaryRecord": good, "Divergence": good.divergences[0], "ContractFinding": good.contract_findings[0]}
    values = {spec.name: getattr(base[name], spec.name) for spec in dataclasses.fields(need(name))}
    with pytest.raises(ValueError) as caught:
        need(name)(**{**values, **changes})
    assert field in str(caught.value), str(caught.value)


# ---------------------------------------------------------------------------
# trial.md


def test_trial_md_shows_the_adversary_section_only_for_a_taken_trial(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "store")
    assert "## Adversary" not in render_trial_md(base_trial(), store), "no record, no section"
    text = section(render_trial_md(taken_trial(), store))
    assert has_row(text, "agent", "fuzzer")
    assert has_row(text, "proposals", "7") and has_row(text, "runs", "6") and has_row(text, "wall_s", "12.5")
    assert has_row(text, "candidate_failure", CANDIDATE_FAILURE), "a divergence names how the candidate run failed"
    assert has_row(text, "out-of-range", "2") and has_row(text, "dtype", "1")
    assert "seed" in text and "budget" in text, "the resolved config is shown"
    assert has_row(text, "x", SHA_X), "a divergence's inputs are shown by their sha256"
    assert has_row(text, "y", "1000.0", "false") and NOTE in text, "a divergence's statistics are shown"
    assert has_row(text, "x", SHA_FINDING) and FINDING in text, "a contract finding is shown with its message"


# ---------------------------------------------------------------------------
# The Parquet mirror


def test_the_parquet_mirror_has_an_adversary_table_with_one_row_per_taken_trial(tmp_path: Path) -> None:
    assert "adversary" in parquet.TABLES and "adversary" in parquet.SCHEMAS, "no Parquet table adversary"
    assert tuple(parquet.SCHEMAS["adversary"].names) == TABLE_COLUMNS
    trials = [taken_trial(), base_trial(UNTAKEN_ID)]
    rows = parquet.trial_rows(trials)["adversary"]
    assert [row["trial_id"] for row in rows] == [TRIAL_ID], "a trial the stage did not take has no row"
    (row,) = rows
    found = example_record()
    assert (row["agent"], row["proposals"], row["runs"], row["wall_s"]) == ("fuzzer", 7, 6, WALL_S)

    def text(value: Any) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=True)

    assert row["config"] == text(found.config)
    assert row["refusals"] == text(found.refusals)
    assert row["divergences"] == text([dataclasses.asdict(entry) for entry in found.divergences])
    assert row["contract_findings"] == text([dataclasses.asdict(entry) for entry in found.contract_findings])
    assert row["agent_log"] is None, "an adversary that keeps no log has a null agent_log (task P16.4)"
    parquet.write_run_parquet(trials, tmp_path / "parquet")
    assert parquet.read_run_parquet(tmp_path / "parquet")["adversary"] == rows


def test_trial_md_names_a_clean_candidate_run_never_as_a_placeholder(tmp_path: Path) -> None:
    found = example_record()
    clean = dataclasses.replace(found.divergences[0], candidate_failure=None)
    trial = dataclasses.replace(base_trial(), adversary=dataclasses.replace(found, divergences=[clean]))
    text = section(render_trial_md(trial, TextStore(tmp_path / "store")))
    assert has_row(text, "candidate_failure", "none (the run was clean)"), "a clean run is a value, not unmeasured"
