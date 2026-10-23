"""The P16 Freeze decision's record, tests/fixtures/adversary/freeze.yaml (task P16.10).

Bible: Adversarial Setups (the seeded set; Fuzzer seeds and freeze), Benchmark Suites (adv-seeded-v0), Agent Rule 5.
Plan: plans/p16-adversarial.md, planning decision Freeze, and P16.10's second acceptance bullet: the record fixes
the sha256 of every seeded candidate and witness, of the fuzzer's and the llm adversary's source files, of
adversary-v0's prompts and the adversary recordings, and each gate recipe's adversary seed and budget and oracle
threshold; this fast test fails on any difference. The record also fixes the suite's manifest, reference programs,
and lassi_io.h, the contract code, and each recipe's whole adversary section, model.recording, and trials.n. A file
added under a frozen directory (a new candidate, recording, or prompt) fails too, so nothing joins the set
unfrozen; README.md files are left out (the record's header says why). Nothing here runs a program; no value is a
measurement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

import lassi.core.runner  # noqa: F401  (registers the components the recipes bind)
from lassi.core.recipe import load_recipe

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "tests" / "fixtures" / "adversary" / "freeze.yaml"
SEEDED = ROOT / "tests" / "fixtures" / "adversary" / "seeded"
RECORD = yaml.safe_load(FREEZE.read_text(encoding="ascii"))
# The directories whose every file, README.md aside, the record must name.
COVERED = ("assets/bench/adv-seeded-v0", "tests/fixtures/adversary/seeded", "assets/prompts/adversary-v0",
           "tests/fixtures/adversary/recordings")
SINGLE = ("assets/bench/adv-seeded-v0.yaml", "assets/harness/c/lassi_io.h", "lassi/agents/fuzzer.py",
          "lassi/agents/llm.py", "lassi/harness/lassi_io.py", "lassi/harness/contract.py")


def sha256(path: Path) -> str:
    """Return the sha256 hex digest of the file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def witness_sha256(witness: dict) -> str:
    """Return the sha256 of an index entry's `witness` value as the record's header defines it (canonical JSON)."""
    return hashlib.sha256(json.dumps(witness, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def test_the_record_names_the_seeded_suite() -> None:
    assert RECORD["suite"] == "adv-seeded-v0"


@pytest.mark.parametrize("name", sorted(RECORD["files"]))
def test_each_frozen_file_has_its_recorded_sha256(name: str) -> None:
    assert sha256(ROOT / name) == RECORD["files"][name], f"{name} changed after the freeze (Freeze decision)"


@pytest.mark.parametrize("directory", COVERED)
def test_every_file_under_a_frozen_directory_is_in_the_record(directory: str) -> None:
    found = {p.relative_to(ROOT).as_posix() for p in (ROOT / directory).glob("**/*")
             if p.is_file() and p.name != "README.md" and "__pycache__" not in p.parts}
    assert found and found <= set(RECORD["files"]), f"unfrozen: {sorted(found - set(RECORD['files']))}"


def test_the_sources_manifest_and_header_are_frozen() -> None:
    assert set(SINGLE) <= set(RECORD["files"])


def test_each_candidates_witness_has_its_recorded_sha256() -> None:
    index = json.loads((SEEDED / "index.json").read_text(encoding="utf-8"))
    # Only the entry's `witness` value is hashed; the rest of the entry is fixed by index.json's file hash.
    assert {entry["id"]: witness_sha256(entry["witness"]) for entry in index["candidates"]} == RECORD["witnesses"]


@pytest.mark.parametrize("name", sorted(RECORD["recipes"]))
def test_each_recipe_keeps_its_adversary_oracle_recording_and_trials(name: str) -> None:
    data, frozen = load_recipe(ROOT / name).data, RECORD["recipes"][name]
    assert data["adversary"] == json.loads(frozen["adversary"])
    assert (data["oracle"]["metric"], data["oracle"]["threshold"]) == (frozen["oracle_metric"],
                                                                       frozen["oracle_threshold"])
    assert (data["model"].get("recording"), data["trials"]["n"]) == (frozen["model_recording"], frozen["trials_n"])


def test_the_gate_values_are_the_plans() -> None:
    fuzzer = json.loads(RECORD["recipes"]["projects/adversary-seeded/seeded-fuzzer.yaml"]["adversary"])
    assert (fuzzer["kind"], fuzzer["seed"], fuzzer["budget"]) == ("fuzzer", 1, {"inputs": 32})


def test_every_adversary_seeded_recipe_is_in_the_record() -> None:
    recipes = {p.relative_to(ROOT).as_posix() for p in (ROOT / "projects" / "adversary-seeded").glob("**/*.yaml")}
    assert recipes == set(RECORD["recipes"])
