"""Tests for the corpus data source, data.corpus (2026-10-23; OQ-076 (1a); lassi.train.data load_corpus).

Bible: Training Module (Safeguards: eval splits are refused, and every
checkpoint logs data split hashes), Project Recipes (Notes, train recipes),
Agent Rules 5 and 10.

The contract these tests fix:

- A corpus is a tracked manifest, <manifests>/<name>/manifest.json, written
  by the tool that built the corpus, and a generated, untracked records
  file, <records root>/<records.file>, never in the repository. The records
  root is TrainOptions.corpus_root, else $LASSI_SCRATCH/train-data; with
  neither, the run is refused naming data.corpus and LASSI_SCRATCH.
- load_corpus refuses, naming data.corpus: a missing or malformed manifest
  (not a JSON object; corpus not the recipe's name; records.file not one
  plain .jsonl name; records.sha256 not 64 hex digits; records.count not
  an integer of at least 1; held_out missing); a records file that is
  missing, resolves outside the records root, or whose sha256 or record
  count differ from the manifest's (the file the tool checked, and no
  other, is trained on); and the line rules of data.synthetic (ASCII, LF,
  one JSON object per line).
- Agent Rule 5: the manifest's held_out names a suite manifest and the
  items the corpus was built without; every eval and unassigned item of
  that suite, read through the bench registry, must be among them, or the
  corpus is refused naming the item and Agent Rule 5. A held-out list may
  also hold train items.
- The TrainData has source corpus, the records in file order, and the split
  hash of {"source": "corpus", "corpus", "records_sha256",
  "manifest_sha256"}. lassi train's provenance data record holds source,
  name, record_count, records_sha256, manifest_sha256, and split_hash.

The manifest, records, and suite are SYNTHETIC files the tests write; no
value here is a measurement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from train_fakes import (
    RECORDS,
    SUITE,
    Log,
    canonical_json,
    fake_probes,
    fake_registry,
    jsonl,
    provenance,
    recipe_data,
    train_data,
    train_run,
    write_recipe,
    write_suite,
)

from lassi.core.runner import RunError

CORPUS = "synthetic-corpus"
RECORDS_FILE = f"{CORPUS}.jsonl"
HELD = ["held", "open"]


def manifest_for(content: bytes, **changes: Any) -> dict[str, Any]:
    """Return a corpus manifest for records `content`; `changes` replace top-level keys."""
    found: dict[str, Any] = {
        "corpus": CORPUS,
        "records": {"file": RECORDS_FILE, "sha256": hashlib.sha256(content).hexdigest(),
                    "count": content.count(b"\n")},
        "held_out": {"suite": SUITE, "items": list(HELD)},
    }
    found.update(changes)
    return found


def write_corpus(root: Path, content: bytes | None = None, manifest: Any = None) -> tuple[Path, Path, Path]:
    """Write the manifest, the records, and the SYNTHETIC suite under `root`; return (manifests, records, suites)."""
    content = jsonl(RECORDS) if content is None else content
    manifests, records, suites = root / "manifests", root / "records", root / "suites"
    (manifests / CORPUS).mkdir(parents=True)
    records.mkdir()
    document = manifest_for(content) if manifest is None else manifest
    text = document if isinstance(document, str) else json.dumps(document, indent=2) + "\n"
    (manifests / CORPUS / "manifest.json").write_bytes(text.encode("ascii"))
    (records / RECORDS_FILE).write_bytes(content)
    write_suite(suites)
    return manifests, records, suites


def load(root: Path, name: str = CORPUS) -> Any:
    """Return load_corpus's (TrainData, identity) for the corpus under `root`."""
    manifests, records, suites = root / "manifests", root / "records", root / "suites"
    return train_data().load_corpus(name, manifests, records, suites)


def test_load_corpus_returns_the_records_and_the_split_hash(tmp_path: Path) -> None:
    manifests, _, _ = write_corpus(tmp_path)
    data, identity = load(tmp_path)
    assert data.source == "corpus" and data.suite is None and data.items == ()
    assert [dict(record) for record in data.records] == [dict(record) for record in RECORDS]
    manifest_sha = hashlib.sha256((manifests / CORPUS / "manifest.json").read_bytes()).hexdigest()
    assert identity == {"source": "corpus", "corpus": CORPUS, "records_sha256": manifest_for(jsonl(RECORDS))[
        "records"]["sha256"], "manifest_sha256": manifest_sha}
    assert data.split_hash == hashlib.sha256(canonical_json(identity).encode("ascii")).hexdigest()


BAD_MANIFESTS = {
    "not-json": "{",
    "not-an-object": "[]",
    "wrong-name": {"corpus": "other"},
    "file-with-slash": {"records": {"file": "../x.jsonl", "sha256": "0" * 64, "count": 2}},
    "file-not-jsonl": {"records": {"file": "x.json", "sha256": "0" * 64, "count": 2}},
    "short-sha": {"records": {"file": RECORDS_FILE, "sha256": "0" * 63, "count": 2}},
    "zero-count": {"records": {"file": RECORDS_FILE, "sha256": "0" * 64, "count": 0}},
    "bool-count": {"records": {"file": RECORDS_FILE, "sha256": "0" * 64, "count": True}},
    "no-held-out": {"held_out": None},
    "held-out-not-a-list": {"held_out": {"suite": SUITE, "items": "held"}},
}


@pytest.mark.parametrize("name", list(BAD_MANIFESTS))
def test_a_malformed_manifest_is_refused_naming_data_corpus(name: str, tmp_path: Path) -> None:
    change = BAD_MANIFESTS[name]
    manifest = change if isinstance(change, str) else manifest_for(jsonl(RECORDS), **change)
    write_corpus(tmp_path, manifest=manifest)
    with pytest.raises(RunError, match="data.corpus"):
        load(tmp_path)


def test_a_missing_manifest_is_refused(tmp_path: Path) -> None:
    write_corpus(tmp_path)
    with pytest.raises(RunError, match="data.corpus") as refused:
        load(tmp_path, name="absent")
    assert "absent" in str(refused.value)


def test_a_records_file_that_is_not_the_checked_one_is_refused(tmp_path: Path) -> None:
    write_corpus(tmp_path)
    (tmp_path / "records" / RECORDS_FILE).write_bytes(jsonl(RECORDS[:1]))
    with pytest.raises(RunError) as refused:
        load(tmp_path)
    assert "data.corpus" in str(refused.value) and "sha256" in str(refused.value)


def test_a_missing_records_file_is_refused(tmp_path: Path) -> None:
    write_corpus(tmp_path)
    (tmp_path / "records" / RECORDS_FILE).unlink()
    with pytest.raises(RunError, match="data.corpus"):
        load(tmp_path)


def test_a_count_that_differs_is_refused(tmp_path: Path) -> None:
    content = jsonl(RECORDS)
    write_corpus(tmp_path, content, manifest_for(content, records={
        "file": RECORDS_FILE, "sha256": hashlib.sha256(content).hexdigest(), "count": 3}))
    with pytest.raises(RunError, match="count"):
        load(tmp_path)


def test_the_line_rules_of_synthetic_data_hold(tmp_path: Path) -> None:
    content = jsonl(RECORDS).replace(b"\n", b"\r\n")
    write_corpus(tmp_path, content, manifest_for(content, records={
        "file": RECORDS_FILE, "sha256": hashlib.sha256(content).hexdigest(), "count": 2}))
    with pytest.raises(RunError, match="data.corpus"):
        load(tmp_path)


@pytest.mark.parametrize("held", [["held"], ["open"], []], ids=["no-unassigned", "no-eval", "none"])
def test_a_held_item_the_corpus_does_not_exclude_is_refused_agent_rule_5(held: list[str], tmp_path: Path) -> None:
    write_corpus(tmp_path, manifest=manifest_for(jsonl(RECORDS), held_out={"suite": SUITE, "items": held}))
    with pytest.raises(RunError) as refused:
        load(tmp_path)
    message = str(refused.value)
    missing = sorted(set(HELD) - set(held))
    assert "Agent Rule 5" in message and missing[0] in message, message


def test_a_held_out_list_may_also_hold_train_items(tmp_path: Path) -> None:
    write_corpus(tmp_path, manifest=manifest_for(jsonl(RECORDS), held_out={"suite": SUITE, "items": [*HELD, "alpha"]}))
    assert load(tmp_path)[0].source == "corpus"


def test_an_unknown_held_out_suite_is_refused(tmp_path: Path) -> None:
    write_corpus(tmp_path, manifest=manifest_for(jsonl(RECORDS), held_out={"suite": "no-such-suite", "items": HELD}))
    with pytest.raises(RunError, match="no-such-suite"):
        load(tmp_path)


def test_the_records_root_is_the_option_then_the_scratch_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = train_data()
    assert module.corpus_root(tmp_path / "option") == tmp_path / "option"
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path / "scratch"))
    assert module.corpus_root(None) == tmp_path / "scratch" / "train-data"
    monkeypatch.delenv("LASSI_SCRATCH")
    with pytest.raises(RunError) as refused:
        module.corpus_root(None)
    assert "data.corpus" in str(refused.value) and "LASSI_SCRATCH" in str(refused.value)


def test_lassi_train_records_the_corpus_in_provenance(tmp_path: Path) -> None:
    manifests, records, suites = write_corpus(tmp_path / "corpus")
    log = Log()
    registry = fake_registry(log, data_sources=frozenset({"synthetic", "corpus"}))
    path = write_recipe(tmp_path / "recipes", recipe_data(data={"corpus": CORPUS}))
    options = train_run().TrainOptions(
        runs_root=tmp_path / "runs-root", train_id="t1", registry=registry, roots=[tmp_path / "recipes"],
        suites_dir=suites, corpus_root=records, corpus_manifests=manifests,
        probes=fake_probes(log),
    )
    train_dir = train_run().run_training(path, options)
    data, identity = load(tmp_path / "corpus")
    record = provenance(train_dir)["data"]
    assert record == {"source": "corpus", "name": CORPUS, "record_count": len(RECORDS),
                      "records_sha256": identity["records_sha256"], "manifest_sha256": identity["manifest_sha256"],
                      "split_hash": data.split_hash}
    assert log.jobs[0].data.records == data.records
