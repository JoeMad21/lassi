"""Tests for the train data sources and the data split hash (task P17.8; lassi.train.data).

Bible: Training Module (Safeguards: eval splits are refused by the trainer,
and every checkpoint logs data split hashes), Project Recipes (train.yaml),
Agent Rules 5 and 10; PHASE-NOTES P17 (training fixtures are synthetic and
outside every bench suite).

The contract these tests fix:

- split_hash(identity) is the sha256 of the identity's canonical JSON:
  sorted keys, separators "," and ":", ASCII only, NaN refused.
- load_synthetic(name, root) reads <root>/<name>, JSON Lines split on LF:
  one JSON object per line, in file order (the last line's LF optional, a
  blank line refused anywhere), as TrainData(source="synthetic")
  whose split hash is that of {"source": "synthetic", "records": [...]}. It
  is a content hash: the file's name, its whitespace, and the order of keys
  inside a line do not change it, and a changed or reordered record does.
- load_bench(recipe, suites_dir) loads <suites_dir>/<bench.suite>.yaml and
  fetches every selected item (bench.items, else every item of the train
  split) with Suite.item(name, purpose="train"), so an eval or unassigned
  item raises EvalSplitError, reported as a RunError naming bench.items
  and Agent Rule 5. It returns TrainData(source="bench") with the suite and
  the sorted item names, and the identity {"source": "bench", "suite",
  "commit", "split": "train", "items", "manifest_sha256"}, whose split hash
  changes when the manifest's bytes or the selected items change. No source
  file is read.

The suite and the fixtures are SYNTHETIC files written by the test. No value
in this module is a measurement.
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
    SUITE_COMMIT,
    SUITE_TEXT,
    Log,
    canonical_json,
    core_name,
    fake_registry,
    jsonl,
    on_bench,
    train_data,
    write_fixture,
    write_recipe,
    write_suite,
)

from lassi.bench import EvalSplitError, Suite
from lassi.core.runner import RunError


def sha256_text(text: str) -> str:
    """Return the sha256 hex digest of ASCII text."""
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def synthetic_identity(records: Any) -> dict[str, Any]:
    """Return the identity a synthetic split hash is taken over."""
    return {"source": "synthetic", "records": [dict(record) for record in records]}


def bench_recipe(tmp_path: Path, **bench: Any) -> Any:
    """Write and load a train recipe that reads the SYNTHETIC suite's train split; `bench` changes its keys."""
    loader = core_name("lassi.core.recipe", "load_train_recipe")
    path = write_recipe(tmp_path / "recipes", on_bench(**bench))
    return loader(path, roots=[tmp_path / "recipes"], registry=fake_registry(Log()))


def spy_on_suite_item(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record every Suite.item call as (item, purpose), passing it through."""
    calls: list[tuple[str, str]] = []
    original = Suite.item

    def spy(self: Suite, name: str, *, purpose: str) -> Any:
        calls.append((name, purpose))
        return original(self, name, purpose=purpose)

    monkeypatch.setattr(Suite, "item", spy)
    return calls


# ---------------------------------------------------------------------------
# The split hash


def test_split_hash_is_sha256_of_canonical_json() -> None:
    identity = {"source": "synthetic", "records": [{"b": 1, "a": "x"}], "z": [3, 1]}
    expected = sha256_text(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    assert train_data().split_hash(identity) == expected
    assert train_data().split_hash(dict(reversed(list(identity.items())))) == expected


def test_split_hash_refuses_nan() -> None:
    with pytest.raises(ValueError):
        train_data().split_hash({"source": "synthetic", "records": [{"a": float("nan")}]})


def test_split_hash_writes_non_ascii_text_as_escapes() -> None:
    identity = {"source": "bench", "suite": "caf\N{LATIN SMALL LETTER E WITH ACUTE}"}
    assert train_data().split_hash(identity) == sha256_text(canonical_json(identity))


# ---------------------------------------------------------------------------
# Synthetic fixtures


def test_load_synthetic_returns_the_records_in_file_order_and_their_hash(tmp_path: Path) -> None:
    write_fixture(tmp_path, jsonl(RECORDS), name="sft-tiny.jsonl")
    data = train_data().load_synthetic("sft-tiny.jsonl", tmp_path)
    assert isinstance(data, core_name("lassi.core.interfaces", "TrainData"))
    assert data.source == "synthetic"
    assert [dict(record) for record in data.records] == list(RECORDS)
    assert data.suite is None and tuple(data.items) == ()
    assert data.split_hash == sha256_text(canonical_json(synthetic_identity(RECORDS)))


def test_synthetic_split_hash_is_a_content_hash(tmp_path: Path) -> None:
    compact = jsonl(RECORDS)
    spaced = b"".join(
        (json.dumps(dict(reversed(list(record.items()))), indent=None, separators=(" ,  ", " :  ")) + "\n").encode(
            "ascii"
        )
        for record in RECORDS
    )
    write_fixture(tmp_path, compact, name="first.jsonl")
    write_fixture(tmp_path, spaced, name="renamed.jsonl")
    write_fixture(tmp_path, compact[:-1], name="no-final-lf.jsonl")
    changed = [RECORDS[0], {**RECORDS[1], "completion": "SYNTHETIC changed"}]
    write_fixture(tmp_path, jsonl(changed), name="changed.jsonl")
    write_fixture(tmp_path, jsonl(reversed(RECORDS)), name="reordered.jsonl")
    load = train_data().load_synthetic
    first = load("first.jsonl", tmp_path).split_hash
    assert load("renamed.jsonl", tmp_path).split_hash == first
    assert load("no-final-lf.jsonl", tmp_path).split_hash == first
    assert load("changed.jsonl", tmp_path).split_hash != first
    assert load("reordered.jsonl", tmp_path).split_hash != first


@pytest.mark.parametrize(
    "content",
    [
        b'{"a": 1}\r\n',
        b'{"a": 1}\n\n{"a": 2}\n',
        b'{"a": 1}\n\n',
        b"",
        b'{"a": 1, "a": 2}\n',
        b"[1]\n",
        b'{"a": "\xc3\xa9"}\n',
    ],
    ids=["crlf", "blank-line", "blank-line-at-end", "empty", "repeated-key", "not-an-object", "non-ascii"],
)
def test_load_synthetic_refusals_name_data_synthetic(content: bytes, tmp_path: Path) -> None:
    write_fixture(tmp_path, content, name="bad.jsonl")
    with pytest.raises(RunError) as refused:
        train_data().load_synthetic("bad.jsonl", tmp_path)
    assert "data.synthetic" in str(refused.value) and "bad.jsonl" in str(refused.value), str(refused.value)


def test_load_synthetic_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RunError) as refused:
        train_data().load_synthetic("absent.jsonl", tmp_path)
    assert "data.synthetic" in str(refused.value) and "absent.jsonl" in str(refused.value), str(refused.value)


# ---------------------------------------------------------------------------
# Bench data through the registry (Agent Rule 5)


def bench_identity(manifest: bytes, items: list[str]) -> dict[str, Any]:
    """Return the identity of the SYNTHETIC suite's train split with `items` selected."""
    return {
        "source": "bench", "suite": SUITE, "commit": SUITE_COMMIT, "split": "train", "items": items,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }


def test_load_bench_takes_every_train_item_through_suite_item(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suites = tmp_path / "suites"
    manifest = write_suite(suites).read_bytes()
    recipe = bench_recipe(tmp_path)
    calls = spy_on_suite_item(monkeypatch)
    data, identity = train_data().load_bench(recipe, suites)
    assert set(calls) == {("alpha", "train"), ("beta", "train")}
    assert identity == bench_identity(manifest, ["alpha", "beta"])
    assert data.source == "bench" and list(data.items) == ["alpha", "beta"] and tuple(data.records) == ()
    assert isinstance(data.suite, Suite) and data.suite.name == SUITE
    assert data.split_hash == sha256_text(canonical_json(identity))


def test_load_bench_refuses_an_eval_item_through_the_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_suite(tmp_path / "suites")
    recipe = bench_recipe(tmp_path, items=["alpha", "open"])
    calls = spy_on_suite_item(monkeypatch)
    with pytest.raises(RunError) as refused:
        train_data().load_bench(recipe, tmp_path / "suites")
    message = str(refused.value)
    assert "bench.items" in message and "open" in message and "Agent Rule 5" in message, message
    assert ("open", "train") in calls
    assert isinstance(refused.value.__cause__, EvalSplitError) or "unassigned" in message, message


def test_bench_split_hash_follows_the_manifest_bytes_and_items(tmp_path: Path) -> None:
    suites = tmp_path / "suites"
    write_suite(suites)
    load_bench = train_data().load_bench
    whole, _ = load_bench(bench_recipe(tmp_path), suites)
    one, identity = load_bench(bench_recipe(tmp_path, items=["beta"]), suites)
    assert identity["items"] == ["beta"] and one.split_hash != whole.split_hash
    write_suite(suites, SUITE_TEXT + "# SYNTHETIC comment that changes the manifest's bytes\n")
    again, _ = load_bench(bench_recipe(tmp_path), suites)
    assert again.split_hash != whole.split_hash
    write_suite(suites)
    assert load_bench(bench_recipe(tmp_path), suites)[0].split_hash == whole.split_hash


def test_bench_item_on_the_returned_handle_refuses_eval_items(tmp_path: Path) -> None:
    write_suite(tmp_path / "suites")
    data, _ = train_data().load_bench(bench_recipe(tmp_path), tmp_path / "suites")
    assert data.bench_item("alpha").split == "train"
    with pytest.raises(EvalSplitError):
        data.bench_item("held")
