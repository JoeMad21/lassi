"""Tests for dev items in the train layer's data sources (task P17.31; OQ-083, option (b); lassi.train.data).

Bible: Training Module (Safeguards: eval splits are refused by the trainer),
Benchmark Suites (split rules: tt-pairs-v0 is a development set,
2026-10-25: its items are never trained on), Agent Rule 5. Plan:
plans/p17-portable.md, P17.31.

The contract these tests fix, each the eval and unassigned behaviour
extended to dev, the registry's fourth split:

- load_bench refuses a dev item that bench.items names: Suite.item(name,
  purpose="train") raises EvalSplitError, reported as a RunError naming
  bench.items, the item, the dev split, and Agent Rule 5.
- load_bench without bench.items selects the train items only; its
  train-only view lists the dev name as held, and no dev SuiteItem is
  reachable from the TrainData.
- load_corpus refuses a corpus whose held_out leaves out a dev item of its
  suite, naming the item and Agent Rule 5, and takes one that lists the
  eval, unassigned, and dev items.

The suite, the corpus manifest, and the records are SYNTHETIC files the
tests write. No value here is a measurement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from train_fakes import (
    RECORDS,
    SUITE,
    SUITE_TEXT,
    Log,
    core_name,
    fake_registry,
    jsonl,
    on_bench,
    train_data,
    write_recipe,
    write_suite,
)

from lassi.bench import EvalSplitError, Suite, SuiteItem
from lassi.core.runner import RunError

# SYNTHETIC: the shared suite (two train items, one eval, one unassigned) plus one dev item.
DEV_SUITE_TEXT = SUITE_TEXT + (
    "  tuned:\n"
    "    split: dev\n"
    "    languages:\n"
    "      omp: {dir: src/tuned-omp, files: [main.cpp]}\n"
)
CORPUS = "synthetic-dev-corpus"
RECORDS_FILE = f"{CORPUS}.jsonl"


def bench_recipe(tmp_path: Path, **bench: Any) -> Any:
    """Write and load a train recipe that reads the SYNTHETIC suite's train split; `bench` changes its keys."""
    loader = core_name("lassi.core.recipe", "load_train_recipe")
    path = write_recipe(tmp_path / "recipes", on_bench(**bench))
    return loader(path, roots=[tmp_path / "recipes"], registry=fake_registry(Log()))


def load_dev_bench(tmp_path: Path, **bench: Any) -> Any:
    """Return load_bench over the four-split suite; fail clearly while the manifest's split dev is refused."""
    suites = tmp_path / "suites"
    write_suite(suites, DEV_SUITE_TEXT)
    try:
        return train_data().load_bench(bench_recipe(tmp_path, **bench), suites)
    except RunError as error:
        if "cannot load the suite manifest" in str(error) and "'dev'" in str(error):
            pytest.fail(f"load_suite refuses split dev, which is not a split yet (task P17.31): {error}")
        raise


def reachable(root: object) -> list[object]:
    """Return every object reachable from `root` through attributes, mapping keys and values, and sequences."""
    seen: dict[int, object] = {}
    stack = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, (str, bytes, int, float, bool, type(None), type, ModuleType)):
            continue
        seen[id(obj)] = obj
        if isinstance(obj, Mapping):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        if hasattr(obj, "__dict__"):
            stack.extend(vars(obj).values())
        for slot in getattr(type(obj), "__slots__", ()):
            if hasattr(obj, slot):
                stack.append(getattr(obj, slot))
    return list(seen.values())


# ---------------------------------------------------------------------------
# bench data


def test_load_bench_refuses_a_dev_item_through_the_registry(tmp_path: Path) -> None:
    """A dev item named in bench.items is refused as a RunError from EvalSplitError, naming the dev split."""
    with pytest.raises(RunError) as refused:
        load_dev_bench(tmp_path, items=["alpha", "tuned"])
    message = str(refused.value)
    assert "bench.items" in message and "tuned" in message and "Agent Rule 5" in message, message
    assert "dev split" in message, message
    assert isinstance(refused.value.__cause__, EvalSplitError), message


def test_load_bench_selects_train_items_only_and_holds_the_dev_name_as_held(tmp_path: Path) -> None:
    """Without bench.items the train items alone are selected; the train-only view lists the dev name as held."""
    data, identity = load_dev_bench(tmp_path)
    assert list(data.items) == ["alpha", "beta"] and identity["items"] == ["alpha", "beta"]
    assert dict(data.suite.held) == {"held": "eval", "open": "unassigned", "tuned": "dev"}
    with pytest.raises(EvalSplitError):
        data.bench_item("tuned")


def test_no_dev_item_is_reachable_from_loaded_bench_data(tmp_path: Path) -> None:
    """No SuiteItem but the train ones, and no Suite, is reachable from the TrainData."""
    data, _ = load_dev_bench(tmp_path)
    found = reachable(data)
    assert not [obj for obj in found if isinstance(obj, Suite)], "TrainData holds the whole suite"
    assert sorted(obj.split for obj in found if isinstance(obj, SuiteItem)) == ["train", "train"]


# ---------------------------------------------------------------------------
# corpus data


def write_dev_corpus(root: Path, held: list[str]) -> None:
    """Write a SYNTHETIC corpus manifest whose held_out lists `held`, its records, and the four-split suite."""
    content = jsonl(RECORDS)
    manifest = {
        "corpus": CORPUS,
        "records": {"file": RECORDS_FILE, "sha256": hashlib.sha256(content).hexdigest(),
                    "count": content.count(b"\n")},
        "held_out": {"suite": SUITE, "items": list(held)},
    }
    (root / "manifests" / CORPUS).mkdir(parents=True)
    (root / "manifests" / CORPUS / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("ascii"))
    (root / "records").mkdir()
    (root / "records" / RECORDS_FILE).write_bytes(content)
    write_suite(root / "suites", DEV_SUITE_TEXT)


def load_dev_corpus(root: Path) -> Any:
    """Return load_corpus's (TrainData, identity) for the corpus under `root`."""
    return train_data().load_corpus(CORPUS, root / "manifests", root / "records", root / "suites")


def test_a_corpus_that_does_not_exclude_a_dev_item_is_refused_agent_rule_5(tmp_path: Path) -> None:
    """A corpus whose held_out lists the eval and unassigned items but not the dev one is refused, naming it."""
    write_dev_corpus(tmp_path, ["held", "open"])
    with pytest.raises(RunError) as refused:
        load_dev_corpus(tmp_path)
    message = str(refused.value)
    assert "data.corpus" in message and "'tuned'" in message and "dev" in message, message
    assert "Agent Rule 5" in message and "cannot load" not in message, message


def test_a_corpus_that_excludes_every_held_item_dev_included_loads(tmp_path: Path) -> None:
    """A corpus whose held_out lists the eval, unassigned, and dev items loads."""
    write_dev_corpus(tmp_path, ["held", "open", "tuned"])
    data, _ = load_dev_corpus(tmp_path)
    assert data.source == "corpus"
    assert [dict(record) for record in data.records] == [dict(record) for record in RECORDS]
