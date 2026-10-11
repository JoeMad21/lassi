"""Tests for corpus input through the bench registry (task P5.11; Agent Rule 5).

Bible: Benchmark Suites (split rules: the bench registry, Suite.item,
refuses an eval or unassigned item to training, prompt tuning, and corpus
harvest with EvalSplitError naming its split, and gives it for
evaluation), Result Record (harvest takes gated train-split attempts
only), Agent Rule 5. Plan: plans/p5-ir.md, task P5.11 (corpus input comes
only through the bench registry: purpose harvest for train items, which
refuses eval and unassigned items; eval and unassigned items are read only
for purpose eval, into a verification table) and its constraint "Agent
Rule 5" (the corpus reads bench items only through lassi.bench.registry,
Suite.item, harvest only for train items).

The contract these tests fix (lassi.corpus):

- harvest_item(suite, name) returns the item through
  suite.item(name, purpose="harvest"), so an eval or unassigned item raises
  EvalSplitError with the registry's own words.
- verification_item(suite, name) returns the item through
  suite.item(name, purpose="eval"); its records go to a verification table
  (tests/corpus/test_corpus_tables.py), never to the corpus or a pair set.

The manifest is SYNTHETIC, written under the test's directory; no file is
read. No value here is a measurement.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from corpus_fixtures import corpus

from lassi import bench

# SYNTHETIC: one train item, one eval item, and one unassigned item.
MANIFEST = (
    "suite: corpus-demo\n"
    "repo: https://example.invalid/corpus-demo.git\n"
    f"commit: {'c' * 40}\n"
    "items:\n"
    "  alpha:\n"
    "    split: train\n"
    "    languages:\n"
    "      cpp: {dir: src/alpha-cpp, files: [main.cpp]}\n"
    "  held:\n"
    "    split: eval\n"
    "    languages:\n"
    "      cpp: {dir: src/held-cpp, files: [main.cpp]}\n"
    "  open:\n"
    "    split: unassigned\n"
    "    languages:\n"
    "      cpp: {dir: src/open-cpp, files: [main.cpp]}\n"
)


def suite(tmp_path: Path) -> bench.Suite:
    """Load the SYNTHETIC suite."""
    path = tmp_path / "corpus-demo.yaml"
    path.write_bytes(MANIFEST.encode("ascii"))
    return bench.load_suite(path)


@pytest.fixture
def purposes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record every Suite.item call as (name, purpose), calling the real method."""
    calls: list[tuple[str, str]] = []
    original = bench.Suite.item

    def spy(self: bench.Suite, name: str, *, purpose: str) -> bench.SuiteItem:
        calls.append((name, purpose))
        return original(self, name, purpose=purpose)

    monkeypatch.setattr(bench.Suite, "item", spy)
    return calls


def test_harvest_reads_a_train_item_through_the_registry_for_purpose_harvest(
    tmp_path: Path, purposes: list[tuple[str, str]]
) -> None:
    loaded = suite(tmp_path)
    found = corpus().harvest_item(loaded, "alpha")
    assert found is loaded.items["alpha"] and found.split == "train"
    assert purposes == [("alpha", "harvest")]


@pytest.mark.parametrize("name", ["held", "open"], ids=["eval", "unassigned"])
def test_harvest_refuses_an_eval_or_unassigned_item_with_the_registrys_words(
    tmp_path: Path, name: str, purposes: list[tuple[str, str]]
) -> None:
    loaded = suite(tmp_path)
    module = corpus()
    with pytest.raises(bench.EvalSplitError) as by_registry:
        loaded.item(name, purpose="harvest")
    with pytest.raises(bench.EvalSplitError) as by_corpus:
        module.harvest_item(loaded, name)
    assert str(by_corpus.value) == str(by_registry.value)
    assert "Agent Rule 5" in str(by_corpus.value)
    assert purposes[-1] == (name, "harvest")


@pytest.mark.parametrize("name", ["held", "open"], ids=["eval", "unassigned"])
def test_an_eval_or_unassigned_item_is_read_for_purpose_eval_for_verification(
    tmp_path: Path, name: str, purposes: list[tuple[str, str]]
) -> None:
    loaded = suite(tmp_path)
    found = corpus().verification_item(loaded, name)
    assert found is loaded.items[name]
    assert purposes == [(name, "eval")]
