"""Tests for dev records and dev items in the corpus (task P17.31; OQ-083, option (b); Agent Rule 5).

Bible: Benchmark Suites (split rules: tt-pairs-v0 is a development set,
2026-10-25: prompt and feature work on its items is allowed, training and
corpus harvest are not), Corpus Pipeline (Record), Agent Rule 5. Plan:
plans/p17-portable.md, P17.31.

The contract these tests fix (lassi.corpus), each the eval and unassigned
behaviour extended to dev, the registry's fourth split:

- CorpusRecord accepts split dev.
- write_corpus and write_pair_set refuse a dev record, before any record is
  written, with EvalSplitError naming the record and Agent Rule 5.
- write_verification takes dev records, VerificationTable holds them, and
  read_verification returns them; read_corpus and read_pair_set refuse a dev
  record found in their table.
- summarize counts dev records on a `split dev: <n>` line after the eval
  and unassigned lines (the registry's SPLITS order).
- harvest_item refuses a dev item with the registry's own words, through
  Suite.item(name, purpose="harvest"); verification_item gives it, through
  Suite.item(name, purpose="eval").

Every record and the manifest are SYNTHETIC. No value here is a measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from corpus_fixtures import by_pair_id, corpus, files_under, make_record

from lassi import bench
from lassi.bench import EvalSplitError

# SYNTHETIC: one train item, one eval item, one unassigned item, and one dev item.
MANIFEST = (
    "suite: corpus-dev\n"
    "repo: https://example.invalid/corpus-dev.git\n"
    f"commit: {'e' * 40}\n"
    "items:\n"
    + "".join(
        f"  {name}:\n"
        f"    split: {split}\n"
        "    languages:\n"
        f"      cpp: {{dir: src/{name}-cpp, files: [main.cpp]}}\n"
        for name, split in (("alpha", "train"), ("held", "eval"), ("open", "unassigned"), ("tuned", "dev"))
    )
)


def dev_record() -> Any:
    """Return a SYNTHETIC dev record; fail clearly while CorpusRecord refuses split dev (task P17.31)."""
    try:
        return make_record("tuned", item="tuned", split="dev")
    except ValueError as error:
        if "split" in str(error):
            pytest.fail(f"CorpusRecord refuses split dev, which is not a split yet (task P17.31): {error}")
        raise


def held_and_dev_records() -> list[Any]:
    """Return a SYNTHETIC eval record, a SYNTHETIC unassigned record, and a SYNTHETIC dev record."""
    return [
        make_record("held", item="held", split="eval"),
        make_record("open", item="open", split="unassigned"),
        dev_record(),
    ]


def dev_suite(tmp_path: Path) -> bench.Suite:
    """Load the SYNTHETIC four-split suite; fail clearly while load_suite refuses split dev (task P17.31)."""
    path = tmp_path / "corpus-dev.yaml"
    path.write_bytes(MANIFEST.encode("ascii"))
    try:
        return bench.load_suite(path)
    except ValueError as error:
        if "split" in str(error) and "'dev'" in str(error):
            pytest.fail(f"load_suite refuses split dev, which is not a split yet (task P17.31): {error}")
        raise


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


# ---------------------------------------------------------------------------
# The record


def test_a_dev_record_builds() -> None:
    """CorpusRecord accepts split dev, the registry's fourth split."""
    assert dev_record().split == "dev"


# ---------------------------------------------------------------------------
# Training tables refuse dev records


def test_the_corpus_writer_refuses_a_dev_record_before_writing(tmp_path: Path) -> None:
    """write_corpus refuses a dev record among train ones with EvalSplitError, writing nothing."""
    out = tmp_path / "out"
    record = dev_record()
    with pytest.raises(EvalSplitError) as refused:
        corpus().write_corpus([make_record("scale"), record], out)
    assert "Agent Rule 5" in str(refused.value) and record.pair_id in str(refused.value), str(refused.value)
    assert not out.exists() or files_under(out) == set()


def test_the_pair_set_writer_refuses_a_dev_record_before_writing(tmp_path: Path) -> None:
    """write_pair_set refuses a dev record among train ones with EvalSplitError, writing nothing."""
    out = tmp_path / "pairs"
    record = dev_record()
    with pytest.raises(EvalSplitError) as refused:
        corpus().write_pair_set([make_record("scale"), record], out)
    assert "Agent Rule 5" in str(refused.value) and record.pair_id in str(refused.value), str(refused.value)
    assert not out.exists() or files_under(out) == set()


# ---------------------------------------------------------------------------
# The verification table holds dev records


def test_a_verification_table_holds_dev_records() -> None:
    """VerificationTable takes a dev record beside eval and unassigned ones."""
    records = held_and_dev_records()
    table = corpus().VerificationTable(records=tuple(records))
    assert sorted(record.split for record in table.records) == ["dev", "eval", "unassigned"]


def test_dev_records_go_into_a_verification_table_and_read_back(tmp_path: Path) -> None:
    """write_verification writes dev records under verification/, and read_verification returns them equal."""
    module = corpus()
    out = tmp_path / "verify"
    module.write_verification(held_and_dev_records(), out)
    assert "verification/suite=demo-suite/item=tuned/part-0.parquet" in files_under(out)
    table = module.read_verification(out)
    assert by_pair_id(table.records) == by_pair_id(held_and_dev_records())


@pytest.mark.parametrize("table", ["pairs", "corpus"])
def test_the_training_and_corpus_readers_refuse_a_dev_record_in_their_table(tmp_path: Path, table: str) -> None:
    """A table of dev records relabeled as a pair set or corpus table is refused by its reader (Agent Rule 5)."""
    module = corpus()
    out = tmp_path / "out"
    record = dev_record()
    module.write_verification([record], out)
    (out / "verification").rename(out / table)
    reader = module.read_pair_set if table == "pairs" else module.read_corpus
    with pytest.raises(EvalSplitError) as refused:
        reader(out)
    message = str(refused.value)
    assert "Agent Rule 5" in message and record.pair_id in message, message


def test_the_summary_counts_dev_records_after_eval_and_unassigned(tmp_path: Path) -> None:
    """summarize lists a split dev line, after the eval and unassigned lines, in the registry's SPLITS order."""
    module = corpus()
    out = tmp_path / "verify"
    module.write_verification(held_and_dev_records(), out)
    lines = module.summarize(out)
    assert [line for line in lines if line.startswith("split ")] == [
        "split eval: 1", "split unassigned: 1", "split dev: 1",
    ]


# ---------------------------------------------------------------------------
# Corpus input through the registry


def test_harvest_refuses_a_dev_item_with_the_registrys_words(tmp_path: Path, purposes: list[tuple[str, str]]) -> None:
    """harvest_item refuses a dev item through Suite.item(purpose="harvest"), in the registry's own words."""
    loaded = dev_suite(tmp_path)
    with pytest.raises(EvalSplitError) as by_registry:
        loaded.item("tuned", purpose="harvest")
    with pytest.raises(EvalSplitError) as by_corpus:
        corpus().harvest_item(loaded, "tuned")
    assert str(by_corpus.value) == str(by_registry.value)
    words = "corpus-dev/tuned is in the dev split and may never be used for corpus harvest (Agent Rule 5)"
    assert str(by_corpus.value) == words
    assert purposes[-1] == ("tuned", "harvest")


def test_a_dev_item_is_read_for_purpose_eval_for_verification(
    tmp_path: Path, purposes: list[tuple[str, str]]
) -> None:
    """verification_item gives a dev item through Suite.item(purpose="eval")."""
    loaded = dev_suite(tmp_path)
    found = corpus().verification_item(loaded, "tuned")
    assert found is loaded.items["tuned"] and found.split == "dev"
    assert purposes == [("tuned", "eval")]
