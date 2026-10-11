"""Tests for the corpus tables: Hive Parquet, Policy B as records, pair sets, verification, dedup (task P5.11).

Bible: Corpus Pipeline (nothing enters a pair set without an execution
round trip; Normalize: deduplicate by structural hash; Record: Parquet),
Result Record (Storage: Hive-partitioned Parquet), Benchmark Suites (split
rules: an eval or unassigned item is refused to training and corpus
harvest), Review Of v0 (the Keep row: Policy B and the Hive Parquet, moved
as failures recorded, never deleted, and dedup by structural hash; the
-Ddfloat row: exclude from pair sets unless built with the Makefile's
values; the whale truncation row), Agent Rules 1, 5, and 11. Plan:
plans/p5-ir.md, task P5.11 (eval and unassigned items are read only for
purpose eval into a verification table that no pair-set writer or training
reader accepts), and the planning decision "v0 migration (P5.2, P5.11)";
plans/spikes/p5-v0-corpus.md, Results 5, the migration map's Policy B row
(a unit that fails a step is recorded with its status, counted, and kept
out of pair sets, and its siblings from the same source stay; nothing is
deleted; test: a failing unit leaves its siblings and appears in the
counts, and no file is removed) and Hive Parquet row (records written
through lassi's Parquet conventions, partitioned by suite and item, with
the suite and split on every record; not moved: the 2 MiB cap and the
(benchmark, pass) dedup; test: write and read back, partition keys, and two
stylistic forms of one module dedup to one record).

The contract these tests fix (lassi.corpus):

- write_corpus(records, out_dir) writes train records, every one it is
  given (no dedup, no cap, failing entries included), as Hive Parquet under
  out_dir/corpus/suite=<suite>/item=<item>/part-0.parquet, as
  lassi.core.parquet writes run tables; split and pair_id are columns of
  every file. It refuses, before any record is written: a record whose split
  is eval or unassigned (lassi.bench EvalSplitError, saying Agent Rule 5),
  two records with one pair_id (ValueError naming it), and an out_dir that
  already holds a corpus table (FileExistsError or ValueError), which it
  never removes or replaces. read_corpus(out_dir) returns the records
  rebuilt and checked, so a round trip gives equal records.
- write_verification(records, out_dir) writes eval and unassigned records
  the same way under out_dir/verification/...; read_verification(out_dir)
  returns a VerificationTable whose `records` are the records rebuilt.
- write_pair_set(records, out_dir) writes the pair set under
  out_dir/pairs/suite=<suite>/item=<item>/part-0.parquet: the records that
  passed every gate step and were built with their source build's defines.
  It returns a mapping of what it left out, by reason, to counts: a failing
  entry under its gate status code, an entry whose build_defines differ
  from its source_defines under "build-defines". It refuses a
  VerificationTable and any eval or unassigned record (EvalSplitError,
  saying Agent Rule 5) before any record is written.
- Every writer claims its table's directory with an exclusive mkdir,
  checks again after the claim that no table of the other kind
  (verification beside corpus or pairs) has appeared, and writes
  <table>/_complete (`records <n>`) last; every reader refuses a table
  whose _complete is missing or malformed or whose count differs.
- read_pair_set(out_dir), the training reader, returns the pair set's
  records. It refuses a directory that holds a verification table
  (EvalSplitError) and one that holds no pair set (ValueError).
- dedup(records, normalize=<seam>) returns an object with `kept` (the first
  record of each structural group, in input order) and `dropped` (the
  pair_ids of the others), so duplicates are counted, never deleted. The
  structural hash is taken over each record's IR after the seam, a
  callable Module -> Module (lassi.core.interfaces.Module; the IRLevel's
  normalize, task P5.6), never over the suite, item, or step. The tests
  pass a SYNTHETIC normalizer that maps two stylistic forms of one module
  to one text, since the real Normalize needs the pinned tools (P5.6).

Every record and text is SYNTHETIC (tests/corpus/corpus_fixtures.py). No
value here is a measurement.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from corpus_fixtures import (
    OTHER_SUITE,
    TWO_MIB,
    big_module,
    by_pair_id,
    chain,
    corpus,
    cpu_ir,
    failed_at,
    files_under,
    lines,
    make_record,
    snapshot,
    tt_ir,
)

from lassi.bench import EvalSplitError
from lassi.core.interfaces import Module

# SYNTHETIC: cpu_ir("scale") in another style: a comment, a blank line, other indentation, other value names.
STYLISTIC_FORM = lines(
    "// SYNTHETIC: the same module in another style",
    'module @"src/demo/scale.c" {',
    "",
    "func.func @scale(%arg0: memref<16xf32>, %arg1: f32) {",
    "  affine.for %i = 0 to 16 {",
    "    %v = affine.load %arg0[%i] : memref<16xf32>  // load",
    "    %w = arith.mulf %v, %arg1 : f32",
    "    affine.store %w, %arg0[%i] : memref<16xf32>",
    "  }",
    "  return",
    "}",
    "}",
)


def mixed_records() -> list[Any]:
    """Return SYNTHETIC train records over two suites and three items, with every kind of field value."""
    module = corpus()
    return [
        make_record("scale"),
        make_record(
            "shift",
            item="beta",
            program_spec="SYNTHETIC program spec\n",
            build_defines={"dfloat": "float"},
            source_defines={"dfloat": "float"},
        ),
        make_record("early", source_ir=None, target_ir=None, gate_status=failed_at("raise"), provenance=chain()),
        make_record("slow", gate_status=module.GateStatus(passed="build", code="timeout")),
        make_record("scale", suite=OTHER_SUITE),
    ]


def policy_b_records() -> list[Any]:
    """Return three SYNTHETIC units of one source; the compute unit failed at build."""
    return [make_record("reader"), make_record("compute", gate_status=failed_at("build")), make_record("writer")]


def held_records() -> list[Any]:
    """Return a SYNTHETIC eval record and a SYNTHETIC unassigned record."""
    return [make_record("held", item="held", split="eval"), make_record("open", item="open", split="unassigned")]


# ---------------------------------------------------------------------------
# Hive Parquet


def test_corpus_records_are_partitioned_by_suite_and_item(tmp_path: Path) -> None:
    out = tmp_path / "out"
    corpus().write_corpus(mixed_records(), out)
    assert files_under(out) == {
        "corpus/suite=demo-suite/item=alpha/part-0.parquet",
        "corpus/suite=demo-suite/item=beta/part-0.parquet",
        "corpus/suite=other-suite/item=alpha/part-0.parquet",
        "corpus/_complete",
    }


def test_corpus_records_read_back_equal(tmp_path: Path) -> None:
    module = corpus()
    records = mixed_records()
    out = tmp_path / "out"
    module.write_corpus(records, out)
    back = list(module.read_corpus(out))
    assert by_pair_id(back) == by_pair_id(records)
    assert all(record.split == "train" and record.suite for record in back)


def test_split_and_pair_id_are_columns_of_every_file(tmp_path: Path) -> None:
    out = tmp_path / "out"
    corpus().write_corpus(mixed_records(), out)
    paths = sorted(out.rglob("*.parquet"))
    assert paths
    for path in paths:
        table = pq.read_table(path)
        assert {"split", "pair_id"} <= set(table.column_names), (path, table.column_names)
        assert set(table.column("split").to_pylist()) == {"train"}


@pytest.mark.parametrize("split", ["eval", "unassigned"])
def test_the_corpus_writer_refuses_an_eval_or_unassigned_record_before_writing(tmp_path: Path, split: str) -> None:
    out = tmp_path / "out"
    record = make_record("held", item="held", split=split)
    with pytest.raises(EvalSplitError) as refused:
        corpus().write_corpus([make_record("scale"), record], out)
    assert "Agent Rule 5" in str(refused.value) and record.pair_id in str(refused.value), str(refused.value)
    assert not out.exists() or files_under(out) == set()


def test_the_corpus_writer_refuses_one_pair_id_twice(tmp_path: Path) -> None:
    out = tmp_path / "out"
    twice = make_record("scale", oracle_io={})
    with pytest.raises(ValueError, match=twice.pair_id):
        corpus().write_corpus([make_record("scale"), twice], out)
    assert not out.exists() or files_under(out) == set()


def test_the_corpus_writer_never_removes_or_replaces_a_table(tmp_path: Path) -> None:
    module = corpus()
    out = tmp_path / "out"
    module.write_corpus(mixed_records(), out)
    (out / "notes.txt").write_bytes(b"SYNTHETIC notes\n")
    before = snapshot(out)
    with pytest.raises((FileExistsError, ValueError)):
        module.write_corpus([make_record("other")], out)
    assert snapshot(out) == before


def test_the_corpus_writer_writes_structural_duplicates_since_dedup_is_its_own_counted_step(tmp_path: Path) -> None:
    module = corpus()
    same = [make_record("scale"), make_record("scale", pair_id="demo-suite/alpha/cpp-tt/scale-again")]
    out = tmp_path / "out"
    module.write_corpus(same, out)
    assert len(list(module.read_corpus(out))) == 2


def test_a_module_over_two_mib_round_trips_whole(tmp_path: Path) -> None:
    module = corpus()
    record = make_record("big", source_ir=big_module(TWO_MIB))
    out = tmp_path / "out"
    module.write_corpus([record], out)
    (back,) = module.read_corpus(out)
    assert back.source_ir == record.source_ir


# ---------------------------------------------------------------------------
# Policy B as records


def test_a_failing_unit_is_recorded_counted_and_kept_out_of_the_pair_set_while_its_siblings_stay(
    tmp_path: Path,
) -> None:
    module = corpus()
    records = policy_b_records()
    corpus_dir, pairs_dir = tmp_path / "corpus-out", tmp_path / "pairs-out"
    corpus_dir.mkdir()
    (corpus_dir / "notes.txt").write_bytes(b"SYNTHETIC notes kept beside the table\n")
    module.write_corpus(records, corpus_dir)
    written = snapshot(corpus_dir)
    back = by_pair_id(module.read_corpus(corpus_dir))
    assert back == by_pair_id(records), "the failing unit is recorded with its status"
    left_out = module.write_pair_set(back, pairs_dir)
    assert dict(left_out) == {module.failure_code("build"): 1}
    pairs = module.read_pair_set(pairs_dir)
    assert sorted(record.pair_id for record in pairs) == [
        "demo-suite/alpha/cpp-tt/reader",
        "demo-suite/alpha/cpp-tt/writer",
    ]
    assert files_under(pairs_dir) == {"pairs/suite=demo-suite/item=alpha/part-0.parquet", "pairs/_complete"}
    assert snapshot(corpus_dir) == written, "nothing is removed"
    assert len(list(module.read_corpus(corpus_dir))) == 3


def test_the_pair_set_writer_leaves_out_entries_built_with_other_defines_than_the_source_builds(
    tmp_path: Path,
) -> None:
    module = corpus()
    other_values = make_record(
        "adv", build_defines={"dfloat": "float", "dlong": "int"}, source_defines={"dfloat": "double", "dlong": "int"}
    )
    same_values = make_record(
        "axh", build_defines={"dfloat": "float", "dlong": "int"}, source_defines={"dfloat": "float", "dlong": "int"}
    )
    plain = make_record("plain")
    left_out = module.write_pair_set([other_values, same_values, plain], tmp_path / "pairs")
    assert dict(left_out) == {"build-defines": 1}
    kept = sorted(record.pair_id for record in module.read_pair_set(tmp_path / "pairs"))
    assert kept == [same_values.pair_id, plain.pair_id]


# ---------------------------------------------------------------------------
# The verification table: eval and unassigned items, which no pair-set writer or training reader accepts


def test_eval_and_unassigned_records_go_into_a_verification_table(tmp_path: Path) -> None:
    module = corpus()
    out = tmp_path / "verify"
    module.write_verification(held_records(), out)
    assert files_under(out) == {
        "verification/suite=demo-suite/item=held/part-0.parquet",
        "verification/suite=demo-suite/item=open/part-0.parquet",
        "verification/_complete",
    }
    table = module.read_verification(out)
    assert isinstance(table, module.VerificationTable)
    assert by_pair_id(table.records) == by_pair_id(held_records())
    assert sorted(record.split for record in table.records) == ["eval", "unassigned"]


def test_the_pair_set_writer_refuses_a_verification_table(tmp_path: Path) -> None:
    module = corpus()
    module.write_verification(held_records(), tmp_path / "verify")
    table = module.read_verification(tmp_path / "verify")
    for given in (table, list(table.records)):
        with pytest.raises(EvalSplitError) as refused:
            module.write_pair_set(given, tmp_path / "pairs")
        assert "Agent Rule 5" in str(refused.value), str(refused.value)
    assert not (tmp_path / "pairs").exists() or files_under(tmp_path / "pairs") == set()


@pytest.mark.parametrize("split", ["eval", "unassigned"])
def test_the_pair_set_writer_refuses_one_held_record_among_train_ones(tmp_path: Path, split: str) -> None:
    module = corpus()
    with pytest.raises(EvalSplitError):
        module.write_pair_set([make_record("scale"), make_record("held", item="held", split=split)], tmp_path / "p")
    assert not (tmp_path / "p").exists() or files_under(tmp_path / "p") == set()


def test_the_training_reader_refuses_a_verification_table(tmp_path: Path) -> None:
    module = corpus()
    module.write_verification(held_records(), tmp_path / "verify")
    with pytest.raises(EvalSplitError) as refused:
        module.read_pair_set(tmp_path / "verify")
    assert "Agent Rule 5" in str(refused.value), str(refused.value)


def test_the_training_reader_refuses_a_corpus_table_that_is_not_a_pair_set(tmp_path: Path) -> None:
    module = corpus()
    module.write_corpus(policy_b_records(), tmp_path / "corpus-out")
    with pytest.raises(ValueError):
        module.read_pair_set(tmp_path / "corpus-out")


def relabel(out: Path, table: str, as_table: str) -> None:
    """Rename a written table's directory to another table's name, as a hand-made or damaged table would be."""
    (out / table).rename(out / as_table)


def test_the_training_reader_refuses_a_record_a_pair_set_leaves_out(tmp_path: Path) -> None:
    module = corpus()
    out = tmp_path / "out"
    module.write_corpus(policy_b_records(), out)
    relabel(out, "corpus", "pairs")
    with pytest.raises(ValueError, match="leaves out"):
        module.read_pair_set(out)


@pytest.mark.parametrize("table", ["pairs", "corpus"])
def test_the_training_and_corpus_readers_refuse_a_held_record_in_their_table(tmp_path: Path, table: str) -> None:
    module = corpus()
    out = tmp_path / "out"
    module.write_verification(held_records(), out)
    relabel(out, "verification", table)
    reader = module.read_pair_set if table == "pairs" else module.read_corpus
    with pytest.raises(EvalSplitError) as refused:
        reader(out)
    message = str(refused.value)
    assert "Agent Rule 5" in message and "may never enter" in message, message
    assert any(record.pair_id in message for record in held_records()), message


@pytest.mark.parametrize("table", ["corpus", "pairs"])
def test_a_verification_table_never_joins_a_directory_that_holds_a_training_table(tmp_path: Path, table: str) -> None:
    module = corpus()
    out = tmp_path / "out"
    (module.write_corpus if table == "corpus" else module.write_pair_set)(policy_b_records(), out)
    before = snapshot(out)
    with pytest.raises(ValueError, match="Agent Rule 5"):
        module.write_verification(held_records(), out)
    assert snapshot(out) == before
    assert sorted(path.name for path in out.iterdir()) == [table]


@pytest.mark.parametrize("table", ["corpus", "pairs"])
def test_a_training_table_never_joins_a_directory_that_holds_a_verification_table(tmp_path: Path, table: str) -> None:
    module = corpus()
    out = tmp_path / "out"
    module.write_verification(held_records(), out)
    before = snapshot(out)
    with pytest.raises(ValueError, match="Agent Rule 5"):
        (module.write_corpus if table == "corpus" else module.write_pair_set)(policy_b_records(), out)
    assert snapshot(out) == before
    assert sorted(path.name for path in out.iterdir()) == ["verification"]


def test_a_failed_write_leaves_no_partial_table_and_the_next_write_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = corpus()
    tables = importlib.import_module("lassi.corpus.tables")

    def partial(data: Any, base_dir: str, **kwargs: Any) -> None:
        """Write one SYNTHETIC file, then fail as a file system that refuses a name would."""
        Path(base_dir, "part-0.parquet").write_bytes(b"SYNTHETIC partial file\n")
        raise OSError("SYNTHETIC: the file system refused a name")

    out = tmp_path / "out"
    with monkeypatch.context() as patch:
        patch.setattr(tables.ds, "write_dataset", partial)
        with pytest.raises(OSError, match="SYNTHETIC"):
            module.write_corpus(mixed_records(), out)
    assert list(out.iterdir()) == []
    module.write_corpus(mixed_records(), out)
    assert sorted(path.name for path in out.iterdir()) == ["corpus"]
    assert by_pair_id(module.read_corpus(out)) == by_pair_id(mixed_records())


def test_an_empty_pair_set_reads_back_as_empty_not_missing(tmp_path: Path) -> None:
    module = corpus()
    out = tmp_path / "pairs-out"
    left_out = module.write_pair_set([make_record("compute", gate_status=failed_at("build"))], out)
    assert dict(left_out) == {module.failure_code("build"): 1}
    assert files_under(out) == {"pairs/_complete"}
    assert module.read_pair_set(out) == []
    assert "records: 0" in module.summarize(out)


@pytest.mark.parametrize("damage", ["no-complete-file", "malformed-complete-file", "a-file-gone"])
def test_a_table_whose_write_did_not_finish_is_never_read(tmp_path: Path, damage: str) -> None:
    module = corpus()
    out = tmp_path / "out"
    module.write_corpus(mixed_records(), out)
    complete = out / "corpus" / "_complete"
    if damage == "no-complete-file":
        complete.unlink()
    elif damage == "malformed-complete-file":
        complete.write_bytes(b"rec")
    else:
        sorted((out / "corpus").rglob("*.parquet"))[0].unlink()
    with pytest.raises(ValueError, match="incomplete" if damage != "a-file-gone" else "_complete file says 5"):
        module.read_corpus(out)
    with pytest.raises(FileExistsError):
        module.write_corpus(mixed_records(), out)


def test_a_writer_that_loses_the_race_for_a_table_leaves_the_winners_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = corpus()
    tables = importlib.import_module("lassi.corpus.tables")
    out = tmp_path / "out"
    checked = tables._refuse_place
    raced: list[bool] = []

    def then_another_writer(out_dir: Path, table: str) -> None:
        """Pass the check, then, once, let another writer write its empty pair set before the claim."""
        checked(out_dir, table)
        if not raced:
            raced.append(True)
            module.write_pair_set([], out_dir)

    monkeypatch.setattr(tables, "_refuse_place", then_another_writer)
    with pytest.raises(FileExistsError, match="never removes or replaces a table"):
        module.write_pair_set([make_record("scale")], out)
    assert files_under(out) == {"pairs/_complete"}
    assert module.read_pair_set(out) == []


@pytest.mark.parametrize(
    ("first", "second"), [("corpus", "verification"), ("pairs", "verification"), ("verification", "corpus")]
)
def test_two_writers_of_the_two_kinds_that_race_never_leave_both_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: str, second: str
) -> None:
    module = corpus()
    tables = importlib.import_module("lassi.corpus.tables")
    writers = {
        "corpus": lambda out_dir: module.write_corpus(policy_b_records(), out_dir),
        "pairs": lambda out_dir: module.write_pair_set(policy_b_records(), out_dir),
        "verification": lambda out_dir: module.write_verification(held_records(), out_dir),
    }
    out = tmp_path / "out"
    checked = tables._refuse_place
    raced: list[bool] = []

    def then_another_writer(out_dir: Path, table: str) -> None:
        """Pass the check, then, once, let a writer of the other kind write its whole table before the claim."""
        checked(out_dir, table)
        if not raced:
            raced.append(True)
            writers[second](out_dir)

    monkeypatch.setattr(tables, "_refuse_place", then_another_writer)
    with pytest.raises(ValueError, match="Agent Rule 5"):
        writers[first](out)
    assert sorted(path.name for path in out.iterdir()) == [second]


# ---------------------------------------------------------------------------
# Dedup by structural hash, through the normalizer seam


def synthetic_normalize(module: Module) -> Module:
    """A SYNTHETIC stand-in for P5.6's Normalize: the stylistic form becomes cpu_ir("scale"); all else is kept."""
    canonical = {STYLISTIC_FORM: cpu_ir("scale")}
    return Module(level=module.level, text=canonical.get(module.text, module.text), tag=module.tag)


def identity(module: Module) -> Module:
    """Return the module unchanged."""
    return module


def test_two_stylistic_forms_of_one_module_dedup_to_one_record() -> None:
    module = corpus()
    first = make_record("scale")
    styled = make_record("scale_b", source_ir=STYLISTIC_FORM, target_ir=tt_ir("scale"))
    result = module.dedup([first, styled], normalize=synthetic_normalize)
    assert list(result.kept) == [first]
    assert list(result.dropped) == [styled.pair_id]


def test_dedup_hashes_the_normalizers_output() -> None:
    module = corpus()
    first = make_record("scale")
    styled = make_record("scale_b", source_ir=STYLISTIC_FORM, target_ir=tt_ir("scale"))
    result = module.dedup([first, styled], normalize=identity)
    assert list(result.kept) == [first, styled] and list(result.dropped) == []


def test_dedup_keys_on_structure_not_on_the_suite_item_or_step() -> None:
    module = corpus()
    one = make_record("scale")
    other_ir_same_item = make_record("shift")
    same_ir_other_item = make_record("scale", item="beta")
    result = module.dedup([one, other_ir_same_item, same_ir_other_item], normalize=identity)
    assert list(result.kept) == [one, other_ir_same_item]
    assert list(result.dropped) == [same_ir_other_item.pair_id]
