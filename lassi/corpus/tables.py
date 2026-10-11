"""The corpus tables: Hive Parquet of corpus records, pair sets, and verification tables (task P5.11).

Three tables, each a directory under an output directory (bible Corpus
Pipeline; Result Record, Storage; Benchmark Suites, split rules; Review Of
v0, the Keep row; Agent Rules 5 and 11):

- corpus/ holds train records, every one given, whatever its gate status
  (write_corpus, read_corpus). This is v0's Policy B moved as records (its
  scripts/lower_one.sh:167-187 at 3ccd280 deleted a snapshot that failed
  its round trip and kept the run's other snapshots): a unit that fails a
  step is recorded with its gate status, its siblings from the same source
  stay, and nothing is deleted. Duplicates are written too, since dedup
  (lassi.corpus.dedup) is a step of its own that counts what it drops.
- pairs/ holds a pair set (write_pair_set, read_pair_set): the train records
  that passed every gate step (Corpus Pipeline: nothing enters a pair set
  without an execution round trip) and were built with their source build's
  defines (Review Of v0, the -Ddfloat row). read_pair_set is the training
  reader: training code reads corpus pairs only through it (none does yet).
- verification/ holds eval and unassigned records (write_verification,
  read_verification, VerificationTable): the round trips of items read for
  purpose eval (lassi.corpus.inputs verification_item). No pair-set writer
  or training reader accepts them (Agent Rule 5).

Layout, v0's Hive Parquet moved (its tools/build_parquet.py:154-190): each
table is written as `<out_dir>/<table>/suite=<suite>/item=<item>/
part-0.parquet`, its partition values URI-encoded as pyarrow does by
default. suite and item are path keys only, so a record read back takes
them from its path; split and pair_id are columns of every file. The other
columns follow CorpusRecord's fields: the text fields as strings, oracle_io
and the two define maps as JSON text with sorted keys, gate_status as
gate_status_passed and gate_status_code, toolchain_pins as one
toolchain_pins_<name> column per lassi.core.record TOOLCHAIN_PIN_NAMES, and
provenance as three list columns, provenance_step, provenance_sha256, and
provenance_parent, in chain order. Not moved from v0: its 2 MiB cap (Agent
Rule 11), its dedup by benchmark and pass name (dedup is by structural
hash), and its zstd level 3 and row groups of 512 (the conventions below
apply).

A complete table is never removed, never replaced: a writer refuses, with
FileExistsError, an output directory whose table directory already exists,
and a verification table never shares an output directory with a corpus
table or a pair set (ValueError), so eval and unassigned records stay apart
from training data. Every refusal comes before any record is written. A
writer claims the table's directory with an exclusive mkdir, so two writers
never share one, then checks again that the output directory holds no table
of the other kind, since two writers of the two kinds can pass the first
check together (the later claim's check sees the earlier claim unless that
writer has already refused or failed and removed it, so two writers of the
two kinds never both finish); a refusal there removes the empty directory it
claimed. It writes the table's COMPLETE file (`_complete`, holding `records
<n>`) last. A write that fails, as on an OSError from the file system, or
that is refused after its claim, removes the unfinished table it made, as
far as the file system lets it; a process killed mid-write leaves one, which
lassi never removes. Every reader refuses a table whose COMPLETE file is
missing or malformed, or whose count differs from the records read, so an
unfinished table is never read as a table. Readers rebuild each record
through CorpusRecord, so every record read is checked again; a table that
does not read back raises ValueError naming its directory.

From lassi.core.parquet, the Parquet mirror of run records, these follow its
conventions: Hive partitioning with string keys, pyarrow's default
compression and row groups, `part-{i}.parquet` file names, nested fields as
`<field>_<key>` columns, maps as JSON text with sorted keys, list columns
for ordered lists (as its requests table), every column nullable, and its
partition-value check (no value ending in '.', and none differing from
another only by letter case). They depart from it where the corpus needs to:
a writer never removes a complete table (write_run_parquet rewrites each
table directory); a table's directory and its COMPLETE file are made even
with no row, so an empty pair set reads back as empty, not missing;
source_ir, target_ir, source_text, and program_spec are large_string
columns, since a corpus has no size cap and one column can pass the 2 GiB of
a string column; write_dataset's partition cap (1024 by default) is set to
the table's partition count, so no corpus is too wide to write; and readers
return checked records, not rows.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds

from lassi.bench.registry import HELD_SPLITS, SPLITS, EvalSplitError
from lassi.core.record import TOOLCHAIN_PIN_NAMES, ToolchainPins
from lassi.corpus.provenance import HashLink
from lassi.corpus.record import CorpusRecord
from lassi.corpus.status import FAILURE_CODES, GateStatus

CORPUS = "corpus"
VERIFICATION = "verification"
PAIRS = "pairs"
TABLES = (CORPUS, VERIFICATION, PAIRS)
PARTITION_COLUMNS = ("suite", "item")
# The reason a pair set leaves out a gated entry built with other defines than its source build's.
BUILD_DEFINES = "build-defines"
# The file a writer writes last in a table's directory, `records <n>`; pyarrow's discovery skips a leading '_'.
COMPLETE = "_complete"
_COMPLETE_TEXT = re.compile(r"records (0|[1-9][0-9]*)\n")

_STRING = pa.string()
_TEXT = pa.large_string()
_STRING_LIST = pa.list_(pa.string())
# Record fields stored as they are, in field order (suite and item are the partition keys).
_PLAIN = (
    "suite", "item", "pair_id", "split", "direction", "source_language", "target", "source_level", "source_ir",
    "target_level", "target_ir", "source_text", "program_spec",
)
_LARGE = {"source_ir", "target_ir", "source_text", "program_spec"}
_JSON = ("oracle_io", "build_defines", "source_defines")
_LINK_KEYS = ("step", "sha256", "parent")
SCHEMA = pa.schema(
    [
        *[(name, _TEXT if name in _LARGE else _STRING) for name in _PLAIN],
        ("oracle_io", _STRING),
        ("gate_status_passed", _STRING),
        ("gate_status_code", _STRING),
        ("build_defines", _STRING),
        ("source_defines", _STRING),
        *[(f"toolchain_pins_{name}", _STRING) for name in TOOLCHAIN_PIN_NAMES],
        *[(f"provenance_{key}", _STRING_LIST) for key in _LINK_KEYS],
    ]
)


@dataclass(frozen=True, kw_only=True)
class VerificationTable:
    """Eval and unassigned records, read for purpose eval only; no pair-set writer or training reader accepts one.

    `records` is kept as a tuple of CorpusRecords, each of split eval or
    unassigned; any other item raises ValueError. The table is not iterable,
    so it cannot pass for a list of records.
    """

    records: tuple[CorpusRecord, ...]

    def __post_init__(self) -> None:
        """Check that every record is an eval or unassigned CorpusRecord; keep them as a tuple."""
        records = tuple(self.records)
        for record in records:
            if not isinstance(record, CorpusRecord):
                raise ValueError(f"VerificationTable.records must hold CorpusRecords, got {record!r}")
            if record.split not in HELD_SPLITS:
                raise ValueError(
                    f"VerificationTable.records: {record.pair_id} is a {record.split} record; a verification table "
                    "holds eval and unassigned records only"
                )
        object.__setattr__(self, "records", records)


# ---------------------------------------------------------------------------
# Writers


def write_corpus(records: Iterable[CorpusRecord], out_dir: Path) -> None:
    """Write train records, every one given, as the corpus table under `out_dir` (Policy B as records).

    Refuses, before any record is written: a VerificationTable or an eval or
    unassigned record (EvalSplitError, saying Agent Rule 5 and naming the
    record), an item that is not a CorpusRecord (TypeError), two records
    with one pair_id (ValueError naming it), partition values that cannot be
    distinct directory names (ValueError), an `out_dir` that is a file
    (ValueError), and one that already holds a corpus table
    (FileExistsError) or a verification table (ValueError).
    """
    given = _train_records(records, "the corpus table")
    _write(given, Path(out_dir), CORPUS)


def write_verification(records: Iterable[CorpusRecord] | VerificationTable, out_dir: Path) -> None:
    """Write eval and unassigned records as the verification table under `out_dir`.

    `records` may be a VerificationTable or CorpusRecords; a train record or
    an item that is not a CorpusRecord raises ValueError (a train record
    belongs to the corpus table). Refused as write_corpus refuses them: two
    records with one pair_id, bad partition values, an `out_dir` that is a
    file, and one that already holds a verification table
    (FileExistsError); and an `out_dir` that holds a corpus table or a pair
    set raises ValueError. Every refusal comes before any record is written.
    """
    table = records if isinstance(records, VerificationTable) else VerificationTable(records=tuple(records))
    _write(list(table.records), Path(out_dir), VERIFICATION)


def write_pair_set(records: Iterable[CorpusRecord], out_dir: Path) -> dict[str, int]:
    """Write the pair set of `records` under `out_dir`; return what it left out, by reason, with counts.

    The pair set holds the records that passed every gate step and whose
    build_defines equal their source_defines. A record that did not pass is
    left out under its gate status code, and a gated one built with other
    defines under BUILD_DEFINES; the returned dict maps each reason present
    to its count, sorted by reason. It refuses, before any record is written,
    what write_corpus refuses, a VerificationTable and any eval or
    unassigned record (EvalSplitError, Agent Rule 5) among them, over every
    record given, left out or not; and an `out_dir` that already holds a
    pair set (FileExistsError) or a verification table (ValueError).
    """
    given = _train_records(records, "a pair set")
    _refuse_twice(given)
    kept: list[CorpusRecord] = []
    left_out: Counter[str] = Counter()
    for record in given:
        reason = _left_out(record)
        if reason is None:
            kept.append(record)
        else:
            left_out[reason] += 1
    _write(kept, Path(out_dir), PAIRS)
    return dict(sorted(left_out.items()))


def _train_records(records: Iterable[CorpusRecord] | VerificationTable, table: str) -> list[CorpusRecord]:
    """Return the records given for a training table; refuse a VerificationTable and every eval or unassigned one."""
    if isinstance(records, VerificationTable):
        raise EvalSplitError(
            f"a verification table holds eval and unassigned records and never becomes {table} (Agent Rule 5)"
        )
    given = list(records)
    for record in given:
        if not isinstance(record, CorpusRecord):
            raise TypeError(f"{table} takes CorpusRecords, got {record!r}")
        if record.split in HELD_SPLITS:
            raise EvalSplitError(
                f"{record.pair_id} is an {record.split} record and may never enter {table}, which training and "
                "corpus harvest read (Agent Rule 5); eval and unassigned records go to a verification table"
            )
    return given


def _left_out(record: CorpusRecord) -> str | None:
    """Return why a pair set leaves `record` out: its gate status code, BUILD_DEFINES, or None to keep it."""
    if not record.gate_status.gated:
        return record.gate_status.code
    if dict(record.build_defines) != dict(record.source_defines):
        return BUILD_DEFINES
    return None


def _write(records: list[CorpusRecord], out_dir: Path, table: str) -> None:
    """Write one table of `records` under `out_dir`; every refusal comes before any record is written.

    The table's directory is claimed with an exclusive mkdir and the output
    directory checked again for a table of the other kind, and the COMPLETE
    file is written last; a write that fails or is refused after the claim
    removes the unfinished table it made, as far as the file system lets it.
    """
    _refuse_twice(records)
    _check_partition_dirs(records)
    _refuse_place(out_dir, table)
    rows = sorted((_row(record) for record in records), key=lambda row: (row["suite"], row["item"], row["pair_id"]))
    try:
        data = pa.Table.from_pylist(rows, schema=SCHEMA)
    except (OverflowError, UnicodeError, pa.ArrowException) as error:
        raise ValueError(f"the {table} records do not fit the Parquet schema: {error}") from error
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out_dir / table
    try:
        table_dir.mkdir()  # the claim
    except FileExistsError:
        raise FileExistsError(_exists_message(table_dir)) from None
    try:
        # Again after the claim, so two writers of the two kinds that race never both finish.
        _refuse_mix(out_dir, table)
        if data.num_rows:
            ds.write_dataset(
                data,
                str(table_dir),
                format="parquet",
                partitioning=_partitioning(),
                basename_template="part-{i}.parquet",
                existing_data_behavior="error",
                max_partitions=len({(record.suite, record.item) for record in records}),
            )
        (table_dir / COMPLETE).write_bytes(f"records {data.num_rows}\n".encode("ascii"))
    except BaseException:
        shutil.rmtree(table_dir, ignore_errors=True)
        raise


def _refuse_twice(records: list[CorpusRecord]) -> None:
    """Refuse two records with one pair_id, naming it."""
    counts = Counter(record.pair_id for record in records)
    twice = sorted(pair_id for pair_id, count in counts.items() if count > 1)
    if twice:
        raise ValueError(f"pair_id {twice[0]} appears in {counts[twice[0]]} records; each pair_id is written once")


def _refuse_place(out_dir: Path, table: str) -> None:
    """Refuse an out_dir that is a file, already holds this table, or would mix a verification table with training."""
    if os.path.lexists(out_dir) and not out_dir.is_dir():
        raise ValueError(f"{out_dir} is not a directory")
    if os.path.lexists(out_dir / table):
        raise FileExistsError(_exists_message(out_dir / table))
    _refuse_mix(out_dir, table)


def _exists_message(table_dir: Path) -> str:
    """Return the refusal of a table directory that already exists."""
    return f"{table_dir} already exists; the corpus never removes or replaces a table, so write to a new directory"


def _refuse_mix(out_dir: Path, table: str) -> None:
    """Refuse an out_dir that holds a table of the other kind: verification beside a corpus table or a pair set."""
    apart = (CORPUS, PAIRS) if table == VERIFICATION else (VERIFICATION,)
    mixed = [name for name in apart if os.path.lexists(out_dir / name)]
    if mixed:
        raise ValueError(
            f"{out_dir} holds a {mixed[0]} table; a verification table never shares a directory with a corpus "
            "table or a pair set, so eval and unassigned records stay apart from training data (Agent Rule 5)"
        )


def _check_partition_dirs(records: list[CorpusRecord]) -> None:
    """Refuse suite and item values that would not map to distinct, faithful directory names on every OS.

    As lassi.core.parquet checks its partitions: Windows drops a trailing
    '.', and Windows and macOS compare names without letter case, so two
    suites, or two items of one suite, that differ only by case would share
    a directory.
    """
    seen: dict[tuple[str, ...], tuple[str, ...]] = {}
    for record in records:
        key = (record.suite, record.item)
        for name, value in zip(PARTITION_COLUMNS, key, strict=True):
            if value.endswith("."):
                raise ValueError(f"partition value {name}={value!r} ends with '.', which Windows drops")
        for depth in (1, 2):
            prefix = key[:depth]
            other = seen.setdefault(tuple(value.casefold() for value in prefix), prefix)
            if other != prefix:
                columns = "/".join(PARTITION_COLUMNS[:depth])
                raise ValueError(
                    f"partition {columns} values {other} and {prefix} differ only by letter case and would share a "
                    "directory"
                )


def _partitioning() -> ds.Partitioning:
    """Return the Hive partitioning by suite and item, both strings."""
    return ds.HivePartitioning(pa.schema([(name, _STRING) for name in PARTITION_COLUMNS]))


def _row(record: CorpusRecord) -> dict[str, Any]:
    """Return the table row of one record, in SCHEMA's columns."""
    row: dict[str, Any] = {name: getattr(record, name) for name in _PLAIN}
    for name in _JSON:
        row[name] = json.dumps(dict(getattr(record, name)), sort_keys=True, ensure_ascii=True)
    row["gate_status_passed"] = record.gate_status.passed
    row["gate_status_code"] = record.gate_status.code
    for name in TOOLCHAIN_PIN_NAMES:
        row[f"toolchain_pins_{name}"] = getattr(record.toolchain_pins, name)
    for key in _LINK_KEYS:
        row[f"provenance_{key}"] = [getattr(link, key) for link in record.provenance]
    return row


# ---------------------------------------------------------------------------
# Readers


def read_corpus(out_dir: Path) -> list[CorpusRecord]:
    """Return the corpus table's records under `out_dir`, rebuilt and checked, sorted by pair_id.

    Raises ValueError when `out_dir` holds no corpus table or one that does
    not read back, and EvalSplitError for an eval or unassigned record in it.
    """
    records = _read(Path(out_dir), CORPUS)
    _train_records(records, "the corpus table")
    return records


def read_verification(out_dir: Path) -> VerificationTable:
    """Return the verification table under `out_dir`, its records rebuilt and checked, sorted by pair_id.

    Raises ValueError when `out_dir` holds none, or one that does not read
    back or holds a train record.
    """
    return VerificationTable(records=tuple(_read(Path(out_dir), VERIFICATION)))


def read_pair_set(out_dir: Path) -> list[CorpusRecord]:
    """Return the pair set's records under `out_dir`, sorted by pair_id: the training reader.

    Training code reads corpus pairs only through this reader (none does
    yet). It refuses a directory that holds a verification table
    (EvalSplitError, Agent Rule 5), one with no pair set or one that does
    not read back (ValueError), an eval or unassigned record in it
    (EvalSplitError), and a record that write_pair_set would have left out
    (ValueError).
    """
    out_dir = Path(out_dir)
    if os.path.lexists(out_dir / VERIFICATION):
        raise EvalSplitError(f"{out_dir} holds a verification table, which no training reader accepts (Agent Rule 5)")
    records = _read(out_dir, PAIRS)
    _train_records(records, "a pair set")
    for record in records:
        reason = _left_out(record)
        if reason is not None:
            raise ValueError(f"{out_dir}: the pair set holds {record.pair_id}, which a pair set leaves out ({reason})")
    return records


def _read(out_dir: Path, table: str) -> list[CorpusRecord]:
    """Read one table under `out_dir` into records sorted by pair_id; ValueError when it is missing or unreadable."""
    table_dir = out_dir / table
    if not table_dir.is_dir():
        raise ValueError(f"{out_dir} holds no {table} table ({table_dir} is not a directory)")
    written = _written_count(table_dir)
    try:
        dataset = ds.dataset(str(table_dir), schema=SCHEMA, format="parquet", partitioning=_partitioning())
        rows = dataset.to_table().to_pylist()
    except (OSError, pa.ArrowException) as error:
        raise ValueError(f"{table_dir} does not read as a corpus table: {' '.join(str(error).split())}") from error
    if len(rows) != written:
        raise ValueError(f"{table_dir} holds {len(rows)} records where its {COMPLETE} file says {written}")
    records = []
    for row in rows:
        try:
            records.append(_record(row))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{table_dir}: the record {row.get('pair_id')!r} is refused: {error}") from error
    return sorted(records, key=lambda record: record.pair_id)


def _written_count(table_dir: Path) -> int:
    """Return the record count a table's COMPLETE file holds; ValueError when the file is missing or malformed."""
    try:
        text = (table_dir / COMPLETE).read_bytes().decode("ascii")
    except (OSError, UnicodeDecodeError):
        text = ""
    found = _COMPLETE_TEXT.fullmatch(text)
    if found is None:
        raise ValueError(
            f"{table_dir} is incomplete: its {COMPLETE} file is missing or malformed, so its write did not finish "
            "(lassi never removes a table; write to a new directory)"
        )
    return int(found.group(1))


def _record(row: dict[str, Any]) -> CorpusRecord:
    """Rebuild one record from its row; a value the record refuses raises ValueError."""
    maps = {}
    for name in _JSON:
        if not isinstance(row[name], str):
            raise ValueError(f"column {name} must hold JSON text, got {row[name]!r}")
        maps[name] = json.loads(row[name])
    lists = [row[f"provenance_{key}"] or [] for key in _LINK_KEYS]
    links = [HashLink(step=step, sha256=sha, parent=parent) for step, sha, parent in zip(*lists, strict=True)]
    return CorpusRecord(
        **{name: row[name] for name in _PLAIN},
        **maps,
        gate_status=GateStatus(passed=row["gate_status_passed"], code=row["gate_status_code"]),
        toolchain_pins=ToolchainPins(**{name: row[f"toolchain_pins_{name}"] for name in TOOLCHAIN_PIN_NAMES}),
        provenance=links,
    )


# ---------------------------------------------------------------------------
# Summary


def summarize(out_dir: Path) -> list[str]:
    """Return plain ASCII lines that count the records of each corpus table under `out_dir`.

    For each table present, in TABLES order: `table <name>`, `records: <n>`,
    `gated: <n>`, one `split <split>: <n>` line per split present (in the
    registry's SPLITS order), and one `status <code>: <n>` line per failure
    code present (in FAILURE_CODES order). Each table is read by its own
    reader, so every record is checked. Raises ValueError naming `out_dir`
    when it is not a directory or holds no table, and the readers' errors.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise ValueError(f"{out_dir} is not a directory")
    present = [table for table in TABLES if (out_dir / table).is_dir()]
    if not present:
        raise ValueError(f"{out_dir} holds no corpus table ({', '.join(f'{table}/' for table in TABLES)})")
    readers = {CORPUS: read_corpus, VERIFICATION: lambda path: read_verification(path).records, PAIRS: read_pair_set}
    lines: list[str] = []
    for table in present:
        records = list(readers[table](out_dir))
        splits = Counter(record.split for record in records)
        codes = Counter(record.gate_status.code for record in records)
        lines += [f"table {table}", f"records: {len(records)}", f"gated: {codes[None]}"]
        lines += [f"split {split}: {splits[split]}" for split in SPLITS if splits[split]]
        lines += [f"status {code}: {codes[code]}" for code in FAILURE_CODES if codes[code]]
    return lines
