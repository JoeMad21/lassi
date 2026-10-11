"""The corpus record: one corpus entry as the Corpus Pipeline's Record step writes it (task P5.11).

A CorpusRecord holds the Record step's fields (bible Corpus Pipeline,
Record: pair_id, direction, source language, target, source IR, target IR,
source text, program spec, oracle I/O hashes, gate status, toolchain pins,
and provenance hashes) plus the bench item's suite, item, and split
(Benchmark Suites, split rules), the IR level of each IR field, and the
defines of the entry's build beside those of its source's own build (Review
Of v0, the -Ddfloat row: record per entry). lassi.corpus.tables writes
records as Hive Parquet and reads them back through this class, so every
record read is checked again.

The IR fields hold normalized custom assembly. Each is checked with
lassi.ir.form's check_ir at its own level, so IR in generic form or
truncated raises that module's GenericFormError or TruncatedIRError (Agent
Rule 11), and a level check_ir does not know is refused: the record holds
no llvm-ir until a later task adds that level to the refusal. There is no
size cap; a module of any size is held whole. A module name that is an
absolute host path is refused: every v0 module was named by its source's
absolute path on the build host (plans/spikes/p5-v0-corpus.md, Results 3),
and normalize_module_names rewrites such names relative to the source root
first. Paths in loc() locations are task P5.6's, which moves locations to a
sidecar.

source_text is the source as decoded text, for a reader of the record. The
provenance chain must start with the link of step "source", which by
contract holds the sha256 of the source file's bytes, hashed with
lassi.corpus.provenance hash_chain, which takes bytes only. The two can
differ for a source that is not UTF-8, and the record compares neither
with the other: it checks the chain's links, not what was hashed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from lassi.bench.registry import SPLITS
from lassi.core.record import TOOLCHAIN_PIN_NAMES, ToolchainPins
from lassi.corpus.provenance import HashLink, check_chain
from lassi.corpus.status import GateStatus
from lassi.ir import form

# The first step of every record's provenance chain: the source file's bytes.
SOURCE_STEP = "source"
# The bench registry's item-name rule (lassi.bench.registry), for suite and item, the corpus tables' partition keys.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DEFINE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# An absolute host path: POSIX or UNC (a leading '/' or '\'), or a Windows drive letter.
_HOST_PATH = re.compile(r"[/\\]|[A-Za-z]:")


@dataclass(frozen=True, kw_only=True)
class CorpusRecord:
    """One corpus entry: a pair's IR, its source, its gate status, and where it came from.

    Every field is required, and a bad value raises ValueError naming the
    field (IR in generic form or truncated raises lassi.ir.form's
    GenericFormError or TruncatedIRError, both ValueErrors):

    - pair_id, direction, source_language, target: non-empty strings with no
      control character. suite and item: bench registry names
      ([A-Za-z0-9][A-Za-z0-9._+-]*). split: one of the registry's SPLITS
      (train, eval, unassigned).
    - source_level, target_level: levels lassi.ir.form knows (form.levels()).
      source_ir, target_ir: a module text, or None for an entry that failed
      before it had one; each passes check_ir at its level, and no module
      name in it is an absolute host path (normalize_module_names).
    - source_text: the source as decoded text. program_spec: text or None.
    - oracle_io: each I/O file's name to the sha256 of its bytes.
    - gate_status: a GateStatus. toolchain_pins: a ToolchainPins.
    - build_defines: each define the entry was built with (a C identifier)
      to its value; source_defines: those its source's own build sets (its
      Makefile's); each {} when none. A pair set leaves out an entry whose
      two differ (lassi.corpus.tables write_pair_set).
    - provenance: HashLinks forming one chain (check_chain) whose first step
      is SOURCE_STEP.

    Every string, mapping keys and values, chain steps, and pins included,
    must encode as UTF-8, as Parquet holds it. The record keeps a copy of
    each mapping as a dict and the chain as a tuple.
    """

    pair_id: str
    suite: str
    item: str
    split: str
    direction: str
    source_language: str
    target: str
    source_level: str
    source_ir: str | None
    target_level: str
    target_ir: str | None
    source_text: str
    program_spec: str | None
    oracle_io: Mapping[str, str]
    gate_status: GateStatus
    build_defines: Mapping[str, str]
    source_defines: Mapping[str, str]
    toolchain_pins: ToolchainPins
    provenance: Sequence[HashLink]

    def __post_init__(self) -> None:
        """Check every field, then keep the mappings as dicts and the provenance chain as a tuple."""
        _check_names(self)
        _check_levels_and_texts(self)
        for name in ("source_ir", "target_ir"):
            _check_ir(self, name)
        _check_maps(self)
        _check_parts(self)
        for name in ("oracle_io", "build_defines", "source_defines"):
            object.__setattr__(self, name, dict(getattr(self, name)))
        object.__setattr__(self, "provenance", tuple(self.provenance))


def _fail(name: str, value: object, rule: str) -> ValueError:
    """Return the ValueError for one bad field value; the value is shown short."""
    shown = repr(value)
    return ValueError(f"CorpusRecord.{name} {rule}, got {shown if len(shown) <= 200 else shown[:200] + '...'}")


def _check_utf8(name: str, value: str) -> None:
    """Refuse a string that does not encode as UTF-8 (one holding a lone surrogate)."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise _fail(name, value, "must encode as UTF-8 (it holds a lone surrogate)") from None


def _check_names(record: CorpusRecord) -> None:
    """Check pair_id, suite, item, split, direction, source_language, and target."""
    for name in ("pair_id", "direction", "source_language", "target", "suite", "item"):
        value = getattr(record, name)
        if not isinstance(value, str) or not value or _CONTROL.search(value):
            raise _fail(name, value, "must be a non-empty string with no control character")
        _check_utf8(name, value)
        if name in ("suite", "item") and not _NAME.fullmatch(value):
            raise _fail(name, value, f"must be a bench registry name ({_NAME.pattern})")
    if record.split not in SPLITS:
        raise _fail("split", record.split, f"must be one of {', '.join(SPLITS)}")


def _check_levels_and_texts(record: CorpusRecord) -> None:
    """Check the two levels against lassi.ir.form, and the types of the text fields."""
    known = form.levels()
    for name in ("source_level", "target_level"):
        if getattr(record, name) not in known:
            raise _fail(name, getattr(record, name), f"must be a level the IR refusal knows ({', '.join(known)})")
    for name in ("source_ir", "target_ir", "source_text", "program_spec"):
        value = getattr(record, name)
        if value is None and name != "source_text":
            continue
        if not isinstance(value, str):
            raise _fail(name, value, "must be a str" if name == "source_text" else "must be a str or None")
        _check_utf8(name, value)


def _check_ir(record: CorpusRecord, name: str) -> None:
    """Run the Agent Rule 11 refusal on one IR field, then refuse a module named by an absolute host path."""
    text = getattr(record, name)
    if text is None:
        return
    level = record.source_level if name == "source_ir" else record.target_level
    form.check_ir(text, level=level, where=f"CorpusRecord.{name}")
    for found in form.module_names(text):
        if _HOST_PATH.match(found.name):
            raise ValueError(
                f'CorpusRecord.{name}: module name "{found.name}" at line {found.line} is an absolute host path; '
                "rewrite it relative to its source root with normalize_module_names first"
            )


def _check_maps(record: CorpusRecord) -> None:
    """Check oracle_io (names to sha256) and the two define maps (C identifiers to values)."""
    for name in ("oracle_io", "build_defines", "source_defines"):
        value = getattr(record, name)
        if not isinstance(value, Mapping):
            raise _fail(name, value, "must be a mapping")
        for key, item in value.items():
            if name == "oracle_io":
                good = isinstance(key, str) and bool(key) and isinstance(item, str) and bool(_SHA256.fullmatch(item))
                rule = "must map each I/O file name to the 64-hex sha256 of its bytes"
            else:
                good = isinstance(key, str) and bool(_DEFINE.fullmatch(key)) and isinstance(item, str)
                rule = "must map each define's name (a C identifier) to its value, a str"
            if not good:
                raise _fail(name, dict(value), rule)
            _check_utf8(name, key)
            _check_utf8(name, item)


def _check_parts(record: CorpusRecord) -> None:
    """Check the gate status, the toolchain pins, and the provenance chain."""
    if not isinstance(record.gate_status, GateStatus):
        raise _fail("gate_status", record.gate_status, "must be a GateStatus")
    if not isinstance(record.toolchain_pins, ToolchainPins):
        raise _fail("toolchain_pins", record.toolchain_pins, "must be a ToolchainPins")
    links: Any = record.provenance
    if isinstance(links, (str, bytes)) or not isinstance(links, Sequence):
        raise _fail("provenance", links, "must be a sequence of HashLinks")
    try:
        check_chain(links)
    except ValueError as error:
        raise ValueError(f"CorpusRecord.provenance: {error}") from None
    if links[0].step != SOURCE_STEP:
        raise _fail("provenance", links[0], f"must start with the {SOURCE_STEP!r} link, the source file's bytes")
    for link in links:
        _check_utf8("provenance", link.step)
    for name in TOOLCHAIN_PIN_NAMES:
        value = getattr(record.toolchain_pins, name)
        if value is not None:
            _check_utf8("toolchain_pins", value)


def normalize_module_names(text: str, source_root: str) -> str:
    """Return `text` with each module name under `source_root` rewritten as its path relative to that root.

    `source_root` is an absolute POSIX path; a trailing '/' changes nothing.
    A module name (lassi.ir.form module_names) that is an absolute host path
    becomes its path relative to the root. A relative name, an unnamed
    module, and every other string, attribute strings holding paths among
    them, stay as they are, so normalizing twice changes nothing. A name
    outside the root, one that leaves it through '..', the root itself, and
    one holding a backslash or a drive letter raise ValueError naming the
    path. The text is read with lassi.ir.form's scan, so a text the scan
    finds cut raises TruncatedIRError.
    """
    if not isinstance(source_root, str) or not source_root.startswith("/"):
        raise ValueError(f"the source root must be an absolute POSIX path, got {source_root!r}")
    root = PurePosixPath(source_root)
    pieces: list[str] = []
    at = 0
    for found in form.module_names(text):
        if _HOST_PATH.match(found.name):
            pieces += [text[at:found.start], _relative(found.name, root)]
            at = found.end
    return "".join([*pieces, text[at:]])


def _relative(name: str, root: PurePosixPath) -> str:
    """Return the POSIX path of `name` relative to `root`; ValueError naming `name` when it is not under it."""
    path = PurePosixPath(name)
    if "\\" in name or not name.startswith("/"):
        raise ValueError(f'module name "{name}" is not a POSIX path under the source root {root}')
    if ".." in path.parts:
        raise ValueError(f'module name "{name}" leaves the source root {root} through ".."')
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError(f'module name "{name}" is outside the source root {root}') from None
    if not relative.parts:
        raise ValueError(f'module name "{name}" is the source root {root} itself, not a file under it')
    return relative.as_posix()
