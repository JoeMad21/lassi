"""Deduplication of corpus records by structural hash (task P5.11).

The Corpus Pipeline's Normalize step ends by deduplicating by structural
hash (bible Corpus Pipeline, Normalize). v0 deduplicated its Parquet by
benchmark and pass name instead (tools/build_parquet.py:103-112 at
3ccd280), a key that ignores the pass index, so a pass name repeated in one
run would keep only its first snapshot (Review Of v0, the Keep row's
finding); that key is not moved (plans/spikes/p5-v0-corpus.md, Results 5).

A record's structural hash is the sha256 over the level and the normalized
text of each of its IR fields, the text being what the normalizer returns:
a callable Module -> Module, the IRLevel's normalize (task P5.6), which
prints a module in canonical custom assembly form, so two stylistic forms of
one module hash alike. Nothing else enters the hash: not the suite, the
item, the split, the step, or the pair_id. dedup keeps the first record of
each structural group, in input order, and returns the pair_ids of the
others, so a duplicate is counted, never deleted (the corpus table keeps
every record; lassi.corpus.tables). A record with no IR at all, one that
failed before it was raised, has no structure to compare and is always
kept.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from lassi.core.interfaces import Module
from lassi.corpus.record import CorpusRecord


@dataclass(frozen=True, kw_only=True)
class Deduped:
    """What dedup kept, the first record of each structural group in input order, and the pair_ids it dropped."""

    kept: tuple[CorpusRecord, ...]
    dropped: tuple[str, ...]


def dedup(records: Iterable[CorpusRecord], *, normalize: Callable[[Module], Module]) -> Deduped:
    """Return the records with structural duplicates dropped and counted (see the module docstring).

    `normalize` takes and returns a lassi.core.interfaces Module; a result
    that is not a Module at the level it was given raises ValueError.
    """
    seen: set[str] = set()
    kept: list[CorpusRecord] = []
    dropped: list[str] = []
    for record in records:
        key = _structural_hash(record, normalize)
        if key is not None and key in seen:
            dropped.append(record.pair_id)
            continue
        if key is not None:
            seen.add(key)
        kept.append(record)
    return Deduped(kept=tuple(kept), dropped=tuple(dropped))


def _structural_hash(record: CorpusRecord, normalize: Callable[[Module], Module]) -> str | None:
    """Return the sha256 over each IR field's level and normalized text; None when the record holds no IR."""
    parts: list[list[str] | None] = []
    for level, text in ((record.source_level, record.source_ir), (record.target_level, record.target_ir)):
        if text is None:
            parts.append(None)
            continue
        module = normalize(Module(level=level, text=text))
        if not isinstance(module, Module) or module.level != level or not isinstance(module.text, str):
            raise ValueError(f"the normalizer must return a Module at level {level!r} with its text, got {module!r}")
        parts.append([level, module.text])
    if parts == [None, None]:
        return None
    return hashlib.sha256(json.dumps(parts, ensure_ascii=True).encode("ascii")).hexdigest()
