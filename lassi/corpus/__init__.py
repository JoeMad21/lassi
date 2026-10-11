"""The corpus: records, gate status, provenance, and Parquet tables (bible Corpus Pipeline; task P5.11).

lassi/corpus/ absorbs v0, the mlir-corpus-pipeline repository at commit
3ccd280 (bible Repository Layout, corpus/; Review Of v0). Task P5.11 moves
the pieces the Review Of v0 table keeps, by the migration map of
plans/spikes/p5-v0-corpus.md (Results 5), each written anew with its tests;
nothing here imports v0 or runs its scripts:

- the exit-code taxonomy: lassi.corpus.status (GATE_STEPS, FAILURE_CODES,
  failure_code, GateStatus);
- the manifest with parent-child hashes: lassi.corpus.provenance (HashLink,
  hash_chain, check_chain), over bytes;
- Policy B, as records that are never deleted, and the Hive Parquet:
  lassi.corpus.tables (write_corpus, write_pair_set, write_verification and
  their readers);
- dedup by structural hash, in place of v0's key: lassi.corpus.dedup.

lassi.corpus.record holds the corpus record (CorpusRecord) and the
module-name normalizer, and lassi.corpus.inputs the corpus's only way to a
bench item (harvest_item, verification_item). v0's LLVM, never pinned, is
replaced by P5's own pins, and its drivers, stream splitter, setup scripts, cir-opt
round trip, 2 MiB cap, and fine-tuning code are not migrated (the spike's
"Not migrated" table). The round-trip driver is task P5.12's.
"""

from lassi.corpus.dedup import Deduped, dedup
from lassi.corpus.inputs import harvest_item, verification_item
from lassi.corpus.provenance import HashLink, check_chain, hash_chain
from lassi.corpus.record import SOURCE_STEP, CorpusRecord, normalize_module_names
from lassi.corpus.status import FAILURE_CODES, GATE_STEPS, STOP_CODES, GateStatus, failure_code
from lassi.corpus.tables import (
    BUILD_DEFINES,
    TABLES,
    VerificationTable,
    read_corpus,
    read_pair_set,
    read_verification,
    summarize,
    write_corpus,
    write_pair_set,
    write_verification,
)

__all__ = [
    "BUILD_DEFINES",
    "FAILURE_CODES",
    "GATE_STEPS",
    "SOURCE_STEP",
    "STOP_CODES",
    "TABLES",
    "CorpusRecord",
    "Deduped",
    "GateStatus",
    "HashLink",
    "VerificationTable",
    "check_chain",
    "dedup",
    "failure_code",
    "harvest_item",
    "hash_chain",
    "normalize_module_names",
    "read_corpus",
    "read_pair_set",
    "read_verification",
    "summarize",
    "verification_item",
    "write_corpus",
    "write_pair_set",
    "write_verification",
]
