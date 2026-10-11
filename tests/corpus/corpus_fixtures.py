"""SYNTHETIC records, IR texts, and helpers shared by the lassi/corpus tests (task P5.11).

Nothing here imports lassi.corpus or lassi.ir at import time: corpus() and
ir_form() import them when a test calls them, so each test fails on its own
while they are missing.

The records follow the contract tests/corpus/test_corpus_record.py states:
a CorpusRecord with the Corpus Pipeline's Record fields (bible Corpus
Pipeline, Record) plus the item's suite, item, and split, the IR level of
each IR field, the build defines and the source build's own defines, and
the provenance hash chain. Every text is SYNTHETIC, written for these
tests: no v0 output, no benchmark source, no eval item text, and no upstream
kernel text; the ttkernel op names are made up. Toolchain pin values are
SYNTHETIC labels, not pins. No value here is a measurement.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from lassi.core.record import ToolchainPins

SUITE = "demo-suite"
OTHER_SUITE = "other-suite"
ITEM = "alpha"
DIRECTION = "cpp-tt"
# A made-up source root, standing for where a host keeps a suite's fetched sources.
SOURCE_ROOT = "/scratch/SYNTHETIC/bench"
SOURCE_TEXT = (
    "/* SYNTHETIC source written for the lassi/corpus tests */\n"
    "void scale(float *a, float s) { for (int i = 0; i < 16; ++i) a[i] *= s; }\n"
)
PINS = ToolchainPins(polygeist="SYNTHETIC-polygeist-pin", tt_mlir="SYNTHETIC-tt-mlir-pin")
# v0's whale cap (tools/build_parquet.py:121-127 at 3ccd280), which the corpus does not keep (Agent Rule 11).
TWO_MIB = 2 * 1024 * 1024


def corpus() -> ModuleType:
    """Import and return lassi.corpus."""
    return importlib.import_module("lassi.corpus")


def ir_form() -> ModuleType:
    """Import and return lassi.ir.form."""
    return importlib.import_module("lassi.ir.form")


def sha(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def lines(*rows: str) -> str:
    """Join rows with LF and end with one LF."""
    return "\n".join(rows) + "\n"


def cpu_ir(name: str = "scale", module: str = "src/demo/scale.c") -> str:
    """Return a SYNTHETIC custom-form cpu-mlir module whose function is @<name>, in a module named `module`."""
    return lines(
        f'module @"{module}" {{',
        f"  func.func @{name}(%arg0: memref<16xf32>, %arg1: f32) {{",
        "    affine.for %i = 0 to 16 {",
        "      %0 = affine.load %arg0[%i] : memref<16xf32>",
        "      %1 = arith.mulf %0, %arg1 : f32",
        "      affine.store %1, %arg0[%i] : memref<16xf32>",
        "    }",
        "    return",
        "  }",
        "}",
    )


def tt_ir(name: str = "scale") -> str:
    """Return a SYNTHETIC custom-form ttkernel-level module whose kernel is @<name>_kernel."""
    return lines(
        "module {",
        f"  func.func @{name}_kernel() {{",
        "    %n = arith.constant 1 : i32",
        "    ttkernel.synthetic_wait(%n) : (i32) -> ()",
        "    ttkernel.synthetic_pop(%n) : (i32) -> ()",
        "    return",
        "  }",
        "}",
    )


def generic_cpu_ir() -> str:
    """Return a SYNTHETIC cpu-mlir module with one op printed in the quoted generic form (arith.addi, line 3)."""
    return lines(
        'module @"src/demo/add.c" {',
        "  func.func @add(%a: i32, %b: i32) -> i32 {",
        '    %0 = "arith.addi"(%a, %b) : (i32, i32) -> i32',
        "    return %0 : i32",
        "  }",
        "}",
    )


def truncated_cpu_ir() -> str:
    """Return a SYNTHETIC cpu-mlir module cut in the middle of an op, as a size cap cuts one."""
    return 'module @"src/demo/add.c" {\n  func.func @add(%a: i32, %b: i32) -> i32 {\n    %0 = arith.addi %a,'


def big_module(at_least: int) -> str:
    """Return a SYNTHETIC custom-form module of many small functions, more than `at_least` bytes long."""
    rows = ['module @"src/demo/big.c" {']
    size, index = 0, 0
    while size <= at_least:
        row = f"  func.func @f{index}(%a: i32) -> i32 {{\n    %0 = arith.addi %a, %a : i32\n    return %0 : i32\n  }}"
        rows.append(row)
        size += len(row) + 1
        index += 1
    rows.append("}")
    return "\n".join(rows) + "\n"


def passed() -> Any:
    """Return the GateStatus of an entry that passed every gate step."""
    module = corpus()
    return module.GateStatus(passed=module.GATE_STEPS[-1], code=None)


def failed_at(step: str) -> Any:
    """Return the GateStatus of an entry that failed at gate step `step`, having passed every step before it."""
    module = corpus()
    index = module.GATE_STEPS.index(step)
    return module.GateStatus(passed=module.GATE_STEPS[index - 1] if index else None, code=module.failure_code(step))


def chain(*outputs: tuple[str, str]) -> list[Any]:
    """Return the provenance chain over the source text and each (step, text) output, hashed as UTF-8 bytes."""
    steps = [("source", SOURCE_TEXT.encode("utf-8")), *((step, text.encode("utf-8")) for step, text in outputs)]
    return list(corpus().hash_chain(steps))


def make_record(unit: str = "scale", **changes: Any) -> Any:
    """Return a valid SYNTHETIC train record of `unit` (a C++ -> TT pair that passed the gate), with `changes`.

    The pair_id is <suite>/<item>/<direction>/<unit> unless `changes` gives
    one. The source and target IR name `unit`, so records of different units
    differ structurally.
    """
    source_ir, target_ir = cpu_ir(unit), tt_ir(unit)
    fields: dict[str, Any] = {
        "suite": SUITE,
        "item": ITEM,
        "split": "train",
        "direction": DIRECTION,
        "source_language": "cpp",
        "target": "tenstorrent",
        "source_level": "cpu-mlir",
        "source_ir": source_ir,
        "target_level": "ttkernel",
        "target_ir": target_ir,
        "source_text": SOURCE_TEXT,
        "program_spec": None,
        "oracle_io": {"c.lassiio": sha(b"SYNTHETIC output c of " + unit.encode("ascii"))},
        "gate_status": passed(),
        "build_defines": {},
        "source_defines": {},
        "toolchain_pins": PINS,
        "provenance": chain(("raise", "// SYNTHETIC raised form\n" + source_ir), ("normalize", source_ir)),
    }
    fields.update(changes)
    fields.setdefault("pair_id", f"{fields['suite']}/{fields['item']}/{fields['direction']}/{unit}")
    return corpus().CorpusRecord(**fields)


def by_pair_id(records: Any) -> list[Any]:
    """Return records sorted by pair_id."""
    return sorted(records, key=lambda record: record.pair_id)


def files_under(root: Path) -> set[str]:
    """Return every file under `root` as a POSIX path relative to it."""
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def snapshot(root: Path) -> dict[str, bytes]:
    """Return every file under `root` with its bytes."""
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
