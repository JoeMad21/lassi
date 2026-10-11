"""Tests for the corpus record: its fields, its IR refusal, and host paths in module names (task P5.11).

Bible: Corpus Pipeline (Record: pair_id, direction, source language,
target, source IR, target IR, source text, program spec, oracle I/O hashes,
gate status, toolchain pins, and provenance hashes), Benchmark Suites (split
rules), Review Of v0 (the -Ddfloat row: record per entry; the 2 MB whale
truncation row: split by function or drop), Agent Rules 1, 5, 10, and 11.
Plan: plans/p5-ir.md, task P5.11 (a corpus record holds the Record step's
fields plus the item's suite and split; its IR fields hold normalized
custom assembly, and a record with generic or truncated IR is refused by
lassi/ir/'s refusal, which needs no pinned tool); plans/spikes/p5-v0-corpus.md,
"Consequences for the plan" (the record holds each entry's build defines,
hashes bytes, and normalizes host paths in module names; no 2 MiB cap) and
Results 3 (every v0 module name holds the absolute host path of its source).

The contract these tests fix (lassi.corpus):

- CorpusRecord is a frozen, keyword-only dataclass with at least these
  fields: pair_id, suite, item, split, direction, source_language, target,
  source_level, source_ir, target_level, target_ir, source_text,
  program_spec, oracle_io, gate_status, build_defines, source_defines,
  toolchain_pins, and provenance. pair_id, suite, item, direction,
  source_language, and target are non-empty strings; split is one of the
  bench registry's SPLITS (train, eval, unassigned); source_ir and
  target_ir are module texts or None (an entry that failed before it had
  one); source_level and target_level are levels lassi.ir.form knows;
  source_text is a str; program_spec a str or None; oracle_io maps an I/O
  file's name to the sha256 of its bytes; gate_status is a GateStatus;
  build_defines maps each define the entry was built with to its value,
  and source_defines those the source's own build sets (its Makefile's),
  each {} when none; toolchain_pins is lassi.core.record.ToolchainPins;
  provenance is the hash chain (tests/corpus/test_corpus_provenance.py). A
  bad value raises ValueError naming the field.
- Each IR field is checked with lassi.ir.form.check_ir at its own level, so
  generic or truncated IR raises lassi.ir.form's GenericFormError or
  TruncatedIRError, and an op on NO_CUSTOM_FORM[level] passes only at that
  level.
- A module name (`module @"<name>"`) that is an absolute host path (POSIX
  `/...` or a drive letter) is refused with a ValueError that says
  "module"; normalize_module_names(text, source_root) rewrites each module
  name under `source_root` (a POSIX path string) to its path relative to
  that root, leaves relative and unnamed modules and every other string as
  they are, and refuses, naming the path, a module name outside the root
  or one that leaves it through '..'.
- There is no size cap: a module over 2 MiB is held whole.

Every text is SYNTHETIC (tests/corpus/corpus_fixtures.py). No value here is
a measurement.
"""

from __future__ import annotations

import dataclasses
from types import ModuleType

import pytest
from corpus_fixtures import (
    SOURCE_ROOT,
    TWO_MIB,
    big_module,
    chain,
    corpus,
    cpu_ir,
    failed_at,
    generic_cpu_ir,
    ir_form,
    lines,
    make_record,
    truncated_cpu_ir,
)

from lassi.core.record import ToolchainPins

RECORD_FIELDS = {
    "pair_id",
    "suite",
    "item",
    "split",
    "direction",
    "source_language",
    "target",
    "source_level",
    "source_ir",
    "target_level",
    "target_ir",
    "source_text",
    "program_spec",
    "oracle_io",
    "gate_status",
    "build_defines",
    "source_defines",
    "toolchain_pins",
    "provenance",
}
ALLOWED_OP = "ttkernel.synthetic_no_format"


def test_the_record_is_frozen_and_holds_the_record_steps_fields_with_suite_item_and_split() -> None:
    module = corpus()
    assert dataclasses.is_dataclass(module.CorpusRecord)
    params = module.CorpusRecord.__dataclass_params__
    assert params.frozen
    names = {item.name for item in dataclasses.fields(module.CorpusRecord)}
    assert RECORD_FIELDS <= names, sorted(RECORD_FIELDS - names)
    assert all(item.kw_only for item in dataclasses.fields(module.CorpusRecord))
    record = make_record()
    assert (record.suite, record.item, record.split) == ("demo-suite", "alpha", "train")
    assert isinstance(record.toolchain_pins, ToolchainPins)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.split = "eval"  # type: ignore[misc]


@pytest.mark.parametrize("split", ["train", "eval", "unassigned"])
def test_every_split_builds(split: str) -> None:
    assert make_record(split=split).split == split


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"split": "test"}, "split"),
        ({"pair_id": ""}, "pair_id"),
        ({"suite": ""}, "suite"),
        ({"item": ""}, "item"),
        ({"source_level": "no-such-level"}, "source_level"),
        ({"target_level": "no-such-level"}, "target_level"),
        ({"oracle_io": {"c.lassiio": "not-a-sha256"}}, "oracle_io"),
        ({"gate_status": "passed"}, "gate_status"),
        ({"build_defines": {"dfloat": 1}}, "build_defines"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_a_bad_field_value_is_refused_naming_the_field(changes: dict[str, object], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        make_record(**changes)


def test_an_entry_that_failed_before_it_had_ir_holds_none() -> None:
    record = make_record(source_ir=None, target_ir=None, gate_status=failed_at("raise"), provenance=chain())
    assert (record.source_ir, record.target_ir) == (None, None)


# ---------------------------------------------------------------------------
# Generic and truncated IR are refused through lassi/ir/


@pytest.mark.parametrize("field", ["source_ir", "target_ir"])
def test_a_record_with_generic_ir_is_refused_by_the_ir_refusal(field: str) -> None:
    form = ir_form()
    level = {"source_ir": "source_level", "target_ir": "target_level"}[field]
    with pytest.raises(form.GenericFormError) as refused:
        make_record(**{field: generic_cpu_ir(), level: "cpu-mlir"})
    assert refused.value.op == "arith.addi"


@pytest.mark.parametrize("field", ["source_ir", "target_ir"])
def test_a_record_with_truncated_ir_is_refused_by_the_ir_refusal(field: str) -> None:
    form = ir_form()
    level = {"source_ir": "source_level", "target_ir": "target_level"}[field]
    with pytest.raises(form.TruncatedIRError):
        make_record(**{field: truncated_cpu_ir(), level: "cpu-mlir"})


@pytest.fixture
def two_level_list(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Replace lassi.ir.form's NO_CUSTOM_FORM: one made-up op allowed at ttkernel, none at cpu-mlir."""
    form = ir_form()
    monkeypatch.setattr(form, "NO_CUSTOM_FORM", {"cpu-mlir": frozenset(), "ttkernel": frozenset({ALLOWED_OP})})
    return form


def allowed_op_module() -> str:
    """Return a SYNTHETIC module holding the made-up op in the quoted generic form."""
    return lines("module {", "  func.func @k() {", f'    "{ALLOWED_OP}"() : () -> ()', "    return", "  }", "}")


def test_an_op_with_no_custom_form_passes_only_at_its_own_level(two_level_list: ModuleType) -> None:
    assert make_record(target_ir=allowed_op_module(), target_level="ttkernel").target_ir == allowed_op_module()
    with pytest.raises(two_level_list.GenericFormError):
        make_record(source_ir=allowed_op_module(), source_level="cpu-mlir")


# ---------------------------------------------------------------------------
# Host paths in module names


@pytest.mark.parametrize(
    "name",
    [f"{SOURCE_ROOT}/src/demo-cuda/main.cu", "C:/SYNTHETIC/bench/src/demo/main.cpp"],
    ids=["posix", "drive-letter"],
)
def test_a_module_named_by_an_absolute_host_path_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="module"):
        make_record(source_ir=cpu_ir(module=name))


def test_module_names_under_the_source_root_become_relative_paths() -> None:
    text = lines(
        f'module @"{SOURCE_ROOT}/src/demo-cuda/main.cu" attributes {{lassi.note = "{SOURCE_ROOT}/kept.txt"}} {{',
        f'  module @"{SOURCE_ROOT}/src/demo-cuda/kernel.cu" {{',
        "  }",
        '  module @"already/relative.cu" {',
        "  }",
        "  module {",
        "  }",
        "}",
    )
    want = lines(
        f'module @"src/demo-cuda/main.cu" attributes {{lassi.note = "{SOURCE_ROOT}/kept.txt"}} {{',
        '  module @"src/demo-cuda/kernel.cu" {',
        "  }",
        '  module @"already/relative.cu" {',
        "  }",
        "  module {",
        "  }",
        "}",
    )
    module = corpus()
    assert module.normalize_module_names(text, SOURCE_ROOT) == want
    assert module.normalize_module_names(text, SOURCE_ROOT + "/") == want
    assert module.normalize_module_names(want, SOURCE_ROOT) == want


def test_a_normalized_module_name_makes_a_valid_record() -> None:
    text = corpus().normalize_module_names(cpu_ir(module=f"{SOURCE_ROOT}/src/demo/scale.c"), SOURCE_ROOT)
    assert make_record(source_ir=text).source_ir == cpu_ir(module="src/demo/scale.c")


@pytest.mark.parametrize(
    "name",
    ["/elsewhere/SYNTHETIC/src/demo/main.cu", f"{SOURCE_ROOT}/../outside/main.cu"],
    ids=["outside-the-root", "dot-dot"],
)
def test_a_module_name_outside_the_source_root_is_refused_naming_it(name: str) -> None:
    with pytest.raises(ValueError) as refused:
        corpus().normalize_module_names(cpu_ir(module=name), SOURCE_ROOT)
    assert name in str(refused.value), str(refused.value)


# ---------------------------------------------------------------------------
# Build defines and size


def test_the_record_holds_the_defines_it_was_built_with_and_the_source_builds_own() -> None:
    record = make_record(
        build_defines={"dfloat": "float", "dlong": "int"}, source_defines={"dfloat": "double", "dlong": "int"}
    )
    assert dict(record.build_defines) == {"dfloat": "float", "dlong": "int"}
    assert dict(record.source_defines) == {"dfloat": "double", "dlong": "int"}


def test_a_module_over_two_mib_is_held_whole() -> None:
    text = big_module(TWO_MIB)
    assert len(text.encode("utf-8")) > TWO_MIB
    record = make_record(source_ir=text)
    assert record.source_ir == text
