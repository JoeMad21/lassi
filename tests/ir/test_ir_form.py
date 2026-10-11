"""Tests for the Agent Rule 11 refusal of generic-form and truncated IR (task P5.11; lassi/ir/).

Bible: Agent Rule 11 (IR is committed in custom assembly format only, never
generic form; corpus IR is never truncated), Readability Standards (MLIR:
custom assembly format only), Repository Layout (ir/), Review Of v0 (the
2 MB whale truncation; the per-op generic-form test). Plan: plans/p5-ir.md,
task P5.11 (the refusal lives in lassi/ir/, needs no pinned tool, and P5.6
reuses it; ops with no custom assembly format at the pin pass by a
per-level list that P5.6 fills), and plans/spikes/p5-v0-corpus.md, Results
3 and 6 (a first-line test finds only a whole module printed generically,
not one op printed generically inside a custom-form module).

The contract these tests fix (lassi.ir.form):

- check_ir(text, level=<level>) returns when its scan of `text` finds no
  cut and no op in generic form off the level's list, and raises an
  IRFormError (a ValueError) otherwise; its known cases (a text with no
  module op, or two top-level modules, passes) are pinned below. It runs
  no process and reads no toolchain.
- GenericFormError(IRFormError): an op printed in MLIR's quoted generic
  form, `"dialect.op"(...)` at the place an op starts (at the start of a
  statement, after any result list such as `%0 =`, `%0:2 =`, or `%a, %b =`),
  with or without results, properties `<{...}>`, attributes `{...}`,
  regions `({...})`, or successors `[^bb]`, unless the op is on the level's
  list NO_CUSTOM_FORM[level]. The error names the first such op in text
  order: its `op` attribute is the op name and its `line` the 1-based line
  where the quoted name stands, and its message names the op, the line
  (`line <n>`), and the level. An allowlisted op's regions are still
  checked.
- TruncatedIRError(IRFormError): a text the scan finds cut: an
  opening brace, parenthesis, or bracket never closed, a closing one with
  no opener (a cut at the front), a string never closed, or an empty text.
  Strings (with their escapes, `\\"` and `\\\\` among them) and `//`
  comments are skipped when balancing, so a brace, a quote, or generic-
  looking text inside them changes nothing, and the arrow `->` in types and
  affine maps is no bracket. Its `line` attribute is the 1-based line of
  the truncation point (where the text ends, for a cut at the end; the
  stray closer's line, for one at the front) and its message says
  `line <n>`.
- NO_CUSTOM_FORM maps each MLIR level the corpus holds (cpu-mlir and
  ttkernel at least) to a frozenset of op names that have no custom
  assembly format at the pin (some ttkernel ops print generically in
  either mode). It starts empty or nearly so and task P5.6 fills it from
  the pinned dialects. check_ir reads the module attribute at call time,
  so a test can replace it. A level it does not hold is refused with a
  ValueError naming the level.

Every module here is SYNTHETIC, written for these tests: no v0 output, no
benchmark source, and no upstream kernel text. The ttkernel op names
`ttkernel.synthetic_*` are made up; the check is textual and knows no
dialect. No value here is a measurement.
"""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
FORM_SOURCE = REPO / "lassi" / "ir" / "form.py"
OP_NAME = re.compile(r"[A-Za-z_][\w$]*(\.[\w$]+)+")
ALLOWED_OP = "ttkernel.synthetic_no_format"


def form() -> ModuleType:
    """Import and return lassi.ir.form, so each test fails on its own while it is missing."""
    return importlib.import_module("lassi.ir.form")


def lines(*rows: str) -> str:
    """Join rows with LF and end with one LF."""
    return "\n".join(rows) + "\n"


def wrapped(*body: str) -> str:
    """Return a custom-form module whose function body starts at line 3 with `body`, indented four spaces."""
    return lines(
        "module {",
        "  func.func @f(%a: i32, %b: i32, %m: memref<4xi32>, %i: index) -> i32 {",
        *("    " + row for row in body),
        "    return %a : i32",
        "  }",
        "}",
    )


# SYNTHETIC: a custom-form cpu-mlir module whose strings and comments hold braces, brackets, parentheses, quotes,
# `//`, an escaped backslash at a string's end, and generic-looking text; an affine map alias holds `->`; a
# dictionary key is a quoted string followed by `=`.
CPU_MODULE = lines(
    "#map = affine_map<(d0) -> (d0 * 2)>",
    'module @"src/demo/scale.c" {',
    '  func.func @scale(%arg0: memref<16xf32>, %arg1: f32) attributes {lassi.note = "SYNTHETIC: { ( [ // and \\" '
    'in a string", lassi.path = "dir\\\\"} {',
    "    affine.for %i = 0 to 16 {",
    "      %0 = affine.load %arg0[%i] : memref<16xf32>",
    '      %1 = arith.mulf %0, %arg1 : f32  // SYNTHETIC: a comment with } ) ] and "arith.mulf"(',
    "      affine.store %1, %arg0[%i] : memref<16xf32>",
    "    }",
    "    return",
    "  }",
    '  func.func private @ext(i32) -> i32 attributes {"lassi.quoted" = 1 : i32}',
    '  // SYNTHETIC: a lone quote " and an open brace {',
    '  // "func.return"() : () -> ()',
    "}",
)

# SYNTHETIC: a ttkernel-level module in custom form; its made-up ttkernel ops print their unquoted name before `(`.
TT_MODULE = lines(
    "module {",
    '  func.func @kernel_main() attributes {lassi.kernel = "SYNTHETIC compute kernel"} {',
    "    %c0 = arith.constant 0 : index",
    "    %c1 = arith.constant 1 : index",
    "    %c4 = arith.constant 4 : index",
    "    %n = arith.constant 1 : i32",
    "    scf.for %t = %c0 to %c4 step %c1 {",
    "      ttkernel.synthetic_wait(%n) : (i32) -> ()",
    "      ttkernel.synthetic_pop(%n) : (i32) -> ()",
    "    }",
    "    return",
    "  }",
    "}",
)

# SYNTHETIC: a custom-form op whose operand list follows a quoted callee, as emitc.call_opaque prints.
QUOTED_OPERAND_MODULE = lines(
    "module {",
    "  func.func @g(%n: i32) -> i32 {",
    '    %0 = emitc.call_opaque "synthetic_helper"(%n) : (i32) -> i32',
    "    return %0 : i32",
    "  }",
    "}",
)

# Each refused text with the op the error names and its line.
GENERIC_CASES = {
    "results": (wrapped('%0 = "arith.addi"(%a, %b) : (i32, i32) -> i32'), "arith.addi", 3),
    "no-results": (wrapped('"memref.store"(%a, %m, %i) : (i32, memref<4xi32>, index) -> ()'), "memref.store", 3),
    "properties": (wrapped('%c = "arith.constant"() <{value = 7 : i32}> : () -> i32'), "arith.constant", 3),
    "attributes": (wrapped('%c = "arith.constant"() {value = 7 : i32} : () -> i32'), "arith.constant", 3),
    "multi-result-count": (
        wrapped('%p:2 = "arith.addui_extended"(%a, %b) : (i32, i32) -> (i32, i1)'),
        "arith.addui_extended",
        3,
    ),
    "multi-result-list": (
        wrapped('%s, %o = "arith.addui_extended"(%a, %b) : (i32, i32) -> (i32, i1)'),
        "arith.addui_extended",
        3,
    ),
    "region": (
        wrapped('%r = "scf.execute_region"() ({', '  "scf.yield"(%a) : (i32) -> ()', "}) : () -> i32"),
        "scf.execute_region",
        3,
    ),
    "inside-a-custom-region": (
        wrapped("scf.for %k = %i to %i step %i {", '  %0 = "arith.addi"(%a, %b) : (i32, i32) -> i32', "}"),
        "arith.addi",
        4,
    ),
    "first-of-two": (
        wrapped(
            "%0 = arith.addi %a, %b : i32",
            '%1 = "arith.muli"(%0, %b) : (i32, i32) -> i32',
            '%2 = "arith.subi"(%1, %b) : (i32, i32) -> i32',
        ),
        "arith.muli",
        4,
    ),
    "successors": (
        lines(
            "module {",
            "  func.func @f(%a: i32) -> i32 {",
            '    "cf.br"(%a)[^bb1] : (i32) -> ()',
            "  ^bb1(%x: i32):",
            "    return %x : i32",
            "  }",
            "}",
        ),
        "cf.br",
        3,
    ),
    "whole-module": (
        lines(
            '"builtin.module"() ({',
            '  "func.func"() <{function_type = () -> (), sym_name = "f"}> ({',
            '    "func.return"() : () -> ()',
            "  }) : () -> ()",
            "}) : () -> ()",
        ),
        "builtin.module",
        1,
    ),
}

# Each truncated text with the line of its truncation point (None: only the error type is fixed).
TRUNCATED_CASES = {
    "cut-mid-op": ("module {\n  func.func @f(%a: i32) -> i32 {\n    %0 = arith.addi %a,", 3),
    "cut-at-a-line-end": ("module {\n  func.func @f() {\n    return\n", None),
    "cut-inside-a-string": ('module {\n  func.func @f() attributes {lassi.note = "SYNTHETIC cut inside a str', 2),
    "cut-inside-parentheses": ("module {\n  func.func private @ext(i32,", 2),
    "cut-at-the-front": (lines("module {", "  func.func @f() {", "    return", "  }", "}", "}"), 6),
}


@pytest.fixture
def two_level_list(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Replace NO_CUSTOM_FORM with a SYNTHETIC list: one made-up op allowed at ttkernel, none at cpu-mlir."""
    module = form()
    monkeypatch.setattr(module, "NO_CUSTOM_FORM", {"cpu-mlir": frozenset(), "ttkernel": frozenset({ALLOWED_OP})})
    return module


# ---------------------------------------------------------------------------
# Custom form passes


@pytest.mark.parametrize(
    ("text", "level"),
    [(CPU_MODULE, "cpu-mlir"), (TT_MODULE, "ttkernel"), (QUOTED_OPERAND_MODULE, "ttkernel")],
    ids=["cpu-mlir-strings-and-comments", "ttkernel-custom-ops", "quoted-operand-after-the-op-name"],
)
def test_a_whole_custom_form_module_passes(text: str, level: str) -> None:
    form().check_ir(text, level=level)


def test_the_error_classes_are_value_errors_under_one_base() -> None:
    module = form()
    assert issubclass(module.IRFormError, ValueError)
    assert issubclass(module.GenericFormError, module.IRFormError)
    assert issubclass(module.TruncatedIRError, module.IRFormError)
    assert not issubclass(module.GenericFormError, module.TruncatedIRError)


# ---------------------------------------------------------------------------
# Generic form is refused


@pytest.mark.parametrize(("text", "op", "line"), list(GENERIC_CASES.values()), ids=list(GENERIC_CASES))
def test_an_op_in_the_quoted_generic_form_is_refused_naming_the_op_and_line(text: str, op: str, line: int) -> None:
    module = form()
    with pytest.raises(module.GenericFormError) as refused:
        module.check_ir(text, level="cpu-mlir")
    assert (refused.value.op, refused.value.line) == (op, line)
    message = str(refused.value)
    assert op in message and f"line {line}" in message and "cpu-mlir" in message, message


def test_an_op_on_the_levels_list_passes_and_the_list_is_per_level(two_level_list: ModuleType) -> None:
    text = wrapped(f'"{ALLOWED_OP}"(%a) : (i32) -> ()')
    two_level_list.check_ir(text, level="ttkernel")
    with pytest.raises(two_level_list.GenericFormError) as refused:
        two_level_list.check_ir(text, level="cpu-mlir")
    assert refused.value.op == ALLOWED_OP and "cpu-mlir" in str(refused.value)


def test_an_allowed_ops_region_is_still_checked(two_level_list: ModuleType) -> None:
    text = wrapped(f'"{ALLOWED_OP}"() ({{', '  %0 = "arith.addi"(%a, %b) : (i32, i32) -> i32', "}) : () -> ()")
    with pytest.raises(two_level_list.GenericFormError) as refused:
        two_level_list.check_ir(text, level="ttkernel")
    assert (refused.value.op, refused.value.line) == ("arith.addi", 4)


def test_the_list_maps_each_corpus_level_to_a_frozenset_of_op_names() -> None:
    table = form().NO_CUSTOM_FORM
    assert {"cpu-mlir", "ttkernel"} <= set(table)
    for level, ops in table.items():
        assert isinstance(ops, frozenset), level
        assert all(isinstance(op, str) and OP_NAME.fullmatch(op) for op in ops), (level, sorted(ops))


def test_a_level_the_list_does_not_hold_is_refused_naming_it() -> None:
    with pytest.raises(ValueError, match="no-such-level"):
        form().check_ir(TT_MODULE, level="no-such-level")


# ---------------------------------------------------------------------------
# Truncated IR is refused


@pytest.mark.parametrize(("text", "line"), list(TRUNCATED_CASES.values()), ids=list(TRUNCATED_CASES))
def test_truncated_ir_is_refused_naming_the_truncation_point(text: str, line: int | None) -> None:
    module = form()
    with pytest.raises(module.TruncatedIRError) as refused:
        module.check_ir(text, level="cpu-mlir")
    if line is not None:
        assert refused.value.line == line
        assert f"line {line}" in str(refused.value), str(refused.value)


def test_an_empty_text_is_refused() -> None:
    module = form()
    with pytest.raises(module.IRFormError):
        module.check_ir("", level="cpu-mlir")


@pytest.mark.parametrize(
    "text",
    [
        lines("func.func @f() {", "  return", "}"),
        lines("module {", "}", "module {", "}"),
    ],
    ids=["no-module-op", "two-top-level-modules"],
)
def test_known_case_a_text_with_no_module_op_or_two_top_level_modules_passes(text: str) -> None:
    # SYNTHETIC. A known case, not a goal (the module docstring): MLIR reads top-level ops as an implicit module,
    # and the textual scan takes such a text; task P5.6's pinned verifiers decide what a level accepts.
    form().check_ir(text, level="cpu-mlir")


# ---------------------------------------------------------------------------
# No pinned tool


def test_the_check_runs_no_process(monkeypatch: pytest.MonkeyPatch) -> None:
    module = form()
    monkeypatch.delenv("LASSI_TOOLCHAINS", raising=False)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("check_ir started a process; the refusal needs no pinned tool")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    module.check_ir(CPU_MODULE, level="cpu-mlir")
    with pytest.raises(module.GenericFormError):
        module.check_ir(GENERIC_CASES["results"][0], level="cpu-mlir")
    with pytest.raises(module.TruncatedIRError):
        module.check_ir(TRUNCATED_CASES["cut-mid-op"][0], level="cpu-mlir")


def test_the_refusal_module_imports_no_process_or_toolchain_module() -> None:
    form()
    tree = ast.parse(FORM_SOURCE.read_bytes().decode("utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    banned = ("subprocess", "lassi.toolchains", "lassi.executors")
    hits = [name for name in names if any(name == ban or name.startswith(ban + ".") for ban in banned)]
    assert not hits, hits
