"""Tests for the Python-subset frontend: Python 3.10 source raised to a structured hub module (task P12.4).

Bible: Language Frontends (the Python scalar and NumPy row; the hub level), Frontend Rules (the published
subset; tags), Component Interfaces (Frontend), IR Strategy (the cpu-mlir row), Harness Contract (the Python
binding; held-out input arguments), Agent Rules 3 and 11; plans/p12-frontends.md, task P12.4 and its planning
decisions "Refusals (P12.4)" and "Harness I/O in raised programs"; plans/spikes/p12-xlang-v0.md, section 8
(what xlang-v0's Python programs need: calls with array arguments, allocation, while loops).

The contract these tests fix:

- Importing lassi.frontends registers the Frontend python-subset (lassi.frontends.python_subset), built as
  factory(). raise_source({path: text}, flags) reads the text with the standard ast module and never executes
  or imports it, and returns Module(level="cpu-mlir", text, tag): MLIR in custom assembly at LLVM 18.0.0git's
  syntax (Polygeist's LLVM, llvm-project 26eb4285), holding only builtin's module, func, arith, math, scf, and
  memref. The tag is lassi.frontends.hub_tag(text): "low" when the text holds an llvm or cf op, else
  "structured"; it reads the dialects and takes no language.
- For each construct below, raising tests/fixtures/frontends/python_subset/<construct>.py gives exactly the
  bytes of <construct>.mlir, written by hand here; a second raise, by the same or a new frontend, or in a
  process with another hash seed, gives the same bytes. Every golden passes lassi.ir.form check_ir at the
  cpu-mlir level (custom form, whole). Their pinned parse and verify is task P12.12's, since no pinned MLIR
  tool runs on Windows.
- lassi.frontends.python_subset's module docstring publishes the subset, the harness I/O form, and every
  refusal code (tests/frontends/test_python_subset_refusals.py tests the refusals).
- Agent Rule 3: lassi/frontends names no project, suite, or model.

The constructs (one fixture pair each): scalar_function (functions over scalars, return, calls between them),
float_arith (+ - * /, unary minus, an int operand promoted), int_arith (+ - * // % & | ^ << >>, and / on
ints), compare_if (comparisons; if, elif, else; and, or, not; a scalar local), conditional_expr (x if c else
y), for_range (range(stop), range(start, stop), range(start, stop, step); augmented assignment), while_loop
(while; an assigned parameter), prange (the binding's parallel-for, with locals private to each iteration),
array_access (element loads and stores with dtype conversion, len(), augmented element assignment),
array_alloc (the binding's allocation form, zero-filled), call_arrays (calls with array arguments, an int
argument to a float parameter, a call inside a loop), math_calls (math.sqrt, exp, log, sin, cos; abs, float,
int), numpy_vectorized (np.asarray and a vectorized expression assigned into an array), harness_io
(read_array, write_array, the imports, and the entry guard of a whole program), annotated_scalars (`x: int = e`,
float and bool, a bool local in a memref<i1> slot), array_alias (`b = a` and an annotated alias), write_shapes
(writes of rank 0 and rank 2, and a write inside a loop), and prange_bounds (prange with a start and a step, and a
range loop reusing its variable's name).

The lowering scheme the goldens follow (P12.4 publishes it; P12.12 checks it with the pinned tools):

- Module: `module {`, then one memref.global per output name, then the binding's externals as `func.func
  private` declarations, each in order of first use, then each function in source order. Two spaces per
  level, one op per line, no blank lines, LF line ends, the closing brace and a line feed last. Module-level
  docstrings, the imports (`import math`, `import sys`, `import numpy as np`, `from typing import
  Annotated`, `from lassi_io import` any of read_array, write_array, prange, never aliased), and the entry
  guard `if __name__ == "__main__": main()` emit nothing; a function docstring emits nothing.
- Types: int i64, float f64, bool i1; a parameter `Annotated[memoryview, "<dtype>"]` is memref<?xT> with T
  f32, f64, i8 (i8 and u8), i32 (i32 and u32), or i64. An element load widens to the Python scalar (extf;
  extsi for i8 and i32; extui for u8 and u32); a store narrows (truncf; trunci), an int stored into a float
  array going through sitofp to f64 first.
- Values: parameters are %arg0, %arg1, ...; every other value, op results and induction variables alike, is
  %0, %1, ... in the order its definition appears in the function's text. Constants are emitted where used,
  never shared: `arith.constant <v> : i64`, a float literal as Python's repr `: f64` (a repr with no "."
  gains ".0" before its exponent, or at its end: 1e-08 is 1.0e-08), `true` or `false`, and loop bounds and
  dims `: index`. A unary minus on a number literal is part of the literal.
- Locals: each scalar name bound in a function, loop variables included, lives in a rank-0 memref.alloca
  slot; reads load it, bindings store it. A parameter the body assigns gets a slot stored from its argument;
  other parameters are read as the argument. Slots open at function entry: assigned parameters in parameter
  order (alloca, then store), then other names in order of first binding. Names bound only inside a prange
  body open there instead, the loop variable first, so each iteration has its own. Array names (an
  allocation, a read, np.asarray of a view, a parameter) are SSA values, bound once.
- Expressions, evaluated left to right as Python does: after both operands, an int operand of a float
  operation is converted with sitofp (left first); / converts both. addi, subi, muli; // floordivsi; a % b as
  a - (a // b) * b (floordivsi, muli, subi, reusing b's value); andi, ori, xori, shli, shrsi; addf, subf,
  mulf, divf; unary minus negf on a float; not as xori with true (operand, then constant). Comparisons: cmpi
  slt, sle, sgt, sge, eq, ne; cmpf olt, ole, ogt, oge, oeq, une. `a and b` and `a or b` short-circuit through
  scf.if -> (i1): and evaluates b in its then region and yields false in its else region; or yields true in
  its then region and evaluates b in its else region. `x if c else y` is scf.if on c yielding each arm.
- Statements: `x = e` stores; `x op= e` loads x, evaluates e, applies, stores; `a[i] = e` evaluates e, then i
  (index_cast to index), narrows, stores; `a[i] op= e` evaluates i, loads, widens, evaluates e, applies,
  narrows, stores at the same index. if, elif, else is scf.if (elif nested in the else region). `for v in
  range(...)` with stop, or start and stop, or start, stop, and a positive int literal step: the start, if
  given, and the stop evaluated, each index_cast in order, then the absent start 0 and the step (1 when
  absent) as index constants, then scf.for, whose body opens by storing the index_cast of the induction
  variable to v's slot. `for v in prange(...)` is the same as scf.parallel, whose body is one
  memref.alloca_scope (private slots first, so each iteration frees its own) followed by an explicit
  scf.yield; a for loop outside the prange body may reuse the name of its loop variable, in a slot of its own.
  `while c:` is scf.while : () -> () with c in its before region and an explicit scf.yield closing its do
  region. `return e` only as a function's last statement: e is evaluated, then each memref.alloc of the
  function is freed with memref.dealloc in allocation order, then the return (a function with no return
  frees them before the return written last). A call is `call` directly in a function body and `func.call`
  inside a region; its arguments are evaluated in order, then an int argument to a float parameter is
  converted with sitofp.
- Builtins: len(a) is index constant 0, memref.dim, index_cast to i64; float() sitofp; int() fptosi; abs()
  math.absf or math.absi; math.sqrt, exp, log, sin, and cos the math ops on f64.
- Allocation: `memoryview(bytearray(<count> * <itemsize>)).cast("<format>")`, the itemsize literal matching
  the format (f, d, b, B, i, I, q): the count evaluated and index_cast, memref.alloc, index constants 0 and 1
  and the element zero, then an scf.for storing it, zero-filled as bytearray is.
- NumPy: `z[:] = <expression>` over ndarrays of one dtype (np.asarray of views), scalars, + - * /, and
  np.sqrt: its scalar parts first, in evaluation order; index constant 0 and memref.dim of z; index constants 0
  and 1; then scf.parallel over z's length, loading each array operand at the induction variable in
  evaluation order and storing the result into z.
- Harness I/O (the planning decision's form, reconciled with P5.7's by P12.12): a program reads its inputs
  and names its outputs through argv (`@inputs/<name>.lassiio`, then `<name>.lassiio`; Harness Contract).
  `v: Annotated[memoryview, "<dtype>"] = read_array(sys.argv[<k>]).data` is `call @lassi_io_read_<dtype>(%k)
  : (i64) -> memref<?xT>` after `arith.constant <k> : i64`. `write_array(sys.argv[<k>], "<name>", "<dtype>",
  (<dims>,), a)` is the constant k; memref.get_global of @lassi_io_name<N> (the name's ASCII bytes and one
  zero byte), memref.cast to memref<?xi8>; the dims evaluated, a memref.alloca of memref<<rank>xi64>, each dim
  stored at an index constant, memref.cast to memref<?xi64>; then `call @lassi_io_write_<dtype>` taking (i64,
  memref<?xi8>, memref<?xi64>, memref<?xT>). A write inside a loop body stands in one memref.alloca_scope, so
  its shape buffer is freed each time. `def main() -> None`, the function the entry guard calls, is raised as
  `func.func @main()`.

Every golden and every module text here is SYNTHETIC test data written for this task. No value in this module
is a measurement.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from python_subset_fixtures import (
    CONSTRUCTS,
    FIXTURES,
    FRONTEND_NAME,
    HUB_DIALECTS,
    HUB_LEVEL,
    LOW,
    REPO,
    STRUCTURED,
    dialect_of,
    frontend_refusal,
    frontends_package,
    golden,
    hub_tag_function,
    op_names,
    python_subset,
    python_subset_module,
    raise_text,
    source,
)

from lassi.core.registry import DEFAULT_REGISTRY
from lassi.ir.form import check_ir

FRONTENDS_DIR = REPO / "lassi" / "frontends"
PROJECT_NAMES = (
    "lassi-repro", "lassi-ee", "lassi-df", "lassi-demo", "hecbench", "xlang", "qwen", "wizardcoder", "a100",
    "mi300x", "gpt-oss",
)

# SYNTHETIC modules that hold an llvm or a cf op, which the tag rule reads as low.
LLVM_MODULE = "module {\n  llvm.func @f() {\n    llvm.return\n  }\n}\n"
CF_MODULE = (
    "module {\n  func.func @f(%arg0: i1) {\n    cf.cond_br %arg0, ^bb1, ^bb2\n  ^bb1:\n    return\n"
    "  ^bb2:\n    return\n  }\n}\n"
)
MIXED_MODULE = (
    "module {\n  llvm.func @g()\n  func.func @f(%arg0: index) {\n    %0 = arith.constant 0 : index\n"
    "    %1 = arith.constant 1 : index\n    scf.for %2 = %0 to %arg0 step %1 {\n      llvm.call @g() : () -> ()\n"
    "    }\n    return\n  }\n}\n"
)

# The script a child process runs to raise every construct; it prints one JSON object of construct -> text.
_CHILD = """
import json, sys
from pathlib import Path
import lassi.frontends
from lassi.core.registry import DEFAULT_REGISTRY
root = Path(sys.argv[1])
names = sys.argv[2:]
frontend = DEFAULT_REGISTRY.get("Frontend", "python-subset").factory()
texts = {}
for name in names:
    text = (root / (name + ".py")).read_bytes().decode("ascii")
    texts[name] = frontend.raise_source({name + ".py": text}, []).text
sys.stdout.write(json.dumps(texts, sort_keys=True))
"""


# ---------------------------------------------------------------------------
# The fixtures themselves (these pass before the task: they check the goldens written here)


def test_every_construct_has_a_source_and_a_golden() -> None:
    sources = {path.stem for path in FIXTURES.glob("*.py")}
    goldens = {path.stem for path in FIXTURES.glob("*.mlir")}
    assert sources == goldens == set(CONSTRUCTS), (sorted(sources ^ set(CONSTRUCTS)), sorted(goldens ^ sources))


@pytest.mark.parametrize("construct", CONSTRUCTS)
def test_every_fixture_source_is_python_3_10(construct: str) -> None:
    ast.parse(source(construct), filename=f"{construct}.py", feature_version=(3, 10))


@pytest.mark.parametrize("construct", CONSTRUCTS)
def test_every_golden_is_custom_form_and_whole(construct: str) -> None:
    text = golden(construct)
    check_ir(text, level=HUB_LEVEL, where=f"{construct}.mlir")
    assert "\r" not in text and "\t" not in text, "LF line ends and blanks only"
    assert text.endswith("}\n") and not text.endswith("\n\n"), "the module closes on the last line"
    assert all(line == line.rstrip() for line in text.split("\n")), "no trailing blanks"
    assert text.startswith("module {\n")


@pytest.mark.parametrize("construct", CONSTRUCTS)
def test_every_golden_holds_only_hub_dialects(construct: str) -> None:
    dialects = {dialect_of(op) for op in op_names(golden(construct))}
    assert dialects <= HUB_DIALECTS, sorted(dialects - HUB_DIALECTS)
    assert {"builtin", "func"} <= dialects


# ---------------------------------------------------------------------------
# Registration and the raise


def test_python_subset_is_a_registered_frontend() -> None:
    frontends_package()
    assert FRONTEND_NAME in DEFAULT_REGISTRY.names("Frontend")
    entry = DEFAULT_REGISTRY.get("Frontend", FRONTEND_NAME)
    assert entry.factory.__module__ == "lassi.frontends.python_subset"
    frontend = entry.factory()
    assert callable(frontend.raise_source)
    assert inspect.getdoc(entry.factory), "the class has a docstring"


@pytest.mark.parametrize("construct", CONSTRUCTS)
def test_each_construct_raises_to_its_golden(construct: str) -> None:
    module = raise_text(source(construct), f"{construct}.py")
    assert module.text == golden(construct)
    assert module.level == HUB_LEVEL
    assert module.tag == STRUCTURED
    assert module.tag == hub_tag_function()(module.text), "the tag is read from the module's dialects"


@pytest.mark.parametrize("construct", CONSTRUCTS)
def test_a_second_raise_gives_the_same_bytes(construct: str) -> None:
    frontend = python_subset()
    text = source(construct)
    first = frontend.raise_source({f"{construct}.py": text}, []).text
    second = frontend.raise_source({f"{construct}.py": text}, []).text
    fresh = python_subset().raise_source({f"{construct}.py": text}, []).text
    assert first.encode("ascii") == second.encode("ascii") == fresh.encode("ascii")


def test_the_raise_does_not_depend_on_the_hash_seed() -> None:
    python_subset()
    outputs = []
    for seed in ("0", "1"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        done = subprocess.run(
            [sys.executable, "-c", _CHILD, str(FIXTURES), *CONSTRUCTS],
            capture_output=True, cwd=REPO, env=environment, timeout=120, check=False,
        )
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
        outputs.append(json.loads(done.stdout.decode("ascii")))
    assert outputs[0] == outputs[1]
    assert outputs[0] == {construct: golden(construct) for construct in CONSTRUCTS}


# ---------------------------------------------------------------------------
# The tag


def test_hub_tag_reads_the_dialects_never_a_language() -> None:
    hub_tag = hub_tag_function()
    parameters = list(inspect.signature(hub_tag).parameters)
    assert len(parameters) == 1, f"hub_tag takes the module text alone, not a language: {parameters}"
    for construct in CONSTRUCTS:
        assert hub_tag(golden(construct)) == STRUCTURED, construct
    assert hub_tag(LLVM_MODULE) == LOW
    assert hub_tag(CF_MODULE) == LOW
    assert hub_tag(MIXED_MODULE) == LOW, "one llvm op beside scf makes a module low"


# ---------------------------------------------------------------------------
# Never executed


def test_a_refused_source_is_never_executed(tmp_path: Path) -> None:
    refusal = frontend_refusal()
    marker = tmp_path / "executed-marker"
    text = f'"""Leaves a marker when run."""\n\nimport os\n\nos.makedirs({str(marker)!r})\n'
    with pytest.raises(refusal):
        raise_text(text)
    assert not marker.exists(), "the frontend ran the source"


def test_a_subset_program_is_never_executed_or_imported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "imported-marker"
    called = tmp_path / "called-marker"
    fake = (
        '"""A stand-in binding that leaves a marker when imported or called."""\n\n'
        f"open({str(marker)!r}, 'w').close()\n\n\n"
        f"def read_array(path):\n    open({str(called)!r}, 'w').close()\n\n\n"
        f"def write_array(*args):\n    open({str(called)!r}, 'w').close()\n"
    )
    (tmp_path / "lassi_io.py").write_bytes(fake.encode("ascii"))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "lassi_io", raising=False)
    monkeypatch.setattr(sys, "argv", ["main.py", str(tmp_path / "x.lassiio"), str(tmp_path / "y.lassiio")])
    module = raise_text(source("harness_io"), "harness_io.py")
    assert module.text == golden("harness_io")
    assert not marker.exists(), "the frontend imported the binding the source names"
    assert not called.exists(), "the frontend ran the source's main"
    assert "lassi_io" not in sys.modules
    assert not (tmp_path / "y.lassiio").exists()


def test_the_frontend_has_no_way_to_run_source_text() -> None:
    module = python_subset_module()
    path = Path(module.__file__ or "")
    files = sorted(path.parent.rglob("*.py")) if path.name == "__init__.py" else [path]
    assert files and all(file.is_file() for file in files), files
    forbidden = re.compile(r"\bexec\(|\beval\(|(?<![\w.])compile\(|__import__|\bimportlib\b|\brunpy\b|\bsubprocess\b")
    found = []
    for path in files:
        for number, line in enumerate(path.read_bytes().decode("ascii").split("\n"), 1):
            if forbidden.search(line):
                found.append(f"{path.relative_to(REPO).as_posix()}:{number}: {line.strip()}")
    assert not found, found


# ---------------------------------------------------------------------------
# The published subset and Agent Rule 3


def test_the_subset_is_published_in_the_module_docstring() -> None:
    doc = inspect.getdoc(python_subset_module()) or ""
    for word in (
        "prange", "range(", "while", "read_array", "write_array", "sys.argv", "Annotated", "memoryview",
        "bytearray", "np.asarray", "math.sqrt", "lassi_io_read_", "lassi_io_write_", "cpu-mlir", "structured",
        "memref.alloca_scope", "memref.dealloc", "MAX_LINE_TOKENS", "nested too deeply",
    ):
        assert word in doc, f"the module docstring does not publish {word!r}"
    assert doc.isascii()


def test_lassi_frontends_names_no_project() -> None:
    frontends_package()
    found = []
    for path in sorted(FRONTENDS_DIR.rglob("*.py")):
        lowered = path.read_bytes().decode("ascii").lower()
        found += [f"{path.name}: {name}" for name in PROJECT_NAMES if name in lowered]
    assert not found, f"Agent Rule 3: no project in lassi/: {found}"
