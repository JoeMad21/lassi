"""The Python-subset frontend: Python 3.10 source raised to a structured hub module (task P12.4).

Importing lassi.frontends registers it as the Frontend python-subset, built as factory(). raise_source({path:
text}, []) reads the one source file with the standard ast module and never executes or imports the source or
anything it names. It returns Module(level="cpu-mlir", text, tag): MLIR in custom assembly at LLVM 18.0.0git's
syntax (Polygeist's LLVM, llvm-project 26eb4285), holding only builtin's module and the func, arith, math, scf, and
memref dialects, tagged by lassi.frontends.hub_tag from the dialects it holds (structured, since it holds no llvm or
cf op), and checked with lassi.ir.form check_ir (custom form, whole) before it is returned. A source outside the
subset raises lassi.core.interfaces.FrontendRefusal with one parse-stage Diagnostic per construct (Refusals,
below), never a partial module. The frontend takes no flags (a flag is a ValueError) and one source file (two or
more are a refusal). The same text gives the same bytes on every raise. The pinned parse and verify of the text
is task P12.12's, since no pinned MLIR tool runs on Windows.

The published subset (the bible's Frontend Rules subset for Python: loops over arrays, arithmetic and math
intrinsics, prange and NumPy vectorized ops; plus the harness binding's calls)

Module level:
- A docstring; the imports `import math`, `import sys`, `import numpy as np`, `from typing import Annotated`,
  `from __future__ import annotations`, and `from lassi_io import` any of read_array, write_array, and prange,
  none aliased; function definitions; and, as the last statement, the entry guard
  `if __name__ == "__main__": main()`, calling the module's function main, the program's entry, which takes no
  argument and returns None (`func.func @main()`; the CPU target's runtime, task P12.12's, calls it). Docstrings,
  imports, and the guard emit nothing.

Functions:
- `def f(p: T, ...) -> R:` with positional parameters only, each annotated, no default and no decorator. T is int
  (i64), float (f64), bool (i1), or `Annotated[memoryview, "<dtype>"]`, an array: a flat memoryview cast to its
  dtype's format, as the binding's read_array(...).data gives, raised as memref<?xT> with T f32 for f32, f64 for
  f64, i8 for i8 and u8, i32 for i32 and u32, and i64 for i64 (f16 and bf16 are outside). R is int, float, bool,
  or None (`func.func @f(...)` with no result). Function names are ASCII, unique, none of the reserved names
  below, none of the C library names the hub module's lowering can call (malloc and free, for memref.alloc and
  memref.dealloc; exp, log, sin, cos, and sqrt, for the math ops), since every function is a public symbol of the
  module, and none starts with lassi_io_. A function named main takes no argument and returns None. Functions
  may call each other in any order, recursion included.
- A function with a result ends with `return <expression>` (an int returned from a float function is converted
  with sitofp), and return stands nowhere else; a function returning None may end with a bare return.

Statements in a function body:
- A docstring first; pass; `x = e`; `x: int = e` (or float, bool); `x op= e`; `a[i] = e`; `a[i] op= e`;
  if, elif, else; `for v in range(stop)`, `range(start, stop)`, or `range(start, stop, step)` with a positive int
  literal step that fits in 64 bits; `for v in prange(...)` with the same arguments; `while c:`; a call as a
  statement; and the array bindings, harness I/O calls, and NumPy assignment below. Conditions of if and while are
  bools.
- Each scalar name keeps the type of its first binding in source order (a loop variable is an int), and every
  binding gives it that type: rebinding an int name to a float is a refusal. A parameter may be assigned; an
  array may not. No local takes a function's name or a reserved name: math, sys, np, numpy, Annotated,
  read_array, write_array, prange, range, len, abs, float, int, bool, memoryview, bytearray, __name__.

Expressions:
- int literals that fit in 64 bits, finite float literals, True and False; names; + - * / // % & | ^ << >> on
  ints (/ gives a float); + - * / on floats, an int operand converted; unary - and + on numbers, ~ on ints, not on
  bools; one comparison (< <= > >= == !=) of two numbers; and, or over bools; `x if c else y` with arms of one
  type; a[i] with one int index (never a negative literal); calls of the module's functions with positional
  arguments (an int argument to a float parameter is converted; an array argument is a view of the parameter's
  dtype); len(a), abs(x), float(x), and int(x); and math.sqrt, math.exp, math.log, math.sin, and math.cos.

Arrays:
- Allocation: `a = memoryview(bytearray(<count> * <itemsize>)).cast("<format>")`, the binding's form, with format
  f, d, b, B, i, I, or q (dtype f32, f64, i8, u8, i32, u32, i64) and its itemsize literal (4, 8, 1, 1, 4, 4, 8);
  zero-filled as bytearray is. An annotation `a: Annotated[memoryview, "<dtype>"] = ...` names the same dtype.
- `x = np.asarray(v)` of an f64 view (or of such an ndarray) is an ndarray sharing v's elements; an ndarray is
  indexed like a view, but is passed neither to a function nor to write_array (pass the view instead).
- `b = a` makes b another name for the array a.
- Each array name is bound once, directly in the function body (not inside a loop or a branch), and used only
  after its binding.

NumPy vectorized ops:
- `z[:] = <expression>` with z an f64 ndarray: the expression combines f64 ndarrays, scalar expressions of the
  subset, + - * /, and np.sqrt of a part holding an ndarray. Its scalar parts are evaluated first, in evaluation
  order; then scf.parallel runs over z's length, loading each ndarray at the induction variable, and stores the
  result into z. np.sqrt stands only there; math.sqrt takes a scalar. A scalar part after the first ndarray, in
  evaluation order, calls no module function that takes an array, since NumPy reads that ndarray before the call,
  which may write it, and the hub module after it.

Parallel-for:
- `for v in prange(...)`, prange imported from lassi_io, is scf.parallel. The loop variable and every scalar the
  body binds are private to each iteration. Each is bound in the iteration, on every path, before it is read (a
  binding inside an inner loop, or in one branch of an if, does not reach past it; one in both branches does).
  None is used outside the loop, except inside another for loop that rebinds the name as its loop variable and
  holds no prange loop binding it; the name then has a slot of its own outside the prange body. Arrays are shared,
  so iterations must write distinct elements, which the frontend cannot check. prange, and write_array, inside a
  prange body are refusals. The binding exports prange as range, so the program runs natively under plain Python,
  no Numba.

Harness I/O (the P12 plan's form; task P12.12 reconciles it with the C route's):
- A program reads its inputs and names its outputs through argv: `@inputs/<name>.lassiio`, then
  `<name>.lassiio` (Harness Contract). A read is `v: Annotated[memoryview, "<dtype>"] = read_array(sys.argv[<k>]).data`
  directly in a function body, k a positive int literal that fits in 64 bits: `arith.constant <k> : i64`, then
  `call @lassi_io_read_<dtype>(%k) : (i64) -> memref<?xT>`. The array it returns belongs to the runtime, and the
  module never frees it.
- A write is the statement `write_array(sys.argv[<k>], "<name>", "<dtype>", (<dims>,), a)`, a a view of that dtype
  and the name 1 to 255 ASCII letters, digits, '_', '.', or '-': the constant k; memref.get_global of
  @lassi_io_name<N>, a `memref.global "private" constant` holding the name's ASCII bytes and one zero byte, and
  memref.cast to memref<?xi8>; the dims evaluated, a memref.alloca of memref<<rank>xi64>, each dim stored at an
  index constant, and memref.cast to memref<?xi64>; then
  `call @lassi_io_write_<dtype>(i64, memref<?xi8>, memref<?xi64>, memref<?xT>)`. k is as for a read. Inside a
  loop body the write's ops stand in one `memref.alloca_scope`, which frees the shape buffer each time.
- The externals are declared `func.func private` at the top of the module. The runtime that provides them is the
  CPU target's (task P12.12), which also captures argv before it calls @main.

The lowering scheme (each golden under tests/fixtures/frontends/python_subset/ shows one construct):
- Module: `module {`, one memref.global per output name, then the externals, each in order of first use, then each
  function in source order; two blanks per level, one op per line, LF line ends, `}` and a line feed last.
- Values: parameters are %arg0, %arg1, ...; every other value, induction variables included, is %0, %1, ... in the
  order its definition appears in the function's text. Constants are emitted where used, never shared: ints and
  loop bounds as `arith.constant <v> : i64` or `: index`, floats as Python's repr (1e-08 is written 1.0e-08), and
  bools as `true` or `false`; a minus on a number literal is part of the literal.
- Scalar locals live in rank-0 memref.alloca slots, loaded on reads and stored on bindings. Slots open at function
  entry: assigned parameters first, in parameter order (alloca, then a store of the argument), then the other
  names in order of first binding; names private to a prange body open at the start of its body, the loop
  variable first, inside the body's memref.alloca_scope, which frees them when the iteration ends (a private name
  that code outside every prange body also binds opens at entry too). Arrays are SSA values; each memref.alloc is
  freed with memref.dealloc, in allocation order, after the return value is evaluated and before the return
  (arrays never leave a function, since results are scalars).
- Expressions evaluate left to right as Python does; after both operands an int operand of a float operation is
  converted with sitofp, left first (/ converts both). // is floordivsi (the CPU target's conversion to llvm may
  need it expanded first, with arith-expand; task P12.12 confirms with the pinned tools), and a % b is
  a - (a // b) * b. Unary
  minus on a float is negf and on an int 0 - x; not is xori with true; ~ is xori with -1. Comparisons are cmpi
  (slt, sle, sgt, sge, eq, ne) or cmpf (olt, ole, ogt, oge, oeq, une). and and or short-circuit through
  `scf.if -> (i1)`; `x if c else y` is scf.if on c yielding each arm.
- An element load widens to the Python scalar (extf for f32; extsi for i8 and i32; extui for u8 and u32); a store
  narrows (truncf; trunci), an int stored into a float array through sitofp first. `a[i] = e` evaluates e, then i;
  `a[i] op= e` evaluates i, loads, widens, evaluates e, applies, narrows, and stores at the same index.
- if, elif, else is scf.if, an elif nested in the else region. A range loop evaluates its start, if given, and its
  stop, index_casts them in order, then writes the absent start 0 and the step as index constants; scf.for's body
  first stores the index_cast of the induction variable into the loop variable's slot. prange is the same with
  scf.parallel, whose body is one memref.alloca_scope, then scf.yield. `while c:` is scf.while : () -> () with c in
  its before region and scf.yield closing its do region. A call is `call` directly in a function body and
  `func.call` inside a region.
- len(a) is memref.dim at index 0 and an index_cast to i64; float() sitofp; int() fptosi; abs() math.absf or
  math.absi; the math functions are the math dialect's ops on f64, an int argument converted.

Refusals: every construct outside the subset gives one Diagnostic, stage "parse", severity "error", file the
path given, line, column (ast col_offset plus 1; a syntax error's is the parser's), and a one-line ASCII message
naming the construct. One cause is counted once: a refused statement's parts are not read again, and a later use
of a name whose binding was refused gives none, whether the binding stood in the function or at module level (an
import, a module-level statement, or a function the subset cannot take). The codes, each with the constructs it
refuses (known cases, not complete):

- subset-syntax: text that does not parse as Python 3.10 (a null byte included), and a logical line of more than
  MAX_LINE_TOKENS tokens, refused before the parser reads it.
- subset-import: an import other than the subset's, or an import inside a function.
- subset-module: a module-level statement other than a docstring, an import, a function, or the entry guard; an
  entry guard that does not call main; a function main that takes an argument or returns a value; a function name
  the subset cannot take (a reserved name or one of the C library names above); a program of more than one source
  file.
- subset-statement: a statement kind outside the subset (try, with, raise, assert, del, global, a nested def, a
  chained assignment, an expression statement that is not a call); a decorator; a parameter form other than
  annotated positional parameters.
- subset-control: break, continue, an early return, a loop else, a loop over anything but range or prange, a range
  step that is not a positive int literal, and a function with a result that does not end with a return.
- subset-expression: an expression or operator outside the subset (**, @, a chained comparison, is, in, a lambda,
  a display, an attribute, a string, a name never bound); and a source nested too deeply for the frontend's
  recursion, which gives one diagnostic with no line in place of all others.
- subset-call: a call of anything but the module's functions, the subset's builtins, and its math functions (such
  as print), and a call with keyword or starred arguments.
- subset-type: a type the subset cannot settle statically: a name rebound with another type, a missing or other
  annotation, an operator or a call on the wrong types, a condition that is not a bool, a shadowed name; and an int
  literal that does not fit in 64 bits, a range step included.
- subset-array: an array use outside the subset: two indices, a slice, an array binding in a loop or a branch, a
  second binding, a use before the binding, an allocation of another form, a view in a vectorized expression, and
  a call of a module function that takes an array after an ndarray in a vectorized expression.
- subset-io: a binding call outside its form: a path that is not sys.argv[<k>] (k past 64 bits included), a read
  without its annotation, a write of an ndarray or with another dtype.
- subset-prange: a scalar shared between a prange body and code outside it, a private scalar read before the
  iteration binds it, prange or write_array inside a prange body, and prange not imported from lassi_io.

Known limits (known cases, not complete): ints are 64-bit, so arithmetic that Python carries past 64 bits wraps,
and a shift by 64 or more is undefined; a division or modulo by zero, a math domain error, an index out of range at
run time, an element value out of its dtype's range, and an f32 overflow raise in Python but not in the hub module;
a negative index at run time counts from the end in Python but reads before the array's start in the hub module;
in z[:] = <expression>, NumPy broadcasts an ndarray of length 1 and raises on a length mismatch, while the hub
module reads every ndarray at each index of z; a scalar read on a path where Python finds it unbound reads an unset
slot; int-to-float conversion and comparison of an int with a float round past 2**53; prange iterations that write
the same element race, and write_array in a function that a prange body calls runs once per iteration, in
parallel; function names are the module's public symbols, so a name other than the reserved ones that a library
the CPU target links also defines may clash with it there (task P12.12 confirms with the pinned lowering).
"""

from __future__ import annotations

import ast
import io
import tokenize
from typing import Mapping, Sequence

from lassi.core.interfaces import FrontendRefusal, Module
from lassi.core.record import Diagnostic
from lassi.core.registry import register
from lassi.frontends.hub import HUB_LEVEL, hub_tag
from lassi.frontends.python_subset_check import CODES, EXPRESSION, MODULE, SYNTAX, check_module, plain
from lassi.frontends.python_subset_lower import lower_module
from lassi.ir.form import check_ir

__all__ = ["CODES", "MAX_LINE_TOKENS", "PythonSubsetFrontend"]

# The most tokens one logical line may hold; a longer one is refused before the parser reads it (see _parse).
# [DESIGN] margin, not a measured limit.
MAX_LINE_TOKENS = 5000


@register("Frontend", "python-subset")
class PythonSubsetFrontend:
    """Raises one Python 3.10 source in the published subset to a structured cpu-mlir hub module; never runs it."""

    name = "python-subset"
    capabilities = frozenset({"diagnostics"})

    def raise_source(self, files: Mapping[str, str], flags: Sequence[str]) -> Module:
        """Return the hub module of the one source in `files`, or raise FrontendRefusal (see the module docstring).

        Raises ValueError for a flag or for no file, TypeError for a path or a text that is not a str, and
        FrontendRefusal for two or more files or for any construct outside the subset.
        """
        given = list(flags)
        if given:
            verb = "was" if len(given) == 1 else "were"
            raise ValueError(f"the python-subset frontend takes no flags, and {len(given)} {verb} given")
        path, text = _one_file(files)
        tree = _parse(path, text)
        try:
            info, diagnostics = check_module(tree, path)
            if diagnostics:
                raise FrontendRefusal(diagnostics)
            hub = lower_module(info)
        except RecursionError:
            raise FrontendRefusal([_nesting(path)]) from None
        check_ir(hub, level=HUB_LEVEL, where=path)
        return Module(level=HUB_LEVEL, text=hub, tag=hub_tag(hub))


def _one_file(files: Mapping[str, str]) -> tuple[str, str]:
    """Return the one source's path and text; no file is a ValueError and more than one a FrontendRefusal."""
    items = list(files.items())
    if not items:
        raise ValueError("the python-subset frontend raises one source file, and none was given")
    for path, text in items:
        if not isinstance(path, str) or not isinstance(text, str):
            kinds = f"{type(path).__name__} and {type(text).__name__}"
            raise TypeError(f"a source is a str path and a str text, not {kinds}")
    if len(items) > 1:
        names = ", ".join(path for path, _ in items)
        message = f"a program of {len(items)} source files ({names}) is outside the subset, which raises one file"
        raise FrontendRefusal(
            [Diagnostic(stage="parse", severity="error", code=MODULE, file=None, message=plain(message))]
        )
    return items[0]


def _parse(path: str, text: str) -> ast.Module:
    """Return the source's syntax tree; text that does not parse as Python 3.10 raises FrontendRefusal.

    A logical line of more than MAX_LINE_TOKENS tokens is refused before the parser reads it. CPython's parser
    recurses on nested expressions, and a recursion that runs past the C stack ends the process, which no handler
    can catch; MAX_LINE_TOKENS is a [DESIGN] margin against that, chosen here and not measured.
    """
    long_line = _long_logical_line(text)
    if long_line is not None:
        reason = f"a logical line of more than {MAX_LINE_TOKENS} tokens"
        message = f"syntax error: {reason} is outside the subset, which reads such a line without parsing it"
        line, column = long_line, 1
    else:
        try:
            return ast.parse(text, filename=path, mode="exec", feature_version=(3, 10))
        except SyntaxError as error:
            line = error.lineno if isinstance(error.lineno, int) and error.lineno >= 1 else 1
            column = error.offset if isinstance(error.offset, int) and error.offset >= 1 else 1
            reason = " ".join(str(error.msg).split())
            message = f"syntax error: the text does not parse as Python 3.10 ({reason}), so it is outside the subset"
        except (ValueError, MemoryError) as error:  # a null byte (ValueError); the parser out of memory
            line, column = 1, 1
            reason = str(error) or type(error).__name__
            message = f"syntax error: the text does not parse as Python 3.10 ({reason}), so it is outside the subset"
    diagnostic = Diagnostic(
        stage="parse", severity="error", code=SYNTAX, file=path, line=line, column=column, message=plain(message)
    )
    raise FrontendRefusal([diagnostic])


def _long_logical_line(text: str) -> int | None:
    """Return the first line of a logical line holding more than MAX_LINE_TOKENS tokens, or None.

    Text the tokenizer cannot read gives None, and the parser then reports why.
    """
    count, start = 0, None
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if start is None:
                start = token.start[0]
            count += 1
            if count > MAX_LINE_TOKENS:
                return start
            if token.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
                count, start = 0, None
    except (tokenize.TokenError, SyntaxError, ValueError):
        return None
    return None


def _nesting(path: str) -> Diagnostic:
    """Return the diagnostic of a source nested too deeply for the frontend's recursion."""
    message = "an expression or statement nested too deeply for the frontend is outside the subset"
    return Diagnostic(stage="parse", severity="error", code=EXPRESSION, file=path, message=message)
