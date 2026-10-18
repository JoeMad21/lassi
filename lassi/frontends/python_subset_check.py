"""The Python-subset frontend's checker: the subset's rules and the type of every value (task P12.4).

check_module(tree, path) reads a module parsed by the standard ast module and returns its ModuleInfo, from which
lassi.frontends.python_subset_lower writes the hub text, with the Diagnostic of every construct outside the subset
in source order. It never executes or imports anything. lassi.frontends.python_subset's docstring publishes the
subset and its refusal codes; this module carries them out.

The module level is read first: its imports, its function names, and each function's signature. Then each function
is read in three steps. The scope step collects every binding in source order (assignments, annotated and augmented
assignments, and for loop variables), the array names, and what each prange body binds. The type step settles each
scalar name's type from its first binding whose type is known, and repeats until nothing changes, so a name read in
a loop before the line that binds it still settles. The check step walks the statements in source order, refuses
each construct outside the subset with one Refuse per statement (a refused statement's parts are not read again),
and records the type of every expression it reads, which the lowering uses. A name whose binding was refused is
poisoned: a later use of it gives no diagnostic of its own, so one cause is counted once. Module-level names are
poisoned the same way (an import, a module-level statement, or a function the subset cannot take), for every
function where no local or parameter takes the name. Before the check step, each read of a prange body's private
scalar that no binding earlier in the same iteration reaches is marked, and the check step refuses it there.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any, NamedTuple, Union

from lassi.core.record import Diagnostic

# The refusal codes, one per class of construct outside the subset (lassi.frontends.python_subset publishes them).
SYNTAX = "subset-syntax"
IMPORT = "subset-import"
MODULE = "subset-module"
STATEMENT = "subset-statement"
CONTROL = "subset-control"
EXPRESSION = "subset-expression"
CALL = "subset-call"
TYPE = "subset-type"
ARRAY = "subset-array"
IO = "subset-io"
PRANGE = "subset-prange"
CODES = (SYNTAX, IMPORT, MODULE, STATEMENT, CONTROL, EXPRESSION, CALL, TYPE, ARRAY, IO, PRANGE)

# Scalar types by their MLIR names, and the result of a call of a function that returns None.
INT = "i64"
FLOAT = "f64"
BOOL = "i1"
NONE = "none"
_WORDS = {INT: "an int", FLOAT: "a float", BOOL: "a bool", NONE: "None"}
_SCALAR_ANNOTATIONS = {"int": INT, "float": FLOAT, "bool": BOOL}

# Each array dtype of the subset with its MLIR element type, and each memoryview format with its dtype and itemsize.
ELEMENTS = {"f32": "f32", "f64": "f64", "i8": "i8", "u8": "i8", "i32": "i32", "u32": "i32", "i64": "i64"}
FORMATS = {
    "f": ("f32", 4), "d": ("f64", 8), "b": ("i8", 1), "B": ("u8", 1), "i": ("i32", 4), "I": ("u32", 4), "q": ("i64", 8)
}

# The imports of the subset: `import <module>` with the name it must bind, and `from <module> import <name>`.
_IMPORTS = {"math": "math", "sys": "sys", "numpy": "np"}
_FROM_IMPORTS = {
    "typing": ("Annotated",),
    "lassi_io": ("read_array", "write_array", "prange"),
    "__future__": ("annotations",),
}
_IMPORT_RULE = (
    "the subset imports math, sys, numpy as np, Annotated from typing, annotations from __future__, and "
    "read_array, write_array, and prange from lassi_io, none aliased"
)
# The names a local or a function may not take, since the subset reads them by name.
RESERVED = frozenset(
    {
        "math", "sys", "np", "numpy", "Annotated", "read_array", "write_array", "prange", "range", "len", "abs",
        "float", "int", "bool", "memoryview", "bytearray", "__name__",
    }
)
# The prefix of the binding's external functions and globals in the hub module, which no function may take.
RESERVED_PREFIX = "lassi_io_"
# The C library names the hub module's lowering can call (memref.alloc and memref.dealloc; the math ops), which no
# function may take, since every function is a public symbol of the module.
C_SYMBOLS = frozenset({"malloc", "free", "exp", "log", "sin", "cos", "sqrt"})
MATH_FUNCTIONS = ("sqrt", "exp", "log", "sin", "cos")
_BUILTINS = ("len", "abs", "float", "int")
# An output name of write_array (the Harness Contract's name rule).
_OUTPUT_NAME = re.compile(r"[A-Za-z0-9_.-]{1,255}")
_INT_RANGE = (-(2**63), 2**63 - 1)

# Each binary operator's symbol, the operators that take ints only, and each comparison's symbol and predicates.
BINARY = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
    ast.MatMult: "@", ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>",
}
_INT_ONLY = (ast.FloorDiv, ast.Mod, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)
_VECTOR_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
COMPARISONS = {
    ast.Lt: ("<", "slt", "olt"), ast.LtE: ("<=", "sle", "ole"), ast.Gt: (">", "sgt", "ogt"),
    ast.GtE: (">=", "sge", "oge"), ast.Eq: ("==", "eq", "oeq"), ast.NotEq: ("!=", "ne", "une"),
}
_OTHER_COMPARISONS = {ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in"}
_UNARY = {ast.USub: "-", ast.UAdd: "+", ast.Not: "not", ast.Invert: "~"}
_STATEMENT_WORDS = {
    ast.Try: "try statement", ast.With: "with statement", ast.AsyncWith: "async with statement",
    ast.AsyncFor: "async for loop", ast.AsyncFunctionDef: "async function definition", ast.Raise: "raise statement",
    ast.Assert: "assert statement", ast.Delete: "del statement", ast.Global: "global statement",
    ast.Nonlocal: "nonlocal statement", ast.If: "if statement", ast.For: "for loop", ast.While: "while loop",
    ast.Return: "return statement", ast.Pass: "pass statement", ast.Break: "break", ast.Continue: "continue",
    ast.Match: "match statement", ast.Import: "import", ast.ImportFrom: "import",
}
_EXPRESSION_WORDS = {
    ast.Lambda: "a lambda", ast.List: "a list", ast.Tuple: "a tuple", ast.Dict: "a dict", ast.Set: "a set",
    ast.ListComp: "a list comprehension", ast.SetComp: "a set comprehension", ast.DictComp: "a dict comprehension",
    ast.GeneratorExp: "a generator expression", ast.JoinedStr: "an f-string", ast.Starred: "a starred expression",
    ast.NamedExpr: "an assignment expression", ast.Await: "an await", ast.Yield: "a yield",
    ast.YieldFrom: "a yield from", ast.Slice: "a slice",
}


@dataclass(frozen=True)
class ArrayType:
    """A one-dimensional array of one dtype: a memoryview cast to the dtype's format, or (numpy) an ndarray of it."""

    dtype: str
    numpy: bool = False

    @property
    def element(self) -> str:
        """Return the MLIR element type."""
        return ELEMENTS[self.dtype]

    @property
    def memref(self) -> str:
        """Return the MLIR type of the array, memref<?x<element>>."""
        return f"memref<?x{self.element}>"

    @property
    def scalar(self) -> str:
        """Return the Python scalar an element reads as: a float for f32 and f64, else an int."""
        return FLOAT if self.dtype.startswith("f") else INT

    @property
    def unsigned(self) -> bool:
        """Return True for u8 and u32, whose elements widen with zero extension."""
        return self.dtype.startswith("u")


ValueType = Union[str, ArrayType]


@dataclass(frozen=True)
class Signature:
    """A function's name, its parameters with their types, and its result type (None for a function returning None)."""

    name: str
    params: tuple[tuple[str, ValueType], ...]
    result: str | None


class ArrayBinding(NamedTuple):
    """How a statement binds an array: kind alloc (source: the count), read (the argv index), or view (a name)."""

    kind: str
    array: ArrayType
    source: Any


@dataclass(frozen=True)
class FunctionInfo:
    """One checked function, as the lowering reads it.

    `slots` maps each scalar name to its type (parameters included); `assigned` lists the scalar parameters the
    body assigns, in parameter order; `entry` the other scalar names whose slots open at function entry, in order
    of first binding; `private` each prange loop's names, its loop variable first, whose slots open in its body.
    `arrays` maps every array name to its type, `array_bindings` each array-binding statement to its binding,
    `types` every expression the check read to its type, and `vector` the nodes of a vectorized expression that
    hold an ndarray.
    """

    node: ast.FunctionDef
    signature: Signature
    slots: dict[str, str]
    assigned: tuple[str, ...]
    entry: tuple[str, ...]
    private: dict[ast.For, tuple[str, ...]]
    arrays: dict[str, ArrayType]
    array_bindings: dict[ast.AST, ArrayBinding]
    types: dict[ast.AST, ValueType]
    vector: frozenset[ast.AST]


@dataclass(frozen=True)
class ModuleInfo:
    """A checked module: its functions in source order and every accepted signature by name."""

    functions: tuple[FunctionInfo, ...]
    signatures: dict[str, Signature]


class Refuse(Exception):
    """A construct outside the subset: its refusal code, the node it stands at, and a one-line message."""

    def __init__(self, code: str, node: ast.AST, message: str) -> None:
        """Keep the code, the node, and the message."""
        super().__init__(message)
        self.code = code
        self.node = node
        self.message = message

    def diagnostic(self, path: str) -> Diagnostic:
        """Return the parse-stage Diagnostic: the node's line, its column as col_offset plus 1, and the message."""
        line = getattr(self.node, "lineno", None)
        offset = getattr(self.node, "col_offset", None)
        return Diagnostic(
            stage="parse", severity="error", code=self.code, file=path, line=line,
            column=None if offset is None else offset + 1, message=plain(self.message),
        )


class Skip(Exception):
    """A construct whose cause was refused already, so it gives no diagnostic of its own."""


class _Unsettled(Exception):
    """A scalar whose type the type step has not settled yet."""


class _Where(NamedTuple):
    """Where a statement stands: directly in the function body or not, and the prange loop whose body holds it."""

    top: bool
    prange: ast.For | None


def plain(text: str) -> str:
    """Return `text` as one line of printable ASCII: each other character becomes its backslash escape."""
    return "".join(char if " " <= char <= "~" else char.encode("unicode_escape").decode("ascii") for char in text)


def describe(node: ast.AST) -> str:
    """Return the source text of `node` on one line, cut after 40 characters."""
    try:
        text = " ".join(ast.unparse(node).split())
    except ValueError:  # an int literal past the digit limit of str()
        return f"a {type(node).__name__.lower()}"
    return text if len(text) <= 40 else f"{text[:37]}..."


def word(kind: ValueType) -> str:
    """Return a type in words, such as `a float`, for a message."""
    return "an array" if isinstance(kind, ArrayType) else _WORDS.get(kind, kind)


def mlir_type(kind: ValueType) -> str:
    """Return the MLIR type of a parameter or value type."""
    return kind.memref if isinstance(kind, ArrayType) else kind


def is_docstring(stmt: ast.stmt) -> bool:
    """Return True for an expression statement that is a string literal."""
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def is_str(node: ast.AST) -> bool:
    """Return True for a string literal."""
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def is_int_literal(node: ast.AST) -> bool:
    """Return True for an int literal (a bool is not one)."""
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool)


def is_name(node: ast.AST, name: str) -> bool:
    """Return True when `node` is the name `name`."""
    return isinstance(node, ast.Name) and node.id == name


def is_named_call(node: ast.AST, name: str) -> bool:
    """Return True for a call of the name `name`."""
    return isinstance(node, ast.Call) and is_name(node.func, name)


def is_prange_loop(stmt: ast.stmt) -> bool:
    """Return True for a for loop over prange(...)."""
    return isinstance(stmt, ast.For) and is_named_call(stmt.iter, "prange")


def folded_literal(node: ast.AST) -> int | float | None:
    """Return the value of a number literal, a unary minus on one included; None for anything else."""
    negative = isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
    inner = node.operand if negative else node
    if not isinstance(inner, ast.Constant) or isinstance(inner.value, bool):
        return None
    if not isinstance(inner.value, (int, float)):
        return None
    return -inner.value if negative else inner.value


def alloc_shaped(node: ast.AST) -> bool:
    """Return True for a call `memoryview(...).cast(...)`, the shape of the subset's allocation."""
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "cast"
        and is_named_call(node.func.value, "memoryview")
    )


def read_shaped(node: ast.AST) -> bool:
    """Return True for `read_array(...).data`, the shape of the subset's input read."""
    return isinstance(node, ast.Attribute) and node.attr == "data" and is_named_call(node.value, "read_array")


def asarray_shaped(node: ast.AST) -> bool:
    """Return True for a call np.asarray(...)."""
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "asarray"
        and is_name(node.func.value, "np")
    )


def np_sqrt_shaped(node: ast.AST) -> bool:
    """Return True for a call np.sqrt(...)."""
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "sqrt"
        and is_name(node.func.value, "np")
    )


def argv_index(node: ast.AST) -> int | None:
    """Return k for `sys.argv[k]` with k an int literal, else None."""
    if not isinstance(node, ast.Subscript) or not is_int_literal(node.slice):
        return None
    base = node.value
    if not (isinstance(base, ast.Attribute) and base.attr == "argv" and is_name(base.value, "sys")):
        return None
    return node.slice.value


def binop_result(op: ast.operator, left: str, right: str) -> str:
    """Return the type of a binary operation on two numbers the check accepted: / and any float give a float."""
    if isinstance(op, ast.Div) or FLOAT in (left, right):
        return FLOAT
    return INT


def article(phrase: str) -> str:
    """Return `phrase` after the indefinite article it takes: `an assert statement`, `a try statement`."""
    return f"{'an' if phrase[:1].lower() in 'aeiou' else 'a'} {phrase}"


def root_name(node: ast.AST) -> str | None:
    """Return the name an attribute chain such as numpy.linalg.norm starts from, or None when it starts elsewhere."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def bound_names(stmt: ast.stmt) -> set[str]:
    """Return every name a statement binds in its own scope; a nested def's, class's, or lambda's body is not read."""
    names: set[str] = set()
    pending: list[ast.AST] = [stmt]
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).partition(".")[0] for alias in node.names if alias.name != "*")
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name is not None:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            names.add(node.rest)
        pending.extend(ast.iter_child_nodes(node))
    return names


def statement_phrase(stmt: ast.stmt) -> str:
    """Return a statement in words for a message, such as `assignment to SIZE` or `try statement`."""
    if isinstance(stmt, ast.Assign):
        return "assignment to " + ", ".join(describe(target) for target in stmt.targets)
    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        return f"assignment to {describe(stmt.target)}"
    if isinstance(stmt, ast.FunctionDef):
        return f"nested function definition {stmt.name}"
    if isinstance(stmt, ast.ClassDef):
        return f"class definition {stmt.name}"
    if isinstance(stmt, ast.Expr):
        return f"expression statement {describe(stmt.value)}"
    return _STATEMENT_WORDS.get(type(stmt), f"{type(stmt).__name__.lower()} statement")


def early_message(name: str) -> str:
    """Return the message of a read of a prange body's private scalar before the iteration binds it."""
    return (
        f"{name} is read in a prange body before the same iteration binds it, which is outside the subset; a scalar "
        "private to a prange body is bound in each iteration, on every path, before it is read"
    )


def _before(node: ast.AST, site: ast.AST) -> bool:
    """Return True when `node` starts before the end of the statement `site`."""
    return (node.lineno, node.col_offset) < (site.end_lineno, site.end_col_offset)


def _place(diagnostic: Diagnostic) -> tuple[int, int]:
    """Return a diagnostic's line and column for sorting; one without a place sorts first."""
    return (diagnostic.line or 0, diagnostic.column or 0)


def check_module(tree: ast.Module, path: str) -> tuple[ModuleInfo, list[Diagnostic]]:
    """Return the checked module and the diagnostics of every construct outside the subset, in source order.

    The ModuleInfo is for the lowering and holds only what the check accepted; it is whole only when the list
    of diagnostics is empty.
    """
    reader = _ModuleReader(tree)
    info = reader.read()
    diagnostics = sorted((refusal.diagnostic(path) for refusal in reader.refusals), key=_place)
    return info, diagnostics


class _ModuleReader:
    """Reads the module level: imports, function names, signatures, and then each function."""

    def __init__(self, tree: ast.Module) -> None:
        """Start before the module's first statement."""
        self.tree = tree
        self.imports: set[str] = set()
        self.refusals: list[Refuse] = []
        self.defs: list[ast.FunctionDef] = []
        self.function_names: frozenset[str] = frozenset()
        self.signatures: dict[str, Signature] = {}
        # Module-level names whose binding was refused (an import, a statement, a function), counted once.
        self.poisoned: set[str] = set()

    def read(self) -> ModuleInfo:
        """Read every module statement, then every signature, then every function body with an accepted signature."""
        body = self.tree.body
        for index, stmt in enumerate(body):
            try:
                self._module_statement(stmt, index, index == len(body) - 1)
            except Refuse as refusal:
                self.refusals.append(refusal)
                self.poisoned |= bound_names(stmt)
        self.function_names = frozenset(node.name for node in self.defs)
        for node in self.defs:
            try:
                self.signatures[node.name] = _read_signature(node, self)
            except Refuse as refusal:
                self.refusals.append(refusal)
                self.poisoned.add(node.name)
            except Skip:
                self.poisoned.add(node.name)
        functions = tuple(
            _FunctionChecker(self, node, self.signatures[node.name]).check()
            for node in self.defs
            if node.name in self.signatures
        )
        return ModuleInfo(functions, dict(self.signatures))

    def require(self, name: str, node: ast.AST, code: str, message: str) -> None:
        """Raise Refuse with `message` unless `name` was imported; Skip when its import was refused, counted already."""
        if name in self.imports:
            return
        if name in self.poisoned:
            raise Skip
        raise Refuse(code, node, message)

    def _module_statement(self, stmt: ast.stmt, index: int, last: bool) -> None:
        """Read one module-level statement; anything but the subset's raises Refuse."""
        if index == 0 and is_docstring(stmt):
            return
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            self._import(stmt)
        elif isinstance(stmt, ast.FunctionDef):
            self._define(stmt)
        elif isinstance(stmt, ast.If) and _is_guard_test(stmt.test):
            self._guard(stmt, last)
        else:
            raise Refuse(
                MODULE, stmt,
                f"the module-level {statement_phrase(stmt)} is outside the subset; a module holds a docstring, "
                "imports, functions, and the entry guard",
            )

    def _import(self, stmt: ast.Import | ast.ImportFrom) -> None:
        """Record the names an import of the subset binds; any other import raises Refuse."""
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if _IMPORTS.get(alias.name) != (alias.asname or alias.name):
                    shown = f"{alias.name} as {alias.asname}" if alias.asname else alias.name
                    raise Refuse(IMPORT, stmt, f"import {shown} is outside the subset; {_IMPORT_RULE}")
            self.imports.update(alias.asname or alias.name for alias in stmt.names)
            return
        module = stmt.module or ""
        allowed = _FROM_IMPORTS.get(module, ()) if stmt.level == 0 else ()
        for alias in stmt.names:
            if alias.name not in allowed or alias.asname:
                shown = f"{alias.name} as {alias.asname}" if alias.asname else alias.name
                source = "." * stmt.level + module
                raise Refuse(IMPORT, stmt, f"from {source} import {shown} is outside the subset; {_IMPORT_RULE}")
        self.imports.update(alias.name for alias in stmt.names)

    def _define(self, stmt: ast.FunctionDef) -> None:
        """Record a function definition; a name the subset cannot take raises Refuse."""
        name = stmt.name
        if not name.isascii() or name in RESERVED or name in C_SYMBOLS or name.startswith(RESERVED_PREFIX):
            raise Refuse(
                MODULE, stmt,
                f"the function name {name} is outside the subset; a function name is ASCII, is none of the names "
                "the subset reads or the C library names its lowering can call (malloc, free, exp, log, sin, cos, "
                f"sqrt), and does not start with {RESERVED_PREFIX}",
            )
        if any(node.name == name for node in self.defs):
            raise Refuse(MODULE, stmt, f"a second function named {name} is outside the subset")
        self.defs.append(stmt)

    def _guard(self, stmt: ast.If, last: bool) -> None:
        """Accept the entry guard: the last statement, calling the module's function main with no argument."""
        call = stmt.body[0].value if len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Expr) else None
        calls_main = isinstance(call, ast.Call) and is_name(call.func, "main") and not call.args and not call.keywords
        if last and not stmt.orelse and calls_main and any(node.name == "main" for node in self.defs):
            return
        raise Refuse(
            MODULE, stmt,
            'the module-level entry guard is the last statement, if __name__ == "__main__": main(), calling the '
            "module's function main; any other form is outside the subset",
        )


def _is_guard_test(test: ast.expr) -> bool:
    """Return True for `__name__ == "__main__"`."""
    return (
        isinstance(test, ast.Compare) and is_name(test.left, "__name__") and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq) and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _read_signature(node: ast.FunctionDef, module: _ModuleReader) -> Signature:
    """Return a function's signature; a decorator, parameter form, or annotation outside the subset raises Refuse.

    The function main, the program's entry, takes no argument and returns None.
    """
    if node.decorator_list:
        raise Refuse(STATEMENT, node.decorator_list[0], f"a decorator on {node.name} is outside the subset")
    args = node.args
    if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg:
        raise Refuse(
            STATEMENT, node,
            f"{node.name} takes positional-only, starred, or keyword-only parameters, which are outside the subset",
        )
    if args.defaults:
        raise Refuse(STATEMENT, args.defaults[0], f"a parameter default of {node.name} is outside the subset")
    params = []
    for arg in args.args:
        _local_name(arg.arg, arg, module.function_names)
        params.append((arg.arg, _annotation(arg.annotation, arg, module, f"the parameter {arg.arg}")))
    signature = Signature(node.name, tuple(params), _result_annotation(node))
    if node.name == "main" and (signature.params or signature.result is not None):
        raise Refuse(
            MODULE, node,
            "the function main is the program's entry, which takes no argument and returns None in the subset",
        )
    return signature


def _annotation(annotation: ast.expr | None, node: ast.AST, module: _ModuleReader, what: str) -> ValueType:
    """Return the type a parameter's annotation names; a missing or other annotation raises Refuse."""
    rule = 'the subset reads int, float, bool, and Annotated[memoryview, "<dtype>"]'
    if annotation is None:
        raise Refuse(TYPE, node, f"{what} has no annotation, which is outside the subset; {rule}")
    if isinstance(annotation, ast.Name) and annotation.id in _SCALAR_ANNOTATIONS:
        return _SCALAR_ANNOTATIONS[annotation.id]
    if _annotated(annotation):
        return _array_annotation(annotation, module)
    raise Refuse(TYPE, annotation, f"the annotation {describe(annotation)} of {what} is outside the subset; {rule}")


def _annotated(annotation: ast.expr) -> bool:
    """Return True for an annotation Annotated[...]."""
    return isinstance(annotation, ast.Subscript) and is_name(annotation.value, "Annotated")


def _array_annotation(annotation: ast.expr, module: _ModuleReader) -> ArrayType:
    """Return the array type of `Annotated[memoryview, "<dtype>"]`; any other form raises Refuse."""
    module.require(
        "Annotated", annotation, TYPE, "Annotated is used but not imported from typing, which is outside the subset"
    )
    parts = annotation.slice if isinstance(annotation, ast.Subscript) else None
    if not (
        isinstance(parts, ast.Tuple) and len(parts.elts) == 2 and is_name(parts.elts[0], "memoryview")
        and is_str(parts.elts[1])
    ):
        raise Refuse(
            TYPE, annotation,
            f"the annotation {describe(annotation)} is outside the subset; an array is "
            'Annotated[memoryview, "<dtype>"]',
        )
    dtype = parts.elts[1].value
    if dtype not in ELEMENTS:
        raise Refuse(
            TYPE, parts.elts[1],
            f"the dtype {dtype!r} is outside the subset; its dtypes are f32, f64, i8, u8, i32, u32, and i64",
        )
    return ArrayType(dtype)


def _result_annotation(node: ast.FunctionDef) -> str | None:
    """Return a function's result type (None for -> None); a missing or other annotation raises Refuse."""
    rule = "a function returns int, float, bool, or None"
    returns = node.returns
    if returns is None:
        raise Refuse(TYPE, node, f"{node.name} has no return annotation, which is outside the subset; {rule}")
    if isinstance(returns, ast.Constant) and returns.value is None:
        return None
    if isinstance(returns, ast.Name) and returns.id in _SCALAR_ANNOTATIONS:
        return _SCALAR_ANNOTATIONS[returns.id]
    raise Refuse(
        TYPE, returns, f"the return annotation {describe(returns)} of {node.name} is outside the subset; {rule}"
    )


def _local_name(name: str, node: ast.AST, functions: frozenset[str]) -> None:
    """Raise Refuse for a local or parameter name that shadows a function of the module or a name the subset reads."""
    if name in RESERVED or name in functions:
        raise Refuse(
            TYPE, node,
            f"the local name {name} shadows a function or a name the subset reads, which is outside the subset",
        )


class _Event(NamedTuple):
    """One binding in a function: the name, where it stands, what it binds, and the loops around it."""

    name: str
    node: ast.AST  # the statement, or a for loop's target
    value: ast.expr | None  # the value bound; None for a loop variable
    annotation: ast.expr | None
    top: bool  # directly in the function body
    prange: ast.For | None  # the prange loop whose body holds it


def _single_target(stmt: ast.stmt) -> ast.expr | None:
    """Return the one target of an assignment, an annotated assignment with a value, or an augmented assignment."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        return stmt.targets[0]
    if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        return stmt.target
    if isinstance(stmt, ast.AugAssign):
        return stmt.target
    return None


def _events(stmts: list[ast.stmt], top: bool, prange: ast.For | None, out: list[_Event]) -> None:
    """Append the name bindings of `stmts`, in source order, descending into if, for, and while bodies."""
    for stmt in stmts:
        target = _single_target(stmt)
        if isinstance(target, ast.Name):
            annotation = stmt.annotation if isinstance(stmt, ast.AnnAssign) else None
            out.append(_Event(target.id, stmt, stmt.value, annotation, top, prange))
        elif isinstance(stmt, ast.For):
            inner = stmt if prange is None and is_prange_loop(stmt) else prange
            if isinstance(stmt.target, ast.Name):
                out.append(_Event(stmt.target.id, stmt.target, None, None, False, inner))
            _events(stmt.body, False, inner, out)
        elif isinstance(stmt, (ast.If, ast.While)):
            _events(stmt.body, False, prange, out)
            _events(stmt.orelse, False, prange, out)


class _Use(NamedTuple):
    """One occurrence of a name: the prange loop whose body holds it, and the for loops around it, outermost first.

    A for loop's target counts as inside that loop.
    """

    name: str
    owner: ast.For | None
    loops: tuple[ast.For, ...]


def _occurrences(stmts: list[ast.stmt], owner: ast.For | None, loops: tuple[ast.For, ...], out: list[_Use]) -> None:
    """Append each name a statement reads or binds, with the prange loop whose body holds it and the loops around."""
    for stmt in stmts:
        if isinstance(stmt, ast.For):
            inner = stmt if owner is None and is_prange_loop(stmt) else owner
            _names(stmt.iter, owner, loops, out)
            _names(stmt.target, inner, (*loops, stmt), out)
            _occurrences(stmt.body, inner, (*loops, stmt), out)
            _occurrences(stmt.orelse, owner, loops, out)
        elif isinstance(stmt, (ast.If, ast.While)):
            _names(stmt.test, owner, loops, out)
            _occurrences(stmt.body, owner, loops, out)
            _occurrences(stmt.orelse, owner, loops, out)
        else:
            _names(stmt, owner, loops, out)


def _names(node: ast.AST, owner: ast.For | None, loops: tuple[ast.For, ...], out: list[_Use]) -> None:
    """Append every name under `node` with `owner` and `loops`."""
    out.extend(_Use(child.id, owner, loops) for child in ast.walk(node) if isinstance(child, ast.Name))


def _covered(use: _Use, owners: list[ast.For]) -> bool:
    """Return True when a use outside the prange loops `owners` stands in a for loop that rebinds the name.

    That loop is not one of `owners` and holds none of them, so the use never sees a value a prange body bound.
    """
    for loop in use.loops:
        if not is_name(loop.target, use.name) or any(loop is owner for owner in owners):
            continue
        if not any(node is owner for node in ast.walk(loop) for owner in owners):
            return True
    return False


def _unbound_reads(stmts: list[ast.stmt], names: set[str], bound: set[str], out: list[ast.AST]) -> None:
    """Append each read of a name in `names` that no binding before it in the same iteration reaches.

    `bound` holds the names bound on every path so far and grows with each binding; a binding inside a loop body
    or in one branch of an if does not reach past it, and one in both branches does.
    """
    for stmt in stmts:
        if isinstance(stmt, ast.If):
            _reads(stmt.test, names, bound, out)
            then, other = set(bound), set(bound)
            _unbound_reads(stmt.body, names, then, out)
            _unbound_reads(stmt.orelse, names, other, out)
            bound |= then & other
        elif isinstance(stmt, (ast.For, ast.While)):
            _reads(stmt.iter if isinstance(stmt, ast.For) else stmt.test, names, bound, out)
            inner = set(bound)
            if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                inner.add(stmt.target.id)
            _unbound_reads(stmt.body, names, inner, out)
        else:
            target = _single_target(stmt)
            if isinstance(stmt, ast.AugAssign) and isinstance(target, ast.Name):
                if target.id in names and target.id not in bound:
                    out.append(target)
            _reads(stmt, names, bound, out)
            if isinstance(target, ast.Name):
                bound.add(target.id)


def _reads(node: ast.AST, names: set[str], bound: set[str], out: list[ast.AST]) -> None:
    """Append each name under `node` that is read, is in `names`, and is not in `bound`."""
    out.extend(
        child for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in names
        and child.id not in bound
    )


class _FunctionChecker:
    """Checks one function: the scope step, the type step, and the check step (see the module docstring)."""

    def __init__(self, module: _ModuleReader, node: ast.FunctionDef, signature: Signature) -> None:
        """Collect the function's bindings and its parameters' types."""
        self.module = module
        self.node = node
        self.signature = signature
        self.params: dict[str, ValueType] = dict(signature.params)
        self.events: list[_Event] = []
        _events(node.body, True, None, self.events)
        self.bound = {event.name for event in self.events}
        self.first: dict[str, _Event] = {}
        for event in self.events:
            self.first.setdefault(event.name, event)
        self.arrays: dict[str, ArrayType | None] = {n: t for n, t in signature.params if isinstance(t, ArrayType)}
        self.array_site: dict[str, ast.AST] = {}
        self.array_bindings: dict[ast.AST, ArrayBinding] = {}
        self.scalars: dict[str, str] = {n: t for n, t in signature.params if not isinstance(t, ArrayType)}
        self.poisoned: set[str] = set()
        self.shared: dict[str, ast.AST] = {}
        self.private: dict[ast.For, list[str]] = {}
        self.early: set[ast.AST] = set()  # reads of a prange body's private scalar before the iteration binds it
        self.typer = _Typer(self, settling=True)

    def check(self) -> FunctionInfo:
        """Run the three steps, appending each refusal to the module's, and return what the lowering reads."""
        self._find_arrays()
        self._settle()
        self._find_private()
        self._find_early_reads()
        assigned, entry = self._slots()
        self.typer = _Typer(self, settling=False)
        self._block(self.node.body, _Where(True, None), function_body=True)
        self._check_ending()
        return FunctionInfo(
            node=self.node, signature=self.signature, slots=dict(self.scalars), assigned=assigned, entry=entry,
            private={loop: tuple(names) for loop, names in self.private.items()},
            arrays={name: kind for name, kind in self.arrays.items() if kind is not None},
            array_bindings=self.array_bindings, types=self.typer.types, vector=frozenset(self.typer.vector_nodes),
        )

    # The scope step and the type step

    def _find_arrays(self) -> None:
        """Mark each name an array binding binds directly in the function body, with its type read leniently."""
        for event in self.events:
            if not event.top or event.value is None or isinstance(event.node, ast.AugAssign):
                continue
            if event.name in self.params or event.name in self.array_site:
                continue
            if self._array_shaped(event.value, event.annotation):
                self.arrays[event.name] = self._array_guess(event.value, event.annotation)
                self.array_site[event.name] = event.node

    def _array_shaped(self, value: ast.expr, annotation: ast.expr | None) -> bool:
        """Return True when a binding binds an array: an array annotation, an array form, or an array's name."""
        if annotation is not None and _annotated(annotation):
            return True
        if alloc_shaped(value) or read_shaped(value) or asarray_shaped(value):
            return True
        return isinstance(value, ast.Name) and value.id in self.arrays

    def _array_guess(self, value: ast.expr, annotation: ast.expr | None) -> ArrayType | None:
        """Return the array type a binding gives, read before the check step; None when it does not settle."""
        if annotation is not None and _annotated(annotation):
            try:
                return _array_annotation(annotation, self.module)
            except (Refuse, Skip):
                return None
        if alloc_shaped(value):
            fmt = value.args[0].value if len(value.args) == 1 and is_str(value.args[0]) else None
            return ArrayType(FORMATS[fmt][0]) if fmt in FORMATS else None
        if asarray_shaped(value):
            source = value.args[0] if len(value.args) == 1 else None
            return ArrayType("f64", numpy=True) if isinstance(source, ast.Name) and source.id in self.arrays else None
        if isinstance(value, ast.Name):
            return self.arrays.get(value.id)
        return None

    def _settle(self) -> None:
        """Settle each scalar name's type from its first binding whose type is known, until nothing changes."""
        pending = [event for event in self.events if event.name not in self.arrays]
        changed = True
        while changed:
            changed = False
            for event in pending:
                if event.name not in self.scalars:
                    found = self._event_type(event)
                    if found is not None:
                        self.scalars[event.name] = found
                        changed = True

    def _event_type(self, event: _Event) -> str | None:
        """Return the type a binding gives its name, or None while it cannot be read (an update never sets one)."""
        if event.value is None:
            return INT
        if isinstance(event.node, ast.AugAssign):
            return None
        if event.annotation is not None:
            annotation = event.annotation
            return _SCALAR_ANNOTATIONS.get(annotation.id) if isinstance(annotation, ast.Name) else None
        try:
            found = self.typer.expr(event.value)
        except (Refuse, Skip, _Unsettled):
            return None
        return found if found in (INT, FLOAT, BOOL) else None

    def _find_private(self) -> None:
        """Find the scalars each prange body binds: private to it, or shared when code outside it uses them.

        A use outside the body is no sharing when another for loop that rebinds the name holds it (_covered): the
        name then has a slot of its own outside the body, beside the private one.
        """
        owners: dict[str, list[ast.For]] = {}
        for event in self.events:
            if event.prange is not None and event.name not in self.arrays:
                found = owners.setdefault(event.name, [])
                if not any(loop is event.prange for loop in found):
                    found.append(event.prange)
        uses: list[_Use] = []
        _occurrences(self.node.body, None, (), uses)
        for name, loops in owners.items():
            outside = any(
                use.name == name and not any(loop is use.owner for loop in loops) and not _covered(use, loops)
                for use in uses
            )
            if outside or name in self.params:
                self.shared[name] = next(e.node for e in self.events if e.name == name and e.prange is not None)
        for event in self.events:
            if event.prange is None or event.name not in owners or event.name in self.shared:
                continue
            names = self.private.setdefault(event.prange, [])
            if event.name not in names:
                names.append(event.name)

    def _find_early_reads(self) -> None:
        """Mark each read of a prange body's private scalar that no binding earlier in the same iteration reaches."""
        found: list[ast.AST] = []
        for loop, names in self.private.items():
            bound = {loop.target.id} if isinstance(loop.target, ast.Name) else set()
            _unbound_reads(loop.body, set(names), bound, found)
        self.early = set(found)

    def _slots(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the assigned scalar parameters and the other scalar names whose slots open at function entry.

        A name private to a prange body opens at entry too when code outside every prange body binds it.
        """
        assigned = tuple(
            name for name, kind in self.signature.params if not isinstance(kind, ArrayType) and name in self.bound
        )
        private = {name for names in self.private.values() for name in names}
        outside = {event.name for event in self.events if event.prange is None}
        entry: list[str] = []
        for event in self.events:
            name = event.name
            if name in self.arrays or name in self.params or name in entry:
                continue
            if name in private and name not in outside:
                continue
            entry.append(name)
        return assigned, tuple(entry)

    def is_poisoned(self, name: str) -> bool:
        """Return True for a name whose binding was refused: here, or at module level when no local takes it."""
        if name in self.poisoned:
            return True
        return name in self.module.poisoned and name not in self.bound and name not in self.params

    # The check step: statements

    def _block(self, stmts: list[ast.stmt], where: _Where, function_body: bool = False) -> None:
        """Check each statement; a refused one gives its diagnostic, and the names it binds are poisoned."""
        for index, stmt in enumerate(stmts):
            if function_body and index == 0 and is_docstring(stmt):
                continue
            try:
                self._statement(stmt, where, function_body and index == len(stmts) - 1)
            except Refuse as refusal:
                self.module.refusals.append(refusal)
                self._poison(stmt)
            except Skip:
                self._poison(stmt)

    def _poison(self, stmt: ast.stmt) -> None:
        """Mark every name a refused statement binds, so its later uses give no diagnostic of their own."""
        self.poisoned |= bound_names(stmt)

    def _statement(self, stmt: ast.stmt, where: _Where, last: bool) -> None:
        """Check one statement of a function body."""
        if isinstance(stmt, ast.Pass):
            return
        handlers = {
            ast.Expr: self._expression_statement, ast.Assign: self._assign, ast.AnnAssign: self._annotated,
            ast.AugAssign: self._augmented, ast.If: self._if, ast.For: self._for, ast.While: self._while,
        }
        handler = handlers.get(type(stmt))
        if handler is not None:
            handler(stmt, where)
        elif isinstance(stmt, ast.Return):
            self._return(stmt, where, last)
        elif isinstance(stmt, (ast.Break, ast.Continue)):
            raise Refuse(CONTROL, stmt, f"{statement_phrase(stmt)} is outside the subset")
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            raise Refuse(IMPORT, stmt, "an import inside a function is outside the subset; imports stand at top level")
        else:
            raise Refuse(STATEMENT, stmt, f"{article(statement_phrase(stmt))} is outside the subset")

    def _expression_statement(self, stmt: ast.Expr, where: _Where) -> None:
        """Check a call used as a statement: write_array, or any call the subset holds."""
        value = stmt.value
        if not isinstance(value, ast.Call):
            raise Refuse(
                STATEMENT, stmt,
                f"the expression statement {describe(value)} is outside the subset; a statement expression is a call",
            )
        if is_named_call(value, "write_array") and not self.is_poisoned("write_array"):
            self._write(value, where)
        else:
            self.typer.call(value, statement=True)

    def _assign(self, stmt: ast.Assign, where: _Where) -> None:
        """Check `x = e` or `a[i] = e`."""
        if len(stmt.targets) != 1:
            raise Refuse(STATEMENT, stmt, "a chained assignment is outside the subset")
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            self._bind(target.id, stmt, stmt.value, None, where)
        elif isinstance(target, ast.Subscript):
            self._store(target, stmt.value)
        else:
            raise Refuse(
                STATEMENT, target,
                f"an assignment to {describe(target)} is outside the subset; a target is a name or a[i]",
            )

    def _annotated(self, stmt: ast.AnnAssign, where: _Where) -> None:
        """Check `x: T = e`."""
        if not isinstance(stmt.target, ast.Name):
            raise Refuse(STATEMENT, stmt, f"an annotated assignment to {describe(stmt.target)} is outside the subset")
        if stmt.value is None:
            raise Refuse(STATEMENT, stmt, f"the annotation of {stmt.target.id} with no value is outside the subset")
        self._bind(stmt.target.id, stmt, stmt.value, stmt.annotation, where)

    def _bind(self, name: str, stmt: ast.stmt, value: ast.expr, annotation: ast.expr | None, where: _Where) -> None:
        """Check a binding of a name: an array binding, or a scalar that keeps one type."""
        _local_name(name, stmt, self.module.function_names)
        if name in self.arrays:
            self._bind_array(name, stmt, value, annotation)
            return
        if self._array_shaped(value, annotation):
            if isinstance(value, ast.Name) and value.id in self.arrays:
                self.typer.array(value)  # refuses a use of the array before its binding, the likelier cause
            raise Refuse(
                ARRAY, stmt,
                f"binding an array to {name} here is outside the subset; an array is bound once, directly in the "
                "function body (not in a loop or a branch), to a name that is not a parameter",
            )
        found = self.typer.expr(value)
        want = self._scalar(name, stmt)
        if annotation is not None:
            declared = _SCALAR_ANNOTATIONS.get(annotation.id) if isinstance(annotation, ast.Name) else None
            if declared != found:
                raise Refuse(
                    TYPE, stmt,
                    f"{name} is annotated {describe(annotation)} and bound to {word(found)}, which is outside the "
                    "subset",
                )
        if found != want:
            raise Refuse(
                TYPE, stmt,
                f"{name} is bound to {word(found)} here and to {word(want)} at its first binding; a name keeps one "
                "type in the subset",
            )
        self._private(name, stmt, where)

    def _scalar(self, name: str, node: ast.AST) -> str:
        """Return a scalar name's settled type; Skip for a poisoned name, else Refuse."""
        if name in self.scalars:
            return self.scalars[name]
        if self.is_poisoned(name):
            raise Skip
        raise Refuse(TYPE, node, f"the type of {name} does not settle from its bindings, which is outside the subset")

    def _private(self, name: str, node: ast.AST, where: _Where) -> None:
        """Raise Refuse at the first binding in a prange body of a scalar that code outside the body also uses."""
        if where.prange is not None and self.shared.get(name) is node:
            raise Refuse(
                PRANGE, node,
                f"{name} is bound in a prange body and also used outside it; a scalar shared across prange "
                "iterations is outside the subset",
            )

    def _bind_array(self, name: str, stmt: ast.stmt, value: ast.expr, annotation: ast.expr | None) -> None:
        """Check the one binding of an array name and record how it binds."""
        if name in self.params:
            raise Refuse(ARRAY, stmt, f"a binding of the array parameter {name} is outside the subset")
        if self.array_site.get(name) is not stmt:
            raise Refuse(ARRAY, stmt, f"a second binding of the array {name} is outside the subset; arrays bind once")
        binding = self._array_value(value, annotation)
        self.arrays[name] = binding.array
        self.array_bindings[stmt] = binding

    def _array_value(self, value: ast.expr, annotation: ast.expr | None) -> ArrayBinding:
        """Return how an array binding binds: an allocation, an input read, np.asarray, or another array's name."""
        if read_shaped(value):
            if annotation is None or not _annotated(annotation):
                raise Refuse(
                    IO, value,
                    'read_array(...).data needs the annotation Annotated[memoryview, "<dtype>"] on its name; '
                    "without it the read is outside the subset",
                )
            array = _array_annotation(annotation, self.module)
            return ArrayBinding("read", array, self._read(value))
        if alloc_shaped(value):
            binding = ArrayBinding("alloc", *self._alloc(value))
        elif asarray_shaped(value):
            binding = ArrayBinding("view", self._asarray(value), value.args[0])
        elif isinstance(value, ast.Name) and value.id in self.arrays:
            binding = ArrayBinding("view", self.typer.array(value), value)
        else:
            raise Refuse(ARRAY, value, f"the array value {describe(value)} is outside the subset")
        if annotation is not None:
            declared = _array_annotation(annotation, self.module) if _annotated(annotation) else None
            if declared != binding.array:
                raise Refuse(
                    TYPE, annotation,
                    f"the annotation {describe(annotation)} does not name the array's type, which is outside the "
                    "subset",
                )
        return binding

    def _read(self, value: ast.Attribute) -> int:
        """Check `read_array(sys.argv[<k>]).data` and return k."""
        call = value.value
        if self.is_poisoned("read_array"):
            raise Skip
        self.module.require(
            "read_array", call, IO, "read_array is used but not imported from lassi_io, which is outside the subset"
        )
        if call.keywords or len(call.args) != 1:
            raise Refuse(IO, call, "read_array takes one argument, sys.argv[<k>], in the subset")
        return self._argv(call.args[0], call, "read_array")

    def _argv(self, node: ast.expr, call: ast.Call, what: str) -> int:
        """Return k for `sys.argv[<k>]`, k a positive int literal that fits in 64 bits; else Refuse at the call."""
        index = argv_index(node)
        if index is None or not 1 <= index <= _INT_RANGE[1]:
            raise Refuse(
                IO, call,
                f"{what} with the path {describe(node)} is outside the subset; a path is sys.argv[<k>], k a positive "
                "int literal that fits in 64 bits",
            )
        self.module.require("sys", call, IO, "sys is used but not imported, which is outside the subset")
        return index

    def _alloc(self, value: ast.Call) -> tuple[ArrayType, ast.expr]:
        """Check `memoryview(bytearray(<count> * <itemsize>)).cast("<format>")`; return the array and the count."""
        form = 'memoryview(bytearray(<count> * <itemsize>)).cast("<format>")'
        view = value.func.value
        inner = view.args[0] if len(view.args) == 1 and not view.keywords else None
        product = None
        if is_named_call(inner, "bytearray") and len(inner.args) == 1 and not inner.keywords:
            product = inner.args[0]
        cast = not value.keywords and len(value.args) == 1 and is_str(value.args[0])
        if not cast or not (isinstance(product, ast.BinOp) and isinstance(product.op, ast.Mult)):
            raise Refuse(ARRAY, value, f"an allocation outside the form {form} is outside the subset")
        fmt = value.args[0].value
        if fmt not in FORMATS:
            raise Refuse(
                ARRAY, value, f"the format {fmt!r} is outside the subset; the formats are f, d, b, B, i, I, and q"
            )
        dtype, size = FORMATS[fmt]
        if not (is_int_literal(product.right) and product.right.value == size):
            raise Refuse(ARRAY, product, f"an allocation of format {fmt!r} is <count> * {size} in the subset")
        if self.typer.expr(product.left) != INT:
            raise Refuse(TYPE, product.left, "the count of an allocation is an int in the subset")
        return ArrayType(dtype), product.left

    def _asarray(self, value: ast.Call) -> ArrayType:
        """Check `np.asarray(<view>)` of an f64 array and return the ndarray's type."""
        if self.is_poisoned("np"):
            raise Skip
        self.module.require(
            "np", value, CALL, "np is used but not imported (import numpy as np), which is outside the subset"
        )
        if value.keywords or len(value.args) != 1 or not isinstance(value.args[0], ast.Name):
            raise Refuse(ARRAY, value, "np.asarray takes one array name in the subset")
        source = self.typer.array(value.args[0])
        if source.dtype != "f64":
            raise Refuse(ARRAY, value, f"np.asarray of a {source.dtype} array is outside the subset; ndarrays are f64")
        return ArrayType("f64", numpy=True)

    def _store(self, target: ast.Subscript, value: ast.expr) -> None:
        """Check `a[i] = e`, or `z[:] = <expression>` over ndarrays."""
        array = self._subscript_array(target)
        if isinstance(target.slice, ast.Slice):
            self._vector_store(target, array, value)
            return
        found = self.typer.expr(value)
        self.typer.index(target)
        self._storable(found, array, target)

    def _subscript_array(self, target: ast.Subscript) -> ArrayType:
        """Return the type of the array a subscript reads or stores; anything but an array name raises Refuse."""
        return self.typer.array(target.value, target)

    def _storable(self, found: str, array: ArrayType, node: ast.AST) -> None:
        """Raise Refuse for a value an array element cannot take: a bool, or a float into an int array."""
        if found not in (INT, FLOAT):
            raise Refuse(TYPE, node, f"storing {word(found)} into an array is outside the subset")
        if array.scalar == INT and found == FLOAT:
            raise Refuse(TYPE, node, f"storing a float into an {array.dtype} array is outside the subset")

    def _vector_store(self, target: ast.Subscript, array: ArrayType, value: ast.expr) -> None:
        """Check `z[:] = <expression>`: z an f64 ndarray, the expression over ndarrays, scalars, + - * /, np.sqrt."""
        part = target.slice
        if part.lower is not None or part.upper is not None or part.step is not None:
            raise Refuse(ARRAY, target, "a slice other than [:] is outside the subset")
        if not array.numpy:
            raise Refuse(
                ARRAY, target,
                f"z[:] = <expression> stores into an ndarray (np.asarray of an f64 view), and {target.value.id} is a "
                "view, which is outside the subset",
            )
        found = self.typer.vector(value)
        if found not in (INT, FLOAT):
            raise Refuse(TYPE, value, f"storing {word(found)} into an array is outside the subset")

    def _augmented(self, stmt: ast.AugAssign, where: _Where) -> None:
        """Check `x op= e` or `a[i] op= e`; the result keeps the target's type."""
        target = stmt.target
        if isinstance(target, ast.Subscript):
            array = self._subscript_array(target)
            if isinstance(target.slice, ast.Slice):
                raise Refuse(ARRAY, target, "an augmented assignment to a slice is outside the subset")
            self.typer.index(target)
            result = self.typer.binop(stmt.op, array.scalar, self.typer.expr(stmt.value), stmt)
            self._storable(result, array, stmt)
            return
        if not isinstance(target, ast.Name):
            raise Refuse(STATEMENT, stmt, f"an augmented assignment to {describe(target)} is outside the subset")
        name = target.id
        _local_name(name, stmt, self.module.function_names)
        if name in self.arrays:
            raise Refuse(ARRAY, stmt, f"an augmented assignment to the array {name} is outside the subset")
        if name not in self.params and self.first[name].node is stmt:
            raise Refuse(TYPE, stmt, f"{name} is updated before it is bound, which is outside the subset")
        if target in self.early:
            raise Refuse(PRANGE, target, early_message(name))
        want = self._scalar(name, stmt)
        result = self.typer.binop(stmt.op, want, self.typer.expr(stmt.value), stmt)
        if result != want:
            raise Refuse(
                TYPE, stmt,
                f"{name} {BINARY[type(stmt.op)]}= gives {word(result)}, but {name} is {word(want)}; a name keeps one "
                "type in the subset",
            )
        self._private(name, stmt, where)

    def _if(self, stmt: ast.If, where: _Where) -> None:
        """Check an if statement and its branches."""
        self._condition(stmt.test, "an if statement")
        inner = where._replace(top=False)
        self._block(stmt.body, inner)
        self._block(stmt.orelse, inner)

    def _while(self, stmt: ast.While, where: _Where) -> None:
        """Check a while loop and its body."""
        if stmt.orelse:
            raise Refuse(CONTROL, stmt, "a while loop with an else clause is outside the subset")
        self._condition(stmt.test, "a while loop")
        self._block(stmt.body, where._replace(top=False))

    def _condition(self, test: ast.expr, what: str) -> None:
        """Raise Refuse unless the condition of `what` is a bool."""
        found = self.typer.expr(test)
        if found != BOOL:
            raise Refuse(TYPE, test, f"the condition of {what} is {word(found)}, not a bool, outside the subset")

    def _for(self, stmt: ast.For, where: _Where) -> None:
        """Check a loop over range(...) or prange(...) and its body."""
        if stmt.orelse:
            raise Refuse(CONTROL, stmt, "a for loop with an else clause is outside the subset")
        if not isinstance(stmt.target, ast.Name):
            raise Refuse(CONTROL, stmt.target, "a for loop that binds anything but one name is outside the subset")
        call = stmt.iter
        if not (is_named_call(call, "range") or is_named_call(call, "prange")):
            raise Refuse(
                CONTROL, call,
                f"a loop over {describe(call)} is outside the subset; loops run over range or prange",
            )
        parallel = is_named_call(call, "prange")
        if parallel and self.is_poisoned("prange"):
            raise Skip
        if parallel:
            self.module.require(
                "prange", call, PRANGE, "prange is used but not imported from lassi_io, which is outside the subset"
            )
        if parallel and where.prange is not None:
            raise Refuse(PRANGE, stmt, "a prange loop inside a prange body is outside the subset")
        self._range_arguments(call)
        name = stmt.target.id
        _local_name(name, stmt.target, self.module.function_names)
        if name in self.arrays:
            raise Refuse(ARRAY, stmt.target, f"the array {name} as a loop variable is outside the subset")
        if self._scalar(name, stmt.target) != INT:
            raise Refuse(
                TYPE, stmt.target,
                f"the loop variable {name} is bound to {word(self.scalars[name])} elsewhere; a name keeps one type",
            )
        inner = _Where(False, stmt if parallel else where.prange)
        self._private(name, stmt.target, inner)
        self._block(stmt.body, inner)

    def _range_arguments(self, call: ast.Call) -> None:
        """Check range's or prange's arguments: int start and stop, and a positive int literal step."""
        kind = call.func.id
        if call.keywords or not 1 <= len(call.args) <= 3:
            raise Refuse(CONTROL, call, f"{kind} with other than one to three arguments is outside the subset")
        for bound in call.args[:2]:
            if self.typer.expr(bound) != INT:
                raise Refuse(TYPE, bound, f"a {kind} bound that is not an int is outside the subset")
        if len(call.args) == 3:
            step = call.args[2]
            if not (is_int_literal(step) and step.value > 0):
                raise Refuse(CONTROL, step, f"a {kind} step that is not a positive int literal is outside the subset")
            self.typer.expr(step)  # an int literal past 64 bits is a refusal, as everywhere

    def _return(self, stmt: ast.Return, where: _Where, last: bool) -> None:
        """Check a return: only as the function's last statement, giving the function's result type."""
        if not (last and where.top):
            raise Refuse(CONTROL, stmt, "an early return is outside the subset; return is a function's last statement")
        name, want = self.signature.name, self.signature.result
        if want is None:
            if stmt.value is not None:
                raise Refuse(TYPE, stmt, f"{name} returns None, so a return with a value is outside the subset")
            return
        if stmt.value is None:
            raise Refuse(TYPE, stmt, f"{name} returns {word(want)}, so a return with no value is outside the subset")
        found = self.typer.expr(stmt.value)
        if not (found == want or (found == INT and want == FLOAT)):
            raise Refuse(TYPE, stmt, f"{name} returns {word(want)}, and returning {word(found)} is outside the subset")

    def _check_ending(self) -> None:
        """Refuse a function with a result whose last statement is not a return."""
        body = self.node.body
        if self.signature.result is not None and not isinstance(body[-1], ast.Return):
            self.module.refusals.append(
                Refuse(
                    CONTROL, self.node,
                    f"{self.signature.name} returns {word(self.signature.result)} but does not end with a return, "
                    "which is outside the subset",
                )
            )

    def _write(self, call: ast.Call, where: _Where) -> None:
        """Check `write_array(sys.argv[<k>], "<name>", "<dtype>", (<dims>,), <view>)`, outside every prange body."""
        self.module.require(
            "write_array", call, IO, "write_array is used but not imported from lassi_io, which is outside the subset"
        )
        if where.prange is not None:
            raise Refuse(
                PRANGE, call,
                "write_array in a prange body is outside the subset; an output is written outside parallel loops",
            )
        if call.keywords or len(call.args) != 5:
            raise Refuse(
                IO, call,
                "write_array takes five positional arguments in the subset: sys.argv[<k>], a name, a dtype, a shape "
                "tuple, and an array",
            )
        path, name, dtype, shape, data = call.args
        self._argv(path, call, "write_array")
        if not (is_str(name) and _OUTPUT_NAME.fullmatch(name.value)):
            raise Refuse(
                IO, call, "write_array's name is a string literal of 1 to 255 ASCII letters, digits, '_', '.', or '-'"
            )
        if not (is_str(dtype) and dtype.value in ELEMENTS):
            raise Refuse(IO, call, "write_array's dtype is a string literal naming one of the subset's dtypes")
        if not isinstance(shape, ast.Tuple):
            raise Refuse(IO, call, "write_array's shape is a tuple of ints in the subset")
        for dim in shape.elts:
            if self.typer.expr(dim) != INT:
                raise Refuse(TYPE, dim, "a write_array dimension that is not an int is outside the subset")
        if not isinstance(data, ast.Name):
            raise Refuse(IO, call, "write_array's data is an array name in the subset")
        array = self.typer.array(data)
        if array.numpy:
            raise Refuse(IO, call, "write_array of an ndarray is outside the subset; pass the view it was made from")
        if array.dtype != dtype.value:
            raise Refuse(IO, call, f"write_array names dtype {dtype.value} for an {array.dtype} array")


class _Typer:
    """Types the expressions of one function, refusing what the subset does not hold, and records each type."""

    def __init__(self, scope: _FunctionChecker, settling: bool) -> None:
        """Type in `scope`; while settling, a name whose type is not settled yet raises _Unsettled."""
        self.scope = scope
        self.settling = settling
        self.types: dict[ast.AST, ValueType] = {}
        self.vector_nodes: set[ast.AST] = set()
        self.seen_ndarray = False  # in a vectorized expression, whether an ndarray came before, in evaluation order

    def expr(self, node: ast.expr) -> str:
        """Return a scalar expression's type and record it; anything outside the subset raises Refuse."""
        found = self._kind(node)
        self.types[node] = found
        return found

    def _kind(self, node: ast.expr) -> str:
        """Return the type of one expression node by its kind."""
        if isinstance(node, ast.Constant):
            return self._constant(node)
        if isinstance(node, ast.Name):
            return self._name(node)
        if isinstance(node, ast.UnaryOp):
            return self._unary(node)
        if isinstance(node, ast.BinOp):
            left = self.expr(node.left)
            return self.binop(node.op, left, self.expr(node.right), node)
        if isinstance(node, ast.BoolOp):
            return self._boolop(node)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.IfExp):
            return self._ifexp(node)
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, ast.Subscript):
            return self._load(node)
        if isinstance(node, ast.Attribute):
            return self._attribute(node)
        phrase = _EXPRESSION_WORDS.get(type(node), f"a {type(node).__name__.lower()}")
        raise Refuse(EXPRESSION, node, f"{phrase} ({describe(node)}) is outside the subset")

    def _constant(self, node: ast.Constant) -> str:
        """Return the type of a literal: a bool, an int that fits in 64 bits, or a finite float."""
        value = node.value
        if isinstance(value, bool):
            return BOOL
        if isinstance(value, (int, float)):
            return self._number(value, node)
        raise Refuse(EXPRESSION, node, f"the constant {describe(node)} is outside the subset")

    def _number(self, value: int | float, node: ast.AST) -> str:
        """Return the type of a number literal; one past 64 bits or not finite raises Refuse."""
        if isinstance(value, int):
            if _INT_RANGE[0] <= value <= _INT_RANGE[1]:
                return INT
            raise Refuse(TYPE, node, "an int literal that does not fit in 64 bits is outside the subset")
        if math.isfinite(value):
            return FLOAT
        raise Refuse(EXPRESSION, node, "a float literal that is not finite is outside the subset")

    def _name(self, node: ast.Name) -> str:
        """Return a scalar name's type: a parameter or a settled local."""
        scope, name = self.scope, node.id
        if name in scope.arrays:
            raise Refuse(ARRAY, node, f"the array {name} used as a scalar is outside the subset")
        if node in scope.early:
            raise Refuse(PRANGE, node, early_message(name))
        if name in scope.scalars:
            return scope.scalars[name]
        if name in scope.bound:
            if self.settling:
                raise _Unsettled(name)
            if scope.is_poisoned(name):
                raise Skip
            raise Refuse(TYPE, node, f"the type of {name} does not settle from its bindings, outside the subset")
        if scope.is_poisoned(name):
            raise Skip
        raise Refuse(EXPRESSION, node, f"the name {name} is not a local or a parameter, which is outside the subset")

    def _unary(self, node: ast.UnaryOp) -> str:
        """Return the type of -x, +x, not b, or ~n; a minus on a number literal is part of the literal."""
        literal = folded_literal(node)
        if literal is not None:
            return self._number(literal, node)
        operand = self.expr(node.operand)
        op = node.op
        if isinstance(op, (ast.USub, ast.UAdd)) and operand in (INT, FLOAT):
            return operand
        if isinstance(op, ast.Not) and operand == BOOL:
            return BOOL
        if isinstance(op, ast.Invert) and operand == INT:
            return INT
        raise Refuse(TYPE, node, f"the operator {_UNARY[type(op)]} on {word(operand)} is outside the subset")

    def binop(self, op: ast.operator, left: str, right: str, node: ast.AST) -> str:
        """Return the type of a binary operation on two numbers; any other operation raises Refuse at `node`."""
        symbol = BINARY[type(op)]
        if isinstance(op, ast.Pow):
            raise Refuse(EXPRESSION, node, f"the power operator {symbol} is outside the subset")
        if isinstance(op, ast.MatMult):
            raise Refuse(EXPRESSION, node, f"the matrix operator {symbol} is outside the subset")
        for side in (left, right):
            if side not in (INT, FLOAT):
                raise Refuse(TYPE, node, f"the operator {symbol} on {word(side)} is outside the subset")
        if isinstance(op, _INT_ONLY) and FLOAT in (left, right):
            raise Refuse(TYPE, node, f"the operator {symbol} on a float is outside the subset; it takes ints")
        return binop_result(op, left, right)

    def _boolop(self, node: ast.BoolOp) -> str:
        """Return the type of `a and b` or `a or b` over bools."""
        for value in node.values:
            found = self.expr(value)
            if found != BOOL:
                keyword = "and" if isinstance(node.op, ast.And) else "or"
                raise Refuse(TYPE, value, f"{keyword} over {word(found)} is outside the subset; it takes bools")
        return BOOL

    def _compare(self, node: ast.Compare) -> str:
        """Return the type of one comparison of two numbers."""
        if len(node.ops) != 1:
            raise Refuse(EXPRESSION, node, "a chained comparison is outside the subset")
        op = node.ops[0]
        if type(op) not in COMPARISONS:
            symbol = _OTHER_COMPARISONS[type(op)]
            raise Refuse(EXPRESSION, node, f"the comparison operator '{symbol}' is outside the subset")
        left = self.expr(node.left)
        right = self.expr(node.comparators[0])
        for side in (left, right):
            if side not in (INT, FLOAT):
                raise Refuse(TYPE, node, f"a comparison of {word(side)} is outside the subset; it compares numbers")
        return BOOL

    def _ifexp(self, node: ast.IfExp) -> str:
        """Return the type of `x if c else y`: c a bool, both arms of one type."""
        test = self.expr(node.test)
        if test != BOOL:
            raise Refuse(TYPE, node.test, f"a conditional expression on {word(test)} is outside the subset")
        body = self.expr(node.body)
        other = self.expr(node.orelse)
        if body != other:
            raise Refuse(
                TYPE, node,
                f"a conditional expression giving {word(body)} or {word(other)} is outside the subset; both arms "
                "give one type",
            )
        return body

    def _load(self, node: ast.Subscript) -> str:
        """Return the type of an element load a[i]: the Python scalar the element reads as."""
        if argv_index(node) is not None or (isinstance(node.value, ast.Attribute) and node.value.attr == "argv"):
            raise Refuse(IO, node, "sys.argv outside read_array and write_array is outside the subset")
        array = self.array(node.value, node)
        self.index(node)
        return array.scalar

    def index(self, node: ast.Subscript) -> None:
        """Check a subscript's index: one int expression, not a negative literal."""
        index = node.slice
        if isinstance(index, ast.Tuple):
            raise Refuse(
                ARRAY, node,
                f"a subscript with {len(index.elts)} indices is outside the subset; an array takes one int index",
            )
        if isinstance(index, ast.Slice):
            raise Refuse(ARRAY, node, "a slice is outside the subset, but in z[:] = <expression> over ndarrays")
        literal = folded_literal(index)
        if isinstance(literal, int) and literal < 0:
            raise Refuse(ARRAY, node, "a negative index is outside the subset")
        if self.expr(index) != INT:
            raise Refuse(ARRAY, node, "an index that is not an int is outside the subset")

    def _attribute(self, node: ast.Attribute) -> str:
        """Refuse an attribute read: sys.argv and read_array(...).data stand only in their forms."""
        root = root_name(node)
        if root is not None and self.scope.is_poisoned(root):
            raise Skip
        if is_name(node.value, "sys") and node.attr == "argv":
            raise Refuse(IO, node, "sys.argv outside read_array and write_array is outside the subset")
        if read_shaped(node):
            raise Refuse(IO, node, "read_array(...).data outside an annotated array binding is outside the subset")
        raise Refuse(EXPRESSION, node, f"the attribute {describe(node)} is outside the subset")

    def array(self, node: ast.expr, subscript: ast.Subscript | None = None) -> ArrayType:
        """Return the type of an array name used as an array, after its binding; anything else raises Refuse.

        `subscript`, when given, is the subscript whose base `node` is, where a refusal then stands. A poisoned
        name gives Skip, since its refused binding was counted already.
        """
        scope = self.scope
        if not (isinstance(node, ast.Name) and node.id in scope.arrays):
            if isinstance(node, ast.Name) and scope.is_poisoned(node.id) and node.id not in scope.scalars:
                raise Skip
            if subscript is not None:
                raise Refuse(
                    ARRAY, subscript, f"a subscript of {describe(node)}, which is not an array, is outside the subset"
                )
            raise Refuse(ARRAY, node, f"{describe(node)} is not an array name, which is outside the subset here")
        found = scope.arrays[node.id]
        if found is None:
            raise Skip
        site = scope.array_site.get(node.id)
        if site is not None and _before(node, site):
            raise Refuse(ARRAY, node, f"the array {node.id} used before its binding is outside the subset")
        self.types[node] = found
        return found

    def call(self, node: ast.Call, statement: bool = False) -> str:
        """Return the type of a call the subset holds (NONE for a function returning None, as a statement only)."""
        if alloc_shaped(node):
            raise Refuse(ARRAY, node, "an allocation outside an array binding is outside the subset")
        if asarray_shaped(node):
            raise Refuse(ARRAY, node, "np.asarray outside an array binding is outside the subset")
        callee = describe(node.func)
        if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
            raise Refuse(CALL, node, f"a call of {callee} with keyword or starred arguments is outside the subset")
        func = node.func
        module = self.scope.module
        root = root_name(func)
        if root is not None and self.scope.is_poisoned(root):
            raise Skip
        if isinstance(func, ast.Name):
            if func.id in module.signatures:
                return self._function(node, module.signatures[func.id], statement)
            if func.id in _BUILTINS:
                return self._builtin(node, func.id)
            self._misplaced(node, func.id)
        elif isinstance(func, ast.Attribute) and is_name(func.value, "math") and func.attr in MATH_FUNCTIONS:
            return self._math(node)
        elif np_sqrt_shaped(node):
            raise Refuse(
                CALL, node,
                "np.sqrt of a scalar is outside the subset; np.sqrt takes a part that holds an ndarray, in "
                "z[:] = <expression>, and math.sqrt takes a scalar",
            )
        raise Refuse(
            CALL, node,
            f"a call of {callee} is outside the subset; the subset calls the module's functions, len, abs, float, "
            "int, and math.sqrt, exp, log, sin, and cos",
        )

    def _misplaced(self, node: ast.Call, name: str) -> None:
        """Raise Refuse for a call of a name the subset reads only in its own forms: the binding's calls and range."""
        if name in ("read_array", "write_array"):
            raise Refuse(IO, node, f"{name} outside its form is outside the subset")
        if name in ("range", "prange"):
            raise Refuse(CONTROL, node, f"{name} outside a for loop header is outside the subset")
        if name in ("memoryview", "bytearray"):
            raise Refuse(ARRAY, node, f"{name} outside an allocation is outside the subset")

    def _function(self, node: ast.Call, signature: Signature, statement: bool) -> str:
        """Return the result type of a call of a module function, checking each argument against its parameter."""
        name = signature.name
        if len(node.args) != len(signature.params):
            raise Refuse(
                CALL, node,
                f"{name} takes {len(signature.params)} arguments; a call with {len(node.args)} is outside the subset",
            )
        for arg, (param, kind) in zip(node.args, signature.params, strict=True):
            if isinstance(kind, ArrayType):
                self._array_argument(arg, param, kind, name)
                continue
            found = self.expr(arg)
            if not (found == kind or (found == INT and kind == FLOAT)):
                raise Refuse(
                    TYPE, arg,
                    f"the parameter {param} of {name} takes {word(kind)}, and passing {word(found)} is outside the "
                    "subset",
                )
        if signature.result is None:
            if not statement:
                raise Refuse(TYPE, node, f"{name} returns None, so using its call as a value is outside the subset")
            return NONE
        return signature.result

    def _array_argument(self, arg: ast.expr, param: str, kind: ArrayType, name: str) -> None:
        """Check an array argument: a view's name of the parameter's dtype."""
        array = self.array(arg)
        if array.numpy:
            raise Refuse(
                ARRAY, arg, f"passing an ndarray to {name} is outside the subset; pass the view it was made from"
            )
        if array.dtype != kind.dtype:
            raise Refuse(
                TYPE, arg,
                f"the parameter {param} of {name} takes an {kind.dtype} array, and passing an {array.dtype} array is "
                "outside the subset",
            )

    def _builtin(self, node: ast.Call, name: str) -> str:
        """Return the type of len(a), abs(x), float(x), or int(x)."""
        if len(node.args) != 1:
            raise Refuse(CALL, node, f"{name} with other than one argument is outside the subset")
        arg = node.args[0]
        if name == "len":
            self.array(arg)
            return INT
        found = self.expr(arg)
        if found not in (INT, FLOAT):
            raise Refuse(TYPE, node, f"{name} of {word(found)} is outside the subset")
        return {"abs": found, "float": FLOAT, "int": INT}[name]

    def _math(self, node: ast.Call) -> str:
        """Return the type of math.sqrt, exp, log, sin, or cos of a number: a float."""
        self.scope.module.require("math", node, CALL, "math is used but not imported, which is outside the subset")
        if len(node.args) != 1:
            raise Refuse(CALL, node, f"{describe(node.func)} with other than one argument is outside the subset")
        found = self.expr(node.args[0])
        if found not in (INT, FLOAT):
            raise Refuse(TYPE, node, f"{describe(node.func)} of {word(found)} is outside the subset")
        return FLOAT

    def vector(self, node: ast.expr) -> str:
        """Return the element type of a vectorized expression, marking each node that holds an ndarray.

        A part that holds no ndarray is a scalar expression, typed as one; a part that holds one is an ndarray
        name, + - * / of two parts, or np.sqrt of a part. A scalar part after the first ndarray, in evaluation
        order, calls no module function that takes an array (_after_ndarray).
        """
        self.seen_ndarray = False
        return self._vector(node)

    def _vector(self, node: ast.expr) -> str:
        """Return the element type of one part of a vectorized expression (see vector)."""
        if not self._holds_array(node):
            found = self.expr(node)
            if self.seen_ndarray:
                self._after_ndarray(node)
            return found
        self.vector_nodes.add(node)
        if isinstance(node, ast.Name):
            if not self.array(node).numpy:
                raise Refuse(
                    ARRAY, node,
                    f"the view {node.id} in a vectorized expression is outside the subset; take np.asarray of it",
                )
            self.seen_ndarray = True
            found = FLOAT
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _VECTOR_OPERATORS):
                symbol = BINARY[type(node.op)]
                raise Refuse(ARRAY, node, f"the operator {symbol} in a vectorized expression is outside the subset")
            left = self._vector(node.left)
            found = self.binop(node.op, left, self._vector(node.right), node)
        else:
            self.scope.module.require(
                "np", node, CALL, "np is used but not imported (import numpy as np), outside the subset"
            )
            if node.keywords or len(node.args) != 1:
                raise Refuse(CALL, node, "np.sqrt with other than one argument is outside the subset")
            self._vector(node.args[0])
            found = FLOAT
        self.types[node] = found
        return found

    def _after_ndarray(self, node: ast.expr) -> None:
        """Refuse a call of a module function that takes an array in a scalar part after an ndarray.

        NumPy reads that ndarray before the call, which may write it, while the hub module evaluates every scalar
        part before its loop reads any ndarray.
        """
        for child in ast.walk(node):
            if not (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)):
                continue
            signature = self.scope.module.signatures.get(child.func.id)
            if signature is not None and any(isinstance(kind, ArrayType) for _, kind in signature.params):
                raise Refuse(
                    ARRAY, child,
                    f"a call of {child.func.id}, which takes an array, after an ndarray in a vectorized expression is "
                    "outside the subset; NumPy reads that ndarray before the call, and the hub module after it",
                )

    def _holds_array(self, node: ast.expr) -> bool:
        """Return True when a vectorized expression's part holds an array name through + - * / and np.sqrt."""
        if isinstance(node, ast.Name):
            return node.id in self.scope.arrays
        if isinstance(node, ast.BinOp):
            return self._holds_array(node.left) or self._holds_array(node.right)
        if np_sqrt_shaped(node):
            return any(self._holds_array(arg) for arg in node.args)
        return False
