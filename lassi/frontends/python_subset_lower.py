"""The Python-subset frontend's lowering: a checked module to hub text (task P12.4).

lower_module(info) writes the hub module of a ModuleInfo whose check gave no diagnostic
(lassi.frontends.python_subset_check), following the scheme lassi.frontends.python_subset's docstring publishes:
MLIR in custom assembly at LLVM 18.0.0git's syntax, with builtin's module and the func, arith, math, scf, and
memref dialects only. It reads the source's syntax tree and the types the check recorded, and runs nothing.

The text is built line by line: two blanks per level, one op per line, LF line ends. Values are numbered per
function in the order their definitions appear in its text (parameters are %arg0, %arg1, ...), so a region's
result is numbered before the ops inside the region. Constants are written where they are used and never shared.
No alloca grows the stack per loop iteration: a prange body's private slots, and a write's shape buffer inside a
loop body, stand in a memref.alloca_scope. Every memref.alloc is freed with memref.dealloc before the return.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from typing import Iterator

from lassi.frontends.python_subset_check import (
    BOOL,
    COMPARISONS,
    FLOAT,
    INT,
    ArrayBinding,
    ArrayType,
    FunctionInfo,
    ModuleInfo,
    Signature,
    argv_index,
    binop_result,
    folded_literal,
    is_docstring,
    is_named_call,
    mlir_type,
)

_FLOAT_OPS = {ast.Add: "addf", ast.Sub: "subf", ast.Mult: "mulf", ast.Div: "divf"}
_INT_OPS = {
    ast.Add: "addi", ast.Sub: "subi", ast.Mult: "muli", ast.FloorDiv: "floordivsi", ast.BitAnd: "andi",
    ast.BitOr: "ori", ast.BitXor: "xori", ast.LShift: "shli", ast.RShift: "shrsi",
}
INDEX = "index"


def lower_module(info: ModuleInfo) -> str:
    """Return the hub module's text: the output-name globals, the binding's externals, then each function."""
    header = _Header()
    functions: list[str] = []
    for function in info.functions:
        functions.extend(_FunctionLowering(info, function, header).lower())
    return "\n".join(["module {", *header.lines(), *functions, "}"]) + "\n"


def float_text(value: float) -> str:
    """Return a float literal as Python's repr, with ".0" added where the repr has no "." (1e-08 is 1.0e-08)."""
    text = repr(value)
    if "." in text:
        return text
    mantissa, _, exponent = text.partition("e")
    return f"{mantissa}.0e{exponent}" if exponent else f"{mantissa}.0"


class _Header:
    """The module-level declarations, each in order of first use: output-name globals, then the binding's externals."""

    def __init__(self) -> None:
        """Start with no declaration."""
        self.names: dict[str, str] = {}
        self.globals: list[str] = []
        self.externals: dict[str, str] = {}

    def name(self, text: str) -> tuple[str, str]:
        """Return the symbol and type of the global holding `text`'s ASCII bytes and one zero byte."""
        kind = f"memref<{len(text) + 1}xi8>"
        if text not in self.names:
            symbol = f"lassi_io_name{len(self.names)}"
            self.names[text] = symbol
            data = ", ".join(str(byte) for byte in text.encode("ascii") + b"\x00")
            self.globals.append(f'  memref.global "private" constant @{symbol} : {kind} = dense<[{data}]>')
        return self.names[text], kind

    def external(self, symbol: str, inputs: str, result: str) -> None:
        """Declare an external function of the binding the first time it is called."""
        if symbol not in self.externals:
            self.externals[symbol] = f"  func.func private @{symbol}({inputs}){result}"

    def lines(self) -> list[str]:
        """Return the declarations: globals first, then externals."""
        return [*self.globals, *self.externals.values()]


class _FunctionLowering:
    """Writes one function's text."""

    def __init__(self, info: ModuleInfo, function: FunctionInfo, header: _Header) -> None:
        """Start the function with no value numbered and no slot open."""
        self.signatures: dict[str, Signature] = info.signatures
        self.f = function
        self.types = function.types
        self.header = header
        self.lines: list[str] = []
        self.count = 0
        self.depth = 2
        self.nested = 0
        self.loops = 0
        self.values: dict[str, str] = {}
        self.slots: dict[str, str] = {}
        self.allocations: list[tuple[str, str]] = []

    def lower(self) -> list[str]:
        """Return the function's lines, from `func.func` to its closing brace."""
        signature = self.f.signature
        params = ", ".join(f"%arg{index}: {mlir_type(kind)}" for index, (_, kind) in enumerate(signature.params))
        result = f" -> {signature.result}" if signature.result else ""
        self.lines.append(f"  func.func @{signature.name}({params}){result} {{")
        self._open_slots(signature)
        body = self.f.node.body
        self._block(body[1:] if is_docstring(body[0]) else body)
        if not isinstance(body[-1], ast.Return):
            self._free()
            self.emit("return")
        self.lines.append("  }")
        return self.lines

    def _open_slots(self, signature: Signature) -> None:
        """Name the parameters, then open the entry slots: assigned parameters (stored), then the other names."""
        for index, (name, _) in enumerate(signature.params):
            self.values[name] = f"%arg{index}"
        for name in self.f.assigned:
            slot = self.alloca(name)
            self.emit(f"memref.store {self.values[name]}, {slot}[] : memref<{self.f.slots[name]}>")
        for name in self.f.entry:
            self.alloca(name)

    # Text helpers

    def emit(self, text: str) -> None:
        """Append one line at the current depth."""
        self.lines.append("  " * self.depth + text)

    def fresh(self) -> str:
        """Return the next value name."""
        name = f"%{self.count}"
        self.count += 1
        return name

    def op(self, text: str) -> str:
        """Append an op with one result and return the result's name."""
        name = self.fresh()
        self.emit(f"{name} = {text}")
        return name

    def constant(self, value: bool | int | float, kind: str) -> str:
        """Append an arith.constant of `value` at type `kind` and return it."""
        if kind == BOOL:
            return self.op(f"arith.constant {'true' if value else 'false'}")
        text = float_text(value) if isinstance(value, float) else str(value)
        return self.op(f"arith.constant {text} : {kind}")

    def alloca(self, name: str) -> str:
        """Open the rank-0 slot of the scalar `name` and return it."""
        slot = self.op(f"memref.alloca() : memref<{self.f.slots[name]}>")
        self.slots[name] = slot
        return slot

    @contextmanager
    def region(self, loop: bool = False) -> Iterator[None]:
        """Write the enclosed lines one level deeper, inside a region of an op; `loop` for a loop's body."""
        self.depth += 1
        self.nested += 1
        self.loops += int(loop)
        try:
            yield
        finally:
            self.depth -= 1
            self.nested -= 1
            self.loops -= int(loop)

    @contextmanager
    def alloca_scope(self) -> Iterator[None]:
        """Write the enclosed lines in a memref.alloca_scope, which frees their allocas when it ends."""
        self.emit("memref.alloca_scope {")
        with self.region():
            yield
        self.emit("}")

    def _free(self) -> None:
        """Free each memref.alloc of the function, in allocation order, before its return."""
        for value, kind in self.allocations:
            self.emit(f"memref.dealloc {value} : {kind}")

    def call_op(self) -> str:
        """Return the call op's name: `call` directly in a func.func body, where func is the default dialect."""
        return "call" if self.nested == 0 else "func.call"

    def to_float(self, value: str, kind: str) -> str:
        """Return `value` as an f64, converting an int with sitofp."""
        return self.op(f"arith.sitofp {value} : i64 to f64") if kind == INT else value

    # Statements

    def _block(self, stmts: list[ast.stmt]) -> None:
        """Write each statement."""
        for stmt in stmts:
            self._statement(stmt)

    def _statement(self, stmt: ast.stmt) -> None:
        """Write one statement the check accepted."""
        if isinstance(stmt, ast.Pass):
            return
        if isinstance(stmt, ast.Expr):
            self._call_statement(stmt.value)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            self._assign(stmt)
        elif isinstance(stmt, ast.AugAssign):
            self._augmented(stmt)
        elif isinstance(stmt, ast.If):
            self._if(stmt)
        elif isinstance(stmt, ast.For):
            self._for(stmt)
        elif isinstance(stmt, ast.While):
            self._while(stmt)
        elif isinstance(stmt, ast.Return):
            self._return(stmt)
        else:
            raise AssertionError(f"the check accepted a {type(stmt).__name__} statement it never lowers")

    def _call_statement(self, call: ast.Call) -> None:
        """Write a call used as a statement."""
        if is_named_call(call, "write_array"):
            self._write(call)
        else:
            self._call(call)

    def _assign(self, stmt: ast.Assign | ast.AnnAssign) -> None:
        """Write an array binding, `x = e`, `a[i] = e`, or `z[:] = <expression>`."""
        target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
        binding = self.f.array_bindings.get(stmt)
        if binding is not None:
            self._array_binding(target.id, binding)
        elif isinstance(target, ast.Name):
            value = self.expr(stmt.value)
            self.emit(f"memref.store {value}, {self.slots[target.id]}[] : memref<{self.f.slots[target.id]}>")
        elif isinstance(target.slice, ast.Slice):
            self._vector_store(target, stmt.value)
        else:
            self._element_store(target, stmt.value)

    def _element_store(self, target: ast.Subscript, value_node: ast.expr) -> None:
        """Write `a[i] = e`: e, then i, then the narrowed store."""
        base, array = self._array(target.value)
        value = self.expr(value_node)
        index = self._index(target.slice)
        stored = self._narrow(value, self.types[value_node], array)
        self.emit(f"memref.store {stored}, {base}[{index}] : {array.memref}")

    def _augmented(self, stmt: ast.AugAssign) -> None:
        """Write `x op= e` (load, e, apply, store) or `a[i] op= e` (i, load, widen, e, apply, narrow, store)."""
        target = stmt.target
        right = self.types[stmt.value]
        if isinstance(target, ast.Name):
            kind, slot = self.f.slots[target.id], self.slots[target.id]
            current = self.op(f"memref.load {slot}[] : memref<{kind}>")
            value = self.expr(stmt.value)
            result = self._binop(stmt.op, current, kind, value, right)
            self.emit(f"memref.store {result}, {slot}[] : memref<{kind}>")
            return
        base, array = self._array(target.value)
        index = self._index(target.slice)
        element = self._widen(self.op(f"memref.load {base}[{index}] : {array.memref}"), array)
        value = self.expr(stmt.value)
        result = self._binop(stmt.op, element, array.scalar, value, right)
        stored = self._narrow(result, binop_result(stmt.op, array.scalar, right), array)
        self.emit(f"memref.store {stored}, {base}[{index}] : {array.memref}")

    def _if(self, stmt: ast.If) -> None:
        """Write scf.if over the condition, an elif nested in the else region."""
        condition = self.expr(stmt.test)
        self.emit(f"scf.if {condition} {{")
        with self.region():
            self._block(stmt.body)
        if stmt.orelse:
            self.emit("} else {")
            with self.region():
                self._block(stmt.orelse)
        self.emit("}")

    def _while(self, stmt: ast.While) -> None:
        """Write scf.while : () -> (), the condition in its before region and the body in its do region."""
        self.emit("scf.while : () -> () {")
        with self.region():
            condition = self.expr(stmt.test)
            self.emit(f"scf.condition({condition})")
        self.emit("} do {")
        with self.region(loop=True):
            self._block(stmt.body)
            self.emit("scf.yield")
        self.emit("}")

    def _for(self, stmt: ast.For) -> None:
        """Write scf.for over range(...), or scf.parallel over prange(...) whose alloca_scope opens private slots."""
        args = stmt.iter.args
        parallel = stmt.iter.func.id == "prange"
        start_node, stop_node = (args[0], args[1]) if len(args) > 1 else (None, args[0])
        start = self.expr(start_node) if start_node is not None else None
        stop = self.expr(stop_node)
        lower = self.op(f"arith.index_cast {start} : i64 to index") if start is not None else None
        upper = self.op(f"arith.index_cast {stop} : i64 to index")
        if lower is None:
            lower = self.constant(0, INDEX)
        step = self.constant(args[2].value if len(args) == 3 else 1, INDEX)
        induction = self.fresh()
        if parallel:
            self.emit(f"scf.parallel ({induction}) = ({lower}) to ({upper}) step ({step}) {{")
        else:
            self.emit(f"scf.for {induction} = {lower} to {upper} step {step} {{")
        outer = dict(self.slots)
        with self.region(loop=True):
            if parallel:
                with self.alloca_scope():
                    self._loop_body(stmt, self.f.private.get(stmt, ()), induction)
                self.emit("scf.yield")
            else:
                self._loop_body(stmt, (), induction)
        self.slots = outer
        self.emit("}")

    def _loop_body(self, stmt: ast.For, private: tuple[str, ...], induction: str) -> None:
        """Write a loop's body: its private slots, the loop variable's store, then its statements."""
        for name in private:
            self.alloca(name)
        value = self.op(f"arith.index_cast {induction} : index to i64")
        self.emit(f"memref.store {value}, {self.slots[stmt.target.id]}[] : memref<{INT}>")
        self._block(stmt.body)

    def _return(self, stmt: ast.Return) -> None:
        """Write the function's return: its value (an int from a float function converted), the frees, the return."""
        if stmt.value is None:
            self._free()
            self.emit("return")
            return
        kind = self.f.signature.result
        value = self.expr(stmt.value)
        if kind == FLOAT:
            value = self.to_float(value, self.types[stmt.value])
        self._free()
        self.emit(f"return {value} : {kind}")

    # Arrays and harness I/O

    def _array(self, node: ast.expr) -> tuple[str, ArrayType]:
        """Return an array name's value and type."""
        return self.values[node.id], self.f.arrays[node.id]

    def _array_binding(self, name: str, binding: ArrayBinding) -> None:
        """Bind an array name: another array's value, an input read, or a zero-filled allocation."""
        array = binding.array
        if binding.kind == "view":
            self.values[name] = self.values[binding.source.id]
            return
        if binding.kind == "read":
            index = self.constant(binding.source, INT)
            symbol = f"lassi_io_read_{array.dtype}"
            self.header.external(symbol, INT, f" -> {array.memref}")
            self.values[name] = self.op(f"{self.call_op()} @{symbol}({index}) : (i64) -> {array.memref}")
            return
        count = self.expr(binding.source)
        size = self.op(f"arith.index_cast {count} : i64 to index")
        base = self.op(f"memref.alloc({size}) : {array.memref}")
        self.allocations.append((base, array.memref))
        start = self.constant(0, INDEX)
        step = self.constant(1, INDEX)
        zero = self.constant(0.0 if array.scalar == FLOAT else 0, array.element)
        position = self.fresh()
        self.emit(f"scf.for {position} = {start} to {size} step {step} {{")
        with self.region():
            self.emit(f"memref.store {zero}, {base}[{position}] : {array.memref}")
        self.emit("}")
        self.values[name] = base

    def _write(self, call: ast.Call) -> None:
        """Write write_array; inside a loop body, in a memref.alloca_scope that frees its shape buffer each time."""
        if self.loops:
            with self.alloca_scope():
                self._write_ops(call)
        else:
            self._write_ops(call)

    def _write_ops(self, call: ast.Call) -> None:
        """Write write_array's ops: the argv index, the name's global, the shape buffer, then the external call."""
        path, name, _, shape, data = call.args
        index = self.constant(argv_index(path), INT)
        symbol, kind = self.header.name(name.value)
        pointer = self.op(f"memref.get_global @{symbol} : {kind}")
        text = self.op(f"memref.cast {pointer} : {kind} to memref<?xi8>")
        dims = [self.expr(dim) for dim in shape.elts]
        dims_kind = f"memref<{len(dims)}xi64>"
        buffer = self.op(f"memref.alloca() : {dims_kind}")
        for position, dim in enumerate(dims):
            at = self.constant(position, INDEX)
            self.emit(f"memref.store {dim}, {buffer}[{at}] : {dims_kind}")
        shape_value = self.op(f"memref.cast {buffer} : {dims_kind} to memref<?xi64>")
        base, array = self._array(data)
        callee = f"lassi_io_write_{array.dtype}"
        inputs = f"i64, memref<?xi8>, memref<?xi64>, {array.memref}"
        self.header.external(callee, inputs, "")
        self.emit(f"{self.call_op()} @{callee}({index}, {text}, {shape_value}, {base}) : ({inputs}) -> ()")

    def _vector_store(self, target: ast.Subscript, value_node: ast.expr) -> None:
        """Write `z[:] = <expression>`: its scalar parts, then scf.parallel over z's length storing each element."""
        base, array = self._array(target.value)
        parts: dict[ast.AST, str] = {}
        self._scalar_parts(value_node, parts)
        zero = self.constant(0, INDEX)
        length = self.op(f"memref.dim {base}, {zero} : {array.memref}")
        start = self.constant(0, INDEX)
        step = self.constant(1, INDEX)
        induction = self.fresh()
        self.emit(f"scf.parallel ({induction}) = ({start}) to ({length}) step ({step}) {{")
        with self.region():
            value = self._vector_value(value_node, induction, parts)
            value = self.to_float(value, self.types[value_node])
            self.emit(f"memref.store {value}, {base}[{induction}] : {array.memref}")
            self.emit("scf.yield")
        self.emit("}")

    def _scalar_parts(self, node: ast.expr, parts: dict[ast.AST, str]) -> None:
        """Write each part of a vectorized expression that holds no ndarray, in evaluation order."""
        if node not in self.f.vector:
            parts[node] = self.expr(node)
        elif isinstance(node, ast.BinOp):
            self._scalar_parts(node.left, parts)
            self._scalar_parts(node.right, parts)
        elif isinstance(node, ast.Call):
            self._scalar_parts(node.args[0], parts)

    def _vector_value(self, node: ast.expr, induction: str, parts: dict[ast.AST, str]) -> str:
        """Write one element of a vectorized expression: each ndarray loaded at the induction variable."""
        if node not in self.f.vector:
            return parts[node]
        if isinstance(node, ast.Name):
            base, array = self._array(node)
            return self.op(f"memref.load {base}[{induction}] : {array.memref}")
        if isinstance(node, ast.BinOp):
            left = self._vector_value(node.left, induction, parts)
            right = self._vector_value(node.right, induction, parts)
            return self._binop(node.op, left, self.types[node.left], right, self.types[node.right])
        argument = self._vector_value(node.args[0], induction, parts)
        return self.op(f"math.sqrt {self.to_float(argument, self.types[node.args[0]])} : f64")

    def _index(self, node: ast.expr) -> str:
        """Write an index expression and its index_cast to index."""
        value = self.expr(node)
        return self.op(f"arith.index_cast {value} : i64 to index")

    def _widen(self, value: str, array: ArrayType) -> str:
        """Widen a loaded element to the Python scalar: extf for f32, extsi or extui for i8, u8, i32, and u32."""
        element = array.element
        if element == "f32":
            return self.op(f"arith.extf {value} : f32 to f64")
        if element in ("i8", "i32"):
            extend = "extui" if array.unsigned else "extsi"
            return self.op(f"arith.{extend} {value} : {element} to i64")
        return value

    def _narrow(self, value: str, kind: str, array: ArrayType) -> str:
        """Narrow a value to an array's element (truncf, trunci); an int bound for a float array is sitofp'd first."""
        if array.scalar == FLOAT:
            value = self.to_float(value, kind)
        element = array.element
        if element == "f32":
            return self.op(f"arith.truncf {value} : f64 to f32")
        if element in ("i8", "i32"):
            return self.op(f"arith.trunci {value} : i64 to {element}")
        return value

    # Expressions

    def expr(self, node: ast.expr) -> str:
        """Write a scalar expression and return its value."""
        if isinstance(node, ast.Constant):
            return self.constant(node.value, self.types[node])
        if isinstance(node, ast.Name):
            return self._name(node.id)
        if isinstance(node, ast.UnaryOp):
            return self._unary(node)
        if isinstance(node, ast.BinOp):
            left = self.expr(node.left)
            right = self.expr(node.right)
            return self._binop(node.op, left, self.types[node.left], right, self.types[node.right])
        if isinstance(node, ast.BoolOp):
            return self._logic(isinstance(node.op, ast.And), node.values)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.IfExp):
            return self._ifexp(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Subscript):
            base, array = self._array(node.value)
            index = self._index(node.slice)
            return self._widen(self.op(f"memref.load {base}[{index}] : {array.memref}"), array)
        raise AssertionError(f"the check accepted a {type(node).__name__} expression it never lowers")

    def _name(self, name: str) -> str:
        """Read a scalar name: load its slot, or use the parameter's value."""
        if name in self.slots:
            return self.op(f"memref.load {self.slots[name]}[] : memref<{self.f.slots[name]}>")
        return self.values[name]

    def _unary(self, node: ast.UnaryOp) -> str:
        """Write -x (negf; 0 - x for an int), +x, not b (xori with true), or ~n (xori with -1)."""
        kind = self.types[node]
        literal = folded_literal(node)
        if literal is not None:
            return self.constant(literal, kind)
        value = self.expr(node.operand)
        op = node.op
        if isinstance(op, ast.UAdd):
            return value
        if isinstance(op, ast.USub) and kind == FLOAT:
            return self.op(f"arith.negf {value} : f64")
        if isinstance(op, ast.USub):
            zero = self.constant(0, INT)
            return self.op(f"arith.subi {zero}, {value} : i64")
        if isinstance(op, ast.Not):
            true = self.constant(True, BOOL)
            return self.op(f"arith.xori {value}, {true} : i1")
        ones = self.constant(-1, INT)
        return self.op(f"arith.xori {value}, {ones} : i64")

    def _binop(self, op: ast.operator, left: str, left_kind: str, right: str, right_kind: str) -> str:
        """Write a binary operation; % is a - (a // b) * b, and an int operand of a float operation is converted."""
        if isinstance(op, ast.Mod):
            quotient = self.op(f"arith.floordivsi {left}, {right} : i64")
            product = self.op(f"arith.muli {quotient}, {right} : i64")
            return self.op(f"arith.subi {left}, {product} : i64")
        if binop_result(op, left_kind, right_kind) == FLOAT:
            left = self.to_float(left, left_kind)
            right = self.to_float(right, right_kind)
            return self.op(f"arith.{_FLOAT_OPS[type(op)]} {left}, {right} : f64")
        return self.op(f"arith.{_INT_OPS[type(op)]} {left}, {right} : i64")

    def _compare(self, node: ast.Compare) -> str:
        """Write cmpi on two ints, or cmpf with an int operand converted."""
        right_node = node.comparators[0]
        left = self.expr(node.left)
        right = self.expr(right_node)
        left_kind, right_kind = self.types[node.left], self.types[right_node]
        _, signed, ordered = COMPARISONS[type(node.ops[0])]
        if FLOAT in (left_kind, right_kind):
            left = self.to_float(left, left_kind)
            right = self.to_float(right, right_kind)
            return self.op(f"arith.cmpf {ordered}, {left}, {right} : f64")
        return self.op(f"arith.cmpi {signed}, {left}, {right} : i64")

    def _logic(self, conjunction: bool, values: list[ast.expr]) -> str:
        """Write `a and b ...` or `a or b ...` as scf.if -> (i1), short-circuiting as Python does."""
        first = self.expr(values[0])
        if len(values) == 1:
            return first
        result = self.fresh()
        self.emit(f"{result} = scf.if {first} -> (i1) {{")
        with self.region():
            then = self._logic(conjunction, values[1:]) if conjunction else self.constant(True, BOOL)
            self.emit(f"scf.yield {then} : i1")
        self.emit("} else {")
        with self.region():
            other = self.constant(False, BOOL) if conjunction else self._logic(conjunction, values[1:])
            self.emit(f"scf.yield {other} : i1")
        self.emit("}")
        return result

    def _ifexp(self, node: ast.IfExp) -> str:
        """Write `x if c else y` as scf.if on c yielding each arm."""
        condition = self.expr(node.test)
        kind = self.types[node]
        result = self.fresh()
        self.emit(f"{result} = scf.if {condition} -> ({kind}) {{")
        with self.region():
            body = self.expr(node.body)
            self.emit(f"scf.yield {body} : {kind}")
        self.emit("} else {")
        with self.region():
            other = self.expr(node.orelse)
            self.emit(f"scf.yield {other} : {kind}")
        self.emit("}")
        return result

    def _call(self, node: ast.Call) -> str | None:
        """Write a call of a module function, a builtin, or a math function; None for a function returning None."""
        func = node.func
        if isinstance(func, ast.Name) and func.id in self.signatures:
            return self._function_call(node, self.signatures[func.id])
        if isinstance(func, ast.Name):
            return self._builtin(node, func.id)
        argument = node.args[0]
        value = self.to_float(self.expr(argument), self.types[argument])
        return self.op(f"math.{func.attr} {value} : f64")

    def _function_call(self, node: ast.Call, signature: Signature) -> str | None:
        """Write a call: the arguments in order, then each int argument to a float parameter converted."""
        values = []
        for arg, (_, kind) in zip(node.args, signature.params, strict=True):
            values.append(self.values[arg.id] if isinstance(kind, ArrayType) else self.expr(arg))
        for position, (arg, (_, kind)) in enumerate(zip(node.args, signature.params, strict=True)):
            if kind == FLOAT:
                values[position] = self.to_float(values[position], self.types[arg])
        inputs = ", ".join(mlir_type(kind) for _, kind in signature.params)
        text = f"{self.call_op()} @{signature.name}({', '.join(values)}) : ({inputs}) -> {signature.result or '()'}"
        if signature.result is None:
            self.emit(text)
            return None
        return self.op(text)

    def _builtin(self, node: ast.Call, name: str) -> str:
        """Write len(a) (memref.dim), abs(x) (math.absf or math.absi), float(x), or int(x)."""
        argument = node.args[0]
        if name == "len":
            base, array = self._array(argument)
            zero = self.constant(0, INDEX)
            length = self.op(f"memref.dim {base}, {zero} : {array.memref}")
            return self.op(f"arith.index_cast {length} : index to i64")
        value = self.expr(argument)
        kind = self.types[argument]
        if name == "abs":
            return self.op(f"math.{'absf' if kind == FLOAT else 'absi'} {value} : {kind}")
        if name == "float":
            return self.to_float(value, kind)
        return value if kind == INT else self.op(f"arith.fptosi {value} : f64 to i64")
