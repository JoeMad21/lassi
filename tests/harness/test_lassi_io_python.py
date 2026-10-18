"""Tests for the Python harness binding assets/harness/python/lassi_io.py (task P12.3).

Bible: Harness Contract (lassi_io format version 1: elements are raw
little-endian bit patterns and neither side converts a value), Frontend
Rules (the harness I/O contract ships a Python module among the bindings
per language), Repository Layout (assets/harness/python/). Plan:
plans/p12-frontends.md, P12.3, and its planning decision that the binding
exports prange as range.

The binding is copied beside a program as a harness file, so it is not a
package module: these tests load it by its file path. The contract they fix:

- write_array(path, name, dtype, shape, data) takes the arguments of
  lassi.harness.lassi_io.write_array: a str or a path, a dtype name (a key
  of lassi.harness.lassi_io.DTYPES), a sequence of ints (empty for a
  scalar), and the little-endian element bytes in C order as any bytes-like
  object (bytes, bytearray, memoryview), or the data of an array the binding
  read. It writes exactly the bytes the reference writes, for every dtype,
  rank 0, and empty arrays.
- read_array(path) returns the binding's array: .name (str), .dtype (the
  dtype name), .shape (the dims, compared as a tuple), and .data, a buffer
  whose bytes (bytes(data)) are the file's element bytes, unchanged. It
  reads every file the reference writes, and the reference reads every file
  it writes.
- The module docstring names the class read_array returns and the class of
  its data: the binding's array type. That data is a writable, flat
  memoryview cast to the dtype's view format (VIEW_FORMATS; f16 and bf16 as
  their 16-bit patterns), so a program indexes element values.
- Like lassi_io.h, the binding refuses to read or write on a big-endian host.
- prange is the builtin range, so a subset program imports it from the
  binding and runs natively under the pinned python (P12.4 reads that import
  as the parallel-for).
- It imports only the standard library of Python 3.10 and nothing from
  lassi, and a program beside it runs with nothing but the standard library.

The refusals are in test_lassi_io_python_refusals.py. The bit patterns are
test inputs; no value here is a measurement.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import math
import re
import shutil
import struct
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from lassi.harness import lassi_io as reference

REPO = Path(__file__).resolve().parents[2]
BINDING = REPO / "assets" / "harness" / "python" / "lassi_io.py"
DTYPE_NAMES = list(reference.DTYPES)

# Six elements per dtype as raw little-endian bit patterns: zeros, signs, extremes, and for the float dtypes
# infinities, quiet and signaling NaNs with payloads, and subnormals, so a binding that converts a value shows.
PATTERNS = {
    "f32": struct.pack("<6I", 0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7F800001),
    "f16": struct.pack("<6H", 0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0x7C01),
    "bf16": struct.pack("<6H", 0x7FC0, 0x7F81, 0xFF80, 0x8000, 0x0001, 0x3F80),
    "i32": struct.pack("<6i", 0, -1, 2**31 - 1, -(2**31), 1, -2),
    "u32": struct.pack("<6I", 0, 1, 2**32 - 1, 2**31, 0xDEADBEEF, 7),
    "i8": struct.pack("<6b", 0, -1, 127, -128, 1, -2),
    "u8": struct.pack("<6B", 0, 1, 255, 128, 127, 42),
    "i64": struct.pack("<6q", 0, -1, 2**63 - 1, -(2**63), 1, -2),
    "f64": struct.pack(
        "<6Q", 1 << 63, 0x7FF0000000000000, 0x7FF0000000000001, 0xFFF0000000000001, 0x7FF8000000000001, 1
    ),
}
# Signaling and quiet NaNs of both signs with payloads, and the smallest subnormal: a value converted to a Python
# float and back can come back with other bits.
NAN_PATTERNS = {
    "f32": struct.pack("<6I", 0x7F800001, 0xFF800001, 0x7FBFFFFF, 0x7FC00001, 0xFFFFFFFF, 0x00000001),
    "f64": struct.pack(
        "<6Q",
        0x7FF0000000000001,
        0xFFF0000000000001,
        0x7FF7FFFFFFFFFFFF,
        0x7FF8000000000001,
        0xFFFFFFFFFFFFFFFF,
        0x0000000000000001,
    ),
}
# The memoryview format of the data the binding reads, per dtype: f16 and bf16 are their 16-bit patterns.
VIEW_FORMATS = {
    "f32": "f",
    "f16": "H",
    "bf16": "H",
    "i32": "i",
    "u32": "I",
    "i8": "b",
    "u8": "B",
    "i64": "q",
    "f64": "d",
}
# Shapes of at most six elements: 2-D, 1-D, rank 0 (a scalar), empty arrays, and rank 4.
SHAPES = {
    "2x3": (2, 3),
    "6": (6,),
    "rank0": (),
    "empty-0": (0,),
    "empty-3x0x2": (3, 0, 2),
    "empty-6x0": (6, 0),
    "1x1x1x6": (1, 1, 1, 6),
}

# Every signature of range, empty, forward, backward, stepped, and past 64 bits.
RANGE_ARGS = [
    (0,),
    (1,),
    (5,),
    (-3,),
    (0, 0),
    (2, 7),
    (7, 2),
    (-5, 5),
    (0, 10, 3),
    (10, 0, -3),
    (10, 0, 3),
    (0, 10, -1),
    (-1, -10, -2),
    (3, 4, 10**20),
    (2**70, 2**70 + 4),
    (0, 2**70, 2**68),
]
# Calls range refuses: no argument, a zero step, a float, a str, four arguments, None.
BAD_RANGE_ARGS = [(), (1, 2, 0), (1.5,), ("3",), (1, 2, 3, 4), (None,)]

# A program beside its harness file: copy one array through the binding, loop over a prange, report lassi modules.
PROGRAM = '''\
"""A program beside its harness file, run with only the standard library on its path."""

import sys

from lassi_io import prange, read_array, write_array

array = read_array(sys.argv[1])
write_array(sys.argv[2], array.name, array.dtype, array.shape, array.data)
total = 0
for i in prange(4):
    total += i
loaded = sorted(name for name in sys.modules if name == "lassi" or name.startswith("lassi."))
print(total, len(loaded))
'''


@functools.lru_cache(maxsize=None)
def binding() -> ModuleType:
    """Load the binding by its file path (registered first, as dataclasses expect); fail when it does not exist."""
    assert BINDING.is_file(), f"no Python binding at {BINDING.relative_to(REPO).as_posix()} (task P12.3 writes it)"
    spec = importlib.util.spec_from_file_location("lassi_io_python_binding", BINDING)
    assert spec is not None and spec.loader is not None, f"cannot load {BINDING}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def data_for(dtype: str, shape: tuple[int, ...]) -> bytes:
    """Return the first product(shape) elements of the dtype's patterns (a byte ramp for a dtype with none)."""
    itemsize = reference.DTYPES[dtype].itemsize
    pattern = PATTERNS.get(dtype, bytes(range(1, 6 * itemsize + 1)))
    return pattern[: math.prod(shape) * itemsize]


def written(module: ModuleType, path: Path, name: str, dtype: str, shape: tuple[int, ...], data: object) -> bytes:
    """Write one array with `module`'s write_array and return the file's bytes."""
    module.write_array(path, name, dtype, shape, data)
    return path.read_bytes()


def contents(array: object) -> tuple[str, str, tuple[int, ...], bytes]:
    """Return (name, dtype, shape, element bytes) of an array either side read."""
    return (array.name, array.dtype, tuple(array.shape), bytes(array.data))


def binding_tree() -> ast.Module:
    """Parse the binding's source; fail when it does not exist."""
    assert BINDING.is_file(), f"no Python binding at {BINDING.relative_to(REPO).as_posix()} (task P12.3 writes it)"
    return ast.parse(BINDING.read_text(encoding="utf-8"), filename=str(BINDING))


def imports_of(tree: ast.Module) -> list[tuple[str, int]]:
    """Return (module, level) of every import statement anywhere in `tree`; level > 0 is a relative import."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.module or "", node.level))
    return found


# ---------------------------------------------------------------------------
# The same bytes as the reference


@pytest.mark.parametrize("dtype", DTYPE_NAMES)
def test_the_binding_writes_the_reference_bytes_in_every_shape(dtype: str, tmp_path: Path) -> None:
    module = binding()
    differ = []
    for shape_id, shape in SHAPES.items():
        data = data_for(dtype, shape)
        ours = written(module, tmp_path / "binding.lassiio", f"{dtype}_out", dtype, shape, data)
        if ours != written(reference, tmp_path / "reference.lassiio", f"{dtype}_out", dtype, shape, data):
            differ.append(shape_id)
    assert not differ, f"shapes whose bytes differ from the reference's (rank 0 and empty arrays included): {differ}"


def test_the_binding_pads_every_name_length_as_the_reference_does(tmp_path: Path) -> None:
    module = binding()
    differ = []
    for rank in range(4):
        shape = tuple(range(1, rank + 1))
        data = bytes(range(math.prod(shape)))
        for length in [*range(1, 18), 254, 255]:
            name = "n" * length
            ours = written(module, tmp_path / "binding.lassiio", name, "u8", shape, data)
            if ours != written(reference, tmp_path / "reference.lassiio", name, "u8", shape, data):
                differ.append((rank, length))
    assert not differ, f"(rank, name length) pairs whose bytes differ from the reference's: {differ}"


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_every_16_bit_pattern_passes_through_the_binding_unchanged(dtype: str, tmp_path: Path) -> None:
    # Every NaN (quiet, signaling, either sign, any payload), both infinities, -0, and every subnormal.
    data = struct.pack("<65536H", *range(1 << 16))
    module = binding()
    ours = written(module, tmp_path / "binding.lassiio", f"all_{dtype}", dtype, (256, 256), data)
    assert ours == written(reference, tmp_path / "reference.lassiio", f"all_{dtype}", dtype, (256, 256), data)
    assert contents(module.read_array(tmp_path / "binding.lassiio")) == (f"all_{dtype}", dtype, (256, 256), data)


@pytest.mark.parametrize("dtype", sorted(NAN_PATTERNS))
def test_nan_payloads_pass_through_the_binding_unchanged(dtype: str, tmp_path: Path) -> None:
    data = NAN_PATTERNS[dtype]
    module = binding()
    reference.write_array(tmp_path / "reference.lassiio", "nan", dtype, (6,), data)
    array = module.read_array(tmp_path / "reference.lassiio")
    assert contents(array) == ("nan", dtype, (6,), data)
    module.write_array(tmp_path / "binding.lassiio", array.name, array.dtype, array.shape, array.data)
    assert (tmp_path / "binding.lassiio").read_bytes() == (tmp_path / "reference.lassiio").read_bytes()


# ---------------------------------------------------------------------------
# Each reads the other's files


@pytest.mark.parametrize("dtype", DTYPE_NAMES)
def test_the_binding_reads_the_files_the_reference_writes_in_every_shape(dtype: str, tmp_path: Path) -> None:
    module = binding()
    differ = []
    for shape_id, shape in SHAPES.items():
        data = data_for(dtype, shape)
        path = tmp_path / f"reference-{shape_id}.lassiio"
        reference.write_array(path, f"{dtype}_in", dtype, shape, data)
        array = module.read_array(path)
        assert type(array.shape) is tuple and all(type(dim) is int for dim in array.shape), array.shape
        if contents(array) != (f"{dtype}_in", dtype, shape, data):
            differ.append(shape_id)
    assert not differ, f"shapes the binding reads otherwise than the reference wrote them: {differ}"


@pytest.mark.parametrize("dtype", DTYPE_NAMES)
def test_the_reference_reads_the_files_the_binding_writes_in_every_shape(dtype: str, tmp_path: Path) -> None:
    module = binding()
    differ = []
    for shape_id, shape in SHAPES.items():
        data = data_for(dtype, shape)
        path = tmp_path / f"binding-{shape_id}.lassiio"
        module.write_array(path, f"{dtype}_out", dtype, shape, data)
        if reference.read_array(path) != reference.LassiArray(f"{dtype}_out", dtype, shape, data):
            differ.append(shape_id)
    assert not differ, f"shapes the reference reads otherwise than the binding wrote them: {differ}"


@pytest.mark.parametrize("dtype", DTYPE_NAMES)
def test_an_array_the_binding_reads_writes_back_to_the_same_bytes_in_every_shape(dtype: str, tmp_path: Path) -> None:
    module = binding()
    differ = []
    for shape_id, shape in SHAPES.items():
        source = tmp_path / f"in-{shape_id}.lassiio"
        reference.write_array(source, "copy", dtype, shape, data_for(dtype, shape))
        array = module.read_array(source)
        module.write_array(tmp_path / "out.lassiio", array.name, array.dtype, array.shape, array.data)
        if (tmp_path / "out.lassiio").read_bytes() != source.read_bytes():
            differ.append(shape_id)
    assert not differ, f"shapes whose array, read and written back by the binding, changed its bytes: {differ}"


@pytest.mark.parametrize(
    "as_data",
    [bytes, bytearray, memoryview, lambda raw: memoryview(raw).cast("d")],
    ids=["bytes", "bytearray", "memoryview", "memoryview of f64"],
)
def test_the_binding_takes_any_bytes_like_data(as_data: Callable[[bytes], object], tmp_path: Path) -> None:
    raw = struct.pack("<3d", 1.5, -0.0, 2.0)
    ours = written(binding(), tmp_path / "binding.lassiio", "d", "f64", (3,), as_data(raw))
    assert ours == written(reference, tmp_path / "reference.lassiio", "d", "f64", (3,), raw)


def test_the_binding_takes_a_str_or_a_path(tmp_path: Path) -> None:
    module = binding()
    data = struct.pack("<3i", 7, -8, 9)
    module.write_array(str(tmp_path / "as-str.lassiio"), "k", "i32", [3], data)
    assert contents(module.read_array(tmp_path / "as-str.lassiio")) == ("k", "i32", (3,), data)
    module.write_array(tmp_path / "as-path.lassiio", "k", "i32", (3,), data)
    assert contents(module.read_array(str(tmp_path / "as-path.lassiio"))) == ("k", "i32", (3,), data)


# ---------------------------------------------------------------------------
# The array type, stated in the docstring


def test_the_module_docstring_states_the_array_type(tmp_path: Path) -> None:
    module = binding()
    path = tmp_path / "x.lassiio"
    reference.write_array(path, "x", "f32", (2,), struct.pack("<2f", 1.0, -2.0))
    array = module.read_array(path)
    doc = module.__doc__ or ""
    assert doc.strip(), "the binding has no module docstring"
    for kind in (type(array), type(array.data)):
        assert re.search(rf"\b{re.escape(kind.__name__)}\b", doc), (
            f"the module docstring does not name {kind.__name__}, the class of what read_array returns or of its data"
        )


@pytest.mark.parametrize("dtype", DTYPE_NAMES)
def test_the_data_is_a_flat_writable_view_in_the_dtype_format(dtype: str, tmp_path: Path) -> None:
    module = binding()
    kind, view = reference.DTYPES[dtype], VIEW_FORMATS[dtype]
    assert (module.DTYPES[dtype].code, module.DTYPES[dtype].itemsize) == (kind.code, kind.itemsize)
    assert module.DTYPES[dtype].view_format == view
    reference.write_array(tmp_path / "in.lassiio", "a", dtype, (2, 3), struct.pack(f"<6{view}", 0, 1, 2, 3, 4, 5))
    data = module.read_array(tmp_path / "in.lassiio").data
    assert isinstance(data, memoryview)
    assert (data.format, data.ndim, len(data), data.readonly) == (view, 1, 6, False)
    assert data.tolist() == [0, 1, 2, 3, 4, 5]
    data[4] = 7
    module.write_array(tmp_path / "out.lassiio", "a", dtype, (2, 3), data)
    assert reference.read_array(tmp_path / "out.lassiio").data == struct.pack(f"<6{view}", 0, 1, 2, 3, 7, 5)


# ---------------------------------------------------------------------------
# The host


def test_a_big_endian_host_is_refused_before_any_file_is_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = binding()
    source, target = tmp_path / "in.lassiio", tmp_path / "out.lassiio"
    reference.write_array(source, "x", "f32", (6,), PATTERNS["f32"])
    monkeypatch.setattr(sys, "byteorder", "big")
    calls = [
        (source, lambda: module.read_array(source)),
        (target, lambda: module.write_array(target, "x", "f32", (6,), PATTERNS["f32"])),
    ]
    for path, call in calls:
        with pytest.raises(module.LassiIOError) as refusal:
            call()
        assert str(refusal.value).startswith(str(path)) and "big-endian" in str(refusal.value)
    assert not target.exists()


# ---------------------------------------------------------------------------
# prange


def test_prange_is_range() -> None:
    assert binding().prange is range


@pytest.mark.parametrize("args", RANGE_ARGS, ids=str)
def test_prange_iterates_as_range(args: tuple[int, ...]) -> None:
    prange = binding().prange
    expected = list(range(*args))
    loop = prange(*args)
    assert list(loop) == expected
    assert list(loop) == expected, "a second pass over the same prange differs from range's"


@pytest.mark.parametrize("args", BAD_RANGE_ARGS, ids=str)
def test_prange_refuses_what_range_refuses(args: tuple[object, ...]) -> None:
    prange = binding().prange
    with pytest.raises(Exception) as ours:
        prange(*args)
    with pytest.raises(Exception) as theirs:
        range(*args)
    assert type(ours.value) is type(theirs.value)


# ---------------------------------------------------------------------------
# Standard library only, nothing from lassi


def test_the_binding_imports_only_the_standard_library() -> None:
    tree = binding_tree()
    tops = {module.split(".")[0] for module, level in imports_of(tree) if level == 0}
    outside = sorted(tops - set(sys.stdlib_module_names))
    assert not outside, f"the binding imports modules outside the standard library: {outside}"
    dynamic = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in {"__import__", "import_module"})
            or (isinstance(node.func, ast.Attribute) and node.func.attr in {"__import__", "import_module"})
        )
    ]
    assert not dynamic, f"the binding loads modules by name at run time, at lines {dynamic}"


def test_the_binding_imports_nothing_from_lassi() -> None:
    found = imports_of(binding_tree())
    relative = sorted(f"{'.' * level}{module}" for module, level in found if level > 0)
    assert not relative, f"the binding has relative imports: {relative}"
    lassi = sorted(module for module, _ in found if module == "lassi" or module.startswith("lassi."))
    assert not lassi, f"the binding imports from lassi: {lassi}"


def test_a_program_beside_the_binding_runs_with_only_the_standard_library(tmp_path: Path) -> None:
    assert BINDING.is_file(), f"no Python binding at {BINDING.relative_to(REPO).as_posix()} (task P12.3 writes it)"
    work = tmp_path / "work"
    work.mkdir()
    shutil.copyfile(BINDING, work / "lassi_io.py")
    (work / "program.py").write_text(PROGRAM, encoding="ascii", newline="\n")
    reference.write_array(work / "in.lassiio", "x", "f32", (2, 3), PATTERNS["f32"])
    # -E ignores PYTHONPATH, -S drops site-packages (where lassi is installed), -B writes no bytecode; the
    # script's directory stays on sys.path, so the program imports the harness file beside it.
    argv = [sys.executable, "-E", "-S", "-B", "program.py", "in.lassiio", "out.lassiio"]
    done = subprocess.run(argv, cwd=work, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, f"the program failed: {done.stderr[-2000:]}"
    assert done.stdout.split() == ["6", "0"], f"prange's sum and the count of lassi modules: {done.stdout!r}"
    assert (work / "out.lassiio").read_bytes() == (work / "in.lassiio").read_bytes()


def test_a_lassi_array_compares_by_identity(tmp_path: Path) -> None:
    module = binding()
    path = tmp_path / "pair.lassiio"
    module.write_array(path, "pair", "f32", (2,), struct.pack("<2f", 1.0, 2.0))
    first, second = module.read_array(path), module.read_array(path)
    assert first == first and hash(first) == hash(first)
    assert first != second, "two reads of one file are two arrays: a LassiArray compares by identity"

