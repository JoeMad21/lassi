"""Tests for the lassi_io file format and its Python reader and writer (task P4.3).

Bible: Harness Contract (programs read inputs and write outputs as binary
files through lassi_io.h; the harness owns inputs and comparison), Frontend
Rules (the harness I/O contract ships bindings per language), Repository
Layout (assets/harness/).

The contract these tests fix, in lassi/harness/lassi_io.py:

- DTYPES: a mapping from dtype name to a DType with .name, .code (the u32
  written in the file), and .itemsize (bytes per element). It holds at
  least these nine dtypes with these codes, which follow the order the task
  lists them; code 0 is never a dtype, and no two dtypes share a code:

      f32 1 4    f16 2 2    bf16 3 2    i32 4 4    u32 5 4
      i8  6 1    u8  7 1    i64  8 8    f64 9 8

  bf16 is a first-class dtype of 2 bytes per element. The project has no
  array library among its dependencies, so the API carries raw bytes: a
  bf16 element is its 16-bit pattern, little-endian.
- write_array(path, name, dtype, shape, data) -> None writes one file that
  holds one named array: `dtype` is a DTYPES key, `shape` a sequence of
  non-negative ints (empty for a scalar), and `data` the little-endian
  element bytes in C order. read_array(path) -> LassiArray reads one back.
  LassiArray(name, dtype, shape, data) compares by value; dtype is the
  DTYPES key, shape a tuple of ints, data bytes. Both functions take a str
  or a Path.
- The file, little-endian throughout, offsets from the start of the file:

      0          8 bytes     magic: the ASCII text LASSIIO and one zero byte
      8          u32         version, 1
      12         u32         dtype code
      16         u32         rank
      20         u64 x rank  dims, in C order
      then       u32         name length in bytes
      then       the name bytes
      then       zero bytes up to the next multiple of 8
      then       the element data, product(dims) * itemsize bytes, and nothing after it

  A rank-0 array is a scalar of one element; a dim of 0 gives an empty
  array with no data bytes.
- A name is 1 to 255 bytes, each an ASCII letter, digit, '_', '.', or '-'.

Three small files are written out here byte for byte, so the format cannot
drift silently; the other layouts are checked against a header this module
builds with struct. The refusals are in test_lassi_io_refusals.py. The
bit patterns are test inputs; no value here is a measurement.
"""

from __future__ import annotations

import importlib
import struct
from pathlib import Path
from types import ModuleType

import pytest

MAGIC = b"LASSIIO\x00"
# name -> (code, itemsize), as the module docstring states.
REQUIRED = {
    "f32": (1, 4),
    "f16": (2, 2),
    "bf16": (3, 2),
    "i32": (4, 4),
    "u32": (5, 4),
    "i8": (6, 1),
    "u8": (7, 1),
    "i64": (8, 8),
    "f64": (9, 8),
}

# Six elements per dtype: zeros, signs, extremes, and for the float dtypes infinities, a quiet NaN, a NaN with a
# payload (or a signaling NaN), and a subnormal, as raw little-endian bit patterns.
PATTERNS = {
    "f32": struct.pack("<6I", 0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001),
    "f16": struct.pack("<6H", 0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x0001),
    "bf16": struct.pack("<6H", 0x7FC0, 0x7F81, 0xFF80, 0x8000, 0x0001, 0x3F80),
    "i32": struct.pack("<6i", 0, -1, 2**31 - 1, -(2**31), 1, -2),
    "u32": struct.pack("<6I", 0, 1, 2**32 - 1, 2**31, 0xDEADBEEF, 7),
    "i8": struct.pack("<6b", 0, -1, 127, -128, 1, -2),
    "u8": struct.pack("<6B", 0, 1, 255, 128, 127, 42),
    "i64": struct.pack("<6q", 0, -1, 2**63 - 1, -(2**63), 1, -2),
    "f64": struct.pack(
        "<6Q", 0, 1 << 63, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000001, 0x0000000000000001
    ),
}
# Shapes every dtype round trips through: 2-D, 1-D, a scalar, empty arrays, and rank 4.
SHAPES = [(2, 3), (6,), (), (0,), (3, 0, 2), (1, 1, 1, 6)]

# Hand-written files, byte for byte: (name, dtype, shape, data, file bytes as hex).
EXAMPLES = {
    "f32-vector": (
        "x",
        "f32",
        (2,),
        struct.pack("<2f", 1.0, -2.0),
        "4c41535349494f00"  # magic: LASSIIO and a zero byte
        "01000000"  # version 1
        "01000000"  # dtype code 1 (f32)
        "01000000"  # rank 1
        "0200000000000000"  # dims[0] = 2
        "01000000"  # name length 1
        "78"  # name "x"; the header is 33 bytes so far
        "00000000000000"  # 7 zero bytes of padding, to offset 40
        "0000803f"  # 1.0
        "000000c0",  # -2.0
    ),
    "bf16-matrix": (
        "in_a",
        "bf16",
        (2, 3),
        struct.pack("<6H", 0x3F80, 0xBF80, 0x7FC0, 0x7F80, 0x8000, 0x0001),
        "4c41535349494f00"  # magic
        "01000000"  # version 1
        "03000000"  # dtype code 3 (bf16)
        "02000000"  # rank 2
        "0200000000000000"  # dims[0] = 2
        "0300000000000000"  # dims[1] = 3
        "04000000"  # name length 4
        "696e5f61"  # name "in_a"; the header is 44 bytes so far
        "00000000"  # 4 zero bytes of padding, to offset 48
        "803f"  # 1.0
        "80bf"  # -1.0
        "c07f"  # quiet NaN
        "807f"  # +infinity
        "0080"  # -0.0
        "0100",  # the smallest subnormal
    ),
    "i64-scalar": (
        "s",
        "i64",
        (),
        struct.pack("<q", -2),
        "4c41535349494f00"  # magic
        "01000000"  # version 1
        "08000000"  # dtype code 8 (i64)
        "00000000"  # rank 0: no dims follow
        "01000000"  # name length 1
        "73"  # name "s"; the header is 25 bytes so far
        "00000000000000"  # 7 zero bytes of padding, to offset 32
        "feffffffffffffff",  # -2
    ),
}

GOOD_NAMES = ["a", "Z", "0", "in_a", "A.b-c_9", "x.y.z", "a-b", "x" * 255]


def lassi_io() -> ModuleType:
    """Import and return lassi.harness.lassi_io."""
    return importlib.import_module("lassi.harness.lassi_io")


def header(name: bytes, code: int, shape: tuple[int, ...]) -> bytes:
    """Build a file header by hand: magic, version 1, dtype code, rank, dims, name length, name, zero padding."""
    head = MAGIC + struct.pack("<III", 1, code, len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
    head += struct.pack("<I", len(name)) + name
    return head + bytes(-len(head) % 8)


def count_of(shape: tuple[int, ...]) -> int:
    """Return the number of elements of `shape` (1 for a scalar)."""
    count = 1
    for dim in shape:
        count *= dim
    return count


def data_for(dtype: str, shape: tuple[int, ...]) -> bytes:
    """Return the first product(shape) elements of the dtype's patterns."""
    itemsize = REQUIRED[dtype][1]
    return PATTERNS[dtype][: count_of(shape) * itemsize]


# ---------------------------------------------------------------------------
# DTYPES


def test_dtypes_hold_the_nine_required_dtypes_with_their_codes_and_sizes() -> None:
    dtypes = lassi_io().DTYPES
    found = {name: (dtypes[name].code, dtypes[name].itemsize) for name in REQUIRED if name in dtypes}
    assert found == REQUIRED
    for name in REQUIRED:
        assert dtypes[name].name == name


def test_dtype_codes_are_unique_and_never_zero() -> None:
    dtypes = lassi_io().DTYPES
    codes = [dtype.code for dtype in dtypes.values()]
    assert 0 not in codes
    assert len(codes) == len(set(codes)), f"two dtypes share a code: {sorted(codes)}"
    assert all(key == dtype.name for key, dtype in dtypes.items())


# ---------------------------------------------------------------------------
# The exact bytes


@pytest.mark.parametrize("example", sorted(EXAMPLES))
def test_the_writer_produces_the_hand_written_bytes(example: str, tmp_path: Path) -> None:
    name, dtype, shape, data, expected = EXAMPLES[example]
    path = tmp_path / f"{name}.lassiio"
    lassi_io().write_array(path, name, dtype, shape, data)
    assert path.read_bytes().hex() == expected


@pytest.mark.parametrize("example", sorted(EXAMPLES))
def test_the_reader_reads_the_hand_written_bytes(example: str, tmp_path: Path) -> None:
    name, dtype, shape, data, expected = EXAMPLES[example]
    path = tmp_path / "example.lassiio"
    path.write_bytes(bytes.fromhex(expected))
    module = lassi_io()
    array = module.read_array(path)
    assert (array.name, array.dtype, array.shape, array.data) == (name, dtype, shape, data)
    assert array == module.LassiArray(name=name, dtype=dtype, shape=shape, data=data)
    assert isinstance(array.shape, tuple) and isinstance(array.data, bytes)


def test_the_data_starts_at_the_next_multiple_of_8_after_the_name(tmp_path: Path) -> None:
    module = lassi_io()
    wrong = []
    for rank in range(4):
        shape = tuple(range(1, rank + 1))
        for length in range(1, 18):
            name = "n" * length
            data = data_for("u8", shape)
            path = tmp_path / f"r{rank}-n{length}.lassiio"
            module.write_array(path, name, "u8", shape, data)
            expected = header(name.encode("ascii"), 7, shape) + data
            if path.read_bytes() != expected:
                wrong.append((rank, length))
    assert not wrong, f"(rank, name length) pairs whose bytes differ from the hand-built header: {wrong}"


# ---------------------------------------------------------------------------
# Round trips


@pytest.mark.parametrize("dtype", sorted(REQUIRED))
def test_every_dtype_round_trips_in_every_shape(dtype: str, tmp_path: Path) -> None:
    module = lassi_io()
    for index, shape in enumerate(SHAPES):
        data = data_for(dtype, shape)
        path = tmp_path / f"{dtype}-{index}.lassiio"
        module.write_array(path, f"{dtype}_{index}", dtype, shape, data)
        array = module.read_array(path)
        assert array == module.LassiArray(f"{dtype}_{index}", dtype, shape, data), shape
        head = header(f"{dtype}_{index}".encode("ascii"), REQUIRED[dtype][0], shape)
        assert path.read_bytes() == head + data, shape


@pytest.mark.parametrize(("dtype", "fmt"), [("bf16", "<H"), ("f16", "<H")])
def test_every_16_bit_pattern_round_trips_bit_for_bit(dtype: str, fmt: str, tmp_path: Path) -> None:
    # Every NaN (quiet, signaling, either sign, any payload), both infinities, -0, and every subnormal.
    data = b"".join(struct.pack(fmt, bits) for bits in range(1 << 16))
    path = tmp_path / f"{dtype}.lassiio"
    module = lassi_io()
    module.write_array(path, f"all_{dtype}", dtype, (256, 256), data)
    array = module.read_array(path)
    assert (array.dtype, array.shape) == (dtype, (256, 256))
    assert array.data == data


@pytest.mark.parametrize("name", GOOD_NAMES, ids=lambda name: name if len(name) < 20 else f"{len(name)} bytes")
def test_names_of_the_allowed_characters_round_trip(name: str, tmp_path: Path) -> None:
    path = tmp_path / "named.lassiio"
    module = lassi_io()
    module.write_array(path, name, "i8", (2,), b"\x01\xff")
    assert module.read_array(path).name == name


def test_the_reader_and_writer_take_a_str_or_a_path(tmp_path: Path) -> None:
    module = lassi_io()
    data = struct.pack("<3i", 7, -8, 9)
    module.write_array(str(tmp_path / "as-str.lassiio"), "k", "i32", [3], data)
    assert module.read_array(tmp_path / "as-str.lassiio") == module.LassiArray("k", "i32", (3,), data)
    module.write_array(tmp_path / "as-path.lassiio", "k", "i32", (3,), data)
    assert module.read_array(str(tmp_path / "as-path.lassiio")) == module.LassiArray("k", "i32", (3,), data)
