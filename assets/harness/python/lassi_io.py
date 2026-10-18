"""lassi_io.py: read and write one named array per binary file (LASSI Harness Contract), for Python programs.

The Python binding of the harness I/O contract, beside assets/harness/c/lassi_io.h (task P12.3). It is a
harness file, copied beside a program and imported as lassi_io, so it uses the standard library of Python
3.10 only and imports nothing from lassi. A program reads its inputs with read_array and writes its outputs
with write_array; the harness writes the inputs and reads the outputs with lassi/harness/lassi_io.py, which
reads and writes the same bytes, and this module refuses what that module refuses.

File format, version 1, little-endian throughout, offsets from the start of the file:

    0      8 bytes     magic: the ASCII text LASSIIO and one zero byte
    8      u32         version, 1
    12     u32         dtype code (DTYPES)
    16     u32         rank
    20     u64 x rank  dims, in C order
    then   u32         name length in bytes, 1 to 255
    then   the name    each byte an ASCII letter, digit, '_', '.', or '-'
    then   zero bytes up to the next multiple of 8, so the data is 8-byte aligned
    then   the data    product(dims) * itemsize bytes in C order, and nothing after it

A rank-0 array is a scalar of one element; a dim of 0 gives an empty array with no data bytes. No value is
converted: an element is its little-endian bit pattern, and a bf16 element is the upper 16 bits of the f32
of the same value.

The array type. read_array returns a LassiArray: name (str), dtype (a DTYPES key), shape (a tuple of ints,
empty for a scalar), and data, a memoryview of the element bytes: writable, one-dimensional (the elements in
C order, so a program indexes a 2-D array as data[i * cols + j]), and cast to the dtype's view format:
f32 'f', f64 'd', i8 'b', u8 'B', i32 'i', u32 'I', i64 'q', and f16 and bf16 'H', their 16-bit patterns,
since Python 3.10's memoryview has no half or bfloat16 format. data[i] is then the element's value (a float
or an int; for f16 and bf16 its bit pattern), and bytes(data) the file's element bytes, unchanged. File I/O
converts nothing, but reading or assigning an f32 element through the view goes through a Python float, so a
signaling NaN copied element by element comes back quiet; copy the bytes to pass bit patterns through. A
LassiArray compares by identity, since a memoryview compares element values and a NaN equals nothing.

write_array(path, name, dtype, shape, data) takes data as bytes, bytearray, or a memoryview of any format,
the data of a LassiArray included, and writes its bytes as they are. A program makes an output's data as
memoryview(bytearray(count * itemsize)).cast(view_format), with both from DTYPES[dtype].

prange is the builtin range itself. A program imports it from here and writes its parallel loops as
"for i in prange(n)"; natively they run as range loops under plain Python, with no Numba, and the
Python-subset frontend (task P12.4) reads that import as the parallel-for.

Refusals: LassiIOError, a ValueError, is the one error of a refusal; its message starts with the file's path
and says why. read_array refuses a bad magic, a version other than 1, an unknown dtype code, a file that ends
inside its header, a bad name length or name, nonzero padding, and data longer or shorter than the dims say;
it checks each header field against the file's size before using it, and it multiplies the dims only until the
product passes the bytes that follow the header (a zero dim gives 0 at once), so its cost is linear in the
rank. write_array refuses data that is not bytes, a bytearray, or a memoryview, an unknown dtype, a bad shape
(a negative or non-int dim, or one past 2**64 - 1), a bad name, and data whose length is not product(shape) *
itemsize, checked the same way against the data's length, before it opens the file. No message formats a huge
number. Both refuse on a big-endian host, as lassi_io.h does, since the data's view reads its bytes in the
host's order. A file that cannot be opened is an OSError, not a refusal.
"""

from __future__ import annotations

import os
import re
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

MAGIC = b"LASSIIO\x00"
VERSION = 1
MAX_NAME_BYTES = 255
ALIGNMENT = 8
# A name: 1 to 255 bytes, each an ASCII letter, digit, '_', '.', or '-'.
_NAME = re.compile(r"[A-Za-z0-9_.-]{1,255}")
_NAME_BYTES = re.compile(rb"[A-Za-z0-9_.-]{1,255}")
_NAME_RULE = f"a name is 1 to {MAX_NAME_BYTES} bytes, each an ASCII letter, digit, '_', '.', or '-'"
_M32 = (1 << 32) - 1
_M64 = (1 << 64) - 1


@dataclass(frozen=True)
class DType:
    """One element type: its name, the u32 code written in the file, its size in bytes, and its view format."""

    name: str
    code: int
    itemsize: int
    view_format: str


# The dtypes, by name, with the Harness Contract's codes and sizes; code 0 is never a dtype.
DTYPES: Mapping[str, DType] = MappingProxyType(
    {
        dtype.name: dtype
        for dtype in (
            DType("f32", 1, 4, "f"),
            DType("f16", 2, 2, "H"),
            DType("bf16", 3, 2, "H"),
            DType("i32", 4, 4, "i"),
            DType("u32", 5, 4, "I"),
            DType("i8", 6, 1, "b"),
            DType("u8", 7, 1, "B"),
            DType("i64", 8, 8, "q"),
            DType("f64", 9, 8, "d"),
        )
    }
)
_BY_CODE = {dtype.code: dtype for dtype in DTYPES.values()}

# The parallel-for of the Python subset: the builtin range itself (see the module docstring).
prange = range


class LassiIOError(ValueError):
    """A lassi_io refusal: a file or an argument that is not exactly valid."""


@dataclass(frozen=True, eq=False)
class LassiArray:
    """One named array as a file holds it.

    dtype is a DTYPES key, shape a tuple of ints (empty for a scalar), and data a writable, one-dimensional
    memoryview of the element bytes in C order, cast to the dtype's view format (see the module docstring).
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    data: memoryview


def write_array(path: str | os.PathLike[str], name: str, dtype: str, shape: Sequence[int], data: object) -> None:
    """Write one named array to `path` in the lassi_io format, replacing any file there.

    `dtype` is a DTYPES key, `shape` a sequence of non-negative ints (empty for a scalar), and `data` the
    little-endian element bytes in C order (bytes, bytearray, or a memoryview, a LassiArray's data
    included). Raises LassiIOError naming the path, before the file is opened, on a big-endian host and for
    data that is not bytes, a bytearray, or a memoryview, an unknown dtype, a bad shape or name, or data whose
    length is not product(shape) * itemsize.
    """
    where = os.fspath(path)
    _check_host(where)
    blob = _encode(where, name, dtype, shape, data)
    with open(where, "wb") as handle:
        handle.write(blob)


def read_array(path: str | os.PathLike[str]) -> LassiArray:
    """Read one named array from the lassi_io file at `path`.

    Raises LassiIOError naming the path on a big-endian host and for any file that is not exactly a valid
    version 1 file (see the module docstring); OSError when the file cannot be opened or read.
    """
    where = os.fspath(path)
    _check_host(where)
    with open(where, "rb") as handle:
        raw = handle.read()
    return _decode(where, raw)


def _refusal(where: str, reason: str) -> LassiIOError:
    """Return the error for a refusal of `where`: its path, then the reason."""
    return LassiIOError(f"{where}: {reason}")


def _check_host(where: str) -> None:
    """Raise LassiIOError naming `where` on a big-endian host, where the data's view would read its bytes reversed."""
    if sys.byteorder != "little":
        raise _refusal(where, "this host is big-endian; lassi_io files are little-endian")


def _shown(value: object) -> str:
    """Return an ASCII rendering of `value` for a message, cut after 60 characters.

    It never fails and stays fast: an int wider than 128 bits is shown by its width and a list or tuple of
    more than 8 items by its length, since str() of a huge int is slow and refused past Python's digit limit.
    """
    if isinstance(value, int) and value.bit_length() > 128:
        return f"<an int of {value.bit_length()} bits>"
    if isinstance(value, (list, tuple)) and len(value) > 8:
        return f"<a {type(value).__name__} of {len(value)} items>"
    try:
        text = ascii(value)
    except ValueError:  # a nested int past the digit limit
        return f"<a {type(value).__name__} too large to show>"
    return text if len(text) <= 60 else f"{text[:60]}... ({len(text)} characters)"


def _byte_count(dims: Sequence[int], itemsize: int, limit: int) -> int | None:
    """Return product(dims) * itemsize, or None as soon as it exceeds `limit`; a zero dim gives 0 at once."""
    if 0 in dims:
        return 0
    total = itemsize
    for dim in dims:
        total *= dim
        if total > limit:
            return None
    return total


def _checked_dims(where: str, shape: object) -> tuple[int, ...]:
    """Return `shape` as a tuple of dims, or raise LassiIOError for a bad shape."""
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        raise _refusal(where, f"bad shape {_shown(shape)}: a shape is a sequence of ints")
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise _refusal(where, f"bad shape {_shown(shape)}: dim {_shown(dim)} is not an int")
        if dim < 0:
            raise _refusal(where, f"bad shape {_shown(shape)}: negative dim {_shown(dim)}")
        if dim > _M64:
            raise _refusal(where, f"bad shape {_shown(shape)}: dim {_shown(dim)} does not fit in a u64")
    if len(shape) > _M32:
        raise _refusal(where, f"bad shape: rank {len(shape)} does not fit in a u32")
    return tuple(shape)


def _encode(where: str, name: object, dtype: object, shape: object, data: object) -> bytes:
    """Return the file bytes of one array, or raise LassiIOError naming `where` for a bad argument."""
    if not isinstance(name, str):
        raise _refusal(where, f"bad name: an array name is a str, not {type(name).__name__}")
    if not _NAME.fullmatch(name):
        raise _refusal(where, f"bad name {_shown(name)}: {_NAME_RULE}")
    if not isinstance(dtype, str) or dtype not in DTYPES:
        raise _refusal(where, f"unknown dtype {_shown(dtype)}; the dtypes are {', '.join(DTYPES)}")
    dims = _checked_dims(where, shape)
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise _refusal(where, f"bad data: the element bytes are bytes-like, not {type(data).__name__}")
    raw = bytes(data)
    kind = DTYPES[dtype]
    expected = _byte_count(dims, kind.itemsize, len(raw))
    if expected != len(raw):
        need = f"more than {len(raw)}" if expected is None else str(expected)
        raise _refusal(where, f"data size: shape {_shown(dims)} of {dtype} needs {need} bytes, got {len(raw)} bytes")
    encoded = name.encode("ascii")
    head = MAGIC + struct.pack("<III", VERSION, kind.code, len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
    head += struct.pack("<I", len(encoded)) + encoded
    return head + bytes(-len(head) % ALIGNMENT) + raw


def _u32_at(where: str, raw: bytes, offset: int, field: str) -> int:
    """Return the u32 at `offset`, or raise LassiIOError when the file ends before it."""
    if offset + 4 > len(raw):
        raise _refusal(where, f"truncated: the file ends inside its header, at its {field} ({len(raw)} bytes)")
    return struct.unpack_from("<I", raw, offset)[0]


def _decode(where: str, raw: bytes) -> LassiArray:
    """Return the array a file's bytes hold, or raise LassiIOError naming `where` (the checks run in file order)."""
    size = len(raw)
    if raw[:8] != MAGIC:
        if size < 8 and MAGIC.startswith(raw):
            raise _refusal(where, f"truncated: the file ends inside its magic ({size} bytes)")
        raise _refusal(where, f"bad magic {_shown(raw[:8])}: a lassi_io file starts with {ascii(MAGIC)}")
    version = _u32_at(where, raw, 8, "version")
    if version != VERSION:
        raise _refusal(where, f"unsupported version {version}: this reader reads version {VERSION}")
    code = _u32_at(where, raw, 12, "dtype code")
    if code not in _BY_CODE:
        raise _refusal(where, f"unknown dtype code {code}; the codes are 1 to {max(_BY_CODE)}")
    dtype = _BY_CODE[code]
    rank = _u32_at(where, raw, 16, "rank")
    dims_end = 20 + 8 * rank
    if dims_end > size:
        raise _refusal(where, f"truncated: rank {rank} needs {8 * rank} bytes of dims, the file has {size} bytes")
    dims = struct.unpack_from(f"<{rank}Q", raw, 20)
    length = _u32_at(where, raw, dims_end, "name length")
    if not 1 <= length <= MAX_NAME_BYTES:
        raise _refusal(where, f"bad name length {length}: {_NAME_RULE}")
    name_end = dims_end + 4 + length
    if name_end > size:
        raise _refusal(where, f"truncated: the file ends inside its name ({size} bytes)")
    name = raw[dims_end + 4 : name_end]
    if not _NAME_BYTES.fullmatch(name):
        raise _refusal(where, f"bad name {_shown(name)}: {_NAME_RULE}")
    header_end = name_end + (-name_end % ALIGNMENT)
    if header_end > size:
        raise _refusal(where, f"truncated: the file ends inside the padding after its name ({size} bytes)")
    if any(raw[name_end:header_end]):
        raise _refusal(where, "bad padding: the bytes between the name and the data must be zero")
    available = size - header_end
    expected = _byte_count(dims, dtype.itemsize, available)
    if expected != available:
        need = f"more than {available}" if expected is None else str(expected)
        raise _refusal(
            where,
            f"data size: dims {_shown(dims)} of {dtype.name} need {need} bytes, "
            f"but {available} bytes follow the header",
        )
    data = memoryview(bytearray(memoryview(raw)[header_end:])).cast(dtype.view_format)
    return LassiArray(name.decode("ascii"), dtype.name, dims, data)
