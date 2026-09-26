"""The lassi_io file format, its reader and writer, and the seeded generator of held-out inputs.

Bible: Harness Contract (programs read inputs and write outputs as binary
files through assets/harness/c/lassi_io.h; the harness owns inputs and
comparison; reward and evaluation use held-out inputs), Frontend Rules
(harness bindings per language). This module is the control plane's side of
that contract: it writes the files a program reads, reads the files it
writes, and generates held-out inputs from a seed. The C and C++ header reads
and writes the same bytes.

File format, version 1 [DESIGN, task P4.3]. One file holds one named array.
Little-endian throughout; offsets from the start of the file:

    0      8 bytes     magic: the ASCII text LASSIIO and one zero byte
    8      u32         version, 1
    12     u32         dtype code (DTYPES)
    16     u32         rank
    20     u64 x rank  dims, in C order
    then   u32         name length in bytes, 1 to 255
    then   the name    each byte an ASCII letter, digit, '_', '.', or '-'
    then   zero bytes up to the next multiple of 8, so the data is 8-byte aligned
    then   the data    product(dims) * itemsize bytes in C order, and nothing after it

A rank-0 array is a scalar of one element; a dim of 0 gives an empty array
with no data bytes. The project has no array library, so the API carries raw
bytes: an element is its little-endian bit pattern (a bf16 element is the
upper 16 bits of the f32 of the same value), and no value is converted.

Refusals: LassiIOError, a ValueError, is the one error of a refusal; its
message starts with the file's path and says why. read_array refuses a bad
magic, a version other than 1, an unknown dtype code, a file that ends inside
its header, a bad name length or name, nonzero padding, and data longer or
shorter than the dims say; it checks each header field against the file's
size before using it, so no field makes it read or allocate more than the
file holds, and it multiplies the dims only until the product passes the
bytes that follow the header (a zero dim gives 0 at once). write_array
refuses an unknown dtype, a bad shape (a negative or non-int dim, or one
past 2**64 - 1), a bad name, and data whose length is not product(shape) *
itemsize, checked the same way against the data's length, before it opens
the file. No message formats a huge number, so every refusal is a
LassiIOError naming the path. A file that cannot be opened is an OSError,
not a refusal.

Held-out inputs: generate_inputs(spec, seed, out_dir) writes one file per
spec entry, <name>.lassiio, with PCG-XSH-RR 64/32 draws (Pcg32) on a stream of
its own per input; the rules for turning draws into values are in its
docstring.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

MAGIC = b"LASSIIO\x00"
VERSION = 1
MAX_NAME_BYTES = 255
ALIGNMENT = 8
SUFFIX = ".lassiio"
# generate_inputs names a file <name>.lassiio, which must fit a 255-byte file name.
MAX_INPUT_NAME_BYTES = 255 - len(SUFFIX)
# A name: 1 to 255 bytes, each an ASCII letter, digit, '_', '.', or '-'.
_NAME = re.compile(r"[A-Za-z0-9_.-]{1,255}")
_NAME_BYTES = re.compile(rb"[A-Za-z0-9_.-]{1,255}")
_NAME_RULE = f"a name is 1 to {MAX_NAME_BYTES} bytes, each an ASCII letter, digit, '_', '.', or '-'"
_M32 = (1 << 32) - 1
_M64 = (1 << 64) - 1


@dataclass(frozen=True)
class DType:
    """One element type: its name, the u32 code written in the file, and its size in bytes."""

    name: str
    code: int
    itemsize: int


# The dtypes, by name. Codes follow the order the task lists them; code 0 is never a dtype.
DTYPES: Mapping[str, DType] = MappingProxyType(
    {
        dtype.name: dtype
        for dtype in (
            DType("f32", 1, 4),
            DType("f16", 2, 2),
            DType("bf16", 3, 2),
            DType("i32", 4, 4),
            DType("u32", 5, 4),
            DType("i8", 6, 1),
            DType("u8", 7, 1),
            DType("i64", 8, 8),
            DType("f64", 9, 8),
        )
    }
)
_BY_CODE = {dtype.code: dtype for dtype in DTYPES.values()}


class LassiIOError(ValueError):
    """A lassi_io refusal: a file, an argument, or an input spec that is not exactly valid."""


@dataclass(frozen=True)
class LassiArray:
    """One named array as a file holds it.

    dtype is a DTYPES key, shape a tuple of ints (empty for a scalar), and data
    the little-endian element bytes in C order. Instances compare by value; a
    list shape is stored as a tuple and a bytes-like data as bytes.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        """Store shape as a tuple and data as bytes, so equal arrays compare equal."""
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "data", bytes(self.data))


# ---------------------------------------------------------------------------
# Reading and writing


def write_array(path: str | os.PathLike[str], name: str, dtype: str, shape: Sequence[int], data: bytes) -> None:
    """Write one named array to `path` in the lassi_io format, replacing any file there.

    `dtype` is a DTYPES key, `shape` a sequence of non-negative ints (empty for
    a scalar), and `data` the little-endian element bytes in C order (bytes,
    bytearray, or memoryview). Raises LassiIOError naming the path, before the
    file is opened, for an unknown dtype, a bad shape or name, or data whose
    length is not product(shape) * itemsize.
    """
    where = os.fspath(path)
    blob = _encode(where, name, dtype, shape, data)
    with open(where, "wb") as handle:
        handle.write(blob)


def read_array(path: str | os.PathLike[str]) -> LassiArray:
    """Read one named array from the lassi_io file at `path`.

    Raises LassiIOError naming the path for any file that is not exactly a
    valid version 1 file (see the module docstring); OSError when the file
    cannot be opened or read.
    """
    where = os.fspath(path)
    with open(where, "rb") as handle:
        raw = handle.read()
    return _decode(where, raw)


def _refusal(where: str, reason: str) -> LassiIOError:
    """Return the error for a refusal of `where`: its path, then the reason."""
    return LassiIOError(f"{where}: {reason}")


def _shown(value: object) -> str:
    """Return an ASCII rendering of `value` for a message, cut after 60 characters.

    It never fails and stays fast: an int wider than 128 bits is shown by its
    width and a list or tuple of more than 8 items by its length, since str()
    of a huge int is slow and refused past Python's digit limit.
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
    """Return product(dims) * itemsize, or None as soon as it exceeds `limit`.

    A zero dim gives 0 at once. The product is never built past `limit` times
    one dim, so a huge shape costs one pass over its dims and no big-number work.
    """
    if 0 in dims:
        return 0
    total = itemsize
    for dim in dims:
        total *= dim
        if total > limit:
            return None
    return total


def _name_problem(name: object) -> str:
    """Return why `name` is not a valid array name, or "" when it is."""
    if not isinstance(name, str):
        return f"bad name: an array name is a str, not {type(name).__name__}"
    if not _NAME.fullmatch(name):
        return f"bad name {_shown(name)}: {_NAME_RULE}"
    return ""


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


def _encode(where: str, name: str, dtype: str, shape: Sequence[int], data: bytes) -> bytes:
    """Return the file bytes of one array, or raise LassiIOError naming `where` for a bad argument."""
    problem = _name_problem(name)
    if problem:
        raise _refusal(where, problem)
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
    return LassiArray(name.decode("ascii"), dtype.name, dims, raw[header_end:])


# ---------------------------------------------------------------------------
# The generator


class Pcg32:
    """PCG-XSH-RR 64/32, the pcg32 generator of the PCG reference implementation, with its seeding.

    Pcg32(initstate, initseq) seeds as pcg32_srandom_r does: state = 0,
    inc = (initseq << 1) | 1 mod 2**64, a step, state += initstate, a step. A
    step is state = state * 6364136223846793005 + inc mod 2**64; next_u32()
    steps and returns the XSH-RR output of the state before the step. Both
    seeds are ints in [0, 2**64).
    """

    MULTIPLIER = 6364136223846793005

    def __init__(self, initstate: int, initseq: int) -> None:
        """Seed the generator; raise ValueError for a seed outside [0, 2**64)."""
        for label, value in (("initstate", initstate), ("initseq", initseq)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _M64:
                raise ValueError(f"Pcg32 {label} must be an int in [0, 2**64), got {_shown(value)}")
        self._state = 0
        self._inc = ((initseq << 1) | 1) & _M64
        self.next_u32()
        self._state = (self._state + initstate) & _M64
        self.next_u32()

    def next_u32(self) -> int:
        """Step the state and return the XSH-RR output of the state before the step."""
        old = self._state
        self._state = (old * self.MULTIPLIER + self._inc) & _M64
        shifted = (((old >> 18) ^ old) >> 27) & _M32
        rotation = old >> 59
        return ((shifted >> rotation) | (shifted << ((-rotation) & 31))) & _M32

    def next_u64(self) -> int:
        """Return two draws as one u64, the first draw in the high 32 bits."""
        high = self.next_u32()
        return (high << 32) | self.next_u32()


def stream_of(name: str) -> int:
    """Return an input's PCG stream (initseq): the first 8 bytes of sha256(name as ASCII), little-endian."""
    return int.from_bytes(hashlib.sha256(name.encode("ascii")).digest()[:8], "little")


@dataclass(frozen=True)
class _FloatFormat:
    """A binary floating-point format: exponent and fraction widths, and 32-bit draws per uniform element."""

    exp_bits: int
    frac_bits: int
    draws: int

    @property
    def width(self) -> int:
        """Return the format's width in bits."""
        return 1 + self.exp_bits + self.frac_bits

    @property
    def bias(self) -> int:
        """Return the exponent bias."""
        return (1 << (self.exp_bits - 1)) - 1


_FLOATS = {
    "f16": _FloatFormat(5, 10, 1),
    "bf16": _FloatFormat(8, 7, 1),
    "f32": _FloatFormat(8, 23, 1),
    "f64": _FloatFormat(11, 52, 2),
}
# Integer dtypes: struct code and signedness.
_INTS = {"i8": ("b", True), "u8": ("B", False), "i32": ("i", True), "u32": ("I", False), "i64": ("q", True)}
_BITS_CODE = {16: "H", 32: "I", 64: "Q"}
_SPEC_KEYS = frozenset({"name", "dtype", "shape", "dist", "lo", "hi"})
_DISTS = {"uniform": _FLOATS, "integers": _INTS}


def _round_bits(value: float, fmt: _FloatFormat) -> int:
    """Return the bit pattern of `value` rounded to the format, to nearest with ties to even.

    `value` is finite and no larger in magnitude than the format's largest
    finite value, so the result is finite. The rounding is exact integer
    arithmetic on the double's significand, the same on every host.
    """
    sign = (1 << (fmt.width - 1)) if math.copysign(1.0, value) < 0 else 0
    if value == 0.0:
        return sign
    mantissa, exponent = math.frexp(abs(value))  # abs(value) = mantissa * 2**exponent, 0.5 <= mantissa < 1
    significand = int(mantissa * (1 << 53))  # exact: abs(value) = significand * 2**(exponent - 53)
    # The result is units * 2**scale, scale being the last place of the value's binade, or of the subnormals.
    scale = max(exponent - 1 - fmt.frac_bits, 1 - fmt.bias - fmt.frac_bits)
    shift = scale - (exponent - 53)
    units, rest = divmod(significand, 1 << shift)
    if shift > 0:
        half = 1 << (shift - 1)
        if rest > half or (rest == half and units & 1):
            units += 1
    if units >> (fmt.frac_bits + 1):  # the rounding carried into the next binade
        units >>= 1
        scale += 1
    if units < (1 << fmt.frac_bits):  # a subnormal, or zero
        return sign | units
    biased = scale + fmt.frac_bits + fmt.bias
    if biased >= (1 << fmt.exp_bits) - 1:
        raise OverflowError(f"{value!r} overflows the format")
    return sign | (biased << fmt.frac_bits) | (units - (1 << fmt.frac_bits))


def _bits_value(bits: int, fmt: _FloatFormat) -> float:
    """Return the value of a finite bit pattern of the format, exactly, as a double."""
    sign = -1.0 if bits >> (fmt.width - 1) else 1.0
    field = (bits >> fmt.frac_bits) & ((1 << fmt.exp_bits) - 1)
    fraction = bits & ((1 << fmt.frac_bits) - 1)
    if field == 0:
        return sign * math.ldexp(fraction, 1 - fmt.bias - fmt.frac_bits)
    return sign * math.ldexp(fraction | (1 << fmt.frac_bits), field - fmt.bias - fmt.frac_bits)


def _key(bits: int, fmt: _FloatFormat) -> int:
    """Return an int that orders finite bit patterns as their values are ordered (both zeros give 0)."""
    magnitude = bits & ((1 << (fmt.width - 1)) - 1)
    return -magnitude if bits >> (fmt.width - 1) else magnitude


def _from_key(key: int, fmt: _FloatFormat) -> int:
    """Return the bit pattern of an ordering key (0 gives +0)."""
    return key if key >= 0 else (1 << (fmt.width - 1)) | -key


@dataclass(frozen=True)
class _InputPlan:
    """One checked spec entry. For floats, lo_key and hi_key order the dtype values the input may take."""

    name: str
    dtype: DType
    dims: tuple[int, ...]
    dist: str
    lo: float | int
    hi: float | int
    lo_key: int = 0
    hi_key: int = 0


def _float_plan(where: str, name: str, dtype: DType, dims: tuple[int, ...], lo: Any, hi: Any) -> _InputPlan:
    """Check a uniform entry's bounds against its float dtype and return its plan."""
    fmt = _FLOATS[dtype.name]
    bounds = []
    for label, value in (("lo", lo), ("hi", hi)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _refusal(where, f"{label} {_shown(value)} is not a number")
        try:
            bound = float(value)
        except OverflowError:
            bound = math.nan
        if isinstance(value, int) and bound != value:
            raise _refusal(where, f"{label} {_shown(value)} is not exactly a double")
        bounds.append(bound)
    low, high = bounds
    if not (math.isfinite(low) and math.isfinite(high)):
        raise _refusal(where, f"the bounds must be finite, got lo {low!r} and hi {high!r}")
    if low >= high:
        raise _refusal(where, f"lo {low!r} must be below hi {high!r}")
    largest = _bits_value((((1 << fmt.exp_bits) - 2) << fmt.frac_bits) | ((1 << fmt.frac_bits) - 1), fmt)
    if max(abs(low), abs(high)) > largest:
        raise _refusal(where, f"[{low!r}, {high!r}) reaches past the finite range of {dtype.name}")
    if not math.isfinite(high - low):
        raise _refusal(where, f"hi - lo overflows a double for [{low!r}, {high!r})")
    lo_bits = _round_bits(low, fmt)
    lo_key = _key(lo_bits, fmt) + (1 if _bits_value(lo_bits, fmt) < low else 0)
    hi_bits = _round_bits(high, fmt)
    hi_key = _key(hi_bits, fmt) - (1 if _bits_value(hi_bits, fmt) >= high else 0)
    if lo_key > hi_key:
        raise _refusal(where, f"no {dtype.name} value lies in [{low!r}, {high!r})")
    return _InputPlan(name, dtype, dims, "uniform", low, high, lo_key, hi_key)


def _int_plan(where: str, name: str, dtype: DType, dims: tuple[int, ...], lo: Any, hi: Any) -> _InputPlan:
    """Check an integers entry's bounds against its integer dtype and return its plan."""
    for label, value in (("lo", lo), ("hi", hi)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise _refusal(where, f"{label} {_shown(value)} is not an int")
    if lo >= hi:
        raise _refusal(where, f"lo {_shown(lo)} must be below hi {_shown(hi)}")
    signed = _INTS[dtype.name][1]
    bits = 8 * dtype.itemsize
    smallest, largest = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if lo < smallest or hi - 1 > largest:
        raise _refusal(
            where, f"[{_shown(lo)}, {_shown(hi)}) does not fit {dtype.name}, which holds [{smallest}, {largest}]"
        )
    return _InputPlan(name, dtype, dims, "integers", lo, hi)


def _plan_entry(where: str, index: int, entry: object) -> _InputPlan:
    """Check one spec entry and return its plan, or raise LassiIOError naming the entry."""
    if not isinstance(entry, Mapping):
        raise _refusal(where, f"input spec entry {index} is not a mapping")
    name = entry.get("name")
    at = f"{where}: input spec entry {index} ({_shown(name)})"
    if set(entry) != _SPEC_KEYS:
        missing, extra = sorted(_SPEC_KEYS - set(entry)), sorted(map(_shown, set(entry) - _SPEC_KEYS))
        raise LassiIOError(f"{at}: the keys must be exactly {sorted(_SPEC_KEYS)}; missing {missing}, extra {extra}")
    problem = _name_problem(name)
    if problem:
        raise LassiIOError(f"{at}: {problem}")
    if len(name) > MAX_INPUT_NAME_BYTES:
        raise LassiIOError(
            f"{at}: the name is {len(name)} bytes; an input's name is at most {MAX_INPUT_NAME_BYTES} bytes, "
            f"so that <name>{SUFFIX} fits a 255-byte file name"
        )
    dtype, dist = entry["dtype"], entry["dist"]
    if not isinstance(dtype, str) or dtype not in DTYPES:
        raise LassiIOError(f"{at}: unknown dtype {_shown(dtype)}; the dtypes are {', '.join(DTYPES)}")
    if not isinstance(dist, str) or dist not in _DISTS:
        raise LassiIOError(f"{at}: unknown dist {_shown(dist)}; the dists are {', '.join(_DISTS)}")
    if dtype not in _DISTS[dist]:
        raise LassiIOError(f"{at}: dist {dist} draws {', '.join(_DISTS[dist])}, not {dtype}")
    dims = _checked_dims(at, entry["shape"])
    plan = _float_plan if dist == "uniform" else _int_plan
    return plan(at, name, DTYPES[dtype], dims, entry["lo"], entry["hi"])


def _uniform_data(plan: _InputPlan, rng: Pcg32) -> bytes:
    """Draw a uniform float input's element bytes (rules in generate_inputs)."""
    fmt = _FLOATS[plan.dtype.name]
    low, width = float(plan.lo), float(plan.hi) - float(plan.lo)
    out = []
    for _ in range(math.prod(plan.dims)):
        if fmt.draws == 1:
            fraction = (rng.next_u32() >> 8) * 2.0**-24
        else:
            fraction = (rng.next_u64() >> 11) * 2.0**-53
        bits = _round_bits(low + width * fraction, fmt)
        key = _key(bits, fmt)
        if key < plan.lo_key or key > plan.hi_key:
            bits = _from_key(min(max(key, plan.lo_key), plan.hi_key), fmt)
        out.append(bits)
    return struct.pack(f"<{len(out)}{_BITS_CODE[fmt.width]}", *out)


def _bounded(rng: Pcg32, bound: int) -> int:
    """Return an unbiased draw in [0, bound) by rejection, as pcg32_boundedrand_r does, for bound <= 2**64."""
    if bound <= 1 << 32:
        threshold = (1 << 32) % bound
        while True:
            draw = rng.next_u32()
            if draw >= threshold:
                return draw % bound
    threshold = (1 << 64) % bound
    while True:
        draw = rng.next_u64()
        if draw >= threshold:
            return draw % bound


def _integer_data(plan: _InputPlan, rng: Pcg32) -> bytes:
    """Draw an integers input's element bytes (rules in generate_inputs)."""
    low, bound = int(plan.lo), int(plan.hi) - int(plan.lo)
    values = [low + _bounded(rng, bound) for _ in range(math.prod(plan.dims))]
    return struct.pack(f"<{len(values)}{_INTS[plan.dtype.name][0]}", *values)


def generate_inputs(spec: Sequence[Mapping[str, Any]], seed: int, out_dir: str | os.PathLike[str]) -> dict[str, Path]:
    """Write one lassi_io file per spec entry into the existing directory `out_dir`; return name -> path.

    Each entry is a mapping with exactly the keys name, dtype, shape, dist,
    lo, and hi, as a suite manifest holds them; the file is <name>.lassiio and
    holds the array under that name. `seed` is an int in [0, 2**64).

    Rules [DESIGN, task P4.3], the same bytes for the same seed on every host:

    - Each input has its own Pcg32 stream: initstate = seed, initseq =
      stream_of(name). Its bytes depend on the seed and its own entry alone.
      Elements are drawn in C order.
    - "uniform" (f32, f16, bf16, f64) draws floats in [lo, hi): an element is
      lo + (hi - lo) * u computed in double, where u = (r >> 8) * 2**-24 from
      one draw r for f32, f16, and bf16, and u = (d >> 11) * 2**-53 from one
      u64 d (two draws, the first in the high half) for f64; it is rounded to
      the dtype to nearest, ties to even. A value that rounds outside
      [lo, hi) becomes the nearest dtype value inside it, so it is never hi.
      lo and hi are finite numbers inside the dtype's finite range, and some
      dtype value must lie in [lo, hi).
    - "integers" (i8, u8, i32, u32, i64) draws integers in [lo, hi): an
      element is lo + an unbiased draw in [0, hi - lo) made by rejection as
      pcg32_boundedrand_r does: with n = hi - lo <= 2**32, draw r until
      r >= 2**32 mod n and take r mod n; with a wider n, the same with u64
      draws and 2**64. lo and hi are ints and [lo, hi) lies in the dtype's
      range.

    Raises LassiIOError (a ValueError), before any file is written, for a
    seed outside [0, 2**64) or a bool, an out_dir that is not a directory, an
    entry that breaks the rules above (lo >= hi, a non-finite bound, an
    unknown dist or dtype, a dist that does not draw the dtype, a bad name,
    a name over MAX_INPUT_NAME_BYTES (247, so <name>.lassiio fits a 255-byte
    file name), a bad shape, missing or extra keys), two entries with one
    name, or a file that already exists.
    """
    where = os.fspath(out_dir)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _M64:
        raise _refusal(where, f"the seed must be an int in [0, 2**64), got {_shown(seed)}")
    if isinstance(spec, (str, bytes, Mapping)) or not isinstance(spec, Sequence):
        raise _refusal(where, "the input spec must be a sequence of mappings")
    if not os.path.isdir(where):
        raise _refusal(where, "the output directory does not exist")
    plans: list[_InputPlan] = []
    for index, entry in enumerate(spec):
        plan = _plan_entry(where, index, entry)
        if any(other.name == plan.name for other in plans):
            raise _refusal(where, f"input spec entry {index}: the name {plan.name!r} is used twice")
        plans.append(plan)
    paths = {plan.name: Path(where) / f"{plan.name}{SUFFIX}" for plan in plans}
    existing = sorted(str(path) for path in paths.values() if os.path.lexists(path))
    if existing:
        raise _refusal(where, f"refusing to replace existing files: {', '.join(existing)}")
    blobs = {}
    for plan in plans:
        rng = Pcg32(seed, stream_of(plan.name))
        data = _uniform_data(plan, rng) if plan.dist == "uniform" else _integer_data(plan, rng)
        blobs[plan.name] = _encode(str(paths[plan.name]), plan.name, plan.dtype.name, plan.dims, data)
    for name, blob in blobs.items():
        paths[name].write_bytes(blob)
    return paths
