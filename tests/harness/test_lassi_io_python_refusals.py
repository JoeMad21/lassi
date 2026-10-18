"""Tests for the refusals of the Python harness binding assets/harness/python/lassi_io.py (task P12.3).

Bible: Harness Contract (lassi_io refusals: what the Python reader and the
Python writer refuse, and how). Plan: plans/p12-frontends.md, P12.3 (the
binding refuses what lassi.harness.lassi_io refuses).

The contract these tests fix: the binding's LassiIOError is a ValueError
and the one error of a refusal; its message starts with the file's path and
says why. Each refusal of the Harness Contract's bullet has one test, which
also checks that lassi.harness.lassi_io refuses the same input, so every
case is a refusal of the reference too:

- read_array refuses a bad magic, a version other than 1, an unknown dtype
  code, a file that ends inside its header (every prefix of a valid file), a
  name length outside 1 to 255, a bad name, nonzero padding, data longer or
  shorter than the dims say, and a header field past the end of the file or
  dims whose product is far beyond it. It multiplies the dims only until the
  product passes the bytes that follow the header, so a huge rank costs
  linear time and a zero dim gives an empty array at once.
- write_array refuses an unknown dtype, a negative or non-int dim or one past
  2**64 - 1, a bad name, and data whose length is not product(shape) *
  itemsize, before it opens the file: no file is created and an existing one
  keeps its bytes. No refusal formats a huge int.
- A file that cannot be opened is an OSError, not a refusal.

Two differential tests compare the binding with the reference: over every
one-byte change, cut, and extension of valid files, and over a table of
write arguments, the two give the same outcome (both refuse, or both read or
write the same thing). The keyword sets are those of
test_lassi_io_refusals.py; the timing bound is a generous guard against
quadratic work, not a measurement. No value here is a measurement.
"""

from __future__ import annotations

import array
import functools
import importlib.util
import struct
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from lassi.harness import lassi_io as reference

REPO = Path(__file__).resolve().parents[2]
BINDING = REPO / "assets" / "harness" / "python" / "lassi_io.py"
MAGIC = b"LASSIIO\x00"
F32 = 1
MAX_U64 = 2**64 - 1
HUGE = 10**5000  # more digits than Python converts to text by default
# A fixed rank in the tens of thousands must finish far inside this; quadratic big-number work took seconds.
FAST_S = 2.0
SENTINEL = b"an existing file a refused write must not touch\n"

MAGIC_WORDS = ("magic",)
VERSION_WORDS = ("version",)
DTYPE_WORDS = ("dtype",)
SIZE_WORDS = ("size", "bytes", "length")
TRUNCATED_WORDS = ("truncated", "short", "end of file", "eof")
NAME_WORDS = ("name",)
PADDING_WORDS = ("padding",)
SHAPE_WORDS = ("shape", "dim")

# A name with one non-ASCII letter (e with an acute accent), built so this file stays ASCII.
NON_ASCII = "caf" + chr(0xE9)
BAD_NAMES = ["", "x" * 256, NON_ASCII, "a b", "a/b", "a\\b", "x:y", "tab\there", "nul\x00", "semi;colon"]
BAD_NAME_IDS = ["empty", "256 bytes", "non-ASCII", "space", "slash", "backslash", "colon", "tab", "NUL", "semicolon"]
# Names of a valid length (1 to 255 bytes) with a byte outside the allowed set.
BAD_FILE_NAMES = [NON_ASCII.encode("utf-8"), b"a b", b"a/b", b"\xff", b"nul\x00", b"x:y"]
BAD_FILE_NAME_IDS = ["non-ASCII", "space", "slash", "byte 0xff", "NUL", "colon"]
F32_DATA = struct.pack("<2f", 1.0, -2.0)


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


def build(
    name: bytes = b"x",
    code: int = F32,
    dims: tuple[int, ...] = (2,),
    data: bytes = F32_DATA,
    *,
    magic: bytes = MAGIC,
    version: int = 1,
    rank: int | None = None,
    name_length: int | None = None,
) -> bytes:
    """Build a file by hand; the keyword arguments override single fields to make it bad."""
    head = magic + struct.pack("<III", version, code, len(dims) if rank is None else rank)
    head += struct.pack(f"<{len(dims)}Q", *dims)
    head += struct.pack("<I", len(name) if name_length is None else name_length) + name
    return head + bytes(-len(head) % 8) + data


VALID = build()  # the name "x" ends at offset 33; padding fills 33 to 39; the data starts at 40
SAMPLES = {
    "f32-vector": VALID,
    "bf16-matrix": build(b"in_a", 3, (2, 3), struct.pack("<6H", 0x3F80, 0xBF80, 0x7FC0, 0x7F80, 0x8000, 0x0001)),
    "i64-scalar": build(b"s", 8, (), struct.pack("<q", -2)),
    "u8-empty": build(b"e", 7, (3, 0), b""),
}


def raised(call: Callable[[], object]) -> BaseException:
    """Call `call` and return the exception it raised; fail when it returned."""
    try:
        call()
    except Exception as error:
        return error
    pytest.fail("the call returned; a refusal was expected")


def check_refusal(module: ModuleType, error: BaseException, path: Path, words: tuple[str, ...] | None) -> None:
    """Check a refusal by `module`: its LassiIOError, a message that starts with the path and says one of `words`."""
    side = "the reference" if module is reference else "the binding"
    message = str(error)
    assert isinstance(error, module.LassiIOError), (
        f"{side} raised {type(error).__name__}, not its LassiIOError: {message[:200]}"
    )
    assert message.startswith(str(path)), f"{side}'s message does not start with {path}: {message[:200]}"
    if words is not None:
        assert any(word in message.lower() for word in words), f"{side}'s message says none of {words}: {message}"


def refused_read(path: Path, data: bytes, words: tuple[str, ...] | None = None) -> None:
    """Write `data` to `path`; check that the reference refuses it, then that the binding refuses it the same way."""
    module = binding()
    path.write_bytes(data)
    check_refusal(reference, raised(lambda: reference.read_array(path)), path, words)
    check_refusal(module, raised(lambda: module.read_array(path)), path, words)


def refused_write(
    tmp_path: Path, name: object, dtype: object, shape: object, data: object, words: tuple[str, ...] | None = None
) -> None:
    """Check that the reference, then the binding, refuses a write before opening the file.

    Each side writes once to a new path, which must not exist afterwards, and
    once over an existing file, which must keep its bytes.
    """
    module = binding()
    for side, label in ((reference, "reference"), (module, "binding")):
        fresh = tmp_path / f"{label}-new.lassiio"
        kept = tmp_path / f"{label}-kept.lassiio"
        kept.write_bytes(SENTINEL)
        for path in (fresh, kept):
            write = functools.partial(side.write_array, path, name, dtype, shape, data)
            check_refusal(side, raised(write), path, words)
        assert not fresh.exists(), f"a refused write by the {label} created the file"
        assert kept.read_bytes() == SENTINEL, f"a refused write by the {label} changed an existing file"


def read_outcome(module: ModuleType, path: Path) -> tuple[object, ...]:
    """Return what reading `path` with `module` gives: ("refused",), the array's contents, or another error."""
    try:
        array = module.read_array(path)
    except Exception as error:
        if isinstance(error, module.LassiIOError) and str(error).startswith(str(path)):
            return ("refused",)
        return ("error", type(error).__name__, str(error)[:120])
    return ("read", array.name, array.dtype, tuple(array.shape), bytes(array.data))


def write_outcome(module: ModuleType, path: Path, args: tuple[object, ...]) -> tuple[object, ...]:
    """Return what writing `args` to `path` with `module` gives: ("refused", file exists), the bytes, or an error."""
    try:
        module.write_array(path, *args)
    except Exception as error:
        if isinstance(error, module.LassiIOError) and str(error).startswith(str(path)):
            return ("refused", path.exists())
        return ("error", type(error).__name__, str(error)[:120])
    return ("written", path.read_bytes())


def test_lassi_io_error_is_a_value_error() -> None:
    assert issubclass(binding().LassiIOError, ValueError)


def test_the_valid_files_these_tests_break_are_read_alike(tmp_path: Path) -> None:
    module = binding()
    for sample, data in SAMPLES.items():
        path = tmp_path / f"{sample}.lassiio"
        path.write_bytes(data)
        assert read_outcome(module, path)[0] == "read", sample
        assert read_outcome(module, path) == read_outcome(reference, path), sample


# ---------------------------------------------------------------------------
# Reading


@pytest.mark.parametrize(
    "magic", [b"LASSIIO\x01", b"lassiio\x00", b"\x00" * 8, b"LASSIIO ", b"OISSAL\x00\x00"], ids=bytes.hex
)
def test_a_bad_magic_is_refused(magic: bytes, tmp_path: Path) -> None:
    refused_read(tmp_path / "bad-magic.lassiio", build(magic=magic), MAGIC_WORDS)


@pytest.mark.parametrize("version", [0, 2, 0xFFFFFFFF])
def test_a_version_other_than_1_is_refused(version: int, tmp_path: Path) -> None:
    refused_read(tmp_path / "bad-version.lassiio", build(version=version), VERSION_WORDS)


@pytest.mark.parametrize("code", [0, 10, 0x7FFFFFFF, 0xFFFFFFFF])
def test_an_unknown_dtype_code_is_refused(code: int, tmp_path: Path) -> None:
    assert code not in {dtype.code for dtype in reference.DTYPES.values()}
    refused_read(tmp_path / "bad-dtype.lassiio", build(code=code), DTYPE_WORDS)


@pytest.mark.parametrize(
    "cut",
    [
        pytest.param(0, id="empty file"),
        pytest.param(5, id="inside the magic"),
        pytest.param(8, id="after the magic"),
        pytest.param(12, id="after the version"),
        pytest.param(16, id="after the dtype code"),
        pytest.param(24, id="inside the dims"),
        pytest.param(28, id="after the dims"),
        pytest.param(30, id="inside the name length"),
        pytest.param(32, id="after the name length"),
        pytest.param(36, id="inside the padding"),
    ],
)
def test_a_file_that_ends_inside_its_header_is_refused(cut: int, tmp_path: Path) -> None:
    refused_read(tmp_path / "truncated.lassiio", VALID[:cut], TRUNCATED_WORDS)


def test_every_prefix_of_a_valid_file_is_refused(tmp_path: Path) -> None:
    module = binding()
    read = []
    for sample, data in SAMPLES.items():
        for cut in range(len(data)):
            path = tmp_path / f"{sample}-{cut:02d}.lassiio"
            path.write_bytes(data[:cut])
            check_refusal(reference, raised(functools.partial(reference.read_array, path)), path, None)
            try:
                module.read_array(path)
            except module.LassiIOError as error:
                assert str(error).startswith(str(path)), f"the message does not start with {path}: {error}"
            else:
                read.append((sample, cut))
    assert not read, f"prefixes the binding read as valid files, as (sample, length): {read}"


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(build(name=b""), id="0"),
        pytest.param(build(name=b"x" * 256), id="256"),
        pytest.param(build(name_length=1000), id="1000"),
        pytest.param(build(name_length=0xFFFFFFFF), id="2**32 - 1"),
    ],
)
def test_a_name_length_outside_1_to_255_is_refused(data: bytes, tmp_path: Path) -> None:
    refused_read(tmp_path / "bad-name-length.lassiio", data, NAME_WORDS)


@pytest.mark.parametrize("name", BAD_FILE_NAMES, ids=BAD_FILE_NAME_IDS)
def test_a_bad_name_in_a_file_is_refused(name: bytes, tmp_path: Path) -> None:
    refused_read(tmp_path / "bad-name.lassiio", build(name=name), NAME_WORDS)


@pytest.mark.parametrize("offset", range(33, 40))
def test_nonzero_padding_is_refused(offset: int, tmp_path: Path) -> None:
    assert VALID[offset] == 0
    data = VALID[:offset] + b"\x01" + VALID[offset + 1 :]
    refused_read(tmp_path / "bad-padding.lassiio", data, PADDING_WORDS)


@pytest.mark.parametrize(
    ("dims", "data"),
    [
        pytest.param((2,), F32_DATA + b"\x00", id="one byte long"),
        pytest.param((2,), struct.pack("<3f", 1.0, -2.0, 3.0), id="one element long"),
        pytest.param((), F32_DATA, id="scalar of two elements"),
        pytest.param((0, 3), b"\x00", id="empty array with data"),
    ],
)
def test_data_longer_than_the_dims_say_is_refused(dims: tuple[int, ...], data: bytes, tmp_path: Path) -> None:
    refused_read(tmp_path / "long.lassiio", build(dims=dims, data=data), SIZE_WORDS)


@pytest.mark.parametrize(
    ("dims", "data"),
    [
        pytest.param((2,), F32_DATA[:-1], id="one byte short"),
        pytest.param((2,), struct.pack("<f", 1.0), id="one element short"),
        pytest.param((2,), b"", id="no data"),
        pytest.param((), b"", id="scalar without data"),
    ],
)
def test_data_shorter_than_the_dims_say_is_refused(dims: tuple[int, ...], data: bytes, tmp_path: Path) -> None:
    refused_read(tmp_path / "short.lassiio", build(dims=dims, data=data), SIZE_WORDS)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(build(rank=0xFFFFFFFF), id="rank 2**32 - 1"),
        pytest.param(build(rank=100), id="rank 100"),
        pytest.param(build(name_length=200), id="name length 200"),
    ],
)
def test_a_header_field_past_the_end_of_the_file_is_refused(data: bytes, tmp_path: Path) -> None:
    refused_read(tmp_path / "huge-field.lassiio", data)


@pytest.mark.parametrize("dims", [(2**40, 2**40), (MAX_U64,), (2**32, 2**32, 2**32), (MAX_U64,) * 300])
def test_dims_whose_product_is_far_beyond_the_file_are_refused(dims: tuple[int, ...], tmp_path: Path) -> None:
    refused_read(tmp_path / "huge-dims.lassiio", build(dims=dims), SIZE_WORDS)


def test_a_rank_of_60000_max_dims_is_refused_in_linear_time(tmp_path: Path) -> None:
    module = binding()
    path = tmp_path / "rank-60000.lassiio"
    path.write_bytes(build(dims=(MAX_U64,) * 60000, data=b"\x00" * 8))
    start = time.perf_counter()
    error = raised(lambda: module.read_array(path))
    elapsed = time.perf_counter() - start
    check_refusal(module, error, path, SIZE_WORDS)
    assert elapsed < FAST_S, f"refusing took {elapsed:.2f} s"


def test_a_zero_dim_among_max_dims_reads_as_an_empty_array_at_once(tmp_path: Path) -> None:
    module = binding()
    dims = (MAX_U64,) * 59999 + (0,)
    path = tmp_path / "empty.lassiio"
    path.write_bytes(build(code=7, dims=dims, data=b""))
    start = time.perf_counter()
    array = module.read_array(path)
    elapsed = time.perf_counter() - start
    assert (array.dtype, tuple(array.shape), bytes(array.data)) == ("u8", dims, b"")
    assert elapsed < FAST_S, f"reading took {elapsed:.2f} s"


def test_a_file_that_cannot_be_opened_is_an_os_error_not_a_refusal(tmp_path: Path) -> None:
    module = binding()
    missing = tmp_path / "missing.lassiio"
    for call in (
        lambda: module.read_array(missing),
        lambda: module.write_array(tmp_path / "no-such-directory" / "out.lassiio", "x", "f32", (2,), F32_DATA),
    ):
        error = raised(call)
        assert isinstance(error, OSError), f"{type(error).__name__}, not an OSError: {error}"
        assert not isinstance(error, module.LassiIOError)


# ---------------------------------------------------------------------------
# Writing


@pytest.mark.parametrize("dtype", ["f128", "float32", "bfloat16", "F32", "", "c64"], ids=lambda dtype: dtype or "empty")
def test_writing_an_unknown_dtype_is_refused(dtype: str, tmp_path: Path) -> None:
    refused_write(tmp_path, "x", dtype, (2,), b"\x00" * 8, DTYPE_WORDS)


@pytest.mark.parametrize("shape", [(-1,), (2, -3), (0, -1)])
def test_writing_a_negative_dim_is_refused(shape: tuple[int, ...], tmp_path: Path) -> None:
    refused_write(tmp_path, "x", "u8", shape, b"", SHAPE_WORDS)


@pytest.mark.parametrize(
    "shape", [(2.0,), ("2",), (True, True), (None,), (2, 1.0)], ids=["float", "str", "bool", "None", "second float"]
)
def test_writing_a_non_int_dim_is_refused(shape: tuple[object, ...], tmp_path: Path) -> None:
    refused_write(tmp_path, "x", "u8", shape, b"\x00\x00", SHAPE_WORDS)


@pytest.mark.parametrize(
    "shape",
    [(MAX_U64 + 1,), (HUGE,), (2, HUGE), (-HUGE,), (HUGE,) * 9],
    ids=["2**64", "huge dim", "huge second dim", "huge negative dim", "nine huge dims"],
)
def test_writing_a_dim_past_2_64_minus_1_is_refused(shape: tuple[int, ...], tmp_path: Path) -> None:
    refused_write(tmp_path, "x", "u8", shape, b"", SHAPE_WORDS)


@pytest.mark.parametrize("name", BAD_NAMES, ids=BAD_NAME_IDS)
def test_writing_a_bad_name_is_refused(name: str, tmp_path: Path) -> None:
    refused_write(tmp_path, name, "u8", (1,), b"\x00", NAME_WORDS)


@pytest.mark.parametrize(
    ("dtype", "shape", "data"),
    [
        pytest.param("f32", (2, 3), b"\x00" * 23, id="one byte short"),
        pytest.param("f32", (2, 3), b"\x00" * 25, id="one byte long"),
        pytest.param("f32", (), b"", id="scalar without data"),
        pytest.param("f64", (), b"\x00" * 16, id="scalar of two elements"),
        pytest.param("f32", (0, 3), b"\x00" * 4, id="empty array with data"),
        pytest.param("f32", (MAX_U64,) * 224, b"\x00" * 8, id="rank 224 of max dims"),
    ],
)
def test_writing_data_of_the_wrong_length_is_refused(
    dtype: str, shape: tuple[int, ...], data: bytes, tmp_path: Path
) -> None:
    refused_write(tmp_path, "x", dtype, shape, data, SIZE_WORDS)


def test_writing_a_rank_of_60000_max_dims_is_refused_in_linear_time(tmp_path: Path) -> None:
    module = binding()
    path = tmp_path / "out.lassiio"
    start = time.perf_counter()
    error = raised(lambda: module.write_array(path, "x", "f64", [MAX_U64] * 60000, b"\x00" * 8))
    elapsed = time.perf_counter() - start
    check_refusal(module, error, path, SIZE_WORDS)
    assert elapsed < FAST_S, f"refusing took {elapsed:.2f} s"
    assert not path.exists()


# ---------------------------------------------------------------------------
# The binding against the reference


@pytest.mark.parametrize("sample", sorted(SAMPLES))
def test_the_binding_refuses_exactly_the_one_byte_changes_the_reference_refuses(sample: str, tmp_path: Path) -> None:
    module = binding()
    valid = SAMPLES[sample]
    path = tmp_path / "changed.lassiio"
    differ = []
    for offset in range(len(valid)):
        for value in sorted({0x00, 0x01, 0x7F, 0x80, 0xFF, valid[offset] ^ 0x01} - {valid[offset]}):
            path.write_bytes(valid[:offset] + bytes([value]) + valid[offset + 1 :])
            ours, theirs = read_outcome(module, path), read_outcome(reference, path)
            if ours != theirs:
                differ.append((offset, value, ours[:2], theirs[:2]))
    assert not differ, f"(offset, byte, binding, reference) where the outcomes differ: {differ[:20]}"


@pytest.mark.parametrize("sample", sorted(SAMPLES))
def test_the_binding_refuses_exactly_the_cuts_and_extensions_the_reference_refuses(sample: str, tmp_path: Path) -> None:
    module = binding()
    valid = SAMPLES[sample]
    variants = [valid[:cut] for cut in range(len(valid) + 1)]
    variants += [valid + extra for extra in (b"\x00", b"\x00" * 8, b"\xff" * 16)]
    path = tmp_path / "variant.lassiio"
    differ = []
    for data in variants:
        path.write_bytes(data)
        ours, theirs = read_outcome(module, path), read_outcome(reference, path)
        if ours != theirs:
            differ.append((len(data), ours[:2], theirs[:2]))
    assert not differ, f"(file length, binding, reference) where the outcomes differ: {differ}"


WRITE_CASES = [
    pytest.param(("x", "f32", (2,), F32_DATA), id="valid"),
    pytest.param(("x", "f32", [2], F32_DATA), id="list shape"),
    pytest.param(("x", "f32", (2,), bytearray(F32_DATA)), id="bytearray"),
    pytest.param(("x", "f32", (2,), memoryview(F32_DATA).cast("f")), id="memoryview of f32"),
    pytest.param(("x", "f32", (1, 2, 1), F32_DATA), id="rank 3"),
    pytest.param(("x", "f32", (), F32_DATA[:4]), id="scalar"),
    pytest.param(("x", "bf16", (2,), F32_DATA), id="bf16 pair"),
    pytest.param(("x", "f32", (0, MAX_U64), b""), id="zero dim before a max dim"),
    pytest.param(("x" * 255, "u8", (1,), b"\x07"), id="255-byte name"),
    pytest.param(("A.b-c_9", "i8", (2,), b"\x01\xff"), id="every name character class"),
    pytest.param(("x", "f32", (4,), F32_DATA), id="data short"),
    pytest.param(("x", "f32", (1,), F32_DATA), id="data long"),
    pytest.param(("x", "f64", (2,), F32_DATA), id="data short for f64"),
    pytest.param(("x", "f32", (-1,), b""), id="negative dim"),
    pytest.param(("x", "f32", (2**64,), b""), id="dim 2**64"),
    pytest.param(("x", "f32", (2.0,), F32_DATA), id="float dim"),
    pytest.param(("x", "f32", (True, True), F32_DATA[:4]), id="bool dims"),
    pytest.param(("x", "f32", "ab", F32_DATA), id="str shape"),
    pytest.param(("x", "f32", None, F32_DATA), id="no shape"),
    pytest.param(("x", "F32", (2,), F32_DATA), id="upper-case dtype"),
    pytest.param(("x", 1, (2,), F32_DATA), id="dtype code"),
    pytest.param(("x", None, (2,), F32_DATA), id="no dtype"),
    pytest.param(("", "f32", (2,), F32_DATA), id="empty name"),
    pytest.param(("x" * 256, "f32", (2,), F32_DATA), id="256-byte name"),
    pytest.param((b"x", "f32", (2,), F32_DATA), id="bytes name"),
    pytest.param(("a b", "f32", (2,), F32_DATA), id="name with a space"),
    pytest.param(("x", "u8", (3,), 3), id="int data"),
    pytest.param(("x", "u8", (2,), [1, 2]), id="list data"),
    pytest.param(("x", "u8", (2,), "ab"), id="str data"),
    pytest.param(("x", "u8", (2,), None), id="no data"),
    pytest.param(("x", "f32", (2,), array.array("f", [1.0, -2.0])), id="array.array data"),
]


@pytest.mark.parametrize("args", WRITE_CASES)
def test_the_binding_refuses_exactly_the_writes_the_reference_refuses(args: tuple[object, ...], tmp_path: Path) -> None:
    module = binding()
    ours = write_outcome(module, tmp_path / "binding.lassiio", args)
    theirs = write_outcome(reference, tmp_path / "reference.lassiio", args)
    assert ours == theirs
