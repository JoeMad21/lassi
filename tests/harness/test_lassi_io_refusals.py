"""Tests for the refusals of the lassi_io reader and writer (task P4.3).

Bible: Harness Contract (the harness owns inputs and comparison, so a file
it cannot read exactly is refused, never guessed at).

The contract these tests fix, in lassi/harness/lassi_io.py (the format is
stated in test_lassi_io_format.py):

- LassiIOError, a subclass of ValueError, is the one error of a refusal.
  Its message holds str(path) of the file, so a caller can tell which file
  was refused, and says why:
  - read_array refuses a bad magic ("magic" in the message), a version other
    than 1 ("version"), a dtype code DTYPES does not hold ("dtype"), data
    longer or shorter than the dims say ("size", "bytes", or "length"), a
    file that ends inside its header ("truncated", "short", "end of file",
    or "eof"), and a name that is empty, longer than 255 bytes, or holds a
    byte other than an ASCII letter, digit, '_', '.', or '-' ("name"). Every
    prefix of a valid file shorter than the file is refused. A rank or a
    name length far beyond the file's size, and dims whose product is far
    beyond it, are refused the same way, without reading or allocating that
    much.
  - write_array refuses a dtype DTYPES does not hold, data whose length is
    not product(shape) * itemsize, a negative dim, and a bad name, before it
    writes anything: the file does not exist afterwards.

The keyword sets are lenient on purpose; the file's path is required. No
value here is a measurement.
"""

from __future__ import annotations

import importlib
import struct
from pathlib import Path
from types import ModuleType

import pytest

MAGIC = b"LASSIIO\x00"
F32 = 1
MAGIC_WORDS = ("magic",)
VERSION_WORDS = ("version",)
DTYPE_WORDS = ("dtype",)
SIZE_WORDS = ("size", "bytes", "length")
TRUNCATED_WORDS = ("truncated", "short", "end of file", "eof")
NAME_WORDS = ("name",)

# A name with one non-ASCII letter (e with an acute accent), built so this file stays ASCII.
NON_ASCII = "caf" + chr(0xE9)
BAD_NAMES = ["", "x" * 256, NON_ASCII, "a b", "a/b", "a\\b", "x:y", "tab\there", "nul\x00", "semi;colon"]
BAD_NAME_IDS = ["empty", "256 bytes", "non-ASCII", "space", "slash", "backslash", "colon", "tab", "NUL", "semicolon"]
BAD_FILE_NAMES = [b"", b"x" * 256, NON_ASCII.encode("utf-8"), b"a b", b"a/b", b"\xff", b"nul\x00"]
BAD_FILE_NAME_IDS = ["empty", "256 bytes", "non-ASCII", "space", "slash", "byte 0xff", "NUL"]


def lassi_io() -> ModuleType:
    """Import and return lassi.harness.lassi_io."""
    return importlib.import_module("lassi.harness.lassi_io")


def build(
    name: bytes = b"x",
    code: int = F32,
    dims: tuple[int, ...] = (2,),
    data: bytes = struct.pack("<2f", 1.0, -2.0),
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


VALID = build()


def refused(path: Path, words: tuple[str, ...] | None = None) -> str:
    """Read `path`, expecting LassiIOError naming the file and, when given, one of `words`; return the message."""
    module = lassi_io()
    with pytest.raises(module.LassiIOError) as info:
        module.read_array(path)
    message = str(info.value)
    assert str(path) in message, f"the error does not name the file {path}: {message}"
    if words is not None:
        assert any(word in message.lower() for word in words), f"the error says none of {words}: {message}"
    return message


def refused_write(
    path: Path, name: str, dtype: str, shape: tuple[int, ...], data: bytes, words: tuple[str, ...]
) -> None:
    """Write, expecting LassiIOError naming the file and one of `words`, and no file afterwards."""
    module = lassi_io()
    with pytest.raises(module.LassiIOError) as info:
        module.write_array(path, name, dtype, shape, data)
    message = str(info.value)
    assert str(path) in message, f"the error does not name the file {path}: {message}"
    assert any(word in message.lower() for word in words), f"the error says none of {words}: {message}"
    assert not path.exists(), "a refused write left a file behind"


def test_lassi_io_error_is_a_value_error() -> None:
    assert issubclass(lassi_io().LassiIOError, ValueError)


def test_the_valid_file_these_tests_break_is_read(tmp_path: Path) -> None:
    path = tmp_path / "valid.lassiio"
    path.write_bytes(VALID)
    module = lassi_io()
    assert module.read_array(path) == module.LassiArray("x", "f32", (2,), struct.pack("<2f", 1.0, -2.0))


# ---------------------------------------------------------------------------
# Reading


@pytest.mark.parametrize(
    "magic", [b"LASSIIO\x01", b"lassiio\x00", b"\x00" * 8, b"LASSIIO ", b"OISSAL\x00\x00"], ids=bytes.hex
)
def test_a_bad_magic_is_refused(magic: bytes, tmp_path: Path) -> None:
    path = tmp_path / "bad-magic.lassiio"
    path.write_bytes(build(magic=magic))
    refused(path, MAGIC_WORDS)


@pytest.mark.parametrize("version", [0, 2, 0xFFFFFFFF])
def test_an_unknown_version_is_refused(version: int, tmp_path: Path) -> None:
    path = tmp_path / "bad-version.lassiio"
    path.write_bytes(build(version=version))
    refused(path, VERSION_WORDS)


@pytest.mark.parametrize("code", [0, 0x7FFFFFFF, 0xFFFFFFFF])
def test_an_unknown_dtype_code_is_refused(code: int, tmp_path: Path) -> None:
    assert code not in {dtype.code for dtype in lassi_io().DTYPES.values()}
    path = tmp_path / "bad-dtype.lassiio"
    path.write_bytes(build(code=code))
    refused(path, DTYPE_WORDS)


@pytest.mark.parametrize(
    ("dims", "data"),
    [
        pytest.param((2,), struct.pack("<2f", 1.0, -2.0) + b"\x00", id="one byte long"),
        pytest.param((2,), struct.pack("<3f", 1.0, -2.0, 3.0), id="one element long"),
        pytest.param((2,), struct.pack("<2f", 1.0, -2.0)[:-1], id="one byte short"),
        pytest.param((2,), struct.pack("<f", 1.0), id="one element short"),
        pytest.param((2,), b"", id="no data"),
        pytest.param((), b"", id="scalar without data"),
        pytest.param((), struct.pack("<2f", 1.0, -2.0), id="scalar of two elements"),
        pytest.param((0, 3), b"\x00", id="empty array with data"),
    ],
)
def test_data_of_the_wrong_size_is_refused(dims: tuple[int, ...], data: bytes, tmp_path: Path) -> None:
    path = tmp_path / "bad-size.lassiio"
    path.write_bytes(build(dims=dims, data=data))
    refused(path, SIZE_WORDS)


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
def test_a_truncated_header_is_refused(cut: int, tmp_path: Path) -> None:
    path = tmp_path / "truncated.lassiio"
    path.write_bytes(VALID[:cut])
    refused(path, TRUNCATED_WORDS)


def test_every_prefix_of_a_valid_file_is_refused(tmp_path: Path) -> None:
    module = lassi_io()
    read = []
    for cut in range(len(VALID)):
        path = tmp_path / f"prefix-{cut:02d}.lassiio"
        path.write_bytes(VALID[:cut])
        try:
            module.read_array(path)
        except module.LassiIOError as error:
            assert str(path) in str(error), f"the error does not name the file {path}: {error}"
        else:
            read.append(cut)
    assert not read, f"prefixes read as valid files, by length: {read}"


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"rank": 0xFFFFFFFF}, id="rank 2**32 - 1"),
        pytest.param({"name_length": 0xFFFFFFFF}, id="name length 2**32 - 1"),
        pytest.param({"name_length": 1000}, id="name length past the end"),
    ],
)
def test_a_header_field_past_the_end_of_the_file_is_refused(overrides: dict, tmp_path: Path) -> None:
    path = tmp_path / "huge-field.lassiio"
    path.write_bytes(build(**overrides))
    refused(path)


@pytest.mark.parametrize("dims", [(2**40, 2**40), (2**64 - 1,), (2**32, 2**32, 2**32)])
def test_dims_whose_product_is_far_beyond_the_file_are_refused(dims: tuple[int, ...], tmp_path: Path) -> None:
    path = tmp_path / "huge-dims.lassiio"
    path.write_bytes(build(dims=dims))
    refused(path, SIZE_WORDS)


@pytest.mark.parametrize("name", BAD_FILE_NAMES, ids=BAD_FILE_NAME_IDS)
def test_a_bad_name_in_a_file_is_refused(name: bytes, tmp_path: Path) -> None:
    path = tmp_path / "bad-name.lassiio"
    path.write_bytes(build(name=name))
    refused(path, NAME_WORDS)


# ---------------------------------------------------------------------------
# Writing


@pytest.mark.parametrize("dtype", ["f128", "float32", "bfloat16", "", "c64"], ids=lambda dtype: dtype or "empty")
def test_writing_an_unknown_dtype_is_refused(dtype: str, tmp_path: Path) -> None:
    refused_write(tmp_path / "out.lassiio", "x", dtype, (2,), b"\x00" * 8, DTYPE_WORDS)


@pytest.mark.parametrize(
    ("shape", "data"),
    [
        pytest.param((2, 3), b"\x00" * 23, id="one byte short"),
        pytest.param((2, 3), b"\x00" * 25, id="one byte long"),
        pytest.param((), b"", id="scalar without data"),
        pytest.param((0, 3), b"\x00" * 4, id="empty array with data"),
    ],
)
def test_writing_data_of_the_wrong_size_is_refused(shape: tuple[int, ...], data: bytes, tmp_path: Path) -> None:
    refused_write(tmp_path / "out.lassiio", "x", "f32", shape, data, SIZE_WORDS)


@pytest.mark.parametrize("shape", [(-1,), (2, -3), (0, -1)])
def test_writing_a_negative_dim_is_refused(shape: tuple[int, ...], tmp_path: Path) -> None:
    module = lassi_io()
    path = tmp_path / "out.lassiio"
    with pytest.raises(module.LassiIOError) as info:
        module.write_array(path, "x", "u8", shape, b"")
    assert str(path) in str(info.value)
    assert not path.exists()


@pytest.mark.parametrize("name", BAD_NAMES, ids=BAD_NAME_IDS)
def test_writing_a_bad_name_is_refused(name: str, tmp_path: Path) -> None:
    refused_write(tmp_path / "out.lassiio", name, "u8", (1,), b"\x00", NAME_WORDS)
