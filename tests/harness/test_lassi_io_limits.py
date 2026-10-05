"""Tests for lassi_io at its limits: huge shapes, huge numbers, and the generator's name length (task P4.3).

Bible: Harness Contract (lassi_io refusals; held-out input files).

The contract these tests fix, in lassi/harness/lassi_io.py (the format is
stated in test_lassi_io_format.py, the other refusals in
test_lassi_io_refusals.py and test_lassi_io_inputs.py):

- Every refusal of read_array and write_array is a LassiIOError whose
  message starts with str(path), however large the shape or its numbers: a
  rank of 224 or more with every dim 2**64 - 1 (a byte count with more
  digits than Python converts to text by default), and in write_array a dim
  far above 2**64 - 1 or far below 0.
- The byte count is checked by multiplying the dims only until the product
  passes the bytes available (read) or the data's length (write), and a zero
  dim gives 0 at once, so a rank of 60000 is refused, or read as an empty
  array, in well under a second.
- generate_inputs refuses a name over 247 bytes, since <name>.lassiio must
  fit a 255-byte file name, before any file is written, and accepts one of
  247 bytes. write_array keeps the 255-byte name rule, since its path is the
  caller's.
- A huge int in a spec (a bound, the seed, a dim) is refused with
  LassiIOError naming the output directory, before any file is written.

The timing bound is a generous guard against quadratic work, not a
measurement. No value here is a measurement.
"""

from __future__ import annotations

import importlib
import struct
import time
from pathlib import Path
from types import ModuleType

import pytest

MAGIC = b"LASSIIO\x00"
MAX_U64 = 2**64 - 1
SIZE_WORDS = ("size", "bytes", "length")
# A fixed rank in the tens of thousands must finish far inside this; quadratic big-number work took seconds.
FAST_S = 2.0
HUGE = 10**5000  # more digits than Python converts to text by default
VALID_ENTRY = {"name": "a", "dtype": "f32", "shape": [4], "dist": "uniform", "lo": -1.0, "hi": 1.0}


def lassi_io() -> ModuleType:
    """Import and return lassi.harness.lassi_io."""
    return importlib.import_module("lassi.harness.lassi_io")


def build(dims: tuple[int, ...], data: bytes, code: int = 1, name: bytes = b"x") -> bytes:
    """Build a file by hand: magic, version 1, dtype code, rank, dims, name, zero padding, data."""
    head = MAGIC + struct.pack("<III", 1, code, len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
    head += struct.pack("<I", len(name)) + name
    return head + bytes(-len(head) % 8) + data


def refusal(error: BaseException, path: Path, words: tuple[str, ...] | None = None) -> None:
    """Check a refusal: a LassiIOError whose message starts with the path and, when given, says one of `words`."""
    module = lassi_io()
    message = str(error)
    assert isinstance(error, module.LassiIOError), f"{type(error).__name__}, not LassiIOError: {message[:200]}"
    assert message.startswith(str(path)), f"the message does not start with {path}: {message[:200]}"
    if words is not None:
        assert any(word in message.lower() for word in words), f"the message says none of {words}: {message[:200]}"


# ---------------------------------------------------------------------------
# Reading


@pytest.mark.parametrize("rank", [224, 300, 1000])
def test_reading_a_huge_rank_of_max_dims_is_a_refusal_naming_the_file(rank: int, tmp_path: Path) -> None:
    path = tmp_path / f"rank-{rank}.lassiio"
    path.write_bytes(build((MAX_U64,) * rank, struct.pack("<2f", 1.0, -2.0)))
    with pytest.raises(Exception) as info:
        lassi_io().read_array(path)
    refusal(info.value, path, SIZE_WORDS)


def test_reading_a_rank_of_60000_max_dims_is_refused_fast(tmp_path: Path) -> None:
    path = tmp_path / "rank-60000.lassiio"
    path.write_bytes(build((MAX_U64,) * 60000, b"\x00" * 8))
    start = time.perf_counter()
    with pytest.raises(Exception) as info:
        lassi_io().read_array(path)
    elapsed = time.perf_counter() - start
    refusal(info.value, path, SIZE_WORDS)
    assert elapsed < FAST_S, f"refusing took {elapsed:.2f} s"


def test_a_zero_dim_among_max_dims_reads_as_an_empty_array_fast(tmp_path: Path) -> None:
    dims = (MAX_U64,) * 59999 + (0,)
    path = tmp_path / "empty.lassiio"
    path.write_bytes(build(dims, b"", code=7))
    start = time.perf_counter()
    array = lassi_io().read_array(path)
    elapsed = time.perf_counter() - start
    assert (array.dtype, array.shape, array.data) == ("u8", dims, b"")
    assert elapsed < FAST_S, f"reading took {elapsed:.2f} s"


# ---------------------------------------------------------------------------
# Writing


@pytest.mark.parametrize("rank", [224, 300, 1000])
def test_writing_a_huge_rank_of_max_dims_is_a_refusal_naming_the_file(rank: int, tmp_path: Path) -> None:
    path = tmp_path / "out.lassiio"
    with pytest.raises(Exception) as info:
        lassi_io().write_array(path, "x", "f32", (MAX_U64,) * rank, b"\x00" * 8)
    refusal(info.value, path, SIZE_WORDS)
    assert not path.exists()


def test_writing_a_rank_of_60000_max_dims_is_refused_fast(tmp_path: Path) -> None:
    path = tmp_path / "out.lassiio"
    start = time.perf_counter()
    with pytest.raises(Exception) as info:
        lassi_io().write_array(path, "x", "f64", [MAX_U64] * 60000, b"\x00" * 8)
    elapsed = time.perf_counter() - start
    refusal(info.value, path, SIZE_WORDS)
    assert elapsed < FAST_S, f"refusing took {elapsed:.2f} s"
    assert not path.exists()


def test_a_zero_dim_among_max_dims_round_trips(tmp_path: Path) -> None:
    module = lassi_io()
    shape = (MAX_U64,) * 223 + (0,)
    path = tmp_path / "empty.lassiio"
    module.write_array(path, "e", "i64", shape, b"")
    assert module.read_array(path) == module.LassiArray("e", "i64", shape, b"")


@pytest.mark.parametrize(
    "shape",
    [(HUGE,), (2, HUGE), (-HUGE,), (MAX_U64 + 1,), (HUGE,) * 9],
    ids=["huge dim", "huge second dim", "huge negative dim", "2**64", "nine huge dims"],
)
def test_writing_a_dim_outside_u64_is_a_refusal_naming_the_file(shape: tuple[int, ...], tmp_path: Path) -> None:
    path = tmp_path / "out.lassiio"
    with pytest.raises(Exception) as info:
        lassi_io().write_array(path, "x", "u8", shape, b"")
    refusal(info.value, path)
    assert not path.exists()


# ---------------------------------------------------------------------------
# The generator


def refused_spec(entry: dict, out: Path, seed: int = 1) -> BaseException:
    """Run generate_inputs on [VALID_ENTRY, entry] into the empty `out`; return the error; check nothing was written."""
    out.mkdir()
    with pytest.raises(Exception) as info:
        lassi_io().generate_inputs([VALID_ENTRY, entry], seed, out)
    assert list(out.iterdir()) == [], "a refused spec left files behind"
    return info.value


def test_the_generator_refuses_a_name_over_247_bytes_before_writing(tmp_path: Path) -> None:
    out = tmp_path / "out"
    error = refused_spec({**VALID_ENTRY, "name": "x" * 248}, out)
    refusal(error, out, ("name",))
    assert "247" in str(error)


def test_the_generator_accepts_a_name_of_247_bytes(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    name = "x" * 247
    paths = lassi_io().generate_inputs([{**VALID_ENTRY, "name": name}], 1, out)
    assert paths == {name: out / f"{name}.lassiio"}
    assert len(f"{name}.lassiio") == 255
    assert lassi_io().read_array(paths[name]).name == name


def test_write_array_keeps_the_255_byte_name_rule(tmp_path: Path) -> None:
    module = lassi_io()
    path = tmp_path / "long-name.lassiio"
    module.write_array(path, "x" * 255, "u8", (1,), b"\x07")
    assert module.read_array(path).name == "x" * 255


@pytest.mark.parametrize(
    "entry",
    [
        {**VALID_ENTRY, "name": "b", "dtype": "f64", "lo": 0.0, "hi": HUGE},
        {**VALID_ENTRY, "name": "b", "dtype": "f64", "lo": -HUGE, "hi": 0.0},
        {**VALID_ENTRY, "name": "b", "dtype": "i64", "dist": "integers", "lo": 0, "hi": HUGE},
        {**VALID_ENTRY, "name": "b", "dtype": "i64", "dist": "integers", "lo": HUGE, "hi": HUGE + 1},
        {**VALID_ENTRY, "name": "b", "shape": [HUGE]},
        {**VALID_ENTRY, "name": "b", "shape": [-HUGE]},
    ],
    ids=["float hi", "float lo", "int hi", "int lo above hi range", "huge dim", "huge negative dim"],
)
def test_a_huge_int_in_a_spec_is_a_refusal_naming_the_directory(entry: dict, tmp_path: Path) -> None:
    out = tmp_path / "out"
    refusal(refused_spec(entry, out), out)


def test_a_huge_seed_is_a_refusal_naming_the_directory(tmp_path: Path) -> None:
    out = tmp_path / "out"
    refusal(refused_spec({**VALID_ENTRY, "name": "b"}, out, seed=HUGE), out)
