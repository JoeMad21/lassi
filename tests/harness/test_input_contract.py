"""Tests for the input-contract check (task P16.1).

Bible: Adversarial Setups (the adversary rule: inputs that violate the input
contract, by type, range, or size, are rejected before execution), Harness
Contract (lassi_io format and refusals; held-out input files), Benchmark
Suites (held-out inputs; the element cap MAX_INPUT_ELEMENTS), Risks And
Questions (out-of-contract inputs from an adversary, answered by an
input-contract check before execution), Agent Rule 5. Plan:
plans/p16-adversarial.md, P16.1 and the planning decision "Input contract
(P16.1)".

The contract these tests fix, in lassi.harness (which module holds it is
the implementation's choice):

- input_contract(spec, declared=None) returns an InputContract. `spec` is
  an item's held-out input spec, generate_inputs entries (name, dtype,
  shape, dist, lo, hi). With no `declared`, the contract is the spec read
  as one: each input's name, dtype, and shape, and every element in
  [lo, hi), with no NaN and no infinity. `declared` is the value of the
  optional manifest key `contract` (tests/bench/test_contract_key.py): one
  entry per held-out input, the keys name, dtype, lo, and hi, and the
  optional allow_nan and allow_inf (true or false, false when left out).
  An entry's range must hold every dtype value of its input's spec range
  (it may widen the range, never narrow it), and its bounds follow
  generate_inputs' bound rules (hi - lo finite in double among them). Its
  lo and hi replace its input's range, and its flags (a float dtype only)
  let a NaN, or an infinity of either sign, through whatever the range;
  the name, dtype, and shape stay the spec's. A shape with a zero dim holds
  no element, whatever its other dims.
- check_inputs(contract, files) reads one proposed input set, a sequence of
  lassi_io file paths in any order, and matches arrays by the name in each
  file's header. It returns when the set is inside the contract and
  otherwise raises InputContractError. The error's `reason` is one of
  CONTRACT_REASONS, its `name` the input (None for an unreadable file), its
  `path` the file (None for a missing input), and its message holds the
  reason and each of the two that is present:
  - unreadable: a file that is not a readable lassi_io file (read_array
    refuses it, with the same message, or the path cannot be read);
  - too-many-elements: a file whose header's dims hold more than
    MAX_INPUT_ELEMENTS elements (lassi.bench.registry), whatever the
    contract's shape, refused before its data is read;
  - missing-input, extra-input, repeated-input: a contract input that no
    file holds, an array name the contract lacks, an array name two files
    hold;
  - dtype, shape: a dtype or a shape other than the contract's;
  - nan, infinity: a NaN, or an infinity of either sign, that the contract
    does not allow; neither reads as out-of-range;
  - out-of-range: a finite element outside [lo, hi), compared by value:
    lo is inside and hi is not; -0 equals +0, so it is inside [0, 1) and
    outside [-1, 0); each smallest subnormal lies on its sign's side of 0;
    a bound between two dtype values admits exactly the values on its side.
  The order, which CONTRACT_REASONS lists: each file in full, in the order
  given (unreadable, too-many-elements); the names in file order
  (repeated-input, extra-input), then missing-input; then input by input in
  the contract's order: dtype, shape, then over the whole array nan, then
  infinity, then out-of-range. It only reads: no file is written, changed,
  or removed. A refusal pickles with its attributes. check_inputs raises
  TypeError for a contract that is not an InputContract and for files that
  are not a sequence of paths, naming the type it got.
- Round trip: the files generate_inputs(spec, seed, dir) writes pass
  check_inputs(input_contract(spec), ...) for every dtype, dist, and seed.

Each test imports lassi.harness when it runs, so a missing name fails that
test alone. Every array here is SYNTHETIC; no value is a measurement. The
boundary bit patterns are written out by hand from each format's layout,
and one test checks that they decode to the values they are named for.
"""

from __future__ import annotations

import importlib
import math
import pickle
import struct
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.bench import registry
from lassi.harness.lassi_io import MAGIC, LassiIOError, check_input_spec, generate_inputs, read_array, write_array

# The refusal reasons in the order the check meets them.
REASONS = (
    "unreadable",
    "too-many-elements",
    "repeated-input",
    "extra-input",
    "missing-input",
    "dtype",
    "shape",
    "nan",
    "infinity",
    "out-of-range",
)
# struct codes: a float dtype's elements are given as bit patterns, an integer dtype's as values.
BITS = {"f16": "H", "bf16": "H", "f32": "I", "f64": "Q"}
INTS = {"i8": "b", "u8": "B", "i32": "i", "u32": "I", "i64": "q"}
SIGN = {"f16": 1 << 15, "bf16": 1 << 15, "f32": 1 << 31, "f64": 1 << 63}
# The bit pattern of 0.5 in each float dtype, a filler inside [-1, 1) and [0, 1); with SIGN, -0.5 for [-1, 0).
HALF = {"f16": 0x3800, "bf16": 0x3F00, "f32": 0x3F000000, "f64": 0x3FE0000000000000}
# Hand-written boundary patterns of each float dtype (checked by test_the_hand_written_patterns_...).
PATTERNS: dict[str, dict[str, int]] = {
    "f16": {
        "zero": 0x0000, "neg_zero": 0x8000, "sub": 0x0001, "neg_sub": 0x8001,
        "one": 0x3C00, "below_one": 0x3BFF, "neg_one": 0xBC00, "below_neg_one": 0xBC01,
    },
    "bf16": {
        "zero": 0x0000, "neg_zero": 0x8000, "sub": 0x0001, "neg_sub": 0x8001,
        "one": 0x3F80, "below_one": 0x3F7F, "neg_one": 0xBF80, "below_neg_one": 0xBF81,
    },
    "f32": {
        "zero": 0x00000000, "neg_zero": 0x80000000, "sub": 0x00000001, "neg_sub": 0x80000001,
        "one": 0x3F800000, "below_one": 0x3F7FFFFF, "neg_one": 0xBF800000, "below_neg_one": 0xBF800001,
    },
    "f64": {
        "zero": 0, "neg_zero": 1 << 63, "sub": 1, "neg_sub": (1 << 63) | 1,
        "one": 0x3FF0000000000000, "below_one": 0x3FEFFFFFFFFFFFFF,
        "neg_one": 0xBFF0000000000000, "below_neg_one": 0xBFF0000000000001,
    },
}
# Quiet, signaling, and negative NaNs, and both infinities, of each float dtype.
NANS = {
    "f16": (0x7E00, 0x7C01, 0xFE00),
    "bf16": (0x7FC0, 0x7F81, 0xFFC0),
    "f32": (0x7FC00000, 0x7F800001, 0xFFC00000),
    "f64": (0x7FF8000000000000, 0x7FF0000000000001, 0xFFF8000000000000),
}
INFS = {
    "f16": (0x7C00, 0xFC00),
    "bf16": (0x7F80, 0xFF80),
    "f32": (0x7F800000, 0xFF800000),
    "f64": (0x7FF0000000000000, 0xFFF0000000000000),
}
FLOATS = tuple(BITS)


def harness() -> ModuleType:
    """Import and return lassi.harness, where check_inputs and input_contract live."""
    return importlib.import_module("lassi.harness")


def entry(name: str, dtype: str, shape: Sequence[int], lo: float | int, hi: float | int) -> dict[str, Any]:
    """Return one generate_inputs entry: uniform for a float dtype, integers for an integer one."""
    dist = "uniform" if dtype in BITS else "integers"
    return {"name": name, "dtype": dtype, "shape": list(shape), "dist": dist, "lo": lo, "hi": hi}


def pack(dtype: str, elements: Sequence[int]) -> bytes:
    """Return little-endian element bytes: bit patterns for a float dtype, values for an integer dtype."""
    code = BITS.get(dtype) or INTS[dtype]
    return struct.pack(f"<{len(elements)}{code}", *elements)


def put(directory: Path, file: str, name: str, dtype: str, shape: Sequence[int], elements: Sequence[int]) -> Path:
    """Write one SYNTHETIC lassi_io file holding array `name` and return its path."""
    path = directory / file
    write_array(path, name, dtype, list(shape), pack(dtype, elements))
    return path


def accepts(contract: Any, files: Sequence[Path]) -> None:
    """Run check_inputs on `files`; any refusal fails the test with its message."""
    h = harness()
    try:
        h.check_inputs(contract, list(files))
    except h.InputContractError as error:
        pytest.fail(f"refused an in-contract set: {error.reason}: {error}")


def refused(contract: Any, files: Sequence[Path], reason: str, name: str | None, path: Path | None = None) -> Any:
    """Require check_inputs to refuse `files` for `reason`, naming input `name` and file `path`; return the error.

    `name` is None for an unreadable file and `path` None for a missing
    input; the error's attributes must be exactly these, and its message
    must start with the reason and hold each one given.
    """
    h = harness()
    with pytest.raises(h.InputContractError) as raised:
        h.check_inputs(contract, list(files))
    error = raised.value
    assert error.reason == reason, f"reason {error.reason!r}, expected {reason!r}: {error}"
    assert str(error).startswith(f"{reason}: "), f"the message starts with its reason {reason!r}: {error}"
    assert error.name == name, f"name {error.name!r}, expected {name!r}: {error}"
    assert error.path == (None if path is None else str(path)), f"path {error.path!r}, expected {path}: {error}"
    for shown in (name, path):
        if shown is not None:
            assert str(shown) in str(error), f"the message names {shown}: {error}"
    return error


def raw_file(path: Path, name: str, code: int, dims: Sequence[int], data: bytes) -> Path:
    """Write a SYNTHETIC version 1 file byte by byte, its header as given whatever its data's size; return its path."""
    head = MAGIC + struct.pack("<III", 1, code, len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
    head += struct.pack("<I", len(name)) + name.encode("ascii")
    path.write_bytes(head + bytes(-len(head) % 8) + data)
    return path


def single(dtype: str, lo: float | int, hi: float | int) -> Any:
    """Return the contract of one input `x` of `dtype`, shape [4], range [lo, hi)."""
    return harness().input_contract([entry("x", dtype, [4], lo, hi)])


def one_element(tmp_path: Path, dtype: str, filler: int, element: int) -> Path:
    """Write x of shape [4], three fillers and `element` at index 2, and return its path."""
    return put(tmp_path, "x.lassiio", "x", dtype, [4], [filler, filler, element, filler])


# ---------------------------------------------------------------------------
# The reasons and the test patterns themselves


def test_the_reasons_are_the_ten_refusal_classes_in_the_order_of_the_check() -> None:
    assert harness().CONTRACT_REASONS == REASONS


def decoded(dtype: str, bits: int) -> float:
    """Return the value of a float bit pattern (bf16 as the upper half of an f32)."""
    if dtype == "f16":
        return struct.unpack("<e", struct.pack("<H", bits))[0]
    if dtype == "bf16":
        return struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    if dtype == "f32":
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


@pytest.mark.parametrize("dtype", FLOATS)
def test_the_hand_written_patterns_are_the_values_they_are_named_for(dtype: str) -> None:
    p = PATTERNS[dtype]
    assert decoded(dtype, p["zero"]) == 0.0 and math.copysign(1.0, decoded(dtype, p["zero"])) == 1.0
    assert decoded(dtype, p["neg_zero"]) == 0.0 and math.copysign(1.0, decoded(dtype, p["neg_zero"])) == -1.0
    assert 0.0 < decoded(dtype, p["sub"]) == -decoded(dtype, p["neg_sub"])
    assert decoded(dtype, p["one"]) == 1.0 and decoded(dtype, p["neg_one"]) == -1.0
    assert decoded(dtype, p["below_one"]) < 1.0 and p["below_one"] + 1 == p["one"]
    assert decoded(dtype, p["below_neg_one"]) < -1.0 and p["below_neg_one"] == p["neg_one"] + 1
    assert decoded(dtype, HALF[dtype]) == 0.5 and decoded(dtype, HALF[dtype] | SIGN[dtype]) == -0.5
    assert all(math.isnan(decoded(dtype, bits)) for bits in NANS[dtype])
    assert [decoded(dtype, bits) for bits in INFS[dtype]] == [math.inf, -math.inf]


# ---------------------------------------------------------------------------
# Accepted sets


def test_an_in_contract_set_is_accepted_in_any_order(tmp_path: Path) -> None:
    contract = harness().input_contract([entry("a", "f32", [4], -1.0, 1.0), entry("k", "i32", [3], 0, 10)])
    a = put(tmp_path, "a.lassiio", "a", "f32", [4], [HALF["f32"]] * 4)
    k = put(tmp_path, "k.lassiio", "k", "i32", [3], [0, 5, 9])
    accepts(contract, [a, k])
    accepts(contract, [k, a])


ROUND_TRIP = [
    entry("h", "f16", [64], 0.1, 0.2),
    entry("bh", "bf16", [64], -0.3, 0.3),
    entry("bn", "bf16", [64], -1.0, 0.0),
    entry("bz", "bf16", [64], 0.0, 100.0),
    entry("f", "f32", [8, 8], -1e-40, 1e-40),
    entry("d", "f64", [64], -1.0, 0.0),
    entry("s", "f64", [], 0.0, 1.0),
    entry("e", "f32", [0], 0.0, 1.0),
    entry("z", "u8", [1 << 21, 0], 0, 256),
    entry("i8", "i8", [64], -128, 128),
    entry("u8", "u8", [64], 0, 256),
    entry("i32", "i32", [64], -5, 5),
    entry("u32", "u32", [64], 0, 1 << 32),
    entry("i64", "i64", [64], -(1 << 63), 1 << 63),
]


@pytest.mark.parametrize("seed", [0, 1, (1 << 64) - 1])
def test_generated_inputs_pass_the_contract_of_their_own_spec(tmp_path: Path, seed: int) -> None:
    paths = generate_inputs(ROUND_TRIP, seed, tmp_path)
    accepts(harness().input_contract(ROUND_TRIP), list(paths.values()))


# ---------------------------------------------------------------------------
# One test per refusal class


@pytest.mark.parametrize(
    "content",
    [b"not a lassi_io file\n", b"", "truncated", "version 2"],
    ids=["garbage", "empty", "truncated", "wrong-version"],
)
def test_a_file_that_is_not_a_readable_lassi_io_file_is_unreadable(tmp_path: Path, content: Any) -> None:
    contract = harness().input_contract([entry("a", "f32", [4], -1.0, 1.0), entry("k", "i32", [3], 0, 10)])
    k = put(tmp_path, "k.lassiio", "k", "i32", [3], [1, 2, 3])
    good = put(tmp_path, "good.lassiio", "a", "f32", [4], [HALF["f32"]] * 4).read_bytes()
    bad = tmp_path / "a.lassiio"
    if content == "truncated":
        bad.write_bytes(good[:-1])
    elif content == "version 2":
        bad.write_bytes(good[:8] + struct.pack("<I", 2) + good[12:])
    else:
        bad.write_bytes(content)
    (tmp_path / "good.lassiio").unlink()
    error = refused(contract, [bad, k], "unreadable", None, bad)
    with pytest.raises(LassiIOError) as direct:
        read_array(bad)
    assert error.detail == str(direct.value), "the refusal carries read_array's own message"


@pytest.mark.parametrize(
    "header",
    [
        MAGIC + struct.pack("<III", 1, 7, 0xFFFFFFFF),
        MAGIC + struct.pack("<III", 1, 7, 1) + struct.pack("<QI", 4, 200) + b"u",
        MAGIC + struct.pack("<III", 1, 99, 0xFFFFFFFF),
    ],
    ids=["rank-past-the-file", "name-past-the-file", "bad-dtype-code"],
)
def test_a_header_that_outruns_its_file_is_unreadable_with_read_arrays_message(tmp_path: Path, header: bytes) -> None:
    contract = harness().input_contract([entry("u", "u8", [4], 0, 256)])
    path = tmp_path / "u.lassiio"
    path.write_bytes(header)
    error = refused(contract, [path], "unreadable", None, path)
    with pytest.raises(LassiIOError) as direct:
        read_array(path)
    assert error.detail == str(direct.value)


def test_a_path_that_cannot_be_read_is_unreadable(tmp_path: Path) -> None:
    contract = harness().input_contract([entry("a", "f32", [4], -1.0, 1.0)])
    absent = tmp_path / "absent.lassiio"
    refused(contract, [absent], "unreadable", None, absent)
    refused(contract, [tmp_path], "unreadable", None, tmp_path)


def test_a_file_past_the_element_cap_is_refused_before_its_shape(tmp_path: Path) -> None:
    cap = registry.MAX_INPUT_ELEMENTS
    contract = harness().input_contract([entry("u", "u8", [16], 0, 256)])
    for shape in ([cap + 1], [1024, 1025]):
        path = tmp_path / "u.lassiio"
        write_array(path, "u", "u8", shape, bytes(math.prod(shape)))
        refused(contract, [path], "too-many-elements", "u", path)
    write_array(tmp_path / "u.lassiio", "u", "u8", [cap], bytes(cap))
    refused(contract, [tmp_path / "u.lassiio"], "shape", "u", tmp_path / "u.lassiio")


def test_the_cap_is_read_from_the_header_before_any_data(tmp_path: Path) -> None:
    cap = registry.MAX_INPUT_ELEMENTS
    contract = harness().input_contract([entry("u", "u8", [16], 0, 256)])
    for dims in ([cap + 1], [1 << 40, 1 << 40], [1 << 63, 1 << 63, 2]):
        short = raw_file(tmp_path / "short.lassiio", "u", 7, dims, bytes(16))
        refused(contract, [short], "too-many-elements", "u", short)
    long = raw_file(tmp_path / "long.lassiio", "u", 7, [16], bytes(16 + (1 << 22)))
    refused(contract, [long], "unreadable", None, long)


def test_a_zero_dim_holds_no_element_whatever_the_other_dims(tmp_path: Path) -> None:
    contract = harness().input_contract([entry("z", "u8", [1 << 21, 0], 0, 256)])
    assert contract.inputs[0].shape == (1 << 21, 0)
    accepts(contract, [put(tmp_path, "z.lassiio", "z", "u8", [1 << 21, 0], [])])
    other = put(tmp_path, "other.lassiio", "z", "u8", [0, 1 << 40], [])
    refused(contract, [other], "shape", "z", other)
    with pytest.raises(ValueError, match="elements"):
        harness().input_contract([entry("z", "u8", [1 << 21, 1], 0, 256)])


def two_inputs(tmp_path: Path) -> tuple[Any, Path, Path]:
    """Return the contract of a (f32 [4]) and k (i32 [3]) and an in-contract file for each."""
    contract = harness().input_contract([entry("a", "f32", [4], -1.0, 1.0), entry("k", "i32", [3], 0, 10)])
    a = put(tmp_path, "a.lassiio", "a", "f32", [4], [HALF["f32"]] * 4)
    k = put(tmp_path, "k.lassiio", "k", "i32", [3], [1, 2, 3])
    return contract, a, k


def test_a_missing_input_is_refused_naming_it(tmp_path: Path) -> None:
    contract, a, k = two_inputs(tmp_path)
    refused(contract, [a], "missing-input", "k")
    refused(contract, [k], "missing-input", "a")
    refused(contract, [], "missing-input", "a")


def test_an_extra_input_is_refused_naming_it(tmp_path: Path) -> None:
    contract, a, k = two_inputs(tmp_path)
    z = put(tmp_path, "z.lassiio", "zeta", "f32", [4], [HALF["f32"]] * 4)
    refused(contract, [a, k, z], "extra-input", "zeta", z)


def test_a_repeated_input_is_refused_naming_it(tmp_path: Path) -> None:
    contract, a, k = two_inputs(tmp_path)
    again = put(tmp_path, "again.lassiio", "a", "f32", [4], [HALF["f32"]] * 4)
    error = refused(contract, [a, k, again], "repeated-input", "a", again)
    assert str(a) in str(error), "the message names the first file too"


@pytest.mark.parametrize(
    ("dtype", "elements"),
    [("i32", [1, 2, 3, 4]), ("u32", [1, 2, 3, 4]), ("f64", [HALF["f64"]] * 4)],
)
def test_a_dtype_other_than_the_contracts_is_refused(tmp_path: Path, dtype: str, elements: list[int]) -> None:
    contract, _, k = two_inputs(tmp_path)
    a = put(tmp_path, "a.lassiio", "a", dtype, [4], elements)
    refused(contract, [a, k], "dtype", "a", a)


def test_bf16_and_f16_are_different_dtypes(tmp_path: Path) -> None:
    contract = harness().input_contract([entry("h", "f16", [2], -1.0, 1.0)])
    h = put(tmp_path, "h.lassiio", "h", "bf16", [2], [HALF["bf16"]] * 2)
    refused(contract, [h], "dtype", "h", h)


@pytest.mark.parametrize("shape", [[2, 2], [5], [3], [4, 1], [1, 4]])
def test_a_shape_other_than_the_contracts_is_refused(tmp_path: Path, shape: list[int]) -> None:
    contract, _, k = two_inputs(tmp_path)
    a = put(tmp_path, "a.lassiio", "a", "f32", shape, [HALF["f32"]] * math.prod(shape))
    refused(contract, [a, k], "shape", "a", a)


def test_a_scalar_and_a_one_element_array_differ_in_shape(tmp_path: Path) -> None:
    scalar = harness().input_contract([entry("s", "f64", [], 0.0, 1.0)])
    vector = harness().input_contract([entry("s", "f64", [1], 0.0, 1.0)])
    as_vector = put(tmp_path, "v.lassiio", "s", "f64", [1], [HALF["f64"]])
    as_scalar = put(tmp_path, "s.lassiio", "s", "f64", [], [HALF["f64"]])
    accepts(scalar, [as_scalar])
    accepts(vector, [as_vector])
    refused(scalar, [as_vector], "shape", "s", as_vector)
    refused(vector, [as_scalar], "shape", "s", as_scalar)


@pytest.mark.parametrize("dtype", FLOATS)
def test_a_nan_the_contract_does_not_allow_is_refused(tmp_path: Path, dtype: str) -> None:
    contract = single(dtype, -1.0, 1.0)
    for bits in NANS[dtype]:
        path = one_element(tmp_path, dtype, HALF[dtype], bits)
        refused(contract, [path], "nan", "x", path)


@pytest.mark.parametrize("dtype", FLOATS)
def test_an_infinity_the_contract_does_not_allow_is_refused(tmp_path: Path, dtype: str) -> None:
    contract = single(dtype, -1.0, 1.0)
    for bits in INFS[dtype]:
        path = one_element(tmp_path, dtype, HALF[dtype], bits)
        refused(contract, [path], "infinity", "x", path)


# ---------------------------------------------------------------------------
# The order of the checks


def test_each_file_is_checked_in_full_before_the_next_and_every_file_before_any_value(tmp_path: Path) -> None:
    contract, _, k = two_inputs(tmp_path)
    garbage = tmp_path / "garbage.lassiio"
    garbage.write_bytes(b"not a lassi_io file\n")
    out_a = put(tmp_path, "out.lassiio", "a", "f32", [4], [0x40000000] * 4)  # 2.0, outside [-1, 1)
    big = raw_file(tmp_path / "big.lassiio", "a", 1, [registry.MAX_INPUT_ELEMENTS + 1], b"")
    refused(contract, [out_a, k, garbage], "unreadable", None, garbage)
    refused(contract, [big, garbage], "too-many-elements", "a", big)
    refused(contract, [garbage, big], "unreadable", None, garbage)


def test_names_are_checked_in_file_order_before_any_input(tmp_path: Path) -> None:
    contract, a, k = two_inputs(tmp_path)
    z = put(tmp_path, "z.lassiio", "zeta", "f32", [4], [HALF["f32"]] * 4)
    again = put(tmp_path, "again.lassiio", "a", "f32", [4], [HALF["f32"]] * 4)
    refused(contract, [z, a, again, k], "extra-input", "zeta", z)
    refused(contract, [a, again, z, k], "repeated-input", "a", again)
    wide = put(tmp_path, "wide.lassiio", "a", "f64", [4], [HALF["f64"]] * 4)
    refused(contract, [wide], "missing-input", "k")


def test_inputs_are_checked_in_the_contracts_order_not_the_files(tmp_path: Path) -> None:
    contract, _, _ = two_inputs(tmp_path)
    out_a = put(tmp_path, "out.lassiio", "a", "f32", [4], [0x40000000] * 4)  # 2.0, outside [-1, 1)
    k64 = put(tmp_path, "k64.lassiio", "k", "i64", [3], [1, 2, 3])
    refused(contract, [k64, out_a], "out-of-range", "a", out_a)


def test_dtype_comes_before_shape_and_shape_before_any_value(tmp_path: Path) -> None:
    contract = single("f32", -1.0, 1.0)
    u8 = put(tmp_path, "u8.lassiio", "x", "u8", [3], [255, 255, 255])
    refused(contract, [u8], "dtype", "x", u8)
    five = put(tmp_path, "five.lassiio", "x", "f32", [5], [NANS["f32"][0]] * 5)
    refused(contract, [five], "shape", "x", five)


@pytest.mark.parametrize("dtype", FLOATS)
def test_a_nan_comes_before_an_infinity_and_an_infinity_before_a_value_out_of_range(tmp_path: Path, dtype: str) -> None:
    contract = single(dtype, -1.0, 1.0)
    one, nan, inf, half = PATTERNS[dtype]["one"], NANS[dtype][0], INFS[dtype][1], HALF[dtype]
    for elements, reason, index in (
        ([one, inf, nan, half], "nan", 2),
        ([one, one, inf, half], "infinity", 2),
        ([half, one, one, half], "out-of-range", 1),
    ):
        path = put(tmp_path, "x.lassiio", "x", dtype, [4], elements)
        error = refused(contract, [path], reason, "x", path)
        assert f"element {index} " in str(error), error


# ---------------------------------------------------------------------------
# Boundaries, exactly: lo accepted, hi refused, -0, the smallest subnormals, the largest value below hi


# (range, pattern, accepted): each float dtype's patterns against [0, 1) and [-1, 0).
FLOAT_BOUNDARIES = [
    ((0.0, 1.0), "zero", True),
    ((0.0, 1.0), "neg_zero", True),
    ((0.0, 1.0), "sub", True),
    ((0.0, 1.0), "below_one", True),
    ((0.0, 1.0), "one", False),
    ((0.0, 1.0), "neg_sub", False),
    ((0.0, 1.0), "neg_one", False),
    ((-1.0, 0.0), "neg_one", True),
    ((-1.0, 0.0), "neg_sub", True),
    ((-1.0, 0.0), "below_neg_one", False),
    ((-1.0, 0.0), "zero", False),
    ((-1.0, 0.0), "neg_zero", False),
    ((-1.0, 0.0), "sub", False),
    ((-1.0, 0.0), "below_one", False),
]


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize(
    ("bounds", "pattern", "accepted"),
    FLOAT_BOUNDARIES,
    ids=[f"[{lo},{hi})-{pattern}-{'in' if ok else 'out'}" for (lo, hi), pattern, ok in FLOAT_BOUNDARIES],
)
def test_float_boundaries_are_exact(
    tmp_path: Path, dtype: str, bounds: tuple[float, float], pattern: str, accepted: bool
) -> None:
    lo, hi = bounds
    filler = HALF[dtype] if lo == 0.0 else HALF[dtype] | SIGN[dtype]
    path = one_element(tmp_path, dtype, filler, PATTERNS[dtype][pattern])
    if accepted:
        accepts(single(dtype, lo, hi), [path])
    else:
        refused(single(dtype, lo, hi), [path], "out-of-range", "x", path)


@pytest.mark.parametrize(
    ("bits", "accepted"),
    [(0x42C7, True), (0x42C8, False), (0x0000, True), (0x8000, True), (0x8001, False)],
    ids=["99.5-largest-below-hi", "100-hi", "zero-lo", "neg-zero", "neg-sub"],
)
def test_a_bf16_range_whose_hi_is_100_is_exact(tmp_path: Path, bits: int, accepted: bool) -> None:
    contract = single("bf16", 0.0, 100.0)
    path = one_element(tmp_path, "bf16", HALF["bf16"], bits)
    if accepted:
        accepts(contract, [path])
    else:
        refused(contract, [path], "out-of-range", "x", path)


# (dtype, lo, hi, bits, accepted): bounds that are no dtype value, so each rounds to a neighbour on one side of it.
ROUNDED_BOUNDARIES = [
    ("f16", 0.1, 0.2, 0x2E66, False),  # 0.0999756, the f16 nearest lo, lies below it
    ("f16", 0.1, 0.2, 0x2E67, True),  # 0.1000366, the smallest f16 in range
    ("f16", 0.1, 0.2, 0x3266, True),  # 0.1999512, the f16 nearest hi, the largest below it
    ("f16", 0.1, 0.2, 0x3267, False),  # 0.2000732
    ("f16", -0.2, -0.1, 0xB267, False),  # -0.2000732
    ("f16", -0.2, -0.1, 0xB266, True),  # -0.1999512, the f16 nearest lo, inside it
    ("f16", -0.2, -0.1, 0xAE67, True),  # -0.1000366, the largest below hi
    ("f16", -0.2, -0.1, 0xAE66, False),  # -0.0999756, the f16 nearest hi, above it
    ("f16", 0.0, 0.3, 0x34CC, True),  # 0.2998047, the largest below hi
    ("f16", 0.0, 0.3, 0x34CD, False),  # 0.3000488, the f16 nearest hi, above it
    ("f16", -0.3, 0.0, 0xB4CD, False),  # -0.3000488, the f16 nearest lo, below it
    ("f16", -0.3, 0.0, 0xB4CC, True),  # -0.2998047, the smallest in range
    ("f32", 1.5e-45, 3e-45, 0x00000001, False),  # the smallest subnormal, nearest lo, below it
    ("f32", 1.5e-45, 3e-45, 0x00000002, True),  # twice it, nearest hi, below it
    ("f32", 1.5e-45, 3e-45, 0x00000003, False),
    ("f32", -3e-45, -1.5e-45, 0x80000003, False),
    ("f32", -3e-45, -1.5e-45, 0x80000002, True),  # nearest lo, inside it
    ("f32", -3e-45, -1.5e-45, 0x80000001, False),  # nearest hi, above it
]


@pytest.mark.parametrize(
    ("dtype", "lo", "hi", "bits", "accepted"),
    ROUNDED_BOUNDARIES,
    ids=[f"{dtype}-[{lo},{hi})-{bits:#x}" for dtype, lo, hi, bits, _ in ROUNDED_BOUNDARIES],
)
def test_bounds_between_dtype_values_admit_exactly_the_values_inside(
    tmp_path: Path, dtype: str, lo: float, hi: float, bits: int, accepted: bool
) -> None:
    assert (lo <= decoded(dtype, bits) < hi) == accepted, "the case table agrees with the decoded value"
    path = one_element(tmp_path, dtype, bits, bits)
    if accepted:
        accepts(single(dtype, lo, hi), [path])
    else:
        refused(single(dtype, lo, hi), [path], "out-of-range", "x", path)


INT_BOUNDARIES = [
    ("i32", 0, 10, 0, True),
    ("i32", 0, 10, 9, True),
    ("i32", 0, 10, 10, False),
    ("i32", 0, 10, -1, False),
    ("i8", -128, 128, -128, True),
    ("i8", -128, 128, 127, True),
    ("u8", 1, 255, 1, True),
    ("u8", 1, 255, 254, True),
    ("u8", 1, 255, 0, False),
    ("u8", 1, 255, 255, False),
    ("u32", 0, 1 << 32, (1 << 32) - 1, True),
    ("u32", 1, 1 << 32, 0, False),
    ("i64", -(1 << 63), 1 << 63, -(1 << 63), True),
    ("i64", -(1 << 63), 1 << 63, (1 << 63) - 1, True),
    ("i64", -5, 5, -5, True),
    ("i64", -5, 5, 4, True),
    ("i64", -5, 5, 5, False),
    ("i64", -5, 5, -6, False),
]


@pytest.mark.parametrize(
    ("dtype", "lo", "hi", "value", "accepted"),
    INT_BOUNDARIES,
    ids=[f"{dtype}-[{lo},{hi})-{value}" for dtype, lo, hi, value, _ in INT_BOUNDARIES],
)
def test_integer_boundaries_are_exact(tmp_path: Path, dtype: str, lo: int, hi: int, value: int, accepted: bool) -> None:
    path = one_element(tmp_path, dtype, lo, value)
    if accepted:
        accepts(single(dtype, lo, hi), [path])
    else:
        refused(single(dtype, lo, hi), [path], "out-of-range", "x", path)


# ---------------------------------------------------------------------------
# A declared contract: wider ranges, and NaN or infinity let through


def test_a_declared_range_replaces_the_spec_range(tmp_path: Path) -> None:
    spec = [entry("x", "f32", [4], 0.0, 1.0), entry("k", "i32", [3], 0, 10)]
    declared = [
        {"name": "x", "dtype": "f32", "lo": -2.0, "hi": 2.0},
        {"name": "k", "dtype": "i32", "lo": -100, "hi": 100},
    ]
    contract = harness().input_contract(spec, declared=declared)
    wide_k = put(tmp_path, "wide_k.lassiio", "k", "i32", [3], [-100, 50, 99])
    hi_k = put(tmp_path, "hi_k.lassiio", "k", "i32", [3], [0, 0, 100])
    for bits in (0xC0000000, 0xBFC00000, 0x3FC00000):  # -2.0 (lo), -1.5, and 1.5, outside the spec's [0, 1)
        accepts(contract, [one_element(tmp_path, "f32", HALF["f32"], bits), wide_k])
    two = one_element(tmp_path, "f32", HALF["f32"], 0x40000000)  # 2.0, the declared hi
    refused(contract, [two, wide_k], "out-of-range", "x", two)
    good_x = one_element(tmp_path, "f32", HALF["f32"], HALF["f32"])
    refused(contract, [good_x, hi_k], "out-of-range", "k", hi_k)
    refused(harness().input_contract(spec), [good_x, wide_k], "out-of-range", "k", wide_k)


# (dtype, spec lo, spec hi, declared lo, declared hi): declared ranges that hold every dtype value of the spec's.
COVERING = [
    ("f32", -1.0, 1.0, -1.0, 1.0),
    ("f32", -1.0, 1.0, -1.00000001, 1.0),  # below -1.0, above the next f32 down: the same f32 values
    ("f32", -1.0, 1.0, -1.0, 0.99999998),  # above 1 - 2**-24, below 1.0: the same f32 values
    ("f16", 0.1, 0.2, 0.1000366, 0.2),  # 0x2E67, the spec's smallest f16 value
    ("i32", 0, 10, 0, 10),
    ("i32", 0, 10, -5, 11),
]
# Declared ranges that leave out at least one dtype value of the spec's range.
NARROWER = [
    ("f32", -1.0, 1.0, -0.5, 1.0),
    ("f32", -1.0, 1.0, -1.0, 0.5),
    ("f32", -1.0, 1.0, -0.99999999, 1.0),  # leaves out -1.0 alone
    ("f32", -1.0, 1.0, -1.0, 1.0 - 2.0**-24),  # leaves out 1 - 2**-24 alone
    ("f16", 0.1, 0.2, 0.1000367, 0.2),  # leaves out 0x2E67 alone
    ("f64", 0.0, 1.0, 5e-324, 1.0),  # leaves out 0 alone
    ("i32", 0, 10, 1, 10),
    ("i32", 0, 10, 0, 9),
    ("u8", 0, 256, 0, 255),
]


def declare(dtype: str, lo: float | int, hi: float | int, spec: tuple[float | int, float | int]) -> Any:
    """Return input_contract of one input x with the spec range `spec` and the declared range [lo, hi)."""
    declared = [{"name": "x", "dtype": dtype, "lo": lo, "hi": hi}]
    return harness().input_contract([entry("x", dtype, [4], *spec)], declared=declared)


@pytest.mark.parametrize(("dtype", "spec_lo", "spec_hi", "lo", "hi"), COVERING)
def test_a_declared_range_may_widen_the_spec_range(
    tmp_path: Path, dtype: str, spec_lo: float | int, spec_hi: float | int, lo: float | int, hi: float | int
) -> None:
    contract = declare(dtype, lo, hi, (spec_lo, spec_hi))
    assert (contract.inputs[0].lo, contract.inputs[0].hi) == (lo, hi)
    accepts(contract, list(generate_inputs([entry("x", dtype, [4], spec_lo, spec_hi)], 3, tmp_path).values()))


@pytest.mark.parametrize(("dtype", "spec_lo", "spec_hi", "lo", "hi"), NARROWER)
def test_a_declared_range_may_never_narrow_the_spec_range(
    dtype: str, spec_lo: float | int, spec_hi: float | int, lo: float | int, hi: float | int
) -> None:
    with pytest.raises(ValueError, match=r"^contract: input 'x': .*narrow"):
        declare(dtype, lo, hi, (spec_lo, spec_hi))


def test_a_declared_range_follows_generate_inputs_bound_rules() -> None:
    biggest = sys.float_info.max
    with pytest.raises(LassiIOError, match="overflows"):
        check_input_spec([entry("x", "f64", [4], -biggest, biggest)], "spec")
    with pytest.raises(ValueError, match="overflows"):
        declare("f64", -biggest, biggest, (-1.0, 1.0))
    assert declare("f64", 0.0, biggest, (0.0, 1.0)).inputs[0].hi == biggest
    with pytest.raises(ValueError, match="no f16 value"):
        declare("f16", 1.0001, 1.0002, (1.0, 2.0))


def test_the_flags_apply_to_a_float_dtype_only() -> None:
    spec = [entry("k", "i32", [4], 0, 10)]
    for flag in ("allow_nan", "allow_inf"):
        with pytest.raises(ValueError, match="float dtype"):
            harness().input_contract(spec, declared=[{"name": "k", "dtype": "i32", "lo": 0, "hi": 10, flag: True}])
        declared = [{"name": "k", "dtype": "i32", "lo": 0, "hi": 10, flag: False}]
        assert harness().input_contract(spec, declared=declared) == harness().input_contract(spec)


@pytest.mark.parametrize("dtype", FLOATS)
def test_allow_nan_lets_a_nan_through_and_no_infinity(tmp_path: Path, dtype: str) -> None:
    declared = [{"name": "x", "dtype": dtype, "lo": -1.0, "hi": 1.0, "allow_nan": True}]
    contract = harness().input_contract([entry("x", dtype, [4], -1.0, 1.0)], declared=declared)
    for bits in NANS[dtype]:
        accepts(contract, [one_element(tmp_path, dtype, HALF[dtype], bits)])
    for bits in INFS[dtype]:
        path = one_element(tmp_path, dtype, HALF[dtype], bits)
        refused(contract, [path], "infinity", "x", path)


@pytest.mark.parametrize("dtype", FLOATS)
def test_allow_inf_lets_either_infinity_through_and_no_nan(tmp_path: Path, dtype: str) -> None:
    declared = [{"name": "x", "dtype": dtype, "lo": -1.0, "hi": 1.0, "allow_inf": True}]
    contract = harness().input_contract([entry("x", dtype, [4], -1.0, 1.0)], declared=declared)
    for bits in INFS[dtype]:
        accepts(contract, [one_element(tmp_path, dtype, HALF[dtype], bits)])
    for bits in NANS[dtype]:
        path = one_element(tmp_path, dtype, HALF[dtype], bits)
        refused(contract, [path], "nan", "x", path)
    path = one_element(tmp_path, dtype, HALF[dtype], PATTERNS[dtype]["one"])
    refused(contract, [path], "out-of-range", "x", path)


def test_flags_left_out_or_false_allow_neither(tmp_path: Path) -> None:
    spec = [entry("x", "f32", [4], -1.0, 1.0)]
    for declared in (
        [{"name": "x", "dtype": "f32", "lo": -1.0, "hi": 1.0}],
        [{"name": "x", "dtype": "f32", "lo": -1.0, "hi": 1.0, "allow_nan": False, "allow_inf": False}],
    ):
        contract = harness().input_contract(spec, declared=declared)
        nan = one_element(tmp_path, "f32", HALF["f32"], NANS["f32"][0])
        refused(contract, [nan], "nan", "x", nan)
        inf = one_element(tmp_path, "f32", HALF["f32"], INFS["f32"][0])
        refused(contract, [inf], "infinity", "x", inf)


# ---------------------------------------------------------------------------
# Arguments, and the refusal across a process boundary


def test_check_inputs_refuses_a_missing_contract_and_files_that_are_not_a_sequence(tmp_path: Path) -> None:
    h = harness()
    contract, a, k = two_inputs(tmp_path)
    with pytest.raises(TypeError, match="InputContract, got NoneType"):
        h.check_inputs(None, [a, k])
    for files in (a, str(a), str(a).encode("ascii"), (path for path in (a, k))):
        with pytest.raises(TypeError, match=f"got a {type(files).__name__}$"):
            h.check_inputs(contract, files)


def test_a_refusal_pickles_with_its_reason_name_path_and_message(tmp_path: Path) -> None:
    contract, a, _ = two_inputs(tmp_path)
    garbage = tmp_path / "garbage.lassiio"
    garbage.write_bytes(b"not a lassi_io file\n")
    out = put(tmp_path, "out.lassiio", "a", "f32", [4], [0x40000000] * 4)
    for error in (
        refused(contract, [a], "missing-input", "k"),
        refused(contract, [garbage], "unreadable", None, garbage),
        refused(contract, [out, a], "repeated-input", "a", a),
    ):
        copy = pickle.loads(pickle.dumps(error))
        assert type(copy) is type(error) and str(copy) == str(error) and copy.args == error.args
        assert (copy.reason, copy.detail, copy.name, copy.path) == (error.reason, error.detail, error.name, error.path)


# ---------------------------------------------------------------------------
# The check only reads


def snapshot(directory: Path) -> dict[str, bytes]:
    """Return every file under `directory` with its bytes."""
    return {str(path.relative_to(directory)): path.read_bytes() for path in sorted(directory.rglob("*"))}


def test_the_check_writes_changes_and_removes_nothing(tmp_path: Path) -> None:
    contract, a, k = two_inputs(tmp_path)
    z = put(tmp_path, "z.lassiio", "z", "f32", [4], [HALF["f32"]] * 4)
    before = snapshot(tmp_path)
    accepts(contract, [a, k])
    refused(contract, [a, k, z], "extra-input", "z", z)
    refused(contract, [a], "missing-input", "k")
    assert snapshot(tmp_path) == before
    assert read_array(a).data == pack("f32", [HALF["f32"]] * 4)
