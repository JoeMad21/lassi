"""Tests for the cpp -> ttnn and omp -> ttnn suite bale-v0-ttnn and its item contracts (task P17.57).

Bible: Benchmark Suites (bale-v0-tt; manifest keys), IR Strategy (the gather
and scatter routes), Harness Contract (lassi_io), Agent Rule 5 (every item
is unassigned: no item's answer in a tracked file or a prompt). Evidence:
plans/spikes/scatter-gather-workarounds.md, sections 3 and 5 and "Compile
probe at the pin (2026-10-29)".

What they fix:

- The suite is a sibling of bale-v0-tt.yaml: the same eight items with the
  same splits, held-out inputs, seeds, outputs, tolerances, and cpp and omp
  languages, the same repo and commit, and a ttnn language in place of ttkh.
- The ttnn language is one module per item, <item>.mlir, tracked, with
  `reference: cpp` and no tracked module, and one support file,
  boundary.yaml, pinned by sha256: the item contract.
- Each contract (the format is the manifest's header) holds against its
  item's declared inputs: every input is an argument or a static scalar and
  every output a result, once and in order; every narrowed input's declared
  range fits its device type (si32 or ui32; ui16 where a sort key); a static
  scalar's range holds one value, its `value`; ig's table goes as bf16
  columns of 8-bit digits whose bias and count cover its declared range,
  each digit exact in bf16, and comes back by recombination; and a padded
  output's length is a result and its padding holds the most entries the
  output can have.
- The conversions below are the contract's rules written as code, the
  statement the TT-NN executor (task P17.58) follows; no held-out input is
  generated here, only the manifest's declared bounds are read.
"""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import Direction, load_suite

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "assets" / "bench" / "bale-v0-ttnn.yaml"
SUITE = load_suite(MANIFEST)
ORIGINAL = load_suite(REPO / "assets" / "bench" / "bale-v0-tt.yaml")
ITEMS = ("histo", "ig", "permute_matrix", "randperm", "sssp", "toposort", "transpose_matrix", "triangle")
KEYS = ("split", "inputs", "outputs", "seed", "tolerance", "unpack_to_dest", "contract", "work", "support",
        "compute", "run_args", "passfail")
# The value range of each device integer type a contract may name, inclusive.
RANGES = {"si32": (-2**31, 2**31 - 1), "ui32": (0, 2**32 - 1), "ui16": (0, 2**16 - 1)}
ARG_KEYS = {"narrow": {"name", "input", "dtype", "shape", "convert"},
            "digits": {"name", "input", "dtype", "shape", "convert", "digits", "bias"}}
RESULT_KEYS = {"widen": {"name", "output", "dtype", "shape", "convert"},
               "recombine": {"name", "output", "dtype", "shape", "convert", "digits", "bias"},
               "trim": {"name", "output", "dtype", "shape", "convert", "length", "at_most"},
               "length": {"name", "dtype", "shape", "convert"}}
# A ttnn.embedding table's width when its output is tiled (spike, section 3).
TILE = 32


def boundary(name: str) -> dict[str, Any]:
    """Return item `name`'s contract, read from its ttnn support file."""
    entry = SUITE.items[name].languages["ttnn"].support["boundary.yaml"]
    return yaml.safe_load((REPO / entry.path).read_text(encoding="ascii"))


def declared(name: str) -> dict[str, dict[str, Any]]:
    """Return item `name`'s declared held-out input entries by input name (its bounds, never its values)."""
    return {entry["name"]: entry for entry in SUITE.items[name].inputs}


def count(shape: list[int]) -> int:
    """Return the element count of a shape ([] is a scalar, one element)."""
    return math.prod(shape)


def bf16(value: float) -> float:
    """Return `value` rounded to bfloat16 (round to nearest, ties to even), as a float."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def narrow(values: list[int], dtype: str) -> list[int]:
    """Return `values` for a narrow argument of `dtype`; ValueError for any value outside the type (the range check)."""
    low, high = RANGES[dtype]
    bad = [value for value in values if not low <= value <= high]
    if bad:
        raise ValueError(f"{len(bad)} values outside {dtype}, the first {bad[0]}")
    return list(values)


def to_digits(values: list[int], digits: int, bias: int, width: int) -> list[list[float]]:
    """Return the bf16 digit rows of `values` (column k: ((v + bias) >> 8k) & 255, zeros after); ValueError outside."""
    rows = []
    for value in values:
        shifted = value + bias
        if not 0 <= shifted < 256**digits:
            raise ValueError(f"{value} + bias is outside [0, 256**{digits})")
        rows.append([bf16(float((shifted >> (8 * k)) & 255)) for k in range(digits)] + [0.0] * (width - digits))
    return rows


def recombine(rows: list[list[float]], digits: int, bias: int) -> list[int]:
    """Return the i64 values of bf16 digit rows; ValueError for a digit that is no integer in [0, 255]."""
    out = []
    for row in rows:
        if any(not (float(digit).is_integer() and 0 <= digit <= 255) for digit in row[:digits]):
            raise ValueError(f"a digit outside the integers 0 to 255: {row[:digits]}")
        out.append(sum(int(digit) << (8 * k) for k, digit in enumerate(row[:digits])) - bias)
    return out


def trim(padded: list[int], length: int) -> list[int]:
    """Return the first `length` elements of a padded result; ValueError when the length is outside [0, padding]."""
    if not 0 <= length <= len(padded):
        raise ValueError(f"length {length} outside [0, {len(padded)}]")
    return padded[:length]


# ---------------------------------------------------------------------------
# The suite


def test_the_sibling_holds_bale_v0_tt_unchanged() -> None:
    assert SUITE.name == "bale-v0-ttnn" and sorted(SUITE.items) == list(ITEMS) == sorted(ORIGINAL.items)
    assert (SUITE.repo, SUITE.commit, SUITE.pin) == (ORIGINAL.repo, ORIGINAL.commit, ORIGINAL.pin)
    for name in ITEMS:
        ours, theirs = SUITE.items[name], ORIGINAL.items[name]
        for key in KEYS:
            assert getattr(ours, key) == getattr(theirs, key), f"{name}.{key} drifted from bale-v0-tt"
        assert set(ours.languages) == {"cpp", "omp", "ttnn"} and ours.split == "unassigned"
        for kept in ("cpp", "omp"):
            assert ours.languages[kept] == theirs.languages[kept], f"{name}.{kept} drifted"


@pytest.mark.parametrize("name", ITEMS)
def test_the_ttnn_language_is_one_module_with_the_cpp_reference_and_its_contract(name: str, tmp_path: Path) -> None:
    spec = SUITE.items[name].languages["ttnn"]
    assert spec.files == (f"{name}.mlir",) and spec.tracked and spec.reference == "cpp"
    assert not (REPO / spec.dir).exists(), "no reference module is tracked: it would be the item's answer"
    (contract,) = spec.support.items()
    path = f"assets/bench/bale-v0-ttnn/{name}/boundary.yaml"
    assert contract[0] == "boundary.yaml" and (contract[1].path, contract[1].tracked) == (path, True)
    data = (REPO / path).read_bytes()
    assert contract[1].sha256 == hashlib.sha256(data).hexdigest() and data.isascii() and b"\r" not in data
    assert SUITE.support_files(name, tmp_path, purpose="eval", language="ttnn") == {"boundary.yaml": data.decode()}
    assert SUITE.reference_language(name, "ttnn") == "cpp"
    for source in ("cpp", "omp"):
        assert SUITE.bench_item(name, Direction(source, "ttnn")).direction == f"{source}-ttnn"


def test_the_manifest_is_plain_ascii_and_names_its_task() -> None:
    data = MANIFEST.read_bytes()
    assert data.isascii() and b"\r" not in data
    text = data.decode("ascii")
    assert "P17.57" in text and "plans/spikes/scatter-gather-workarounds.md" in text and "ttir-ttnn-compile" in text
    assert "ttkh" not in text.split("suite: bale-v0-ttnn", 1)[1], "ttnn takes the place of ttkh"


# ---------------------------------------------------------------------------
# The contracts against their items


@pytest.mark.parametrize("name", ITEMS)
def test_each_contract_names_every_input_and_output_once_in_order(name: str) -> None:
    data, item = boundary(name), SUITE.items[name]
    assert set(data) <= {"item", "static", "args", "results"} and data["item"] == name
    statics = [entry["input"] for entry in data.get("static", [])]
    args = [entry["input"] for entry in data["args"]]
    assert sorted(statics + args) == sorted(declared(name)), "every input once, as an argument or a static scalar"
    assert args == [entry["name"] for entry in item.inputs if entry["name"] not in statics], "arguments in input order"
    outputs = [entry["output"] for entry in data["results"] if entry["convert"] != "length"]
    assert outputs == list(item.outputs), "every output once, in the item's order"
    names = [entry["name"] for entry in data["args"] + data["results"]]
    assert len(names) == len(set(names))
    for entry in data["args"]:
        assert set(entry) - {"fits"} == ARG_KEYS[entry["convert"]], entry
    for entry in data["results"]:
        assert set(entry) - {"output_shape"} == RESULT_KEYS[entry["convert"]], entry


@pytest.mark.parametrize("name", ITEMS)
def test_each_narrowed_input_fits_its_device_type(name: str) -> None:
    found = declared(name)
    narrowed = [entry for entry in boundary(name)["args"] if entry["convert"] == "narrow"]
    assert narrowed, "every item passes at least one array"
    for entry in narrowed:
        spec = found[entry["input"]]
        assert spec["dtype"] == "i64" and spec["dist"] == "integers"
        assert count(entry["shape"]) == count(spec["shape"]), "the same elements, row-major"
        for dtype in (entry["dtype"], entry.get("fits", entry["dtype"])):
            low, high = RANGES[dtype]
            assert low <= spec["lo"] and spec["hi"] - 1 <= high, f"{name}.{entry['input']} does not fit {dtype}"


@pytest.mark.parametrize("name", ITEMS)
def test_each_static_input_is_a_scalar_whose_range_holds_one_value(name: str) -> None:
    found = declared(name)
    for entry in boundary(name).get("static", []):
        spec = found[entry["input"]]
        assert spec["shape"] == [] and (spec["lo"], spec["hi"]) == (entry["value"], entry["value"] + 1)


@pytest.mark.parametrize("name", ITEMS)
def test_each_result_is_a_device_type_and_a_padded_one_names_its_length_and_bound(name: str) -> None:
    results = {entry["name"]: entry for entry in boundary(name)["results"]}
    found = declared(name)
    for entry in results.values():
        assert entry["dtype"] in ("si32", "bf16") and all(dim > 0 for dim in entry["shape"])
        if entry["convert"] == "widen":
            written = entry.get("output_shape", entry["shape"])
            assert entry["dtype"] == "si32" and count(written) == count(entry["shape"])
        if entry["convert"] == "trim":
            length = results[entry["length"]]
            assert (length["convert"], length["dtype"], length["shape"]) == ("length", "si32", [1])
            assert entry["dtype"] == "si32" and len(entry["shape"]) == 1
            assert count(found[entry["at_most"]]["shape"]) <= entry["shape"][0], "the padding holds every entry"
    lengths = {entry["name"] for entry in results.values() if entry["convert"] == "length"}
    assert lengths == {entry["length"] for entry in results.values() if entry["convert"] == "trim"}


def test_the_padded_outputs_are_the_two_sparse_matrices_columns() -> None:
    padded = {name: [entry["output"] for entry in boundary(name)["results"] if entry["convert"] == "trim"]
              for name in ITEMS}
    assert {name: found for name, found in padded.items() if found} == {
        "permute_matrix": ["column"], "transpose_matrix": ["column"]}


def test_igs_table_goes_as_8_bit_digit_columns_that_cover_its_range() -> None:
    (table, index), (tgt,) = boundary("ig")["args"], boundary("ig")["results"]
    spec = declared("ig")["table"]
    assert (table["convert"], table["dtype"], tgt["convert"], tgt["dtype"]) == ("digits", "bf16", "recombine", "bf16")
    digits, bias, width = table["digits"], table["bias"], table["shape"][1]
    assert (tgt["digits"], tgt["bias"], tgt["shape"]) == (digits, bias, table["shape"])
    assert table["shape"][0] == count(spec["shape"]) == count(tgt["output_shape"])
    assert spec["lo"] + bias >= 0 and spec["hi"] - 1 + bias < 256**digits, "the digits cover the declared range"
    assert digits <= width and width % TILE == 0, "a tiled embedding's table is a multiple of 32 wide"
    assert (index["convert"], index["dtype"]) == ("narrow", "si32")
    assert all(bf16(float(digit)) == digit for digit in range(256)), "every 8-bit digit is exact in bf16"
    assert bf16(257.0) != 257.0, "a 9-bit digit is not, so the digits are 8 bits"


# ---------------------------------------------------------------------------
# The conversions, as the contracts state them


def test_igs_table_round_trips_through_bf16_digits_at_its_bounds() -> None:
    table, spec = boundary("ig")["args"][0], declared("ig")["table"]
    values = [spec["lo"], spec["lo"] + 1, -1, 0, 1, 255, 256, 2**32 + 7, spec["hi"] - 1]
    rows = to_digits(values, table["digits"], table["bias"], table["shape"][1])
    assert all(len(row) == table["shape"][1] and row[table["digits"]:] == [0.0] * (len(row) - table["digits"])
               for row in rows)
    assert recombine(rows, table["digits"], table["bias"]) == values


def test_the_conversions_refuse_what_the_contracts_refuse() -> None:
    assert narrow([-2**31, 0, 2**31 - 1], "si32") == [-2**31, 0, 2**31 - 1]
    for value, dtype in ((2**31, "si32"), (-2**31 - 1, "si32"), (-1, "ui32"), (65536, "ui16")):
        with pytest.raises(ValueError, match="outside"):
            narrow([value], dtype)
    table = boundary("ig")["args"][0]
    for value in (-table["bias"] - 1, 256 ** table["digits"] - table["bias"]):
        with pytest.raises(ValueError, match="outside"):
            to_digits([value], table["digits"], table["bias"], table["shape"][1])
    for row in ([256.0, 0.0], [0.5, 0.0], [-1.0, 0.0]):
        with pytest.raises(ValueError, match="digit"):
            recombine([row], 2, 0)
    assert trim([4, 5, 6, 0, 0], 3) == [4, 5, 6] and trim([0, 0], 0) == []
    for length in (-1, 6):
        with pytest.raises(ValueError, match="length"):
            trim([4, 5, 6, 0, 0], length)
