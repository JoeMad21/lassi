"""Tests for the optional manifest key `contract` and an item's input contract (task P16.1).

Bible: Benchmark Suites (held-out inputs; manifest keys), Harness Contract
(held-out input files), Adversarial Setups (the adversary rule), Agent Rule
5. Plan: plans/p16-adversarial.md, P16.1 and the planning decision "Input
contract (P16.1)". The check itself is fixed in
tests/harness/test_input_contract.py. What these tests fix, in
lassi.bench.registry:

- An item with held-out inputs may declare `contract`: a list with one
  entry per held-out input, each with the keys name (a str), dtype, lo, and
  hi and the optional allow_nan and allow_inf (true or false; true only on
  a float dtype). Its dtype is that input's dtype. Its bounds follow
  generate_inputs' bound rules for the dtype: for a float dtype, finite
  numbers inside the dtype's finite range (its largest finite value is
  allowed), lo below hi; for an integer dtype, ints with lo below hi and
  [lo, hi) inside the dtype's range. Its range holds every dtype value of
  the input's own range, so it may widen the range and never narrow it.
  load_suite refuses a malformed key (`contract: null` among them), and the
  key on an item with no held-out inputs, with a ValueError that names the
  item and `contract`.
- load_suite's element cap counts a shape's elements as the check does: a
  zero dim gives none, whatever the other dims.
- SuiteItem.contract is the item's InputContract (lassi.harness):
  input_contract(inputs, declared=<the key>) when the item declares the key,
  input_contract(inputs) when it does not, and None for an item with no
  held-out inputs.
- The key never changes the held-out inputs: inputs, outputs, seed,
  program_args, and the bytes stage_inputs writes are those of the same
  item without it.

Items are read through Suite.item with purpose eval only (Agent Rule 5).
The manifest is SYNTHETIC; no value here is a measurement.
"""

from __future__ import annotations

import copy
import importlib
import math
import struct
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from lassi import bench
from lassi.bench import registry
from lassi.harness.lassi_io import write_array

SUITE = "keyed-fixture"
ITEM = "vadd"
PLAIN = "plain"
INPUTS = [
    {"name": "a", "dtype": "f32", "shape": [2, 4], "dist": "uniform", "lo": -1.0, "hi": 1.0},
    {"name": "k", "dtype": "i32", "shape": [8], "dist": "integers", "lo": 0, "hi": 10},
]
CONTRACT = [
    {"name": "a", "dtype": "f32", "lo": -2.0, "hi": 2.0},
    {"name": "k", "dtype": "i32", "lo": -100, "hi": 100},
]
FLT_MAX = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]


def harness() -> ModuleType:
    """Import and return lassi.harness, where check_inputs and input_contract live."""
    return importlib.import_module("lassi.harness")


def manifest_data(contract: Any = None) -> dict[str, Any]:
    """Return the SYNTHETIC manifest: item vadd with held-out inputs (and `contract` when given), plain without."""
    item: dict[str, Any] = {
        "split": "unassigned",
        "seed": 7,
        "inputs": copy.deepcopy(INPUTS),
        "outputs": ["c"],
        "languages": {"cpp": {"dir": "src/vadd", "files": ["vadd.cpp"]}},
    }
    if contract is not None:
        item["contract"] = copy.deepcopy(contract)
    plain = {"split": "eval", "run_args": ["10"], "languages": {"cpp": {"dir": "src/plain", "files": ["plain.cpp"]}}}
    return {
        "suite": SUITE,
        "repo": "https://example.invalid/keyed.git",
        "commit": "c" * 40,
        "items": {ITEM: item, PLAIN: plain},
    }


def write_manifest(directory: Path, data: dict[str, Any]) -> Path:
    """Write `data` as <directory>/<suite>.yaml with LF newlines and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{data['suite']}.yaml"
    path.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    return path


def load(directory: Path, data: dict[str, Any]) -> bench.Suite:
    """Write and load a manifest."""
    return bench.load_suite(write_manifest(directory, data))


def files(directory: Path, a: list[float], k: list[int]) -> list[Path]:
    """Write SYNTHETIC a (f32 [2, 4]) and k (i32 [8]) files and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    write_array(directory / "a.lassiio", "a", "f32", [2, 4], struct.pack("<8f", *a))
    write_array(directory / "k.lassiio", "k", "i32", [8], struct.pack("<8i", *k))
    return [directory / "a.lassiio", directory / "k.lassiio"]


def reason(contract: Any, paths: list[Path]) -> str | None:
    """Return the refusal reason of check_inputs, or None when it accepts."""
    h = harness()
    try:
        h.check_inputs(contract, paths)
    except h.InputContractError as error:
        return error.reason
    return None


# ---------------------------------------------------------------------------
# The item's contract


def test_without_the_key_the_contract_is_the_spec_read_as_one(tmp_path: Path) -> None:
    item = load(tmp_path / "m", manifest_data()).item(ITEM, purpose="eval")
    assert isinstance(item.contract, harness().InputContract)
    assert item.contract == harness().input_contract(item.inputs)
    assert reason(item.contract, files(tmp_path / "in", [-1.0, 0.5] * 4, [0, 9] * 4)) is None
    assert reason(item.contract, files(tmp_path / "wide", [-1.5, 0.5] * 4, [0, 9] * 4)) == "out-of-range"
    assert reason(item.contract, files(tmp_path / "hi", [1.0, 0.5] * 4, [0, 9] * 4)) == "out-of-range"
    assert reason(item.contract, files(tmp_path / "k", [0.5] * 8, [10] * 8)) == "out-of-range"


def test_the_key_gives_each_input_its_declared_range(tmp_path: Path) -> None:
    item = load(tmp_path / "m", manifest_data(CONTRACT)).item(ITEM, purpose="eval")
    assert item.contract == harness().input_contract(item.inputs, declared=CONTRACT)
    assert reason(item.contract, files(tmp_path / "wide", [-2.0, 1.5] * 4, [-100, 99] * 4)) is None
    assert reason(item.contract, files(tmp_path / "hi", [2.0, 0.5] * 4, [0] * 8)) == "out-of-range"
    assert reason(item.contract, files(tmp_path / "k", [0.5] * 8, [100] * 8)) == "out-of-range"


def test_the_key_can_let_a_nan_or_an_infinity_through(tmp_path: Path) -> None:
    declared = copy.deepcopy(CONTRACT)
    declared[0]["allow_nan"] = True
    item = load(tmp_path / "m", manifest_data(declared)).item(ITEM, purpose="eval")
    assert reason(item.contract, files(tmp_path / "nan", [math.nan, 0.5] * 4, [0] * 8)) is None
    assert reason(item.contract, files(tmp_path / "inf", [math.inf, 0.5] * 4, [0] * 8)) == "infinity"
    declared[0]["allow_nan"], declared[0]["allow_inf"] = False, True
    item = load(tmp_path / "m2", manifest_data(declared)).item(ITEM, purpose="eval")
    assert reason(item.contract, files(tmp_path / "inf2", [math.inf, -math.inf] * 4, [0] * 8)) is None
    assert reason(item.contract, files(tmp_path / "nan2", [math.nan, 0.5] * 4, [0] * 8)) == "nan"


def test_an_item_without_held_out_inputs_has_no_contract(tmp_path: Path) -> None:
    assert load(tmp_path, manifest_data(CONTRACT)).item(PLAIN, purpose="eval").contract is None


def test_the_bounds_may_reach_the_dtype_limits(tmp_path: Path) -> None:
    declared = [
        {"name": "a", "dtype": "f32", "lo": -FLT_MAX, "hi": FLT_MAX},
        {"name": "k", "dtype": "i32", "lo": -(1 << 31), "hi": 1 << 31},
    ]
    item = load(tmp_path / "m", manifest_data(declared)).item(ITEM, purpose="eval")
    assert reason(item.contract, files(tmp_path / "x", [-FLT_MAX, 1e38] * 4, [-(1 << 31), (1 << 31) - 1] * 4)) is None


def test_the_key_never_changes_the_held_out_inputs(tmp_path: Path) -> None:
    plain = load(tmp_path / "plain", manifest_data()).item(ITEM, purpose="eval")
    keyed = load(tmp_path / "keyed", manifest_data(CONTRACT)).item(ITEM, purpose="eval")
    assert [dict(entry) for entry in keyed.inputs] == [dict(entry) for entry in plain.inputs] == INPUTS
    assert (keyed.outputs, keyed.seed) == (plain.outputs, plain.seed) == (("c",), 7)
    assert registry.program_args(keyed) == registry.program_args(plain)
    written = {}
    for label, item in (("plain", plain), ("keyed", keyed)):
        workdir = tmp_path / f"work-{label}"
        workdir.mkdir()
        assert registry.stage_inputs(item, workdir) == registry.program_args(plain)
        staged = workdir / registry.INPUTS_DIR
        written[label] = {path.name: path.read_bytes() for path in sorted(staged.iterdir())}
    assert written["keyed"] == written["plain"] and sorted(written["plain"]) == ["a.lassiio", "k.lassiio"]


# ---------------------------------------------------------------------------
# A malformed key is refused, naming the item


def changed(change: Any) -> list[Any]:
    """Return a copy of CONTRACT with `change` applied to it."""
    declared = copy.deepcopy(CONTRACT)
    change(declared)
    return declared


MALFORMED = {
    "not-a-list": {"a": {"dtype": "f32", "lo": -2.0, "hi": 2.0}},
    "empty": [],
    "entry-not-a-mapping": ["a", CONTRACT[1]],
    "missing-entry": [CONTRACT[0]],
    "unknown-input": [*CONTRACT, {"name": "z", "dtype": "f32", "lo": 0.0, "hi": 1.0}],
    "repeated-entry": [CONTRACT[0], CONTRACT[0], CONTRACT[1]],
    "dtype-mismatch": changed(lambda d: d[0].update(dtype="f64")),
    "unknown-dtype": changed(lambda d: d[0].update(dtype="f8")),
    "missing-hi": changed(lambda d: d[0].pop("hi")),
    "missing-name": changed(lambda d: d[0].pop("name")),
    "unknown-key": changed(lambda d: d[0].update(extra=1)),
    "hi-past-finite-range": changed(lambda d: d[0].update(hi=3.5e38)),
    "lo-past-finite-range": changed(lambda d: d[0].update(lo=-1e39)),
    "infinite-hi": changed(lambda d: d[0].update(hi=math.inf)),
    "nan-lo": changed(lambda d: d[0].update(lo=math.nan)),
    "lo-not-below-hi": changed(lambda d: d[0].update(lo=2.0)),
    "float-bound-not-a-number": changed(lambda d: d[0].update(lo="0")),
    "bool-bound": changed(lambda d: d[0].update(lo=True)),
    "int-bound-not-an-int": changed(lambda d: d[1].update(lo=0.5)),
    "int-hi-past-dtype": changed(lambda d: d[1].update(hi=(1 << 31) + 1)),
    "int-lo-past-dtype": changed(lambda d: d[1].update(lo=-(1 << 31) - 1)),
    "int-lo-not-below-hi": changed(lambda d: d[1].update(lo=100)),
    "allow-nan-not-a-bool": changed(lambda d: d[0].update(allow_nan="yes")),
    "allow-inf-not-a-bool": changed(lambda d: d[0].update(allow_inf=1)),
    "allow-nan-on-an-int": changed(lambda d: d[1].update(allow_nan=True)),
    "allow-inf-on-an-int": changed(lambda d: d[1].update(allow_inf=True)),
    "name-not-a-str": changed(lambda d: d[0].update(name=1)),
    # a's held-out inputs are drawn from [-1, 1) and k's from [0, 10): a contract may not leave out any of their values.
    "float-narrower-than-inputs": changed(lambda d: d[0].update(lo=0.0, hi=0.5)),
    "float-lo-above-the-inputs": changed(lambda d: d[0].update(lo=-0.99999999)),
    "int-narrower-than-inputs": changed(lambda d: d[1].update(lo=1)),
    "int-hi-below-the-inputs": changed(lambda d: d[1].update(hi=9)),
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_a_malformed_key_is_refused_naming_the_item(tmp_path: Path, case: str) -> None:
    load(tmp_path / "good", manifest_data(CONTRACT))
    path = write_manifest(tmp_path / "bad", manifest_data(MALFORMED[case]))
    with pytest.raises(ValueError) as raised:
        bench.load_suite(path)
    message = str(raised.value).replace(str(path), "<manifest>")
    assert ITEM in message and "contract" in message, message


def test_a_null_key_is_refused_naming_the_item(tmp_path: Path) -> None:
    data = manifest_data()
    data["items"][ITEM]["contract"] = None
    path = write_manifest(tmp_path, data)
    with pytest.raises(ValueError) as raised:
        bench.load_suite(path)
    message = str(raised.value).replace(str(path), "<manifest>")
    assert ITEM in message and "contract" in message and "null" in message, message


def test_flags_set_false_on_an_int_are_accepted(tmp_path: Path) -> None:
    declared = changed(lambda d: d[1].update(allow_nan=False, allow_inf=False))
    item = load(tmp_path, manifest_data(declared)).item(ITEM, purpose="eval")
    assert item.contract == harness().input_contract(item.inputs, declared=CONTRACT)


def test_the_element_cap_counts_a_zero_dim_as_no_elements(tmp_path: Path) -> None:
    data = manifest_data()
    data["items"][ITEM]["inputs"][1]["shape"] = [1 << 21, 0]
    item = load(tmp_path / "zero", data).item(ITEM, purpose="eval")
    assert item.inputs[1]["shape"] == [1 << 21, 0]
    assert item.contract.inputs[1].shape == (1 << 21, 0)
    data["items"][ITEM]["inputs"][1]["shape"] = [1 << 21, 1]
    with pytest.raises(ValueError, match="element"):
        load(tmp_path / "past", data)


def test_the_key_needs_held_out_inputs(tmp_path: Path) -> None:
    load(tmp_path / "good", manifest_data(CONTRACT))
    data = manifest_data()
    data["items"][PLAIN]["contract"] = copy.deepcopy(CONTRACT)
    path = write_manifest(tmp_path / "bad", data)
    with pytest.raises(ValueError) as raised:
        bench.load_suite(path)
    message = str(raised.value).replace(str(path), "<manifest>")
    assert PLAIN in message and "contract" in message, message
