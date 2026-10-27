"""Tests for the analytic work counts of the tt-pairs-v0 family (task P17.42; plans/spikes/tt-perf.md, (C)).

Bible: Benchmark Suites (tt-pairs-v0), Evaluation Protocol (Reporting Rules),
Agent Rules 1 and 2; the owner's answer of 2026-10-26 (the spike's option (a)
of OQ-106): FLOPs are the analytic useful work of each item, from
its shapes, the same for every candidate that solves it, never a count read
from the device.

The manifest key these tests fix (lassi.bench.registry), optional per item:

    work: {flop: <int or null>, ops: {<class>: <int>}, bytes: <int>, basis: <note>}

- flop: the floating-point operations of the item's computation, 2MNK for
  a matmul and one per element for an elementwise add; null for an item
  whose work is not floating-point arithmetic (a copy) or only
  transcendental (exp), whose count goes under ops.
- ops: named operation counts (add, mac, exp), each an int of at least 0.
- bytes: the fewest bytes the device must move: each input read once and
  each output written once, in the device's data format (bf16, 2 bytes).
- basis: a non-empty note saying how the counts follow from the shapes.

load_suite reads it into SuiteItem.work (registry Work), refusing another
key, a missing key, a negative or non-integer count, and an empty basis,
naming `work`. The three manifests of the family (tt-pairs-v0, -kh, and
-ttkernel) give every item the same work, and every count follows from the
item's input shapes here. The counts are arithmetic, not measurements.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import registry

REPO = Path(__file__).resolve().parents[2]
FAMILY = ("tt-pairs-v0", "tt-pairs-v0-kh", "tt-pairs-v0-ttkernel")
# bf16 on the device: two bytes per element (the items' circular buffers are Float16_b; tests/bench/test_tt_pairs.py).
DEVICE_BYTES = 2


def manifest(name: str) -> Path:
    """Return assets/bench/<name>.yaml."""
    return REPO / "assets" / "bench" / f"{name}.yaml"


def elements(shape: list[int]) -> int:
    """Return the element count of a shape."""
    return math.prod(shape)


def expected_work(item: registry.SuiteItem) -> dict[str, Any]:
    """Return the work an item's shapes give: its flop, ops, and bytes (each output as the device writes it)."""
    shapes = [list(entry["shape"]) for entry in item.inputs]
    if item.name.startswith("matmul"):
        (m, k), (k2, n) = shapes
        assert k == k2
        out = m * n
        return {"flop": 2 * m * n * k, "ops": {"mac": m * n * k},
                "bytes": DEVICE_BYTES * (elements(shapes[0]) + elements(shapes[1]) + out)}
    size = elements(shapes[0])
    if item.name == "eltwise_binary":
        return {"flop": size, "ops": {"add": size}, "bytes": DEVICE_BYTES * (2 * size + size)}
    if item.name == "eltwise_sfpu":
        return {"flop": None, "ops": {"exp": size}, "bytes": DEVICE_BYTES * (size + size)}
    if item.name == "loopback":
        return {"flop": None, "ops": {}, "bytes": DEVICE_BYTES * (size + size)}
    raise AssertionError(f"no expected work for {item.name}")


@pytest.mark.parametrize("name", FAMILY)
def test_every_item_declares_the_work_its_shapes_give(name: str) -> None:
    suite = registry.load_suite(manifest(name))
    for item in suite.items.values():
        work = item.work
        assert isinstance(work, registry.Work), f"{name}/{item.name} declares no work"
        expected = expected_work(item)
        assert (work.flop, dict(work.ops), work.bytes) == (expected["flop"], expected["ops"], expected["bytes"])
        assert work.basis.strip() and work.basis.isascii()


def test_the_spikes_table_values() -> None:
    items = registry.load_suite(manifest("tt-pairs-v0")).items
    assert items["matmul_single_core"].work.flop == 524_288_000
    assert items["matmul_multi_core"].work.flop == 524_288_000
    assert items["matmul_multi_core"].work.bytes == 2_457_600
    assert (items["eltwise_binary"].work.flop, items["eltwise_binary"].work.bytes) == (65_536, 393_216)
    assert (items["eltwise_sfpu"].work.ops["exp"], items["eltwise_sfpu"].work.bytes) == (65_536, 262_144)
    assert (items["loopback"].work.flop, items["loopback"].work.bytes) == (None, 204_800)


def test_the_family_agrees_item_for_item() -> None:
    suites = [registry.load_suite(manifest(name)) for name in FAMILY]
    for item in suites[0].items:
        works = [suite.items[item].work for suite in suites]
        assert works[1:] == works[:-1], item


def write(tmp_path: Path, work: Any) -> Path:
    """Write a one-item SYNTHETIC tracked suite with `work` on its item."""
    item: dict[str, Any] = {"split": "unassigned",
                            "languages": {"cpp": {"dir": "assets/x", "tracked": True, "files": ["x.cpp"]}}}
    if work is not None:
        item["work"] = work
    path = tmp_path / "synthetic-work.yaml"
    path.write_text(yaml.safe_dump({"suite": "synthetic-work", "items": {"one": item}}), encoding="ascii")
    return path


GOOD = {"flop": 10, "ops": {"add": 10}, "bytes": 60, "basis": "SYNTHETIC: one add per element"}


def test_an_item_without_work_has_none(tmp_path: Path) -> None:
    assert registry.load_suite(write(tmp_path, None)).items["one"].work is None
    assert registry.load_suite(write(tmp_path, GOOD)).items["one"].work == registry.Work(
        flop=10, ops={"add": 10}, bytes=60, basis="SYNTHETIC: one add per element")


@pytest.mark.parametrize(
    "work",
    [
        {**GOOD, "extra": 1},
        {key: value for key, value in GOOD.items() if key != "bytes"},
        {**GOOD, "flop": -1},
        {**GOOD, "flop": 1.5},
        {**GOOD, "flop": True},
        {**GOOD, "bytes": "60"},
        {**GOOD, "ops": {"add": -2}},
        {**GOOD, "ops": ["add"]},
        {**GOOD, "basis": " "},
        "work",
    ],
    ids=["extra-key", "no-bytes", "negative-flop", "float-flop", "bool-flop", "text-bytes", "negative-op",
         "ops-list", "blank-basis", "not-a-mapping"],
)
def test_a_malformed_work_is_refused_naming_it(tmp_path: Path, work: Any) -> None:
    with pytest.raises(ValueError, match="work"):
        registry.load_suite(write(tmp_path, work))
