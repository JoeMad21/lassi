"""Tests that every manifest's held-out inputs are unchanged and pass their item's contract (task P16.1).

Bible: Benchmark Suites (held-out inputs), Harness Contract (held-out input
files: a seed gives the same bytes on every host), Adversarial Setups (the
adversary rule; confirmed counterexamples never join the held-out inputs of
an eval or unassigned item), Agent Rule 5. Plan: plans/p16-adversarial.md,
P16.1 ("generate_inputs' bytes for every item of every manifest are
unchanged"; "every tt-pairs-v0 item's held-out inputs pass its contract").

- DIGESTS holds the sha256 of every file generate_inputs writes for every
  item, of every manifest under assets/bench/, that declares held-out
  inputs. tt-pairs-v0's were recorded at commit 9793777, before any P16.1
  change, and adv-seeded-v0's when the suite was added (task P16.8), each by
  loading each manifest with load_suite and running
  generate_inputs(item.inputs, item.seed, <a temporary directory>). They are
  known-answer values of the generator, not measurements. lassi-hecbench-10
  declares no held-out inputs, so its record is empty. A suite added later
  with held-out inputs fails test_every_suite_with_held_out_inputs_is_recorded
  until its digests are recorded here.
- Every item's generated held-out inputs, in every manifest, pass
  check_inputs(item.contract, ...) (tests/harness/test_input_contract.py),
  every tt-pairs-v0 item among them.

Items are read through Suite.item with purpose eval only; nothing here
trains, tunes a prompt, or harvests, and the held-out inputs are written
only to a temporary directory and never changed (Agent Rule 5).
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import ModuleType

import pytest

from lassi import bench
from lassi.harness.lassi_io import generate_inputs

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = REPO / "assets" / "bench"
TT_PAIRS = ("eltwise_binary", "eltwise_sfpu", "loopback", "matmul_multi_core", "matmul_single_core")
# (suite, item, input) -> sha256 of <input>.lassiio: tt-pairs-v0's recorded at commit 9793777, adv-seeded-v0's
# when the suite was added (task P16.8); see the module docstring.
DIGESTS = {
    ("tt-pairs-v0", "eltwise_binary", "a"): "d6cf97a41f008e43657ce6fd14e69540db4f3b599924d81b5210036f76dcc939",
    ("tt-pairs-v0", "eltwise_binary", "b"): "78ddaed19b8dc345a5ecf097c38151ee86f482610cfd53b5a9bc7fe00c987b33",
    ("tt-pairs-v0", "eltwise_sfpu", "src0"): "0518e55edbd31fa87790e6b861a604ff3dc710aa631712553160121b7534f4eb",
    ("tt-pairs-v0", "loopback", "input"): "17920f77b799fc5d41295ca74399512e0bee1bfae8ee54882c4dd3d76fdc8327",
    ("tt-pairs-v0", "matmul_multi_core", "a"): "cb7ec4f9be4fd79c9dc15744c6cdb9c28f6e1853ad67908e9a320bd03daed9fb",
    ("tt-pairs-v0", "matmul_multi_core", "b"): "33c0d78cca66b4376e416d6f0f252e5efe12d0167991a4c1e5b45de422b1235b",
    ("tt-pairs-v0", "matmul_single_core", "a"): "d932532bcdace3cc9ce9f4d051b5a162eed28b55cc212d571e23c07575f6c614",
    ("tt-pairs-v0", "matmul_single_core", "b"): "2c59e61289c1a409ece2274b7666804e20597a71f0fe69249d6134c40fb99565",
    # adv-seeded-v0 (task P16.8), recorded when the suite was added, the same way.
    ("adv-seeded-v0", "pair_max", "x"): "72e8deb6d40647dad6e456bc8b303ef6506783cb45d1944843e97b84fbef480e",
    ("adv-seeded-v0", "piecewise_gate", "x"): "d4d862d51eeb3b3d4987f8fedb582c6f6e8279cff1923628893a5044f29a71ec",
    ("adv-seeded-v0", "piecewise_ops", "x"): "0fe99f9cf390ac65587cf957f89e30d3764a09f687052b73c33a08f23db73d69",
    ("adv-seeded-v0", "prefix_sum", "x"): "5ec454cf1992433075c7253e7141f77d9212eb246b7d83bb56b0e97f1ea7416f",
    ("adv-seeded-v0", "scale_i64", "x"): "1789264892c2376a62bf82c58174ad7d5e46de9747f9ebb3c7d4e885882dc463",
    ("adv-seeded-v0", "signed_root", "x"): "efc14187fdfe04ce8532f297e17eae3670b84c30e1660493471e5bcd96af5bec",
}
# tt-pairs-v0-ttkernel (task P17.36) and tt-pairs-v0-kh (task P17.39) hold tt-pairs-v0's items with the same inputs
# and seeds, so their files are tt-pairs-v0's, byte for byte (tests/bench/test_ttkernel_suite.py and
# tests/bench/test_kh_suite.py check that the entries are equal).
DIGESTS.update({
    (sibling, item, name): digest
    for sibling in ("tt-pairs-v0-ttkernel", "tt-pairs-v0-kh")
    for (suite, item, name), digest in list(DIGESTS.items())
    if suite == "tt-pairs-v0"
})
RECORDED_SUITES = ("adv-seeded-v0", "lassi-hecbench-10", "tt-pairs-v0", "tt-pairs-v0-ttkernel", "tt-pairs-v0-kh")


def harness() -> ModuleType:
    """Import and return lassi.harness, where check_inputs lives."""
    return importlib.import_module("lassi.harness")


@pytest.fixture(scope="module")
def held_out(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, str], dict[str, Path]]:
    """Generate every manifest item's held-out inputs once: (suite, item) -> input name -> file."""
    generated: dict[tuple[str, str], dict[str, Path]] = {}
    for manifest in sorted(MANIFESTS.glob("*.yaml")):
        suite = bench.load_suite(manifest)
        for name in sorted(suite.items):
            item = suite.item(name, purpose="eval")
            if item.inputs:
                out = tmp_path_factory.mktemp(f"{suite.name}-{name}")
                generated[(suite.name, name)] = generate_inputs(item.inputs, item.seed, out)
    return generated


def test_every_suite_with_held_out_inputs_is_recorded(held_out: dict[tuple[str, str], dict[str, Path]]) -> None:
    unrecorded = sorted({suite for suite, _ in held_out} - set(RECORDED_SUITES))
    assert unrecorded == [], f"record the held-out input digests of {unrecorded} in DIGESTS"
    assert set(RECORDED_SUITES) <= {bench.load_suite(path).name for path in MANIFESTS.glob("*.yaml")}


def test_the_recorded_suites_still_declare_exactly_the_recorded_inputs(
    held_out: dict[tuple[str, str], dict[str, Path]],
) -> None:
    found = {
        (suite, item, name)
        for (suite, item), paths in held_out.items()
        if suite in RECORDED_SUITES
        for name in paths
    }
    assert found == set(DIGESTS)


def test_generate_inputs_gives_the_recorded_bytes_for_every_item(
    held_out: dict[tuple[str, str], dict[str, Path]],
) -> None:
    changed = {
        key: digest
        for key, expected in DIGESTS.items()
        if (digest := hashlib.sha256(held_out[key[:2]][key[2]].read_bytes()).hexdigest()) != expected
    }
    assert changed == {}, f"held-out input bytes changed: {changed}"


def test_every_items_held_out_inputs_pass_its_contract(held_out: dict[tuple[str, str], dict[str, Path]]) -> None:
    assert {("tt-pairs-v0", item) for item in TT_PAIRS} <= set(held_out), "every tt-pairs-v0 item is checked"
    h = harness()
    refused = {}
    for (suite, name), paths in sorted(held_out.items()):
        item = bench.load_suite(MANIFESTS / f"{suite}.yaml").item(name, purpose="eval")
        try:
            h.check_inputs(item.contract, list(paths.values()))
        except h.InputContractError as error:
            refused[(suite, name)] = f"{error.reason}: {error}"
    assert refused == {}, f"items whose own held-out inputs are refused: {refused}"
