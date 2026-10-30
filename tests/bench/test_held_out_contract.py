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
  change, adv-seeded-v0's when the suite was added (task P16.8), and
  bale-v0's when that suite was added (task P17.40), each by
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
    # bale-v0 (task P17.40), recorded when the suite was added, the same way.
    ("bale-v0", "histo", "table_size"): "72c7b485c395de5e7870a64297d8522de4e3efb1a831220be3e199232ff8858d",
    ("bale-v0", "histo", "index"): "3e64996a2e86e42d63b501571531444f118814d1978e06b2dd8ae65284d23708",
    ("bale-v0", "ig", "table"): "4b37dc3b82356edb418dcdc25c6a21914391023fdaa2bb845ec9a08b50dc394a",
    ("bale-v0", "ig", "index"): "a7ebbc50c79479011fb541648fcbfa2aa6fe28148ab06ea177c02c236eafe11b",
    ("bale-v0", "triangle", "num_vertices"): "71bfa507b1eb86befe242cf42ca98b8cf247ff19e0d5052fdd2a818968114ec8",
    ("bale-v0", "triangle", "edge_u"): "b3821383ae49846e59086b56c6541093d0376b8455d75895302c4f9499a82c0d",
    ("bale-v0", "triangle", "edge_v"): "a9b4928c3e9ea18e311bc0387d9f44e332684dd16e4ab85171030745bdb63e36",
    # bale-v0-tt (task P17.43), recorded when the suite was added, the same way.
    ("bale-v0-tt", "histo", "table_size"): "c9181ad6373ed627294febc5f34af5dd4658239bc50bbcfb6906db1c1af8a8fb",
    ("bale-v0-tt", "histo", "index"): "2bd21dde9682c23aaf67c284c177f92cd99fe80b0821808d04c37754a5b8227c",
    ("bale-v0-tt", "ig", "table"): "7bf0d7c0dc89802ebe71c95e6aae354be5501bc9bd9101659b3504855205fb47",
    ("bale-v0-tt", "ig", "index"): "ba607aa695c8057f0698e6d9b09b46c9055ed040a498099c4043b2b78acf1875",
    ("bale-v0-tt", "permute_matrix", "num_rows"): "fa48dffb5bd061f221f6de2aae1a9e497ad208767f8260a93aed126f0d52ecb6",
    ("bale-v0-tt", "permute_matrix", "nz_row"): "76d422327ddc9d5779d2e3bb6cc039812384df71690a8f4b5b5f7eef436244bf",
    ("bale-v0-tt", "permute_matrix", "nz_col"): "bd454e1b9c9cf9d1fc89ecf46f8fcdbcaaf456cec4aa9a5ccdd0d672174b796b",
    ("bale-v0-tt", "permute_matrix", "row_key"): "99cf7336dd80f131eb9d7352440904b8034c0e51b8ef17c01ad9f3d4f9cc5cfb",
    ("bale-v0-tt", "permute_matrix", "col_key"): "ae2fa387158ba87c4e5b23653ac502c6a6f19f64cfa74f79f27bf803c6741874",
    ("bale-v0-tt", "randperm", "num_darts"): "2b894cd1e590b59d77f20548cd5f3eb59c1035ac34588a533b96c4f376b73489",
    ("bale-v0-tt", "randperm", "darts"): "7f205382c265d4e546a0d54f16546bde2fe004c480bec5557167a2b433d12540",
    ("bale-v0-tt", "sssp", "num_vertices"): "ce7a07443aebc516f4eb60b7929bd395f4095cfdcb9866fa48d2fa5466182d76",
    ("bale-v0-tt", "sssp", "edge_tail"): "476060ee552bfb67d9326cefa54551c02e359989668b3ceeef97f97b9141fae4",
    ("bale-v0-tt", "sssp", "edge_head"): "f0955ed739a49e0706d256a34963b0ab141a88aa6a4fd2c680116ad5b8f25631",
    ("bale-v0-tt", "sssp", "edge_weight"): "cd136d9737c551084308a4ed53d37041c71ff2493a04d01600462f3b1bec4c94",
    ("bale-v0-tt", "toposort", "num_rows"): "fa48dffb5bd061f221f6de2aae1a9e497ad208767f8260a93aed126f0d52ecb6",
    ("bale-v0-tt", "toposort", "extra_u"): "62d25ed815f7d1a35e50c299a20b8725f1f4967eb40229d33b26eb7c32573cc2",
    ("bale-v0-tt", "toposort", "extra_v"): "7a6071064b5a7c5ad162d109a249cdaf6227bfcd51da13754a7db38e0e5c316f",
    ("bale-v0-tt", "toposort", "row_key"): "107abd469870f3bac521882938ed80a9691cb3e7c01d60d8529ad55e397a74bf",
    ("bale-v0-tt", "toposort", "col_key"): "d01562e358bcf17b240d3f90879a505c5dc248afa8c549ff51450c8592c2ffde",
    ("bale-v0-tt", "transpose_matrix", "num_rows"): "f2bd54f39bfc99e97de688fbf9f5ee242026e18de050ee0cf420a3be4c11bf8d",
    ("bale-v0-tt", "transpose_matrix", "num_cols"): "6a8760131ddcee2c75ec6902d5934e76ac38cd03df263eab94550af8206a146d",
    ("bale-v0-tt", "transpose_matrix", "entry_row"): "43ef8f26a5f1ae06fe74f8fa92c416f4dd3223c6aca1261ddddc9d8dd66c9544",
    ("bale-v0-tt", "transpose_matrix", "entry_col"): "4b6f83b6d1dec4fd66b3ad310ff28c9ef588cb1d73637da87fb9ffc1b9e02003",
    ("bale-v0-tt", "triangle", "num_vertices"): "ce7a07443aebc516f4eb60b7929bd395f4095cfdcb9866fa48d2fa5466182d76",
    ("bale-v0-tt", "triangle", "edge_u"): "1721acaef35aea61c6bc61a9c9dd262bb9d758f0236a370ded39c18125efd37a",
    ("bale-v0-tt", "triangle", "edge_v"): "cea30188682358b2f0bdd965c7162bad2a60950651d985b03e2068ab13e0f03e",
}
# tt-pairs-v0-ttkernel (task P17.36), tt-pairs-v0-kh (task P17.39), and tt-pairs-v0-ttir and tt-pairs-v0-d2m (task
# P17.48) hold tt-pairs-v0's items with the same inputs and seeds, so their files are tt-pairs-v0's, byte for byte
# (tests/bench/test_ttkernel_suite.py, tests/bench/test_kh_suite.py, and tests/bench/test_ttir_d2m_suites.py check
# that the entries are equal).
DIGESTS.update({
    (sibling, item, name): digest
    for sibling in ("tt-pairs-v0-ttkernel", "tt-pairs-v0-kh", "tt-pairs-v0-ttir", "tt-pairs-v0-d2m")
    for (suite, item, name), digest in list(DIGESTS.items())
    if suite == "tt-pairs-v0"
})
RECORDED_SUITES = (
    "adv-seeded-v0", "bale-v0", "bale-v0-tt", "lassi-hecbench-10", "tt-pairs-v0", "tt-pairs-v0-ttkernel",
    "tt-pairs-v0-kh", "tt-pairs-v0-ttir", "tt-pairs-v0-d2m",
)


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
