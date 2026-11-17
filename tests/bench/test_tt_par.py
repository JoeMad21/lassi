"""Tests for the development items of the multi-core suite tt-par-v0 (task P17.67; OQ-129).

Bible: Benchmark Suites (a new suite and its splits), Harness Contract
(lassi_io, the 2**20-element input cap, the CPU -> TT guard), Oracles
(binary_io), ttsim Facts (unpack_to_dest), Agent Rules 3 and 5. Plan:
plans/p17-portable.md, "The multi-core C++ -> TT test" and P17.67;
plans/spikes/tt-par-upstream.md for each example's kernels, their sha256,
its own check, and the sizes the owner chose (its Recommendation; OQ-129,
"All five, resized"). They check tt-par-v0 as tests/bench/test_tt_pairs.py
and tests/bench/test_tt_pairs_work.py check tt-pairs-v0. What they fix:

- assets/bench/tt-par-v0.yaml loads, is plain ASCII with LF line ends,
  names tasks P17.67 and P17.68 and OQ-129 in its header, and pins tt-metal
  as tt-pairs-v0.yaml does (same repo and commit, `pin: tt-metal`).
- Its dev items are exactly the five development items the owner chose,
  each given for evaluation and prompt tuning and refused to training and
  corpus harvest (Agent Rule 5). The held-out items of task P17.68 join the
  same manifest with split eval; these tests select the dev items by split
  and constrain only them, so an eval item does not fail them.
- Each dev item has a seed of its own, distinct from every other item's in
  the manifest and from every tt-pairs-v0 seed; inputs at the owner's
  sizes, each within MAX_INPUT_ELEMENTS; an output that spans more tiles
  (1024-element units for pad_multi_core's row-major output) than an 8x9
  grid has cores (72, the largest grid the spike names); the tolerance of
  the upstream example's own check (exact equality for pad_multi_core,
  which has none); and analytic work consistent with its shapes.
- Each dev item has languages cpp and tt, both tracked under
  assets/bench/tt-par-v0/<item>/; the tt support holds lassi_io.h and the
  upstream kernels, unmodified, each named by an `upstream:` path under
  tt_metal/programming_examples/ whose tail is its support name, with the
  sha256 the spike lists. No tracked file under a dev item's directory is
  an upstream file: none bears a listed kernel's name or bytes. (A check
  for runs of three consecutive non-blank lines shared with the upstream
  host programs, as tt-train-v0's rule 3 counts them, needs those
  programs' text, which is not tracked, so it is not made here.)
- Every tracked program opens with its provenance line; the C++ program
  includes nothing from tt-metal; each TT program names each of its item's
  support kernels as a string literal and no other kernel path, reads
  host_compute False under the CPU -> TT guard, and keeps its
  unpack_to_dest note's claims (no fp32_dest_acc_en, no
  unpack_to_dest_mode, Float16_b the only DataFormat).
- tt-train-v0's include list names none of the five upstream example
  directories nor the directories their kernels come from.

Every count here is arithmetic over the items' shapes or a digest of a
file at the pin as the spike recorded it; none is a measurement. Until the
manifest exists, every test fails naming it.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

from lassi.bench import registry
from lassi.core.tolerance import Tolerance
from lassi.harness.contract import MAX_INPUT_ELEMENTS
from lassi.toolchains import pins
from lassi.toolchains._cxx_scan import IDENT, LITERAL, lex
from lassi.toolchains.ttmetal_build import TtMetalHost

REPO = Path(__file__).resolve().parents[2]
SUITE = "tt-par-v0"
MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
SIBLING = REPO / "assets" / "bench" / "tt-pairs-v0.yaml"
PROGRAMS = REPO / "assets" / "bench" / SUITE
TRAIN_SPEC = REPO / "assets" / "train" / "tt-train-v0" / "dataset.yaml"
SPIKE = REPO / "plans" / "spikes" / "tt-par-upstream.md"
HARNESS = "assets/harness/c/lassi_io.h"
HEADER_NAMES = ("P17.67", "P17.68", "OQ-129")
# The five development items the owner chose on 2026-11-17 (OQ-129, "All five, resized").
DEV_ITEMS = (
    "matmul_multicore_reuse",
    "matmul_multicore_reuse_mcast",
    "pad_multi_core",
    "vecadd_multi_core",
    "vecadd_sharding",
)
LANGUAGES = ("cpp", "tt")
FORBIDDEN = ("train", "harvest")
GIVEN = ("eval", "prompt-tuning")
PLACEHOLDER = "PLACEHOLDER"
# Each dev item's seed, chosen for task P17.67 (11 to 15, one per item), clear of tt-pairs-v0's 1 to 5.
SEEDS = {
    "vecadd_multi_core": 11,
    "vecadd_sharding": 12,
    "pad_multi_core": 13,
    "matmul_multicore_reuse": 14,
    "matmul_multicore_reuse_mcast": 15,
}
# Each dev item's inputs, in order, as (dtype, shape, dist, lo, hi): the spike's Recommendation, which the owner chose
# (OQ-129 (a)). vecadd_multi_core at upstream's 640 tiles; vecadd_sharding raised to 448 tiles; pad_multi_core's
# source [512, 128]; both matmuls at M 1568, K 256, N 1024; the ranges are upstream's draws, except pad's [-1, 1).
VEC_640 = ("bf16", [640 * 1024], "uniform", 0.0, 10.0)
VEC_448 = ("bf16", [448 * 1024], "uniform", 0.0, 10.0)
MATMUL = [("bf16", [1568, 256], "uniform", -0.4, 0.6), ("bf16", [256, 1024], "uniform", -0.3, 0.7)]
INPUTS = {
    "vecadd_multi_core": [VEC_640, VEC_640],
    "vecadd_sharding": [VEC_448, VEC_448],
    "pad_multi_core": [("bf16", [512, 128], "uniform", -1.0, 1.0)],
    "matmul_multicore_reuse": MATMUL,
    "matmul_multicore_reuse_mcast": MATMUL,
}
# pad_multi_core pads each row on the right to this many columns ([512, 128] -> [512, 256]; OQ-129).
PADDED_COLUMNS = 256
# A tile, and pad_multi_core's unit of its row-major output: 1024 elements.
TILE = 32 * 32
# The largest compute grid the spike names: 8 x 9 (ttsim's, inferred from the descriptors), over card 0's 8 x 7.
GRID_CORES = 8 * 9
# The output tiles (1024-element units for pad_multi_core) the owner's sizes give (the spike's Recommendation).
OUTPUT_TILES = {
    "vecadd_multi_core": 640,
    "vecadd_sharding": 448,
    "pad_multi_core": 128,
    "matmul_multicore_reuse": 1568,
    "matmul_multicore_reuse_mcast": 1568,
}
# Each item's tolerance: its upstream example's own check (the spike, section 3 of each example); pad_multi_core has
# none, so exact equality, max_abs 0, chosen in task P17.67. binary_io passes a PCC at or above its threshold.
TOLERANCES = {
    "vecadd_multi_core": Tolerance(metric="max_abs", threshold=0.3),
    "vecadd_sharding": Tolerance(metric="max_abs", threshold=0.2),
    "pad_multi_core": Tolerance(metric="max_abs", threshold=0),
    "matmul_multicore_reuse": Tolerance(metric="pcc", threshold=0.99),
    "matmul_multicore_reuse_mcast": Tolerance(metric="pcc", threshold=0.98),
}
# bf16 on the device: two bytes per element.
DEVICE_BYTES = 2
# The spike's analytic work at these sizes (its Recommendation; arithmetic, never a measurement).
SPIKE_WORK = {
    "vecadd_multi_core": (655_360, 3_932_160),
    "vecadd_sharding": (458_752, 2_752_512),
    "pad_multi_core": (None, 393_216),
    "matmul_multicore_reuse": (822_083_584, 4_538_368),
    "matmul_multicore_reuse_mcast": (822_083_584, 4_538_368),
}
EXAMPLES_DIR = "tt_metal/programming_examples"
# The upstream example directory of each dev item, under tt_metal/programming_examples (the spike's Question).
EXAMPLE_DIRS = {
    "vecadd_multi_core": "vecadd_multi_core",
    "vecadd_sharding": "vecadd_sharding",
    "pad_multi_core": "pad_multi_core",
    "matmul_multicore_reuse": "matmul/matmul_multicore_reuse",
    "matmul_multicore_reuse_mcast": "matmul/matmul_multicore_reuse_mcast",
}
# The kernels each example launches, relative to tt_metal/programming_examples, with the sha256 of each file's bytes
# at the pin, each file's git blob id and size checked against the pinned tree: plans/spikes/tt-par-upstream.md, the
# kernel table of each example (the scratch hash list it was read from is not tracked). Paths and digests only; no
# tt-metal text.
MATMUL_COMMON = "matmul/matmul_common/kernels"
KERNEL_SHA256 = {
    "vecadd_multi_core/kernels/interleaved_tile_read_multi_core.cpp":
        "c044be02a23eaf2447e77ff6b37d0023355b6a2a4bed5a2686f24dd64780970f",
    "vecadd_multi_core/kernels/tile_write_multi_core.cpp":
        "fd273d6022c361740530a2bfbc34664a39676ccd0c04306f484b266fb5a3fc4b",
    "vecadd_multi_core/kernels/add_multi_core.cpp":
        "0d0c8fb1301dc5e0cc06349d43c9bf0ef1496166b79a6b13e9db99c23599260d",
    "vecadd_sharding/kernels/add_sharding.cpp":
        "9b89bd03d6e8dedec93c8a778f9aa204f56fa3f8d639f350f32fd41ec316b2fd",
    "pad_multi_core/kernels/pad_reader_dims_rm_interleaved.cpp":
        "31ad8a0d23a9dca3eebc6da3d4b8c740c23a02e27e4ed711262318773e17482b",
    "pad_multi_core/kernels/pad_writer_dims_rm_interleaved.cpp":
        "ee711f0765c06c770261681e5cd6b6cacb9cd99023945ad4c366113a5af4903b",
    f"{MATMUL_COMMON}/dataflow/reader_bmm_tile_layout.cpp":
        "8bc4f61e7780d664ed7825757fdfdc440f05fe50c8060cdfb809803b681267ef",
    f"{MATMUL_COMMON}/dataflow/writer_bmm_tile_layout.cpp":
        "e17aec12dd837bba1d91e66ec455611df745ff07c897f6172acd821329c1b600",
    f"{MATMUL_COMMON}/compute/bmm_large_block_zm.cpp":
        "8c59140dbaf49e584af6fd1ea443d00be684f3a708bfb0fd8a7ff735539aec30",
    f"{MATMUL_COMMON}/dataflow/reader_bmm_tile_layout_in0_sender_in1_sender.cpp":
        "31dd377003eb2bc302bd508a9603591e28838f8d6d0a0e4fc28581c0ba50cebd",
    f"{MATMUL_COMMON}/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp":
        "c6d1e619e3c792806183c50b2c50394889e1574bf8c260e52806325ee310cd07",
    f"{MATMUL_COMMON}/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp":
        "793023368a44508b537c7c119ccc00928fdb2e8ff8ec30249c31b4a01d126988",
    f"{MATMUL_COMMON}/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp":
        "c893a77b3566f5af187c55f9afffe7c5c9ecce442ad51c7a5470053859df4ca0",
}
KERNELS = {
    "vecadd_multi_core": (
        "vecadd_multi_core/kernels/interleaved_tile_read_multi_core.cpp",
        "vecadd_multi_core/kernels/tile_write_multi_core.cpp",
        "vecadd_multi_core/kernels/add_multi_core.cpp",
    ),
    "vecadd_sharding": ("vecadd_sharding/kernels/add_sharding.cpp",),
    "pad_multi_core": (
        "pad_multi_core/kernels/pad_reader_dims_rm_interleaved.cpp",
        "pad_multi_core/kernels/pad_writer_dims_rm_interleaved.cpp",
    ),
    "matmul_multicore_reuse": (
        f"{MATMUL_COMMON}/dataflow/reader_bmm_tile_layout.cpp",
        f"{MATMUL_COMMON}/dataflow/writer_bmm_tile_layout.cpp",
        f"{MATMUL_COMMON}/compute/bmm_large_block_zm.cpp",
    ),
    "matmul_multicore_reuse_mcast": (
        f"{MATMUL_COMMON}/dataflow/reader_bmm_tile_layout_in0_sender_in1_sender.cpp",
        f"{MATMUL_COMMON}/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp",
        f"{MATMUL_COMMON}/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp",
        f"{MATMUL_COMMON}/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp",
        f"{MATMUL_COMMON}/dataflow/writer_bmm_tile_layout.cpp",
        f"{MATMUL_COMMON}/compute/bmm_large_block_zm.cpp",
    ),
}
# The guard's tag for a program that creates kernels but no compute kernel (lassi.toolchains.ttmetal_guard).
TAG = "guard-data-movement-only"
DATA_MOVEMENT_ONLY = frozenset({"pad_multi_core"})
FIRST_LINE_START = "// Written for LASSI-DF tt-par-v0 (task P17.67)"
TT_FIRST_LINE = FIRST_LINE_START + "; runs the unmodified upstream {item} kernels from tt-metal 5280a9cf."
SOURCE_SUFFIXES = (".cpp", ".cc", ".c", ".hpp", ".h")


def raw() -> dict[str, Any]:
    """Return the manifest as YAML reads it; fail naming it when it does not exist."""
    if not MANIFEST.is_file():
        pytest.fail(f"{MANIFEST.relative_to(REPO).as_posix()} does not exist (task P17.67)")
    return yaml.safe_load(MANIFEST.read_bytes().decode("utf-8"))


def suite() -> registry.Suite:
    """Load the manifest with load_suite; fail naming it when it does not exist."""
    if not MANIFEST.is_file():
        pytest.fail(f"{MANIFEST.relative_to(REPO).as_posix()} does not exist (task P17.67)")
    return registry.load_suite(MANIFEST)


def dev_items(data: dict[str, Any]) -> list[str]:
    """Return the names of the manifest's items in split dev, sorted."""
    return sorted(name for name, entry in data["items"].items() if entry.get("split") == "dev")


def placeholders(value: Any, where: str = "") -> list[str]:
    """Return the key paths under `value` whose value is a string starting with PLACEHOLDER."""
    if isinstance(value, str):
        return [where] if value.startswith(PLACEHOLDER) else []
    if isinstance(value, dict):
        return [found for key, item in value.items() for found in placeholders(item, f"{where}/{key}")]
    if isinstance(value, list):
        return [found for index, item in enumerate(value) for found in placeholders(item, f"{where}/{index}")]
    return []


def program(item: str, language: str) -> Path:
    """Return the path of the item's program in `language`, as the loaded manifest names it."""
    spec = suite().items[item].languages[language]
    return REPO / spec.dir / spec.files[0]


def program_text(item: str, language: str) -> str:
    """Return the text of the item's program in `language`; fail when it does not exist."""
    path = program(item, language)
    if not path.is_file():
        pytest.fail(f"{item}: the manifest names {path} but it does not exist")
    return path.read_bytes().decode("utf-8")


def kernel_literals(text: str, name: str) -> set[str]:
    """Return the string literals of a program's code (comments aside) that read as a kernel or source path."""
    found = set()
    for token in lex(text, name).tokens:
        if token.kind != LITERAL or not token.text.startswith('"'):
            continue
        value = token.text[1:-1]
        if not value or any(char.isspace() for char in value):
            continue
        if value.endswith(SOURCE_SUFFIXES) or "kernels" in PurePosixPath(value).parts[:-1]:
            found.add(value)
    return found


def output_elements(item: str, inputs: tuple[Any, ...]) -> int:
    """Return the element count of the item's one output, from its input shapes and its computation."""
    shapes = [list(entry["shape"]) for entry in inputs]
    if item.startswith("matmul"):
        (m, _), (_, n) = shapes
        return m * n
    if item == "pad_multi_core":
        [(rows, _)] = shapes
        return rows * PADDED_COLUMNS
    return math.prod(shapes[0])


def expected_work(item: str, inputs: tuple[Any, ...]) -> tuple[int | None, dict[str, int], int]:
    """Return (flop, ops, bytes) the item's shapes give: each input read and the output written once, in bf16."""
    shapes = [list(entry["shape"]) for entry in inputs]
    read = sum(math.prod(shape) for shape in shapes)
    written = output_elements(item, inputs)
    if item.startswith("matmul"):
        (m, k), (k2, n) = shapes
        assert k == k2, f"{item}: inner dims {k} and {k2}"
        return 2 * m * n * k, {"mac": m * n * k}, DEVICE_BYTES * (read + written)
    if item == "pad_multi_core":
        return None, {}, DEVICE_BYTES * (read + written)
    size = math.prod(shapes[0])
    return size, {"add": size}, DEVICE_BYTES * (read + written)


# ---------------------------------------------------------------------------
# The manifest


def test_the_manifest_is_plain_ascii_with_lf_and_its_header_names_the_tasks() -> None:
    """The manifest is plain ASCII with LF line ends, and its header comment names P17.67, P17.68, and OQ-129."""
    raw()
    data = MANIFEST.read_bytes()
    assert data.isascii() and b"\r" not in data, "plain ASCII with LF line ends"
    text = data.decode("ascii")
    header = text.split("\nsuite:", 1)[0]
    assert header != text, "the manifest has a top-level suite key"
    assert all(line.startswith("#") for line in header.splitlines() if line.strip()), "the header is comments"
    for name in HEADER_NAMES:
        assert name in header, f"the header names {name}"


def test_the_manifest_pins_tt_metal_as_tt_pairs_v0_does() -> None:
    """Suite tt-par-v0 has tt-pairs-v0's repo and commit, the tt-metal pin's URL and COMMIT, and pin tt-metal."""
    data, sibling, pin = raw(), registry.load_suite(SIBLING), pins.read_pin("tt-metal")
    assert data["suite"] == SUITE
    assert (data["repo"], data["commit"]) == (sibling.repo, sibling.commit)
    assert (data["repo"], data["commit"], data["pin"]) == (pin["URL"], pin["COMMIT"], "tt-metal")
    loaded = suite()
    assert (loaded.name, loaded.repo, loaded.commit, loaded.pin) == (SUITE, sibling.repo, sibling.commit, "tt-metal")


def test_the_manifest_holds_no_placeholder_and_loads() -> None:
    """The manifest is complete: no PLACEHOLDER value, and load_suite loads it."""
    assert placeholders(raw()) == [], f"the manifest is complete; load_suite refuses a {PLACEHOLDER}"
    assert set(DEV_ITEMS) <= set(suite().items)


def test_the_dev_items_are_exactly_the_five_the_owner_chose() -> None:
    """The items in split dev are the five development items (OQ-129); P17.68's held-out items are eval."""
    assert dev_items(raw()) == list(DEV_ITEMS)
    loaded = suite()
    assert sorted(name for name, item in loaded.items.items() if item.split == "dev") == list(DEV_ITEMS)


@pytest.mark.parametrize("item", DEV_ITEMS)
def test_a_dev_item_is_given_for_prompt_tuning_and_refused_to_training_and_harvest(item: str) -> None:
    """A dev item serves evaluation and prompt tuning and is refused to training and corpus harvest (Agent Rule 5)."""
    loaded = suite()
    for purpose in GIVEN:
        assert loaded.item(item, purpose=purpose).split == "dev"
    for purpose in FORBIDDEN:
        with pytest.raises(registry.EvalSplitError, match="dev split"):
            loaded.item(item, purpose=purpose)


def test_each_dev_seed_is_its_own_and_clear_of_tt_pairs_v0() -> None:
    """Each dev item has its chosen seed, no other item of the manifest shares it, and no tt-pairs-v0 item has it."""
    items = suite().items
    assert {name: items[name].seed for name in DEV_ITEMS} == SEEDS
    seeds = [item.seed for item in items.values() if item.seed is not None]
    assert len(seeds) == len(set(seeds)), f"seeds repeat within {SUITE}: {sorted(seeds)}"
    sibling = {item.seed for item in registry.load_suite(SIBLING).items.values()}
    assert not set(SEEDS.values()) & sibling, f"a dev seed is a tt-pairs-v0 seed: {sorted(sibling)}"


@pytest.mark.parametrize("item", DEV_ITEMS)
def test_a_dev_items_inputs_are_the_owners_sizes_within_the_cap(item: str) -> None:
    """Each input has the owner's dtype, shape, and range, and at most MAX_INPUT_ELEMENTS elements; one output."""
    found = suite().items[item]
    assert found.run_args == ()
    declared = [(e["dtype"], list(e["shape"]), e["dist"], e["lo"], e["hi"]) for e in found.inputs]
    assert declared == [tuple(entry) for entry in INPUTS[item]], f"{item}: inputs {declared}"
    names = [entry["name"] for entry in found.inputs]
    assert len(names) == len(set(names)), f"{item}: input names repeat"
    for entry in found.inputs:
        count = math.prod(entry["shape"])
        assert 0 < count <= MAX_INPUT_ELEMENTS, f"{item}: {entry['name']} holds {count} elements"
    assert len(found.outputs) == 1, f"{item}: one output, got {found.outputs}"


@pytest.mark.parametrize("item", DEV_ITEMS)
def test_a_dev_items_output_spans_more_tiles_than_an_8x9_grid_has_cores(item: str) -> None:
    """The output spans more tiles (1024-element units for pad) than 72 cores, at the owner's sizes."""
    found = suite().items[item]
    elements = output_elements(item, found.inputs)
    assert elements % TILE == 0, f"{item}: {elements} output elements are not whole tiles"
    assert elements // TILE == OUTPUT_TILES[item]
    assert elements // TILE > GRID_CORES


@pytest.mark.parametrize("item", DEV_ITEMS)
def test_a_dev_items_tolerance_is_its_examples_own_check(item: str) -> None:
    """The tolerance is the upstream example's own check, and exact equality for pad_multi_core."""
    assert suite().items[item].tolerance == TOLERANCES[item]


@pytest.mark.parametrize("item", DEV_ITEMS)
def test_a_dev_item_declares_the_work_its_shapes_give(item: str) -> None:
    """work is present, its flop, ops, and bytes follow from the shapes, and they equal the spike's arithmetic."""
    found = suite().items[item]
    work = found.work
    assert isinstance(work, registry.Work), f"{item} declares no work"
    flop, ops, nbytes = expected_work(item, found.inputs)
    assert (work.flop, dict(work.ops), work.bytes) == (flop, ops, nbytes)
    assert (work.flop, work.bytes) == SPIKE_WORK[item]
    assert work.basis.strip() and work.basis.isascii()


# ---------------------------------------------------------------------------
# Languages, support files, and the tracked programs


@pytest.mark.parametrize("item", DEV_ITEMS)
def test_a_dev_items_languages_are_tracked_cpp_and_tt(item: str) -> None:
    """cpp and tt, each tracked at assets/bench/tt-par-v0/<item>/<language>/<item>.cpp; cpp gets lassi_io.h alone."""
    found = suite().items[item]
    assert sorted(found.languages) == list(LANGUAGES)
    for language in LANGUAGES:
        spec = found.languages[language]
        assert spec.tracked and spec.dir == f"assets/bench/{SUITE}/{item}/{language}"
        assert spec.files == (f"{item}.cpp",)
        assert spec.reference is None
    assert found.support == {}, "every support file belongs to one language"
    cpp, tt = found.languages["cpp"].support, found.languages["tt"].support
    assert set(cpp) == {"lassi_io.h"} and (cpp["lassi_io.h"].tracked, cpp["lassi_io.h"].path) == (True, HARNESS)
    assert (tt["lassi_io.h"].tracked, tt["lassi_io.h"].path) == (True, HARNESS)


@pytest.mark.parametrize("item", DEV_ITEMS)
def test_a_tt_support_kernel_is_upstream_by_path_and_the_spikes_sha256(item: str) -> None:
    """Each tt kernel is an upstream path under tt_metal/programming_examples, its name its tail, with its sha256."""
    tt = suite().items[item].languages["tt"].support
    kernels = {name: entry for name, entry in tt.items() if name != "lassi_io.h"}
    assert sorted(kernels) == sorted(KERNELS[item]), f"{item}: kernels {sorted(kernels)}"
    spike = SPIKE.read_text(encoding="utf-8").splitlines()
    for name, entry in kernels.items():
        assert not entry.tracked, f"{name}: no tt-metal text is tracked"
        assert entry.path == f"{EXAMPLES_DIR}/{name}", f"{name}: upstream {entry.path}"
        assert entry.path.endswith(f"/{name}")
        assert "kernels" in PurePosixPath(name).parts[:-1], f"{name}: ttmetal-host places it, never compiles it"
        assert entry.sha256 == KERNEL_SHA256[name], f"{name}: sha256 {entry.sha256}"
        basename = PurePosixPath(name).name
        assert any(entry.sha256 in line and basename in line for line in spike), (
            f"{name}: plans/spikes/tt-par-upstream.md lists {entry.sha256} beside {basename}")


@pytest.mark.parametrize("item", DEV_ITEMS)
def test_no_tracked_file_of_a_dev_item_is_an_upstream_file(item: str) -> None:
    """The item's directory holds only its two programs, and none bears a listed kernel's name or bytes."""
    found = suite().items[item]
    root = PROGRAMS / item
    tracked = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    expected = sorted(f"{language}/{found.languages[language].files[0]}" for language in LANGUAGES)
    assert tracked == expected, f"{item}: files {tracked}"
    names = {PurePosixPath(path).name for path in KERNEL_SHA256}
    digests = set(KERNEL_SHA256.values())
    for relative in tracked:
        assert PurePosixPath(relative).name not in names, f"{item}/{relative} has an upstream kernel's name"
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert digest not in digests, f"{item}/{relative} is an upstream kernel's bytes"


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("item", DEV_ITEMS)
def test_a_program_opens_with_its_provenance_line_and_is_plain_ascii(item: str, language: str) -> None:
    """Every tracked program is plain ASCII with LF line ends and opens with the P17.67 provenance line."""
    text = program_text(item, language)
    assert text.isascii() and "\r" not in text, f"{item} ({language}): plain ASCII with LF line ends"
    first = text.split("\n", 1)[0]
    assert first.startswith(FIRST_LINE_START), f"{item} ({language}): {first}"
    if language == "tt":
        assert first == TT_FIRST_LINE.format(item=item)


@pytest.mark.parametrize("item", DEV_ITEMS)
def test_a_cpp_program_is_host_only(item: str) -> None:
    """The C++ program includes nothing from tt-metal and names no kernel path."""
    text = program_text(item, "cpp")
    includes = [name for name, _ in lex(text, f"{item}.cpp").includes]
    assert "lassi_io.h" in includes
    assert not [name for name in includes if name.startswith(("tt-metalium/", "tt_metal/", "tt-"))], includes
    assert kernel_literals(text, f"{item}.cpp") == set(), f"{item}: the C++ program names a kernel"


@pytest.mark.parametrize("item", DEV_ITEMS)
def test_a_tt_program_names_exactly_its_support_kernels(item: str) -> None:
    """The TT program names each support kernel as a string literal, and no other kernel or source path."""
    tt = suite().items[item].languages["tt"].support
    kernels = {name for name in tt if name != "lassi_io.h"}
    named = kernel_literals(program_text(item, "tt"), f"{item}.cpp")
    assert named == kernels, f"{item}: names {sorted(named)}, support {sorted(kernels)}"


@pytest.mark.parametrize("item", DEV_ITEMS)
def test_a_tt_program_passes_the_cpu_to_tt_guard(item: str) -> None:
    """The CPU -> TT guard reads host_compute False; pad_multi_core, with no compute kernel, carries the tag."""
    text = program_text(item, "tt")
    harness = {"lassi_io.h": (REPO / HARNESS).read_bytes().decode("utf-8")}
    reading = TtMetalHost.host_compute_guard({f"{item}.cpp": text}, harness)
    codes = [diagnostic.code for diagnostic in reading.diagnostics]
    assert reading.host_compute is False, f"{item}: host_compute {reading.host_compute}; {reading.diagnostics}"
    assert codes == ([TAG] if item in DATA_MOVEMENT_ONLY else []), f"{item}: {reading.diagnostics}"


@pytest.mark.parametrize("item", DEV_ITEMS)
def test_a_tt_program_keeps_its_unpack_to_dest_note(item: str) -> None:
    """The note says not taken, and the program's code names no fp32_dest_acc_en or unpack_to_dest_mode."""
    note = suite().items[item].unpack_to_dest
    assert note and "unpack_to_dest" in note and note.startswith("not taken"), f"{item}: {note!r}"
    tokens = lex(program_text(item, "tt"), f"{item}.cpp").tokens
    names = {token.text for token in tokens if token.kind == IDENT}
    for name in ("fp32_dest_acc_en", "unpack_to_dest_mode", "UnpackToDestFp32"):
        assert name not in names, f"{item}: names {name} (ttsim issue #18's path; P4.9)"
    formats = {
        tokens[index + 2].text
        for index in range(len(tokens) - 2)
        if tokens[index].text == "DataFormat" and tokens[index + 1].text == "::"
    }
    assert formats <= {"Float16_b"}, f"{item}: every circular buffer is Float16_b, got {sorted(formats)}"


# ---------------------------------------------------------------------------
# tt-train-v0


def test_tt_train_v0_includes_none_of_the_development_examples() -> None:
    """tt-train-v0's include list names no dev item's example directory and no directory its kernels come from."""
    items = suite().items
    directories = {EXAMPLE_DIRS[name] for name in dev_items(raw())}
    for name in DEV_ITEMS:
        for kernel, entry in items[name].languages["tt"].support.items():
            if kernel != "lassi_io.h":
                directories.add(entry.path.removeprefix(f"{EXAMPLES_DIR}/").split("/kernels/", 1)[0])
    assert {"vecadd_multi_core", "matmul/matmul_common", "matmul/matmul_multicore_reuse_mcast"} <= directories
    spec = yaml.safe_load(TRAIN_SPEC.read_text(encoding="utf-8"))
    assert spec["root"] == EXAMPLES_DIR
    included = list(spec["include"])
    clashes = sorted(
        (example, directory)
        for example in included
        for directory in directories
        if example == directory or example.startswith(f"{directory}/") or directory.startswith(f"{example}/")
    )
    assert clashes == [], f"tt-train-v0 includes a development example: {clashes}"
