"""Tests for the Tier A suite tt-pairs-v0 and its dry-run recipe (task P4.13; P4.14 confirms the splits).

Bible: Benchmark Suites (Tier A, split rules), Harness Contract (lassi_io,
the CPU -> TT guard), Oracles (binary_io), ttsim Facts (unpack_to_dest),
Agent Rule 5. Plan: plans/p4-ttsim.md, P4.13 and the planning decision
"Tier A (P4.13)"; plans/spikes/p4-ttsim-runtime.md (P4.9) for each item's
kernels, declared tolerance, and unpack_to_dest reading. What they fix:

- assets/bench/tt-pairs-v0.yaml pins tt-metal at P4.2's commit
  (toolchains/tt-metal.pin COMMIT and URL, `pin: tt-metal`) and lists the
  five Tier A items, each `unassigned` (OQ-025, option (d)) and refused to
  training, prompt tuning, and corpus harvest (Agent Rule 5).
- Each item's TT version is a host program written here
  (assets/bench/tt-pairs-v0/<item>/tt/<item>.cpp, tracked) with the
  upstream example's kernels, unmodified, as support files fetched from the
  pinned tt-metal with their sha256, each at the relative path the program
  names (a tail of its upstream path, under a directory named kernels),
  plus assets/harness/c/lassi_io.h; its
  C++ version (assets/bench/tt-pairs-v0/<item>/cpp/<item>.cpp, tracked)
  gets lassi_io.h. Each item declares its input specs, outputs, seed, the
  tolerance of its upstream example's own check (P4.9), and an
  unpack_to_dest note.
- Each TT host program reads host_compute False under the CPU -> TT guard
  (TtMetalHost.host_compute_guard with lassi_io.h as its harness): loopback
  with exactly the data-movement-only tag, the four compute items with no
  guard diagnostic. Each keeps the unpack_to_dest reading: its code
  (comments aside) names no fp32_dest_acc_en and no unpack_to_dest_mode,
  and every DataFormat it names is Float16_b.
- tests/fixtures/recipes/p4-tier-a-dry-run.yaml loads and resolves.

The manifest is complete: one test asserts that it holds no PLACEHOLDER
and loads, and the per-item tests load it as it is, so a PLACEHOLDER that
returns (load_suite refuses it) or a missing program fails them, never
skips them. No value here is a measurement.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

from lassi import bench
from lassi.bench import registry
from lassi.core import runner as _runner  # noqa: F401  (registers every component a recipe may bind)
from lassi.core.recipe import load_recipe
from lassi.core.tolerance import Tolerance
from lassi.toolchains import pins
from lassi.toolchains._cxx_scan import lex
from lassi.toolchains.ttmetal_build import TtMetalHost

REPO = Path(__file__).resolve().parents[2]
SUITE = "tt-pairs-v0"
MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
PROGRAMS = REPO / "assets" / "bench" / SUITE
HARNESS = "assets/harness/c/lassi_io.h"
RECIPE = REPO / "tests" / "fixtures" / "recipes" / "p4-tier-a-dry-run.yaml"
ITEMS = ("eltwise_binary", "eltwise_sfpu", "loopback", "matmul_multi_core", "matmul_single_core")
LANGUAGES = ("cpp", "tt")
FORBIDDEN = ("train", "prompt-tuning", "harvest")
PLACEHOLDER = "PLACEHOLDER"
# Each item's declared tolerance: its upstream example's own check (plans/spikes/p4-ttsim-runtime.md, "The
# examples' own checks"). Exact equality is max_abs 0 (bible Benchmark Suites, item tolerance).
TOLERANCES = {
    "loopback": Tolerance(metric="max_abs", threshold=0),
    "eltwise_binary": Tolerance(metric="max_abs", threshold=0.01),
    "eltwise_sfpu": Tolerance(metric="max_abs", threshold=0.05),
    "matmul_single_core": Tolerance(metric="pcc", threshold=0.97),
    "matmul_multi_core": Tolerance(metric="pcc", threshold=0.97),
}
# The kernels each upstream example names after OVERRIDE_KERNEL_PREFIX at the pin (read on the build host in rx
# 20261004-222751-exec-f0dd), each relative to tt_metal/programming_examples. Paths only; no tt-metal text.
EXAMPLES_DIR = "tt_metal/programming_examples"
KERNELS = {
    "loopback": ("loopback/kernels/loopback_dram_copy.cpp",),
    "eltwise_binary": (
        "eltwise_binary/kernels/dataflow/read_tiles.cpp",
        "eltwise_binary/kernels/dataflow/write_tile.cpp",
        "eltwise_binary/kernels/compute/tiles_add.cpp",
    ),
    "eltwise_sfpu": (
        "eltwise_sfpu/kernels/dataflow/read_tile.cpp",
        "eltwise_sfpu/kernels/dataflow/write_tile.cpp",
        "eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp",
    ),
    "matmul_single_core": (
        "matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
        "matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
        "matmul/matmul_single_core/kernels/compute/mm.cpp",
    ),
    "matmul_multi_core": (
        "matmul/matmul_multi_core/kernels/dataflow/reader_mm_output_tiles_partitioned.cpp",
        "matmul/matmul_multi_core/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        "matmul/matmul_multi_core/kernels/compute/mm.cpp",
    ),
}
TAG = "guard-data-movement-only"
DATA_MOVEMENT_ONLY = frozenset({"loopback"})
# The first line of every program (the P4.13 brief, from OQ-018 practice: no tt-metal text enters a tracked file).
FIRST_LINE_START = "// Written for LASSI-DF tt-pairs-v0 (task P4.13)"
FIRST_LINE = FIRST_LINE_START + "; runs the unmodified upstream {item} kernels from tt-metal 5280a9cf."


def raw() -> dict[str, Any]:
    """Return the manifest as YAML reads it, before any check."""
    return yaml.safe_load(MANIFEST.read_bytes().decode("utf-8"))


def placeholders(value: Any, where: str = "") -> list[str]:
    """Return the key paths under `value` whose value is a string starting with PLACEHOLDER."""
    if isinstance(value, str):
        return [where] if value.startswith(PLACEHOLDER) else []
    if isinstance(value, dict):
        return [found for key, item in value.items() for found in placeholders(item, f"{where}/{key}")]
    if isinstance(value, list):
        return [found for index, item in enumerate(value) for found in placeholders(item, f"{where}/{index}")]
    return []


def suite() -> bench.Suite:
    """Load the manifest as it is; a PLACEHOLDER anywhere in it raises ValueError, which fails the calling test."""
    return bench.load_suite(MANIFEST)


def program(item: str, language: str) -> Path:
    """Return where the item's program in `language` is tracked."""
    return PROGRAMS / item / language / f"{item}.cpp"


# ---------------------------------------------------------------------------
# The manifest


def test_the_manifest_pins_tt_metal_at_the_install_commit() -> None:
    pin, data = pins.read_pin("tt-metal"), raw()
    assert data["suite"] == SUITE
    assert (data["repo"], data["commit"], data["pin"]) == (pin["URL"], pin["COMMIT"], "tt-metal")


def test_the_manifest_lists_the_five_tier_a_items_each_unassigned() -> None:
    items = raw()["items"]
    assert sorted(items) == list(ITEMS)
    for item, entry in items.items():
        assert entry["split"] == "unassigned", f"{item}: every Tier A item stays unassigned (OQ-025)"


def test_the_manifest_holds_no_placeholder_and_loads() -> None:
    assert placeholders(raw()) == [], f"the manifest is complete; load_suite refuses a {PLACEHOLDER}"
    assert sorted(suite().items) == list(ITEMS)


@pytest.mark.parametrize("item", ITEMS)
def test_an_item_is_unassigned_and_refused_to_every_purpose_rule_5_forbids(item: str) -> None:
    loaded = suite()
    assert loaded.item(item, purpose="eval").split == "unassigned"
    for purpose in FORBIDDEN:
        with pytest.raises(bench.EvalSplitError, match="unassigned"):
            loaded.item(item, purpose=purpose)


@pytest.mark.parametrize("item", ITEMS)
def test_an_item_declares_its_tolerance_inputs_and_unpack_to_dest_note(item: str) -> None:
    found = suite().items[item]
    assert found.tolerance == TOLERANCES[item]
    assert found.inputs and found.outputs and found.seed is not None
    assert found.run_args == ()
    for entry in found.inputs:
        count = 1
        for dim in entry["shape"]:
            count *= dim
        assert 0 < count <= registry.MAX_INPUT_ELEMENTS, f"{item}: {entry['name']} holds {count} elements"
    assert found.unpack_to_dest and "unpack_to_dest" in found.unpack_to_dest


@pytest.mark.parametrize("item", ITEMS)
def test_an_items_programs_are_tracked_and_its_kernels_are_the_upstream_ones(item: str) -> None:
    found = suite().items[item]
    assert sorted(found.languages) == list(LANGUAGES)
    for language in LANGUAGES:
        spec = found.languages[language]
        assert spec.tracked and spec.dir == f"assets/bench/{SUITE}/{item}/{language}"
        assert spec.files == (f"{item}.cpp",)
    assert found.support == {}, "every support file belongs to one language"
    cpp, tt = found.languages["cpp"].support, found.languages["tt"].support
    assert set(cpp) == {"lassi_io.h"} and (cpp["lassi_io.h"].tracked, cpp["lassi_io.h"].path) == (True, HARNESS)
    assert (tt["lassi_io.h"].tracked, tt["lassi_io.h"].path) == (True, HARNESS)
    kernels = {name: entry for name, entry in tt.items() if name != "lassi_io.h"}
    assert sorted(entry.path for entry in kernels.values()) == sorted(f"{EXAMPLES_DIR}/{k}" for k in KERNELS[item])
    text = program_text(item, "tt")
    for name, entry in kernels.items():
        assert not entry.tracked and entry.sha256 and re.fullmatch(r"[0-9a-f]{64}", entry.sha256)
        assert "kernels" in PurePosixPath(name).parts[:-1], f"{name}: ttmetal-host places it, never compiles it"
        assert entry.path.endswith(f"/{name}"), f"{name} is a tail of the upstream path {entry.path}"
        assert f'"{name}"' in text, f"{item}: the TT host program names its kernel at {name}, where the build puts it"


# ---------------------------------------------------------------------------
# The programs: the guard and the unpack_to_dest reading


def program_text(item: str, language: str) -> str:
    """Return a program's text; fail when it does not exist."""
    path = program(item, language)
    if not path.is_file():
        pytest.fail(f"{item}: the manifest entry is complete but {path} does not exist")
    return path.read_bytes().decode("utf-8")


@pytest.mark.parametrize("item", ITEMS)
def test_a_tt_host_program_passes_the_cpu_to_tt_guard(item: str) -> None:
    text = program_text(item, "tt")
    harness = {"lassi_io.h": (REPO / HARNESS).read_bytes().decode("utf-8")}
    reading = TtMetalHost.host_compute_guard({f"{item}.cpp": text}, harness)
    codes = [diagnostic.code for diagnostic in reading.diagnostics]
    assert reading.host_compute is False, f"{item}: host_compute {reading.host_compute}; {reading.diagnostics}"
    assert codes == ([TAG] if item in DATA_MOVEMENT_ONLY else []), f"{item}: {reading.diagnostics}"


@pytest.mark.parametrize("item", ITEMS)
def test_a_tt_host_program_keeps_the_unpack_to_dest_reading(item: str) -> None:
    # Code tokens only (lassi.toolchains._cxx_scan.lex drops comments), so a comment may say what is not set.
    tokens = lex(program_text(item, "tt"), f"{item}.cpp").tokens
    names = {token.text for token in tokens if token.kind == "IDENT"}
    for name in ("fp32_dest_acc_en", "unpack_to_dest_mode", "UnpackToDestFp32"):
        assert name not in names, f"{item}: names {name} (ttsim issue #18's path; P4.9)"
    formats = {
        tokens[index + 2].text
        for index in range(len(tokens) - 2)
        if tokens[index].text == "DataFormat" and tokens[index + 1].text == "::"
    }
    assert formats <= {"Float16_b"}, f"{item}: every circular buffer is Float16_b, got {sorted(formats)}"


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("item", ITEMS)
def test_a_program_opens_with_its_provenance_line_and_is_plain_ascii(item: str, language: str) -> None:
    text = program_text(item, language)
    assert text.isascii() and "\r" not in text, f"{item} ({language}): plain ASCII with LF line ends"
    first = text.split("\n", 1)[0]
    if language == "tt":
        assert first == FIRST_LINE.format(item=item)
    else:
        assert first.startswith(FIRST_LINE_START)


# ---------------------------------------------------------------------------
# The dry-run recipe


def test_the_dry_run_recipe_loads_and_resolves() -> None:
    recipe = load_recipe(RECIPE)
    data = recipe.data
    assert data["model"] == {"backend": "mock", "id": "mock-reference"}
    assert data["bench"]["suite"] == SUITE and data["bench"]["split"] == "unassigned"
    assert sorted(data["bench"]["items"]) == list(ITEMS)
    assert data["directions"] == [{"source": "cpp", "target": "tt"}, {"source": "tt", "target": "cpp"}]
    assert data["fixes"]["baseline_both"] is True
    assert data["toolchain"] == {"cpp": "gcc-native", "tt": "ttmetal-host"}
    assert data["executor"]["cpp"] == {"kind": "native"}
    assert data["executor"]["tt"] == {"kind": "ttsim", "arch": "wormhole_b0", "dispatch": "slow"}
    assert data["oracle"]["kind"] == "binary_io" and data["oracle"]["threshold"] == "from_baseline"
    assert data["stages"] == ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
    assert data["trials"]["n"] == 1
    bound = {(binding.interface, binding.name) for binding in recipe.bindings}
    assert {("Executor", "native"), ("Executor", "ttsim"), ("Oracle", "binary_io"), ("LLMBackend", "mock")} <= bound
