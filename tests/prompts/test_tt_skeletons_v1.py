"""Tests for tt-host-skeleton-v1 and tt-kh-skeleton-v1, the skeletons with the compute grid by default (P17.71).

Bible: Project Recipes (Notes, the Skeleton bullet), Harness Contract (lassi_io.h; the CPU -> TT guard), Toolchain
Pins (tt-metal 5280a9cf), Agent Rules 4 (the skeleton is opt-in) and 5 (no item's program, held-out or not, shapes a
skeleton). The owner's answer to OQ-096 of 2026-10-25, "Item-agnostic only", holds for both: they are written from
the v0 skeletons and the pinned tt-metalium headers, never from a suite item's program or its upstream example.

The contract these tests fix for assets/skeletons/tt-host-skeleton-v1/ (language tt) and
assets/skeletons/tt-kh-skeleton-v1/ (language ttkh):

1. MANIFEST.yaml follows the v0 manifest's format: the skeleton, its language, the pin (tt-metal) and the pin's
   COMMIT, and the v0 skeleton's files in v0's order, each with its sha256 and size; its sha256 is that of the
   lines "<sha256>  <path>\\n" of the files in order; lassi.prompts.skeletons loads it. The v0 skeletons are
   unchanged, and tt-kh-skeleton-v1's host.cpp is tt-host-skeleton-v1's byte for byte, as in v0.
2. Every file is plain ASCII with LF line ends. Each file's BLANK blocks are its v0 file's, same names and order,
   each closed and holding code. host.cpp keeps v0's helpers, harness calls, and frame: outside the blanks cores,
   circular_buffers, kernels, and runtime_args it is v0's host.cpp, with the includes of work_split.hpp and
   core_coord.hpp added and the header comment rewritten.
3. The cores blank takes the device's compute grid (compute_with_storage_grid_size) and splits the output tiles
   over it with split_work_to_cores, whose all_cores, cores_1, and cores_2 are CoreRangeSets; no single
   CoreCoord core = {0, 0} remains. Every circular buffer and kernel goes on all_cores. The runtime_args blank
   calls SetRuntimeArgs once per kernel and core, over corerange_to_cores of each core group, with that core's
   first tile and tile count, the next core's first tile following.
4. tt-kh-skeleton-v1's three kernel skeletons read their first tile and tile count with get_arg_val at the positions
   the host passes them, and loop over those tiles only; the compute kernel reads no compile arg.
5. The CPU -> TT guard reads host.cpp (with the kernels, for kh) as no host compute, with no diagnostic.
6. No file of either skeleton names an item of any suite under assets/bench (read through the item keys only) or a
   tt-pairs-v0 item, kernel, or file; the BLANK markers' names are left out of the suite check, since contract
   point 2 fixes them as v0's and one, layout, is also a suite's item key, a generic word that names no item's
   program here. No file shares a run of three non-blank lines with a tt-par-v0 held-out
   file (P17.68's leak test, run here on the files before git tracks them; a failure names only the skeleton's
   line, never the held-out text).

Nothing here compiles a skeleton or runs it: the compile at the pin and a one-tile run on ttsim are the build
host's (P17.71). No value here is a measurement.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from string import Template

import pytest
import yaml

from lassi.prompts import skeletons
from lassi.toolchains.pins import read_pin
from lassi.toolchains.ttmetal_guard import read_host_compute

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "core"))
sys.path.insert(0, str(REPO / "tests" / "bench"))

import test_tt_par_heldout as heldout  # noqa: E402
from test_template_context import named_tier_a  # noqa: E402

SKELETONS = REPO / "assets" / "skeletons"
HOST_V1, KH_V1 = "tt-host-skeleton-v1", "tt-kh-skeleton-v1"
# Each v1 skeleton: its language and the v0 skeleton it is a version of.
V1 = {HOST_V1: ("tt", "tt-host-skeleton-v0"), KH_V1: ("ttkh", "tt-kh-skeleton-v0")}
KERNELS = ("kernels/dataflow/reader.cpp", "kernels/compute/compute.cpp", "kernels/dataflow/writer.cpp")
FILES = {HOST_V1: ("host.cpp", "block.txt"), KH_V1: ("host.cpp", "block.txt", *KERNELS)}
ALL_FILES = [(name, path) for name, paths in FILES.items() for path in (*paths, "MANIFEST.yaml")]
CODE_FILES = [(name, path) for name, paths in FILES.items() for path in paths if path.endswith(".cpp")]
# The v0 skeletons' manifest digests: the bytes P17.33 and P17.39 committed, which P17.71 leaves unchanged.
V0_SHA256 = {
    "tt-host-skeleton-v0": "313151639efa7ba70274c4f7be9d4375e9ff02bc733e7940693cdbfbba01a0da",
    "tt-kh-skeleton-v0": "01598e8b4dcaabd146cde63c17e16e628180ce51bfb4688a1382c2acd809cfe2",
}
# The host blanks P17.71 changes; every other blank of host.cpp is v0's byte for byte.
CHANGED = ("cores", "circular_buffers", "kernels", "runtime_args")
ADDED_INCLUDES = ("tt-metalium/core_coord.hpp", "tt-metalium/work_split.hpp")
API_PACK = REPO / "assets" / "context" / "tt-metalium-5280a9cf-full" / "MANIFEST.yaml"
KH_PACK = REPO / "assets" / "context" / "tt-metalium-5280a9cf-kh"
HARNESS = REPO / "assets" / "harness" / "c" / "lassi_io.h"
BENCH = REPO / "assets" / "bench"
INCLUDE = re.compile(r"^\s*#\s*include\s*([<\"])([^>\"]+)[>\"]", re.MULTILINE)
BLANK = re.compile(r"^\s*// BLANK ([A-Za-z0-9_]+): (\S.*)$")
END = re.compile(r"^\s*// END BLANK ([A-Za-z0-9_]+)\s*$")
MARKER_STARTS = ("// BLANK", "// END BLANK")
# A BLANK or END BLANK marker's name, which the item-name check leaves out: the names are v0's (contract point 2).
MARKER_NAME = re.compile(r"(// (?:END )?BLANK) ([A-Za-z0-9_]+)")
KERNEL_CALL = re.compile(r'KernelHandle (\w+) = CreateKernel\(\s*program,\s*"([^"]+)",\s*(\w+),')
RUNTIME_CALL = re.compile(r"SetRuntimeArgs\(program, (\w+), (\w+), \{([^{}]*)\}\);")
GROUP_LOOP = re.compile(r"for \(const CoreCoord& core : corerange_to_cores\((\w+)\)\) \{")
ARG_READ = re.compile(r"(\w+) = get_arg_val<uint32_t>\((\d+)\);")
CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;()]*>)?\s*\(")
NOT_CALLS = {"for", "if", "while", "return", "sizeof", "kernel_main", "static_cast", "uint32_t"}
LOOP = "for (uint32_t tile = first_tile; tile < first_tile + n_tiles; ++tile) {"
V0_LOOP = "for (uint32_t tile = 0; tile < n_tiles; ++tile) {"
# The calls v1's host.cpp makes, each as written: v0's (but its one core) and the multi-core ones (P17.71).
HOST_CALLS = (
    "using namespace tt::tt_metal;",
    "distributed::MeshDevice::create_unit_mesh(0)",
    "->mesh_command_queue()",
    "distributed::MeshBuffer::create(",
    "distributed::EnqueueWriteMeshBuffer(",
    "CreateProgram()",
    "CircularBufferConfig(",
    "CreateCircularBuffer(",
    "TensorAccessorArgs(*",
    "DataMovementConfig{",
    "ComputeConfig{",
    "CreateKernel(",
    "SetRuntimeArgs(",
    "->address()",
    "distributed::EnqueueMeshWorkload(",
    "distributed::Finish(",
    "distributed::EnqueueReadMeshBuffer(",
    "tilize_nfaces(",
    "untilize_nfaces(",
    "device->close();",
    "lassi_io_read(",
    "lassi_io_write(",
    "lassi_io_free(",
    "CoreCoord grid = device->compute_with_storage_grid_size();",
    "split_work_to_cores(grid, n_tiles);",
    "corerange_to_cores(cores_1)",
    "corerange_to_cores(cores_2)",
)
# What each kernel must say, in the pinned forms (tt_metal/hw/inc/api at 5280a9cf), per core.
KERNEL_FORMS = {
    "kernels/dataflow/reader.cpp": (
        '#include "api/dataflow/dataflow_api.h"', "void kernel_main()", "get_arg_val<uint32_t>(0)",
        "get_arg_val<uint32_t>(1)", "get_arg_val<uint32_t>(2)", "TensorAccessorArgs<0>()", "TensorAccessor(",
        "cb_reserve_back(", "get_write_ptr(", "noc_async_read_page(tile, in0, l1_addr)", "noc_async_read_barrier()",
        "cb_push_back(", "tt::CBIndex::c_0", LOOP,
    ),
    "kernels/compute/compute.cpp": (
        '#include "api/compute/common.h"', '#include "api/compute/tile_move_copy.h"',
        '#include "api/compute/eltwise_unary/eltwise_unary.h"', "void kernel_main()", "get_arg_val<uint32_t>(0)",
        "get_arg_val<uint32_t>(1)", "init_sfpu(", "copy_tile_init(", "cb_wait_front(", "tile_regs_acquire()",
        "copy_tile(", "tile_regs_commit()", "tile_regs_wait()", "cb_reserve_back(", "pack_tile(", "cb_push_back(",
        "cb_pop_front(", "tile_regs_release()", "tt::CBIndex::c_0", "tt::CBIndex::c_16", LOOP,
    ),
    "kernels/dataflow/writer.cpp": (
        '#include "api/dataflow/dataflow_api.h"', "void kernel_main()", "get_arg_val<uint32_t>(0)",
        "get_arg_val<uint32_t>(1)", "get_arg_val<uint32_t>(2)", "TensorAccessorArgs<0>()", "TensorAccessor(",
        "cb_wait_front(", "get_read_ptr(", "noc_async_write_page(tile, out0, l1_addr)", "noc_async_write_barrier()",
        "cb_pop_front(", "tt::CBIndex::c_16", LOOP,
    ),
}


def text_of(path: Path) -> str:
    """Return a skeleton file as ASCII text; fail when it is missing or not plain ASCII with LF line ends."""
    assert path.is_file(), f"{path} is missing"
    data = path.read_bytes()
    assert data.isascii() and b"\r" not in data, f"{path} is not plain ASCII with LF line ends"
    return data.decode("ascii")


def v1_text(name: str, path: str) -> str:
    """Return file `path` of the v1 skeleton `name`."""
    return text_of(SKELETONS / name / path)


def v0_text(name: str, path: str) -> str:
    """Return file `path` of the v0 skeleton that the v1 skeleton `name` is a version of."""
    return text_of(SKELETONS / V1[name][1] / path)


def code_of(text: str) -> str:
    """Return C++ `text` with each // comment removed (the skeletons hold no block comment and no // in a string)."""
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def sha256(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def blanks(text: str) -> list[tuple[str, int, int]]:
    """Return each BLANK block of `text` as (name, BLANK line index, END line index), in order; fail on a bad pair."""
    found: list[tuple[str, int, int]] = []
    open_name, open_at = None, -1
    for index, line in enumerate(text.splitlines()):
        start, end = BLANK.match(line), END.match(line)
        if start:
            assert open_name is None, f"line {index + 1}: BLANK {start.group(1)} opens inside BLANK {open_name}"
            open_name, open_at = start.group(1), index
        elif end:
            assert end.group(1) == open_name, f"line {index + 1}: END BLANK {end.group(1)} closes no open BLANK"
            found.append((open_name, open_at, index))
            open_name = None
        else:
            marker = line.strip().startswith(MARKER_STARTS)
            assert not marker, f"line {index + 1}: a BLANK marker not in the form the block names: {line!r}"
    assert open_name is None, f"BLANK {open_name} has no END BLANK {open_name}"
    return found


def blank_texts(text: str) -> dict[str, str]:
    """Return {name: the block's lines from its BLANK line to its END BLANK line} of `text`."""
    lines = text.splitlines()
    return {name: "\n".join(lines[start:end + 1]) for name, start, end in blanks(text)}


def frame(text: str, start: str) -> list[str]:
    """Return `text` from its first line that starts with `start` on, each BLANK block cut to its END BLANK line."""
    lines = text.splitlines()
    kept = lines[next(i for i, line in enumerate(lines) if line.startswith(start)):]
    out: list[str] = []
    inside = False
    for line in kept:
        if BLANK.match(line):
            inside = True
        elif END.match(line):
            inside = False
            out.append(line)
        elif not inside:
            out.append(line)
    return out


def manifest(name: str) -> dict:
    """Return the v1 skeleton's MANIFEST.yaml."""
    document = yaml.safe_load(v1_text(name, "MANIFEST.yaml"))
    assert isinstance(document, dict), f"{name}/MANIFEST.yaml is not a mapping"
    return document


def suite_item_names() -> set[str]:
    """Return every item key of every suite manifest under assets/bench (nothing else of a suite is read)."""
    names: set[str] = set()
    paths = sorted(BENCH.glob("*.yaml"))
    assert paths, "assets/bench holds the suite manifests"
    for path in paths:
        document = yaml.safe_load(path.read_bytes().decode("utf-8"))
        items = document.get("items") if isinstance(document, dict) else None
        assert isinstance(items, dict) and items, f"{path.name} lists no items by key"
        names.update(str(key) for key in items)
    return names


def named_items(text: str, names: set[str]) -> list[str]:
    """Return each name of `names` that `text` holds as a whole word, in any letter case, sorted."""
    return sorted(name for name in names
                  if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text, re.IGNORECASE))


# ---------------------------------------------------------------------------
# The manifests and the loader (contract point 1)


@pytest.mark.parametrize("name", sorted(V1))
def test_the_manifest_names_the_skeleton_its_language_the_pin_and_v0s_files(name: str) -> None:
    document = manifest(name)
    language, v0 = V1[name]
    assert (document["skeleton"], document["language"], document["pin"]) == (name, language, "tt-metal")
    assert document["commit"] == read_pin("tt-metal")["COMMIT"], "the skeleton is written for the tt-metal pin"
    v0_paths = [entry["path"] for entry in yaml.safe_load(text_of(SKELETONS / v0 / "MANIFEST.yaml"))["files"]]
    assert [entry["path"] for entry in document["files"]] == v0_paths == list(FILES[name])
    lines = ""
    for entry in document["files"]:
        data = (SKELETONS / name / entry["path"]).read_bytes()
        assert entry["sha256"] == sha256(data), f"{name}/{entry['path']}: its sha256 in MANIFEST.yaml is stale"
        assert entry["bytes"] == len(data), f"{name}/{entry['path']}: its size in MANIFEST.yaml is stale"
        lines += f"{entry['sha256']}  {entry['path']}\n"
    assert document["sha256"] == sha256(lines.encode("ascii")), "the manifest's sha256 is not that of its entries"


@pytest.mark.parametrize("name", sorted(V1))
def test_the_loader_reads_the_skeleton_as_it_reads_v0(name: str) -> None:
    tree = SKELETONS / name
    assert skeletons.check_skeleton(tree) == []
    loaded = skeletons.load_skeleton(tree)
    assert (loaded.name, loaded.language) == (name, V1[name][0])
    assert loaded.sha256 == manifest(name)["sha256"]
    assert loaded.host == v1_text(name, "host.cpp") and loaded.block == v1_text(name, "block.txt")
    assert [path for path, _ in loaded.kernels] == list(FILES[name][2:])
    assert loaded.sha256 != skeletons.load_skeleton(SKELETONS / V1[name][1]).sha256


@pytest.mark.parametrize("v0", sorted(V0_SHA256))
def test_the_v0_skeletons_are_unchanged(v0: str) -> None:
    tree = SKELETONS / v0
    assert skeletons.check_skeleton(tree) == []
    assert skeletons.load_skeleton(tree).sha256 == V0_SHA256[v0], f"{v0} changed (P17.71 leaves it as it was)"


def test_the_kh_host_program_is_the_host_skeletons_byte_for_byte() -> None:
    assert (SKELETONS / KH_V1 / "host.cpp").read_bytes() == (SKELETONS / HOST_V1 / "host.cpp").read_bytes()


# ---------------------------------------------------------------------------
# Text, blanks, and v0's frame (contract point 2)


@pytest.mark.parametrize(("name", "path"), ALL_FILES)
def test_every_file_is_plain_ascii_with_lf_line_ends(name: str, path: str) -> None:
    assert v1_text(name, path)


@pytest.mark.parametrize(("name", "path"), CODE_FILES)
def test_the_blanks_are_v0s_in_the_same_order_each_holding_code(name: str, path: str) -> None:
    text = v1_text(name, path)
    names = [blank for blank, _, _ in blanks(text)]
    assert names == [blank for blank, _, _ in blanks(v0_text(name, path))], f"{name}/{path}: {names}"
    lines = text.splitlines()
    for blank, start, end in blanks(text):
        assert code_of("\n".join(lines[start + 1:end])).strip(), f"{name}/{path}: BLANK {blank} holds no code"


def test_host_keeps_v0s_helpers_harness_calls_and_frame() -> None:
    v1, v0 = v1_text(HOST_V1, "host.cpp"), v0_text(HOST_V1, "host.cpp")
    start = "using namespace"
    assert frame(v1, start) == frame(v0, start), "outside its blanks host.cpp is v0's from `using namespace` on"
    v1_blanks, v0_blanks = blank_texts(v1), blank_texts(v0)
    for blank in v0_blanks:
        if blank not in CHANGED:
            assert v1_blanks[blank] == v0_blanks[blank], f"BLANK {blank} differs from v0's"
    for blank in CHANGED:
        assert v1_blanks[blank] != v0_blanks[blank], f"BLANK {blank} is v0's; P17.71 changes it"


def test_host_includes_v0s_headers_plus_work_split_and_core_coord() -> None:
    v1 = INCLUDE.findall(code_of(v1_text(HOST_V1, "host.cpp")))
    v0 = INCLUDE.findall(code_of(v0_text(HOST_V1, "host.cpp")))
    assert sorted(v1) == sorted(v0 + [("<", name) for name in ADDED_INCLUDES])
    document = yaml.safe_load(API_PACK.read_bytes().decode("ascii"))
    prefix = "tt_metal/api/"
    pack = {source["path"][len(prefix):] for entry in document["entries"] for source in entry["sources"]
            if source["path"].startswith(prefix + "tt-metalium/")}
    allowed = pack | {"tt-metalium/work_split.hpp"}
    for bracket, name in v1:
        if bracket == '"':
            assert name == "lassi_io.h", f'host.cpp includes "{name}"; only "lassi_io.h" is quoted'
        else:
            assert name in allowed, f"host.cpp includes <{name}>, which neither the API pack nor P17.71 names"


@pytest.mark.parametrize("path", KERNELS)
def test_each_kernel_keeps_v0s_includes_frame_and_unchanged_blanks(path: str) -> None:
    v1, v0 = v1_text(KH_V1, path), v0_text(KH_V1, path)
    assert INCLUDE.findall(code_of(v1)) == INCLUDE.findall(code_of(v0))
    v1_frame = [line.replace(LOOP, V0_LOOP) for line in frame(v1, "#include")]
    assert v1_frame == frame(v0, "#include"), f"{path}: outside its blanks and its loop line it is v0's"
    v1_blanks, v0_blanks = blank_texts(v1), blank_texts(v0)
    for blank, text in v0_blanks.items():
        if not blank.endswith(("_args", "_loop")):
            assert v1_blanks[blank] == text, f"{path}: BLANK {blank} differs from v0's"
    assert code_of(v1).count("void kernel_main()") == 1
    assert "MAIN" not in code_of(v1) and "namespace NAMESPACE" not in v1


def test_the_header_comment_says_item_agnostic_and_names_its_sources() -> None:
    head = v1_text(HOST_V1, "host.cpp").split("#include", 1)[0]
    for words in ("tt-host-skeleton-v1", "item-agnostic", "5280a9cf", "tt-metalium", "tt-host-skeleton-v0",
                  "work_split.hpp", "compute grid"):
        assert words in " ".join(head.replace("//", " ").split()), f"the header comment does not say {words!r}"


# ---------------------------------------------------------------------------
# The compute grid, the core ranges, and the runtime args (contract point 3)


@pytest.mark.parametrize("call", HOST_CALLS)
def test_host_makes_the_call_in_the_pinned_form(call: str) -> None:
    assert call in code_of(v1_text(HOST_V1, "host.cpp")), f"host.cpp does not write {call!r}"


@pytest.mark.parametrize("call", ["distributed::MeshBuffer::create(", "CircularBufferConfig(", "CreateCircularBuffer("])
def test_buffers_and_circular_buffers_go_through_one_helper(call: str) -> None:
    assert code_of(v1_text(HOST_V1, "host.cpp")).count(call) == 1, f"{call!r} is written once, inside its helper"


def test_the_cores_blank_splits_the_output_tiles_over_the_compute_grid() -> None:
    block = blank_texts(v1_text(HOST_V1, "host.cpp"))["cores"]
    code = code_of(block)
    assert "CoreCoord grid = device->compute_with_storage_grid_size();" in code
    binding = re.search(r"auto \[(\w+), (\w+), (\w+), (\w+), (\w+), (\w+)\] =\s*split_work_to_cores\(grid, n_tiles\);",
                        code)
    assert binding, "the cores blank binds split_work_to_cores(grid, n_tiles)'s six results"
    assert binding.groups() == ("n_cores", "all_cores", "cores_1", "cores_2", "units_1", "units_2")
    assert "CoreRangeSet" in block, "the cores blank says all_cores and the two groups are CoreRangeSets"
    whole = code_of(v1_text(HOST_V1, "host.cpp"))
    assert "CoreCoord core = {0, 0}" not in whole and not re.search(r"\bcore\s*=\s*\{\s*0\s*,\s*0\s*\}", whole)


def test_every_circular_buffer_and_kernel_goes_on_all_cores() -> None:
    code = code_of(v1_text(HOST_V1, "host.cpp"))
    cbs = re.findall(r"make_cb\(program, (\w+), tt::CBIndex::c_\d+, \d+\);", code)
    assert cbs and set(cbs) == {"all_cores"}, cbs
    kernels = KERNEL_CALL.findall(code)
    assert [path for _, path, _ in kernels] == list(KERNELS)
    assert {spec for _, _, spec in kernels} == {"all_cores"}, kernels


def runtime_args() -> dict[str, list[tuple[str, list[str]]]]:
    """Return {core group: [(kernel handle, its runtime args as written)]} of the runtime_args blank's loops."""
    code = code_of(blank_texts(v1_text(HOST_V1, "host.cpp"))["runtime_args"])
    parts = GROUP_LOOP.split(code)
    assert len(parts) == 5, "the runtime_args blank loops once over corerange_to_cores of each core group"
    found: dict[str, list[tuple[str, list[str]]]] = {}
    for group, body in zip(parts[1::2], parts[2::2], strict=True):
        calls = RUNTIME_CALL.findall(body)
        assert all(spec == "core" for _, spec, _ in calls), f"each SetRuntimeArgs in {group}'s loop is per core"
        found[group] = [(handle, [arg.strip() for arg in args.split(",")]) for handle, _, args in calls]
    return found


def test_the_runtime_args_blank_gives_each_core_its_first_tile_and_tile_count() -> None:
    code = code_of(blank_texts(v1_text(HOST_V1, "host.cpp"))["runtime_args"])
    assert "uint32_t first_tile = 0;" in code
    loops = runtime_args()
    assert list(loops) == ["cores_1", "cores_2"]
    handles = [handle for handle, _, _ in KERNEL_CALL.findall(code_of(v1_text(HOST_V1, "host.cpp")))]
    for group, count in (("cores_1", "units_1"), ("cores_2", "units_2")):
        assert [handle for handle, _ in loops[group]] == handles, f"one SetRuntimeArgs per kernel per core of {group}"
        for handle, args in loops[group]:
            assert args[-2:] == ["first_tile", count], f"{handle} on {group} gets {args}"
        assert f"first_tile += {count};" in code, f"the next core's first tile follows a core of {group}"
    assert code.count("SetRuntimeArgs(") == 2 * len(handles)
    whole = code_of(v1_text(HOST_V1, "host.cpp"))
    assert whole.count("SetRuntimeArgs(") == code.count("SetRuntimeArgs("), "every SetRuntimeArgs is in the blank"


# ---------------------------------------------------------------------------
# The kernel skeletons (contract point 4)


def test_the_kernels_lie_at_the_paths_the_host_skeleton_launches() -> None:
    launched = re.findall(r'"(kernels/[^"]+\.cpp)"', code_of(v1_text(KH_V1, "host.cpp")))
    assert launched == list(KERNELS)


@pytest.mark.parametrize("path", KERNELS)
def test_each_kernel_makes_its_calls_in_the_pinned_form(path: str) -> None:
    code = code_of(v1_text(KH_V1, path))
    for form in KERNEL_FORMS[path]:
        assert form in code, f"{path} does not write {form!r}"
    if path == "kernels/compute/compute.cpp":
        assert "get_compile_time_arg_val(" not in code, "the compute kernel's tile count is a runtime arg"
    assert V0_LOOP not in code, f"{path} still loops from tile 0"


@pytest.mark.parametrize("path", KERNELS)
def test_each_kernel_reads_its_first_tile_and_count_where_the_host_passes_them(path: str) -> None:
    code = code_of(v1_text(KH_V1, path))
    reads = {name: int(index) for name, index in ARG_READ.findall(code)}
    handle = {launched: handle for handle, launched, _ in KERNEL_CALL.findall(code_of(v1_text(KH_V1, "host.cpp")))}
    for group, count in (("cores_1", "units_1"), ("cores_2", "units_2")):
        args = dict(runtime_args()[group])[handle[path]]
        assert reads.get("first_tile") == args.index("first_tile"), f"{path} reads first_tile at {reads}"
        assert reads.get("n_tiles") == args.index(count), f"{path} reads its tile count at {reads}"
        assert sorted(reads.values()) == list(range(len(args))), f"{path} reads every runtime arg once: {reads}"


def test_every_kernel_api_call_is_declared_in_the_kernels_pack() -> None:
    texts = [KH_PACK / name for name in ("metalium_guide.ttkh.txt", "host_api.ttkh.txt", "kernel_api.ttkh.txt")]
    if not all(path.is_file() for path in texts):
        pytest.skip("the kernels pack is not built here; build it with uv run tools/build_tt_context.py --kernels")
    from lassi.core.stages import pack_declarations
    from lassi.prompts.assets import load_tree, pack_text

    pack = pack_text(KH_PACK.name, load_tree(KH_PACK))
    declared = {declaration.name for declaration in pack_declarations(pack)}
    macros = set(re.findall(r"^#define ([A-Za-z_][A-Za-z0-9_]*)\(", pack, re.MULTILINE))
    for path in KERNELS:
        calls = set(CALL.findall(code_of(v1_text(KH_V1, path)))) - NOT_CALLS
        missing = sorted(call for call in calls if call not in declared and call not in macros)
        assert not missing, f"{path} calls {missing}, which the pinned kernel API declares nowhere"


# ---------------------------------------------------------------------------
# The CPU -> TT guard (contract point 5)


def test_the_guard_reads_the_host_skeleton_as_no_host_compute() -> None:
    reading = read_host_compute({"host.cpp": v1_text(HOST_V1, "host.cpp")}, {"lassi_io.h": text_of(HARNESS)})
    assert reading.host_compute is False, reading
    assert reading.diagnostics == (), f"the guard tags or notes the skeleton: {reading.diagnostics}"


def test_the_guard_reads_the_kh_skeleton_with_its_kernels_as_no_host_compute() -> None:
    files = {"host.cpp": v1_text(KH_V1, "host.cpp"), **{path: v1_text(KH_V1, path) for path in KERNELS}}
    reading = read_host_compute(files, {"lassi_io.h": text_of(HARNESS)})
    assert reading.host_compute is False and reading.diagnostics == (), reading


# ---------------------------------------------------------------------------
# block.txt, Agent Rule 5, and the leak test (contract point 6)


@pytest.mark.parametrize("name", sorted(V1))
def test_the_block_renders_with_v0s_fields_and_says_how_to_use_them(name: str) -> None:
    template = v1_text(name, "block.txt")
    fields = {match.group("named") or match.group("braced") for match in Template.pattern.finditer(template)
              if match.group("named") or match.group("braced")}
    v0 = v0_text(name, "block.txt")
    v0_fields = {match.group("named") or match.group("braced") for match in Template.pattern.finditer(v0)
                 if match.group("named") or match.group("braced")}
    assert fields == v0_fields, f"{name}/block.txt uses {sorted(fields)}"
    loaded = skeletons.load_skeleton(SKELETONS / name)
    rendered = skeletons.block_text(loaded)
    assert v1_text(name, "host.cpp") in rendered
    for path, text in loaded.kernels:
        assert f"// FILE: {path}\n{text}" in rendered
    words = " ".join(template.split())
    for phrase in ("BLANK", "END BLANK", "exactly", "all_cores", "first tile", "tile count"):
        assert phrase in words, f"{name}/block.txt does not say {phrase!r}"


@pytest.mark.parametrize(("name", "path"), ALL_FILES)
def test_no_item_of_any_suite_is_named(name: str, path: str) -> None:
    text = v1_text(name, path)
    v0_names = {blank for v0_path in FILES[name] if v0_path.endswith(".cpp")
                for blank, _, _ in blanks(v0_text(name, v0_path))}
    marked = {marker for _, marker in MARKER_NAME.findall(text)}
    assert marked <= v0_names, f"{name}/{path} marks blanks v0 has not: {sorted(marked - v0_names)}"
    unmarked = MARKER_NAME.sub(r"\1", text)
    assert named_items(unmarked, suite_item_names()) == [], f"{name}/{path} names a suite item (Agent Rule 5)"
    assert named_tier_a(text) == [], f"{name}/{path} names a tt-pairs-v0 item or file (Agent Rule 5)"


def test_no_file_shares_three_lines_with_a_held_out_file() -> None:
    held = {run for item in heldout.HELD_OUT for text in heldout.held_out_files(item).values()
            for run in heldout.runs(text)}
    assert held, "the held-out files give runs to compare"
    hits = [f"{name}/{path}:{line}" for name, path in ALL_FILES
            for run, line in heldout.runs(v1_text(name, path)).items() if run in held]
    assert hits == [], f"runs of {heldout.RUN_LINES} lines shared with a held-out file start at: {', '.join(hits)}"
