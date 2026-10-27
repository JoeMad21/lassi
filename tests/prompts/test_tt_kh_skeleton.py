"""Tests for tt-kh-skeleton-v0, the item-agnostic host and kernel skeletons of the C++ -> TT kernels arm (P17.39).

Bible: Project Recipes (Notes, the Skeleton bullet), Harness Contract (lassi_io.h; the CPU -> TT guard; a kernel
compiled by the kernel JIT at first launch), Toolchain Pins (tt-metal 5280a9cf), Agent Rule 5. The owner's answer
to OQ-096 of 2026-10-25, "Item-agnostic only", holds for every skeleton: these are written from the pinned headers
and tt-metal's documentation and examples outside the five tt-pairs-v0 items, never from an item's program.

The contract these tests fix for assets/skeletons/tt-kh-skeleton-v0/:

1. MANIFEST.yaml names the skeleton, its language (ttkh), the pin (tt-metal) and the pin's COMMIT, and lists
   host.cpp, block.txt, and the three kernel skeletons, each with its sha256 and size; lassi.prompts.skeletons
   reads it.
2. host.cpp is tt-host-skeleton-v0's host.cpp byte for byte, the host skeleton that compiled and linked at the pin
   from a clean commit (bible, the Skeleton bullet; results/p17-skeleton-build/).
3. The kernel skeletons lie at the paths the host skeleton's CreateKernel calls name (a reader, a compute kernel,
   and a writer), each plain ASCII, including only <cstdint> and "api/..." kernel headers, defining
   void kernel_main(), and reading the args in the order the host skeleton passes them; every BLANK block has its
   END BLANK line and holds code; every pinned kernel API call they make is declared in the kernels pack when it
   is built here.
4. block.txt renders with $skeleton and $kernels, says to fill only the blanks, to write every kernel at the path
   the prompt lists, and to keep each call's form; no file names a tt-pairs-v0 item, kernel, or file.

Nothing here compiles a skeleton; the kernels compile only when a program launches them on ttsim, which a window
run checks. No value here is a measurement.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest
import yaml

from lassi.prompts import skeletons
from lassi.toolchains.pins import read_pin
from lassi.toolchains.ttmetal_guard import read_host_compute

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "core"))

from test_template_context import named_tier_a  # noqa: E402

NAME = "tt-kh-skeleton-v0"
TREE = REPO / "assets" / "skeletons" / NAME
HOST_TREE = REPO / "assets" / "skeletons" / "tt-host-skeleton-v0"
KERNELS = ("kernels/dataflow/reader.cpp", "kernels/compute/compute.cpp", "kernels/dataflow/writer.cpp")
HARNESS = REPO / "assets" / "harness" / "c" / "lassi_io.h"
KH_PACK = REPO / "assets" / "context" / "tt-metalium-5280a9cf-kh"
INCLUDE = re.compile(r"^\s*#\s*include\s*([<\"])([^>\"]+)[>\"]", re.MULTILINE)
BLANK = re.compile(r"^\s*// BLANK ([A-Za-z0-9_]+): (\S.*)$")
END = re.compile(r"^\s*// END BLANK ([A-Za-z0-9_]+)\s*$")
CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;()]*>)?\s*\(")
# Words a call regex finds that are no API call: C++ keywords, the entry point, and the loop's own names.
NOT_CALLS = {"for", "if", "while", "return", "sizeof", "kernel_main", "static_cast", "uint32_t"}
# What each kernel must say, in the pinned forms (tt_metal/hw/inc/api at 5280a9cf) and the host skeleton's order.
FORMS = {
    "kernels/dataflow/reader.cpp": (
        '#include "api/dataflow/dataflow_api.h"', "void kernel_main()", "get_arg_val<uint32_t>(0)",
        "get_arg_val<uint32_t>(1)", "TensorAccessorArgs<0>()", "TensorAccessor(", "cb_reserve_back(",
        "get_write_ptr(", "noc_async_read_page(", "noc_async_read_barrier()", "cb_push_back(", "tt::CBIndex::c_0",
    ),
    "kernels/compute/compute.cpp": (
        '#include "api/compute/common.h"', '#include "api/compute/tile_move_copy.h"',
        '#include "api/compute/eltwise_unary/eltwise_unary.h"', "void kernel_main()",
        "get_compile_time_arg_val(0)", "init_sfpu(", "copy_tile_init(", "cb_wait_front(", "tile_regs_acquire()",
        "copy_tile(", "tile_regs_commit()", "tile_regs_wait()", "cb_reserve_back(", "pack_tile(", "cb_push_back(",
        "cb_pop_front(", "tile_regs_release()", "tt::CBIndex::c_0", "tt::CBIndex::c_16",
    ),
    "kernels/dataflow/writer.cpp": (
        '#include "api/dataflow/dataflow_api.h"', "void kernel_main()", "get_arg_val<uint32_t>(0)",
        "get_arg_val<uint32_t>(1)", "TensorAccessorArgs<0>()", "TensorAccessor(", "cb_wait_front(",
        "get_read_ptr(", "noc_async_write_page(", "noc_async_write_barrier()", "cb_pop_front(", "tt::CBIndex::c_16",
    ),
}


def text_of(path: Path) -> str:
    """Return a skeleton file as ASCII text; fail when it is missing or not plain ASCII with LF line ends."""
    assert path.is_file(), f"{path} is missing"
    data = path.read_bytes()
    assert data.isascii() and b"\r" not in data, f"{path} is not plain ASCII with LF line ends"
    return data.decode("ascii")


def code_of(text: str) -> str:
    """Return C++ `text` with each // comment removed (the skeletons hold no block comment and no // in a string)."""
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def test_the_manifest_names_the_skeleton_its_language_the_pin_and_every_file() -> None:
    document = yaml.safe_load(text_of(TREE / "MANIFEST.yaml"))
    assert (document["skeleton"], document["language"], document["pin"]) == (NAME, "ttkh", "tt-metal")
    assert document["commit"] == read_pin("tt-metal")["COMMIT"]
    assert [entry["path"] for entry in document["files"]] == ["host.cpp", "block.txt", *KERNELS]
    lines = ""
    for entry in document["files"]:
        data = (TREE / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest() and entry["bytes"] == len(data), entry["path"]
        lines += f"{entry['sha256']}  {entry['path']}\n"
    assert document["sha256"] == hashlib.sha256(lines.encode("ascii")).hexdigest()
    loaded = skeletons.load_skeleton(TREE)
    assert loaded.language == "ttkh" and [path for path, _ in loaded.kernels] == list(KERNELS)


def test_the_host_skeleton_is_the_one_that_built_at_the_pin() -> None:
    assert (TREE / "host.cpp").read_bytes() == (HOST_TREE / "host.cpp").read_bytes()
    digest = hashlib.sha256((TREE / "host.cpp").read_bytes()).hexdigest()
    assert digest == "5756f7dab5783acf29886a846fe644aa50bd33962fe1bd2135d02e88e0917515", "the measured host.cpp"


def test_the_kernels_lie_at_the_paths_the_host_skeleton_launches() -> None:
    launched = re.findall(r'"(kernels/[^"]+\.cpp)"', code_of(text_of(TREE / "host.cpp")))
    assert launched == list(KERNELS)


@pytest.mark.parametrize("path", KERNELS)
def test_each_kernel_includes_only_kernel_api_headers_and_defines_kernel_main(path: str) -> None:
    text = text_of(TREE / path)
    includes = INCLUDE.findall(code_of(text))
    assert includes, f"{path} includes nothing"
    for bracket, name in includes:
        assert (bracket, name) == ("<", "cstdint") or (bracket == '"' and name.startswith("api/")), name
    assert code_of(text).count("void kernel_main()") == 1
    assert "MAIN" not in code_of(text) and "namespace NAMESPACE" not in text


@pytest.mark.parametrize("path", KERNELS)
def test_each_kernel_makes_its_calls_in_the_pinned_form(path: str) -> None:
    code = code_of(text_of(TREE / path))
    for form in FORMS[path]:
        assert form in code, f"{path} does not write {form!r}"


@pytest.mark.parametrize("path", KERNELS)
def test_every_blank_has_an_end_and_holds_code(path: str) -> None:
    lines = text_of(TREE / path).splitlines()
    opened: tuple[str, int] | None = None
    found = []
    for index, line in enumerate(lines):
        start, end = BLANK.match(line), END.match(line)
        if start:
            assert opened is None, f"{path}:{index + 1}: a BLANK opens inside another"
            opened = (start.group(1), index)
        elif end:
            assert opened is not None and end.group(1) == opened[0], f"{path}:{index + 1}: END BLANK closes nothing"
            assert code_of("\n".join(lines[opened[1] + 1:index])).strip(), f"{path}: BLANK {opened[0]} is empty"
            found.append(opened[0])
            opened = None
    assert opened is None and found and len(found) == len(set(found)), found


def test_the_block_renders_both_fields_and_says_how_to_use_them() -> None:
    loaded = skeletons.load_skeleton(TREE)
    rendered = skeletons.block_text(loaded)
    assert text_of(TREE / "host.cpp") in rendered
    for path in KERNELS:
        assert f"// FILE: {path}\n{text_of(TREE / path)}" in rendered
    words = " ".join(text_of(TREE / "block.txt").split())
    for phrase in ("BLANK", "END BLANK", "exactly", "the path the prompt lists", "kernel_main"):
        assert phrase in words, phrase


@pytest.mark.parametrize("name", ["host.cpp", "block.txt", *KERNELS])
def test_no_tt_pairs_item_is_named(name: str) -> None:
    assert named_tier_a(text_of(TREE / name)) == [], f"{name} names a tt-pairs-v0 item or file (Agent Rule 5)"


def test_the_guard_reads_the_host_skeleton_with_its_kernels_as_no_host_compute() -> None:
    files = {"host.cpp": text_of(TREE / "host.cpp"), **{path: text_of(TREE / path) for path in KERNELS}}
    reading = read_host_compute(files, {"lassi_io.h": text_of(HARNESS)})
    assert reading.host_compute is False and reading.diagnostics == (), reading


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
        calls = set(CALL.findall(code_of(text_of(TREE / path)))) - NOT_CALLS
        missing = sorted(call for call in calls if call not in declared and call not in macros)
        assert not missing, f"{path} calls {missing}, which the pinned kernel API declares nowhere"
