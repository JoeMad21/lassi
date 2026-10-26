"""Tests for tt-host-skeleton-v0, the item-agnostic C++ -> TT host program skeleton (OQ-096, option (a)).

Bible: Project Recipes (Notes, the Skeleton bullet), Harness Contract (lassi_io.h; the CPU -> TT guard),
Toolchain Pins (the tt-metal pin, 5280a9cf), Agent Rule 5 (no tt-pairs-v0 item's program enters a prompt). The
owner's answer to OQ-096 of 2026-10-25: the skeleton is item-agnostic only, written from the pinned headers and
tt-metal's programming examples outside the five tt-pairs-v0 items.

The contract these tests fix for assets/skeletons/tt-host-skeleton-v0/:

1. MANIFEST.yaml names the skeleton, its language (tt), the pin (tt-metal) and the pin's COMMIT, and lists
   host.cpp and block.txt, each with its sha256 and size in bytes; its sha256 is that of the lines
   "<sha256>  <path>\\n" of the files in order, and lassi.prompts.skeletons reads the skeleton only when every
   file matches it.
2. host.cpp is plain ASCII with LF line ends. Its includes are tt-metalium headers that the API pack
   assets/context/tt-metalium-5280a9cf-full names among its sources, and "lassi_io.h", nothing else.
3. Every `// BLANK <name>: <what to fill>` line has one later `// END BLANK <name>` line, the blocks neither nest
   nor repeat a name, and each holds code.
4. Every lassi_io_write precedes every lassi_io_free in its code, so no output is written from a freed input's dims.
5. It makes the calls the tt-host-v0 generate prompt names, DRAM MeshBuffers through one helper and circular
   buffers through one helper, and the CPU -> TT guard reads it as no host compute with a compute kernel.
6. block.txt is plain ASCII, renders with the one field `skeleton`, and tells the model to keep every call and
   fill only the BLANK blocks; neither file names a tt-pairs-v0 item, kernel, or file.

Nothing here compiles the skeleton: the compile check runs on the build host from a commit, against the pinned
tree (the ttmetal-host toolchain's command line). No value here is a measurement.
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

from test_template_context import named_tier_a  # noqa: E402

NAME = "tt-host-skeleton-v0"
TREE = REPO / "assets" / "skeletons" / NAME
HOST = TREE / "host.cpp"
BLOCK = TREE / "block.txt"
API_PACK = REPO / "assets" / "context" / "tt-metalium-5280a9cf-full" / "MANIFEST.yaml"
HARNESS = REPO / "assets" / "harness" / "c" / "lassi_io.h"
INCLUDE = re.compile(r"^\s*#\s*include\s*([<\"])([^>\"]+)[>\"]", re.MULTILINE)
BLANK = re.compile(r"^\s*// BLANK ([A-Za-z0-9_]+): (\S.*)$")
END = re.compile(r"^\s*// END BLANK ([A-Za-z0-9_]+)\s*$")
MARKER_STARTS = ("// BLANK", "// END BLANK")
# The calls the tt-host-v0 generate prompt names, each as the skeleton must write it.
CALLS = (
    "using namespace tt::tt_metal;",
    "distributed::MeshDevice::create_unit_mesh(0)",
    "->mesh_command_queue()",
    "distributed::MeshBuffer::create(",
    "distributed::ReplicatedBufferConfig",
    "distributed::DeviceLocalBufferConfig",
    "BufferType::DRAM",
    "distributed::EnqueueWriteMeshBuffer(",
    "CreateProgram()",
    "CircularBufferConfig(",
    ".set_page_size(",
    "tt::DataFormat::Float16_b",
    "CreateCircularBuffer(",
    "tt::CBIndex::c_0",
    "tt::CBIndex::c_16",
    "TensorAccessorArgs(*",
    ".append_to(",
    "DataMovementConfig{",
    "ComputeConfig{",
    "CreateKernel(",
    "SetRuntimeArgs(",
    "->address()",
    "CoreCoord core = {0, 0};",
    "distributed::MeshWorkload workload;",
    "workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));",
    "distributed::EnqueueMeshWorkload(",
    "distributed::Finish(",
    "distributed::EnqueueReadMeshBuffer(",
    "tilize_nfaces(",
    "untilize_nfaces(",
    "device->close();",
    "lassi_io_read(",
    "lassi_io_write(",
    "lassi_io_free(",
)


def text_of(path: Path) -> str:
    """Return a skeleton file as ASCII text; fail when it is missing or not plain ASCII with LF line ends."""
    assert path.is_file(), f"{path} is missing"
    data = path.read_bytes()
    assert data.isascii() and b"\r" not in data, f"{path} is not plain ASCII with LF line ends"
    return data.decode("ascii")


def code_of(text: str) -> str:
    """Return C++ `text` with each // comment removed (the skeleton holds no block comment and no // in a string)."""
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def manifest() -> dict:
    """Return the skeleton's MANIFEST.yaml."""
    document = yaml.safe_load(text_of(TREE / "MANIFEST.yaml"))
    assert isinstance(document, dict), "MANIFEST.yaml is not a mapping"
    return document


def sha256(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# The manifest (contract point 1)


def test_the_manifest_names_the_skeleton_its_language_and_the_pin() -> None:
    document = manifest()
    assert document["skeleton"] == NAME
    assert document["language"] == "tt"
    assert document["pin"] == "tt-metal"
    assert document["commit"] == read_pin("tt-metal")["COMMIT"], "the skeleton is written for the tt-metal pin"


def test_the_manifest_lists_each_file_with_its_sha256_and_size() -> None:
    document = manifest()
    assert [entry["path"] for entry in document["files"]] == ["host.cpp", "block.txt"]
    lines = ""
    for entry in document["files"]:
        data = (TREE / entry["path"]).read_bytes()
        assert entry["sha256"] == sha256(data), f"{entry['path']}: its sha256 in MANIFEST.yaml is stale"
        assert entry["bytes"] == len(data), f"{entry['path']}: its size in MANIFEST.yaml is stale"
        lines += f"{entry['sha256']}  {entry['path']}\n"
    assert document["sha256"] == sha256(lines.encode("ascii")), "the manifest's sha256 is not that of its entries"


def test_the_loader_reads_the_skeleton_with_the_manifest_sha256() -> None:
    assert skeletons.check_skeleton(TREE) == []
    loaded = skeletons.load_skeleton(TREE)
    assert loaded.name == NAME and loaded.language == "tt"
    assert loaded.sha256 == manifest()["sha256"]
    assert loaded.host == text_of(HOST) and loaded.block == text_of(BLOCK)


# ---------------------------------------------------------------------------
# host.cpp (contract points 2 to 5)


def pack_headers() -> set[str]:
    """Return the include names of the tt-metalium headers the API pack names among its sources."""
    document = yaml.safe_load(API_PACK.read_bytes().decode("ascii"))
    prefix = "tt_metal/api/"
    return {source["path"][len(prefix):] for entry in document["entries"] for source in entry["sources"]
            if source["path"].startswith(prefix + "tt-metalium/")}


def test_host_includes_only_pack_headers_and_the_harness_header() -> None:
    allowed = pack_headers()
    assert "tt-metalium/host_api.hpp" in allowed, "the API pack's manifest names its headers"
    includes = INCLUDE.findall(code_of(text_of(HOST)))
    assert includes, "host.cpp includes nothing"
    for bracket, name in includes:
        if bracket == '"':
            assert name == "lassi_io.h", f'host.cpp includes "{name}"; only "lassi_io.h" is quoted'
        else:
            assert name in allowed, f"host.cpp includes <{name}>, which the API pack does not name"
    assert ('"', "lassi_io.h") in includes


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


def test_every_blank_has_an_end_and_holds_code() -> None:
    text = text_of(HOST)
    found = blanks(text)
    assert len(found) >= 8, f"host.cpp marks only {len(found)} blanks"
    names = [name for name, _, _ in found]
    assert len(names) == len(set(names)), f"a blank name repeats: {names}"
    lines = text.splitlines()
    for name, start, end in found:
        body = code_of("\n".join(lines[start + 1:end])).strip()
        assert body, f"BLANK {name} holds no placeholder code"


def test_every_output_write_precedes_every_input_free() -> None:
    code = code_of(text_of(HOST))
    writes = [match.start() for match in re.finditer(r"\blassi_io_write\s*\(", code)]
    frees = [match.start() for match in re.finditer(r"\blassi_io_free\s*\(", code)]
    assert writes and frees, "host.cpp writes its outputs and frees its inputs through the harness header"
    assert max(writes) < min(frees), "a lassi_io_free comes before a lassi_io_write"


@pytest.mark.parametrize("call", CALLS)
def test_host_makes_the_call_in_the_pinned_form(call: str) -> None:
    assert call in code_of(text_of(HOST)), f"host.cpp does not write {call!r}"


@pytest.mark.parametrize("call", ["distributed::MeshBuffer::create(", "CircularBufferConfig(", "CreateCircularBuffer("])
def test_buffers_and_circular_buffers_go_through_one_helper(call: str) -> None:
    assert code_of(text_of(HOST)).count(call) == 1, f"{call!r} is written once, inside its helper"


def test_the_header_comment_says_item_agnostic_and_names_its_sources() -> None:
    head = text_of(HOST).split("#include", 1)[0]
    for words in ("item-agnostic", "5280a9cf", "add_2_integers_in_riscv", "shard_data_rm", "tt-metalium"):
        assert words in head, f"the header comment does not say {words!r}"


def test_the_guard_reads_no_host_compute_and_a_compute_kernel() -> None:
    reading = read_host_compute({"host.cpp": text_of(HOST)}, {"lassi_io.h": text_of(HARNESS)})
    assert reading.host_compute is False, reading
    assert reading.diagnostics == (), f"the guard tags or notes the skeleton: {reading.diagnostics}"


# ---------------------------------------------------------------------------
# block.txt and Agent Rule 5 (contract point 6)


def test_the_block_renders_with_the_skeleton_field_only() -> None:
    template = text_of(BLOCK)
    names = {match.group("named") or match.group("braced") for match in Template.pattern.finditer(template)
             if match.group("named") or match.group("braced")}
    assert names == {skeletons.BLOCK_FIELD}, f"block.txt uses the fields {sorted(names)}"
    rendered = skeletons.block_text(skeletons.load_skeleton(TREE))
    assert text_of(HOST) in rendered
    for words in ("BLANK", "END BLANK", "exactly"):
        assert words in rendered.replace(text_of(HOST), ""), f"block.txt does not say {words!r}"


@pytest.mark.parametrize("path", [HOST, BLOCK], ids=["host.cpp", "block.txt"])
def test_no_tt_pairs_item_is_named(path: Path) -> None:
    assert named_tier_a(text_of(path)) == [], f"{path.name} names a tt-pairs-v0 item or file (Agent Rule 5)"
