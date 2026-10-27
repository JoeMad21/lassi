"""Tests for the kernels pack of tools/build_tt_context.py, tt-metalium-5280a9cf-kh (task P17.39).

Bible: Project Recipes (Notes, the Lookup bullet: the context packs the
tool builds), Design Principle 3, Agent Rules 5 and 10, Toolchain Pins
(tt-metal 5280a9cf). The owner's direction of 2026-10-26: the model writes
kernels as well as the host program, so a C++ -> TT prompt that asks for
kernels needs the kernel API at the pin beside the host API.

The contract these tests fix for `--kernels`:

- The pack serves the language ttkh: its entries are metalium_guide.ttkh,
  host_api.ttkh, and kernel_api.ttkh, in that order, and the manifest says
  language ttkh and records the whole pack's limit (the host part's limit
  and KERNEL_LIMIT_BYTES) and every header cut.
- The guide follows the default pack's rule but keeps the compute kernel
  sections (no section left out), without their code blocks, images, and
  the sentences that point at them.
- host_api.ttkh is the fuller pack's reference, built by the same rules
  from the same tree, head file, and template.
- kernel_api.ttkh holds declarations only, no comments and no conditional
  preprocessor lines, of every header under tt_metal/hw/inc/api at the pin
  (KERNEL_SOURCES, each with its sha256): first the kernel headers the
  tt-kh-v0 generate template includes, in its order (kernel_roots, read
  from the template at build time), then the rest in path order, filled
  whole to KERNEL_LIMIT_BYTES; each part starts with the include line
  `// ==== #include <api/...> ====` from the tracked kernel head file, so
  the lookup (lassi.core.stages pack_declarations) reads it as it reads a
  host part.
- Every fragment still passes the Agent Rule 5 check against the Tier A
  references, and no source is a programming example.
- Only the manifest is tracked; the fragments are gitignored.

Synthetic tests run the tool's parts on text written here. Tests that need
the pinned files skip unless LASSI_TT_METAL_TREE names a tree at the pin
holding the guide, the host headers, the kernel headers, and the Tier A
files. No value here is a measurement.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "build_tt_context.py"
KH_NAME = "tt-metalium-5280a9cf-kh"
KH_PACK = f"context/{KH_NAME}"
FULL_PACK = "context/tt-metalium-5280a9cf-full"
MANIFEST = "MANIFEST.yaml"
KERNEL_TEMPLATE = REPO / "assets" / "prompts" / "tt-kh-v0" / "generate.txt"
KERNEL_HEAD = REPO / "assets" / "context" / "tt-metalium-5280a9cf-kh.head.txt"
KEYS = ["metalium_guide.ttkh", "host_api.ttkh", "kernel_api.ttkh"]
# The kernel headers tt-kh-v0's generate.txt includes, in its order (the template's quoted "api/..." names).
ROOTS = ["api/dataflow/dataflow_api.h", "api/compute/common.h", "api/compute/tile_move_copy.h",
         "api/compute/eltwise_unary/eltwise_unary.h", "api/compute/matmul.h"]
COMPUTE_SECTIONS = ("### Register control and Data Flow within the Compute Kernels", "#### Compute APIs")


def load_tool() -> ModuleType:
    """Load tools/build_tt_context.py as a module."""
    spec = importlib.util.spec_from_file_location("build_tt_context_for_kh", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    """Return the tool loaded as a module."""
    return load_tool()


def run_tool(tool: ModuleType, *args: str) -> tuple[int, str]:
    """Run the tool's main(argv) in this process; return (exit status, everything it printed)."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        status = tool.main(list(args))
    return int(status or 0), sink.getvalue()


def pinned_tree() -> Path | None:
    """Return the tree LASSI_TT_METAL_TREE names when it holds the kernel headers too, else None."""
    named = os.environ.get("LASSI_TT_METAL_TREE", "")
    tree = Path(named) if named else None
    if tree is None or not (tree / "tt_metal" / "hw" / "inc" / "api" / "dataflow" / "dataflow_api.h").is_file():
        return None
    return tree


# ---------------------------------------------------------------- the parts, on text written here


def test_kernel_roots_are_the_api_includes_the_kh_template_names_in_order(tool: ModuleType) -> None:
    template = KERNEL_TEMPLATE.read_text(encoding="ascii")
    assert tool.kernel_roots(template) == [f"tt_metal/hw/inc/{name}" for name in ROOTS]
    assert tool.KERNEL_TEMPLATE_FILE == KERNEL_TEMPLATE and tool.KERNEL_HEAD_FILE == KERNEL_HEAD


def test_kernel_sources_are_the_kernel_api_headers_with_their_sha256(tool: ModuleType) -> None:
    sources = tool.KERNEL_SOURCES
    assert len(sources) >= 100 and all(path.startswith("tt_metal/hw/inc/api/") for path in sources)
    assert all(path.endswith(".h") and len(digest) == 64 for path, digest in sources.items())
    assert all(f"tt_metal/hw/inc/{name}" in sources for name in ROOTS)
    assert not any(path.startswith(tool.EXAMPLES) for path in sources), "no source is a programming example"


def test_kernel_order_puts_the_roots_first_then_the_rest_in_path_order(tool: ModuleType) -> None:
    paths = ["tt_metal/hw/inc/api/b.h", "tt_metal/hw/inc/api/a/z.h", "tt_metal/hw/inc/api/c.h"]
    order = tool.kernel_order(paths, ["tt_metal/hw/inc/api/c.h"])
    assert order == ["tt_metal/hw/inc/api/c.h", "tt_metal/hw/inc/api/a/z.h", "tt_metal/hw/inc/api/b.h"]


def test_a_kernel_part_holds_its_include_line_and_declarations_without_conditionals(tool: ModuleType) -> None:
    head = tool.read_head(KERNEL_HEAD)
    text = (
        "// SPDX-License-Identifier: SYNTHETIC\n#pragma once\n#include \"api/compute/common.h\"\n"
        "#ifdef TRISC_MATH\n#include \"llk_math.h\"\n#endif\n"
        "#define get_value(i) value_at<i>()\n"
        "namespace ckernel {\n/** SYNTHETIC doc comment */\n"
        "ALWI void synthetic_tile(uint32_t cb, uint32_t tile) {\n    MATH((llk_synthetic(cb, tile)));\n}\n}\n"
    )
    part = tool.kernel_part("tt_metal/hw/inc/api/compute/synthetic.h", text, head)
    assert part.startswith("// ==== #include <api/compute/synthetic.h> ====\n\n")
    assert "ALWI void synthetic_tile(uint32_t cb, uint32_t tile);" in part
    assert "#define get_value(i) value_at<i>()" in part, "a macro the kernel calls stays"
    for left_out in ("#ifdef", "#endif", '#include "', "#pragma", "SPDX", "doc comment", "llk_synthetic"):
        assert left_out not in part, left_out


def test_fit_kernels_keeps_the_roots_and_skips_a_header_that_does_not_fit(tool: ModuleType) -> None:
    head = tool.read_head(KERNEL_HEAD)
    small = "void small_call(int a);\n"
    headers = [("tt_metal/hw/inc/api/root.h", small), ("tt_metal/hw/inc/api/big.h", "void big(int a);\n" * 400),
               ("tt_metal/hw/inc/api/late.h", small)]
    fragment, cut = tool.fit_kernels(headers, head, limit=2048, keep=1)
    assert fragment.key == "kernel_api.ttkh" and cut == ("tt_metal/hw/inc/api/big.h",)
    assert "<api/root.h>" in fragment.text and "<api/late.h>" in fragment.text
    assert len(fragment.text.encode("utf-8")) <= 2048
    assert fragment.sources[:2] == (tool.KERNEL_HEAD_SOURCE, tool.KERNEL_TEMPLATE_SOURCE)


def test_the_kh_guide_keeps_the_compute_kernel_sections(tool: ModuleType) -> None:
    assert tool.KH_GUIDE_LEFT_OUT == () and tool.GUIDE_LEFT_OUT == (COMPUTE_SECTIONS[0],)
    for heading in tool.KH_GUIDE_DANGLING:
        assert heading in tool.GUIDE_DANGLING or heading == COMPUTE_SECTIONS[0]


def test_the_kernel_head_file_is_plain_ascii_and_names_the_include_form(tool: ModuleType) -> None:
    data = KERNEL_HEAD.read_bytes()
    assert data.isascii() and b"\r" not in data
    head = tool.read_head(KERNEL_HEAD)
    assert head.include_line == "// ==== #include <{include}> ===="
    for words in ("5280a9cf", "tt_metal/hw/inc/api", "kernel_main()", "Declarations only"):
        assert words in head.preamble, words


def test_main_builds_the_kernels_pack_for_ttkh(tool: ModuleType, tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def build_kernels(tree: Path, limit: int = tool.FULL_LIMIT_BYTES) -> object:
        calls.append(limit)
        fragments = tuple(tool.Fragment(key, (tool.GUIDE,), f"SYNTHETIC {key}\n") for key in KEYS)
        return tool.Pack(fragments, {}, ("tt_metal/hw/inc/api/cut.h",), limit + tool.KERNEL_LIMIT_BYTES, "ttkh")

    monkeypatch.setattr(tool, "build_kernels", build_kernels)
    monkeypatch.setattr(tool, "SOURCES", {tool.GUIDE: "1" * 64})
    out = tmp_path / "assets"
    status, output = run_tool(tool, "--tree", str(tmp_path), "--out", str(out), "--kernels")
    assert status == 0, output
    assert calls == [tool.FULL_LIMIT_BYTES], "the host part fills the fuller pack's limit"
    loaded = yaml.safe_load((out / KH_PACK / MANIFEST).read_text(encoding="ascii"))
    assert loaded["language"] == "ttkh" and [entry["key"] for entry in loaded["entries"]] == KEYS
    assert loaded["limit_bytes"] == tool.FULL_LIMIT_BYTES + tool.KERNEL_LIMIT_BYTES
    assert loaded["cut_by_limit"] == ["tt_metal/hw/inc/api/cut.h"]
    assert sorted(path.name for path in (out / KH_PACK).iterdir()) == sorted([MANIFEST, *(f"{k}.txt" for k in KEYS)])


def test_only_the_manifest_is_tracked() -> None:
    lines = (REPO / ".gitignore").read_bytes().decode("ascii").splitlines()
    for line in (f"/assets/{KH_PACK}/*", f"!/assets/{KH_PACK}/MANIFEST.yaml"):
        assert line in lines, f".gitignore lacks {line}"


# ---------------------------------------------------------------- the tracked manifest


def manifest() -> dict:
    """Return the tracked manifest of the kernels pack."""
    path = REPO / "assets" / KH_PACK / MANIFEST
    assert path.is_file(), f"{path} is missing; build it with uv run tools/build_tt_context.py --kernels --tree <tree>"
    return yaml.safe_load(path.read_bytes().decode("ascii"))


def test_the_tracked_manifest_serves_ttkh_with_three_entries_from_the_pin(tool: ModuleType) -> None:
    found = manifest()
    assert (found["generator"], found["commit"], found["language"]) == (
        "tools/build_tt_context.py", tool.PIN, "ttkh")
    assert [entry["key"] for entry in found["entries"]] == KEYS
    kernel = found["entries"][2]["sources"]
    assert [source["path"] for source in kernel[:2]] == [tool.KERNEL_HEAD_SOURCE, tool.KERNEL_TEMPLATE_SOURCE]
    headers = [source for source in kernel[2:]]
    assert [source["path"] for source in headers[:len(ROOTS)]] == [f"tt_metal/hw/inc/{name}" for name in ROOTS]
    for source in headers:
        assert source["sha256"] == tool.KERNEL_SOURCES[source["path"]], source["path"]
    full = yaml.safe_load((REPO / "assets" / FULL_PACK / MANIFEST).read_bytes().decode("ascii"))
    assert found["entries"][1]["sources"] == full["entries"][1]["sources"], "the host part is the fuller pack's"
    assert found["entries"][1]["sha256"] == full["entries"][1]["sha256"]


def test_the_tracked_kernel_head_and_template_hashes_are_current(tool: ModuleType) -> None:
    kernel = manifest()["entries"][2]["sources"]
    assert kernel[0]["sha256"] == hashlib.sha256(KERNEL_HEAD.read_bytes()).hexdigest(), "rebuild: the head changed"
    assert kernel[1]["sha256"] == hashlib.sha256(KERNEL_TEMPLATE.read_bytes()).hexdigest(), (
        "rebuild: tt-kh-v0's generate.txt changed")


# ---------------------------------------------------------------- with the pinned files


def test_a_fresh_build_matches_the_tracked_manifest(tool: ModuleType, tmp_path: Path) -> None:
    tree = pinned_tree()
    if tree is None:
        pytest.skip("needs LASSI_TT_METAL_TREE naming a tree at 5280a9cf with the guide, the host and kernel "
                    "headers, and the Tier A files")
    status, output = run_tool(tool, "--tree", str(tree), "--out", str(tmp_path), "--kernels")
    assert status == 0, output[-3000:]
    built = (tmp_path / KH_PACK / MANIFEST).read_bytes()
    assert built == (REPO / "assets" / KH_PACK / MANIFEST).read_bytes(), "regenerate the kernels pack's manifest"


def test_the_built_pack_declares_the_kernel_calls_the_prompt_names() -> None:
    texts = [REPO / "assets" / KH_PACK / f"{key}.txt" for key in KEYS]
    if not all(path.is_file() for path in texts):
        pytest.skip("the kernels pack is not built here; build it with uv run tools/build_tt_context.py --kernels")
    from lassi.core.stages import pack_declarations, pack_parts
    from lassi.prompts.assets import load_tree, pack_text

    pack = pack_text(KH_NAME, load_tree(REPO / "assets" / KH_PACK))
    names = {declaration.name for declaration in pack_declarations(pack)}
    for name in ("cb_wait_front", "cb_reserve_back", "get_write_ptr", "noc_async_read_page", "noc_async_write_page",
                 "noc_async_read_barrier", "tile_regs_acquire", "copy_tile", "pack_tile", "init_sfpu",
                 "TensorAccessor", "TensorAccessorArgs", "CreateKernel", "CircularBufferConfig"):
        assert name in names, name
    assert set(f"{name}" for name in ROOTS) <= set(pack_parts(pack))
    assert "tt-metalium/host_api.hpp" in pack_parts(pack)
