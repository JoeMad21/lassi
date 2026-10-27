"""Tests for tools/build_tt_context.py, the generator of the tt-metalium context pack (owner, 2026-10-23).

Bible: Design Principle 3 (context packs are versioned files under assets/),
Repository Layout (assets/context: generated text gitignored, only manifests
tracked, OQ-018), Agent Rule 5 (no Tier A test program enters a prompt, and
no content choice depends on a Tier A item or a result on one), Toolchain
Pins (tt-metal 5280a9cf).

The contract these tests fix: the tool reads a tt-metal tree at the pin (the
installed tree, or files fetched from GitHub at the commit, each at its tree
path), checks every source against the sha256 its SOURCES table records,
and writes assets/context/tt-metalium-5280a9cf/: MANIFEST.yaml and two
fragments. The guide is METALIUM_GUIDE.md cut by a rule on its headings
(the title and the GUIDE_PARTS parts, less GUIDE_LEFT_OUT), with no code
block and no worked example. The API reference holds declarations only, no
comments, of the ROOTS headers (the tt-metalium includes the tt-host-v0
generate template names, in its order), then of the headers that define
the names that template uses (read from it at build time: every identifier
of the template that a header of the walk defines, in the order of first
use), then of the rest of the tt-metalium headers the roots include,
transitively, in breadth-first include order from all roots together,
filled in that order to LIMIT_BYTES for the whole pack: a whole header that
does not fit the room left is skipped and the next one tried, and a root is
never skipped; its own text comes from the tracked head file, and the head
file and the template are its first sources.
It refuses, writing nothing, a source off the pin, a Tier A kernel, host
program, or walkthrough missing from the tree, a fragment holding a run of
RUN_LINES or more consecutive non-blank lines of any Tier A reference file,
a pack over LIMIT_BYTES, and a guide and roots that alone pass it.

Synthetic tests run the tool's parts on C++ and Markdown written here.
Tests that need the pinned tt-metal files skip unless LASSI_TT_METAL_TREE
names a tree at the pin or $LASSI_TOOLCHAINS/tt-metal@5280a9cf exists. No
value here is a measurement.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import os
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from lassi.toolchains.pins import read_pin

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "build_tt_context.py"
PACK = "context/tt-metalium-5280a9cf"
MANIFEST = "MANIFEST.yaml"
PIN = "5280a9cfb00998fd49667a29523d03aee905c129"
TIER_A = REPO / "assets" / "bench" / "tt-pairs-v0.yaml"
EXAMPLES = "tt_metal/programming_examples/"
API_DIR = "tt_metal/api/tt-metalium/"
# The tt-metalium includes assets/prompts/tt-host-v0/generate.txt tells the model a tt program uses, in its order.
ROOTS = tuple(f"{API_DIR}{name}" for name in
              ("host_api.hpp", "distributed.hpp", "tensor_accessor_args.hpp", "tilize_utils.hpp"))
GENERATE_TEMPLATE = REPO / "assets" / "prompts" / "tt-host-v0" / "generate.txt"
TEMPLATE_SOURCE = "assets/prompts/tt-host-v0/generate.txt"
HEAD_FILE = REPO / "assets" / "context" / "tt-metalium-5280a9cf.head.txt"
HEAD_SOURCE = "assets/context/tt-metalium-5280a9cf.head.txt"
# The names the template uses that a header of the walk defines at the pin, in the order of first use (the audit
# of 2026-10-24 read the same 13).
DERIVED_NAMES = ("MeshDevice", "MeshBuffer", "ReplicatedBufferConfig", "DeviceLocalBufferConfig", "BufferType",
                 "CircularBufferConfig", "DataFormat", "TensorAccessorArgs", "DataMovementConfig", "ComputeConfig",
                 "CoreCoord", "MeshWorkload", "MeshCoordinateRange")
LIMIT = 40 * 1024


def load_tool() -> ModuleType:
    """Load tools/build_tt_context.py as a module; fail clearly while it does not exist."""
    assert TOOL.is_file(), "tools/build_tt_context.py is missing"
    spec = importlib.util.spec_from_file_location("build_tt_context", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    """Return the tool loaded as a module."""
    return load_tool()


def pinned_tree() -> Path | None:
    """Return a tt-metal tree at the pin from LASSI_TT_METAL_TREE or $LASSI_TOOLCHAINS, or None."""
    named = os.environ.get("LASSI_TT_METAL_TREE", "")
    if named and Path(named).is_dir():
        return Path(named)
    toolchains = os.environ.get("LASSI_TOOLCHAINS", "")
    prefix = read_pin("tt-metal")["PREFIX_NAME"]  # the pinned install (the Tracy build since task P17.42)
    installed = Path(toolchains) / prefix if toolchains else None
    return installed if installed is not None and installed.is_dir() else None


TREE_SKIP = "needs a tt-metal tree at 5280a9cf: set LASSI_TT_METAL_TREE, or install it under $LASSI_TOOLCHAINS"


def run_tool(tool: ModuleType, *args: str) -> tuple[int, str]:
    """Run the tool's main(argv) in this process; return (exit status, everything it printed)."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        status = tool.main(list(args))
    return int(status or 0), sink.getvalue()


@pytest.fixture(scope="module")
def fresh(tool: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return an assets root holding a fresh pack built from the pinned tree; skip without one."""
    tree = pinned_tree()
    if tree is None:
        pytest.skip(TREE_SKIP)
    out = tmp_path_factory.mktemp("assets")
    status, output = run_tool(tool, "--tree", str(tree), "--out", str(out))
    assert status == 0, f"the tool exited {status}: {output[-3000:]}"
    return out


def fresh_manifest(root: Path) -> dict:
    """Return the loaded MANIFEST.yaml of the pack under `root`."""
    return yaml.safe_load((root / PACK / MANIFEST).read_text(encoding="ascii"))


def fragment_texts(root: Path) -> dict[str, str]:
    """Return {key: text} of every fragment the pack's manifest under `root` lists."""
    tree = root / PACK
    return {entry["key"]: (tree / f"{entry['key']}.txt").read_bytes().decode("utf-8")
            for entry in fresh_manifest(root)["entries"]}


def bfs_includes(tree: Path, roots: tuple[str, ...] = ROOTS) -> list[str]:
    """Return the tt-metalium include closure of `roots` in `tree`, breadth first, read independently of the tool."""
    order, queue, seen = [], list(roots), set(roots)
    while queue:
        path = queue.pop(0)
        order.append(path)
        text = (tree / path).read_bytes().decode("utf-8")
        for line in text.split("\n"):
            found = re.match(r'\s*#\s*include\s*[<"](tt-metalium/[^>"]+)[>"]', line)
            target = f"tt_metal/api/{found.group(1)}" if found else None
            if target is not None and target not in seen:
                assert (tree / target).is_file(), f"{path} includes {target}, which the tree lacks"
                seen.add(target)
                queue.append(target)
    return order


# ---------------------------------------------------------------- function bodies


def test_strip_bodies_replaces_function_bodies_and_keeps_scopes(tool: ModuleType) -> None:
    source = (
        "namespace a::b {\n"
        "struct S {\n"
        "    int f(int x) const { return x + 1; }\n"
        "    void g();\n"
        "};\n"
        "enum class E : int { A = 1, B };\n"
        "template <typename T>\n"
        "void h(std::vector<T>& v, bool blocking = false) {\n"
        "    if (v.empty()) { return; }\n"
        "}\n"
        "}  // namespace a::b\n"
    )
    got = tool.strip_bodies(source)
    assert "return" not in got, got
    assert "int f(int x) const;" in got
    assert "void h(std::vector<T>& v, bool blocking = false);" in got
    for kept in ("namespace a::b {", "struct S {", "enum class E : int { A = 1, B };", "}  // namespace a::b"):
        assert kept in got, f"{kept!r} is missing from {got!r}"


def test_strip_bodies_drops_init_lists_and_keeps_member_and_default_initializers(tool: ModuleType) -> None:
    source = (
        "class C {\n"
        "public:\n"
        "    C(int x) : x_(x), y_{2}, z_{} { setup(); }\n"
        "    void run(const Config& c = Config{}, std::vector<int> v = {}) { go(c, v); }\n"
        "    int x_ = 0;\n"
        "    std::vector<int> y_{1, 2};\n"
        "    CoreCoord core = {0, 0};\n"
        "};\n"
    )
    got = tool.strip_bodies(source)
    assert "C(int x);" in got, got
    assert "setup" not in got and "go(" not in got, got
    assert "void run(const Config& c = Config{}, std::vector<int> v = {});" in got
    for kept in ("int x_ = 0;", "std::vector<int> y_{1, 2};", "CoreCoord core = {0, 0};"):
        assert kept in got, f"{kept!r} is missing from {got!r}"


def test_strip_bodies_strips_operator_bodies(tool: ModuleType) -> None:
    source = (
        "constexpr bool operator<=(const P& a, const P& b) { return (a < b) or (a == b); }\n"
        "struct Q {\n"
        "    Q& operator=(const Q& other) { copy(other); return *this; }\n"
        "    bool operator==(const Q& other) const = default;\n"
        "};\n"
    )
    got = tool.strip_bodies(source)
    assert "return" not in got, got
    assert "constexpr bool operator<=(const P& a, const P& b);" in got
    assert "Q& operator=(const Q& other);" in got
    assert "bool operator==(const Q& other) const = default;" in got


def test_strip_bodies_ignores_braces_in_comments_and_strings(tool: ModuleType) -> None:
    source = (
        "// a comment with { an open brace\n"
        "/* and a block comment } with a close one */\n"
        'inline const char* name() { return "}{"; }\n'
        "void after();\n"
    )
    got = tool.strip_bodies(source)
    assert "inline const char* name();" in got, got
    assert "void after();" in got
    assert "// a comment with { an open brace" in got


# ---------------------------------------------------------------- declarations


def test_declarations_keep_public_members_only_and_drop_deprecated_friend_forward_and_specializations(
    tool: ModuleType,
) -> None:
    # An override stays: the reference may hold a derived class without its base (MeshDevice without IDevice), so
    # a derived class's overrides are the only place its callers read those members.
    source = (
        "namespace n {\n"
        "class Hidden;\n"
        "class Base {\n"
        "public:\n"
        "    virtual bool close() = 0;\n"
        "};\n"
        "class Device : public Base {\n"
        "    friend class Impl;\n"
        "    int secret_;\n"
        "protected:\n"
        "    Device(int id);\n"
        "public:\n"
        "    // Opens the device.\n"
        "    static Device open(int id);\n"
        "    bool close() override;\n"
        '    [[deprecated("use open")]]\n'
        "    static Device make(int id);\n"
        '    struct [[deprecated("use Spec")]] OldSpec {\n'
        "        int size;\n"
        "    };\n"
        "private:\n"
        "    void helper();\n"
        "};\n"
        "template <>\n"
        "int convert<int>(int x);\n"
        "// Internal helpers.\n"
        "namespace detail {\n"
        "int internal_helper();\n"
        "}  // namespace detail\n"
        "}  // namespace n\n"
        "template <>\n"
        "struct fmt::formatter<n::Device> {\n"
        "    auto format(const n::Device& d, format_context& ctx) const -> format_context::iterator;\n"
        "};\n"
    )
    got = tool.declarations(source)
    for gone in ("Hidden", "friend", "secret_", "Device(int id);", "protected:", "helper", "make(", "OldSpec",
                 "private:", "convert", "formatter", "template <>", "detail", "internal_helper", "Internal helpers"):
        assert gone not in got, f"{gone!r} should be left out of {got!r}"
    for kept in ("virtual bool close() = 0;", "// Opens the device.", "static Device open(int id);", "public:",
                 "bool close() override;"):
        assert kept in got, f"{kept!r} is missing from {got!r}"


def test_declarations_drop_includes_pragmas_and_license_lines(tool: ModuleType) -> None:
    source = (
        "// SPDX-FileCopyrightText: (c) 2025 Example\n"
        "//\n"
        "// SPDX-License-Identifier: Apache-2.0\n"
        "\n"
        "#pragma once\n"
        "#include <vector>\n"
        "\n"
        "/** \\mainpage Front page\n"
        " * Welcome.\n"
        " * */\n"
        "\n"
        "// clang-format off\n"
        "/**\n"
        " * Makes a thing.\n"
        " *\n"
        " * | Argument | Description                  | Type | Valid Range | Required |\n"
        " * |----------|------------------------------|------|-------------|----------|\n"
        " * | id       | The id of the thing          | int  | 0 to 7      | Yes      |\n"
        " * | flag     | An extra column follows      | bool |             | No       | |\n"
        " */\n"
        "// clang-format on\n"
        "int make(int id, bool flag = false);\n"
    )
    got = tool.declarations(source)
    for gone in ("SPDX", "#pragma", "#include", "clang-format", "mainpage", "Welcome"):
        assert gone not in got, f"{gone!r} should be left out of {got!r}"
    assert " * Makes a thing." in got
    assert " * | Argument | Description | Valid Range |\n" in got, "the Type and Required columns are left out"
    assert " * | id | The id of the thing | 0 to 7 |\n" in got, "table cells keep single spaces"
    assert " * | flag | An extra column follows | |\n" in got, "a cell past the header row's columns is left out"
    assert "|---" not in got, "a table's separator row carries nothing and is left out"
    assert got.endswith("int make(int id, bool flag = false);\n")


def test_declarations_drop_empty_scopes_but_keep_enumerators(tool: ModuleType) -> None:
    source = (
        "namespace n {\n"
        "class OnlyForward;\n"
        "}  // namespace n\n"
        "namespace m {\n"
        "enum class Kind { A, B };\n"
        "}  // namespace m\n"
    )
    got = tool.declarations(source)
    assert "namespace n" not in got, got
    assert "enum class Kind { A, B };" in got and "namespace m {" in got


def test_declarations_without_comments_keep_the_code_and_drop_every_comment(tool: ModuleType) -> None:
    source = (
        "namespace n {\n"
        "// ==== a banner ====\n"
        "/**\n"
        " * Opens a thing. | Argument | Description |\n"
        " */\n"
        "int open(int id /* the id */, const char* name = \"// not a comment\");  // trailing\n"
        "\n\n\n"
        "struct Spec {\n"
        "    int size;  ///< the size\n"
        "};\n"
        "}  // namespace n\n"
    )
    assert tool.declarations(source, comments=False) == (
        "namespace n {\n\n"
        "int open(int id , const char* name = \"// not a comment\");\n\n"
        "struct Spec {\n    int size;\n};\n}\n"
    ), repr(tool.declarations(source, comments=False))
    assert "Opens a thing." in tool.declarations(source), "with comments (the default) the doc comments stay"


# ---------------------------------------------------------------- the include closure


def test_include_closure_is_breadth_first_in_include_order_within_tt_metalium(tool: ModuleType) -> None:
    texts = {
        f"{API_DIR}root.hpp": ("#pragma once\n#include <vector>\n#include <tt-metalium/b.hpp>\n"
                               "#include <tt_stl/span.hpp>\n#include <tt-metalium/c.hpp>\n"),
        f"{API_DIR}b.hpp": "#include <tt-metalium/d.hpp>\n#include <tt-metalium/c.hpp>\n#include <umd/x.hpp>\n",
        f"{API_DIR}c.hpp": "#include <tt-metalium/b.hpp>\n#include <tt-metalium/sub/e.h>\n",
        f"{API_DIR}d.hpp": "int d();\n",
        f"{API_DIR}sub/e.h": "#include <tt-metalium/root.hpp>\n",
        f"{API_DIR}unreached.hpp": "int u();\n",
    }
    got = tool.include_closure(texts, roots=(f"{API_DIR}root.hpp",))
    assert got == [f"{API_DIR}{name}" for name in ("root.hpp", "b.hpp", "c.hpp", "d.hpp", "sub/e.h")], got


def test_include_closure_walks_from_all_roots_together(tool: ModuleType) -> None:
    # Every root comes first, in order, and then the includes level by level: r2's include x comes after r1's
    # include w though r1's own includes reach deeper (w includes y), because y is one level further down.
    texts = {
        f"{API_DIR}r1.hpp": "#include <tt-metalium/w.hpp>\n#include <tt-metalium/r2.hpp>\n",
        f"{API_DIR}r2.hpp": "#include <tt-metalium/x.hpp>\n#include <tt-metalium/w.hpp>\n",
        f"{API_DIR}r3.hpp": "int r3();\n",
        f"{API_DIR}w.hpp": "#include <tt-metalium/y.hpp>\n",
        f"{API_DIR}x.hpp": "int x();\n",
        f"{API_DIR}y.hpp": "int y();\n",
    }
    roots = tuple(f"{API_DIR}{name}" for name in ("r1.hpp", "r2.hpp", "r3.hpp"))
    got = tool.include_closure(texts, roots=roots)
    assert got == [f"{API_DIR}{name}" for name in ("r1.hpp", "r2.hpp", "r3.hpp", "w.hpp", "x.hpp", "y.hpp")], got


def test_include_closure_refuses_an_include_that_sources_do_not_record(tool: ModuleType) -> None:
    texts = {f"{API_DIR}root.hpp": "#include <tt-metalium/missing.hpp>\n"}
    with pytest.raises(tool.Refusal, match="missing.hpp"):
        tool.include_closure(texts, roots=(f"{API_DIR}root.hpp",))
    with pytest.raises(tool.Refusal, match="root header .*other.hpp"):
        tool.include_closure(texts, roots=(f"{API_DIR}root.hpp", f"{API_DIR}other.hpp"))


def test_roots_are_the_includes_the_generate_template_names_in_order(tool: ModuleType) -> None:
    # The roots come from what the prompt set already shows the model, not from Tier A (Agent Rule 5).
    named = re.findall(r"<(tt-metalium/[^>]+)>", GENERATE_TEMPLATE.read_bytes().decode("utf-8"))
    assert named, f"{GENERATE_TEMPLATE} names no tt-metalium include"
    assert [f"tt_metal/api/{name}" for name in dict.fromkeys(named)] == list(tool.ROOTS) == list(ROOTS)


def test_sources_are_the_guide_then_tt_metalium_headers_from_the_roots(tool: ModuleType) -> None:
    assert tool.ROOTS == ROOTS and tool.API_DIR == API_DIR
    paths = list(tool.SOURCES)
    assert paths[0] == tool.GUIDE == "METALIUM_GUIDE.md"
    assert paths[1:1 + len(ROOTS)] == list(ROOTS), "the include order starts at the roots, in their order"
    outside = [path for path in paths[1:] if not path.startswith(API_DIR)]
    assert not outside, f"sources outside tt-metalium: {outside}"
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in tool.SOURCES.values())
    assert not hasattr(tool, "HEADERS"), "the headers follow from the include rule, not from a list"


def test_fit_reference_keeps_every_header_that_fits(tool: ModuleType) -> None:
    head = tool.Head(preamble="# Head\n", include_line="// <{include}>", sha256="0" * 64)
    guide = tool.Fragment(key="metalium_guide.tt", sources=("METALIUM_GUIDE.md",), text="G\n")
    headers = [(f"{API_DIR}a.hpp", "int a();\n"), (f"{API_DIR}b.hpp", "int b();\n"),
               (f"{API_DIR}c.hpp", "int c();\n")]
    whole, cut = tool.fit_reference(guide, headers, head, limit=10_000)
    assert cut == () and whole.text == (
        "# Head\n\n// <tt-metalium/a.hpp>\n\nint a();\n\n// <tt-metalium/b.hpp>\n\nint b();\n\n"
        "// <tt-metalium/c.hpp>\n\nint c();\n"
    ), repr(whole.text)
    assert whole.sources == (HEAD_SOURCE, TEMPLATE_SOURCE, *(path for path, _ in headers))
    two = tool.pack_size([guide, tool.fit_reference(guide, headers[:2], head, limit=10_000)[0]])
    fragment, cut = tool.fit_reference(guide, headers, head, limit=two)
    assert cut == (f"{API_DIR}c.hpp",) and fragment.sources[-1] == f"{API_DIR}b.hpp"
    with pytest.raises(tool.Refusal, match="limit"):
        tool.fit_reference(guide, headers, head, limit=8)
    # The roots are never cut: when the guide and the first `keep` headers pass the limit, the build is refused,
    # naming the sizes, rather than dropping a root.
    with pytest.raises(tool.Refusal, match=rf"guide \(2 bytes\) and the first 3 headers .* over the {two}-byte limit"):
        tool.fit_reference(guide, headers, head, limit=two, keep=3)
    assert tool.fit_reference(guide, headers, head, limit=two, keep=2)[1] == (f"{API_DIR}c.hpp",)


def test_fit_reference_skips_a_header_that_does_not_fit_and_keeps_a_later_one_that_does(tool: ModuleType) -> None:
    # The walk root, big, small, tail: big does not fit in the room the root leaves, so it is skipped, and the walk
    # goes on: small fits and is kept, and tail, which no longer fits after small, is skipped too.
    head = tool.Head(preamble="# Head\n", include_line="// <{include}>", sha256="0" * 64)
    guide = tool.Fragment(key="metalium_guide.tt", sources=("METALIUM_GUIDE.md",), text="G\n")
    headers = [(f"{API_DIR}root.hpp", "int root();\n"), (f"{API_DIR}big.hpp", "int big();\n" * 50),
               (f"{API_DIR}small.hpp", "int s();\n"), (f"{API_DIR}tail.hpp", "int tail_header();\n")]
    with_small = tool.fit_reference(guide, [headers[0], headers[2]], head, limit=10_000)[0]
    limit = tool.pack_size([guide, with_small])
    fragment, skipped = tool.fit_reference(guide, headers, head, limit=limit, keep=1)
    assert fragment.sources == (HEAD_SOURCE, TEMPLATE_SOURCE, f"{API_DIR}root.hpp", f"{API_DIR}small.hpp"), \
        fragment.sources
    assert skipped == (f"{API_DIR}big.hpp", f"{API_DIR}tail.hpp")
    assert fragment.text == with_small.text and tool.pack_size([guide, fragment]) <= limit
    # A root is never skipped, however big: the roots that do not fit refuse the build instead.
    with pytest.raises(tool.Refusal, match="first 2 headers"):
        tool.fit_reference(guide, headers, head, limit=limit, keep=2)


def test_fit_reference_gives_every_header_as_declarations_only_the_roots_included(tool: ModuleType) -> None:
    head = tool.Head(preamble="# Head\n", include_line="// <{include}>", sha256="0" * 64)
    guide = tool.Fragment(key="metalium_guide.tt", sources=("METALIUM_GUIDE.md",), text="G\n")
    headers = [(f"{API_DIR}r1.hpp", "// Root one.\nint r1();\n"), (f"{API_DIR}r2.hpp", "/** Root two. */\nint r2();\n"),
               (f"{API_DIR}later.hpp", "// A later header.\nint later();\n")]
    fragment, skipped = tool.fit_reference(guide, headers, head, limit=10_000, keep=2)
    assert skipped == ()
    assert fragment.text == (
        "# Head\n\n// <tt-metalium/r1.hpp>\n\nint r1();\n\n// <tt-metalium/r2.hpp>\n\nint r2();\n\n"
        "// <tt-metalium/later.hpp>\n\nint later();\n"
    ), repr(fragment.text)
    assert fragment.text == tool.api_text(headers, head)
    for comment in ("Root one", "Root two", "A later header"):
        assert comment not in fragment.text, f"{comment!r}: no header keeps its comments, a root included"


def test_defines_finds_a_definition_and_not_a_forward_declaration_or_a_mention(tool: ModuleType) -> None:
    assert tool.defines("class MeshDevice : public IDevice {\n};\n", "MeshDevice")
    assert tool.defines("enum class BufferType : uint8_t { DRAM, L1 };\n", "BufferType")
    assert tool.defines("struct [[nodiscard]] Spec {\n};\n", "Spec")
    assert tool.defines("using KernelHandle = std::uint32_t;\n", "KernelHandle")
    assert not tool.defines("class MeshDevice;\n", "MeshDevice")
    assert not tool.defines("void open(MeshDevice& device);\n", "MeshDevice")
    assert not tool.defines("// class MeshDevice {\n", "MeshDevice"), "a comment defines nothing"
    assert not tool.defines("struct MeshDeviceConfig {\n};\n", "MeshDevice")
    assert not tool.defines("struct Outer::Inner {\n};\n", "Outer")
    assert not tool.defines("enum class DataFormat : uint8_t;\n", "DataFormat"), "an opaque enum is no definition"


def test_walk_order_puts_the_headers_that_define_the_prompt_names_after_the_roots(tool: ModuleType) -> None:
    # Each definition has a member, as the declarations leave out a scope that holds nothing.
    texts = {
        f"{API_DIR}root.hpp": "void run(Config c, Device d);\n",
        f"{API_DIR}a.hpp": "class Device;\nint a();\n",
        f"{API_DIR}b.hpp": "int b();\n",
        f"{API_DIR}device.hpp": "class Device {\npublic:\n    int id();\n};\n",
        f"{API_DIR}config.hpp": "struct Config {\n    int x;\n};\nstruct Other {\n    int y;\n};\n",
        f"{API_DIR}again.hpp": "struct Config {\n    int z;\n};\n",
    }
    declared = tool.declared_headers(texts, list(texts))
    assert list(declared) == list(texts) and "class Device;" not in declared[f"{API_DIR}a.hpp"]
    root = (f"{API_DIR}root.hpp",)
    order, chosen = tool.walk_order(declared, roots=root, names=("Config", "Device", "Other"))
    assert chosen == {"Config": f"{API_DIR}config.hpp", "Device": f"{API_DIR}device.hpp",
                      "Other": f"{API_DIR}config.hpp"}, "the first header of the walk that defines each name"
    assert order == [f"{API_DIR}{name}" for name in
                     ("root.hpp", "config.hpp", "device.hpp", "a.hpp", "b.hpp", "again.hpp")], order
    assert tool.walk_order(declared, roots=root, names=())[0] == list(texts)
    with pytest.raises(tool.Refusal, match="Missing"):
        tool.walk_order(declared, roots=root, names=("Missing",))


def test_prompt_names_are_the_template_identifiers_a_header_defines_in_order_of_first_use(tool: ModuleType) -> None:
    template = ("Open a Device with open_device(), then a Config{.size}; a Device again, then Missing, "
                "Alias, and 32x32 tiles; device::Config is named twice.\n")
    declared = {f"{API_DIR}a.hpp": "struct Config {\n    int size;\n};\nenum class Missing : int;\n",
                f"{API_DIR}b.hpp": "class Device : public Base {\npublic:\n    int id();\n};\nusing Alias = int;\n"}
    assert tool.prompt_names(template, declared) == ["Device", "Config", "Alias"]
    assert tool.prompt_names("Nothing here is defined.\n", declared) == []


def derived_names(template: str, tree: Path, closure: list[str]) -> list[str]:
    """Return, read independently of the tool, the template's identifiers some walk header defines, by first use.

    A header defines a name when its code, comments cut, holds a class,
    struct, union, or enum of that name with a body, or a `using name =`;
    `struct std::hash<...> {` specializes a member of std and defines no std.
    """
    codes = [re.sub(r"//[^\n]*|/\*.*?\*/", "", (tree / path).read_bytes().decode("utf-8"), flags=re.S)
             for path in closure]
    found = []
    for name in dict.fromkeys(re.findall(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*", template)):
        pattern = re.compile(rf"\b(class|struct|union|enum( class| struct)?)\s+{name}\b(?!\s*::)[^;{{()]*\{{"
                             rf"|\busing\s+{name}\s*=")
        if any(pattern.search(code) for code in codes):
            found.append(name)
    return found


def test_the_derived_prompt_names_match_an_independent_reading_of_the_template(tool: ModuleType) -> None:
    tree = pinned_tree()
    if tree is None:
        pytest.skip(TREE_SKIP)
    template = GENERATE_TEMPLATE.read_bytes().decode("utf-8")
    assert tool.TEMPLATE_FILE == GENERATE_TEMPLATE and tool.TEMPLATE_SOURCE == TEMPLATE_SOURCE
    closure = bfs_includes(tree)
    declared = tool.declared_headers(tool.read_sources(tree), closure)
    assert tool.prompt_names(template, declared) == derived_names(template, tree, closure) == list(DERIVED_NAMES)
    assert not hasattr(tool, "PROMPT_NAMES"), "the names are read from the template, not from a hand list"


def test_the_header_chosen_for_each_prompt_name_defines_it(tool: ModuleType) -> None:
    tree = pinned_tree()
    if tree is None:
        pytest.skip(TREE_SKIP)
    closure = bfs_includes(tree)
    declared = tool.declared_headers(tool.read_sources(tree), closure)
    names = tool.prompt_names(GENERATE_TEMPLATE.read_bytes().decode("utf-8"), declared)
    order, chosen = tool.walk_order(declared, names=names)
    assert list(chosen) == names
    for name, path in chosen.items():
        definition = re.compile(rf"\b(class|struct|enum class)\s+{name}\b[^;{{]*\{{|\busing\s+{name}\s*=")
        assert definition.search((tree / path).read_bytes().decode("utf-8")), f"{path} does not define {name}"
        earlier = [other for other in closure[:closure.index(path)]
                   if definition.search((tree / other).read_bytes().decode("utf-8"))]
        assert not earlier, f"{name} is defined earlier in the walk, in {earlier}"
    first = list(dict.fromkeys([*ROOTS, *(chosen[name] for name in names)]))
    assert order[:len(first)] == first and sorted(order) == sorted(closure)


# ---------------------------------------------------------------- the head file


def test_read_head_splits_the_preamble_from_the_include_line(tool: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "head.txt"
    path.write_bytes(b"# Title\n\nSome text.\n// == {include} ==\n")
    head = tool.read_head(path)
    assert head.preamble == "# Title\n\nSome text.\n" and head.include_line == "// == {include} =="
    assert head.line("tt-metalium/a.hpp") == "// == tt-metalium/a.hpp =="
    assert head.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("data", [
    b"# Title\nno slot\n",
    b"// {include}\n",
    b"# Title\n// {include}\ntrailing text\n",
    b"# Title\n// {include} and {include}\n",
    b"# Title\r\n// {include}\r\n",
    b"# Title\n// {include}",
    b"# Title " + bytes([0xC3, 0xA9]) + b"\n// {include}\n",
], ids=["no-slot", "no-preamble", "text-after", "two-slots", "crlf", "no-final-newline", "not-ascii"])
def test_read_head_refuses_a_malformed_head(tool: ModuleType, tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "head.txt"
    path.write_bytes(data)
    with pytest.raises(tool.Refusal, match="head file"):
        tool.read_head(path)


def test_the_committed_head_file_is_the_tools_head(tool: ModuleType) -> None:
    assert tool.HEAD_FILE == HEAD_FILE and tool.HEAD_SOURCE == HEAD_SOURCE
    head = tool.read_head()
    assert head.preamble.startswith("# TT-Metalium host API reference (tt-metal 5280a9cf)\n")
    assert head.line("tt-metalium/host_api.hpp") == "// ==== #include <tt-metalium/host_api.hpp> ===="
    source = TOOL.read_text(encoding="utf-8")
    assert "TT-Metalium host API reference" not in source and "// ====" not in source, \
        "the pack's own prose lives in the head file, not in the tool (Design Principle 3)"


# ---------------------------------------------------------------- the guide


GUIDE_SAMPLE = (
    "# Title\n\n"
    "## Table of Contents\n\n- [Kept](#kept)\n\n"
    "## Kept\n\nFirst paragraph. The text shown above is gone.\n\n<img src=\"x.webp\">\n"
    "Second paragraph, right after an image.\n\n"
    "```c++\nint code();\n```\n\nThird paragraph.\n\nFourth paragraph. For example:\n\n```c++\nint e();\n```\n\n"
    "### Kept child\n\nChild text.\n\n"
    "### Left out\n\nInternals.\n\n#### Under left out\n\nDeeper internals.\n\n"
    "### Kept after\n\nAfter text.\n\n"
    "## Dropped\n\nNot wanted.\n\n### Dropped child\n\nNot wanted either.\n"
)


def test_guide_keeps_its_parts_by_heading_and_drops_code_images_cuts_and_dangling_sentences(
    tool: ModuleType,
) -> None:
    # The blocks of "## Kept", numbered from 0: the first paragraph, the image, the second paragraph, the code
    # block, the third paragraph, the fourth, and its code block; the cut (4, 4) is the third paragraph.
    got = tool.guide_text(
        GUIDE_SAMPLE, parts=("## Kept",), left_out=("### Left out",), cuts={"## Kept": ((4, 4),)},
        dangling={"## Kept": ("The text shown above is gone.", "For example:")},
    )
    assert got == (
        "# Title\n\n## Kept\n\nFirst paragraph.\n\nSecond paragraph, right after an image.\n\n"
        "Fourth paragraph.\n\n### Kept child\n\nChild text.\n\n### Kept after\n\nAfter text.\n"
    ), repr(got)


def test_kept_headings_follow_the_part_rule(tool: ModuleType) -> None:
    headings = [heading for heading, _ in tool.guide_blocks(GUIDE_SAMPLE)]
    assert tool.kept_headings(headings, ("## Kept",), ("### Left out",)) == [
        "# Title", "## Kept", "### Kept child", "### Kept after"]


def test_guide_refuses_a_layout_it_does_not_expect(tool: ModuleType) -> None:
    with pytest.raises(tool.Refusal, match="## Missing"):
        tool.guide_text("# Title\n\nText.\n", parts=("## Missing",), left_out=(), cuts={}, dangling={})
    with pytest.raises(tool.Refusal, match="## Kept"):
        tool.guide_text("## Kept\n\nOne.\n", parts=("## Kept",), left_out=(), cuts={"## Kept": ((0, 4),)},
                        dangling={})
    with pytest.raises(tool.Refusal, match="not a section"):
        tool.guide_text(GUIDE_SAMPLE, parts=("## Kept",), left_out=(), cuts={"## Dropped": ((0, 0),)}, dangling={})
    with pytest.raises(tool.Refusal, match="Absent sentence"):
        tool.guide_text(GUIDE_SAMPLE, parts=("## Kept",), left_out=(), cuts={},
                        dangling={"## Kept": ("Absent sentence.",)})


def test_the_guide_rule_names_no_tier_a_item(tool: ModuleType) -> None:
    items = set(yaml.safe_load(TIER_A.read_text(encoding="utf-8"))["items"])
    named = [*tool.GUIDE_PARTS, *tool.GUIDE_LEFT_OUT, *tool.GUIDE_DANGLING,
             *(sentence for sentences in tool.GUIDE_DANGLING.values() for sentence in sentences)]
    assert not [text for text in named for item in items if item in text.lower().replace(" ", "_")], named


# ---------------------------------------------------------------- Agent Rule 5


def test_tier_a_runs_find_a_three_line_run_whatever_the_indentation(tool: ModuleType) -> None:
    reference = "void kernel_main() {\n    uint32_t a = get_arg_val<uint32_t>(0);\n\n    cb_wait_front(cb, 1);\n}\n"
    fragment = "Intro.\nvoid kernel_main() {\nuint32_t a = get_arg_val<uint32_t>(0);\n        cb_wait_front(cb, 1);\n"
    hits = tool.tier_a_runs(fragment, {"ref.cpp": reference})
    assert hits and "ref.cpp" in hits[0], hits


def test_tier_a_runs_ignore_two_line_runs_and_unrelated_text(tool: ModuleType) -> None:
    reference = "line one;\nline two;\nline three;\n"
    assert tool.tier_a_runs("line one;\nline two;\nsomething else;\n", {"ref.cpp": reference}) == []
    assert tool.tier_a_runs("line two;\nline one;\nline three;\n", {"ref.cpp": reference}) == []


def test_check_pack_refuses_a_tier_a_run_and_a_pack_over_the_limit(tool: ModuleType) -> None:
    fragment = tool.Fragment(key="host_api.tt", sources=("a.hpp",), text="alpha();\nbeta();\ngamma();\n")
    with pytest.raises(tool.Refusal, match="Rule 5"):
        tool.check_pack([fragment], {"tier-a/x.cpp": "  alpha();\n  beta();\n  gamma();\n"})
    with pytest.raises(tool.Refusal, match="limit"):
        tool.check_pack([fragment], {}, limit=8)
    tool.check_pack([fragment], {"tier-a/x.cpp": "alpha();\nbeta();\n"})


def synthetic_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: ModuleType) -> Path:
    """Point the tool at a one-item Tier A suite written here and return a tree holding its kernel only."""
    kernel = f"{EXAMPLES}ex/kernels/k.cpp"
    tree = tmp_path / "tree"
    (tree / kernel).parent.mkdir(parents=True)
    (tree / kernel).write_bytes(b"void kernel_main() {}\n")
    suite = {"commit": PIN, "items": {"ex": {"languages": {"tt": {"support": {"ex/kernels/k.cpp": {
        "upstream": kernel, "sha256": hashlib.sha256(b"void kernel_main() {}\n").hexdigest()}}}}}}}
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump(suite), encoding="utf-8")
    (tmp_path / "tracked").mkdir()
    monkeypatch.setattr(tool, "TIER_A_MANIFEST", tmp_path / "suite.yaml")
    monkeypatch.setattr(tool, "TIER_A_TRACKED", tmp_path / "tracked")
    monkeypatch.setattr(tool, "SHARED_EXAMPLE_DIRS", ())
    monkeypatch.setattr(tool, "TIER_A_EXAMPLE_FILES", {
        f"{EXAMPLES}ex/ex.cpp": hashlib.sha256(b"int main() {}\n").hexdigest(),
        f"{EXAMPLES}ex/ex.md": hashlib.sha256(b"# Walkthrough\n").hexdigest(),
    })
    return tree


def test_tier_a_references_refuse_a_tree_missing_a_host_program_or_walkthrough(
    tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = synthetic_suite(tmp_path, monkeypatch, tool)
    (tree / EXAMPLES / "ex" / "ex.cpp").write_bytes(b"int main() {}\n")
    with pytest.raises(tool.Refusal, match=r"ex/ex\.md is missing"):
        tool.tier_a_references(tree)
    (tree / EXAMPLES / "ex" / "ex.md").write_bytes(b"# Another walkthrough\n")
    with pytest.raises(tool.Refusal, match=r"ex/ex\.md has another sha256"):
        tool.tier_a_references(tree)
    (tree / EXAMPLES / "ex" / "ex.md").write_bytes(b"# Walkthrough\n")
    (tree / EXAMPLES / "ex" / "CMakeLists.txt").write_bytes(b"project(ex)\n")
    got = tool.tier_a_references(tree)
    assert set(got) == {f"{EXAMPLES}ex/{name}" for name in ("kernels/k.cpp", "ex.cpp", "ex.md", "CMakeLists.txt")}


def test_tier_a_example_files_cover_every_example_the_suite_names(tool: ModuleType) -> None:
    suite = yaml.safe_load(TIER_A.read_text(encoding="utf-8"))
    upstream = {spec["upstream"] for item in suite["items"].values() for language in item["languages"].values()
                for spec in language.get("support", {}).values() if "upstream" in spec}
    folders = {path.split("/kernels/", 1)[0] for path in upstream}
    assert len(folders) == 5, folders
    for folder in folders:
        named = [path for path in tool.TIER_A_EXAMPLE_FILES if path.startswith(f"{folder}/")]
        assert any(path.endswith(".cpp") for path in named), f"{folder} has no host program listed"
        assert any(path.endswith(".md") for path in named), f"{folder} has no walkthrough listed"
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in tool.TIER_A_EXAMPLE_FILES.values())


def test_no_source_is_a_programming_example_or_a_tier_a_file(tool: ModuleType) -> None:
    upstream = set()
    for item in yaml.safe_load(TIER_A.read_text(encoding="utf-8"))["items"].values():
        for language in item["languages"].values():
            upstream.update(spec["upstream"] for spec in language.get("support", {}).values() if "upstream" in spec)
    assert len(upstream) == 13, f"the Tier A manifest names {len(upstream)} upstream kernels"
    for path in tool.SOURCES:
        assert not path.startswith(EXAMPLES), f"{path} is a programming example"
        assert path not in upstream and path not in tool.TIER_A_EXAMPLE_FILES, f"{path} is a Tier A file"


def test_pin_is_the_tt_metal_pin_and_the_tier_a_commit(tool: ModuleType) -> None:
    pin = (REPO / "toolchains" / "tt-metal.pin").read_text(encoding="utf-8")
    assert re.search(rf"^COMMIT={tool.PIN}$", pin, flags=re.M), "the tool's PIN is not toolchains/tt-metal.pin's"
    assert tool.PIN == PIN
    assert yaml.safe_load(TIER_A.read_text(encoding="utf-8"))["commit"] == PIN


def test_tool_refuses_a_source_off_the_pin_and_writes_nothing(tool: ModuleType, tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    for path in tool.SOURCES:
        (tree / path).parent.mkdir(parents=True, exist_ok=True)
        (tree / path).write_bytes(b"// not the pinned bytes\n")
    out = tmp_path / "assets"
    status, output = run_tool(tool, "--tree", str(tree), "--out", str(out))
    assert status == 1, output
    assert "sha256" in output and tool.GUIDE in output, output
    assert not out.exists(), "a refusal writes nothing"


def test_tool_refuses_a_tree_without_a_source(tool: ModuleType, tmp_path: Path) -> None:
    out = tmp_path / "assets"
    status, output = run_tool(tool, "--tree", str(tmp_path / "empty"), "--out", str(out))
    assert status == 1 and "refused" in output, output
    assert not out.exists()


def test_manifest_lists_each_fragment_with_its_sources_sha256_and_the_cut(tool: ModuleType) -> None:
    fragments = [
        tool.Fragment(key="metalium_guide.tt", sources=("METALIUM_GUIDE.md",), text="Guide text.\n"),
        tool.Fragment(key="host_api.tt", sources=(HEAD_SOURCE, "a.hpp"), text="int a();\n"),
    ]
    hashes = {"METALIUM_GUIDE.md": "1" * 64, HEAD_SOURCE: "2" * 64, "a.hpp": "3" * 64}
    text = tool.manifest_text(fragments, hashes, ("b.hpp",))
    assert text.isascii() and "\r" not in text
    loaded = yaml.safe_load(text)
    assert loaded["generator"] == "tools/build_tt_context.py"
    assert loaded["commit"] == PIN and loaded["limit_bytes"] == tool.LIMIT_BYTES == LIMIT
    assert [entry["key"] for entry in loaded["entries"]] == ["metalium_guide.tt", "host_api.tt"]
    second = loaded["entries"][1]
    assert second["sources"] == [{"path": HEAD_SOURCE, "sha256": "2" * 64}, {"path": "a.hpp", "sha256": "3" * 64}]
    assert second["sha256"] == hashlib.sha256(b"int a();\n").hexdigest()
    assert loaded["cut_by_limit"] == ["b.hpp"]


# ---------------------------------------------------------------- the pinned tree


def test_committed_manifest_matches_a_fresh_build(fresh: Path) -> None:
    committed = (REPO / "assets" / PACK / MANIFEST).read_bytes()
    assert (fresh / PACK / MANIFEST).read_bytes() == committed, "regenerate the pack and commit its manifest"


def test_fresh_fragments_match_their_manifest_sha256(fresh: Path) -> None:
    tree = fresh / PACK
    entries = fresh_manifest(fresh)["entries"]
    assert [entry["key"] for entry in entries] == ["metalium_guide.tt", "host_api.tt"]
    for entry in entries:
        data = (tree / f"{entry['key']}.txt").read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["key"]


def test_tool_is_deterministic(tool: ModuleType, fresh: Path, tmp_path: Path) -> None:
    again = tmp_path / "again"
    status, output = run_tool(tool, "--tree", str(pinned_tree()), "--out", str(again))
    assert status == 0, output
    names = sorted(path.name for path in (fresh / PACK).iterdir())
    assert names == sorted(path.name for path in (again / PACK).iterdir()) == [MANIFEST, "host_api.tt.txt",
                                                                              "metalium_guide.tt.txt"]
    for name in names:
        assert (fresh / PACK / name).read_bytes() == (again / PACK / name).read_bytes(), name


def test_sources_list_the_include_closure_of_the_roots_in_order(tool: ModuleType) -> None:
    tree = pinned_tree()
    if tree is None:
        pytest.skip(TREE_SKIP)
    closure = bfs_includes(tree)
    assert tuple(closure[:len(ROOTS)]) == ROOTS and all(path.startswith(API_DIR) for path in closure), closure
    assert tool.include_closure(tool.read_sources(tree)) == closure
    assert [path for path in tool.SOURCES if path != tool.GUIDE] == closure


def test_fresh_reference_fills_the_limit_in_order_skipping_what_does_not_fit(
    tool: ModuleType, fresh: Path,
) -> None:
    tree = pinned_tree()
    assert tree is not None
    closure = bfs_includes(tree)
    declared = tool.declared_headers(tool.read_sources(tree), closure)
    order, _ = tool.walk_order(declared, names=list(DERIVED_NAMES))
    manifest = fresh_manifest(fresh)
    sources = [source["path"] for source in manifest["entries"][1]["sources"]]
    assert sources[:2] == [HEAD_SOURCE, TEMPLATE_SOURCE]
    kept, cut = sources[2:], manifest["cut_by_limit"]
    assert sorted(kept + cut) == sorted(closure) and not set(kept) & set(cut), "kept and skipped split the walk"
    assert kept == [path for path in order if path in kept] and cut == [path for path in order if path in cut]
    assert tuple(kept[:len(ROOTS)]) == ROOTS, "every root is kept; only headers past the roots are skipped"
    texts = fragment_texts(fresh)
    includes = re.findall(r"^// ==== #include <([^>]+)> ====$", texts["host_api.tt"], flags=re.M)
    assert includes == [path.split("tt_metal/api/", 1)[1] for path in kept]
    assert all(include.startswith("tt-metalium/") for include in includes), includes
    # Replay the rule: every header as declarations only, the roots always kept, then, in order, a header kept
    # exactly when the pack still fits with it.
    guide = tool.Fragment(tool.GUIDE_KEY, (tool.GUIDE,), texts["metalium_guide.tt"])
    head = tool.read_head()
    so_far = [(path, (tree / path).read_bytes().decode("utf-8")) for path in ROOTS]
    for path in order[len(ROOTS):]:
        trial = [*so_far, (path, (tree / path).read_bytes().decode("utf-8"))]
        fits = tool.pack_size([guide, tool.Fragment(tool.API_KEY, (), tool.api_text(trial, head))]) <= LIMIT
        assert (path in kept) == fits, f"{path}: {'fits but was skipped' if fits else 'kept but does not fit'}"
        so_far = trial if fits else so_far
    assert tool.api_text(so_far, head) == texts["host_api.tt"]
    assert not re.search(r"^\s*(//(?! ==== #include)|/\*|\*)", texts["host_api.tt"], flags=re.M), \
        "no comment is left in the reference but the include lines"


def test_fresh_pack_fits_the_limit(tool: ModuleType, fresh: Path) -> None:
    sizes = [len(text.encode("utf-8")) for text in fragment_texts(fresh).values()]
    assert sum(sizes) + len(sizes) - 1 <= tool.LIMIT_BYTES == LIMIT, sizes


def test_fresh_reference_keeps_no_function_body(fresh: Path) -> None:
    reference = fragment_texts(fresh)["host_api.tt"]
    code = [line for line in reference.split("\n") if not line.lstrip().startswith(("//", "*", "/*"))]
    bodies = [line for line in code if re.match(r"\s*(return\b|if \(|for \()", line)]
    assert not bodies, f"function body lines remain: {bodies[:5]}"


def test_fresh_guide_follows_the_heading_rule(tool: ModuleType, fresh: Path) -> None:
    tree = pinned_tree()
    assert tree is not None
    source = (tree / tool.GUIDE).read_bytes().decode("utf-8")
    headings = [line for line in source.replace("\r\n", "\n").split("\n") if re.match(r"#+ ", line)]
    wanted, part, below = [], None, None
    for heading in headings:
        level = len(heading) - len(heading.lstrip("#"))
        if below is not None and level > below:
            continue
        below = None
        part = heading if level == 2 else (None if level == 1 else part)
        if level == 1 or (part in tool.GUIDE_PARTS and heading not in tool.GUIDE_LEFT_OUT):
            wanted.append(heading)
        elif part in tool.GUIDE_PARTS:
            below = level
    guide = fragment_texts(fresh)["metalium_guide.tt"]
    got = [line for line in guide.split("\n") if re.match(r"#+ ", line)]
    assert got == wanted, got


def test_fresh_guide_holds_no_code_block_image_worked_example_or_dangling_sentence(
    tool: ModuleType, fresh: Path,
) -> None:
    guide = fragment_texts(fresh)["metalium_guide.tt"]
    assert "```" not in guide and "<img" not in guide
    assert "vector addition" not in guide.lower(), "the guide's vector addition example is the eltwise_binary example"
    assert guide.startswith("# TT Architecture and Metalium Guide\n")
    for sentences in tool.GUIDE_DANGLING.values():
        for sentence in sentences:
            assert sentence not in guide, f"{sentence!r} points at text the pack leaves out"


def test_fresh_pack_holds_no_run_of_any_tier_a_file(tool: ModuleType, fresh: Path) -> None:
    references = tool.tier_a_references(pinned_tree())
    tracked = [label for label in references if label.startswith("assets/bench/tt-pairs-v0/")]
    upstream = [label for label in references if label.startswith(EXAMPLES)]
    assert len(tracked) == 10, f"the tracked Tier A programs: {tracked}"
    assert len(upstream) >= 13 + len(tool.TIER_A_EXAMPLE_FILES), "every required Tier A file is read from the tree"
    assert set(tool.TIER_A_EXAMPLE_FILES) <= set(references)
    for key, text in fragment_texts(fresh).items():
        hits = tool.tier_a_runs(text, references)
        assert not hits, f"{key} holds Tier A text: {hits[:5]}"


SPAN = 60


def tier_a_spans(tool: ModuleType, tree: Path, width: int = SPAN) -> dict[str, str]:
    """Return {span: Tier A file} of every `width`-character window of every Tier A file, whitespace collapsed."""
    spans: dict[str, str] = {}
    for label, text in tool.tier_a_references(tree).items():
        flat = " ".join(text.split())
        for k in range(len(flat) - width + 1):
            spans.setdefault(flat[k:k + width], label)
    return spans


def test_fresh_guide_holds_no_sixty_character_span_of_any_tier_a_file(tool: ModuleType, fresh: Path) -> None:
    # A sliding character window over whitespace-normalized text, so a sentence copied from a Tier A file is caught
    # wherever its line breaks fall (the split_work_to_cores paragraph matched a comment of matmul_multi_core.cpp).
    tree = pinned_tree()
    assert tree is not None
    spans = tier_a_spans(tool, tree)
    flat = " ".join(fragment_texts(fresh)["metalium_guide.tt"].split())
    hits = [(spans[flat[k:k + SPAN]], flat[k:k + SPAN]) for k in range(len(flat) - SPAN + 1)
            if flat[k:k + SPAN] in spans]
    assert not hits, f"the guide holds Tier A text: {hits[:3]}"
    uncut = tool.guide_text((tree / tool.GUIDE).read_bytes().decode("utf-8"),
                            cuts={key: value for key, value in tool.GUIDE_CUTS.items() if "SPMD" not in key})
    flat = " ".join(uncut.split())
    assert any(flat[k:k + SPAN] in spans for k in range(len(flat) - SPAN + 1)), \
        "without the SPMD cut the check finds the matmul_multi_core comment, so it can fail"


# ---------------------------------------------------------------- the fuller pack (--name, --limit-bytes)
#
# The tool builds a second pack by the same rules under another name with a larger byte limit (2026-10-25): the
# default pack's bytes and manifest stay as they were, and the fuller pack's manifest records its own limit.

FULL_NAME = "tt-metalium-5280a9cf-full"
FULL_PACK = f"context/{FULL_NAME}"
FULL_LIMIT = 128 * 1024


def routed_build(tool: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the tool's build with one returning a SYNTHETIC pack; return the list each call's limit lands in."""
    limits: list[int] = []

    def build(tree: Path, limit: int = tool.LIMIT_BYTES) -> object:
        limits.append(limit)
        fragments = (tool.Fragment("metalium_guide.tt", ("METALIUM_GUIDE.md",), "SYNTHETIC guide.\n"),
                     tool.Fragment("host_api.tt", (HEAD_SOURCE, TEMPLATE_SOURCE), "int synthetic();\n"))
        return tool.Pack(fragments, {HEAD_SOURCE: "2" * 64, TEMPLATE_SOURCE: "3" * 64}, ("b.hpp",), limit)

    monkeypatch.setattr(tool, "build", build)
    monkeypatch.setattr(tool, "SOURCES", {"METALIUM_GUIDE.md": "1" * 64})
    return limits


def test_main_builds_a_named_pack_with_its_own_limit_beside_the_default(
    tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits = routed_build(tool, monkeypatch)
    out = tmp_path / "assets"
    status, output = run_tool(tool, "--tree", str(tmp_path), "--out", str(out), "--name", FULL_NAME,
                              "--limit-bytes", str(FULL_LIMIT))
    assert status == 0, output
    assert limits == [FULL_LIMIT], "the limit reaches the build"
    assert not (out / PACK).exists(), "a named pack leaves the default pack's directory alone"
    loaded = yaml.safe_load((out / FULL_PACK / MANIFEST).read_text(encoding="ascii"))
    assert loaded["limit_bytes"] == FULL_LIMIT and loaded["cut_by_limit"] == ["b.hpp"]
    assert sorted(path.name for path in (out / FULL_PACK).iterdir()) == [MANIFEST, "host_api.tt.txt",
                                                                        "metalium_guide.tt.txt"]
    assert f"of {FULL_LIMIT} bytes" in output, output
    status, output = run_tool(tool, "--tree", str(tmp_path), "--out", str(out))
    assert status == 0, output
    assert limits == [FULL_LIMIT, LIMIT], "without the options the build gets LIMIT_BYTES"
    assert yaml.safe_load((out / PACK / MANIFEST).read_text(encoding="ascii"))["limit_bytes"] == LIMIT


def test_the_default_options_give_the_manifest_text_of_before(tool: ModuleType) -> None:
    fragments = [tool.Fragment(key="host_api.tt", sources=(HEAD_SOURCE,), text="int a();\n")]
    hashes = {HEAD_SOURCE: "2" * 64}
    default = tool.manifest_text(fragments, hashes, ("b.hpp",))
    assert tool.manifest_text(fragments, hashes, ("b.hpp",), tool.LIMIT_BYTES) == default
    assert f"limit_bytes: {LIMIT}\n" in default and "name" not in yaml.safe_load(default), "no new field"
    other = tool.manifest_text(fragments, hashes, ("b.hpp",), FULL_LIMIT)
    assert other == default.replace(f"limit_bytes: {LIMIT}\n", f"limit_bytes: {FULL_LIMIT}\n")
    assert tool.PACK_TREE == PACK and tool.pack_tree() == PACK and tool.pack_tree(FULL_NAME) == FULL_PACK


@pytest.mark.parametrize(("option", "value"), [
    ("--name", "../tt-metalium-5280a9cf-full"), ("--name", "other-pack"), ("--name", "tt-metalium-5280a9cf-"),
    ("--name", "tt-metalium-5280a9cf/full"), ("--name", "tt-metalium-5280a9cf-Full"), ("--limit-bytes", "0"),
    ("--limit-bytes", "-5"),
], ids=["parent", "other-name", "empty-suffix", "slash", "capital", "zero", "negative"])
def test_main_refuses_a_bad_name_or_limit_and_writes_nothing(
    tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str, value: str,
) -> None:
    limits = routed_build(tool, monkeypatch)
    out = tmp_path / "assets"
    status, output = run_tool(tool, "--tree", str(tmp_path), "--out", str(out), option, value)
    assert status == 1 and "refused" in output and option.lstrip("-") in output, output
    assert limits == [] and not out.exists(), "a refusal builds and writes nothing"


@pytest.fixture(scope="module")
def fresh_full(tool: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return an assets root holding the fuller pack built from the pinned tree; skip without one."""
    tree = pinned_tree()
    if tree is None:
        pytest.skip(TREE_SKIP)
    out = tmp_path_factory.mktemp("assets-full")
    status, output = run_tool(tool, "--tree", str(tree), "--out", str(out), "--name", FULL_NAME,
                              "--limit-bytes", str(FULL_LIMIT))
    assert status == 0, f"the tool exited {status}: {output[-3000:]}"
    return out


def test_committed_full_manifest_matches_a_fresh_build(fresh_full: Path) -> None:
    committed = (REPO / "assets" / FULL_PACK / MANIFEST).read_bytes()
    assert (fresh_full / FULL_PACK / MANIFEST).read_bytes() == committed, "regenerate the full pack's manifest"


def test_fresh_full_pack_fits_its_limit_and_keeps_every_header_the_default_keeps(
    tool: ModuleType, fresh: Path, fresh_full: Path,
) -> None:
    full = yaml.safe_load((fresh_full / FULL_PACK / MANIFEST).read_text(encoding="ascii"))
    default = fresh_manifest(fresh)
    assert full["limit_bytes"] == FULL_LIMIT
    kept = [source["path"] for source in full["entries"][1]["sources"]]
    assert set(source["path"] for source in default["entries"][1]["sources"]) <= set(kept)
    assert full["entries"][0] == default["entries"][0], "the guide is the default pack's"
    texts = [(fresh_full / FULL_PACK / f"{entry['key']}.txt").read_bytes() for entry in full["entries"]]
    assert sum(len(text) for text in texts) + len(texts) - 1 <= FULL_LIMIT
    assert len(full["cut_by_limit"]) < len(default["cut_by_limit"])
