"""Tests for the documentation corpus: chunking, the Agent Rule 5 exclusion, and the manifest's hashes.

Bible: Agent Rule 5 (evaluation splits are untouchable), Design Principle 3
(data is versioned files), the Repository Layout note on generated text
(OQ-018). Owner direction of 2026-10-23: a retriever over the Tenstorrent
documentation, the fallback beside the API-reference context pack.

The contract these tests fix (lassi.prompts.rag_corpus, tools/build_tt_rag.py):

- chunk_file cuts Markdown and reStructuredText at headings and code at
  blank lines, never cuts a fenced or literal block at a blank line, packs
  blocks under max_lines and max_chars, never cuts a line, leaves a code
  file's leading SPDX license comment out, and gives each chunk the exact
  text of its line range.
- Rule 5: no indexed chunk holds a run of 3 consecutive lines of any Tier A
  reference (the suite's tracked programs and every file under a Tier A
  example directory of the tree) under either normalization, applied alike
  to both sides: code (each line's code alone: // and /* */ comments cut,
  a block comment's middle line starting with *, and brace-only lines
  skipped) and literal (each non-blank line stripped, inner whitespace
  collapsed). Those files, the pages the spec lists, and a file whose runs
  are mostly reference runs are excluded whole; any other reference run is
  excised line by line and recorded, and the chunks keep the file's text,
  comments included. Of a chunk an excision cut, a piece of fewer than 3
  non-blank lines, or one holding a code fence whose pair the excision cut
  off, is dropped and recorded as excised.
- The similarity report records each indexed code file whose token
  similarity to its closest Tier A code reference is 0.7 or more, such as
  a renamed copy no run shows; the build records it and does not refuse.
- The manifest records the pin, every selected file's sha256, the
  parameters, the exclusions, the excised lines, and the index's sha256,
  and a rebuild from the same tree gives the same bytes.

The oracle here (reference_runs, chunk_runs) is this file's own, written
from the rule's words, not from the implementation. The Tier A references
are the tracked files in assets/bench/tt-pairs-v0/; the upstream kernels are
checked only when a pinned tree is available (LASSI_TT_RAG_TREE, or the
install under $LASSI_TOOLCHAINS), and the real index only when one is
(LASSI_TT_RAG_INDEX, or the default under $LASSI_SCRATCH); otherwise those
tests skip and name tools/build_tt_rag.py. Every other corpus is SYNTHETIC
text written here. No value here is a measurement.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from lassi.prompts import rag_corpus
from lassi.prompts.retrieval import INDEX_FORMAT, TOKENIZER, Chunk, Index, load_index
from lassi.toolchains.pins import read_pin

REPO = Path(__file__).resolve().parents[2]
PIN = read_pin("tt-metal")
SPEC = REPO / "assets" / "rag" / "tt-metal-docs" / "corpus.yaml"
MANIFEST = SPEC.parent / "MANIFEST.yaml"
SUITE = REPO / "assets" / "bench" / "tt-pairs-v0.yaml"
TRACKED_DIR = REPO / "assets" / "bench" / "tt-pairs-v0"
RUN = 3
BACKSLASH = chr(92)


def tool() -> ModuleType:
    """Import tools/build_tt_rag.py by path and return the module."""
    spec = importlib.util.spec_from_file_location("build_tt_rag", REPO / "tools" / "build_tt_rag.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def literal_form(line: str) -> str:
    """Return a line stripped with inner whitespace collapsed ('' for a blank line): the literal normalization."""
    return " ".join(line.split())


def code_form(line: str) -> str:
    """Return a line's code as the rule's words define it ('' for none), scanned here character by character.

    A */ with no /* or // before it closes a comment begun above; then each
    /* */ span is skipped, and an unclosed /* or a // ends the line. What is
    left, stripped and collapsed, is no code when it starts with * or holds
    only braces, parentheses, semicolons, and commas (one brace at least).
    """
    head, close, tail = line.partition("*/")
    if close and "/*" not in head and "//" not in head:
        line = tail
    kept, index = [], 0
    while index < len(line):
        if line.startswith("//", index):
            break
        if line.startswith("/*", index):
            end = line.find("*/", index + 2)
            if end < 0:
                break
            kept.append(" ")
            index = end + 2
            continue
        kept.append(line[index])
        index += 1
    text = " ".join("".join(kept).split())
    if text.startswith("*") or (set(text) & set("{}") and set(text) <= set("{}();, ")):
        return ""
    return text


FORMS: dict[str, Callable[[str], str]] = {"code": code_form, "literal": literal_form}


def runs_of(text: str, size: int = RUN, form: Callable[[str], str] = literal_form) -> set[tuple[str, ...]]:
    """Return every run of `size` consecutive lines of `text` under `form`, the lines it leaves empty skipped."""
    lines = [norm for norm in (form(line) for line in text.replace("\r\n", "\n").split("\n")) if norm]
    return {tuple(lines[i:i + size]) for i in range(len(lines) - size + 1)}


def tracked_references() -> dict[str, str]:
    """Return {repository path: text} of every tracked Tier A program, found by listing the tracked directory."""
    return {path.relative_to(REPO).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted(TRACKED_DIR.rglob("*.cpp"))}


def tier_a_dirs_from_suite() -> dict[str, str]:
    """Return {example directory: item} read from the suite manifest's upstream kernel paths, independently."""
    data = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for item, spec in data["items"].items():
        for language in spec["languages"].values():
            for entry in (language.get("support") or {}).values():
                if "upstream" in entry:
                    found[entry["upstream"].split("/kernels/")[0]] = item
    return found


def reference_runs(references: dict[str, str]) -> dict[str, set[tuple[str, ...]]]:
    """Return {normalization: the union of every reference's runs under it}, for code and literal."""
    return {kind: set().union(*(runs_of(text, form=form) for text in references.values()))
            for kind, form in FORMS.items()}


def offending_chunks(index: Index, runs: dict[str, set[tuple[str, ...]]]) -> list[str]:
    """Return '<path>:<first line>' for every chunk of `index` holding a reference run under either normalization."""
    return [f"{chunk.path}:{chunk.start_line}" for chunk in index.chunks
            if any(runs_of(chunk.text, form=form) & runs[kind] for kind, form in FORMS.items())]


def distinctive_lines(text: str, count: int) -> list[str]:
    """Return the first `count` consecutive non-blank lines of `text` that are each long code lines."""
    lines = [line for line in text.split("\n") if line.strip()]
    for start in range(len(lines) - count + 1):
        window = lines[start:start + count]
        if all(len(line.strip()) > 25 and not line.strip().startswith(("#", "//")) for line in window):
            return window
    raise AssertionError("no distinctive window in the tracked reference")


# ---------------------------------------------------------------------------
# Chunking


def chunks_of(path: str, text: str, max_lines: int = 40, max_chars: int = 3000) -> list[Chunk]:
    """Return chunk_file's chunks of `text` at `path`."""
    return rag_corpus.chunk_file(path, text, max_lines=max_lines, max_chars=max_chars)


def assert_exact(text: str, chunks: Sequence[Chunk]) -> None:
    """Assert each chunk's text is exactly its line range of `text`."""
    lines = rag_corpus.split_lines(text)
    for chunk in chunks:
        assert chunk.text == "\n".join(lines[chunk.start_line - 1:chunk.end_line]), chunk


MARKDOWN = """# Guide

Intro paragraph about the device.

## Setup

Install the runtime first.

```cpp
int first = 1;

int second = 2;
```

## Use

Run the program.
"""


def test_markdown_is_cut_at_headings_with_heading_paths() -> None:
    chunks = chunks_of("docs/guide.md", MARKDOWN)
    assert [(c.start_line, c.end_line, c.heading) for c in chunks] == [
        (1, 3, "Guide"), (5, 13, "Guide > Setup"), (15, 17, "Guide > Use")]
    assert_exact(MARKDOWN, chunks)


def test_a_fenced_block_is_never_cut_at_its_blank_line() -> None:
    chunks = chunks_of("docs/guide.md", MARKDOWN, max_lines=6)
    fenced = [c for c in chunks if "int first" in c.text]
    assert len(fenced) == 1 and "int second" in fenced[0].text
    assert_exact(MARKDOWN, chunks)


RST = """=======
Manual
=======

Opening words.

Circular buffers
----------------

Prose about buffers.

.. code-block:: cpp

   cb_reserve_back(cb, 1);

   cb_push_back(cb, 1);

Semaphores
----------

Prose about semaphores.
"""


def test_rst_titles_and_literal_blocks() -> None:
    chunks = chunks_of("docs/manual.rst", RST)
    assert [(c.start_line, c.heading) for c in chunks] == [
        (1, "Manual"), (7, "Manual > Circular buffers"), (18, "Manual > Semaphores")]
    literal = [c for c in chunks if "cb_reserve_back" in c.text][0]
    assert "cb_push_back" in literal.text
    assert_exact(RST, chunks)


HEADER = """// SPDX-FileCopyrightText: (c) 2026 Synthetic Authors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * The file's own doc comment stays.
 */
#pragma once

// Returns one.
int one();

// Returns two.
int two();
"""


def test_a_code_file_drops_its_license_comment_and_keeps_its_doc_comment() -> None:
    chunks = chunks_of("api/x.hpp", HEADER)
    assert chunks[0].start_line == 5
    assert all("SPDX" not in c.text for c in chunks)
    assert "The file's own doc comment stays." in chunks[0].text
    assert_exact(HEADER, chunks)


def test_blocks_pack_under_max_lines_and_a_long_block_is_cut_at_whole_lines() -> None:
    body = "\n".join(f"int value_{n} = {n};" for n in range(25)) + "\n\nint tail = 0;\n"
    chunks = chunks_of("src/long.cpp", body, max_lines=10)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 10), (11, 20), (21, 25), (27, 27)]
    assert all(c.end_line - c.start_line + 1 <= 10 for c in chunks)
    assert_exact(body, chunks)


def test_a_line_longer_than_max_chars_stays_whole() -> None:
    text = "short line\n" + "x" * 500 + "\nanother short line\n"
    chunks = chunks_of("docs/wide.md", text, max_chars=100)
    assert any(c.text == "x" * 500 for c in chunks)
    assert_exact(text, chunks)


def test_line_numbers_count_newlines_only_and_crlf_is_read_as_lf() -> None:
    text = "# Title\r\n\r\nA line with a form feed \x0c inside.\r\nNext line.\r\n"
    chunks = chunks_of("docs/crlf.md", text)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 4)]
    assert "\r" not in chunks[0].text and chunks[0].text.endswith("Next line.")


def test_doxygen_inputs_follow_continuations_and_additions() -> None:
    text = (f"PROJECT_NAME = x\nINPUT = a/one.h {BACKSLASH}\n        a/two.h {BACKSLASH}\n        a/one.h\n"
            f"RECURSIVE = NO\nINPUT += b/three.h\n# INPUT = commented/out.h\n")
    assert rag_corpus.doxygen_inputs(text) == ["a/one.h", "a/two.h", "b/three.h"]


@pytest.mark.parametrize(("pattern", "path", "expected"), [
    ("docs/**/*.rst", "docs/a.rst", True),
    ("docs/**/*.rst", "docs/x/y/a.rst", True),
    ("docs/**/*.rst", "docs/a.md", False),
    ("docs/*.rst", "docs/x/a.rst", False),
    ("METALIUM_GUIDE.md", "METALIUM_GUIDE.md", True),
    ("tt_metal/api/tt-metalium/**/*.hpp", "tt_metal/api/tt-metalium/host_api.hpp", True),
])
def test_glob_match(pattern: str, path: str, expected: bool) -> None:
    assert rag_corpus.glob_match(pattern, path) is expected


def numbered(text: str) -> list[tuple[int, str]]:
    """Return (line number, line) for every line of `text`, as the run check takes them."""
    return list(enumerate(rag_corpus.split_lines(text), start=1))


def test_the_run_check_finds_runs_across_blank_lines_and_reindentation() -> None:
    check = rag_corpus.RunCheck({"ref.cpp": "alpha();\nbeta();\n\ngamma();\ndelta();\n"}, RUN)
    found = check.matches(numbered("prose\n    alpha();\n\n  beta();\ngamma();\nprose again\n"))
    assert (found.matched["code"], found.total["code"]) == (1, 3)
    assert (found.matched["literal"], found.total["literal"]) == (1, 3)
    hits = found.lines()
    assert sorted(hits) == [2, 4, 5] and set(hits.values()) == {"ref.cpp"}


@pytest.mark.parametrize(("line", "code"), [
    ("    total += step;  // keep a running sum", "total += step;"),
    ("// a whole-line comment", ""),
    ("call(/*count=*/3, /*flag*/ true);", "call( 3, true);"),
    ("first(); /* a comment that runs on", "first();"),
    (" * the middle of a block comment", ""),
    ("   comment end */ resume();", "resume();"),
    ("*/", ""),
    ("    }", ""),
    ("});", ""),
    ("} else {", "} else {"),
    ("", ""),
    ("print(a // b);", "print(a"),
])
def test_code_line_keeps_only_code(line: str, code: str) -> None:
    assert rag_corpus.code_line(line) == code_form(line) == code


def test_normalized_lines_skip_comments_and_braces_and_literal_lines_keep_them() -> None:
    lines = ["int a = 1;  // one", "", "// two", "/**", " * three", " */", "}", "b();"]
    assert rag_corpus.normalized_lines(lines, 10) == [(10, "int a = 1;"), (17, "b();")]
    assert [number for number, _ in rag_corpus.literal_lines(lines, 10)] == [10, 12, 13, 14, 15, 16, 17]


# SYNTHETIC: a reference whose code lines are interleaved with every comment form, and a text quoting those code
# lines bare, with one comment of its own between them.
INTERLEAVED_REFERENCE = """void kernel_main() {
    // The source address comes first.
    uint32_t interleaved_source_address = get_arg_val<uint32_t>(0);  // in DRAM
    /* The tile count follows it. */
    uint32_t interleaved_tile_count = get_arg_val<uint32_t>(1);
    for (uint32_t interleaved_tile = 0; interleaved_tile < interleaved_tile_count; interleaved_tile++) {
        /*
         * One tile per pass.
         */
        interleaved_read_one_tile(interleaved_source_address, interleaved_tile);
    }
}
"""
INTERLEAVED_CODE = [line.split("//")[0].rstrip() for line in INTERLEAVED_REFERENCE.split("\n")
                    if "interleaved_" in line]
BARE_QUOTE = "\n".join([*INTERLEAVED_CODE[:2], "// then the loop", *INTERLEAVED_CODE[2:], "}"])


def test_a_comment_interleaved_reference_is_caught_by_the_code_normalization_only() -> None:
    """The old normalization (literal) sees no run here; stripping comments on both sides finds the copy."""
    assert not runs_of(BARE_QUOTE) & runs_of(INTERLEAVED_REFERENCE)
    assert runs_of(BARE_QUOTE, form=code_form) & runs_of(INTERLEAVED_REFERENCE, form=code_form)
    check = rag_corpus.RunCheck({"ref.cpp": INTERLEAVED_REFERENCE}, RUN)
    found = check.matches(numbered("prose before\n" + BARE_QUOTE + "\nprose after\n"))
    assert found.matched["literal"] == 0 and found.matched["code"] == 2
    assert sorted(found.lines()) == [2, 3, 5, 6]


# ---------------------------------------------------------------------------
# A SYNTHETIC tree built through the tool


TILIZE_HEADER = """// SPDX-FileCopyrightText: (c) 2026 Synthetic Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

// Converts a row-major matrix of m x n values into tiles of faces.
template <typename T>
std::vector<T> tilize_nfaces(const std::vector<T>& input, uint32_t m, uint32_t n);

// Converts tiles of faces back to a row-major matrix of m x n values.
template <typename T>
std::vector<T> untilize_nfaces(const std::vector<T>& input, uint32_t m, uint32_t n);

// Swaps the two halves of every face; used only by the tests.
template <typename T>
std::vector<T> swap_face_halves(const std::vector<T>& input);
"""

UPSTREAM_KERNEL = """void kernel_main() {
    uint32_t synthetic_source_address = get_arg_val<uint32_t>(0);
    uint32_t synthetic_target_address = get_arg_val<uint32_t>(1);
    uint32_t synthetic_tile_count = get_arg_val<uint32_t>(2);
    for (uint32_t synthetic_tile = 0; synthetic_tile < synthetic_tile_count; synthetic_tile++) {
        synthetic_copy_one_tile(synthetic_source_address, synthetic_target_address, synthetic_tile);
    }
}
"""


# An upstream Tier A file whose code is interleaved with comments, and a page quoting its code bare among enough
# prose that the page stays under max_file_overlap. With the synthetic spec's 12-line chunks, the three lines before
# the fence and the fenced quote make one chunk; the excision leaves of it a piece holding the opening fence alone
# (unpaired-fence), the page's own comment line (short-piece), and the closing brace and fence (short-piece). The
# "Own code" section is a chunk no excision touches, kept whole with its comment.
INTERLEAVED_PATH = "tt_metal/programming_examples/loopback/kernels/interleaved_reader.cpp"
OWN_COMMENT = "// a comment the matching strips"
BARE_PAGE = ("# Bare\n\n" + "".join(f"Synthetic bare prose line {n}.\n" for n in range(24))
             + "\nThe reader's loop follows.\nIt is quoted without its comments.\nRead it beside the reference.\n\n"
             + "```cpp\n" + BARE_QUOTE + "\n```\n\nThat is the loop.\n\n## Own code\n\n"
             + f"```cpp\nint bare_own_code = 0;  {OWN_COMMENT}\nint bare_more_code = 1;\n```\n")
# A copy of UPSTREAM_KERNEL with every local renamed: no line of it is a reference line, so only the similarity
# report sees it.
RENAMED_PATH = "tt_metal/programming_examples/renamed/renamed.cpp"
UPSTREAM_PATH = "tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp"


def synthetic_tree(root: Path) -> dict[str, bytes]:
    """Write the SYNTHETIC tree under `root` and return {path: bytes} of every file written."""
    matmul = tracked_references()["assets/bench/tt-pairs-v0/matmul_single_core/tt/matmul_single_core.cpp"]
    eltwise = tracked_references()["assets/bench/tt-pairs-v0/eltwise_binary/tt/eltwise_binary.cpp"]
    quote = distinctive_lines(matmul, RUN)
    pair = distinctive_lines(eltwise, 2)
    kernel_quote = [line for line in UPSTREAM_KERNEL.split("\n") if "get_arg_val" in line]
    code = "\n".join(f"   {line.strip()}" for line in quote)
    files = {
        "lassi-fetch.txt": f"{PIN['COMMIT']}\ncorpus synthetic-docs: a SYNTHETIC test tree\n",
        "METALIUM_GUIDE.md": "# Synthetic guide\n\nKernels talk through circular buffers.\n\n## Reading\n\n"
                             "A reader kernel calls noc_async_read and then a barrier.\n",
        "docs/Doxyfile": f"PROJECT_NAME = synthetic\nINPUT = tt_metal/hw/inc/api/dataflow/dataflow_api.h {BACKSLASH}\n"
                         "        tt_metal/api/tt-metalium/host_api.hpp\nRECURSIVE = NO\n",
        "docs/source/guide.rst": "Guide\n=====\n\nOverview\n--------\n\nThe overview explains the mesh device.\n\n"
                                 f"Example\n-------\n\n.. code-block:: cpp\n\n   int synthetic_setup = 0;\n{code}\n"
                                 "   int synthetic_teardown = 1;\n\nReading\n-------\n\n"
                                 "Reading tiles back is covered elsewhere.\n",
        "docs/source/near.md": "# Near\n\nTwo quoted lines follow.\n\n```cpp\n" + "\n".join(pair)
                               + "\nint synthetic_breaks_the_run = 2;\n```\n",
        "docs/source/kernels.md": "# Kernels\n\n" + "".join(f"Synthetic prose line {n} about kernels.\n"
                                                             for n in range(12))
                                  + "\nThe arguments arrive in order:\n\n```cpp\n" + "\n".join(kernel_quote)
                                  + "\n```\n\nThat is all.\n",
        "docs/source/bare.md": BARE_PAGE,
        "docs/source/walkthrough.rst": "Walkthrough\n===========\n\nA synthetic walkthrough page.\n",
        "docs/source/latin1.rst": "Caf\xe9\n====\n",
        "docs/source/image.png": "not an image",
        "tech_reports/notes.md": "# Notes\n\nThe fabric routes packets between chips.\n",
        "tt_metal/hw/inc/api/dataflow/dataflow_api.h": "// SPDX-License-Identifier: Apache-2.0\n\n"
                                                        "// Starts an asynchronous read over the NoC.\n"
                                                        "void noc_async_read(uint64_t src, uint32_t dst);\n",
        "tt_metal/hw/inc/api/dataflow/other.h": "// not listed by the Doxyfile\nvoid other();\n",
        "tt_metal/api/tt-metalium/host_api.hpp": "// Creates a program.\nProgram CreateProgram();\n",
        "tt_metal/api/tt-metalium/tilize_utils.hpp": TILIZE_HEADER,
        UPSTREAM_PATH: UPSTREAM_KERNEL,
        INTERLEAVED_PATH: INTERLEAVED_REFERENCE,
        RENAMED_PATH: UPSTREAM_KERNEL.replace("synthetic_", "renamed_"),
        "tt_metal/programming_examples/loopback/loopback.md": "# Loopback\n\nThe Tier A walkthrough.\n",
        "tt_metal/programming_examples/matmul/matmul_single_core/matmul_single_core.cpp": matmul,
        "tt_metal/programming_examples/copycat/copycat.cpp": "// a copy of a Tier A program\n" + eltwise,
        "tt_metal/programming_examples/hello/hello.cpp": "// Prints a greeting from the host.\nint main() {}\n",
    }
    written: dict[str, bytes] = {}
    for path, text in files.items():
        data = text.encode("latin-1") if path.endswith("latin1.rst") else text.encode("utf-8")
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
        written[path] = data
    return written


def synthetic_spec(path: Path) -> Path:
    """Write the SYNTHETIC corpus spec at `path`: the real spec's rules, small chunks, one listed page."""
    data = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    data["corpus"] = "synthetic-docs"
    data["exclude"] = [{"path": "docs/source/walkthrough.rst", "reason": "a synthetic walkthrough"}]
    data["chunking"] = {"max_lines": 12, "max_chars": 600, "min_piece_lines": 3}
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


class Built:
    """One build of the SYNTHETIC tree: its tree, files, spec, index, manifest, and the tool's status."""

    def __init__(self, root: Path) -> None:
        self.tree = root / "tree"
        self.files = synthetic_tree(self.tree)
        self.spec = synthetic_spec(root / "corpus.yaml")
        self.index_path = root / "out" / "index.json"
        self.manifest_path = root / "out" / "MANIFEST.yaml"
        self.status = run_tool("build", "--tree", self.tree, "--corpus", self.spec, "--index", self.index_path,
                               "--manifest", self.manifest_path)
        self.manifest = rag_corpus.read_manifest(self.manifest_path)
        self.index = load_index(self.index_path, sha256=self.manifest["index"]["sha256"])

    def reason(self, path: str) -> str:
        """Return the exclusion reason of `path`; fail when it is not excluded."""
        return next(row["reason"] for row in self.manifest["excluded"] if row["path"] == path)


def run_tool(*args: str | Path) -> int:
    """Run tools/build_tt_rag.py main with `args` (paths turned to strings) and return its status."""
    return tool().main([str(arg) for arg in args])


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    """Build the SYNTHETIC tree once for this module."""
    return Built(tmp_path_factory.mktemp("rag"))


def indexed_paths(built: Built) -> set[str]:
    """Return the paths of the files with indexed chunks."""
    return {row["path"] for row in built.manifest["files"]}


def test_the_build_selects_the_spec_globs_and_the_doxygen_inputs_only(built: Built) -> None:
    assert built.status == 0
    selected = indexed_paths(built) | {row["path"] for row in built.manifest["excluded"]}
    expected = {path for path in built.files if path not in (
        "lassi-fetch.txt", "docs/Doxyfile", "docs/source/image.png", "tt_metal/hw/inc/api/dataflow/other.h")}
    assert selected == expected


def test_every_file_of_a_tier_a_example_directory_is_excluded(built: Built) -> None:
    kernel = "tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp"
    assert built.reason(kernel) == "tier-a-item: loopback"
    assert built.reason("tt_metal/programming_examples/loopback/loopback.md") == "tier-a-item: loopback"
    assert built.reason("tt_metal/programming_examples/matmul/matmul_single_core/matmul_single_core.cpp") \
        == "tier-a-item: matmul_single_core"
    assert {row["path"]: row["item"] for row in built.manifest["tier_a"]["dirs"]} == tier_a_dirs_from_suite()


def test_no_indexed_chunk_holds_three_lines_of_a_tier_a_reference(built: Built) -> None:
    references = tracked_references()
    references.update({path: data.decode("utf-8") for path, data in built.files.items()
                       if any(path.startswith(f"{d}/") for d in tier_a_dirs_from_suite())})
    assert offending_chunks(built.index, reference_runs(references)) == []


SHORT = "short-piece: fewer than 3 lines left beside an excision"
UNPAIRED = "unpaired-fence: an excision cut a code fence off from its pair"


def excised_rows(built: Built, path: str) -> list[tuple[int, int, str]]:
    """Return (first, last, reason) of every excised row of `path`, in manifest order."""
    return [(row["first"], row["last"], row["reason"]) for row in built.manifest["excised"] if row["path"] == path]


def line_of(text: str, needle: str) -> int:
    """Return the 1-based number of the first line of `text` holding `needle`."""
    return next(n for n, line in enumerate(rag_corpus.split_lines(text), start=1) if needle in line)


def test_a_quoted_reference_run_is_excised_and_the_short_piece_after_it_dropped(built: Built) -> None:
    guide = [c for c in built.index.chunks if c.path == "docs/source/guide.rst"]
    text = "\n".join(c.text for c in guide)
    assert "The overview explains the mesh device." in text and "Reading tiles back" in text
    assert "synthetic_setup" in text and "synthetic_teardown" not in text
    page = built.files["docs/source/guide.rst"].decode("utf-8")
    lines = rag_corpus.split_lines(page)
    quote = distinctive_lines(tracked_references()[
        "assets/bench/tt-pairs-v0/matmul_single_core/tt/matmul_single_core.cpp"], RUN)
    quoted = [n for n, line in enumerate(lines, start=1) if line.strip() in {q.strip() for q in quote}]
    teardown = line_of(page, "synthetic_teardown")
    assert excised_rows(built, "docs/source/guide.rst") == [
        (quoted[0], quoted[-1], "tier-a-run: assets/bench/tt-pairs-v0/matmul_single_core/tt/matmul_single_core.cpp"),
        (teardown, teardown, SHORT)]
    assert not any(c.start_line <= n <= c.end_line for c in guide for n in [*quoted, teardown])


def test_a_run_of_an_upstream_tier_a_file_is_excised_too(built: Built) -> None:
    kernels = [c for c in built.index.chunks if c.path == "docs/source/kernels.md"]
    assert kernels and not any("get_arg_val" in c.text for c in kernels)
    page = built.files["docs/source/kernels.md"].decode("utf-8")
    first, last = line_of(page, "get_arg_val<uint32_t>(0)"), line_of(page, "get_arg_val<uint32_t>(2)")
    assert excised_rows(built, "docs/source/kernels.md") == [
        (line_of(page, "The arguments arrive"), first - 1, SHORT), (first, last, f"tier-a-run: {UPSTREAM_PATH}"),
        (last + 1, line_of(page, "That is all."), SHORT)]


def test_a_comment_interleaved_reference_run_is_excised_with_the_pieces_it_breaks(built: Built) -> None:
    page = "docs/source/bare.md"
    text = built.files[page].decode("utf-8")
    assert not runs_of(text) & runs_of(INTERLEAVED_REFERENCE), "the literal normalization alone would miss it"
    lines = rag_corpus.split_lines(text)
    quoted = [n for n, line in enumerate(lines, start=1) if line in INTERLEAVED_CODE]
    reason = f"tier-a-run: {INTERLEAVED_PATH}"
    opening = line_of(text, "The reader's loop follows.")
    assert excised_rows(built, page) == [
        (opening, quoted[0] - 1, UNPAIRED), (quoted[0], quoted[1], reason), (quoted[1] + 1, quoted[1] + 1, SHORT),
        (quoted[2], quoted[3], reason), (quoted[3] + 1, quoted[3] + 2, SHORT)]
    assert lines[quoted[0] - 2] == "```cpp" and lines[quoted[1]] == "// then the loop" and lines[quoted[3] + 1] == "```"
    chunks = [c for c in built.index.chunks if c.path == page]
    kept = "\n".join(c.text for c in chunks)
    assert not any(code.strip() in kept for code in INTERLEAVED_CODE) and "// then the loop" not in kept
    assert "Synthetic bare prose line 23." in kept and "That is the loop." in kept
    own = [c for c in chunks if OWN_COMMENT in c.text]
    assert len(own) == 1 and own[0].heading == "Bare > Own code", "a chunk no excision touched stays whole"
    assert all(sum(line.lstrip().startswith("```") for line in c.text.split("\n")) % 2 == 0 for c in chunks)
    assert_exact(text, chunks)


def test_a_renamed_copy_the_run_check_misses_is_in_the_similarity_report(built: Built) -> None:
    assert RENAMED_PATH in indexed_paths(built) and not excised_rows(built, RENAMED_PATH)
    rows = {row["path"]: row for row in built.manifest["similar"]}
    assert rows[RENAMED_PATH] == {"path": RENAMED_PATH, "similarity": 1.0, "reference": UPSTREAM_PATH}
    assert "tt_metal/programming_examples/hello/hello.cpp" not in rows
    assert all(row["similarity"] >= 0.7 and not row["path"].endswith((".md", ".rst")) for row in rows.values())
    assert built.manifest["tier_a"]["similarity"]["report_at"] == 0.7


def test_similarity_tokens_read_a_renamed_copy_as_the_same_program() -> None:
    prefixes = rag_corpus.load_corpus_spec(SPEC).api_prefixes
    tokens = rag_corpus.similarity_tokens(UPSTREAM_KERNEL, prefixes)
    disguised = ("#include <cstdint>\n" + UPSTREAM_KERNEL.replace("synthetic_", "other_").replace("(2)", "(5)")
                 + "/* a trailing\n   comment */ // and another\n")
    assert rag_corpus.similarity_tokens(disguised, prefixes) == tokens
    assert "get_arg_val" in tokens and "v" in tokens and "L" in tokens and "synthetic_tile" not in tokens


def test_the_similarity_report_records_only_texts_at_or_above_its_line() -> None:
    prefixes = rag_corpus.load_corpus_spec(SPEC).api_prefixes
    texts = {"b/copy.cpp": UPSTREAM_KERNEL.replace("synthetic_", "copied_"), "a/other.cpp": TILIZE_HEADER}
    report = rag_corpus.similarity_report(texts, {"ref/kernel.cpp": UPSTREAM_KERNEL}, report_at=0.7,
                                          api_prefixes=prefixes)
    assert report == [rag_corpus.SimilarFile("b/copy.cpp", 1.0, "ref/kernel.cpp")]
    assert rag_corpus.similarity_report(texts, {"ref/kernel.cpp": TILIZE_HEADER}, report_at=1.0,
                                        api_prefixes=prefixes) == [rag_corpus.SimilarFile("a/other.cpp", 1.0,
                                                                                          "ref/kernel.cpp")]


def test_two_quoted_lines_are_not_a_run_and_stay(built: Built) -> None:
    pair = distinctive_lines(tracked_references()["assets/bench/tt-pairs-v0/eltwise_binary/tt/eltwise_binary.cpp"], 2)
    near = "\n".join(c.text for c in built.index.chunks if c.path == "docs/source/near.md")
    assert all(line in near for line in pair)
    assert not [row for row in built.manifest["excised"] if row["path"] == "docs/source/near.md"]


def test_a_copy_of_a_tier_a_program_is_excluded_whole(built: Built) -> None:
    reason = built.reason("tt_metal/programming_examples/copycat/copycat.cpp")
    assert reason.startswith("tier-a-overlap: ") and "tt/eltwise_binary.cpp" in reason
    assert "tt_metal/programming_examples/hello/hello.cpp" in indexed_paths(built)


def test_a_listed_page_and_a_file_that_is_not_utf8_are_excluded(built: Built) -> None:
    assert built.reason("docs/source/walkthrough.rst") == "listed: a synthetic walkthrough"
    assert built.reason("docs/source/latin1.rst") == "not-utf8"


def test_a_query_for_tilize_nfaces_returns_its_declaration_chunk(built: Built) -> None:
    lines = TILIZE_HEADER.split("\n")
    declared = next(n for n, line in enumerate(lines, start=1) if "tilize_nfaces(" in line and "untilize" not in line)
    top = built.index.search("tilize_nfaces", 3)[0]
    assert top.path == "tt_metal/api/tt-metalium/tilize_utils.hpp"
    assert top.start_line <= declared <= top.end_line
    assert "std::vector<T> tilize_nfaces(const std::vector<T>& input" in top.text


def test_the_manifest_records_every_file_hash_and_the_index_hash(built: Built) -> None:
    manifest = built.manifest
    for row in manifest["files"] + manifest["excluded"]:
        assert row["sha256"] == hashlib.sha256(built.files[row["path"]]).hexdigest(), row["path"]
    data = built.index_path.read_bytes()
    assert manifest["index"]["sha256"] == hashlib.sha256(data).hexdigest() == built.index.sha256
    assert manifest["index"]["bytes"] == len(data)
    assert manifest["index"]["files"] == len(manifest["files"])
    assert manifest["index"]["chunks"] == sum(row["chunks"] for row in manifest["files"]) == len(built.index)
    assert (manifest["index"]["format"], manifest["index"]["tokenizer"]) == (INDEX_FORMAT, TOKENIZER)
    assert manifest["corpus_sha256"] == hashlib.sha256(built.spec.read_bytes()).hexdigest()
    assert manifest["source"] == {"pin": "tt-metal", "repo": PIN["URL"], "commit": PIN["COMMIT"]}
    assert manifest["tier_a"]["references"] == [
        {"path": path, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        for path, text in tracked_references().items()]
    assert built.manifest_path.read_bytes().isascii()


def test_a_rebuild_gives_the_same_bytes_and_check_agrees(built: Built, tmp_path: Path) -> None:
    index, manifest = tmp_path / "index.json", tmp_path / "MANIFEST.yaml"
    assert run_tool("build", "--tree", built.tree, "--corpus", built.spec, "--index", index,
                    "--manifest", manifest) == 0
    assert index.read_bytes() == built.index_path.read_bytes()
    assert manifest.read_bytes() == built.manifest_path.read_bytes()
    assert run_tool("check", "--tree", built.tree, "--corpus", built.spec, "--index", built.index_path,
                    "--manifest", built.manifest_path) == 0


def test_check_names_a_changed_file(built: Built, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    changed = tmp_path / "tree"
    synthetic_tree(changed)
    (changed / "tech_reports" / "notes.md").write_text("# Notes\n\nChanged.\n", encoding="utf-8")
    assert run_tool("check", "--tree", changed, "--corpus", built.spec, "--index", built.index_path,
                    "--manifest", built.manifest_path) == 1
    err = capsys.readouterr().err
    assert "files differs" in err and "index.json has sha256" in err


# ---------------------------------------------------------------------------
# The tracked manifest


# Files the tracked spec must keep excluded whole, pinned here so that dropping one from corpus.yaml fails: the
# walkthroughs and code found by the Rule 5 audits of the retriever (each reason is in corpus.yaml).
PINNED_EXCLUSIONS = (
    "METALIUM_GUIDE.md",
    "docs/source/tt-metalium/tt_metal/examples/custom_sfpi_add.rst",
    "docs/source/tt-metalium/tt_metal/examples/custom_sfpi_smoothstep.rst",
    "tech_reports/prog_examples/sfpu_eltwise_chain/sfpu_eltwise_chain.md",
    "tt_metal/programming_examples/add_2_integers_in_compute/add_2_integers_in_compute.md",
    "tt_metal/programming_examples/vecadd_multi_core/kernels/interleaved_tile_read_multi_core.cpp",
    "tt_metal/programming_examples/vecadd_multi_core/vecadd_multi_core.cpp",
    "tt_metal/programming_examples/distributed/3_distributed_eltwise_add/distributed_eltwise_add.cpp",
)
# The reasons an excised row may give: a reference run's lines, or a piece of a cut chunk too weak to stand alone.
EXCISED_REASONS = ("tier-a-run: ", "short-piece: ", "no-text: ", "unpaired-fence: ")


@pytest.fixture(scope="module")
def tracked_manifest() -> dict:
    """Return the tracked manifest, read and shape-checked."""
    return rag_corpus.read_manifest(MANIFEST)


def test_the_tracked_manifest_names_the_pin_and_the_tracked_spec(tracked_manifest: dict) -> None:
    assert MANIFEST.read_bytes().isascii()
    assert tracked_manifest["corpus_sha256"] == hashlib.sha256(SPEC.read_bytes()).hexdigest()
    assert tracked_manifest["source"] == {"pin": "tt-metal", "repo": PIN["URL"], "commit": PIN["COMMIT"]}
    index = tracked_manifest["index"]
    assert (index["format"], index["tokenizer"]) == (INDEX_FORMAT, TOKENIZER)
    assert index["files"] == len(tracked_manifest["files"])
    assert index["chunks"] == sum(row["chunks"] for row in tracked_manifest["files"])


def test_the_tracked_manifest_records_the_current_tier_a_references(tracked_manifest: dict) -> None:
    """A changed tracked program makes the manifest stale: rebuild with tools/build_tt_rag.py build."""
    assert tracked_manifest["tier_a"]["references"] == [
        {"path": path, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        for path, text in tracked_references().items()]
    assert {row["path"]: row["item"] for row in tracked_manifest["tier_a"]["dirs"]} == tier_a_dirs_from_suite()


def test_the_tracked_manifest_indexes_nothing_of_a_tier_a_item_or_a_listed_page(tracked_manifest: dict) -> None:
    dirs = tier_a_dirs_from_suite()
    indexed = [row["path"] for row in tracked_manifest["files"]]
    assert not [path for path in indexed if any(path.startswith(f"{d}/") for d in dirs)]
    reasons = {row["path"]: row["reason"] for row in tracked_manifest["excluded"]}
    for entry in yaml.safe_load(SPEC.read_text(encoding="utf-8"))["exclude"]:
        assert reasons.get(entry["path"]) == f"listed: {entry['reason']}"
    for directory, item in dirs.items():
        assert any(path.startswith(f"{directory}/") and reason == f"tier-a-item: {item}"
                   for path, reason in reasons.items()), directory
    listed = {entry["path"] for entry in yaml.safe_load(SPEC.read_text(encoding="utf-8"))["exclude"]}
    for path in PINNED_EXCLUSIONS:
        assert path in listed and reasons.get(path, "").startswith("listed: "), path
    assert all(row["reason"].startswith(EXCISED_REASONS) for row in tracked_manifest["excised"])
    paths = indexed + list(reasons)
    assert len(paths) == len(set(paths)) and indexed == sorted(indexed)


def test_the_tracked_similarity_report_names_indexed_code_files_at_or_above_its_line(tracked_manifest: dict) -> None:
    report_at = yaml.safe_load(SPEC.read_text(encoding="utf-8"))["tier_a"]["similarity"]["report_at"]
    assert tracked_manifest["tier_a"]["similarity"]["report_at"] == report_at
    indexed = {row["path"] for row in tracked_manifest["files"]}
    rows = tracked_manifest["similar"]
    assert [row["path"] for row in rows] == sorted({row["path"] for row in rows})
    for row in rows:
        assert row["path"] in indexed and not row["path"].endswith((".md", ".rst")), row
        assert report_at <= row["similarity"] <= 1 and row["reference"], row


# ---------------------------------------------------------------------------
# The real tree and index, when this host has them


def real_tree() -> Path:
    """Return the pinned tree (LASSI_TT_RAG_TREE, or the install under $LASSI_TOOLCHAINS); skip when absent."""
    named = os.environ.get("LASSI_TT_RAG_TREE")
    toolchains = os.environ.get("LASSI_TOOLCHAINS")
    tree = Path(named) if named else Path(toolchains) / PIN["PREFIX_NAME"] if toolchains else None
    if tree is None or not ((tree / "lassi-install.txt").is_file() or (tree / "lassi-fetch.txt").is_file()):
        pytest.skip("no pinned tt-metal tree here; fetch one with uv run tools/build_tt_rag.py fetch and name it "
                    "in LASSI_TT_RAG_TREE")
    return tree


def real_index(manifest: dict) -> Index:
    """Return the real index checked against the manifest's sha256; skip when this host has none."""
    named = os.environ.get("LASSI_TT_RAG_INDEX")
    scratch = os.environ.get("LASSI_SCRATCH")
    default = Path(scratch) / "rag" / f"{manifest['corpus']}@{PIN['VERSION']}" / "index.json" if scratch else None
    path = Path(named) if named else default
    if path is None or not path.is_file():
        pytest.skip("no built index here; build one with uv run tools/build_tt_rag.py build")
    return load_index(path, sha256=manifest["index"]["sha256"])


def test_the_real_index_holds_no_run_of_a_tier_a_reference(tracked_manifest: dict) -> None:
    """Against the tracked programs always, and the upstream Tier A files when the pinned tree is here too."""
    index = real_index(tracked_manifest)
    references = tracked_references()
    if os.environ.get("LASSI_TT_RAG_TREE") or os.environ.get("LASSI_TOOLCHAINS"):
        tree = real_tree()
        for directory in tier_a_dirs_from_suite():
            for path in sorted((tree / directory).rglob("*")):
                if path.is_file() and path.suffix in (".cpp", ".hpp", ".h", ".md"):
                    references[path.relative_to(tree).as_posix()] = path.read_text(encoding="utf-8")
    assert offending_chunks(index, reference_runs(references)) == []


def test_the_real_index_returns_the_tilize_nfaces_declaration(tracked_manifest: dict) -> None:
    top = real_index(tracked_manifest).search("tilize_nfaces", 3)[0]
    assert top.path == "tt_metal/api/tt-metalium/tilize_utils.hpp"
    assert "tilize_nfaces(" in top.text


def test_the_real_tree_rebuilds_the_tracked_manifest() -> None:
    tree = real_tree()
    assert run_tool("check", "--tree", tree) == 0
