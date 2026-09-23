"""Tests for the FILE-block format in lassi/core/files.py: the writer (P0.4) and the parser (P0.5).

Multi-file model output uses one fenced block per file whose first line
inside the fence is `// FILE: <relative path>` (bible Execution Backends,
Harness Contract). render_file_blocks writes that format: the mock LLM backend
uses it to return a bench item's reference target, and the P0.5 parser reads
it. The fence is max(3, longest backtick run in the file text + 1) backticks,
and the language tag comes from language_for, the one suffix table that
trial.md also uses (the golden test in tests/core/test_trial_md.py checks that
trial.md output does not change). Every expected text below was written by
hand from those rules; no value in this module is a measurement.

parse_file_blocks reads model output back into files. It splits on "\\n" only,
closes a block only on a fence at least as long as the opening one, and turns
each problem into a compile-stage error Diagnostic that tells the model what to
fix: a refused path (absolute, "..", backslash, control character), a second
block for the same path, a block the text never closes, and an expected file
with no block, which the Harness Contract makes a build error fed back to the
model. The parser is looked up at call time (see parse), so the writer tests
run on their own.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import pytest

from lassi.core import files as file_blocks
from lassi.core import trial_md
from lassi.core.record import Diagnostic

fence_for = file_blocks.fence_for
language_for = file_blocks.language_for
render_file_blocks = file_blocks.render_file_blocks

FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
PROJECT_NAMES = ("lassi-repro", "lassi-ee", "lassi-df", "hecbench", "qwen", "wizardcoder", "a100", "mi300x", "gpt-oss")
E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"

# The suffix table trial.md used before P0.4 (lassi/core/trial_md.py at P0.3); language_for must keep it.
LANGUAGE_TABLE = {
    ".cu": "cuda",
    ".cuh": "cuda",
    ".c": "c",
    ".h": "cpp",
    ".hpp": "cpp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".f": "fortran",
    ".f90": "fortran",
    ".f95": "fortran",
    ".py": "python",
    ".rs": "rust",
    ".cs": "csharp",
    ".mlir": "mlir",
}

TWO_FILES = {
    "src/main.cu": '#include "kernels/scale.cuh"\nint main() { return 0; }\n',
    "kernels/scale.cuh": "__device__ float scale(float x);\n",
}
TWO_FILES_TEXT = (
    "```cuda\n"
    "// FILE: kernels/scale.cuh\n"
    "__device__ float scale(float x);\n"
    "```\n"
    "\n"
    "```cuda\n"
    "// FILE: src/main.cu\n"
    '#include "kernels/scale.cuh"\n'
    "int main() { return 0; }\n"
    "```\n"
)


def module_source(name: str) -> str:
    """Return the ASCII source text of an importable module, failing when it does not exist or is not ASCII."""
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin, f"missing module {name}"
    raw = Path(spec.origin).read_bytes()
    assert raw.isascii(), f"{name} has non-ASCII source text"
    return raw.decode("ascii")


def public_defs(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    """Return public top-level classes and functions, plus the public methods of public classes."""
    found: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, *FUNCTION_NODES)) or node.name.startswith("_"):
            continue
        found.append((node.name, node))
        if isinstance(node, ast.ClassDef):
            found += [(f"{node.name}.{item.name}", item) for item in node.body if isinstance(item, FUNCTION_NODES)]
    return [(name, node) for name, node in found if not name.split(".")[-1].startswith("_")]


def is_typed(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True when every parameter except self or cls, and the return value, are annotated."""
    args = node.args
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    params += [a for a in (args.vararg, args.kwarg) if a is not None]
    params = [p for p in params if p.arg not in ("self", "cls")]
    return node.returns is not None and all(p.annotation is not None for p in params)


# ---------------------------------------------------------------------------
# language_for


@pytest.mark.parametrize(("suffix", "language"), sorted(LANGUAGE_TABLE.items()))
def test_language_for_keeps_the_trial_md_table(suffix: str, language: str) -> None:
    assert language_for(f"src/file{suffix}") == language
    assert language_for(f"src/FILE{suffix.upper()}") == language


@pytest.mark.parametrize("path", ["Makefile", "notes.txt", "doc/readme.md", "run.sh", "data.bin", "a.cu.bak"])
def test_language_for_falls_back_to_text(path: str) -> None:
    assert language_for(path) == "text"


def test_language_for_reads_only_the_last_suffix() -> None:
    assert language_for("build.v2/kernel.cu") == "cuda"
    assert language_for("archive.tar.c") == "c"


def test_trial_md_uses_the_files_table() -> None:
    # One table serves both: trial_md imports it from lassi.core.files and keeps no suffix table of its own.
    tree = ast.parse(module_source("lassi.core.trial_md"))
    imports_files = any(
        isinstance(node, ast.ImportFrom)
        and node.level == 0
        and (
            node.module == "lassi.core.files"
            or (node.module == "lassi.core" and any(alias.name == "files" for alias in node.names))
        )
        for node in ast.walk(tree)
    )
    assert imports_files, "lassi.core.trial_md does not import from lassi.core.files"
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert not literals & set(LANGUAGE_TABLE), "lassi.core.trial_md still holds its own suffix table"
    assert not hasattr(trial_md, "_LANGUAGES")


# ---------------------------------------------------------------------------
# fence_for: the one fence rule, shared with trial.md


@pytest.mark.parametrize(
    ("text", "fence"),
    [("", "```"), ("no ticks\n", "```"), ("a `b` ``c``\n", "```"), ("x ``` y\n", "````"), ("`" * 5, "`" * 6)],
    ids=["empty", "none", "short-runs", "triple", "five"],
)
def test_fence_for_is_one_longer_than_the_longest_run_and_at_least_three(text: str, fence: str) -> None:
    assert fence_for(text) == fence


def test_trial_md_uses_the_files_fence_rule() -> None:
    assert trial_md.fence_for is fence_for
    assert not hasattr(trial_md, "_BACKTICK_RUN"), "lassi.core.trial_md still holds its own fence rule"


# ---------------------------------------------------------------------------
# render_file_blocks: exact text


def test_two_files_render_exactly_in_sorted_order() -> None:
    assert render_file_blocks(TWO_FILES) == TWO_FILES_TEXT


def test_insertion_order_does_not_matter() -> None:
    reordered = dict(reversed(list(TWO_FILES.items())))
    assert list(reordered) != list(TWO_FILES)
    assert render_file_blocks(reordered) == TWO_FILES_TEXT
    assert render_file_blocks(MappingProxyType(TWO_FILES)) == TWO_FILES_TEXT


def test_language_tags_follow_the_suffix() -> None:
    files = {"solver.f90": "end\n", "lib/util.c": "int util(void);\n", "Makefile": "all:\n\tnvcc main.cu\n"}
    expected = (
        "```text\n// FILE: Makefile\nall:\n\tnvcc main.cu\n```\n"
        "\n"
        "```c\n// FILE: lib/util.c\nint util(void);\n```\n"
        "\n"
        "```fortran\n// FILE: solver.f90\nend\n```\n"
    )
    assert render_file_blocks(files) == expected


def test_single_file_is_one_block() -> None:
    assert render_file_blocks({"main.cu": "int x;\n"}) == "```cuda\n// FILE: main.cu\nint x;\n```\n"


def test_a_backtick_run_lengthens_that_block_fence_only() -> None:
    files = {"doc/notes.md": "Use ```cuda fences and `````x````` runs.\n", "main.cu": "int x; // `a` ``b``\n"}
    expected = (
        "``````text\n// FILE: doc/notes.md\nUse ```cuda fences and `````x````` runs.\n``````\n"
        "\n"
        "```cuda\n// FILE: main.cu\nint x; // `a` ``b``\n```\n"
    )
    assert render_file_blocks(files) == expected


def test_a_triple_backtick_run_gets_a_four_backtick_fence() -> None:
    files = {"kernels/scale.cuh": "// Usage:\n// ```\n// out[i] = scale(in[i]);\n// ```\n"}
    expected = "````cuda\n// FILE: kernels/scale.cuh\n// Usage:\n// ```\n// out[i] = scale(in[i]);\n// ```\n````\n"
    assert render_file_blocks(files) == expected


def test_a_missing_final_newline_is_added() -> None:
    assert render_file_blocks({"main.cu": "int main() {}"}) == "```cuda\n// FILE: main.cu\nint main() {}\n```\n"


def test_file_text_is_otherwise_copied_unchanged() -> None:
    # Blank lines, trailing spaces, tabs, CRLF line breaks, and non-ASCII text are kept byte for byte.
    text = f"\n\nint a;  \r\nint b;\t\r\n// caf{E_ACUTE}\n\n"
    assert render_file_blocks({"win.c": text}) == f"```c\n// FILE: win.c\n{text}```\n"


def test_output_ends_with_exactly_one_newline() -> None:
    out = render_file_blocks({"a.py": "x = 1\n\n\n", "b.py": "y = 2"})
    assert out.endswith("```\n")
    assert not out.endswith("\n\n")
    assert out == "```python\n// FILE: a.py\nx = 1\n\n\n```\n\n```python\n// FILE: b.py\ny = 2\n```\n"


def test_dotted_names_that_are_not_parent_segments_are_allowed() -> None:
    files = {"notes/v1..v2.txt": "a\n", "..hidden/x.c": "b\n"}
    expected = "```c\n// FILE: ..hidden/x.c\nb\n```\n\n```text\n// FILE: notes/v1..v2.txt\na\n```\n"
    assert render_file_blocks(files) == expected


def test_inner_spaces_and_non_ascii_letters_are_allowed() -> None:
    files = {f"my dir/caf{E_ACUTE} 2.c": "int x;\n"}
    assert render_file_blocks(files) == f"```c\n// FILE: my dir/caf{E_ACUTE} 2.c\nint x;\n```\n"


# ---------------------------------------------------------------------------
# render_file_blocks: refusals


@pytest.mark.parametrize("files", [{}, MappingProxyType({})], ids=["dict", "mapping-proxy"])
def test_an_empty_mapping_is_refused(files: Mapping[str, str]) -> None:
    with pytest.raises(ValueError):
        render_file_blocks(files)


BAD_PATHS = [
    pytest.param("", id="empty"),
    pytest.param("/abs/main.cu", id="absolute-posix"),
    pytest.param("/main.cu", id="root-file"),
    pytest.param("C:/src/main.cu", id="drive-slash"),
    pytest.param("c:main.cu", id="drive-relative"),
    pytest.param("Z:", id="bare-drive"),
    pytest.param("src\\main.cu", id="backslash"),
    pytest.param("..", id="dotdot"),
    pytest.param("../main.cu", id="dotdot-first"),
    pytest.param("src/../main.cu", id="dotdot-middle"),
    pytest.param("src/..", id="dotdot-last"),
    # The FILE line must stay one line for any line splitter, str.splitlines included.
    pytest.param("a\nb.c", id="newline"),
    pytest.param("a\rb.c", id="carriage-return"),
    pytest.param("a\x00b.c", id="nul"),
    pytest.param("a\x0bb.c", id="vertical-tab"),
    pytest.param("a\x1cb.c", id="file-separator"),
    pytest.param("a\x7fb.c", id="delete"),
    pytest.param("a\x85b.c", id="next-line-c1"),
    pytest.param("a\x9fb.c", id="last-c1"),
    pytest.param("a\N{LINE SEPARATOR}b.c", id="line-separator"),
    pytest.param("a\N{PARAGRAPH SEPARATOR}b.c", id="paragraph-separator"),
    # A normalized path has one spelling, so two keys never name one file.
    pytest.param("src//main.cu", id="empty-segment"),
    pytest.param("src/", id="trailing-slash"),
    pytest.param("./main.cu", id="dot-first"),
    pytest.param("src/./main.cu", id="dot-middle"),
    pytest.param(".", id="dot"),
    pytest.param(" main.cu", id="leading-space"),
    pytest.param("main.cu ", id="trailing-space"),
    pytest.param("main.cu\N{NO-BREAK SPACE}", id="trailing-no-break-space"),
    # A compiler command line reads a word starting with "-" or "@" as an option or a response file.
    pytest.param("-x.cu", id="leading-dash"),
    pytest.param("-optf=a.txt,b.c", id="nvcc-options-file"),
    pytest.param("-o/tmp/x.c", id="output-option"),
    pytest.param("-", id="bare-dash"),
    pytest.param("@rsp.cpp", id="response-file"),
    pytest.param("-dir/a.cu", id="dash-directory"),
]


@pytest.mark.parametrize("path", BAD_PATHS)
def test_a_bad_path_is_refused(path: str) -> None:
    with pytest.raises(ValueError):
        render_file_blocks({path: "int x;\n"})


@pytest.mark.parametrize("path", ["../escape.cu", "/etc/passwd", "a\\b.c", "-run.cu"])
def test_a_bad_path_is_refused_among_good_ones(path: str) -> None:
    with pytest.raises(ValueError):
        render_file_blocks({"a.cu": "int a;\n", path: "int b;\n", "z.cu": "int z;\n"})


def test_a_dash_or_at_sign_after_the_first_character_is_allowed() -> None:
    # Only the first character of the whole path reaches the start of a command-line word.
    files = {"src/-x.cu": "int a;\n", "a-b.cu": "int b;\n", "x@y.c": "int c;\n", "_-@.h": "int d;\n"}
    text = render_file_blocks(files)
    assert "// FILE: src/-x.cu\n" in text
    assert file_blocks.parse_file_blocks(text) == file_blocks.ParsedFiles(files=files, diagnostics=[])


# ---------------------------------------------------------------------------
# parse_file_blocks (P0.5): helpers


def parse(text: str, expected: Sequence[str] | None = None) -> file_blocks.ParsedFiles:
    """Call lassi.core.files.parse_file_blocks, looked up now so the writer tests above run without it.

    With `expected` None the argument is left out, so its default is used.
    """
    if expected is None:
        return file_blocks.parse_file_blocks(text)
    return file_blocks.parse_file_blocks(text, expected)


# (stage, severity, code, file, line, column)
Where = tuple[str, str, str | None, str | None, int | None, int | None]


def where(diagnostic: Diagnostic) -> Where:
    """Return every Diagnostic field except the message, whose wording is left to the parser."""
    d = diagnostic
    return (d.stage, d.severity, d.code, d.file, d.line, d.column)


def problem(code: str, file: str | None, line: int | None) -> Where:
    """Return what where() gives for a parser problem: a compile-stage error with no column."""
    return ("compile", "error", code, file, line, None)


def outcome(text: str, expected: Sequence[str] | None = None) -> tuple[dict[str, str], list[Where]]:
    """Parse `text` and return its files and the where() of each diagnostic, checking the record types."""
    parsed = parse(text, expected)
    assert isinstance(parsed.files, dict)
    assert isinstance(parsed.diagnostics, list)
    for diagnostic in parsed.diagnostics:
        assert isinstance(diagnostic, Diagnostic)
        assert isinstance(diagnostic.message, str) and diagnostic.message.strip(), diagnostic
    return parsed.files, [where(diagnostic) for diagnostic in parsed.diagnostics]


# ---------------------------------------------------------------------------
# parse_file_blocks: the record and the round trip


def test_parsed_files_is_a_frozen_record() -> None:
    parsed = parse(TWO_FILES_TEXT)
    assert isinstance(parsed, file_blocks.ParsedFiles)
    assert [f.name for f in dataclasses.fields(file_blocks.ParsedFiles)] == ["files", "diagnostics"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        parsed.files = {}  # type: ignore[misc]
    assert (parsed.files, parsed.diagnostics) == (TWO_FILES, [])


ROUND_TRIPS = [
    pytest.param(TWO_FILES, id="two-files-in-subdirectories"),
    pytest.param({"doc/usage.md": "Build it:\n```sh\nnvcc -o main main.cu\n```\n"}, id="four-backtick-fence"),
    pytest.param(
        {"doc/nested.md": "`````\n````md\n```\n````\n`````\n", "a.cu": "int a; // ``x``\n"},
        id="nested-shorter-fences",
    ),
    pytest.param({"win.c": "int a;\r\nint b;\r\n\r\n"}, id="crlf"),
    pytest.param(
        {"page.c": "/* one */\n\x0c\n/* two */\x0b\x1c\x1d\x1e\x85\N{LINE SEPARATOR}\N{PARAGRAPH SEPARATOR}\n"},
        id="form-feed-and-other-line-breaks",
    ),
    pytest.param(
        {
            "a.py": "\n\nx = 1  \n\t\n\n",
            "b/c/d.h": "   ```   \n",
            "Makefile": "all:\n\tnvcc main.cu\n",
            f"my dir/caf{E_ACUTE} 2.c": f"// caf{E_ACUTE}\n",
            "src/kernels/scale.cuh": "__device__ float scale(float x);\n",
        },
        id="several-files",
    ),
    pytest.param({"a.c": "// FILE: b.c\nint x;\n"}, id="file-marker-inside-a-file"),
    pytest.param({"notes/v1..v2.txt": "a\n", "..hidden/x.c": "b\n"}, id="dotted-names"),
]


@pytest.mark.parametrize("files", ROUND_TRIPS)
def test_round_trip_through_render_file_blocks(files: dict[str, str]) -> None:
    text = render_file_blocks(files)
    assert outcome(text) == (files, [])
    assert outcome(text, list(files)) == (files, [])


def test_a_text_without_final_newline_comes_back_with_one() -> None:
    files = {"main.cu": "int main() {}", "empty.c": "", "cr.c": "x\r"}
    expected = {"main.cu": "int main() {}\n", "empty.c": "\n", "cr.c": "x\r\n"}
    assert outcome(render_file_blocks(files)) == (expected, [])


TEXT_PIECES = (
    "a",
    "int x;",
    " ",
    "\t",
    "`",
    "``",
    "```",
    "````",
    "\n",
    "\r",
    "\r\n",
    "\x0c",
    "\x0b",
    "\x1c",
    "\x85",
    "\N{LINE SEPARATOR}",
    E_ACUTE,
    "// FILE: y.c",
    "   ```",
    "```cuda",
    "~~~",
)
PATH_SEGMENTS = ("src", "k", "a.cu", "b.cuh", "c d.c", "v1..v2", "..x", f"caf{E_ACUTE}.h", "Makefile")


def random_files(rng: random.Random) -> dict[str, str]:
    """Return a mapping render_file_blocks accepts: 1 to 4 valid paths, texts joined from TEXT_PIECES."""
    files: dict[str, str] = {}
    for _ in range(rng.randint(1, 4)):
        path = "/".join(rng.choice(PATH_SEGMENTS) for _ in range(rng.randint(1, 3)))
        files[path] = "".join(rng.choice(TEXT_PIECES) for _ in range(rng.randint(0, 12)))
    return files


def test_round_trip_over_seeded_random_mappings() -> None:
    rng = random.Random(20260923)
    for _ in range(300):
        files = random_files(rng)
        expected = {path: text if text.endswith("\n") else text + "\n" for path, text in files.items()}
        assert outcome(render_file_blocks(files)) == (expected, []), files


# ---------------------------------------------------------------------------
# parse_file_blocks: model responses


def test_blocks_are_found_among_prose() -> None:
    text = (
        "Here is the translation.\n"
        "\n"
        "```cuda\n"
        "// FILE: main.cu\n"
        '#include "kernels/scale.cuh"\n'
        "int main() { return 0; }\n"
        "```\n"
        "\n"
        "The header stays small:\n"
        "\n"
        "```cpp\n"
        "// FILE: kernels/scale.cuh\n"
        "__device__ float scale(float x);\n"
        "```\n"
        "Done.\n"
    )
    files = {
        "main.cu": '#include "kernels/scale.cuh"\nint main() { return 0; }\n',
        "kernels/scale.cuh": "__device__ float scale(float x);\n",
    }
    assert outcome(text) == (files, [])


def test_blocks_without_a_file_line_are_skipped() -> None:
    text = (
        "The format looks like this:\n"
        "```text\n"
        "Example:\n"
        "// FILE: fake.cu\n"
        "```\n"
        "```cuda\n"
        "__global__ void k() {}\n"
        "```\n"
        "```cuda\n"
        "\n"
        "// FILE: late.cu\n"
        "int late;\n"
        "```\n"
        "// FILE: outside.cu\n"
        "```cuda\n"
        "// FILE: real.cu\n"
        "int real;\n"
        "```\n"
    )
    assert outcome(text) == ({"real.cu": "int real;\n"}, [])


def test_file_line_spacing_is_flexible() -> None:
    text = "```cuda\n\t  // FILE:   src/main.cu  \t\r\nint x;\n```\n```c\n// FILE:util.c\nint u;\n```\n"
    assert outcome(text) == ({"src/main.cu": "int x;\n", "util.c": "int u;\n"}, [])


@pytest.mark.parametrize(
    ("written", "normalized"),
    [
        ("./main.cu", "main.cu"),
        ("src//a.cu", "src/a.cu"),
        ("src/./b.cu", "src/b.cu"),
        ("././c.cu", "c.cu"),
        ("./src/.//d.cu", "src/d.cu"),
    ],
)
def test_dot_and_empty_segments_are_dropped(written: str, normalized: str) -> None:
    assert outcome(f"```cuda\n// FILE: {written}\nint x;\n```\n") == ({normalized: "int x;\n"}, [])


def test_a_crlf_response_keeps_its_carriage_returns() -> None:
    text = "Here you go:\r\n```cuda\r\n// FILE: main.cu\r\nint main() {\r\n  return 0;\r\n}\r\n```\r\n"
    assert outcome(text) == ({"main.cu": "int main() {\r\n  return 0;\r\n}\r\n"}, [])
    bad = "```cuda\r\n// FILE: ../x.cu\r\nint x;\r\n```\r\n"
    assert outcome(bad) == ({}, [problem("bad-path", "../x.cu", 2)])


# ---------------------------------------------------------------------------
# parse_file_blocks: fences


def test_fences_may_have_up_to_three_leading_spaces() -> None:
    text = (
        " ```cuda\n// FILE: one.cu\nint one;\n ```\n"
        "    ```cuda\n    // FILE: four.cu\n    int four;\n    ```\n"
        "   ```cuda\n// FILE: three.cu\nint three;\n   ```\n"
    )
    assert outcome(text) == ({"one.cu": "int one;\n", "three.cu": "int three;\n"}, [])


def test_an_indented_fence_line_inside_a_block_is_content() -> None:
    text = "```cuda\n// FILE: c.cu\n    ```\nint c;\n```\n"
    assert outcome(text) == ({"c.cu": "    ```\nint c;\n"}, [])


def test_a_longer_closing_fence_and_trailing_blanks_close_the_block() -> None:
    text = "```cuda\n// FILE: a.cu\nint a;\n``````\n```cuda\n// FILE: b.cu\nint b;\n```  \t\n"
    assert outcome(text) == ({"a.cu": "int a;\n", "b.cu": "int b;\n"}, [])


def test_a_fence_line_with_an_info_string_does_not_close() -> None:
    text = "```cuda\n// FILE: a.cu\nint a;\n```cuda\nint b;\n```\n"
    assert outcome(text) == ({"a.cu": "int a;\n```cuda\nint b;\n"}, [])


def test_an_info_string_with_a_backtick_is_not_an_opening_fence() -> None:
    # Line 1 is not a fence, so the FILE line is outside any block; line 4 opens a block with no content.
    assert outcome("```cuda `x`\n// FILE: a.cu\nint a;\n```\n") == ({}, [])


# ---------------------------------------------------------------------------
# parse_file_blocks: diagnostics


BAD_PARSED_PATHS = [
    pytest.param("/etc/passwd", id="absolute"),
    pytest.param("/main.cu", id="root-file"),
    pytest.param("//main.cu", id="double-slash-root"),
    pytest.param("C:/src/main.cu", id="drive-slash"),
    pytest.param("c:main.cu", id="drive-relative"),
    pytest.param("src\\main.cu", id="backslash"),
    pytest.param("../main.cu", id="dotdot-first"),
    pytest.param("src/../main.cu", id="dotdot-middle"),
    pytest.param("./../main.cu", id="dot-then-dotdot"),
    pytest.param("src/..", id="dotdot-last"),
    pytest.param("a\x1bb.cu", id="escape-character"),
    pytest.param("a\x7fb.cu", id="delete"),
    pytest.param("a\x85b.cu", id="next-line-c1"),
    # Words a compiler reads as options: nvcc's options file and output flag, a host-compiler flag, a response file.
    pytest.param("-optf=a.txt,b.c", id="nvcc-options-file"),
    pytest.param("-x.cu", id="leading-dash"),
    pytest.param("./-o.cu", id="dot-slash-dash"),
    pytest.param("-Xcompiler=-specs=evil.c", id="host-compiler-flag"),
    pytest.param("@rsp.cpp", id="response-file"),
]


@pytest.mark.parametrize("path", BAD_PARSED_PATHS)
def test_a_refused_path_drops_its_block(path: str) -> None:
    text = f"```cuda\n// FILE: {path}\nint x;\n```\n```cuda\n// FILE: ok.cu\nint ok;\n```\n"
    assert outcome(text) == ({"ok.cu": "int ok;\n"}, [problem("bad-path", path, 2)])


@pytest.mark.parametrize("line", ["// FILE:", "// FILE:   \t"], ids=["bare", "blank"])
def test_an_empty_path_is_refused_without_a_file(line: str) -> None:
    assert outcome(f"```cuda\n{line}\nint x;\n```\n") == ({}, [problem("bad-path", None, 2)])


def test_a_second_block_for_the_same_path_is_a_duplicate() -> None:
    text = "```cuda\n// FILE: main.cu\nint first;\n```\n\n```cuda\n// FILE: ./main.cu\nint second;\n```\n"
    assert outcome(text) == ({"main.cu": "int first;\n"}, [problem("duplicate-file", "./main.cu", 7)])


UNCLOSED = [
    pytest.param("```cuda\n// FILE: main.cu\nint x;\n", {}, 2, "main.cu", id="text-ends"),
    pytest.param("```cuda\n// FILE: ./main.cu\nint x;\n", {}, 2, "./main.cu", id="path-as-written"),
    pytest.param("```cuda\n// FILE:  \nint x;\n", {}, 2, None, id="no-path"),
    pytest.param("````cuda\n// FILE: main.cu\nint x;\n```\n", {}, 2, "main.cu", id="shorter-fence"),
    pytest.param("```cuda\n// FILE: main.cu\nint x;\n```cuda\n", {}, 2, "main.cu", id="info-string-line"),
    pytest.param("```cuda\n// FILE: main.cu\nint x;\n    ```\n", {}, 2, "main.cu", id="indented-four"),
    pytest.param(
        "```c\n// FILE: a.c\nint a;\n```\n```cuda\n// FILE: b.cu\nint b;",
        {"a.c": "int a;\n"},
        6,
        "b.cu",
        id="after-a-good-block",
    ),
]


@pytest.mark.parametrize(("text", "files", "line", "path"), UNCLOSED)
def test_an_unclosed_block_is_dropped(text: str, files: dict[str, str], line: int, path: str | None) -> None:
    assert outcome(text) == (files, [problem("unclosed-block", path, line)])


def test_an_expected_file_in_an_unclosed_block_is_not_also_missing() -> None:
    # The file has a block, so only unclosed-block is reported: its message says how to fix it.
    for written in ("main.cu", "./main.cu"):
        text = f"```cuda\n// FILE: {written}\nint x;\n"
        assert outcome(text, ["main.cu"]) == ({}, [problem("unclosed-block", written, 2)])
    text = "```cuda\n// FILE: main.cu\nint x;\n"
    assert outcome(text, ["main.cu", "util.h"]) == (
        {},
        [problem("unclosed-block", "main.cu", 2), problem("missing-file", "util.h", None)],
    )


def test_an_expected_file_whose_block_path_is_refused_is_missing() -> None:
    text = "```cuda\n// FILE: /main.cu\nint x;\n```\n"
    expected = [problem("bad-path", "/main.cu", 2), problem("missing-file", "main.cu", None)]
    assert outcome(text, ["main.cu"]) == ({}, expected)


def test_a_block_with_no_lines_gives_a_single_newline() -> None:
    # The file text is the content lines joined with "\n" plus a final "\n", so no lines give "\n".
    for close in ("```\n", "```\r\n", "```"):
        assert outcome(f"```cuda\n// FILE: empty.cu\n{close}") == ({"empty.cu": "\n"}, [])
    assert outcome("```cuda\n// FILE: blank.cu\n\n```\n") == ({"blank.cu": "\n"}, [])
    assert outcome("```cuda\n// FILE: two.cu\n\n\n```\n") == ({"two.cu": "\n\n"}, [])


def test_a_missing_file_names_the_normalized_expected_path() -> None:
    assert outcome("", ["./src//a.cu"]) == ({}, [problem("missing-file", "src/a.cu", None)])
    message = parse("", ["./src//a.cu"]).diagnostics[0].message
    assert "// FILE: src/a.cu" in message, message


def test_a_repeated_expected_path_is_reported_once() -> None:
    assert outcome("", ["a.cu", "a.cu", "./a.cu", "b.cu"]) == (
        {},
        [problem("missing-file", "a.cu", None), problem("missing-file", "b.cu", None)],
    )


@pytest.mark.parametrize("path", ["./", "", "../x.cu", "/abs.cu", "src/../x.cu", "a\\b.cu", "-x.cu", "@r.cpp"])
def test_a_bad_expected_path_raises(path: str) -> None:
    # Expected paths come from the harness, not the model, so a bad one is a caller error.
    with pytest.raises(ValueError, match="expected path"):
        parse("```cuda\n// FILE: main.cu\nint x;\n```\n", ["main.cu", path])


# ---------------------------------------------------------------------------
# parse_file_blocks: what each message tells the model


def only_message(text: str, expected: Sequence[str] = ()) -> str:
    """Parse `text` and return the message of its one diagnostic."""
    diagnostics = parse(text, expected).diagnostics
    assert len(diagnostics) == 1, diagnostics
    return diagnostics[0].message


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("../main.cu", "'..'"),
        ("/etc/passwd", "absolute"),
        ("src\\main.cu", "backslash"),
        ("-optf=a.txt,b.c", "option"),
        ("@rsp.cpp", "option"),
    ],
)
def test_a_bad_path_message_names_the_path_the_reason_and_the_fix(path: str, reason: str) -> None:
    message = only_message(f"```cuda\n// FILE: {path}\nint x;\n```\n")
    assert repr(path) in message, message
    assert "line 2" in message, message
    assert reason in message, message
    assert "relative path" in message, message


def test_a_duplicate_file_message_names_both_lines_and_the_fix() -> None:
    text = "intro\n```cuda\n// FILE: main.cu\nint a;\n```\n```cuda\n// FILE: ./main.cu\nint b;\n```\n"
    message = only_message(text)
    assert "'main.cu'" in message, message
    assert "line 7" in message and "line 3" in message, message
    assert "exactly one block" in message, message


@pytest.mark.parametrize("fence", ["```", "````", "``````"])
def test_an_unclosed_block_message_names_the_closing_fence(fence: str) -> None:
    message = only_message(f"{fence}cuda\n// FILE: cut.cu\nint cut;\n")
    assert "'cut.cu'" in message, message
    assert "line 2" in message, message
    assert message.endswith(f"End the block with a line of {fence}"), message


# ---------------------------------------------------------------------------
# parse_file_blocks: more fence cases


@pytest.mark.parametrize(
    "tail",
    ["\x0c", "\x0b", "\r", " \t\r", "\N{NO-BREAK SPACE}", "\x1c"],
    ids=["form-feed", "vertical-tab", "cr", "blanks-and-cr", "no-break-space", "file-separator"],
)
def test_a_closing_fence_may_end_with_any_whitespace(tail: str) -> None:
    assert outcome(f"```cuda\n// FILE: a.cu\nint a;\n```{tail}\nafter\n") == ({"a.cu": "int a;\n"}, [])


def test_a_closing_fence_with_other_text_is_content() -> None:
    text = "```cuda\n// FILE: a.cu\n``` x\n```\n"
    assert outcome(text) == ({"a.cu": "``` x\n"}, [])


def test_two_backticks_do_not_open_a_block() -> None:
    # Line 1 is not a fence; line 4 opens a fence with nothing after it, so nothing is read.
    assert outcome("``cuda\n// FILE: a.cu\nint a;\n```\n") == ({}, [])


def test_a_missing_expected_file_is_a_compile_error() -> None:
    text = "```cuda\n// FILE: main.cu\nint x;\n```\n"
    expected = ("kernels/scale.cuh", "main.cu", "util/io.h")
    for given in (expected, list(expected)):
        parsed = parse(text, given)
        assert parsed.files == {"main.cu": "int x;\n"}
        assert [where(d) for d in parsed.diagnostics] == [
            problem("missing-file", "kernels/scale.cuh", None),
            problem("missing-file", "util/io.h", None),
        ]
        for diagnostic, path in zip(parsed.diagnostics, ("kernels/scale.cuh", "util/io.h"), strict=True):
            assert f"// FILE: {path}" in diagnostic.message, diagnostic.message
            assert "block" in diagnostic.message.lower(), diagnostic.message


def test_expected_paths_are_normalized_before_comparing() -> None:
    text = "```cuda\n// FILE: ./a.cu\nint a;\n```\n```cuda\n// FILE: src/b.cu\nint b;\n```\n"
    files = {"a.cu": "int a;\n", "src/b.cu": "int b;\n"}
    assert outcome(text, ["a.cu", "src//b.cu", "./src/./b.cu"]) == (files, [])


def test_files_that_were_not_expected_are_kept() -> None:
    text = "```cuda\n// FILE: main.cu\nint x;\n```\n```cuda\n// FILE: extra.cuh\nint y;\n```\n"
    assert outcome(text, ["main.cu"]) == ({"main.cu": "int x;\n", "extra.cuh": "int y;\n"}, [])


def test_empty_text_with_expected_files() -> None:
    assert outcome("") == ({}, [])
    assert outcome("", ["main.cu"]) == ({}, [problem("missing-file", "main.cu", None)])


def test_line_numbers_count_newlines_only() -> None:
    # Form feeds, NEL, file separators, and line separators before the block are not line breaks here.
    text = "Intro\x0cpage\N{LINE SEPARATOR}two\x85three\x1cfour\r\n```cuda\n// FILE: ../x.cu\nint x;\n```\n"
    assert outcome(text) == ({}, [problem("bad-path", "../x.cu", 3)])


def test_diagnostics_come_in_text_order_with_missing_files_last() -> None:
    text = (
        "```cuda\n"  # 1
        "// FILE: /abs.cu\n"  # 2: bad-path
        "int a;\n"
        "```\n"
        "```cuda\n"  # 5
        "// FILE: ok.cu\n"  # 6
        "int ok;\n"
        "```\n"
        "```cuda\n"  # 9
        "// FILE: ./ok.cu\n"  # 10: duplicate-file
        "int again;\n"
        "```\n"
        "```cuda\n"  # 13
        "// FILE: cut.cu\n"  # 14: unclosed-block
        "int cut"
    )
    assert outcome(text, ["z.cu", "ok.cu", "a.cu"]) == (
        {"ok.cu": "int ok;\n"},
        [
            problem("bad-path", "/abs.cu", 2),
            problem("duplicate-file", "./ok.cu", 10),
            problem("unclosed-block", "cut.cu", 14),
            problem("missing-file", "z.cu", None),
            problem("missing-file", "a.cu", None),
        ],
    )


# ---------------------------------------------------------------------------
# Module hygiene (Agent Rule 3, Readability Standards)


def test_files_module_documents_the_format() -> None:
    tree = ast.parse(module_source("lassi.core.files"))
    doc = ast.get_docstring(tree)
    assert doc, "lassi.core.files has no module docstring"
    assert "// FILE:" in doc, doc


def test_files_module_documented_typed_ascii() -> None:
    tree = ast.parse(module_source("lassi.core.files"))
    undocumented = [qual for qual, node in public_defs(tree) if not ast.get_docstring(node)]
    assert not undocumented, f"lassi.core.files: no docstring on {undocumented}"
    untyped = [qual for qual, node in public_defs(tree) if isinstance(node, FUNCTION_NODES) and not is_typed(node)]
    assert not untyped, f"lassi.core.files: missing type hints on {untyped}"
    names = {qual for qual, _ in public_defs(tree)}
    assert {"fence_for", "language_for", "render_file_blocks"} <= names


def test_files_module_holds_no_project_code() -> None:
    lowered = module_source("lassi.core.files").lower()
    found = [word for word in PROJECT_NAMES if word in lowered]
    assert not found, f"lassi.core.files names projects: {found}"


def test_files_module_defines_the_parser() -> None:
    # The documented-and-typed check above covers these once they exist.
    names = {qual for qual, _ in public_defs(ast.parse(module_source("lassi.core.files")))}
    assert {"ParsedFiles", "parse_file_blocks"} <= names, sorted(names)
