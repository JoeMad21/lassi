"""Tests for the FILE-block format writer in lassi/core/files.py (P0.4).

Multi-file model output uses one fenced block per file whose first line
inside the fence is `// FILE: <relative path>` (bible Execution Backends,
Harness Contract). render_file_blocks writes that format: the mock LLM backend
uses it to return a bench item's reference target, and the P0.5 parser reads
it. The fence is max(3, longest backtick run in the file text + 1) backticks,
and the language tag comes from language_for, the one suffix table that
trial.md also uses (the golden test in tests/core/test_trial_md.py checks that
trial.md output does not change). Every expected text below was written by
hand from those rules; no value in this module is a measurement.
"""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pytest

from lassi.core import files as file_blocks
from lassi.core import trial_md

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
]


@pytest.mark.parametrize("path", BAD_PATHS)
def test_a_bad_path_is_refused(path: str) -> None:
    with pytest.raises(ValueError):
        render_file_blocks({path: "int x;\n"})


@pytest.mark.parametrize("path", ["../escape.cu", "/etc/passwd", "a\\b.c"])
def test_a_bad_path_is_refused_among_good_ones(path: str) -> None:
    with pytest.raises(ValueError):
        render_file_blocks({"a.cu": "int a;\n", path: "int b;\n", "z.cu": "int z;\n"})


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
