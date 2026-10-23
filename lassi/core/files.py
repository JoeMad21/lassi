"""The FILE-block format of multi-file model output (bible Harness Contract).

Each file is one fenced block whose first line inside the fence is
`// FILE: <relative path>`, followed by the file's text:

    ```<lang>
    // FILE: kernels/scale.cuh
    <file text>
    ```

The fence is backticks, max(3, longest backtick run in the file text + 1)
long (fence_for, which trial.md also uses), so a backtick run inside the file
never closes the block early. `<lang>` comes from the file suffix through
language_for, the one suffix table that trial.md also uses (cuda, c, cpp,
fortran, python, rust, csharp, mlir, and text for anything else).

The file text is copied unchanged, except that a final newline is added when
it is missing, so the closing fence sits on its own line. A reader of this
format therefore gets every file back ending with a newline: "x" and "x\\n"
render the same, and so do "" and "\\n".

A path is relative, POSIX, and normalized, so each file has exactly one
spelling: it is not empty, has no leading "/" or drive (such as "C:"), no
backslash, no empty, "." or ".." segment (so no "//" and no trailing "/"), no
leading or trailing whitespace, and no control character or line separator
(the FILE line must stay one line for any line splitter, str.splitlines
included). It also does not start with "-" or "@": a toolchain puts source
paths on a compiler command line, where such a word is read as an option or
a response file, so a file name could change the compile flags. It holds no
lone surrogate (not Unicode text, so it cannot be written as a file name), and
no segment is longer than MAX_SEGMENT_BYTES in UTF-8, the longest file name
Linux accepts (NAME_MAX).

parse_file_blocks reads model output back into files. It splits the text on
"\\n" only, so "\\r", form feeds, and other line breaks stay inside a line. A
block opens on a line of up to 3 spaces, 3 or more backticks, and an optional
info string without backticks; it is a FILE block when its first line inside,
after optional spaces or tabs, starts with `// FILE:`. It closes on the first
later line of up to 3 spaces, at least as many backticks as the opening fence,
and nothing else but whitespace (a final "\\r" included). Other fenced blocks
are skipped whole. Every file written by render_file_blocks reads back
unchanged.

Model output gets one leniency: a leading "./" and any "." or empty segment
are dropped from a path ("./main.cu" reads as "main.cu"). The result must then
pass the same path check as the writer, so absolute paths, drives,
backslashes, ".." segments, and control characters are refused. Each problem
becomes a compile-stage error Diagnostic that tells the model what to fix
(bible Harness Contract: a missing file is a build error fed back to the
model): bad-path, duplicate-file, unclosed-block, and missing-file.

Unfenced FILE lines are a second, opt-in leniency (parse_file_blocks'
unfenced_file_lines, which the recipe key extract.unfenced_file_lines turns
on; off by default). It applies only to a text that holds no FILE block at
all, closed or unclosed; any other text reads exactly as without it. Every
line outside every fenced block that starts, after optional spaces or tabs,
with `// FILE:` then opens a file, which runs to the line before the next
such line or to the end of the text; text before the first is not read. The
text is copied unchanged, a final newline added when missing, and the text's
own final "\\n" ends the last file rather than adding an empty line to it.
The same path normalization and checks apply (bad-path, duplicate-file,
missing-file; never unclosed-block), and each file read this way gets one
parse-stage warning Diagnostic, UNFENCED_FILE, at its FILE line: a note on
how the reply was read, not a build error, so it changes no rung of the
stage ladder, and it is not one of the compile-stage code warnings df-v0
counts. Known cases, not complete: a FILE line followed by a fenced block
reads the fence lines into the file, prose after the last file's code is
read as part of it, and a fence opened inside a file and never closed hides
every later FILE line.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from lassi.core.record import Diagnostic

FILE_MARKER = "// FILE: "
# The code of the parse-stage warning on each file read from an unfenced FILE line (parse_file_blocks).
UNFENCED_FILE = "unfenced-file"
# The longest path segment in UTF-8 bytes: NAME_MAX on Linux, where the build host writes the files.
MAX_SEGMENT_BYTES = 255

_LANGUAGES = {
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
_BACKTICK_RUN = re.compile(r"`+")
_DRIVE = re.compile(r"[A-Za-z]:")
# C0 and C1 controls (NEL, U+0085, among them) and the Unicode line and paragraph separators.
_LINE_BREAKING = re.compile(r"[\x00-\x1f\x7f-\x9f\N{LINE SEPARATOR}\N{PARAGRAPH SEPARATOR}]")
# A lone surrogate code point, which JSON can decode from a model reply but which is not Unicode text.
_SURROGATE = re.compile("[\ud800-\udfff]")

# An opening fence, matched against a whole line: up to 3 spaces, 3 or more backticks, and an
# info string without backticks, for example "```cuda" or "   ````".
_OPENING_FENCE = re.compile(r" {0,3}(?P<fence>`{3,})[^`]*")
# A closing fence, matched against a whole line: up to 3 spaces, backticks (at least as many as the
# opening fence, checked in code), then only whitespace, a final CR included, for example "```  \r".
_CLOSING_FENCE = re.compile(r" {0,3}(?P<fence>`{3,})\s*")
# The first characters a compiler command line reads as an option ("-o") or a response file ("@args").
_OPTION_STARTS = ("-", "@")
# The FILE line, matched at the start of the first line inside a block, for example "// FILE: src/main.cu".
_FILE_LINE = re.compile(r"[ \t]*// FILE:(?P<path>.*)")


def language_for(path: str) -> str:
    """Return the fenced-block language for a file path, from its last suffix in lowercase; 'text' when unknown."""
    return _LANGUAGES.get(PurePosixPath(path).suffix.lower(), "text")


def fence_for(text: str) -> str:
    """Return the backtick fence for `text`: one longer than its longest backtick run, and at least three."""
    longest = max((len(run) for run in _BACKTICK_RUN.findall(text)), default=0)
    return "`" * max(3, longest + 1)


def _check_path(path: str) -> None:
    """Raise ValueError unless `path` is a normalized relative POSIX path that stays inside the build directory."""
    if not path:
        raise ValueError("a file path must not be empty")
    if path.startswith("/") or _DRIVE.match(path):
        raise ValueError(f"file path {path!r} is absolute; it must be relative")
    if "\\" in path:
        raise ValueError(f"file path {path!r} contains a backslash; use '/' between segments")
    if _LINE_BREAKING.search(path):
        raise ValueError(f"file path {path!r} contains a control character or line separator")
    if _SURROGATE.search(path):
        raise ValueError(f"file path {path!r} contains a lone surrogate, which is not Unicode text")
    if path != path.strip():
        raise ValueError(f"file path {path!r} has leading or trailing whitespace")
    if path.startswith(_OPTION_STARTS):
        raise ValueError(
            f"file path {path!r} starts with {path[0]!r}, which a compiler command line reads as an option; "
            "start the name with a letter, a digit, or '_'"
        )
    segments = path.split("/")
    if ".." in segments:
        raise ValueError(f"file path {path!r} has a '..' segment")
    if "" in segments or "." in segments:
        raise ValueError(f"file path {path!r} is not normalized: it has an empty or '.' segment")
    longest = max(len(segment.encode("utf-8")) for segment in segments)
    if longest > MAX_SEGMENT_BYTES:
        raise ValueError(
            f"file path {path!r} has a segment of {longest} bytes in UTF-8; the limit is {MAX_SEGMENT_BYTES}"
        )


def render_file_blocks(files: Mapping[str, str]) -> str:
    """Return `files` (relative path -> text) as FILE blocks in sorted path order.

    Blocks are separated by one blank line and the whole text ends with exactly
    one newline. Each file's text is copied unchanged, except that a final
    newline is added when missing so the closing fence sits on its own line.
    Raises ValueError for an empty mapping or a bad path (see the module docstring).
    """
    if not files:
        raise ValueError("render_file_blocks needs at least one file")
    paths = sorted(files)
    for path in paths:
        _check_path(path)
    return "\n".join(_block(path, files[path]) for path in paths)


def _block(path: str, text: str) -> str:
    """Return one FILE block, ending with its closing fence and a newline."""
    fence = fence_for(text)
    body = text if text.endswith("\n") else text + "\n"
    return f"{fence}{language_for(path)}\n{FILE_MARKER}{path}\n{body}{fence}\n"


@dataclass(frozen=True)
class ParsedFiles:
    """What parse_file_blocks read: the files (relative path -> text) and the problems found, as Diagnostics."""

    files: dict[str, str]
    diagnostics: list[Diagnostic]


def parse_file_blocks(text: str, expected: Sequence[str] = (), *, unfenced_file_lines: bool = False) -> ParsedFiles:
    """Read the FILE blocks in model output `text` (see the module docstring for the format).

    Each file's text is the lines between its FILE line and the closing
    fence joined with "\\n", plus a final "\\n", so every file comes back
    ending with a newline (a block with no lines gives "\\n"). Paths are
    normalized for model output, then checked as render_file_blocks checks
    them. Every problem is a compile-stage error Diagnostic with no column,
    whose file is the path as written (None when the FILE line names none)
    and whose line is the 1-based line of the FILE line in `text`:

    - bad-path: the path is refused; the block is dropped.
    - duplicate-file: a second block for the same path; the first is kept.
    - unclosed-block: the text ends inside the block; it is dropped, since a
      cut-off file must not compile silently.
    - missing-file: a path in `expected` (normalized the same way) has no
      block at all; its file is that normalized path and its line None. An
      expected file whose block is unclosed gets only the unclosed-block
      error, which says how to fix it.

    Diagnostics come in text order, then missing-file in `expected` order,
    once per path. Files that are not in `expected` are kept without a
    diagnostic. `expected` comes from the caller, not the model, so an
    expected path that fails the path check raises ValueError.

    With `unfenced_file_lines` and no FILE block in `text`, closed or
    unclosed, the files are read from its unfenced FILE lines instead (the
    module docstring), with the same checks and line numbers, and each file
    kept also gets a parse-stage UNFENCED_FILE warning, in text order among
    the errors. Any other text reads as without it.
    """
    wanted = _expected_paths(expected)
    lines = text.split("\n")
    blocks = list(_file_blocks(lines))
    if unfenced_file_lines and not blocks:
        blocks = list(_unfenced_files(lines))
    files, named, diagnostics = _read_blocks(blocks)
    diagnostics += [_missing_file(path) for path in wanted if path not in named]
    return ParsedFiles(files=files, diagnostics=diagnostics)


def _read_blocks(blocks: Sequence[_Block]) -> tuple[dict[str, str], set[str], list[Diagnostic]]:
    """Return the files `blocks` give, every normalized path they name, and their diagnostics in text order.

    A block's problem drops it (parse_file_blocks); a kept unfenced block,
    one with no fence, adds its UNFENCED_FILE warning.
    """
    files: dict[str, str] = {}
    first_lines: dict[str, int] = {}
    named: set[str] = set()
    diagnostics: list[Diagnostic] = []
    for block in blocks:
        path = _normalize(block.written)
        named.add(path)
        if block.content is None:
            diagnostics.append(_unclosed_block(block))
            continue
        problem = _path_problem(block, path, first_lines)
        if problem is not None:
            diagnostics.append(problem)
            continue
        files[path] = "\n".join(block.content) + "\n"
        first_lines[path] = block.line
        if not block.fence:
            diagnostics.append(_unfenced_file(block, path))
    return files, named, diagnostics


def _expected_paths(expected: Sequence[str]) -> list[str]:
    """Return the expected paths normalized, without repeats, in order; raise ValueError for a bad one."""
    paths = list(dict.fromkeys(_normalize(path) for path in expected))
    for path in paths:
        try:
            _check_path(path)
        except ValueError as error:
            raise ValueError(f"bad expected path: {error}") from error
    return paths


def _missing_file(path: str) -> Diagnostic:
    """Return the missing-file Diagnostic for an expected path that no FILE block names."""
    message = (
        f"the expected file {path!r} is missing: return it in a fenced block whose first line is `// FILE: {path}`"
    )
    return _parse_error("missing-file", path, None, message)


@dataclass(frozen=True)
class _Block:
    """One fenced block whose first line is a FILE line, or one unfenced FILE line and its file, as found."""

    written: str  # the path as written on the FILE line, with surrounding whitespace stripped
    line: int  # the 1-based line number of the FILE line
    fence: str  # the backticks of the opening fence; "" for an unfenced FILE line
    content: list[str] | None  # the lines after the FILE line; None when no closing fence follows


def _file_blocks(lines: list[str]) -> Iterator[_Block]:
    """Yield the FILE blocks among `lines` in order, skipping every other fenced block whole."""
    index = 0
    while index < len(lines):
        opening = _OPENING_FENCE.fullmatch(lines[index])
        if opening is None:
            index += 1
            continue
        fence = opening["fence"]
        close = _closing_index(lines, index + 1, len(fence))
        header = _FILE_LINE.match(lines[index + 1]) if index + 1 < len(lines) else None
        if header is not None:
            content = None if close is None else lines[index + 2 : close]
            yield _Block(written=header["path"].strip(), line=index + 2, fence=fence, content=content)
        index = len(lines) if close is None else close + 1


def _unfenced_files(lines: list[str]) -> Iterator[_Block]:
    """Yield a block with no fence for each FILE line outside every fenced block, in order.

    Its content is the lines after it up to the next such FILE line or the
    end of the text; the empty string after the text's final "\\n" is left
    out, so that line feed ends the last file instead of adding an empty
    line to it. Lines before the first FILE line are not read.
    """
    headers = [(index, _FILE_LINE.match(lines[index])) for index in _outside_fences(lines)]
    starts = [(index, header["path"].strip()) for index, header in headers if header is not None]
    if not starts:
        return
    ends = [index for index, _ in starts[1:]] + [len(lines)]
    for (start, written), end in zip(starts, ends, strict=True):
        content = lines[start + 1 : end]
        if end == len(lines) and content and content[-1] == "":
            content.pop()
        yield _Block(written=written, line=start + 1, fence="", content=content)


def _outside_fences(lines: list[str]) -> Iterator[int]:
    """Yield the index of each line outside every fenced block; a block the text never closes runs to its end."""
    index = 0
    while index < len(lines):
        opening = _OPENING_FENCE.fullmatch(lines[index])
        if opening is None:
            yield index
            index += 1
            continue
        close = _closing_index(lines, index + 1, len(opening["fence"]))
        index = len(lines) if close is None else close + 1


def _closing_index(lines: list[str], start: int, width: int) -> int | None:
    """Return the index of the first line from `start` on that closes a fence `width` backticks long, else None."""
    for index in range(start, len(lines)):
        closing = _CLOSING_FENCE.fullmatch(lines[index])
        if closing is not None and len(closing["fence"]) >= width:
            return index
    return None


def _normalize(path: str) -> str:
    """Drop "." and empty segments from a relative path, so "./src//a.cu" becomes "src/a.cu".

    A path that starts with "/" is returned unchanged, so the path check
    still refuses it as absolute.
    """
    if path.startswith("/"):
        return path
    return "/".join(segment for segment in path.split("/") if segment not in ("", "."))


def _path_problem(block: _Block, path: str, first_lines: Mapping[str, int]) -> Diagnostic | None:
    """Return the bad-path or duplicate-file Diagnostic for a closed block with normalized `path`, else None.

    The message calls a block with no fence "the unfenced FILE line".
    """
    written = block.written or None
    what, kept = ("the FILE block", "the first block") if block.fence else ("the unfenced FILE line", "the first")
    try:
        _check_path(path)
    except ValueError as error:
        message = (
            f"{what} at line {block.line} was dropped: {error}. Name each file with a relative path "
            "inside the build directory, with '/' between segments, such as 'src/main.cu'"
        )
        return _parse_error("bad-path", written, block.line, message)
    if path in first_lines:
        message = (
            f"{what} at line {block.line} repeats the file {path!r} from line {first_lines[path]}; "
            f"it was dropped and {kept} is kept. Return each file in exactly one block"
        )
        return _parse_error("duplicate-file", written, block.line, message)
    return None


def _unfenced_file(block: _Block, path: str) -> Diagnostic:
    """Return the parse-stage UNFENCED_FILE warning for a file kept from an unfenced FILE line.

    A note on how the reply was read, not a build error: the stage ladder
    reads only errors, and df-v0 counts only compile- and jit-stage warnings.
    """
    message = (
        f"the file {path!r} at line {block.line} was read without a fence; the format asks for one fenced code "
        f"block per file whose first line is `{FILE_MARKER}{path}`, so return each file that way"
    )
    return Diagnostic(
        stage="parse", severity="warning", code=UNFENCED_FILE, file=block.written or None, line=block.line,
        column=None, message=message,
    )


def _unclosed_block(block: _Block) -> Diagnostic:
    """Return the unclosed-block Diagnostic for a block the text never closes."""
    message = (
        f"the FILE block at line {block.line} ({block.written!r}) is never closed, so the file was dropped "
        f"because it may be cut short. End the block with a line of {block.fence}"
    )
    return _parse_error("unclosed-block", block.written or None, block.line, message)


def _parse_error(code: str, file: str | None, line: int | None, message: str) -> Diagnostic:
    """Return a compile-stage error Diagnostic for a problem in the FILE blocks; the column is always None."""
    return Diagnostic(stage="compile", severity="error", code=code, file=file, line=line, column=None, message=message)
