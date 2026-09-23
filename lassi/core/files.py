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
included).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import PurePosixPath

FILE_MARKER = "// FILE: "

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
    if path != path.strip():
        raise ValueError(f"file path {path!r} has leading or trailing whitespace")
    segments = path.split("/")
    if ".." in segments:
        raise ValueError(f"file path {path!r} has a '..' segment")
    if "" in segments or "." in segments:
        raise ValueError(f"file path {path!r} is not normalized: it has an empty or '.' segment")


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
