"""The tt-metal host toolchain adapter, registered as Toolchain "ttmetal-host" (bible Component Interfaces, Toolchain).

TtMetalHost builds a TT host program against the pinned tt-metal tree with
one command, the pinned host clang++-20 compiling and linking at once:

    <clang++-20> HOST_DEFINES HOST_INCLUDES -idirafter . HOST_FLAGS
        HOST_LINK_FLAGS -o main <host sources> HOST_LINK_LIBRARIES

The five HOST_* values are argv words separated by spaces in
toolchains/tt-metal.pin (PIN "tt-metal"), read when the toolchain is
constructed, with {TREE} replaced by the tree in POSIX form. They are the
pinned build's line for the gate's example (build_Release/build.ninja in the
tree), with the changes the pin's comments record: no OVERRIDE_KERNEL_PREFIX
define, no linker dependency file, and no -Werror (OQ-027, option (b)).
"-idirafter ." is the one word added: the build directory is searched after
every pinned include directory for an angle-bracket include, so a header
written beside the program is found at the path the program names (a quoted
include searches the including file's directory first). The
link names the tree's libraries by path with an rpath naming their
directories, so a program needs no loader variable. Nothing runs at build
time, so the build sets no TT_METAL_* variable; the toolchain keeps the tree
as its attribute `tree` for the executor that runs its programs.

Sources: a C++ file (.cpp, .cc, .cxx) with a directory named "kernels" in
its path is a kernel source. build() writes it at its path in the build
directory, beside the artifact, and never gives it to the compiler, since the
kernel JIT compiles it at first launch (bible Harness Contract). At the pin
the JIT looks a kernel up in the process's working directory first
(tt_metal/impl/kernels/kernel.cpp), so an executor that runs the program in
its artifact's directory, as lassi.executors.native does, has the JIT find
the kernel at the path the program names. Every other C++ file is a host
source, compiled in sorted order; headers and harness files are written
beside them and never compiled.

The compiler is the build host's clang++-20, pinned by path and version in
the pin's EXECUTABLE and EXPECT_VERSION, so the class declares no PIN_BIN,
as gcc-native does. Unlike gcc-native it builds against an installed tree,
<resolved toolchains root>/<PREFIX_NAME>, which the stage runner passes as
`tree` after check_tree accepts it, before any process starts
(lassi.core.runner build_toolchain): the tree's lassi-install.txt must name
the pin's NAME and COMMIT on its first line, and its lassi-cpm-sources.txt,
the packages tt-metal's CMake fetched through CPM, must equal
toolchains/tt-metal-cpm-sources.txt line for line; the --version check
follows. It declares `diagnostics` and `emits_warnings`, and no offload
capability: without -Werror a warning in a host program stays a warning.

parse_diagnostics reads the stderr of clang and of ld.lld, which links
through -fuse-ld=lld, with the patterns shared with the other adapters
(lassi.toolchains._stderr):

- A driver line, `<tool>: <severity>: <message>` from clang, clang++, or
  ld.lld with an optional version suffix (such as "clang++-20: error:
  linker command failed with exit code 1 (use -v to see invocation)" or
  "ld.lld: error: undefined symbol: helper(int)"), has no place, and a
  fatal error is an error. It is tried before the place pattern, since its
  message may quote a name the model chose.
- `<file>:<line>:<column>: <severity>: <message>` is one Diagnostic, read
  as the shared GCC style pattern reads it: error, warning, and note as
  printed, `fatal error` read as an error, and a trailing ` [<option>]`
  whose option starts with "-" (clang's `[-Werror,-Wunused-parameter]`
  included) as the code. A leading "./" is dropped from the file, since
  clang may name a header it found beside the including file that way;
  otherwise file and line are kept as printed. The column is clang's as
  printed, kept only for a built file (fold_built_column): a header of the
  pinned tree, a harness file, or a parse with no files gets none.
- Nothing comes from the include chain ("In file included from ..."), the
  source echo and caret lines, the "N errors generated." summary, or
  ld.lld's ">>> " context lines: none of them matches a pattern.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath, PurePosixPath

from lassi.core.record import Diagnostic
from lassi.core.registry import register
from lassi.toolchains import pins
from lassi.toolchains._base import OUTPUT, CommandRunner, CompilerToolchain
from lassi.toolchains._stderr import (
    GCC,
    NO_FILES,
    LinePattern,
    compile_diagnostic,
    fold_built_column,
    parse_stderr,
)

# The pin file stem, and the pin keys that hold the host build line, in the order the command uses them.
PIN_NAME = "tt-metal"
HOST_KEYS = ("HOST_DEFINES", "HOST_INCLUDES", "HOST_FLAGS", "HOST_LINK_FLAGS", "HOST_LINK_LIBRARIES")
# The word a HOST_* value writes for the installed tree.
TREE_WORD = "{TREE}"
# The words added after the pinned include directories: the build directory, searched last.
BUILD_DIR_INCLUDE = ("-idirafter", ".")
# A directory of this name in a C++ file's path makes it a kernel source, which the host compiler never builds.
KERNEL_DIR = "kernels"
# The install's record (toolchains/tt-metal.sh writes it once every check passed) and its CMake-fetched packages list.
INSTALL_RECORD = "lassi-install.txt"
TREE_CPM_SOURCES = "lassi-cpm-sources.txt"
# The tracked copy of the packages list the install recorded (results/p4-tt-install), beside the pin files.
TRACKED_CPM_SOURCES = "tt-metal-cpm-sources.txt"


def host_words(pin: Mapping[str, str], tree: PurePath) -> dict[str, list[str]]:
    """Return each HOST_* key's argv words, with {TREE} replaced by `tree` in POSIX form.

    Raises ValueError naming every HOST_* key the pin lacks or leaves empty.
    """
    missing = [key for key in HOST_KEYS if not pin.get(key)]
    if missing:
        raise ValueError(
            f"toolchains/{PIN_NAME}.pin has no {', '.join(missing)}, the pinned host build line of a TT host program"
        )
    posix = tree.as_posix()
    return {key: [word.replace(TREE_WORD, posix) for word in pin[key].split()] for key in HOST_KEYS}


def _entries(text: str, *, comments: bool) -> list[str]:
    """Return the lines of a packages list without trailing whitespace, blank lines left out.

    With `comments`, lines starting with "#" are left out too (the tracked
    list); the tree's own list has none, so there such a line is an entry.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    return [line for line in lines if line and not (comments and line.startswith("#"))]


def _read_text(path: Path) -> str:
    """Return a file's text, decoded as UTF-8 with errors replaced; OSError when it cannot be read."""
    return path.read_bytes().decode("utf-8", errors="replace")


def check_install_record(tree: Path, pin: Mapping[str, str]) -> None:
    """Raise ValueError naming the file unless `tree` holds lassi-install.txt whose first line names NAME and COMMIT.

    toolchains/tt-metal.sh writes the record only after every check of the
    install passed, and its first line is "<NAME> <COMMIT> <URL>"; a build
    that stopped leaves lassi-install.unfinished instead.
    """
    name, commit = pin.get("NAME", ""), pin.get("COMMIT", "")
    if not name or not commit:
        raise ValueError(f"toolchains/{PIN_NAME}.pin has no NAME or no COMMIT, so no install can be checked against it")
    record = tree / INSTALL_RECORD
    try:
        lines = _read_text(record).splitlines()
    except OSError as error:
        raise ValueError(
            f"{record} cannot be read ({error.strerror or error}), so {tree} is not an install of the pinned "
            f"{name} (a build that stopped leaves lassi-install.unfinished); install it on the build host with "
            f"toolchains/{PIN_NAME}.sh"
        ) from None
    words = lines[0].split()[:2] if lines else []
    if words != [name, commit]:
        found = " ".join(words) or "nothing"
        raise ValueError(
            f"the first line of {record} names {found!r}, not {name} {commit}; the tree is not the pinned install, "
            f"so install the pin on the build host with toolchains/{PIN_NAME}.sh"
        )


def check_cpm_sources(tree: Path) -> None:
    """Raise ValueError naming the file unless the tree's lassi-cpm-sources.txt equals the tracked list.

    The tracked list is toolchains/tt-metal-cpm-sources.txt, read from
    lassi.toolchains.pins.PINS_DIR; its "#" lines are comments. Both lists
    are compared line for line, trailing whitespace and blank lines left out,
    so a package fetched at another commit, with other changed files, added,
    or missing is refused.
    """
    tracked_path = pins.PINS_DIR / TRACKED_CPM_SOURCES
    try:
        tracked = _entries(_read_text(tracked_path), comments=True)
    except OSError as error:
        raise ValueError(
            f"cannot read {tracked_path}, the tracked list of the packages tt-metal's CMake fetched "
            f"({error.strerror or error})"
        ) from None
    if not tracked:
        raise ValueError(f"{tracked_path} lists no package, so the tree's {TREE_CPM_SOURCES} cannot be checked")
    listed_path = tree / TREE_CPM_SOURCES
    try:
        listed = _entries(_read_text(listed_path), comments=False)
    except OSError as error:
        raise ValueError(
            f"{listed_path} cannot be read ({error.strerror or error}); the install records the packages its CMake "
            f"fetched there, and without it they cannot be compared with {tracked_path}"
        ) from None
    if listed != tracked:
        raise ValueError(
            f"{listed_path} differs from the tracked list {tracked_path} ({_first_difference(listed, tracked)}); "
            "the tree was configured with other CMake-fetched packages than the pinned install, so it is refused"
        )


def _first_difference(listed: Sequence[str], tracked: Sequence[str]) -> str:
    """Return where two packages lists first differ: an entry number with both entries, or which one ends first."""
    for number, (ours, theirs) in enumerate(zip(listed, tracked, strict=False), start=1):
        if ours != theirs:
            return f"entry {number} is {ours!r} in the tree and {theirs!r} in the tracked list"
    return f"the tree lists {len(listed)} packages and the tracked list {len(tracked)}"


# A driver line with no place, for example (SYNTHETIC, in clang's and lld's formats)
#   clang++-20: error: linker command failed with exit code 1 (use -v to see invocation)
#   ld.lld: error: undefined symbol: helper(int)
# The tool may carry a version suffix (clang++-20, clang-20). A place line on a built file reads
# "<file>:<line>:<column>: ...", so a file named like a tool never matches this pattern.
_DRIVER = re.compile(
    r"(?:clang\+\+|clang|ld\.lld)(?:-[0-9]+(?:\.[0-9]+)*)?: "
    r"(?P<severity>fatal error|error|warning|note): (?P<message>.*)"
)
_DRIVER_SEVERITY = {"fatal error": "error", "error": "error", "warning": "warning", "note": "note"}


def _driver(match: re.Match[str]) -> Diagnostic:
    """Return the Diagnostic for a driver line, which names no file; a fatal error is an error."""
    return compile_diagnostic(_DRIVER_SEVERITY[match["severity"]], match["message"])


def _fold_place(
    diagnostic: Diagnostic, lines: list[str], index: int, files: Mapping[str, str], patterns: Sequence[LinePattern]
) -> tuple[Diagnostic, int]:
    """Drop a leading "./" from the file, then keep the column only on a built file (fold_built_column)."""
    if diagnostic.file is not None and diagnostic.file.startswith("./"):
        diagnostic = dataclasses.replace(diagnostic, file=diagnostic.file[2:])
    return fold_built_column(diagnostic, lines, index, files, patterns)


# Tried in this order on each line: the driver pattern before the place pattern (see _DRIVER).
_PATTERNS = (
    LinePattern(_DRIVER, _driver),
    dataclasses.replace(GCC, fold=_fold_place),
)


def parse_diagnostics(stderr: str, files: Mapping[str, str] = NO_FILES) -> list[Diagnostic]:
    """Return the compile-stage Diagnostics in clang's and ld.lld's `stderr`, in order (see the module docstring).

    `files` (relative path -> text) are the files built: a place line keeps
    its column only on one of them. Lines that match no pattern are skipped.
    """
    return parse_stderr(stderr, files, _PATTERNS)


def is_kernel_source(path: str) -> bool:
    """Return True for a path with a directory named kernels in it, which the host compiler never builds."""
    return KERNEL_DIR in PurePosixPath(path).parts[:-1]


@register("Toolchain", "ttmetal-host")
class TtMetalHost(CompilerToolchain):
    """Builds a TT host program with the pinned host clang++-20 against the pinned tt-metal tree."""

    name = "ttmetal-host"
    capabilities = frozenset({"diagnostics", "emits_warnings"})
    SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx")
    # The pinned host clang++-20 and tree: toolchains/tt-metal.pin, whose EXECUTABLE is an absolute path on the build
    # host and whose PREFIX_NAME is the tree. No PIN_BIN, since the compiler is not installed under the tree.
    PIN = PIN_NAME

    def __init__(
        self,
        *,
        tree: Path,
        executable: str = "clang++-20",
        runner: CommandRunner | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        """Keep the tree, the clang++ executable, the command runner (None means subprocess_runner), and the timeout.

        The host build line is read from toolchains/tt-metal.pin now, with
        {TREE} replaced by `tree`; ValueError when the pin lacks a HOST_* key.
        """
        super().__init__(executable=executable, runner=runner, timeout_s=timeout_s)
        self.tree = Path(tree)
        self._words = host_words(pins.read_pin(self.PIN), self.tree)

    @classmethod
    def check_tree(cls, tree: Path, pin: Mapping[str, str]) -> None:
        """Raise ValueError naming the file unless `tree` is the pinned install with the tracked packages.

        The stage runner calls this before any process starts (see the module
        docstring): the install record first, then the CMake-fetched packages
        list, then the pin's host build line.
        """
        check_install_record(tree, pin)
        check_cpm_sources(tree)
        host_words(pin, tree)

    def command(self, sources: Sequence[str]) -> list[str]:
        """Return the compile and link command for the host `sources`, as given (see the module docstring)."""
        words = self._words
        return [
            self.executable,
            *words["HOST_DEFINES"],
            *words["HOST_INCLUDES"],
            *BUILD_DIR_INCLUDE,
            *words["HOST_FLAGS"],
            *words["HOST_LINK_FLAGS"],
            "-o",
            OUTPUT,
            *sources,
            *words["HOST_LINK_LIBRARIES"],
        ]

    def parse(self, stderr: str, files: Mapping[str, str]) -> list[Diagnostic]:
        """Return the Diagnostics in clang's and ld.lld's `stderr` (see parse_diagnostics)."""
        return parse_diagnostics(stderr, files)

    def _sources(self, files: Mapping[str, str]) -> list[str]:
        """Return the host sources in sorted order: the C++ sources that are not kernel sources."""
        return [path for path in super()._sources(files) if not is_kernel_source(path)]

    def _no_sources(self) -> Diagnostic:
        """Return the error for a build with no host source; a kernel source never counts as one."""
        suffixes = ", ".join(self.SOURCE_SUFFIXES)
        message = (
            f"no host source to compile: {self.name} compiles files ending in {suffixes} outside any "
            f"{KERNEL_DIR}/ directory, and a kernel source is placed for the kernel JIT, never compiled here; return "
            "at least one host source, in a fenced block whose first line is // FILE: <path>"
        )
        return compile_diagnostic("error", message, code="no-sources")
