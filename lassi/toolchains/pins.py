"""Toolchain pin files: toolchains/<name>.pin in the repository (bible Toolchain Pins; Agent Rule 10).

A pin file is a shell-sourced list of KEY=value lines, read by the install
script toolchains/<name>.sh. Blank lines and lines starting with "#" are
skipped. A value is either a double-quoted string, whose quotes are removed,
or a bare word with no whitespace, quote, or "#"; a trailing comment after
whitespace is not part of the value. The keys the runner reads:

- NAME and VERSION: the pin's name and version; VERSION is what a Trial's
  toolchain_pins records.
- PREFIX_NAME: the install directory under the toolchains root
  ($LASSI_TOOLCHAINS), written <NAME>@<VERSION>, for a toolchain the
  project installs.
- EXECUTABLE: for a host compiler the project does not install (a
  toolchain class with PIN and no PIN_BIN, such as toolchains/gcc.pin),
  its absolute path on the build host, which the runner uses as given; such
  a pin has no PREFIX_NAME and no install script, unless its class defines
  check_tree and builds against the pin's installed tree, which
  PREFIX_NAME then names (toolchains/tt-metal.pin, with its install script).
- EXPECT_VERSION: text the pinned compiler's `--version` must print, checked
  in the compile sandbox before the first build.
- COMPILER_SUBDIR (optional): the compiler directory inside the prefix,
  which a toolchain class may name in its PIN_BIN as {COMPILER_SUBDIR}.
- CUDA_HOME_FROM (optional): the PREFIX_NAME of another pin that the
  compiler needs at run time; the runner passes <toolchains root>/<it> to the
  compile as NVHPC_CUDA_HOME (PREFIX_VARIABLES), and the Trial records that
  pin's VERSION too.

Other keys are read by the pin's install script or its toolchain class, such
as COMMIT and the HOST_* build line of toolchains/tt-metal.pin, which
lassi.toolchains.ttmetal_build reads. PINS_DIR also holds the tracked lists a
toolchain class compares an install with (toolchains/tt-metal-cpm-sources.txt).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

# Where the committed pin files live: toolchains/ at the repository root.
PINS_DIR = Path(__file__).resolve().parents[2] / "toolchains"

# Pin keys that name another pin's install prefix, and the compile environment variable that gets
# <toolchains root>/<that prefix>. On a host without a GPU driver, nvc++ finds the pinned CUDA through
# NVHPC_CUDA_HOME, or it looks for another CUDA version and fails (toolchains/nvhpc.pin).
PREFIX_VARIABLES = {"CUDA_HOME_FROM": "NVHPC_CUDA_HOME"}

# One KEY=value line, stripped: the key, a quoted or bare value, and an optional trailing comment.
_LINE = re.compile(r"([A-Z][A-Z0-9_]*)=(\"[^\"]*\"|[^\s\"#]*)(\s+#.*)?")
# A pin name: one file name stem, so a name never reaches outside PINS_DIR.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def pin_path(name: str) -> Path:
    """Return the path of toolchains/<name>.pin; raise ValueError for a name that is not a plain file stem."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"a pin name must match {_NAME.pattern}, got {name!r}")
    return PINS_DIR / f"{name}.pin"


def read_pin(name: str) -> dict[str, str]:
    """Return the KEY=value pairs of toolchains/<name>.pin, with surrounding quotes removed.

    Raises ValueError naming the file and the line for a line that is not a
    KEY=value line, and FileNotFoundError when the pin file does not exist.
    A key given twice keeps its last value, as the shell does.
    """
    path = pin_path(name)
    pairs: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _LINE.fullmatch(stripped)
        if match is None:
            raise ValueError(f"{path}:{number}: not a KEY=value line: {line!r}")
        pairs[match.group(1)] = match.group(2).strip('"')
    return pairs


def linked_prefixes(pin: Mapping[str, str]) -> dict[str, str]:
    """Return the compile environment variables a pin sets, each mapped to the other pin's PREFIX_NAME it names.

    For toolchains/nvhpc.pin this is {"NVHPC_CUDA_HOME": "cuda@<version>"}; a
    pin without such keys gives {}.
    """
    return {variable: pin[key] for key, variable in PREFIX_VARIABLES.items() if key in pin}


def prefix_pin_name(prefix: str) -> str:
    """Return the pin name of an install prefix written <NAME>@<VERSION>, such as "cuda" for "cuda@12.6.3"."""
    name, at, version = prefix.partition("@")
    if not at or not name or not version:
        raise ValueError(f"an install prefix must be written <name>@<version>, got {prefix!r}")
    return name
