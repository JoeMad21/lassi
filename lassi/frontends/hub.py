"""The hub level's tag rule, which every frontend applies to the module it returns (bible Frontend Rules; task P12.4).

A frontend tags a hub module from the dialects the module holds, never from the language it read: `low` when the
text holds an op of the llvm or cf dialect, `structured` otherwise (Frontend Rules: structured is scf, affine, and
linalg; low is llvm and cf). A low module is not translation material until it is raised to structured.

hub_tag(text) reads op names textually, with no pinned tool: on each line that starts an op, after an optional
result list such as `%0 = ` or `%0:2 = `, the first word, or the quoted name of an op in generic form. Known cases,
not complete: an op that follows another op on its line is not read (the MLIR printer puts one op on a line), and
neither is an op inside a string or a comment; an llvm attribute or type, such as `{llvm.emit_c_interface}` or
`!llvm.ptr`, names no op and leaves the tag structured.
"""

from __future__ import annotations

import re

# The IR level of the hub (bible IR Strategy, the cpu-mlir row): MLIR at Polygeist's LLVM.
HUB_LEVEL = "cpu-mlir"
STRUCTURED = "structured"
LOW = "low"
# The dialects whose ops make a module low.
LOW_DIALECTS = ("llvm", "cf")

# An op's name where a line starts one: after an optional result list, the first word, or a generic op's quoted name.
_OP = re.compile(r'^\s*(?:%[\w#:$.-]+(?:\s*,\s*%[\w#:$.-]+)*\s*=\s*)?"?([A-Za-z_][\w$.]*)')
# Lines that start no op: a region's close, a comment, a block label, an attribute or type alias.
_NO_OP = ("}", "//", "^", "#", "!")


def hub_tag(text: str) -> str:
    """Return `low` when `text` holds an op of the llvm or cf dialect, else `structured`; the language is not read."""
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith(_NO_OP):
            continue
        match = _OP.match(line)
        if match is None:
            continue
        dialect, dot, _ = match.group(1).partition(".")
        if dot and dialect in LOW_DIALECTS:
            return LOW
    return STRUCTURED
