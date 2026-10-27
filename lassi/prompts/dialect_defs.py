"""The definitions of an MLIR dialect at its pin, by name, for the feedback of a verify error (task P17.41).

plans/spikes/mlir-issues.md, Half 1: most of the model's ttkernel attempts
ended at ttmlir-opt's parse or verify step, on types and ops the dialect
does not have and on ops' assembly formats, and one parser message often
came back unchanged over several corrections. A compile correction under the
recipe's dialect_lookup section (lassi.core.recipe DIALECT_LOOKUP;
lassi.core.stages) therefore carries, for the errors of that step, the exact
definition of each op or type an error names or points at, and for a name
the dialect does not have its nearest op and type names.

A definitions tree is a manifest tree under <assets root>/context/
(lassi.prompts.assets load_tree: MANIFEST.yaml and one `<key>.txt` per
entry, each checked against its sha256), which
tools/build_ttkernel_context.py builds from the pinned dialect sources and
of which only the manifest is tracked (OQ-018 keeps upstream text out of
the repository). Its entries' texts, joined, are a run of blocks: a head
line BLOCK_HEAD naming the definition, then its text, exactly as the
sources hold it, up to the next head line or the end. A name is
`<dialect>.<mnemonic>` for an op, `!<dialect>.<mnemonic>` for a type, and
`#<dialect>.<mnemonic>` for an attribute, the spellings a module uses; every
name of a tree is in one dialect. Text before the first head line is a
preamble and holds no definition.

Which names an error gives (error_names, line_names): the type a message
calls unknown (MLIR's `unknown type `<name>` in dialect `<dialect>``), and
every name of the dialect the message holds, quoted or not; and the names on
the line of the model's file the error points at. On a line, a name with no
sigil is an op, unless an `=` follows it (an attribute's key in an
attribute dictionary, such as `ttkernel.thread = ...`), and `!`/`#` mark a
type or an attribute; a type's or attribute's parameters (`<...>`) are not
part of its name. nearest_names ranks every name of the tree against one the
tree lacks by difflib's ratio over a folded form (the mnemonic in lower case
without `_`, `.`, or `:`), so `!ttkernel.tensor_accessor_args` finds
`!ttkernel.TensorAccessorArgs` and `ttkernel.InterleavedAddrGenFast` finds
`!ttkernel.interleaved_addr_gen_fast`.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from lassi.prompts import assets

# The head line of one definition block; {name} is the definition's name.
BLOCK_HEAD = "// ==== {name} ===="
_HEAD = re.compile(r"^// ==== (?P<name>[!#]?[A-Za-z_][\w.:<>* ]*?) ====$", re.MULTILINE)
# The least difflib ratio, over the folded forms, at which nearest_names lists a name [DESIGN]: as the C++ lookup's
# NEAREST_MIN_RATIO (lassi.core.stages), a one-letter slip in a ten-letter name scores 0.9, and unrelated names are
# expected, not measured, to score lower.
NEAREST_MIN_RATIO = 0.75
# MLIR's words for a type its dialect lacks (mlir/lib/AsmParser), as ttmlir-opt printed them in the spike's records.
_UNKNOWN_TYPE = r"unknown\s+type\s+`(?P<name>[^`\s]+)`\s+in\s+dialect\s+`{dialect}`"
# A name of the dialect in text: an optional sigil, the dialect, and the mnemonic (parts joined by `.` or `::`).
_TOKEN = (
    r"(?<![\w.!#$%])(?P<sigil>[!#]?){dialect}\.(?P<mnemonic>[A-Za-z_]\w*(?:(?:::|\.)[A-Za-z_]\w*)*)"
    r"(?P<after>\s*=?)"
)


@dataclass(frozen=True)
class DialectDefs:
    """A loaded definitions tree: its name, its dialect, and each definition's text by name, in tree order."""

    name: str
    dialect: str
    definitions: Mapping[str, str] = field(default_factory=dict)


def parse_defs(text: str) -> dict[str, str]:
    """Return {name: definition text} of a definitions text's blocks, in order (the module docstring).

    Each text runs from the line after its head to the next head line, with
    surrounding blank lines and line feeds stripped. Raises ValueError for a
    text with no block, a name given twice, or an empty block.
    """
    heads = list(_HEAD.finditer(text))
    if not heads:
        raise ValueError("the definitions text holds no block (no line `// ==== <name> ====`)")
    found: dict[str, str] = {}
    for number, head in enumerate(heads):
        end = heads[number + 1].start() if number + 1 < len(heads) else len(text)
        name, body = head.group("name"), text[head.end():end].strip("\n")
        if name in found:
            raise ValueError(f"the definitions text names {name!r} twice")
        if not body.strip():
            raise ValueError(f"the definition block of {name!r} is empty")
        found[name] = body
    return found


def dialect_of(definitions: Mapping[str, str]) -> str:
    """Return the one dialect the definitions' names are in; ValueError when they name none or several."""
    dialects = {name.lstrip("!#").split(".", 1)[0] for name in definitions}
    if len(dialects) != 1:
        raise ValueError(f"a definitions tree serves one dialect; its names are in {sorted(dialects) or 'none'}")
    return dialects.pop()


def load_defs(name: str, root: Path | None = None) -> DialectDefs:
    """Load the definitions tree `name` under <root>/context/ (root: lassi.prompts.assets.default_root()).

    Raises ValueError when the tree does not exist, when a file is missing or
    differs from its manifest (lassi.prompts.assets load_tree, naming the
    tree's generator), and when its text is no definitions text (parse_defs,
    dialect_of).
    """
    tree = (assets.default_root() if root is None else Path(root)) / "context" / name
    if not (tree / assets.MANIFEST).is_file():
        raise ValueError(f"no definitions tree {name!r}: {tree / assets.MANIFEST} does not exist")
    texts = assets.load_tree(tree)
    try:
        definitions = parse_defs(assets.PACK_JOINER.join(texts.values()))
        dialect = dialect_of(definitions)
    except ValueError as error:
        raise ValueError(f"the definitions tree {tree}: {error}") from error
    return DialectDefs(name=name, dialect=dialect, definitions=MappingProxyType(definitions))


def _names(dialect: str, text: str) -> list[str]:
    """Return the dialect's names in `text`, each once, in order; a sigil-less name before `=` is left out."""
    found: list[str] = []
    for match in re.finditer(_TOKEN.format(dialect=re.escape(dialect)), text):
        if not match.group("sigil") and "=" in match.group("after"):
            continue
        name = f"{match.group('sigil')}{dialect}.{match.group('mnemonic')}"
        if name not in found:
            found.append(name)
    return found


def error_names(dialect: str, message: str) -> list[str]:
    """Return the names one error message of the dialect's tools gives, each once, in order (module docstring)."""
    unknown = [f"!{dialect}.{match.group('name')}" for match in
               re.finditer(_UNKNOWN_TYPE.format(dialect=re.escape(dialect)), message)]
    return list(dict.fromkeys([*unknown, *_names(dialect, message)]))


def line_names(dialect: str, line: str) -> list[str]:
    """Return the names on one line of a module, each once, in order (module docstring)."""
    return _names(dialect, line)


def _folded(name: str) -> str:
    """Return the form nearest_names compares: the mnemonic in lower case without `_`, `.`, or `:`."""
    mnemonic = name.lstrip("!#").split(".", 1)[-1]
    return re.sub(r"[_.:]", "", mnemonic.lower())


def nearest_names(name: str, defs: DialectDefs, count: int) -> tuple[str, ...]:
    """Return up to `count` names of `defs` nearest `name`, nearest first, each at NEAREST_MIN_RATIO or above.

    The score is difflib's ratio of the folded forms (_folded); ties go to
    a name of the same kind (op, type, or attribute) as `name`, then to the
    earlier in tree order. `name` itself is never listed.
    """
    matcher = difflib.SequenceMatcher()
    matcher.set_seq2(_folded(name))
    kind = name[:1] if name[:1] in "!#" else ""
    scored: list[tuple[float, int, int, str]] = []
    for order, candidate in enumerate(defs.definitions):
        if candidate == name:
            continue
        matcher.set_seq1(_folded(candidate))
        if matcher.real_quick_ratio() < NEAREST_MIN_RATIO or matcher.quick_ratio() < NEAREST_MIN_RATIO:
            continue
        ratio = matcher.ratio()
        if ratio >= NEAREST_MIN_RATIO:
            same = 0 if (candidate[:1] if candidate[:1] in "!#" else "") == kind else 1
            scored.append((-ratio, same, order, candidate))
    return tuple(candidate for *_, candidate in sorted(scored)[:count])
