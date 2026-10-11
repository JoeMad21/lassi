"""The Agent Rule 11 refusal: IR in generic form and truncated IR (task P5.11).

IR enters the corpus in custom assembly format only, never in generic form,
and whole, never truncated (bible Agent Rule 11; Readability Standards, MLIR;
Review Of v0, the 2 MB whale truncation row). check_ir(text, level=<level>)
reads one MLIR module text and raises an IRFormError, a ValueError, when
it finds a cut or an op in generic form (the rules below). The check is
textual: it starts no process, reads no pinned tool, and knows no dialect,
so lassi.corpus runs it on every record and task P5.6's IRLevels reuse it
beside the pinned verifiers.

The scan reads the text once as tokens: quoted strings, `//` comments, the
brackets `(`, `)`, `[`, `]`, `{`, and `}`, the sign `=`, and words (any
other run of characters without blanks). Strings and comments are skipped
when balancing brackets and when looking for ops, so a bracket, a quote, or
generic-looking text inside them changes nothing. Angle brackets are not
balanced, since `->` and `>=` reuse their characters; every op of a module
stands inside the module's braces, so a cut inside `<...>` still leaves a
brace open.

TruncatedIRError: a text the scan finds cut. Its `line` is the
1-based line of the truncation point, and its message says `line <n>`:

- a string that does not close on the line it opens on (a backslash and the
  character after it never close one, so `\\"` and `\\\\` are read as
  escapes; `line` is the string's line);
- a bracket still open where the text ends (`line` is the text's last line);
- a closing bracket with nothing open, as when the text starts after a cut,
  or one that closes another kind of bracket (`line` is its line);
- a text with no token: empty, or only blanks and comments (line 1).

GenericFormError: an op printed in MLIR's quoted generic form, a string
followed by `(` (blanks, line ends, and comments between them aside) where
an op starts. An op starts at the first token of a line, right after `{`,
or right after `=` when the token before the `=` is a result such as `%0`,
`%0:2`, or the last of `%a, %b`. A quoted string not followed by `(`, such
as a dictionary key (`{"key" = 1}`), or one that stands elsewhere, such as
a symbol name (`@"name"`) or the callee of a custom-form op
(`emitc.call_opaque "f"(%x)`), is no op. An op on NO_CUSTOM_FORM[level]
passes, and the scan goes on inside its regions. A truncated text is
refused as truncated even when it also holds a generic op; otherwise the
error names the first refused op in text order: its `op` is the op name,
its `line` the line where the quoted name stands, and its `level` the level
checked, all three named in its message.

NO_CUSTOM_FORM maps each MLIR level the corpus holds to the ops that have no
custom assembly format at that level's pin (some ttkernel ops print
generically in either mode). It covers cpu-mlir and ttkernel and starts with
no op; task P5.6 fills it from the pinned dialects and checks it against the
pinned printer. check_ir reads it when called and refuses a level it does
not hold with a ValueError naming the level, so the corpus holds no llvm-ir
text until a later task adds that level with a truncation rule of its own
(LLVM IR is not MLIR, and the generic-form rule does not apply to it).

Known cases, not complete: a generic op without results that follows
another op on its line, not right after `{`, is not seen (a module printed
one op per line has none); a cut between two
top-level ops leaves a whole text, which no textual check can tell from a
whole module; a text with no module op, or with two top-level modules,
passes, since MLIR reads top-level ops as an implicit module; and the scan
parses no type, attribute, or escape beyond the backslash rule above, so it
is no verifier (task P5.6 runs the pinned ones).

module_names(text) reads the quoted names of a text's module ops with the
same scan, for lassi.corpus's host-path rule.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Mapping, NamedTuple, NoReturn

# The ops with no custom assembly format at each level's pin; task P5.6 fills it from the pinned dialects.
NO_CUSTOM_FORM: Mapping[str, frozenset[str]] = MappingProxyType({"cpu-mlir": frozenset(), "ttkernel": frozenset()})

# One token per match, after any blanks on its line. Blanks at the very end match nothing and are passed over.
_TOKEN = re.compile(
    r"[^\S\n]*(?:"
    r"(?P<newline>\n)"
    r"|(?P<comment>//[^\n]*)"
    r'|(?P<string>"(?:[^"\\\n]|\\[^\n])*")'
    r'|(?P<unclosed>")'
    r"|(?P<bracket>[()\[\]{}])"
    r"|(?P<sign>=)"
    r'|(?P<word>[^\s"()\[\]{}=/]+|/))'
)
_OPENER = {")": "(", "]": "[", "}": "{"}
_MODULE_OPS = frozenset({"module", "builtin.module"})
_RULE_11 = "Agent Rule 11 refuses truncated IR (split an oversized module by function, or drop it and count it)"


class IRFormError(ValueError):
    """IR that Agent Rule 11 refuses: an op printed in generic form, or a text the scan finds cut."""


class GenericFormError(IRFormError):
    """An op printed in the quoted generic form: `op` is its name, `line` its line, and `level` the level checked."""

    def __init__(self, message: str, *, op: str, line: int, level: str) -> None:
        """Keep the op, its line, and the level beside the message, which names all three."""
        super().__init__(message)
        self.op = op
        self.line = line
        self.level = level


class TruncatedIRError(IRFormError):
    """A text the scan finds cut: `line` is the 1-based line of the truncation point."""

    def __init__(self, message: str, *, line: int) -> None:
        """Keep the line beside the message, which says `line <n>`."""
        super().__init__(message)
        self.line = line


class ModuleName(NamedTuple):
    """A module op's quoted name as written (escapes undecoded), its line, and its offsets inside the quotes."""

    name: str
    line: int
    start: int
    end: int


def levels() -> tuple[str, ...]:
    """Return the levels check_ir knows, read from NO_CUSTOM_FORM when called."""
    return tuple(NO_CUSTOM_FORM)


def check_ir(text: str, *, level: str, where: str = "") -> None:
    """Return when the scan of `text` finds no cut and no generic op outside NO_CUSTOM_FORM[level]; else raise.

    Raises ValueError naming `level` when NO_CUSTOM_FORM does not hold it,
    TypeError when `text` is not a str, TruncatedIRError when the scan finds
    a cut, and GenericFormError for the first op in the quoted
    generic form that is not on NO_CUSTOM_FORM[level] (see the module
    docstring for both rules). `where`, when given, names the text at the
    start of every message, as in `CorpusRecord.source_ir: ...`.
    """
    table = NO_CUSTOM_FORM
    prefix = f"{where}: " if where else ""
    if level not in table:
        raise ValueError(f"{prefix}unknown IR level {level!r}; the Agent Rule 11 refusal knows {', '.join(table)}")
    if not isinstance(text, str):
        raise TypeError(f"{prefix}IR text must be a str, got {type(text).__name__}")
    generic, _ = _Scanner(text, prefix).run()
    allowed = table[level]
    for op, line in generic:
        if op not in allowed:
            raise GenericFormError(
                f"{prefix}op {op} at line {line} is printed in the quoted generic form; Agent Rule 11 takes {level} "
                f"IR in custom assembly format only, and {op} is not on NO_CUSTOM_FORM[{level!r}], the level's ops "
                "with no custom format at its pin",
                op=op,
                line=line,
                level=level,
            )


def module_names(text: str) -> tuple[ModuleName, ...]:
    """Return the quoted name of each module op in `text`, in text order.

    A module op is `module` or `builtin.module` followed by `@"<name>"`; a
    bare symbol name (`@name`), which cannot hold a path, and an unnamed
    module give none. The text is read with check_ir's scan, so a name
    inside a string or a comment is not read, and a text the scan finds cut
    raises TruncatedIRError.
    """
    return tuple(_Scanner(text, "").run()[1])


class _Scanner:
    """One pass over a module text: its brackets, the quoted names where an op starts, and its module names."""

    def __init__(self, text: str, prefix: str) -> None:
        """Start before the first token of `text`; `prefix` starts every error message."""
        self.text = text
        self.prefix = prefix
        self.line = 1
        self.first = True  # no token yet on the current line
        self.count = 0  # tokens read, comments aside
        self.open: list[tuple[str, int]] = []  # each open bracket with its line
        self.recent = ("", "")  # the two tokens before the current one, the nearer last; a string reads '"'
        self.waiting: tuple[str, int] | None = None  # a quoted name where an op starts, until the next token
        self.generic: list[tuple[str, int]] = []
        self.names: list[ModuleName] = []

    def run(self) -> tuple[list[tuple[str, int]], list[ModuleName]]:
        """Scan the whole text; return each (op, line) in quoted generic form and each module name.

        Raises TruncatedIRError at the first truncation point.
        """
        for match in _TOKEN.finditer(self.text):
            kind = match.lastgroup
            if kind == "newline":
                self.line += 1
                self.first = True
                continue
            if kind == "comment":
                continue
            value = match.group(kind)
            if self.waiting is not None and value == "(":
                self.generic.append(self.waiting)
            self.waiting = None
            if kind == "unclosed":
                self._cut(self.line, "a string opens here and does not close on its line")
            if kind == "string":
                self._string(value, match.start(kind))
                value = '"'
            elif kind == "bracket":
                self._bracket(value)
            self.recent = (self.recent[1], value)
            self.first = False
            self.count += 1
        self._end()
        return self.generic, self.names

    def _string(self, value: str, start: int) -> None:
        """Read a quoted string at offset `start`: a module's name, a name where an op starts, or neither."""
        before, last = self.recent
        if last == "@" and before in _MODULE_OPS:
            self.names.append(ModuleName(value[1:-1], self.line, start + 1, start + len(value) - 1))
        elif self.first or last == "{" or (last == "=" and before.startswith("%")):
            self.waiting = (value[1:-1], self.line)

    def _bracket(self, value: str) -> None:
        """Open a bracket, or close the innermost open one; a closer that fits none is a truncation point."""
        if value not in _OPENER:
            self.open.append((value, self.line))
            return
        if not self.open:
            self._cut(self.line, f"a '{value}' closes nothing, so the text starts after a cut")
        opener, line = self.open.pop()
        if opener != _OPENER[value]:
            self._cut(self.line, f"a '{value}' closes the '{opener}' opened at line {line}")

    def _end(self) -> None:
        """Refuse a text with no token, or one that ends with a bracket open."""
        if not self.count:
            self._cut(1, "the text holds no IR: it is empty, or only blanks and comments")
        if self.open:
            last = self.line - 1 if self.text.endswith("\n") else self.line
            opener, line = self.open[-1]
            self._cut(last, f"the text ends with the '{opener}' opened at line {line} still open")

    def _cut(self, line: int, reason: str) -> NoReturn:
        """Raise the TruncatedIRError for a truncation point at `line`."""
        raise TruncatedIRError(f"{self.prefix}IR cut at line {line}: {reason}; {_RULE_11}", line=line)
