"""Build the ttkernel API reference from the pinned dialect sources (task P17.44; plans/spikes/mlir-issues.md).

Usage:

    uv run tools/build_ttkernel_reference.py build --tree <dir> [--out <assets dir>]
    uv run tools/build_ttkernel_reference.py check --tree <dir> [--out <assets dir>]

The tree is tt-mlir at toolchains/tt-mlir.pin's COMMIT, exactly as
tools/build_ttkernel_context.py takes it (its check_tree: an install's
src/tt-mlir, or a directory its `fetch` filled), and the sources are that
tool's: TTKernelOps.td, TTKernelOpsTypes.td, TTKernelOpsEnums.td, and
TTKernelOps.cpp. In the window of 2026-10-27 (rx 20261027-005611-window-7998)
no cpp -> ttkernel attempt built, and the records of 2026-10-26 show
the main source of wrong syntax: the reference card the model wrote of the
dialect in summarize_context invented forms. This reference replaces that
card with the dialect's own facts, rendered by rule, never written by hand.

`build` writes context/REF/ under `--out` (default: assets/), a context pack
(lassi.prompts.assets load_tree) of one entry, KEY, whose keys end in the
language ttkernel:

- Types: every type of TTKernelOpsTypes.td as a module spells it,
  `!ttkernel.<mnemonic>`, with its parameters in the order its
  assemblyFormat prints them, and its summary.
- Attributes: every attribute, `#ttkernel.<mnemonic>`; an enum attribute
  with its cases' strings from TTKernelOpsEnums.td, a struct attribute with
  its format and parameters.
- Ops: every op of TTKernelOps.td, under its section's title, in file
  order, one line each (op_line): the op as its assemblyFormat prints it,
  with each operand `%<name>` and each attribute `<name>`, an optional part
  in [ ], then ` : (<operand types>) -> <result types>`, each operand type
  spelled from its constraint (spell: i32 for I32, index | i32 for
  IndexLike, !ttkernel.cb<...> for TTKernel_CB); an op whose format holds
  an attribute other than an integer, bool, or string literal, and an op
  with no assembly format, are given in MLIR's generic form instead,
  `"ttkernel.<op>"(<operands>) <{<attribute> = <value>}> : (...) -> ...`,
  which ttmlir-opt parses for every op (the op classes of TTKernelBase.td
  set no assembly format at the pin, read 2026-10-29); an op whose parser is
  hand-written gives the format comment TTKernelOps.cpp holds for it. The
  op's summary follows ` // ` unless it only repeats the op's name.

Only the manifest is tracked; the reference is tt-mlir's dialect rendered
and is gitignored, as OQ-018 keeps upstream text out of the repository. No
source is a tt-pairs-v0 item (Agent Rule 5). `check` builds in memory and
names every difference with the files under --out (status 1); a refusal is
status 2. The same tree always gives the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lassi.toolchains import pins  # noqa: E402

GENERATOR = "tools/build_ttkernel_reference.py"
PIN = "tt-mlir"
REF = "ttkernel-ref-5f396fd6"
LANGUAGE = "ttkernel"
KEY = f"ttkernel_ref.{LANGUAGE}"
DIALECT = "ttkernel"
# Attribute constraints a custom format prints as a plain literal, with the words the reference gives each.
LITERAL_ATTRS = (
    (re.compile(r"[SU]?I\d+Attr"), "an integer literal"),
    (re.compile(r"BoolAttr"), "true or false"),
    (re.compile(r"StrAttr"), 'a string literal "..."'),
    (re.compile(r"F\d+Attr"), "a float literal"),
)
# An integer attribute constraint (its value in the generic form is `<integer> : i<width>`), and the builtin integer
# and float type constraints (I32 is spelled i32, F32 f32).
_GENERIC_INT = re.compile(r"(?P<sign>[SU]?)I(?P<width>\d+)Attr")
_BUILTIN_INT = re.compile(r"(?P<sign>[SU]?)I(?P<width>\d+)")
_BUILTIN_FLOAT = re.compile(r"F(?P<width>\d+)")
_WRAPPED = re.compile(r"(?P<kind>Optional|Variadic|OptionalAttr|DefaultValuedAttr)<(?P<inner>.*)>")
_ANY_OF = re.compile(r"AnyTypeOf<\[(?P<inner>.*)\]>")
_FORMAT_TOKEN = re.compile(
    r"`(?P<lit>[^`]*)`|\$(?P<var>\w+)|(?P<struct>struct\(params\))|(?P<ftype>functional-type)\((?P<fargs>[^)]*)\)"
    r"|(?P<dict>attr-dict-with-keyword|attr-dict)|(?P<close>\)\?)|(?P<open>\()|(?P<anchor>\^)"
)
# The markers of an optional group's bounds among a format's pieces (_format_pieces), shown as [ ] (_joined).
_OPEN, _CLOSE = "<optional-group>", "</optional-group>"
_LET = r"^\s*let {field}\s*=\s*"
_TYPE_DEF = re.compile(r'^def (?P<def>\w+)\s*:\s*TTKernel_Type<\s*"\w+"\s*,\s*"(?P<mnemonic>\w+)"')
_ENUM_ATTR = re.compile(
    r'^def (?P<def>\w+)\s*:\s*EnumAttr<\s*TTKernel_Dialect\s*,\s*(?P<enum>\w+)\s*,\s*"(?P<mnemonic>\w+)"')
_STRUCT_ATTR = re.compile(r'^def (?P<def>\w+)\s*:\s*TTKernel_Attr<\s*"\w+"\s*,\s*"(?P<mnemonic>\w+)"')
_ENUM_DEF = re.compile(r'^def (?P<def>\w+)\s*:?\s*I32EnumAttr<\s*"\w+"\s*,\s*"[^"]*"\s*,\s*\[(?P<cases>[^\]]*)\]',
                       re.DOTALL)
_CASE = re.compile(r'I32EnumAttrCase<\s*"(?P<symbol>\w+)"\s*,\s*-?\d+\s*(?:,\s*"(?P<text>[^"]*)")?\s*>')
HEAD = (
    "// The ttkernel dialect of tt-mlir {version} (commit {commit}): every type, attribute, and op, rendered by\n"
    "// {generator} from {sources} at that commit.\n"
    "// An op line is the op's textual form: %name is an operand, <name> an attribute (\"where\" says how it is\n"
    "// written), [ ] an optional part, and after \" : \" come the operand types in order, \" -> \", and the result\n"
    "// type, () when there is none; a | b is either type. A type with parameters is written with them, such as\n"
    "// !ttkernel.cb<...> with its parameters below. A line in quotes, \"ttkernel.<op>\"(...), is MLIR's generic\n"
    "// form, which every op accepts: properties go in <{{ }}>, a [ ] one only when set.\n"
)


@dataclass(frozen=True)
class Arg:
    """One argument or result of an op: its name, its constraint as written, and how it is declared."""

    name: str
    constraint: str
    attribute: bool
    optional: bool
    variadic: bool


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; 0 on success, 1 when check finds a difference, 2 on a refusal."""
    args = _parser().parse_args(argv)
    try:
        built = build(Path(args.tree), pins.read_pin(PIN))
        out = Path(args.out)
        if args.command == "build":
            write(built, out)
            return 0
        problems = compare(built, out)
    except (OSError, ValueError) as error:
        print(f"build_ttkernel_reference: {error}", file=sys.stderr)
        return 2
    for problem in problems:
        print(problem)
    return 1 if problems else 0


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "check"):
        command = sub.add_parser(name)
        command.add_argument("--tree", required=True)
        command.add_argument("--out", default=str(REPO / "assets"))
    return parser


def context_tool() -> ModuleType:
    """Return tools/build_ttkernel_context.py, whose sources, tree check, and TableGen readers this tool shares."""
    path = REPO / "tools" / "build_ttkernel_context.py"
    spec = importlib.util.spec_from_file_location("build_ttkernel_context", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {path}")
    module = sys.modules.get(spec.name)
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def split_top(text: str) -> list[str]:
    """Return the parts of `text` split at commas outside <>, [], and (), each stripped; empty parts dropped."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "<[(":
            depth += 1
        elif char in ">])":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def let_value(body: str, field: str) -> str:
    """Return one `let <field> = ...;` value of a def body on one line, whitespace collapsed; "" when absent."""
    match = re.search(_LET.format(field=field), body, re.MULTILINE)
    if match is None:
        return ""
    rest = body[match.end():]
    if rest.startswith("[{"):
        value = rest[2:rest.find("}]")]
    else:
        depth, quoted, end = 0, False, len(rest)
        for index, char in enumerate(rest):
            if char == '"':
                quoted = not quoted
            elif not quoted and char in "(<[":
                depth += 1
            elif not quoted and char in ")>]":
                depth -= 1
            elif not quoted and char == ";" and depth <= 0:
                end = index
                break
        value = rest[:end]
    value = " ".join(value.split())
    return value[1:-1] if len(value) > 1 and value[0] == value[-1] == '"' else value


def parse_args(spec: str, attributes: bool = True) -> list[Arg]:
    """Return the Args of an `(ins ...)` or `(outs ...)` list, in order; `attributes` False reads none as one."""
    inner = spec.strip()
    inner = re.sub(r"^\(\s*(ins|outs)\b", "", inner).rstrip(")").strip() if inner else ""
    found: list[Arg] = []
    for part in split_top(inner):
        constraint, _, name = part.rpartition(":$")
        constraint = constraint.strip()
        kind, core = _unwrapped(constraint)
        found.append(Arg(name.strip(), core, attributes and core.endswith("Attr"),
                         kind in ("Optional", "OptionalAttr", "DefaultValuedAttr", "UnitAttr")
                         or core == "UnitAttr", kind == "Variadic"))
    return found


def _unwrapped(constraint: str) -> tuple[str, str]:
    """Return (wrapper kind or "", the constraint inside it); DefaultValuedAttr<X, "v"> gives X."""
    match = _WRAPPED.fullmatch(constraint)
    if match is None:
        return "", constraint
    inner = match.group("inner")
    if match.group("kind") == "DefaultValuedAttr":
        inner = split_top(inner)[0]
    return match.group("kind"), inner.strip()


@dataclass(frozen=True)
class Dialect:
    """What spell and the op lines read: type spellings by def name, constraint bodies, and attribute spellings."""

    types: Mapping[str, str]
    constraints: Mapping[str, str]
    attributes: Mapping[str, str]


def spell(constraint: str, dialect: Dialect) -> str:
    """Return how a module writes a value of type constraint `constraint`, alternatives joined by " | "."""
    kind, core = _unwrapped(constraint.strip())
    if kind:
        return spell(core, dialect)
    alternatives = _ANY_OF.fullmatch(core)
    if alternatives is not None:
        return " | ".join(spell(part, dialect) for part in split_top(alternatives.group("inner")))
    if core in dialect.constraints:
        return spell(dialect.constraints[core], dialect)
    if core in dialect.types:
        return dialect.types[core]
    integer, real = _BUILTIN_INT.fullmatch(core), _BUILTIN_FLOAT.fullmatch(core)
    if integer is not None:
        return f"{integer.group('sign').lower()}i{integer.group('width')}"
    if real is not None:
        return f"f{real.group('width')}"
    return {"Index": "index", "AnyType": "<any type>"}.get(core, core)


def attribute_words(constraint: str, dialect: Dialect) -> str | None:
    """Return how a custom format writes an attribute of `constraint` as a literal, or None when it is no literal."""
    for pattern, words in LITERAL_ATTRS:
        if pattern.fullmatch(constraint):
            return words
    return None


def generic_value(arg: Arg, dialect: Dialect) -> str:
    """Return the value of attribute `arg` in the generic form's property dictionary."""
    integer = _GENERIC_INT.fullmatch(arg.constraint)
    if integer is not None:
        return f"<integer> : {integer.group('sign').lower()}i{integer.group('width')}"
    if arg.constraint == "BoolAttr":
        return "true | false"
    if arg.constraint == "StrAttr":
        return '"..."'
    return dialect.attributes.get(arg.constraint, f"<{arg.constraint}>")


def type_list(args: Sequence[Arg], dialect: Dialect) -> str:
    """Return `(t1, t2[, t3])`: each operand's spelled type, an optional one in [ ], a variadic one with `...`."""
    text = ""
    for index, arg in enumerate(args):
        separator = ", " if index else ""
        spelled = spell(arg.constraint, dialect) + ("..." if arg.variadic else "")
        text += f"[{separator}{spelled}]" if arg.optional else f"{separator}{spelled}"
    return f"({text})"


def result_text(results: Sequence[Arg], dialect: Dialect) -> str:
    """Return the result part after `->`: `()` for none, the one type, or `(t1, t2)`."""
    if not results:
        return "()"
    if len(results) == 1 and not results[0].optional:
        return spell(results[0].constraint, dialect)
    return type_list(results, dialect)


def custom_text(fmt: str, mnemonic: str, args: Sequence[Arg], results: Sequence[Arg], dialect: Dialect) -> str:
    """Return the op as its declarative assemblyFormat `fmt` prints it (the module docstring)."""
    by_name = {arg.name: arg for arg in args}
    operands = [arg for arg in args if not arg.attribute]

    def var(name: str) -> str:
        """Return how the op line shows `$name`: %operand, <attribute>, or {...} for a region."""
        arg = by_name.get(name)
        if arg is None:
            return "{...}"
        return f"<{arg.name}>" if arg.attribute else f"%{arg.name}" + ("..." if arg.variadic else "")

    def functional(fargs: str) -> str:
        """Return functional-type(<fargs>, results) as `(operand types) -> result types`."""
        named = fargs.split(",")[0].strip()
        listed = operands if named == "operands" else [by_name[named.lstrip("$")]]
        return f"{type_list(listed, dialect)} -> {result_text(results, dialect)}"

    return _joined([f"{DIALECT}.{mnemonic}", *_format_pieces(fmt, var, functional)])


def _format_pieces(fmt: str, var: Callable[[str], str], functional: Callable[[str], str] | None = None,
                   struct: str = "") -> list[str]:
    """Return the pieces an assemblyFormat prints, in order: literals, `var` of each $name, and group markers.

    An optional group's parentheses become _OPEN and _CLOSE (shown as
    [ ]), attr-dict prints nothing, `functional` renders functional-type,
    and struct(params) prints `struct`.
    """
    pieces: list[str] = []
    for match in _FORMAT_TOKEN.finditer(fmt):
        if match.group("struct") is not None:
            pieces.append(struct)
        elif match.group("lit") is not None:
            pieces.append(match.group("lit").strip() or " ")
        elif match.group("var") is not None:
            pieces.append(var(match.group("var")))
        elif match.group("ftype") is not None and functional is not None:
            pieces.append(functional(match.group("fargs")))
        elif match.group("open") is not None:
            pieces.append(_OPEN)
        elif match.group("close") is not None:
            pieces.append(_CLOSE)
    return pieces


def _joined(pieces: Sequence[str]) -> str:
    """Return the pieces joined as MLIR's printer spaces them, an optional group's markers shown as [ ].

    No space goes after ( [ < or an opening marker, none before , ) ] > or
    either marker, and none between the op's name and its first (.
    """
    text = pieces[0]
    for previous, piece in zip(pieces, pieces[1:], strict=False):
        if piece == " ":
            continue
        tight = previous in ("(", "[", "<", _OPEN) or piece in (",", ")", "]", ">", _OPEN, _CLOSE) or (
            piece == "(" and previous == pieces[0])
        text += piece if tight else f" {piece}"
    return text.replace(_OPEN, "[").replace(_CLOSE, "]")


def generic_text(mnemonic: str, args: Sequence[Arg], results: Sequence[Arg], dialect: Dialect) -> str:
    """Return the op in MLIR's generic form, its attributes as properties (the module docstring)."""
    operands = [arg for arg in args if not arg.attribute]
    names = ""
    for index, arg in enumerate(operands):
        separator = ", " if index else ""
        name = f"%{arg.name}" + ("..." if arg.variadic else "")
        names += f"[{separator}{name}]" if arg.optional else f"{separator}{name}"
    props = []
    for arg in (arg for arg in args if arg.attribute):
        entry = arg.name if arg.constraint == "UnitAttr" else f"{arg.name} = {generic_value(arg, dialect)}"
        props.append(f"[{entry}]" if arg.optional else entry)
    properties = f" <{{{', '.join(props)}}}>" if props else ""
    return (f'"{DIALECT}.{mnemonic}"({names}){properties} : {type_list(operands, dialect)} -> '
            f"{result_text(results, dialect)}")


def op_line(mnemonic: str, body: str, cpp: str, def_name: str, dialect: Dialect) -> str:
    """Return the reference line of one op def (the module docstring): its form, then a `where` and its summary."""
    tool = context_tool()
    args = parse_args(let_value(body, "arguments"))
    results = parse_args(let_value(body, "results"), attributes=False)
    fmt = let_value(body, "assemblyFormat")
    named = {match.group(1) for match in re.finditer(r"\$(\w+)", fmt)}
    shown = [arg for arg in args if arg.attribute and arg.name in named]
    unliteral = [arg for arg in shown if attribute_words(arg.constraint, dialect) is None]
    if re.search(r"^\s*let hasCustomAssemblyFormat\s*=\s*1\s*;", body, re.MULTILINE):
        found = tool.custom_format(cpp, def_name.removeprefix("TTKernel_"))
        if found is None:
            raise ValueError(f"{DIALECT}.{mnemonic} ({def_name}) has a hand-written parser, but "
                             f"{tool.CPP_SOURCE} holds no format comment for it")
        first, last, lines = found
        words = " ".join(" ".join(line.removeprefix("//").split()) for line in lines)
        operands = ", ".join(f"%{arg.name}: {'[' if arg.optional else ''}{spell(arg.constraint, dialect)}"
                             f"{']' if arg.optional else ''}" for arg in args if not arg.attribute)
        text = (f"{DIALECT}.{mnemonic}: its parser is hand-written ({tool.CPP_SOURCE}:{first}-{last}): {words}; "
                f"operands {operands}; result {result_text(results, dialect)}")
    elif not fmt or unliteral:
        text = generic_text(mnemonic, args, results, dialect)
    else:
        text = custom_text(fmt, mnemonic, args, results, dialect)
        where = [f"<{arg.name}> is {attribute_words(arg.constraint, dialect)}" for arg in shown]
        if where:
            text += f" where {', '.join(where)}"
    if results and not text.startswith(f"{DIALECT}.{mnemonic}:"):
        text = f"%{results[0].name} = {text}" if len(results) == 1 else f"%r:{len(results)} = {text}"
    summary = first_sentence(let_value(body, "summary"))
    if summary and _folded(summary) != _folded(mnemonic):
        text += f"  // {summary}"
    return text


def _folded(text: str) -> str:
    """Return `text` lower-cased without underscores, spaces, dots, colons, or a trailing period."""
    return re.sub(r"[_ .:]|call$", "", text.lower().rstrip("."))


def type_section(types_text: str) -> tuple[list[str], dict[str, str]]:
    """Return the reference lines of every type, and each type def's spelling in an op line."""
    tool = context_tool()
    lines, spellings = [], {}
    for _, text in tool.td_defs(types_text):
        found = _TYPE_DEF.match(text)
        if found is None:
            continue
        name = f"!{DIALECT}.{found.group('mnemonic')}"
        params = parameters(text)
        fmt = let_value(text, "assemblyFormat")
        shown = _param_form(fmt, params) if fmt else ", ".join(param for param, _ in params)
        spellings[found.group("def")] = f"{name}<...>" if params else name
        lines.append(_described(f"{name}<{shown}>" if params else name, params, let_value(text, "summary")))
    return lines, spellings


def parameters(text: str) -> list[tuple[str, str]]:
    """Return (name, kind) of each parameter of a type or attribute def, in order; a C++ kind loses its quotes."""
    inner = re.sub(r"^\(\s*ins\b", "", let_value(text, "parameters")).rstrip(")").strip()
    found = []
    for part in split_top(inner):
        kind, _, name = part.rpartition(":$")
        kind = kind.strip()
        found.append((name.strip(), kind[1:-1] if len(kind) > 1 and kind[0] == kind[-1] == '"' else kind))
    return found


def _described(line: str, params: Sequence[tuple[str, str]], summary: str) -> str:
    """Return `line` with `where <param> is <kind>, ...` and the summary's first sentence after `//`."""
    if params:
        line += "  where " + ", ".join(f"{name} is {kind}" for name, kind in params)
    sentence = first_sentence(summary)
    return f"{line}  // {sentence}" if sentence else line


def first_sentence(text: str) -> str:
    """Return `text` up to and including its first period that a space follows, or all of it."""
    head, found, _ = text.partition(". ")
    return f"{head}." if found else text


def _param_form(fmt: str, params: Sequence[tuple[str, str]]) -> str:
    """Return a type's or attribute's assemblyFormat between its outer `<` and `>`, its $params as names.

    `struct(params)` prints every parameter as `<name> = <value>`, comma
    separated, in order.
    """
    struct = ", ".join(f"{name} = ..." for name, _ in params)
    text = _joined(["", *_format_pieces(fmt, lambda name: name, struct=struct)]).strip()
    return text[1:-1].strip() if text.startswith("<") and text.endswith(">") else text


def attribute_section(types_text: str, enums_text: str) -> tuple[list[str], dict[str, str]]:
    """Return the reference lines of every attribute, and each attribute def's spelling in a generic form."""
    tool = context_tool()
    cases = {}
    for _, text in tool.td_defs(enums_text):
        enum = _ENUM_DEF.match(text)
        if enum is not None:
            cases[enum.group("def")] = [match.group("text") or match.group("symbol")
                                        for symbol in split_top(enum.group("cases"))
                                        for match in [_CASE.search(_case_text(enums_text, symbol))] if match]
    lines, spellings = [], {}
    for _, text in tool.td_defs(types_text):
        enum, struct = _ENUM_ATTR.match(text), _STRUCT_ATTR.match(text)
        if enum is not None:
            spelled = f"#{DIALECT}.{enum.group('mnemonic')}<{' | '.join(cases.get(enum.group('enum'), ['...']))}>"
            spellings[enum.group("def")] = spelled
            lines.append(spelled)
        elif struct is not None:
            name = f"#{DIALECT}.{struct.group('mnemonic')}"
            params = parameters(text)
            form = _param_form(let_value(text, "assemblyFormat"), params)
            lines.append(_described(f"{name}<{form}>", params, let_value(text, "summary")))
            spellings[struct.group("def")] = name + "<...>"
    return lines, spellings


def _case_text(enums_text: str, symbol: str) -> str:
    """Return the text of enum case def `symbol` in the enums file, or the symbol itself when it is inline."""
    match = re.search(rf"^def {re.escape(symbol)}\b[^;]*;", enums_text, re.MULTILINE)
    return match.group(0) if match else symbol


def reference_text(read: Mapping[str, bytes], pin: Mapping[str, str]) -> str:
    """Return the reference (the module docstring) from the sources `read` {tree path: bytes}."""
    tool = context_tool()
    ops_text, cpp = read[tool.OPS_SOURCE].decode("utf-8"), read[tool.CPP_SOURCE].decode("utf-8")
    types_text, enums_text = (read[path].decode("utf-8") for path in tool.TYPE_SOURCES)
    type_lines, type_spellings = type_section(types_text)
    attribute_lines, attribute_spellings = attribute_section(types_text, enums_text)
    constraints = {match.group("name"): match.group(0).split(":", 1)[1].strip().rstrip(";").strip()
                   for match in tool._CONSTRAINT.finditer(ops_text)}
    dialect = Dialect(type_spellings, constraints, attribute_spellings)
    sections = [(match.start(), match.group("title")) for match in tool._SECTION.finditer(ops_text)]
    sources = ", ".join(Path(path).name for path in (tool.OPS_SOURCE, *tool.TYPE_SOURCES, tool.CPP_SOURCE))
    out = [HEAD.format(version=pin["VERSION"], commit=pin["COMMIT"], generator=GENERATOR, sources=sources),
           "Types:", *type_lines, "", "Attributes:", *attribute_lines]
    current = None
    for def_name, body in tool.td_defs(ops_text):
        match = tool._DEF.match(body)
        if match is None:
            continue
        start = ops_text.find(body)
        title = next((name for begin, name in reversed(sections) if begin < start), "Ops")
        if title != current:
            out += ["", f"{title}:"]
            current = title
        out.append(op_line(match.group(2), body, cpp, def_name, dialect))
    return "\n".join(out) + "\n"


def build(tree: Path, pin: Mapping[str, str]) -> dict[str, dict[str, bytes]]:
    """Return {tree name: {file path: bytes}} for the reference pack, its manifest included."""
    tool = context_tool()
    tool.check_tree(tree, pin)
    sources = (tool.OPS_SOURCE, *tool.TYPE_SOURCES, tool.CPP_SOURCE)
    read = {path: (tree / path).read_bytes() for path in sources}
    text = reference_text(read, pin)
    if not text.isascii():
        raise ValueError("the reference holds a character that is not ASCII; the prompts are plain ASCII")
    data = text.encode("ascii")
    document = {
        "generator": GENERATOR, "repo": pin["URL"], "commit": pin["COMMIT"], "language": LANGUAGE,
        "entries": [{"key": KEY, "sources": [{"path": path, "sha256": _digest(read[path])} for path in sources],
                     "sha256": _digest(data)}],
        "bytes": len(data),
    }
    head = (
        f"# Generated by {GENERATOR} from tt-mlir at {pin['COMMIT']} (toolchains/tt-mlir.pin); do not edit.\n"
        "# Only this manifest is tracked; the reference beside it renders tt-mlir's dialect and is gitignored\n"
        "# (OQ-018). No source is a tt-pairs-v0 item (Agent Rule 5). The verbatim_context stage carries it.\n"
    )
    manifest = (head + yaml.safe_dump(document, sort_keys=False, width=120)).encode("ascii")
    return {f"context/{REF}": {f"{KEY}.txt": data, "MANIFEST.yaml": manifest}}


def _digest(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def write(built: Mapping[str, Mapping[str, bytes]], out: Path) -> None:
    """Write every built file under `out`."""
    for tree, files in built.items():
        for path, data in files.items():
            target = out / tree / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def compare(built: Mapping[str, Mapping[str, bytes]], out: Path) -> list[str]:
    """Return one message per built file that is missing under `out` or differs from it."""
    problems = []
    for tree, files in built.items():
        for path, data in files.items():
            target = out / tree / path
            if not target.is_file():
                problems.append(f"{target} is missing")
            elif target.read_bytes() != data:
                problems.append(f"{target} differs from what the tree builds")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
