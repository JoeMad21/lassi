"""Build an MLIR dialect's API reference from the pinned dialect sources (tasks P17.44 and P17.47).

Usage:

    uv run tools/build_ttkernel_reference.py fetch --dest <dir>
    uv run tools/build_ttkernel_reference.py build --tree <dir> [--dialect D] [--out <assets dir>]
    uv run tools/build_ttkernel_reference.py check --tree <dir> [--dialect D] [--out <assets dir>]

D is one of DIALECTS: ttkernel (the default, task P17.44), ttir, or d2m
(task P17.47). The tree is tt-mlir at toolchains/tt-mlir.pin's COMMIT, as
tools/build_ttkernel_context.py takes it (its check_tree: an install's
src/tt-mlir, or a directory whose lassi-fetch.txt names the commit). `fetch`
fills such a directory with every dialect's sources at the commit from
GitHub, each checked against the git blob sha1 that the git trees API lists
for it. In the window of 2026-10-27 (rx 20261027-005611-window-7998) no cpp
-> ttkernel attempt built, and the records of 2026-10-26 show the main source
of wrong syntax: the reference card the model wrote of the dialect in
summarize_context invented forms. A reference replaces that card with the
dialect's own facts, rendered by rule, never written by hand.

`build` writes context/<REF>/ under `--out` (default: assets/), a context
pack (lassi.prompts.assets load_tree) of one entry whose key ends in the
dialect's name, its language (ttkernel_ref.ttkernel, ttir_ref.ttir,
d2m_ref.d2m):

- Types: every type of the dialect's type sources as a module spells it,
  `!<dialect>.<mnemonic>`, with its parameters in the order its
  assemblyFormat prints them, and its summary.
- Attributes: every attribute, `#<dialect>.<mnemonic>`; an enum attribute
  with its cases' strings from the enum sources, a struct attribute with its
  format and parameters.
- Ops: every op of the op sources, one line each (op_line): the op as its
  assemblyFormat prints it, with each operand `%<name>` and each attribute
  `<name>`, an optional part in [ ], then ` : (<operand types>) -> <result
  types>`, each operand type spelled from its constraint (spell: i32 for
  I32, index | i32 for IndexLike, !ttkernel.cb<...> for TTKernel_CB,
  tensor<...> for AnyRankedTensor); an op whose format holds an attribute
  other than an integer, bool, or string literal, and an op with no assembly
  format, are given in MLIR's generic form instead, `"<dialect>.<op>"(<operands>)
  <{<attribute> = <value>}> : (...) -> ...`, which ttmlir-opt parses for
  every op; an op whose parser is hand-written gives the format comment its
  C++ source holds for it. The op's summary follows ` // ` unless it only
  repeats the op's name. An op's arguments, results, and format come from
  its def or, when the def sets none, from the TableGen class it derives
  from (TTIR's elementwise ops set theirs in their class).

TTKernel's ops are listed under the sections of TTKernelOps.td, in file
order, with the rules of task P17.44, unchanged, so its tracked manifest
holds byte for byte. TTIR and D2M (task P17.47) are larger and lean on
ttcore, so their references are grouped by op family, the section of the
.td file and the class an op derives from (families), and take the rules
P17.44 did not need (extended): the attributes a declarative format leaves
to its attribute dictionary, written in { } where attr-dict stands; an enum
parameter's cases; an enum attribute with no format of its own written
`#<dialect><mnemonic case>`, as MLIR prints it; operandSegmentSizes and the
regions in a generic form; and a section listing the ttcore types and
attributes the ops name (and those every module of the dialect writes,
DialectSources.foreign_kept), spelled from TTCoreOpsTypes.td and
TTCoreOpsEnums.td. Descriptions, builders, verifiers, traits, interfaces,
and C++ declarations are left out of every reference: a description's
examples are prose an op's form does not check (TTIR's show forms such as
`ttir.abs(%x)` that its ops, with no assembly format, do not parse), and the
rest says nothing of an op's textual form.

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
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lassi.toolchains import pins  # noqa: E402

GENERATOR = "tools/build_ttkernel_reference.py"
PIN = "tt-mlir"
# The pin's commit as the pack names carry it (toolchains/tt-mlir.pin COMMIT).
SHORT = "5f396fd6"
_TTKERNEL_IR = "include/ttmlir/Dialect/TTKernel/IR"
_TTIR_IR = "include/ttmlir/Dialect/TTIR/IR"
_D2M_IR = "include/ttmlir/Dialect/D2M/IR"
_TTCORE_IR = "include/ttmlir/Dialect/TTCore/IR"


@dataclass(frozen=True)
class DialectSources:
    """One dialect's reference: its names, its pinned sources, and the rules it is rendered with (DIALECTS).

    `prefix` is the TableGen prefix of its defs (TTKernel_), `name` the
    prefix of its textual forms (ttkernel.) and the language of its pack.
    `ops` hold the op defs, `types` the type and attribute defs, `enums` the
    enums, `bases` the op classes and type constraints the ops derive from
    or name (read, never listed), `foreign` another dialect's sources (its
    ...Types.td, whose types and attributes the ops name, and ...Enums.td:
    ttcore's), `cpp` the C++ that holds a hand-written parser's format
    comment. `foreign_kept` names the foreign defs listed even when no op
    names one. `families` and `extended` choose the rules (the module
    docstring); ttkernel's are off, so its reference stays task P17.44's.
    """

    name: str
    prefix: str
    ops: tuple[str, ...]
    types: tuple[str, ...]
    enums: tuple[str, ...]
    cpp: tuple[str, ...]
    bases: tuple[str, ...] = ()
    foreign: tuple[str, ...] = ()
    foreign_kept: tuple[str, ...] = ()
    families: bool = False
    extended: bool = False

    @property
    def ref(self) -> str:
        """Return the pack's name, <name>-ref-<SHORT>."""
        return f"{self.name}-ref-{SHORT}"

    @property
    def key(self) -> str:
        """Return the pack's one entry key, <name>_ref.<name>."""
        return f"{self.name}_ref.{self.name}"

    @property
    def sources(self) -> tuple[str, ...]:
        """Return every source the reference reads, in the manifest's order."""
        return (*self.ops, *self.types, *self.enums, *self.bases, *self.foreign, *self.cpp)


_TTCORE_SOURCES = (f"{_TTCORE_IR}/TTCoreOpsTypes.td", f"{_TTCORE_IR}/TTCoreOpsEnums.td")
# The ttcore defs every TTIR or D2M module of a device program writes whether or not an op names them: the tile
# element type, a tensor's layout, its memory space and grid, and a generic op's iterator types [DESIGN].
_TTCORE_KEPT = ("TTCore_Tile", "TTCore_MetalLayoutAttr", "TTCore_MemorySpaceAttr", "TTCore_GridAttr",
                "TTCore_IteratorTypeAttr")
TTKERNEL = DialectSources(
    name="ttkernel", prefix="TTKernel", ops=(f"{_TTKERNEL_IR}/TTKernelOps.td",),
    types=(f"{_TTKERNEL_IR}/TTKernelOpsTypes.td",), enums=(f"{_TTKERNEL_IR}/TTKernelOpsEnums.td",),
    cpp=("lib/Dialect/TTKernel/IR/TTKernelOps.cpp",),
)
TTIR = DialectSources(
    name="ttir", prefix="TTIR", ops=(f"{_TTIR_IR}/TTIROps.td",),
    types=(f"{_TTIR_IR}/TTIROpsTypes.td", f"{_TTIR_IR}/TTIROpsAttrs.td"), enums=(f"{_TTIR_IR}/TTIROpsEnums.td",),
    cpp=(), bases=(f"{_TTIR_IR}/TTIRBase.td",), foreign=_TTCORE_SOURCES, foreign_kept=_TTCORE_KEPT,
    families=True, extended=True,
)
D2M = DialectSources(
    name="d2m", prefix="D2M", ops=(f"{_D2M_IR}/D2MOps.td", f"{_D2M_IR}/D2MGenericRegionOps.td"),
    types=(f"{_D2M_IR}/D2MOpsTypes.td", f"{_D2M_IR}/D2MOpsAttrs.td"), enums=(f"{_D2M_IR}/D2MOpsEnums.td",),
    cpp=("lib/Dialect/D2M/IR/D2MGenericRegionOps.cpp",), bases=(f"{_D2M_IR}/D2MBase.td",),
    foreign=_TTCORE_SOURCES, foreign_kept=_TTCORE_KEPT, families=True, extended=True,
)
DIALECTS = {dialect.name: dialect for dialect in (TTKERNEL, TTIR, D2M)}
# The ttkernel reference's names, as task P17.44 gave them.
REF = TTKERNEL.ref
LANGUAGE = TTKERNEL.name
KEY = TTKERNEL.key
DIALECT = TTKERNEL.name
# Attribute constraints a custom format prints as a plain literal, with the words the reference gives each.
LITERAL_ATTRS = (
    (re.compile(r"[SU]?I\d+Attr"), "an integer literal"),
    (re.compile(r"BoolAttr"), "true or false"),
    (re.compile(r"StrAttr"), 'a string literal "..."'),
    (re.compile(r"F\d+Attr"), "a float literal"),
)
# Builtin type constraints and how a module writes a value of each; others resolve through the sources.
BUILTIN_TYPES = {
    "Index": "index", "AnyType": "<any type>", "AnyRankedTensor": "tensor<...>", "AnyTensor": "tensor<...>",
    "AnyNon0RankedMemRef": "memref<...>", "AnyMemRef": "memref<...>", "AnyRankedOrUnrankedMemRef": "memref<...>",
    "AnyFloat": "<float type>", "AnyInteger": "<integer type>", "AnySignlessInteger": "<integer type>",
    "NoneType": "none",
}
# Builtin attribute constraints and their values in a generic form's property dictionary.
BUILTIN_ATTRS = {
    "I32ArrayAttr": "[<integer> : i32, ...]", "I64ArrayAttr": "[<integer> : i64, ...]",
    "DenseI32ArrayAttr": "array<i32: <integer>, ...>", "DenseI64ArrayAttr": "array<i64: <integer>, ...>",
    "DenseBoolArrayAttr": "array<i1: true | false, ...>", "AffineMapAttr": "affine_map<...>",
    "AffineMapArrayAttr": "[affine_map<...>, ...]", "TypeAttr": "<type>", "Builtin_StringAttr": '"..."',
    "SymbolRefAttr": "@<symbol>", "FlatSymbolRefAttr": "@<symbol>", "ElementsAttr": "dense<...> : <tensor type>",
    "DenseElementsAttr": "dense<...> : <tensor type>", "I64ElementsAttr": "dense<...> : tensor<...xi64>",
    "ArrayAttr": "[...]", "IndexAttr": "<integer> : index", "F32Attr": "<float> : f32", "F64Attr": "<float> : f64",
}
# The C++ classes a `Type<pred, "summary", "<class>">` constraint names, and the builtin constraint each stands for.
_CPP_TYPES = {"::mlir::MemRefType": "AnyMemRef", "::mlir::RankedTensorType": "AnyRankedTensor"}
# An integer attribute constraint (its value in the generic form is `<integer> : i<width>`), and the builtin integer
# and float type constraints (I32 is spelled i32, F32 f32).
_GENERIC_INT = re.compile(r"(?P<sign>[SU]?)I(?P<width>\d+)Attr")
_BUILTIN_INT = re.compile(r"(?P<sign>[SU]?)I(?P<width>\d+)")
_BUILTIN_FLOAT = re.compile(r"F(?P<width>\d+)")
_WRAPPED = re.compile(r"(?P<kind>Optional|Variadic|OptionalAttr|DefaultValuedAttr|DefaultValuedOptionalAttr|Arg|Res"
                      r"|ConfinedAttr)<(?P<inner>.*)>")
_ANY_OF = re.compile(r"AnyTypeOf<\[(?P<inner>.*)\]>")
_FORMAT_TOKEN = re.compile(
    r"`(?P<lit>[^`]*)`|\$(?P<var>\w+)|(?P<struct>struct\(params\))|(?P<ftype>functional-type)\((?P<fargs>[^)]*)\)"
    r"|(?P<dict>attr-dict-with-keyword|attr-dict)|(?P<close>\)\?)|(?P<open>\()|(?P<anchor>\^)"
    r"|(?P<tdir>type|qualified)\(\s*\$(?P<targ>\w+)\s*\)"
    r"|(?P<custom>custom<\w+>)\((?P<cargs>(?:[^()]|\([^()]*\))*)\)"
)
# The markers of an optional group's bounds among a format's pieces (_format_pieces), shown as [ ] (_joined).
_OPEN, _CLOSE = "<optional-group>", "</optional-group>"
_LET = r"^\s*let {field}\s*=\s*"
_DAG = r"^\s*dag {name}\s*=\s*"
_RECORD = re.compile(r"^(?P<kind>def|class)\s+(?P<name>\w+)", re.MULTILINE)
_SECTION = re.compile(r"^//===-+===//\n// (?P<title>[^\n]+)\n//===-+===//$", re.MULTILINE)
_MNEMONIC = re.compile(r':\s*\w+<\s*"(?P<mnemonic>[^"]+)"')
_ENUM_DEF = re.compile(
    r'^def (?P<def>\w+)\s*:?\s*I32EnumAttr<\s*"(?P<cpp>\w+)"\s*,\s*"[^"]*"\s*,\s*\[(?P<cases>[^\]]*)\]', re.DOTALL)
_CASE = re.compile(r'I32EnumAttrCase<\s*"(?P<symbol>\w+)"\s*,\s*-?\d+\s*(?:,\s*"(?P<text>[^"]*)")?\s*>')
_ARRAY_ATTR = re.compile(r"^def (?P<def>\w+)\s*:\s*TypedArrayAttrBase<\s*(?P<element>\w+)")
_PARAMETER = re.compile(r'\w*Parameter<\s*"(?P<cpp>[\w:]+)"')
HEAD = (
    "// The ttkernel dialect of tt-mlir {version} (commit {commit}): every type, attribute, and op, rendered by\n"
    "// {generator} from {sources} at that commit.\n"
    "// An op line is the op's textual form: %name is an operand, <name> an attribute (\"where\" says how it is\n"
    "// written), [ ] an optional part, and after \" : \" come the operand types in order, \" -> \", and the result\n"
    "// type, () when there is none; a | b is either type. A type with parameters is written with them, such as\n"
    "// !ttkernel.cb<...> with its parameters below. A line in quotes, \"ttkernel.<op>\"(...), is MLIR's generic\n"
    "// form, which every op accepts: properties go in <{{ }}>, a [ ] one only when set.\n"
)
FAMILY_HEAD = (
    "// The {name} dialect of tt-mlir {version} (commit {commit}): every type, attribute, and op, rendered by\n"
    "// {generator} from {sources} at that commit.\n"
    "// Ops are grouped by family: the section of the .td file that defines them and the TableGen class they\n"
    "// derive from. An op line is the op's textual form: %name is an operand, x... a variadic one, <name> an\n"
    "// attribute (\"where\" says how it is written), [ ] an optional part, and after \" : \" come the operand types\n"
    "// in order, \" -> \", and the result type, () when there is none; a | b is either type, tensor<...> any ranked\n"
    "// tensor, memref<...> any memref. A type or attribute with parameters is written with them, as listed below.\n"
    "// A line in quotes, \"{name}.<op>\"(...), is MLIR's generic form, which every op accepts: properties go in\n"
    "// <{{ }}>, a [ ] one only when set, then the op's regions in ({{...}}). In an op's own form, {{ }} holds the\n"
    "// attributes its format leaves to the attribute dictionary. Descriptions, builders, verifiers, traits, and\n"
    "// interfaces are left out.\n"
)


@dataclass(frozen=True)
class Arg:
    """One argument or result of an op: its name, its constraint as written, and how it is declared."""

    name: str
    constraint: str
    attribute: bool
    optional: bool
    variadic: bool


@dataclass(frozen=True)
class Record:
    """One top-level TableGen def or class: its name, the class it derives from first, its text, and its offset."""

    kind: str
    name: str
    parent: str
    text: str
    start: int


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; 0 on success, 1 when check finds a difference, 2 on a refusal."""
    args = _parser().parse_args(argv)
    try:
        pin = pins.read_pin(PIN)
        if args.command == "fetch":
            count = fetch(Path(args.dest), pin)
            print(f"build_ttkernel_reference: {count} files of tt-mlir at {pin['COMMIT']} in {args.dest}")
            return 0
        built = build(Path(args.tree), pin, DIALECTS[args.dialect])
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
    fetcher = sub.add_parser("fetch")
    fetcher.add_argument("--dest", required=True)
    for name in ("build", "check"):
        command = sub.add_parser(name)
        command.add_argument("--tree", required=True)
        command.add_argument("--dialect", choices=sorted(DIALECTS), default=TTKERNEL.name)
        command.add_argument("--out", default=str(REPO / "assets"))
    return parser


def context_tool() -> ModuleType:
    """Return tools/build_ttkernel_context.py, whose tree check, fetch helpers, and format comments this tool shares."""
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


def fetch(dest: Path, pin: Mapping[str, str]) -> int:
    """Download every dialect's sources at the pin's COMMIT into `dest`, each checked by its blob sha1; their count.

    The record it leaves names the commit as the context tool's does, so
    check_tree accepts the directory.
    """
    tool = context_tool()
    if dest.resolve() == REPO or REPO in dest.resolve().parents:
        raise ValueError(f"fetch writes tt-mlir's text outside this repository, not under {REPO}")
    commit, repo = pin["COMMIT"], pin["URL"].removesuffix(".git").removeprefix("https://github.com/")
    listing = json.loads(tool._get(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"))
    blobs = {entry["path"]: entry["sha"] for entry in listing["tree"] if entry["type"] == "blob"}
    paths = sorted({path for dialect in DIALECTS.values() for path in dialect.sources})
    for path in paths:
        if path not in blobs:
            raise ValueError(f"{path} is not in tt-mlir's tree at {commit}")
        data = tool._get(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}")
        if tool.blob_sha1(data) != blobs[path]:
            raise ValueError(f"{path} at {commit} does not match its git blob sha1 {blobs[path]}")
        (dest / path).parent.mkdir(parents=True, exist_ok=True)
        (dest / path).write_bytes(data)
    (dest / tool.FETCH_RECORD).write_bytes(f"tt-mlir {commit} {GENERATOR}\n".encode("ascii"))
    return len(paths)


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
    return "" if match is None else _value_at(body, match.end())


def _value_at(body: str, start: int) -> str:
    """Return the TableGen value that starts at `start` of `body`, through its `;`, on one line, quotes removed."""
    rest = body[start:]
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


def records(text: str) -> list[Record]:
    """Return every top-level `def` and `class` of a TableGen file, in file order, each with its whole text.

    A record runs through the `;` that ends its header or the `}` that
    closes its body, counted outside strings, `//` comments, and `[{ }]`
    code blocks.
    """
    found: list[Record] = []
    end = 0
    for match in _RECORD.finditer(text):
        if match.start() < end:
            continue
        end, parent = _record_end(text, match.end())
        found.append(Record(match.group("kind"), match.group("name"), parent, text[match.start():end],
                            match.start()))
    return found


def _record_end(text: str, index: int) -> tuple[int, str]:
    """Return (the end of the record whose header continues at `index`, the first class it derives from)."""
    depth, parent = 0, ""
    while index < len(text):
        char = text[index]
        if char == '"' or text.startswith("//", index):
            index = _skipped(text, index)
            continue
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        elif depth == 0 and char == ":" and not parent:
            named = re.match(r"\s*(\w+)", text[index + 1:])
            parent = named.group(1) if named else ""
        elif depth == 0 and char == ";":
            return index + 1, parent
        elif depth == 0 and char == "{":
            return _body_end(text, index), parent
        index += 1
    return len(text), parent


def _body_end(text: str, index: int) -> int:
    """Return the index after the `}` that closes the body opened at `index`."""
    depth = 0
    while index < len(text):
        if text.startswith("[{", index):
            close = text.find("}]", index + 2)
            index = len(text) if close == -1 else close + 2
            continue
        char = text[index]
        if char == '"' or text.startswith("//", index):
            index = _skipped(text, index)
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return len(text)


def _skipped(text: str, index: int) -> int:
    """Return the index after the string literal or `//` comment that starts at `index`."""
    if text[index] == '"':
        index += 1
        while index < len(text) and text[index] not in '"\n':
            index += 2 if text[index] == "\\" else 1
        return index + 1
    newline = text.find("\n", index)
    return len(text) if newline == -1 else newline


def uncommented(text: str) -> str:
    """Return `text` with every `//` comment cut from its line, a `//` inside a string literal kept."""
    lines = []
    for line in text.split("\n"):
        index, quoted = 0, False
        while index < len(line):
            if line[index] == '"':
                quoted = not quoted
            elif not quoted and line.startswith("//", index):
                line = line[:index].rstrip()
                break
            index += 2 if line[index] == "\\" else 1
        lines.append(line)
    return "\n".join(lines)


def parse_args(spec: str, attributes: bool = True) -> list[Arg]:
    """Return the Args of an `(ins ...)` or `(outs ...)` list, in order; `attributes` False reads none as one."""
    inner = spec.strip()
    inner = re.sub(r"^\(\s*(ins|outs)\b", "", inner).rstrip(")").strip() if inner else ""
    found: list[Arg] = []
    for part in split_top(inner):
        constraint, _, name = part.rpartition(":$")
        constraint = constraint.strip()
        kind, core = _unwrapped(constraint)
        found.append(Arg(name.strip(), core, attributes and (core.endswith("Attr") or core.startswith("AnyAttrOf<")),
                         kind in ("Optional", "OptionalAttr", "DefaultValuedAttr", "DefaultValuedOptionalAttr",
                                  "UnitAttr") or core == "UnitAttr", kind == "Variadic"))
    return found


def _unwrapped(constraint: str) -> tuple[str, str]:
    """Return (wrapper kind or "", the constraint inside it); DefaultValuedAttr<X, "v"> and Arg<X, ...> give X.

    A wrapper inside a wrapper that keeps the value one value (Arg, Res,
    ConfinedAttr) is unwrapped too, so Arg<Optional<X>> gives Optional and X.
    """
    match = _WRAPPED.fullmatch(constraint)
    if match is None:
        return "", constraint
    inner = match.group("inner")
    kind = match.group("kind")
    if kind in ("DefaultValuedAttr", "DefaultValuedOptionalAttr", "Arg", "Res", "ConfinedAttr"):
        inner = split_top(inner)[0]
    if kind in ("Arg", "Res", "ConfinedAttr"):
        return _unwrapped(inner.strip())
    return kind, inner.strip()


@dataclass(frozen=True)
class Dialect:
    """What spell and the op lines read: type spellings by def name, constraint bodies, and attribute spellings.

    `name` is the textual prefix of the dialect's ops; `enums` maps a C++
    enum name to its cases joined by " | " (extended rules only).
    """

    types: Mapping[str, str]
    constraints: Mapping[str, str]
    attributes: Mapping[str, str]
    name: str = TTKERNEL.name
    enums: Mapping[str, str] = field(default_factory=dict)
    extended: bool = False


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
    return BUILTIN_TYPES.get(core, core)


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
    if arg.constraint in dialect.attributes:
        return dialect.attributes[arg.constraint]
    alternatives = re.fullmatch(r"AnyAttrOf<\[(?P<inner>.*)\]>", arg.constraint)
    if alternatives is not None:
        return " | ".join(generic_value(Arg(arg.name, part, True, False, False), dialect)
                          for part in split_top(alternatives.group("inner")))
    return BUILTIN_ATTRS.get(arg.constraint, f"<{arg.constraint}>")


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
    typed = {**{arg.name: arg for arg in results}, **by_name}
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

    def type_of(name: str) -> str:
        """Return type($name): the spelled type of operand or result `name` (var's text for any other name)."""
        arg = typed.get(name)
        if arg is None:
            return var(name)
        return spell(arg.constraint, dialect) + ("..." if arg.variadic else "")

    def dictionary() -> str:
        """Return what attr-dict stands for: { } of the attributes the format does not name (extended rules)."""
        named = set(re.findall(r"\$(\w+)", fmt))
        return _dictionary([arg for arg in args if arg.attribute and arg.name not in named], dialect)

    pieces = _format_pieces(fmt, var, functional, type_of=type_of,
                            dictionary=dictionary if dialect.extended else None)
    return _joined([f"{dialect.name}.{mnemonic}", *pieces])


def _dictionary(attributes: Sequence[Arg], dialect: Dialect) -> str:
    """Return `{a = <value>, [b = <value>]}` of `attributes`, the whole in [ ] when each is optional; "" for none."""
    if not attributes:
        return ""
    every = all(arg.optional for arg in attributes)
    entries = []
    for arg in attributes:
        entry = arg.name if arg.constraint == "UnitAttr" else f"{arg.name} = {generic_value(arg, dialect)}"
        entries.append(f"[{entry}]" if arg.optional and not every else entry)
    text = "{" + ", ".join(entries) + "}"
    return f"[{text}]" if every else text


def _format_pieces(fmt: str, var: Callable[[str], str], functional: Callable[[str], str] | None = None,
                   struct: str = "", type_of: Callable[[str], str] | None = None,
                   dictionary: Callable[[], str] | None = None) -> list[str]:
    """Return the pieces an assemblyFormat prints, in order: literals, `var` of each $name, and group markers.

    An optional group's parentheses become _OPEN and _CLOSE (shown as
    [ ]), attr-dict prints `dictionary()` (nothing when it is None or
    empty), `functional` renders functional-type, struct(params) prints
    `struct`, type($x) prints `type_of(x)` (the name when it is None), and
    custom<D>(...) prints the directive with its arguments' names. A literal
    newline or space prints nothing.
    """
    pieces: list[str] = []
    for match in _FORMAT_TOKEN.finditer(fmt):
        if match.group("struct") is not None:
            pieces.append(struct)
        elif match.group("lit") is not None:
            text = match.group("lit").strip()
            pieces.append(" " if text in ("", "\\n") else text)
        elif match.group("var") is not None:
            pieces.append(var(match.group("var")))
        elif match.group("ftype") is not None and functional is not None:
            pieces.append(functional(match.group("fargs")))
        elif match.group("dict") is not None and dictionary is not None:
            pieces.append(dictionary() or " ")
        elif match.group("open") is not None:
            pieces.append(_OPEN)
        elif match.group("close") is not None:
            pieces.append(_CLOSE)
        elif match.group("tdir") is not None:
            name = match.group("targ")
            pieces.append(type_of(name) if type_of is not None and match.group("tdir") == "type" else var(name))
        elif match.group("custom") is not None:
            names = ", ".join(var(name) for name in re.findall(r"\$(\w+)", match.group("cargs")))
            pieces.append(f"{match.group('custom')}({names})")
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


def generic_text(mnemonic: str, args: Sequence[Arg], results: Sequence[Arg], dialect: Dialect, *,
                 segments: bool = False, regions: bool = False) -> str:
    """Return the op in MLIR's generic form, its attributes as properties (the module docstring).

    `segments` adds operandSegmentSizes (an op with AttrSizedOperandSegments),
    and `regions` the op's regions after the properties.
    """
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
    if segments:
        props.append("operandSegmentSizes = array<i32: " + ", ".join(f"<count of %{arg.name}>" for arg in operands)
                     + ">")
    properties = f" <{{{', '.join(props)}}}>" if props else ""
    return (f'"{dialect.name}.{mnemonic}"({names}){properties}{" ({...})" if regions else ""} : '
            f"{type_list(operands, dialect)} -> {result_text(results, dialect)}")


@dataclass(frozen=True)
class OpDef:
    """One op as the reference reads it: its mnemonic, def name, fields resolved through its classes, and C++."""

    mnemonic: str
    def_name: str
    arguments: str
    results: str
    fmt: str
    custom: bool
    summary: str
    segments: bool
    regions: bool


def op_def(record: Record, classes: Mapping[str, Record]) -> OpDef:
    """Return the OpDef of an op record, each field from the record or, when it sets none, its nearest class."""
    chain = _chain(record, classes)
    mnemonic = _MNEMONIC.search(record.text)
    return OpDef(
        mnemonic="" if mnemonic is None else mnemonic.group("mnemonic"), def_name=record.name,
        arguments=_resolved(chain, "arguments"), results=_resolved(chain, "results"),
        fmt=_inherited(chain, "assemblyFormat"), custom=_inherited(chain, "hasCustomAssemblyFormat") == "1",
        summary=let_value(record.text, "summary"),
        segments=any("AttrSizedOperandSegments" in _header(link.text) for link in chain),
        regions=bool(_inherited(chain, "regions")),
    )


def _chain(record: Record, classes: Mapping[str, Record]) -> list[Record]:
    """Return `record` and the classes it derives from, nearest first, as far as the sources define them."""
    chain, seen = [record], {record.name}
    while chain[-1].parent in classes and chain[-1].parent not in seen:
        seen.add(chain[-1].parent)
        chain.append(classes[chain[-1].parent])
    return chain


def _header(text: str) -> str:
    """Return a record's text up to its body's `{`, or all of it when it has no body."""
    brace = text.find("{")
    return text if brace == -1 else text[:brace]


def _inherited(chain: Sequence[Record], name: str) -> str:
    """Return the first `let <name>` value along `chain`; "" when none sets it."""
    for link in chain:
        value = let_value(uncommented(link.text), name)
        if value:
            return value
    return ""


def _resolved(chain: Sequence[Record], name: str) -> str:
    """Return `let <name>` along `chain` as one dag: a `dag` field it names is substituted, !con(a, b) joined."""
    value = _inherited(chain, name)
    if not value:
        return ""

    def dag(term: str) -> str:
        """Return the dag text of one term: itself when it is a dag, else the `dag <term>` field it names."""
        term = term.strip()
        if term.startswith("("):
            return term
        for link in chain:
            found = re.search(_DAG.format(name=re.escape(term)), uncommented(link.text), re.MULTILINE)
            if found is not None:
                return _value_at(uncommented(link.text), found.end())
        return term

    joined = re.fullmatch(r"!con\((?P<terms>.*)\)", value)
    if joined is None:
        return dag(value)
    kind, parts = "ins", []
    for term in split_top(joined.group("terms")):
        inner = re.match(r"\(\s*(?P<kind>ins|outs)\b(?P<rest>.*)\)$", dag(term))
        if inner is not None:
            kind = inner.group("kind")
            parts += split_top(inner.group("rest"))
    return f"({kind} {', '.join(parts)})"


def op_line(op: OpDef, cpp: Mapping[str, str], dialect: Dialect) -> str:
    """Return the reference line of one op (the module docstring): its form, then a `where` and its summary."""
    args = parse_args(op.arguments)
    results = parse_args(op.results, attributes=False)
    named = {match.group(1) for match in re.finditer(r"\$(\w+)", op.fmt)}
    shown = [arg for arg in args if arg.attribute and arg.name in named]
    unliteral = [arg for arg in shown if attribute_words(arg.constraint, dialect) is None]
    prefix = f"{dialect.name}.{op.mnemonic}"
    if op.custom:
        text = _hand_written(op, args, results, cpp, dialect)
    elif not op.fmt or unliteral:
        text = generic_text(op.mnemonic, args, results, dialect, segments=dialect.extended and op.segments,
                            regions=dialect.extended and op.regions)
    else:
        text = custom_text(op.fmt, op.mnemonic, args, results, dialect)
        where = [f"<{arg.name}> is {attribute_words(arg.constraint, dialect)}" for arg in shown]
        if where:
            text += f" where {', '.join(where)}"
    if results and not text.startswith(f"{prefix}:"):
        text = f"%{results[0].name} = {text}" if len(results) == 1 else f"%r:{len(results)} = {text}"
    summary = first_sentence(op.summary)
    if summary and _folded(summary) != _folded(op.mnemonic) and not _restates(summary, op.mnemonic, dialect):
        text += f"  // {summary}"
    return text


def _hand_written(op: OpDef, args: Sequence[Arg], results: Sequence[Arg], cpp: Mapping[str, str],
                  dialect: Dialect) -> str:
    """Return the line of an op whose parser is hand-written: the format comment its C++ holds, and its types."""
    class_name = op.def_name.split("_", 1)[1]
    found = format_comment(cpp, class_name)
    if found is None:
        raise ValueError(f"{dialect.name}.{op.mnemonic} ({op.def_name}) has a hand-written parser, but "
                         f"{', '.join(cpp) or 'no C++ source read'} holds no format comment for it")
    path, first, last, lines = found
    words = " ".join(" ".join(line.strip().removeprefix("//").split()) for line in lines)
    operands = ", ".join(f"%{arg.name}: {'[' if arg.optional else ''}{spell(arg.constraint, dialect)}"
                         f"{'...' if arg.variadic and dialect.extended else ''}{']' if arg.optional else ''}"
                         for arg in args if not arg.attribute)
    return (f"{dialect.name}.{op.mnemonic}: its parser is hand-written ({path}:{first}-{last}): {words}; "
            f"operands {operands}; result {result_text(results, dialect)}")


def format_comment(cpp: Mapping[str, str], class_name: str) -> tuple[str, int, int, list[str]] | None:
    """Return (C++ path, first line, last line, comment lines) of `class_name`'s format comment; None when none.

    The context tool's rule comes first (a `// <class> custom assembly
    format` line, then a `// Format:` line and the comment lines after it);
    then the same heading in any case, then the first comment line after it
    that starts `// Format:` and the comment lines that follow it, indented
    or not (D2M's hand-written parsers keep their comment in the parse
    function).
    """
    tool = context_tool()
    for path, text in cpp.items():
        found = tool.custom_format(text, class_name)
        if found is not None:
            return path, *found
    for path, text in cpp.items():
        lines = text.split("\n")
        heads = [index for index, line in enumerate(lines)
                 if line.strip().lower() == f"// {class_name} custom assembly format".lower()]
        start = next((index for index in range(heads[0], len(lines)) if lines[index].strip().startswith(
            "// Format:")), None) if heads else None
        if start is None:
            continue
        end = start
        while end + 1 < len(lines) and lines[end + 1].strip().startswith("//"):
            end += 1
        return path, start + 1, end + 1, lines[start:end + 1]
    return None


def _folded(text: str) -> str:
    """Return `text` lower-cased without underscores, spaces, dots, colons, or a trailing period."""
    return re.sub(r"[_ .:]|call$", "", text.lower().rstrip("."))


def _restates(summary: str, mnemonic: str, dialect: Dialect) -> bool:
    """Return True when a summary is the dialect's name, the op's name, and op (extended rules: D2M Tile Add Op)."""
    if not dialect.extended:
        return False
    words = [word for word in re.split(r"[\s_.:]+", summary.lower().rstrip(".")) if word]
    trimmed = [word for word in words if word not in (dialect.name, "op", "operation")]
    return "".join(trimmed) == _folded(mnemonic)


def first_sentence(text: str) -> str:
    """Return `text` up to and including its first period that a space follows, or all of it."""
    head, found, _ = text.partition(". ")
    return f"{head}." if found else text


def parameters(text: str) -> list[tuple[str, str]]:
    """Return (name, kind) of each parameter of a type or attribute def, in order; a C++ kind loses its quotes."""
    inner = re.sub(r"^\(\s*ins\b", "", let_value(uncommented(text), "parameters")).rstrip(")").strip()
    found = []
    for part in split_top(inner):
        kind, _, name = part.rpartition(":$")
        kind = kind.strip()
        found.append((name.strip(), kind[1:-1] if len(kind) > 1 and kind[0] == kind[-1] == '"' else kind))
    return found


def _described(line: str, params: Sequence[tuple[str, str]], summary: str, dialect: Dialect) -> str:
    """Return `line` with `where <param> is <kind>, ...` and the summary's first sentence after `//`.

    Under the extended rules a parameter whose C++ kind is an enum of the
    sources is given as its cases.
    """
    if params:
        line += "  where " + ", ".join(f"{name} is {_kind_words(kind, dialect)}" for name, kind in params)
    sentence = first_sentence(summary)
    return f"{line}  // {sentence}" if sentence else line


def _kind_words(kind: str, dialect: Dialect) -> str:
    """Return how the reference names a parameter's kind: an enum's cases (extended rules), else the kind."""
    if not dialect.extended:
        return kind
    wrapped = _PARAMETER.match(kind)
    cpp = (wrapped.group("cpp") if wrapped else kind).split("::")[-1]
    return dialect.enums.get(cpp, kind)


def _param_form(fmt: str, params: Sequence[tuple[str, str]]) -> str:
    """Return a type's or attribute's assemblyFormat between its outer `<` and `>`, its $params as names.

    `struct(params)` prints every parameter as `<name> = <value>`, comma
    separated, in order.
    """
    struct = ", ".join(f"{name} = ..." for name, _ in params)
    text = _joined(["", *_format_pieces(fmt, lambda name: name, struct=struct)]).strip()
    return text[1:-1].strip() if text.startswith("<") and text.endswith(">") else text


@dataclass(frozen=True)
class Listed:
    """The Types and Attributes sections of one dialect: lines by def name in file order, and their spellings."""

    types: dict[str, str]
    attributes: dict[str, str]
    type_spellings: dict[str, str]
    attribute_spellings: dict[str, str]
    enums: dict[str, str]


def listed(sources: DialectSources, name: str, prefix: str, type_texts: Sequence[str], enum_texts: Sequence[str],
           enums: Mapping[str, str] | None = None) -> Listed:
    """Return the type and attribute lines and spellings of the defs `type_texts` hold for one dialect.

    `name` and `prefix` are the dialect's (ttcore's for the foreign
    sources); `enums` adds C++ enum cases another dialect's sources hold.
    """
    cases, cpp_cases = _enum_cases(enum_texts)
    dialect = Dialect({}, {}, {}, name, {**(enums or {}), **cpp_cases}, sources.extended)
    out = Listed({}, {}, {}, {}, cpp_cases)
    type_def = re.compile(rf'^def (?P<def>\w+)\s*:\s*{prefix}_Type<\s*"\w+"\s*,\s*"(?P<mnemonic>\w+)"')
    for text in type_texts:
        for record in records(text):
            found = type_def.match(record.text)
            if found is not None:
                _add_type(out, record, found.group("def"), f"!{name}.{found.group('mnemonic')}", dialect)
            else:
                _add_attribute(out, record, cases, name, prefix, dialect)
    return out


def _enum_cases(enum_texts: Sequence[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return ({enum def: case strings}, {C++ enum name: cases joined by " | "}) of the enum sources."""
    cases: dict[str, list[str]] = {}
    cpp: dict[str, str] = {}
    for enums_text in enum_texts:
        for record in records(enums_text):
            enum = _ENUM_DEF.match(record.text)
            if enum is not None:
                found = [match.group("text") or match.group("symbol") for symbol in split_top(enum.group("cases"))
                         for match in [_CASE.search(_case_text(enums_text, symbol))] if match]
                cases[enum.group("def")] = found
                cpp[enum.group("cpp")] = " | ".join(found)
    return cases, cpp


def _add_type(out: Listed, record: Record, def_name: str, spelled: str, dialect: Dialect) -> None:
    """Add one type def's line and its spelling in an op line to `out`."""
    params = parameters(record.text)
    fmt = let_value(record.text, "assemblyFormat")
    shown = _param_form(fmt, params) if fmt else ", ".join(param for param, _ in params)
    out.type_spellings[def_name] = f"{spelled}<...>" if params else spelled
    out.types[def_name] = _described(f"{spelled}<{shown}>" if params else spelled, params,
                                     let_value(record.text, "summary"), dialect)


def _add_attribute(out: Listed, record: Record, cases: Mapping[str, list[str]], name: str, prefix: str,
                   dialect: Dialect) -> None:
    """Add one attribute def's line and spelling to `out`: an enum, a struct (parameters), or a typed array."""
    enum = re.match(rf'^def (?P<def>\w+)\s*:\s*EnumAttr<\s*{prefix}_Dialect\s*,\s*(?P<enum>\w+)\s*,\s*'
                    rf'"(?P<mnemonic>\w+)"', record.text)
    struct = re.match(rf'^def (?P<def>\w+)\s*:\s*{prefix}_Attr<\s*"\w+"\s*,\s*"(?P<mnemonic>\w+)"', record.text)
    if struct is None and dialect.extended:
        struct = re.match(rf'^def (?P<def>\w+)\s*:\s*AttrDef<\s*{prefix}_Dialect\s*,\s*"(?P<mnemonic>\w+)"',
                          record.text)
    array = _ARRAY_ATTR.match(record.text)
    if enum is not None:
        listed_cases = " | ".join(cases.get(enum.group("enum"), ["..."]))
        fmt = let_value(record.text, "assemblyFormat")
        bare = dialect.extended and "<" not in fmt
        spelled = (f"#{name}<{enum.group('mnemonic')} {listed_cases}>" if bare
                   else f"#{name}.{enum.group('mnemonic')}<{listed_cases}>")
        out.attribute_spellings[enum.group("def")] = spelled
        out.attributes[enum.group("def")] = spelled
    elif struct is not None:
        mnemonic = (let_value(record.text, "mnemonic") or struct.group("mnemonic")) if dialect.extended else (
            struct.group("mnemonic"))
        spelled = f"#{name}.{mnemonic}"
        params = parameters(record.text)
        if dialect.extended and not params:
            out.attributes[struct.group("def")] = _described(spelled, (), let_value(record.text, "summary"), dialect)
            out.attribute_spellings[struct.group("def")] = spelled
            return
        form = _param_form(let_value(record.text, "assemblyFormat"), params)
        out.attributes[struct.group("def")] = _described(f"{spelled}<{form}>", params,
                                                         let_value(record.text, "summary"), dialect)
        out.attribute_spellings[struct.group("def")] = spelled + "<...>"
    elif array is not None and dialect.extended:
        out.attribute_spellings[array.group("def")] = f"[<{array.group('element')}>, ...]"


def _case_text(enums_text: str, symbol: str) -> str:
    """Return the text of enum case def `symbol` in the enums file, or the symbol itself when it is inline."""
    match = re.search(rf"^def {re.escape(symbol)}\b[^;]*;", enums_text, re.MULTILINE)
    return match.group(0) if match else symbol


def _arrays_spelled(spellings: dict[str, str]) -> None:
    """Replace each typed array's `[<element def>, ...]` with its element's spelling, where the element is known."""
    for def_name, spelled in list(spellings.items()):
        element = re.fullmatch(r"\[<(?P<element>\w+)>, \.\.\.\]", spelled)
        if element is not None and element.group("element") in spellings:
            spellings[def_name] = f"[{spellings[element.group('element')]}, ...]"


def constraints(texts: Sequence[str], extended: bool) -> dict[str, str]:
    """Return {def name: constraint} of every type constraint the texts define as AnyTypeOf (or Type, extended)."""
    found: dict[str, str] = {}
    for text in texts:
        for record in records(text):
            if record.kind != "def":
                continue
            value = " ".join(record.text.split(":", 1)[1].split()).rstrip(";").strip()
            if record.parent == "AnyTypeOf" and (extended or "\n" not in record.text.strip()):
                found[record.name] = value
            elif extended and record.parent == "Type":
                cpp = next((builtin for name, builtin in _CPP_TYPES.items() if f'"{name}"' in value), None)
                if cpp is not None:
                    found[record.name] = cpp
    return found


def reference_text(read: Mapping[str, bytes], pin: Mapping[str, str], sources: DialectSources = TTKERNEL) -> str:
    """Return the reference (the module docstring) from the sources `read` {tree path: bytes}."""
    texts = {path: read[path].decode("utf-8") for path in sources.sources}
    foreign_types = [texts[path] for path in sources.foreign if path.endswith("Types.td")]
    foreign = listed(sources, "ttcore", "TTCore", foreign_types,
                     [texts[path] for path in sources.foreign if path.endswith("Enums.td")])
    own = listed(sources, sources.name, sources.prefix, [texts[path] for path in sources.types],
                 [texts[path] for path in sources.enums], foreign.enums)
    attributes = {**foreign.attribute_spellings, **own.attribute_spellings}
    _arrays_spelled(attributes)
    dialect = Dialect({**foreign.type_spellings, **own.type_spellings},
                      constraints([texts[path] for path in (*sources.ops, *sources.bases, *sources.foreign)],
                                  sources.extended), attributes, sources.name, {**foreign.enums, **own.enums},
                      sources.extended)
    classes = {record.name: record for path in (*sources.ops, *sources.bases) for record in records(texts[path])
               if record.kind == "class"}
    ops = [(path, record) for path in sources.ops for record in records(texts[path])
           if _is_op(record, classes, sources.prefix)]
    cpp = {path: texts[path] for path in sources.cpp}
    names = ", ".join(Path(path).name for path in sources.sources)
    if sources.families:
        head = FAMILY_HEAD.format(name=sources.name, version=pin["VERSION"], commit=pin["COMMIT"],
                                  generator=GENERATOR, sources=names)
    else:
        head = HEAD.format(version=pin["VERSION"], commit=pin["COMMIT"], generator=GENERATOR, sources=names)
    out = [head, "Types:", *own.types.values(), "", "Attributes:", *own.attributes.values()]
    if sources.foreign:
        used = _foreign_used(ops, classes, dialect, foreign, sources)
        out += ["", "ttcore types and attributes the ops name (and those every module writes, such as a tensor's "
                "layout):", *used]
    out += _op_sections(ops, texts, classes, cpp, dialect, sources)
    return "\n".join(out) + "\n"


def _is_op(record: Record, classes: Mapping[str, Record], prefix: str) -> bool:
    """Return True for an op def: <prefix>_...Op deriving from a <prefix>_...Op class or a class that does."""
    if record.kind != "def" or not re.fullmatch(rf"{prefix}_\w+Op", record.name) or not _MNEMONIC.search(
            _header(record.text)):
        return False
    return any(re.fullmatch(rf"{prefix}_\w*Op", link.parent) for link in _chain(record, classes))


def _foreign_used(ops: Sequence[tuple[str, Record]], classes: Mapping[str, Record], dialect: Dialect,
                  foreign: Listed, sources: DialectSources) -> list[str]:
    """Return the lines of the foreign types and attributes the ops name (through constraints), and the kept ones."""
    words: set[str] = set(sources.foreign_kept)
    pending = [text for _, record in ops for text in (_resolved(_chain(record, classes), "arguments"),
                                                       _resolved(_chain(record, classes), "results"))]
    while pending:
        for word in re.findall(r"\w+", pending.pop()):
            if word not in words:
                words.add(word)
                pending += [dialect.constraints.get(word, ""), dialect.attributes.get(word, "")]
    for def_name, spelled in foreign.attribute_spellings.items():
        element = re.match(r"\[<(?P<element>\w+)>", spelled)
        if def_name in words and element is not None:
            words.add(element.group("element"))
    lines = [line for def_name, line in foreign.types.items() if def_name in words]
    return lines + [line for def_name, line in foreign.attributes.items() if def_name in words]


def _op_sections(ops: Sequence[tuple[str, Record]], texts: Mapping[str, str], classes: Mapping[str, Record],
                 cpp: Mapping[str, str], dialect: Dialect, sources: DialectSources) -> list[str]:
    """Return the op lines under their titles: sections in file order, or families gathered in order of first op."""
    out: list[str] = []
    current = None
    groups: dict[str, list[str]] = {}
    for path, record in ops:
        sections = [(match.start(), match.group("title")) for match in _SECTION.finditer(texts[path])]
        title = next((name for begin, name in reversed(sections) if begin < record.start), "Ops")
        line = op_line(op_def(record, classes), cpp, dialect)
        if sources.families:
            groups.setdefault(_family(title, record, classes, sources), []).append(line)
            continue
        if title != current:
            out += ["", f"{title}:"]
            current = title
        out.append(line)
    for title, lines in groups.items():
        out += ["", f"{title}:", *lines]
    return out


def _family(title: str, record: Record, classes: Mapping[str, Record], sources: DialectSources) -> str:
    """Return an op's family title: its section and, unless it derives from the base op class, that class."""
    if record.parent == f"{sources.prefix}_Op":
        return title
    parent = classes.get(record.parent)
    summary = first_sentence(let_value(parent.text, "summary")) if parent is not None else ""
    return f"{title}, class {record.parent}" + (f" ({summary})" if summary else "")


def build(tree: Path, pin: Mapping[str, str], sources: DialectSources = TTKERNEL) -> dict[str, dict[str, bytes]]:
    """Return {tree name: {file path: bytes}} for one dialect's reference pack, its manifest included."""
    context_tool().check_tree(tree, pin)
    read = {path: (tree / path).read_bytes() for path in sources.sources}
    text = reference_text(read, pin, sources)
    if not text.isascii():
        raise ValueError("the reference holds a character that is not ASCII; the prompts are plain ASCII")
    data = text.encode("ascii")
    document = {
        "generator": GENERATOR, "repo": pin["URL"], "commit": pin["COMMIT"], "language": sources.name,
        "entries": [{"key": sources.key, "sources": [{"path": path, "sha256": _digest(read[path])}
                                                     for path in sources.sources],
                     "sha256": _digest(data)}],
        "bytes": len(data),
    }
    head = (
        f"# Generated by {GENERATOR} from tt-mlir at {pin['COMMIT']} (toolchains/tt-mlir.pin); do not edit.\n"
        "# Only this manifest is tracked; the reference beside it renders tt-mlir's dialect and is gitignored\n"
        "# (OQ-018). No source is a tt-pairs-v0 item (Agent Rule 5). The verbatim_context stage carries it.\n"
    )
    manifest = (head + yaml.safe_dump(document, sort_keys=False, width=120)).encode("ascii")
    return {f"context/{sources.ref}": {f"{sources.key}.txt": data, "MANIFEST.yaml": manifest}}


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
