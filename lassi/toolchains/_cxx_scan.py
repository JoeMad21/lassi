"""Generic C and C++ text reading for static checks of host programs (task P4.12).

This module reads C and C++ source text without a compiler or a
preprocessor run. It names nothing from any device library:
lassi.toolchains._cxx_flow builds a data-flow analysis on it, and
lassi.toolchains.ttmetal_guard the CPU -> TT guard (bible Harness Contract).
Nothing here opens a file or starts a process, and every pass is an
iteration with an explicit stack, so no input can exhaust Python recursion.

Lexing (lex):

- CRLF and a lone CR become LF, then every backslash-newline is spliced
  out. Each token keeps the 1-based original line of its first character.
- `//` runs to the end of the line; `/* */` is a block comment. An
  unterminated block comment makes the text unreadable (Unreadable)
  wherever it starts, a directive line or a dropped region included.
- A string, char, or raw-string literal (prefixes u8, u, U, L; `R"d(...)d"`
  with a delimiter of up to 16 characters) is one LITERAL token. In code,
  a string or char literal that a newline or the end of the text
  interrupts, or a raw string with no terminator, makes the text
  unreadable. In a directive line or a dropped region such a quote only
  ends the line: the rest of the line is one OTHER token (an apostrophe in
  an `#error` line or in `#if 0` prose is not code) [a choice: that text
  is never compiled as code, and reading it so keeps more programs
  readable].
- pp-numbers keep `'` digit separators, signed exponents, and suffixes.
  Identifiers match `[A-Za-z_$][A-Za-z0-9_$]*`. Punctuators are C++20's, by
  longest match; digraphs are not mapped. Any other character outside a
  literal or a comment (a non-ASCII letter, `@`) is an OTHER token.
- A line whose first token is `#` (comments count as blanks) is a
  directive: its tokens go to `directives`, never to the code tokens.
  Regions under `#if 0` or `#if false` (that one token as the condition)
  are dropped up to the matching `#else`, `#elif`, or `#endif`, nesting
  counted over `#if`, `#ifdef`, and `#ifndef`; the `#else` or `#elif`
  branch is kept. No other condition is evaluated, so every arm of every
  other conditional is read as code. Directives inside a dropped region
  are not recorded.
- Every `#define N ...` outside a dropped region records the macro name N
  (`defines`); one whose body is a single identifier, `#define A B`, also
  records the alias A -> B (`aliases`; the first in a file wins). No macro
  is expanded here (lassi.toolchains._cxx_flow expands an alias one level
  in a callee's name). `#include "N"` and `#include <N>` outside a dropped
  region are recorded in order (include_names).
- Brackets are matched with a stack; a closer pairs with the nearest open
  bracket of its kind (brackets opened inside that pair and left open pair
  with the same closer), a closer with no opener of its kind is left
  unmatched, and a bracket still open at the end pairs with the end.

Includes (resolve_include): a quoted name is looked up beside the including
file first (posixpath.join of its directory and the name, normalized), then
at the build-directory root; an angle name only at the root. At each
candidate the model's files are checked before the harness files. A
candidate that leaves the build directory is skipped, and a name with no
candidate among the files resolves to nothing.

Structure (parse_program, walk_body):

- At file, namespace, or class scope a function definition is a name
  (possibly qualified, a destructor, an operator, or followed by a
  template argument list), a balanced `(...)`, then only qualifiers
  (const, volatile, noexcept, throw, or __attribute__ with an optional
  `(...)`, override, final, `&`, `&&`, `[[...]]`, try, and a `-> type` of
  at most TEMPLATE_SCAN_LIMIT tokens) and an optional constructor
  initializer list, then `{`. Every other `{` at those scopes is a
  namespace, a class, an enum, an `extern "C"` block, or an initializer.
- A file- or namespace-scope declarator followed by `(...)` and `;` is a
  function declaration, unless the parentheses hold an initializer: their
  first token is a number, a literal, `{`, a unary `-`, `+`, `!`, `~`, or
  `&`, one of true, false, nullptr, this, sizeof, or new, a file-scope
  variable of the same file recorded before it (the file's own scope is
  read first, in text order, then the namespace bodies in it), or a macro
  name of the host text. Then it declares a variable
  (`std::vector<float> g(1024);`) [a choice: without types the reader
  cannot tell `T g(n)` with a variable n from a prototype, and a prototype
  must not become a variable].
- Member functions are the ones defined in a class, struct, or union body,
  or named `C::f` where C is a class defined in the host text. A class's
  data members are the names its body declares outside function
  definitions and declarations (a constexpr declaration declares none).
  Its host bases are the classes of the host text that its base clause
  names; a class whose base clause holds any other name (a qualifier or a
  template argument included) is open.
- Parameters split at depth-0 commas with `<...>` counted as brackets; a
  parameter's name is its last non-keyword identifier before `=` or its end
  (the type's last name, for an unnamed parameter whose type ends in one).
  It is reference-declared when it holds `&` or `&&`, else pointer-declared
  when it holds `*` or `[` (outside template lists) or when its type's
  head, the last name before its own outside template lists, is a view
  (VIEW_TYPES below: a view points into its argument's storage), and
  value-declared otherwise; the pointer- and reference-declared ones are
  out-ports. A parameter whose last token is `...` makes the function
  variadic.
- Inside a body a `{` that starts a statement opens a block; every other
  `{` belongs to the statement it is in (an initializer, or a lambda body).
  An attribute `[[...]]` that starts a statement is passed over, so the
  statement it leads is walked as itself (the `{` after `[[likely]]` opens
  a block). A class, struct, or union defined inside a body is passed
  over, so its member functions are never read.
- A lambda introducer is a `[` that does not follow an identifier (a
  keyword other than this and operator does not count), a number, a
  literal, `)`, or `]`, is not `[[`, and is followed by an optional
  template list, parameters, specifiers, and a `{` body. A lambda body is
  a block of the enclosing function; its parameters are declared in it,
  and so is the name of each init capture (an identifier followed by `=`,
  `(`, or `{`, after an optional `&`, which makes it reference-declared,
  and an optional `...`), with its declaration at the introducer's `]`,
  so the capture's own initializer names the enclosing scope's variables.
  A value init capture has no initializer, so it starts with no value; a
  reference init capture has its initializer (from after the `=` to the
  capture's end, or inside its brackets), and lassi.toolchains._cxx_flow
  links it with that initializer's base unless the lambda is enclosed
  (Positions there).
- A declaration starts with a type sequence (identifiers, `::`, balanced
  `<...>`, `[[...]]`, fundamental type keywords, and specifier keywords)
  followed by declarators: optional `*`, `&`, `&&`, const, volatile,
  `__restrict`, `__restrict__`, and `restrict`, the name, optional array
  bounds, then one of `=`, `(`, `{`, `;`, `,`, or `:` (or the end of the
  range). A declarator's types are the names in its type sequence,
  template arguments included, and its head is the sequence's last name
  outside template lists (None when a fundamental type keyword such as
  auto ends it). A structured binding, a type sequence holding auto then
  `[n1, n2, ...]` (after an optional `&` or `&&`, which makes each name
  reference-declared; value-declared otherwise), declares each name as a
  binding name (Declarator.binding), with the binding's initializer as its
  own (in a range-for header, with none).
- A name is reference-declared when its declarator holds `&` or `&&`, else
  pointer-declared when it holds `*`, array-declared when it holds only
  `[` (an array owns its storage, so its initializer is a value and never
  an alias: `uint64_t s[1] = {a.count}` links s with nothing), and
  value-declared otherwise. A value-declared name is pointer-declared after
  all, so it aliases the storage it points into, when it has an
  initializer and its head is a view (VIEW_TYPES: span, mdspan,
  string_view, basic_string_view), or when its `=` initializer is a pointer
  or iterator producer (Code.producer: `&X...`; a static_cast,
  reinterpret_cast, const_cast, dynamic_cast, or C cast to a pointer type;
  a std::begin, std::end, std::cbegin, std::cend, std::data, or
  std::addressof call; or a name chain ending in a data(), begin(), end(),
  cbegin(), cend(), rbegin(), or rend() call with no arguments, alone or
  followed by `+` or `-`).
- Statements that start with return, if, else, while, for, do, switch,
  case, default, goto, break, continue, delete, throw, co_return,
  co_yield, or co_await are not declarations, and a lone `X =` is an
  assignment.
- A declaration in a for, if, while, or switch header (an init statement,
  a condition, or a range-for's element), and a catch parameter, belong to
  a block that starts at the keyword and ends with the statement the
  header controls, braced or not (for an if, with its else branch), and
  that statement is walked inside the block, so after it a same-named
  outer variable is seen again (a header that declares nothing has a
  block over itself alone). That end is found by passing if, for, while,
  and switch headers to the statement they control and a leading
  attribute `[[...]]` to the statement it leads, continuing an if whose
  header was passed at its else, and taking a braced block's `}`, a try
  statement's last handler's `}`, the `;` after a do statement's
  `while (...)`, or another statement's `;`; a loop's range ends there
  too. A condition (an if or switch header's last `;` part, a while
  header, or a for's second part) is a declaration only when every name
  it declares has an `=` or braced initializer, as C++ requires, so
  `x && y`, `x & y`, `x * y`, and `ok && f(v)` declare nothing; a for's
  third part is an expression. A range-for header, after an optional init
  statement (read as a for-init), declares its element when it holds one
  declarator with no initializer, and each name of a structured binding
  there (lassi.toolchains._cxx_flow gives each the range's values); any
  other range-for header declares nothing (a misparse). A catch parameter
  is named as a parameter is, with no value.
- A use of N at token r resolves (Body.resolve) to the first of: the latest
  declaration of N before r in the innermost block holding r, then in each
  enclosing block outward; a parameter named N; in a member function, the
  function's THIS symbol when N is `this`, or when no file-scope variable
  is named N and either N is a data member of the function's class or of
  one of its host bases, or one of those classes is open and N is no macro
  name of the host text; a file-scope variable named N in any host file
  (one symbol per name, the first by file rank); the function's own symbol
  for N (one per name; `this` outside a member function gets one too).

Expressions (Code methods; lassi.toolchains._cxx_flow reads every value
through mentions and base):

- mentions(E) are E's identifiers except keywords (`this` counts); a callee
  (an identifier followed by `(`, `{`, or a balanced `<...>` and then one
  of them: a call, a functional cast, or a braced temporary [`{` is a
  choice: a variable is never followed by a brace]); an identifier after
  `.`, `->`, `.*`, `->*`, or `::`, or before `::`; everything inside a
  subscript or an attribute `[...]`; the contents of sizeof, alignof,
  alignas, decltype, typeid, and noexcept (sizeof without parentheses skips
  its operand); a lambda's introducer, template list, parameters, and
  specifiers (its body is read); the template arguments of a callee or a
  cast [a choice: they are types]; and X in `X.m` or `X->m` when m is in
  METADATA. Code.is_mention also counts a name after `::` that the caller
  names as a file-scope variable, so `ns::x` and `::x` mention x;
  mentions() names none.
- base(E) strips, repeatedly, unary `&` and `*`, an opening parenthesis, a
  C cast, static_cast, reinterpret_cast, const_cast, dynamic_cast, and
  bit_cast or std::bit_cast with their template list, and std::addressof,
  std::data, std::begin, std::end, and std::cbegin calls. Then E has no
  base when it starts with `new` or a call, or with a qualified name
  (`ns::x`, `::x`) whose last name is no file-scope variable the caller
  names (base() names none); otherwise its base is that last name or E's
  first mention, with the METADATA rule ignored. Code.base_index with
  `metadata` set, as the flow analysis calls it, also gives no base when
  that mention is the object of a METADATA member (`a.dims`, `v.size()`).

Bounds. More than MAX_DEPTH (256) brackets open at once in a file's code
make the text unreadable. A template-list scan covers at most
TEMPLATE_SCAN_LIMIT (256) tokens and finds no list past that; a receiver of
more than TEMPLATE_SCAN_LIMIT steps back is not read (Code.receiver), and a
trailing return type of more than TEMPLATE_SCAN_LIMIT tokens and bracket
groups ends no function or lambda head (Code.trailing_type_end). A Budget
given to parse_program is spent by those scans, by each statement the body
walk reads, by each search for the end of a statement a header controls
(a step for each header or attribute passed, for each braced block, a
try statement's handlers included, and for each token from another
statement's start to its end; a search keeps the end it finds for each
statement start it reaches with no if waiting for its else, and a later
search that reaches such a start in that state stops there, so an
else-if chain is searched once), by each block a name lookup passes
(Body.lookup), and by the flow analysis, so a reading that needs more
steps stops with Unreadable; lexing, bracket matching, and the
file-structure pass outside those scans spend none. An else-if chain
whose headers each declare a name nests one block per branch, so the
lookups in it spend steps that grow with the square of its length. In
unbraced ifs nested one in another whose headers each declare a name,
the search for each one's end passes every if nested in it, since those
are reached with an if waiting for its else and no end is kept for them,
so those searches spend steps that grow with the square of the nesting
depth.

No value in this module is a measurement.
"""

from __future__ import annotations

import posixpath
import re
from bisect import bisect_left, bisect_right
from collections.abc import Container, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import accumulate

# Token kinds.
IDENT, NUMBER, LITERAL, PUNCT, OTHER = "IDENT", "NUMBER", "LITERAL", "PUNCT", "OTHER"

# More brackets than this open at once make the text unreadable (Bounds in the module docstring).
MAX_DEPTH = 256
# The most tokens a scan for a template argument list covers before it gives up.
TEMPLATE_SCAN_LIMIT = 256
# The longest raw-string delimiter C++ allows.
RAW_DELIMITER_LIMIT = 16

KEYWORDS = frozenset(
    """
    alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t char32_t class
    compl concept const consteval constexpr constinit const_cast continue co_await co_return co_yield decltype
    default delete do double dynamic_cast else enum explicit export extern false float for friend goto if inline int
    long mutable namespace new noexcept not not_eq nullptr operator or or_eq private protected public register
    reinterpret_cast requires return short signed sizeof static static_assert static_cast struct switch template
    this thread_local throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t
    while xor xor_eq
    """.split()
)
# The keywords a type sequence may hold: fundamental types, then qualifiers and specifiers (Structure above).
TYPE_KEYWORDS = frozenset(
    "auto bool char char8_t char16_t char32_t wchar_t short int long signed unsigned float double void".split()
)
DECL_SPECIFIERS = frozenset(
    "const volatile static constexpr constinit inline extern mutable register thread_local typename struct class "
    "union enum".split()
)
# Statements that start with one of these are never declarations.
NOT_DECLARATIONS = frozenset(
    "return if else while for do switch case default goto break continue delete throw co_return co_yield "
    "co_await".split()
)
CAST_KEYWORDS = frozenset({"static_cast", "reinterpret_cast", "const_cast", "dynamic_cast"})
UNEVALUATED = frozenset({"sizeof", "alignof", "alignas", "decltype", "typeid", "noexcept"})
# The member names whose object an expression does not mention: they describe an array, never hold its values.
METADATA = frozenset(
    {"name", "dtype", "rank", "dims", "count", "nbytes", "size", "empty", "capacity", "length", "shape"}
)
# The std calls base() looks through: each returns its argument's storage.
STD_BASE_HELPERS = frozenset({"addressof", "data", "begin", "end", "cbegin"})
# The std calls and the no-argument methods whose result points into the storage of their argument or object.
POINTER_HELPERS = STD_BASE_HELPERS | {"cend"}
POINTER_METHODS = frozenset({"data", "begin", "end", "cbegin", "cend", "rbegin", "rend"})
# The view types: a value of one points into the storage it was made from.
VIEW_TYPES = frozenset({"span", "mdspan", "string_view", "basic_string_view"})
# Declarator qualifiers that are not keywords (`float* __restrict out`).
RESTRICT_WORDS = frozenset({"__restrict", "__restrict__", "restrict"})
CLASS_KEYS = frozenset({"class", "struct", "union"})

OPENERS = frozenset({"(", "[", "{"})
CLOSERS = frozenset({")", "]", "}"})
PAIRS = {")": "(", "]": "[", "}": "{"}
ASSIGN_OPS = frozenset({"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="})
MEMBER_ACCESS = frozenset({".", "->", "::", ".*", "->*"})
# A template argument scan stops at these: they cannot sit inside a template argument list at its own level.
TEMPLATE_STOPS = frozenset(
    {";", "{", "}", ")", "]", "&&", "||", "?", "==", "!=", "<=", ">=", "<=>"} | ASSIGN_OPS
)
# What ends the left side of an assignment, scanning back from its operator at one nesting level.
SEGMENT_STOPS = frozenset({",", ";", "?", ":", "&&", "||"})
SEGMENT_KEYWORDS = frozenset({"return", "co_return", "throw", "case", "else", "do"})
# Statements the body walk passes over to their end.
SKIPPED_STATEMENTS = frozenset(
    {"goto", "break", "continue", "using", "typedef", "static_assert", "asm", "namespace", "template", "friend"}
)
# Words a C cast's parenthesized type may hold besides names, `::`, `*`, `&`, and template lists.
CAST_TYPE_WORDS = TYPE_KEYWORDS | frozenset({"const", "volatile", "struct", "class", "union", "enum", "typename"})

# Declared-name kinds (Structure in the module docstring).
VALUE, POINTER, REFERENCE, ARRAY = "value", "pointer", "reference", "array"
# Symbol roles.
LOCAL, PARAM, THIS, RET, GLOBAL, FREE = "local", "param", "this", "ret", "global", "free"

_PUNCTUATORS = (
    ">>=", "<<=", "<=>", "->*", "...",
    "::", "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", ".*", "##",
    "{", "}", "[", "]", "(", ")", "<", ">", ";", ":", ",", ".", "?", "~", "!",
    "+", "-", "*", "/", "%", "^", "&", "|", "=", "#",
)
_PUNCT = re.compile("|".join(re.escape(item) for item in _PUNCTUATORS))
_PREFIX = r"(?:u8|u|U|L)?"
_RAW_OPEN = re.compile(_PREFIX + r'R"([^ ()\\\t\v\f\n"]{0,' + str(RAW_DELIMITER_LIMIT) + r"})\(")
_STRING = re.compile(_PREFIX + r'"(?:[^"\\\n]|\\[^\n])*"')
_CHAR = re.compile(_PREFIX + r"'(?:[^'\\\n]|\\[^\n])*'")
_QUOTE_START = re.compile(_PREFIX + "[\"']")
_NUMBER = re.compile(r"\.?[0-9](?:[eEpP][+-]|'[0-9A-Za-z_]|[0-9A-Za-z_.])*")
_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_SPACE = re.compile(r"[ \t\f\v]+")
_INCLUDE = re.compile(r'#\s*include\s*(?:"([^"\n]*)"|<([^>\n]*)>)')


class Unreadable(Exception):
    """Text the reader cannot read; `reason` says why ("an unterminated comment in <path>", ...), naming the file."""

    def __init__(self, reason: str) -> None:
        """Keep the reason."""
        super().__init__(reason)
        self.reason = reason


class Budget:
    """A count of reader steps left for one reading, shared by every pass that can repeat work.

    spend() raises Unreadable("the reader exceeded its budget of <limit>
    steps") once more than `limit` steps are spent, so a reading always
    ends. A step is one unit of scanning or analysis work (a token scanned,
    a definition fired, a summary flow applied); the counts are fixed by the
    code, so the same text always spends the same steps.
    """

    def __init__(self, limit: int) -> None:
        """Start with `limit` steps."""
        self.limit = limit
        self.left = limit

    def spend(self, steps: int) -> None:
        """Spend `steps`; raise Unreadable when the budget is used up."""
        self.left -= steps
        if self.left < 0:
            raise Unreadable(f"the reader exceeded its budget of {self.limit} steps")


@dataclass(frozen=True, slots=True)
class Token:
    """One token: its kind, text, 1-based original line, and offset in the spliced text."""

    kind: str
    text: str
    line: int
    pos: int


@dataclass(frozen=True)
class Directive:
    """One directive line outside a dropped region: its tokens (the first is "#"), line, and spliced text."""

    tokens: tuple[Token, ...]
    line: int
    text: str


@dataclass(frozen=True)
class Lexed:
    """One file read by lex: code tokens, their bracket matches, directives, aliases, and include names.

    `match[i]` is, for an opener, the index of its closer (len(tokens) when
    it has none); for a closer, the index of its opener (-1 when it has
    none); -1 for any other token.
    """

    path: str
    tokens: tuple[Token, ...]
    match: tuple[int, ...]
    directives: tuple[Directive, ...]
    aliases: Mapping[str, str]
    includes: tuple[tuple[str, bool], ...]
    defines: frozenset[str] = frozenset()


def lex(text: str, path: str) -> Lexed:
    """Return `text` lexed as the file `path` (see the module docstring); Unreadable when it cannot be read."""
    return _Lexer(text, path).run()


def include_names(lexed: Lexed) -> list[tuple[str, bool]]:
    """Return (name, quoted) for each `#include` outside a dropped region, in order."""
    return list(lexed.includes)


def resolve_include(
    including: str, name: str, quoted: bool, files: Mapping[str, str], harness: Mapping[str, str]
) -> str | None:
    """Return the build-directory path an include reads, or None (see the module docstring, Includes)."""
    candidates = [posixpath.normpath(name)] if name else []
    if quoted and name:
        candidates.insert(0, posixpath.normpath(posixpath.join(posixpath.dirname(including), name)))
    for candidate in candidates:
        if candidate in (".", "..") or candidate.startswith(("../", "/")):
            continue
        if candidate in files or candidate in harness:
            return candidate
    return None


@dataclass
class _Condition:
    """One open #if, #ifdef, or #ifndef: whether text was dropped before it, and whether it started a drop."""

    outer: bool
    drops: bool


class _Lexer:
    """The state of one lex() call."""

    def __init__(self, text: str, path: str) -> None:
        """Normalize line ends, splice lines, and index the newlines and splices for line numbers."""
        self.path = path
        parts = text.replace("\r\n", "\n").replace("\r", "\n").split("\\\n")
        self.text = "".join(parts)
        self.splices = list(accumulate(len(part) for part in parts[:-1]))
        self.newlines = [match.start() for match in re.finditer("\n", self.text)]
        self.tokens: list[Token] = []
        self.directives: list[Directive] = []
        self.aliases: dict[str, str] = {}
        self.defines: set[str] = set()
        self.includes: list[tuple[str, bool]] = []
        self.conditions: list[_Condition] = []
        self.dropping = False

    def line(self, pos: int) -> int:
        """Return the original 1-based line of the spliced offset `pos`."""
        return 1 + bisect_left(self.newlines, pos) + bisect_right(self.splices, pos)

    def run(self) -> Lexed:
        """Return the Lexed file; Unreadable for an unterminated comment or literal, or brackets nested too deep."""
        text, end = self.text, len(self.text)
        pos, line_start = 0, True
        while pos < end:
            if text[pos] == "\n":
                pos, line_start = pos + 1, True
                continue
            skipped = self._blank(pos)
            if skipped > pos:
                pos = skipped
                continue
            if text[pos] == "#" and line_start and not text.startswith("##", pos):
                pos = self._directive(pos)
                continue
            token, pos = self._token(pos, lenient=self.dropping)
            line_start = False
            if not self.dropping:
                self.tokens.append(token)
        tokens = tuple(self.tokens)
        match = tuple(_match(tokens, self.path))
        return Lexed(
            self.path, tokens, match, tuple(self.directives), dict(self.aliases), tuple(self.includes),
            frozenset(self.defines),
        )

    def _blank(self, pos: int) -> int:
        """Return the offset after the spaces or the comment at `pos` (pos when there is none)."""
        text = self.text
        found = _SPACE.match(text, pos)
        if found:
            return found.end()
        if text.startswith("//", pos):
            stop = text.find("\n", pos)
            return len(text) if stop < 0 else stop
        if text.startswith("/*", pos):
            stop = text.find("*/", pos + 2)
            if stop < 0:
                raise Unreadable(f"an unterminated comment in {self.path}")
            return stop + 2
        return pos

    def _token(self, pos: int, *, lenient: bool) -> tuple[Token, int]:
        """Return the token at `pos` and the offset after it."""
        text = self.text
        if text[pos] in "uULR\"'":
            literal = self._literal(pos, lenient)
            if literal is not None:
                return literal
        for kind, pattern in ((IDENT, _IDENT), (NUMBER, _NUMBER), (PUNCT, _PUNCT)):
            found = pattern.match(text, pos)
            if found:
                return Token(kind, found.group(), self.line(pos), pos), found.end()
        return Token(OTHER, text[pos], self.line(pos), pos), pos + 1

    def _literal(self, pos: int, lenient: bool) -> tuple[Token, int] | None:
        """Return the literal at `pos` and the offset after it, or None when `pos` starts no literal."""
        text = self.text
        raw = _RAW_OPEN.match(text, pos)
        if raw:
            stop = text.find(")" + raw.group(1) + '"', raw.end())
            if stop >= 0:
                end = stop + len(raw.group(1)) + 2
                return Token(LITERAL, text[pos:end], self.line(pos), pos), end
            return self._unterminated(pos, lenient)
        start = _QUOTE_START.match(text, pos)
        if start is None:
            return None
        full = (_STRING if text[start.end() - 1] == '"' else _CHAR).match(text, pos)
        if full:
            return Token(LITERAL, full.group(), self.line(pos), pos), full.end()
        return self._unterminated(pos, lenient)

    def _unterminated(self, pos: int, lenient: bool) -> tuple[Token, int]:
        """Return the rest of the line as one OTHER token when lenient; else raise Unreadable."""
        if not lenient:
            raise Unreadable(f"an unterminated literal in {self.path}")
        stop = self.text.find("\n", pos)
        stop = len(self.text) if stop < 0 else stop
        return Token(OTHER, self.text[pos:stop], self.line(pos), pos), stop

    def _directive(self, pos: int) -> int:
        """Read the directive line at `pos`, apply it, and return the offset of its line's end."""
        text, end = self.text, len(self.text)
        tokens: list[Token] = []
        here = pos
        while here < end and text[here] != "\n":
            skipped = self._blank(here)
            if skipped > here:
                here = skipped
                continue
            token, here = self._token(here, lenient=True)
            tokens.append(token)
        dropped = self.dropping
        self._condition(tokens)
        if not dropped:
            self.directives.append(Directive(tuple(tokens), self.line(pos), text[pos:here]))
            self._record(tokens, text[pos:here])
        return here

    def _condition(self, tokens: Sequence[Token]) -> None:
        """Apply a conditional directive to the drop state (see the module docstring)."""
        word = tokens[1].text if len(tokens) > 1 else ""
        if word in ("if", "ifdef", "ifndef"):
            zero = word == "if" and len(tokens) == 3 and tokens[2].text in ("0", "false")
            starts = zero and not self.dropping
            self.conditions.append(_Condition(outer=self.dropping, drops=starts))
            self.dropping = self.dropping or starts
        elif word in ("else", "elif") and self.conditions and self.conditions[-1].drops:
            self.dropping = self.conditions[-1].outer
            self.conditions[-1].drops = False
        elif word == "endif" and self.conditions:
            self.dropping = self.conditions.pop().outer

    def _record(self, tokens: Sequence[Token], line_text: str) -> None:
        """Record a macro name, an alias, or an include name from a directive outside a dropped region."""
        word = tokens[1].text if len(tokens) > 1 else ""
        if word == "define" and len(tokens) > 2 and tokens[2].kind == IDENT:
            self.defines.add(tokens[2].text)
        if word == "define" and len(tokens) == 4 and tokens[2].kind == IDENT and tokens[3].kind == IDENT:
            self.aliases.setdefault(tokens[2].text, tokens[3].text)
        elif word == "include":
            found = _INCLUDE.match(line_text)
            if found:
                quoted = found.group(1) is not None
                self.includes.append((found.group(1) if quoted else found.group(2), quoted))


def _match(tokens: Sequence[Token], path: str) -> list[int]:
    """Return the bracket matches of `tokens` (see Lexed); Unreadable when more than MAX_DEPTH are open at once."""
    end = len(tokens)
    match = [-1] * end
    stack: list[int] = []
    for index, token in enumerate(tokens):
        if token.kind != PUNCT:
            continue
        if token.text in OPENERS:
            stack.append(index)
            if len(stack) > MAX_DEPTH:
                raise Unreadable(f"brackets nested deeper than {MAX_DEPTH} in {path}")
        elif token.text in CLOSERS:
            want = PAIRS[token.text]
            depth = len(stack) - 1
            while depth >= 0 and tokens[stack[depth]].text != want:
                depth -= 1
            if depth < 0:
                continue
            while len(stack) > depth + 1:
                match[stack.pop()] = index
            opener = stack.pop()
            match[opener], match[index] = index, opener
    for opener in stack:
        match[opener] = end
    return match


@dataclass(frozen=True)
class Lambda:
    """A lambda expression: its introducer, parameter list (None when it has none), and body, as bracket indices."""

    intro: tuple[int, int]
    params: tuple[int, int] | None
    body: tuple[int, int]


@dataclass(frozen=True)
class Callee:
    """What precedes a call's `(`: the name, its index, qualifiers, and a member call's receiver range.

    `qualifiers` holds the names before the last `::`, outermost first, with
    "" for a leading global `::`. `receiver` is the token range of the
    object of a member call (`X.f(...)` or `X->f(...)`), or None.
    """

    name: str
    index: int
    qualifiers: tuple[str, ...]
    member: bool
    receiver: tuple[int, int] | None
    open: int
    close: int


class Code:
    """One file's code tokens with their bracket matches, and the scanning helpers built on them.

    `budget`, when given, is spent by every scan that can repeat over the
    same tokens (template lists, receivers, lambda heads), so the scans of
    one reading are bounded together (Budget).
    """

    def __init__(self, tokens: Sequence[Token], match: Sequence[int], budget: Budget | None = None) -> None:
        """Keep the tokens and matches; the separator index and the lambda table are built on first use."""
        self.tokens = tokens
        self.match = match
        self.n = len(tokens)
        self.budget = budget
        self._separators: list[int] | None = None
        self._lambdas: dict[int, Lambda | None] = {}

    def spend(self, steps: int) -> None:
        """Spend `steps` of the reading's budget, when there is one."""
        if self.budget is not None:
            self.budget.spend(steps)

    def text(self, i: int) -> str:
        """Return the text of token `i`, or "" outside the tokens."""
        return self.tokens[i].text if 0 <= i < self.n else ""

    def kind(self, i: int) -> str:
        """Return the kind of token `i`, or "" outside the tokens."""
        return self.tokens[i].kind if 0 <= i < self.n else ""

    def is_name(self, i: int) -> bool:
        """Return True for an identifier that is not a keyword."""
        return self.kind(i) == IDENT and self.tokens[i].text not in KEYWORDS

    def is_punct(self, i: int, texts: frozenset[str] | tuple[str, ...]) -> bool:
        """Return True when token `i` is a punctuator among `texts`."""
        return self.kind(i) == PUNCT and self.tokens[i].text in texts

    def close(self, i: int) -> int:
        """Return the closer of the opener at `i` (n when it has none), or `i` for any other token."""
        return self.match[i] if self.is_punct(i, OPENERS) else i

    def template_end(self, i: int) -> int:
        """Return the index of the `>` (or `>>`) that closes the template list opening at `i`, or -1."""
        depth, j = 0, i
        limit = min(self.n, i + TEMPLATE_SCAN_LIMIT)
        found = -1
        while j < limit:
            t = self.tokens[j].text
            if self.tokens[j].kind == PUNCT:
                if t == "<":
                    depth += 1
                elif t in (">", ">>"):
                    depth -= 1 if t == ">" else 2
                    if depth <= 0:
                        found = j
                        break
                elif t in ("(", "["):
                    j = self.match[j]
                    if j >= self.n:
                        break
                elif t in TEMPLATE_STOPS:
                    break
            j += 1
        self.spend(min(j, limit) - i + 1)
        return found

    def template_open_back(self, j: int) -> int:
        """Return the index of the `<` that opens the template list closing at `j` (`>` or `>>`), or -1."""
        depth, k = 0, j
        limit = max(-1, j - TEMPLATE_SCAN_LIMIT)
        found = -1
        while k > limit:
            t = self.tokens[k].text
            if self.tokens[k].kind == PUNCT:
                if t in (">", ">>"):
                    depth += 1 if t == ">" else 2
                elif t == "<":
                    depth -= 1
                    if depth <= 0:
                        found = k if depth == 0 else -1
                        break
                elif t in (")", "]"):
                    k = self.match[k]
                    if k < 0:
                        break
                elif t in TEMPLATE_STOPS or t in ("(", "["):
                    break
            k -= 1
        self.spend(j - max(k, limit) + 1)
        return found

    def is_callee(self, i: int) -> bool:
        """Return True when identifier `i` is followed by `(` or `{`, directly or after a balanced `<...>`."""
        following = self.text(i + 1)
        if following in ("(", "{"):
            return True
        if following == "<":
            end = self.template_end(i + 1)
            return end >= 0 and self.text(end + 1) in ("(", "{")
        return False

    def lambda_at(self, i: int) -> Lambda | None:
        """Return the lambda whose introducer `[` is token `i`, or None (see the module docstring; memoized)."""
        if i in self._lambdas:
            return self._lambdas[i]
        found = self._find_lambda(i)
        self._lambdas[i] = found
        return found

    def _find_lambda(self, i: int) -> Lambda | None:
        """Return the lambda whose introducer `[` is token `i`, or None."""
        if not self.is_punct(i, ("[",)) or self.text(i + 1) == "[" or self.match[i] >= self.n:
            return None
        before = self.tokens[i - 1] if i > 0 else None
        if before is not None:
            if before.kind in (NUMBER, LITERAL) or before.text in (")", "]"):
                return None
            if before.kind == IDENT and (before.text not in KEYWORDS or before.text in ("this", "operator")):
                return None
        intro_end = self.match[i]
        k = intro_end + 1
        if self.text(k) == "<":
            end = self.template_end(k)
            if end < 0:
                return None
            k = end + 1
        params = None
        if self.text(k) == "(":
            if self.match[k] >= self.n:
                return None
            params, k = (k, self.match[k]), self.match[k] + 1
        k = self._lambda_specifiers(k)
        if not self.is_punct(k, ("{",)):
            return None
        return Lambda((i, intro_end), params, (k, self.match[k]))

    def _lambda_specifiers(self, k: int) -> int:
        """Return the index after a lambda's specifiers and trailing return type, from `k`.

        A trailing return type longer than TEMPLATE_SCAN_LIMIT tokens and
        bracket groups is not read: the index returned is then -1, which is
        no `{`, so no lambda is found there.
        """
        while k < self.n:
            t = self.text(k)
            if t in ("mutable", "constexpr", "consteval", "static"):
                k += 1
            elif t in ("noexcept", "throw"):
                k = self.close(k + 1) + 1 if self.text(k + 1) == "(" else k + 1
            elif t == "[" and self.text(k + 1) == "[":
                k = self.close(k) + 1
            elif t == "->":
                return self.trailing_type_end(k + 1, self.n, ("{", ";", ")", "]", "}"), OPENERS)
            else:
                return k
        return k

    def trailing_type_end(
        self, k: int, hi: int, stops: tuple[str, ...], groups: frozenset[str] | tuple[str, ...]
    ) -> int:
        """Return the index of the first `stops` token from `k` (skipping `groups` brackets), or -1 past the limit.

        The scan covers at most TEMPLATE_SCAN_LIMIT tokens and bracket
        groups, so a trailing return type never costs more than that.
        """
        steps = 0
        while k < hi and self.text(k) not in stops:
            steps += 1
            if steps > TEMPLATE_SCAN_LIMIT:
                self.spend(steps)
                return -1
            k = self.close(k) + 1 if self.is_punct(k, groups) else k + 1
        self.spend(steps + 1)
        return k

    def skip_unevaluated(self, i: int) -> int:
        """Return the index after sizeof, alignof, alignas, decltype, typeid, or noexcept at `i` and its operand."""
        j = i + 1
        if self.text(j) == "...":
            j += 1
        if self.text(j) == "(":
            return self.close(j) + 1
        if self.text(i) == "noexcept":
            return i + 1
        while j < self.n:
            if self.kind(j) == IDENT or self.is_punct(j, (".", "->", "::", "*", "&")):
                j += 1
            elif self.is_punct(j, ("[",)):
                j = self.close(j) + 1
            else:
                break
        return max(j, i + 1)

    def skip_excluded(self, i: int) -> int | None:
        """Return the index after the excluded region token `i` starts, or None when it starts none.

        The regions are mentions' exclusions: a subscript or attribute, an
        unevaluated operand, a cast's or callee's template list, and a
        lambda's introducer, parameters, and specifiers (the index returned
        for a lambda is its body's `{`, so the body is read).
        """
        token = self.tokens[i]
        if token.kind == PUNCT and token.text == "[":
            found = self.lambda_at(i)
            return found.body[0] if found is not None else self.match[i] + 1
        if token.kind != IDENT:
            return None
        if token.text in UNEVALUATED:
            return self.skip_unevaluated(i)
        if self.text(i + 1) == "<" and (token.text in CAST_KEYWORDS or self.is_name(i) and self.is_callee(i)):
            end = self.template_end(i + 1)
            return end + 1 if end > i else None
        return None

    def is_mention(self, i: int, metadata: bool = True, qualified: Container[str] = ()) -> bool:
        """Return True when token `i` is an identifier mentions(E) counts (see the module docstring).

        A name in `qualified` also counts when it ends a qualified name
        (`ns::x` or `::x`); the caller resolves it to its file-scope symbol.
        """
        token = self.tokens[i]
        if token.kind != IDENT:
            return False
        this = token.text == "this"
        if token.text in KEYWORDS and not this:
            return False
        if self.text(i - 1) in MEMBER_ACCESS and self.kind(i - 1) == PUNCT:
            if self.text(i - 1) != "::" or token.text not in qualified:
                return False
        following = self.text(i + 1)
        if following == "::" or not this and self.is_callee(i):
            return False
        return not (metadata and self.is_metadata_object(i))

    def is_metadata_object(self, i: int) -> bool:
        """Return True when token `i` is the object X of `X.m` or `X->m` with m in METADATA."""
        return self.text(i + 1) in (".", "->") and self.text(i + 2) in METADATA

    def iter_mentions(self, lo: int, hi: int, metadata: bool = True, qualified: Container[str] = ()) -> Iterator[int]:
        """Yield the index of each mention in the token range [lo, hi), in order (is_mention's arguments)."""
        i = lo
        while i < hi:
            skip = self.skip_excluded(i)
            if skip is not None:
                i = max(skip, i + 1)
                continue
            if self.is_mention(i, metadata, qualified):
                yield i
            i += 1

    def base_index(self, lo: int, hi: int, metadata: bool = False, qualified: Container[str] = ()) -> int | None:
        """Return the index of the base token of the expression [lo, hi), or None (see the module docstring).

        With `metadata`, an expression whose base is the object of a
        METADATA member (`a.dims`, `v.size()`) has no base. A qualified name
        whose last name is in `qualified` is a base (`ns::x[i]` has base x).
        """
        start = self._strip(lo, hi)
        if start is None or start >= hi:
            return None
        if self.text(start) == "::" or self.is_name(start) and self.text(start + 1) == "::":
            found = self._qualified_last(start)
            if found is None or self.text(found) not in qualified or self.is_callee(found):
                return None
        elif self.text(start) == "new" or self.is_name(start) and self.is_callee(start):
            return None
        else:
            found = next(self.iter_mentions(start, hi, metadata=False), None)
        if found is not None and metadata and self.is_metadata_object(found):
            return None
        return found

    def _qualified_last(self, start: int) -> int | None:
        """Return the index of the last name of the qualified name starting at `start`, or None."""
        j = start + 1 if self.text(start) == "::" else start
        while self.is_name(j) and self.text(j + 1) == "::":
            j += 2
        return j if self.is_name(j) else None

    def is_operand(self, lo: int, hi: int) -> bool:
        """Return True when [lo, hi) is, after base()'s stripping, one name with only accesses, subscripts, calls.

        `c`, `c.data()`, `&c[0]`, `*p`, `this->m`, and `bufs[i].get()` are
        operands; `a != nullptr`, `true`, `n * 4`, and `1` are not.
        """
        j = self._strip(lo, hi)
        if j is None or j >= hi:
            return False
        if self.text(j) == "::" or self.is_name(j) and self.text(j + 1) == "::":
            last = self._qualified_last(j)
            if last is None:
                return False
            j = last
        elif not (self.is_name(j) or self.text(j) == "this"):
            return False
        j += 1
        while j < hi:
            t = self.text(j)
            if t in (".", "->") and self.is_name(j + 1):
                j += 2
            elif t in ("[", "("):
                j = self.close(j) + 1
            elif t == ")":
                j += 1
            else:
                return False
        return True

    def producer(self, lo: int, hi: int) -> bool:
        """Return True when the expression [lo, hi) yields a pointer or iterator into its base's storage.

        The producers are `&X...`, a static_cast, reinterpret_cast,
        const_cast, or C cast to a pointer type, a std::begin, std::end,
        std::cbegin, std::cend, std::data, or std::addressof call, and a
        name chain that ends with a call of data(), begin(), end(), cbegin(),
        cend(), rbegin(), or rend() with no arguments, alone or followed by
        `+` or `-` (pointer arithmetic).
        """
        while lo < hi and self.text(lo) == "(" and self.match[lo] == hi - 1:
            lo, hi = lo + 1, hi - 1
        if lo >= hi:
            return False
        t = self.text(lo)
        if t == "&":
            return True
        if t in CAST_KEYWORDS and self.text(lo + 1) == "<":
            end = self.template_end(lo + 1)
            return end > lo and any(self.text(k) == "*" for k in range(lo + 2, end))
        if t == "(":
            end = self._c_cast_end(lo, hi)
            return end is not None and any(self.text(k) == "*" for k in range(lo + 1, end - 1))
        if t == "std" and self.text(lo + 1) == "::" and self.text(lo + 3) == "(":
            return self.text(lo + 2) in POINTER_HELPERS
        return self._pointer_chain(lo, hi)

    def _pointer_chain(self, lo: int, hi: int) -> bool:
        """Return True when [lo, hi) starts with a name chain ending in a POINTER_METHODS call, then ends or `+`/`-`."""
        if not (self.is_name(lo) or self.text(lo) == "this"):
            return False
        j, last = lo + 1, None
        while j < hi:
            t = self.text(j)
            if t in (".", "->") and self.is_name(j + 1):
                last, j = j + 1, j + 2
            elif t == "[":
                last, j = None, self.close(j) + 1
            elif t == "(" and last is not None:
                if self.text(last) in POINTER_METHODS and self.close(j) == j + 1:
                    j, last = j + 2, -1
                    if j >= hi or self.text(j) in ("+", "-"):
                        return True
                    continue
                last, j = None, self.close(j) + 1
            else:
                break
        return False

    def _strip(self, lo: int, hi: int) -> int | None:
        """Return the index after the casts, `&`, `*`, parentheses, and std helpers that lead [lo, hi)."""
        s = lo
        while s < hi:
            t = self.tokens[s].text
            if t in ("&", "*"):
                s += 1
                continue
            if t == "(":
                end = self._c_cast_end(s, hi)
                s = end if end is not None else s + 1
                continue
            helper = self.text(s + 2) if t == "std" and self.text(s + 1) == "::" else ""
            if t in CAST_KEYWORDS or t == "bit_cast" or helper == "bit_cast":
                at = s + 3 if helper else s + 1
                end = self.template_end(at) if self.text(at) == "<" else -1
                if end < 0:
                    return s
                s = end + 1
            elif helper in STD_BASE_HELPERS and self.text(s + 3) == "(":
                s += 4
            else:
                return s
        return s

    def _c_cast_end(self, s: int, hi: int) -> int | None:
        """Return the index after a C cast `(type)` at `s` that is followed by its operand, or None."""
        m = self.match[s]
        if m + 1 >= hi or m <= s + 1 or not (self.kind(s + 1) == IDENT or self.text(s + 1) == "::"):
            return None
        typey, j = False, s + 1
        while j < m:
            token = self.tokens[j]
            if token.kind == IDENT:
                if token.text in KEYWORDS:
                    if token.text not in CAST_TYPE_WORDS:
                        return None
                    typey = True
                j += 1
            elif token.text in ("*", "&", "&&", "::"):
                typey = typey or token.text != "::"
                j += 1
            elif token.text == "<":
                end = self.template_end(j)
                if end < 0 or end >= m:
                    return None
                typey, j = True, end + 1
            else:
                return None
        following = self.tokens[m + 1]
        if following.kind in (NUMBER, LITERAL) or following.kind == IDENT and following.text not in ("and", "or"):
            return m + 1
        allowed = ("(", "!", "~", "::", "*", "&", "-", "+", "++", "--") if typey else ("(", "!", "~", "::")
        return m + 1 if following.text in allowed else None

    def split_commas(self, lo: int, hi: int, angles: bool = False) -> list[tuple[int, int]]:
        """Return the ranges of [lo, hi) between depth-0 commas ([] for an empty range); `angles` counts `<...>`."""
        if lo >= hi:
            return []
        parts: list[tuple[int, int]] = []
        start, i = lo, lo
        while i < hi:
            t = self.tokens[i].text
            if self.tokens[i].kind == PUNCT:
                if t in OPENERS:
                    i = self.match[i] + 1
                    continue
                if angles and t == "<":
                    end = self.template_end(i)
                    if lo <= end < hi:
                        i = end + 1
                        continue
                if t == ",":
                    parts.append((start, i))
                    start = i + 1
            i += 1
        parts.append((start, hi))
        return parts

    def split_at(self, lo: int, hi: int, separator: str) -> list[tuple[int, int]]:
        """Return the ranges of [lo, hi) between depth-0 `separator` tokens."""
        parts: list[tuple[int, int]] = []
        start, i = lo, lo
        while i < hi:
            if self.is_punct(i, OPENERS):
                i = self.match[i] + 1
                continue
            if self.is_punct(i, (separator,)):
                parts.append((start, i))
                start = i + 1
            i += 1
        parts.append((start, hi))
        return parts

    def statement_end(self, i: int, hi: int) -> int:
        """Return the index of the `;` (or stray closer) that ends the statement starting at `i`, or `hi`."""
        j = i
        while j < hi:
            if self.kind(j) == PUNCT:
                t = self.tokens[j].text
                if t == ";" or t in CLOSERS:
                    return j
                if t in OPENERS:
                    j = self.match[j] + 1
                    continue
            j += 1
        return hi

    def next_separator(self, i: int) -> int:
        """Return the index of the next `,`, `;`, or closer at token `i`'s nesting level (n when there is none)."""
        if self._separators is None:
            self._separators = self._build_separators()
        return self._separators[i] if 0 <= i < self.n else self.n

    def _build_separators(self) -> list[int]:
        """Return next_separator for every token, in one right-to-left pass with a stack of nesting levels."""
        found = [self.n] * self.n
        levels = [self.n]
        for i in range(self.n - 1, -1, -1):
            token = self.tokens[i]
            found[i] = levels[-1]
            if token.kind != PUNCT:
                continue
            if token.text in CLOSERS:
                opener = self.match[i]
                if opener >= 0 and self.match[opener] == i:
                    levels.append(i)
                else:
                    levels[-1] = i
            elif token.text in OPENERS:
                closer = self.match[i]
                if closer < self.n and self.match[closer] == i and len(levels) > 1:
                    levels.pop()
                found[i] = levels[-1]
            elif token.text in (",", ";"):
                levels[-1] = i
        return found

    def segment_start(self, k: int, lo: int) -> int:
        """Return where the left side of the assignment operator at `k` starts, scanning back to `lo` at most."""
        j = k - 1
        while j >= lo:
            token = self.tokens[j]
            if token.kind == PUNCT:
                if token.text in CLOSERS:
                    opener = self.match[j]
                    if opener < lo:
                        return j + 1
                    j = opener - 1
                    continue
                if token.text in OPENERS or token.text in SEGMENT_STOPS or token.text in ASSIGN_OPS:
                    return j + 1
            elif token.kind == IDENT and token.text in SEGMENT_KEYWORDS:
                return j + 1
            j -= 1
        return lo

    def callee_at(self, p: int) -> Callee | None:
        """Return the callee of the call whose `(` is token `p`, or None when no name precedes it."""
        j = p - 1
        if self.text(j) in (">", ">>"):
            opener = self.template_open_back(j)
            if opener < 1:
                return None
            j = opener - 1
        if not self.is_name(j):
            return None
        index, qualifiers, q = j, [], j
        while self.text(q - 1) == "::":
            if self.is_name(q - 2):
                qualifiers.append(self.text(q - 2))
                q -= 2
            else:
                qualifiers.append("")
                q -= 1
                break
        qualifiers.reverse()
        member = self.is_punct(q - 1, (".", "->"))
        receiver = self.receiver(q - 1) if member else None
        return Callee(self.text(index), index, tuple(qualifiers), member, receiver, p, self.close(p))

    def receiver(self, dot: int) -> tuple[int, int] | None:
        """Return the token range of the object of the member access at `dot` (`.` or `->`), or None.

        A receiver longer than TEMPLATE_SCAN_LIMIT steps back (names, member
        accesses, and bracket groups) is not read: None, so a long method
        chain costs no more than that per call.
        """
        j, start, steps = dot - 1, None, 0
        while j >= 0:
            steps += 1
            if steps > TEMPLATE_SCAN_LIMIT:
                self.spend(steps)
                return None
            token = self.tokens[j]
            if token.kind == PUNCT and token.text in (")", "]"):
                opener = self.match[j]
                if opener < 0:
                    break
                start, j = opener, opener - 1
                if j >= 0 and (self.is_name(j) or self.is_punct(j, (")", "]"))):
                    continue
                break
            if token.kind == IDENT and (token.text == "this" or token.text not in KEYWORDS):
                start = j
                if self.is_punct(j - 1, (".", "->", "::")):
                    j -= 2
                    continue
            break
        self.spend(steps)
        return None if start is None else (start, dot)


def mentions(tokens: Sequence[Token]) -> list[Token]:
    """Return the identifier tokens the expression `tokens` mentions (module docstring), in order, unresolved.

    `tokens` are code tokens as lex gives them; every call's arguments are
    read here (the flow analysis applies the call rules that need function
    summaries).
    """
    code = Code(tokens, _match(tokens, "<expression>"))
    return [tokens[i] for i in code.iter_mentions(0, code.n)]


def base(tokens: Sequence[Token]) -> Token | None:
    """Return the token that is the base of the expression `tokens` (module docstring), or None when it has none."""
    code = Code(tokens, _match(tokens, "<expression>"))
    index = code.base_index(0, code.n)
    return None if index is None else tokens[index]


# ---------------------------------------------------------------------------
# Declarations and parameters


@dataclass(frozen=True)
class Declarator:
    """One declared name: its token, kind, initializer range, the token where its value is defined, its types.

    `at` is the token after the name and its array bounds (`=`, `(`, `{`,
    `:`, or the end); `types` are the identifiers of the declared type;
    `head` is the type's last name (`vector` in `std::vector<float>`, `Job`
    in `const Job`), or None when a fundamental type keyword such as auto
    or float ends the type; `binding` marks a name of a structured binding.
    """

    name: int
    kind: str
    init: tuple[int, int] | None
    at: int
    types: frozenset[str]
    head: str | None = None
    binding: bool = False


@dataclass(frozen=True)
class Param:
    """One parameter: its name (None when unnamed), name token, kind, out-port flag, types, and default flag."""

    name: str | None
    index: int
    kind: str
    out: bool
    types: frozenset[str]
    default: bool


def parse_declaration(code: Code, lo: int, hi: int, *, range_for: bool = False) -> list[Declarator] | None:
    """Return the declarators of the declaration [lo, hi), or None when it is not one (see the module docstring).

    With `range_for`, [lo, hi) is a range-for header's declaration, which
    ends at its `:`, so a structured binding there has no initializer.
    """
    if lo >= hi or code.text(lo) in NOT_DECLARATIONS:
        return None
    units, j = _type_sequence(code, lo, hi)
    binding = j
    while binding < hi and code.is_punct(binding, ("&", "&&")):
        binding += 1
    if code.is_punct(binding, ("[",)) and any(code.text(i) == "auto" for i, kind in units if kind == "type"):
        return _binding(code, binding, hi, range_for, REFERENCE if binding > j else VALUE)
    if j < hi and code.is_punct(j, ("*", "&", "&&")):
        type_units, first = units, j
    else:
        if len(units) < 2 or units[-1][1] != "name" or units[-2][1] == "scope":
            return None
        type_units, first = units[:-1], units[-1][0]
    if not type_units or type_units[-1][1] == "scope" or all(kind == "spec" for _, kind in type_units):
        return None
    types = frozenset(code.text(i) for i in range(lo, first) if code.is_name(i))
    head = _head(code, type_units)
    declarators: list[Declarator] = []
    k = first
    while k < hi:
        found, k = _declarator(code, k, hi, types, head)
        if found is None:
            break
        declarators.append(found)
        if k < hi and code.is_punct(k, (",",)):
            k += 1
            continue
        break
    return declarators or None


def _head(code: Code, units: Sequence[tuple[int, str]]) -> str | None:
    """Return the last name of a type sequence's units, or None when a fundamental type keyword comes after it."""
    for index, kind in reversed(units):
        if kind in ("name", "template"):
            return code.text(index)
        if kind == "type":
            return None
    return None


def _binding(code: Code, k: int, hi: int, range_for: bool, kind: str) -> list[Declarator] | None:
    """Return the names of the structured binding `[n1, n2, ...]` at `k` as `kind` declarators of its initializer.

    `kind` is REFERENCE when `&` or `&&` precedes the `[`, else VALUE. In a
    range-for header (`range_for`) a binding that ends the range gives its
    names with no initializer, the range giving their values.
    """
    closer = code.close(k)
    names = [a for a, b in code.split_commas(k + 1, min(closer, hi)) if b - a == 1 and code.is_name(a)]
    at = closer + 1
    if names and range_for and at == hi:
        return [Declarator(name, kind, None, at, frozenset(), binding=True) for name in names]
    if not names or at >= hi or code.text(at) not in ("=", "(", "{"):
        return None
    if code.text(at) == "=":
        init = (at + 1, min(code.next_separator(at), hi))
    else:
        init = (at + 1, min(code.close(at), hi))
    return [Declarator(name, kind, init, at, frozenset(), binding=True) for name in names]


def _type_sequence(code: Code, lo: int, hi: int) -> tuple[list[tuple[int, str]], int]:
    """Return the units of the type sequence starting at `lo` ((index, kind) pairs) and the index after it."""
    units: list[tuple[int, str]] = []
    j = lo
    while j < hi:
        token = code.tokens[j]
        if token.kind == IDENT:
            if token.text in KEYWORDS:
                if token.text not in TYPE_KEYWORDS and token.text not in DECL_SPECIFIERS:
                    break
                units.append((j, "type" if token.text in TYPE_KEYWORDS else "spec"))
                j += 1
                continue
            if code.text(j + 1) == "<":
                end = code.template_end(j + 1)
                if j < end < hi:
                    units.append((j, "template"))
                    j = end + 1
                    continue
            units.append((j, "name"))
        elif token.text == "::" and token.kind == PUNCT:
            units.append((j, "scope"))
        elif token.text == "[" and code.text(j + 1) == "[":
            j = code.close(j) + 1
            continue
        else:
            break
        j += 1
    return units, j


def _declarator(
    code: Code, k: int, hi: int, types: frozenset[str], head: str | None = None
) -> tuple[Declarator | None, int]:
    """Return the declarator starting at `k` and the index after it (None when there is none there).

    A value-declared name whose type's head is a view (VIEW_TYPES), or whose
    `=` initializer is a pointer or iterator producer (Code.producer), is
    pointer-declared: it aliases the storage it points into.
    """
    pointer = reference = array = False
    while k < hi:
        t = code.text(k)
        if t == "*":
            pointer = True
        elif t in ("&", "&&"):
            reference = True
        elif t in RESTRICT_WORDS and code.is_name(k + 1):
            pass
        elif t not in ("const", "volatile"):
            break
        k += 1
    if k >= hi or not code.is_name(k):
        return None, k
    name = k
    k += 1
    while k < hi and code.is_punct(k, ("[",)) and code.text(k + 1) != "[":
        array, k = True, code.close(k) + 1
    end = code.text(k) if k < hi else ";"
    if k > hi or end not in ("=", "(", "{", ";", ",", ":"):
        return None, k
    init, at = None, k
    if k < hi and end == "=":
        stop = min(code.next_separator(k), hi)
        init, k = (k + 1, stop), stop
    elif k < hi and end in ("(", "{"):
        closer = code.close(k)
        init, k = (k + 1, min(closer, hi)), closer + 1
    kind = REFERENCE if reference else POINTER if pointer else ARRAY if array else VALUE
    if kind == VALUE and init is not None and (head in VIEW_TYPES or end == "=" and code.producer(*init)):
        kind = POINTER
    return Declarator(name, kind, init, at, types, head), k


def parse_param(code: Code, lo: int, hi: int) -> Param | None:
    """Return the parameter declared by [lo, hi), or None for an empty range, `void`, or `...`."""
    if lo >= hi or hi - lo == 1 and code.text(lo) in ("void", "..."):
        return None
    end = code.split_at(lo, hi, "=")[0][1]
    default = end < hi
    pointer = reference = array = False
    names: list[int] = []
    outer: list[int] = []
    j = lo
    while j < end:
        t = code.text(j)
        if t == "<" and code.kind(j) == PUNCT:
            closer = code.template_end(j)
            if j < closer < end:
                names.extend(i for i in range(j, closer) if code.is_name(i))
                j = closer + 1
                continue
        if t == "*":
            pointer = True
        elif t in ("&", "&&"):
            reference = True
        elif t == "[" and code.kind(j) == PUNCT:
            array, j = True, code.close(j) + 1
            continue
        elif code.is_name(j):
            names.append(j)
            outer.append(j)
        j += 1
    index = names[-1] if names else -1
    view = len(outer) > 1 and outer[-1] == index and code.text(outer[-2]) in VIEW_TYPES
    kind = REFERENCE if reference else POINTER if pointer or array or view else VALUE
    name = code.text(index) if index >= 0 else None
    types = frozenset(code.text(i) for i in names if i != index)
    return Param(name, index, kind, kind != VALUE, types, default)


def init_captures(code: Code, intro: tuple[int, int]) -> list[tuple[int, str, tuple[int, int]]]:
    """Return the name token, kind, and initializer range of each init capture in the lambda introducer `intro`.

    `intro` is the introducer's brackets. An init capture is an identifier
    followed by `=`, `(`, or `{`, after an optional `&` (reference-declared;
    value-declared otherwise) and `...`; its initializer runs from after
    the `=` to the capture's end, or is the inside of the brackets.
    """
    found: list[tuple[int, str, tuple[int, int]]] = []
    for a, b in code.split_commas(intro[0] + 1, intro[1]):
        j, kind = a, VALUE
        if j < b and code.is_punct(j, ("&",)):
            j, kind = j + 1, REFERENCE
        if j < b and code.is_punct(j, ("...",)):
            j += 1
        if j + 1 < b and code.is_name(j) and code.is_punct(j + 1, ("=", "(", "{")):
            end = b if code.text(j + 1) == "=" else min(code.close(j + 1), b)
            found.append((j, kind, (j + 2, end)))
    return found


# ---------------------------------------------------------------------------
# File structure


@dataclass(frozen=True)
class Function:
    """One function definition in a host file.

    `class_name` is the class of a member function (None for a free one,
    "" for an anonymous class); `scope_class` is the class whose body holds
    the definition, before parse_program reads `C::f` names; `namespaced`
    is True inside a named or anonymous namespace.
    """

    rank: int
    name: str
    qualifiers: tuple[str, ...]
    scope_class: str | None
    class_name: str | None
    namespaced: bool
    head: int
    params: tuple[Param | None, ...]
    variadic: bool
    body: tuple[int, int]

    @property
    def member(self) -> bool:
        """Return True for a member function."""
        return self.class_name is not None

    def admits(self, count: int) -> bool:
        """Return True when a call with `count` arguments can bind to this definition."""
        required = sum(1 for param in self.params if param is None or not param.default)
        return required <= count and (self.variadic or count <= len(self.params))


@dataclass(frozen=True)
class GlobalDecl:
    """One file-scope variable: its file rank and declarator."""

    rank: int
    declarator: Declarator


@dataclass
class FileStructure:
    """What parse_file finds at file, namespace, and class scope.

    `members` maps each class to its data members' names and declared type
    names (constexpr members excluded: they hold no object state);
    `base_names` maps each class to the names in its base clause.
    """

    functions: list[Function] = field(default_factory=list)
    classes: set[str] = field(default_factory=set)
    globals: list[GlobalDecl] = field(default_factory=list)
    members: dict[str, dict[str, frozenset[str]]] = field(default_factory=dict)
    base_names: dict[str, set[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class _Scope:
    """A container being read: the class whose body it is (None outside a class) and whether it is a namespace."""

    class_name: str | None
    namespaced: bool


def parse_file(code: Code, rank: int, defines: Container[str] = ()) -> FileStructure:
    """Return the functions, classes, data members, and file-scope variables of one file's code.

    `defines` are the macro names of the host text, which a parenthesized
    initializer may name (`std::vector<float> g(N);`).
    """
    return _TopLevel(code, rank, defines).run()


class _TopLevel:
    """The state of one parse_file call: a stack of containers still to read."""

    def __init__(self, code: Code, rank: int, defines: Container[str] = ()) -> None:
        """Keep the code and start with the whole file as the one container."""
        self.code = code
        self.rank = rank
        self.defines = defines
        self.found = FileStructure()
        self.global_names: set[str] = set()
        self.work: list[tuple[int, int, _Scope]] = [(0, code.n, _Scope(None, False))]

    def run(self) -> FileStructure:
        """Read every container and return what was found, functions in text order."""
        while self.work:
            lo, hi, scope = self.work.pop()
            i = lo
            while i < hi:
                i = max(self._statement(i, hi, scope), i + 1)
        self.found.functions.sort(key=lambda item: item.head)
        return self.found

    def _push(self, opener: int, scope: _Scope) -> int:
        """Queue the body of the container whose `{` is `opener`; return the index after its closer."""
        closer = self.code.close(opener)
        self.work.append((opener + 1, min(closer, self.code.n), scope))
        return closer + 1

    def _statement(self, i: int, hi: int, scope: _Scope) -> int:
        """Read the file-scope statement at `i`; return the index after it."""
        code = self.code
        t = code.text(i)
        if code.kind(i) == PUNCT:
            return code.close(i) + 1 if t == "{" else i + 1
        if t == "namespace":
            j = i + 1
            while j < hi and code.text(j) not in ("{", ";", "="):
                j += 1
            if code.is_punct(j, ("{",)):
                return self._push(j, _Scope(scope.class_name, True))
            return code.statement_end(j, hi) + 1
        if t == "extern" and code.kind(i + 1) == LITERAL and code.text(i + 2) == "{":
            return self._push(i + 2, scope)
        if t == "template":
            end = code.template_end(i + 1) if code.text(i + 1) == "<" else -1
            return end + 1 if end > i else i + 1
        if t in ("using", "typedef", "static_assert", "friend", "asm", "concept", "enum"):
            return code.statement_end(i, hi) + 1
        if t in ("public", "private", "protected") and code.text(i + 1) == ":":
            return i + 2
        if t in CLASS_KEYS:
            head = class_head(code, i, hi)
            if head is not None:
                name, opener = head
                self.found.classes.add(name)
                self.found.members.setdefault(name, {})
                self.found.base_names.setdefault(name, set()).update(class_bases(code, i, opener))
                after = self._push(opener, _Scope(name, scope.namespaced))
                return code.statement_end(after, hi) + 1
        return self._declaration(i, hi, scope)

    def _declaration(self, i: int, hi: int, scope: _Scope) -> int:
        """Read a declaration or function definition from `i`; return the index after it."""
        code = self.code
        j, assigned = i, False
        while j < hi:
            t, punct = code.text(j), code.kind(j) == PUNCT
            if punct and t == ";":
                if scope.class_name is None:
                    self._globals(i, j)
                else:
                    self._members(scope.class_name, i, j)
                return j + 1
            if punct and t == "=" and code.text(j - 1) != "operator":
                assigned = True
            elif punct and t == "(" and not assigned:
                done = self._function(j, hi, scope, i)
                if done is not None:
                    return done
            elif punct and t in CLOSERS:
                return j + 1
            j = code.close(j) + 1 if punct and t in OPENERS else j + 1
        return hi

    def _function(self, p: int, hi: int, scope: _Scope, start: int) -> int | None:
        """Return the index after the function defined or declared with parameters at `p`, or None for neither.

        A declaration whose parentheses hold an initializer rather than
        parameters (_initializer_parens) is a file-scope variable: it is
        recorded from the statement's `start`.
        """
        code = self.code
        head = function_name(code, p)
        closer = code.close(p)
        if head is None or closer >= hi:
            return None
        k = _qualifiers_end(code, closer + 1, hi)
        if k is None:
            return None
        if code.is_punct(k, (":",)):
            k = _initializers_end(code, k, hi)
            if k is None:
                return None
        if code.is_punct(k, ("{",)):
            self._record(head, p, closer, k, scope)
            return code.close(k) + 1
        if code.is_punct(k, (";",)) or code.text(k) == "=" and code.text(k + 1) in ("0", "default", "delete"):
            if code.is_punct(k, (";",)) and scope.class_name is None and self._initializer_parens(p, closer):
                self._globals(start, k, parens=True)
            return code.statement_end(k, hi) + 1
        return None

    def _initializer_parens(self, p: int, closer: int) -> bool:
        """Return True when the parentheses (p, closer) of a declaration hold an initializer, not parameters.

        They do when the first token in them is a number, a literal, `{`, a
        unary operator, one of true, false, nullptr, this, sizeof, or new, a
        file-scope variable already recorded from this file (the module
        docstring, Structure), or a macro name.
        """
        code, first = self.code, p + 1
        if first >= closer:
            return False
        t = code.text(first)
        if code.kind(first) in (NUMBER, LITERAL) or t in ("{", "-", "+", "!", "~", "&"):
            return True
        if t in ("true", "false", "nullptr", "this", "sizeof", "new"):
            return True
        return code.is_name(first) and (t in self.global_names or t in self.defines)

    def _record(self, head: tuple[str, tuple[str, ...], int], p: int, closer: int, body: int, scope: _Scope) -> None:
        """Record the function definition named `head` with parameters (p, closer) and body at `body`."""
        code = self.code
        name, qualifiers, index = head
        parts = code.split_commas(p + 1, closer, angles=True)
        variadic = any(code.text(b - 1) == "..." for a, b in parts if b > a)
        kept = [(a, b) for a, b in parts if not (b - a == 1 and code.text(a) == "...")]
        params = tuple(parse_param(code, a, b) for a, b in kept)
        if params == (None,):
            params = ()
        self.found.functions.append(
            Function(self.rank, name, qualifiers, scope.class_name, None, scope.namespaced, index, params, variadic,
                     (body, code.close(body)))
        )

    def _globals(self, lo: int, hi: int, parens: bool = False) -> None:
        """Record the variables a file- or namespace-scope declaration [lo, hi) declares.

        A declarator followed by `(` is a function declaration unless
        `parens` says its parentheses hold an initializer.
        """
        declarators = parse_declaration(self.code, lo, hi)
        for declarator in declarators or ():
            if parens or self.code.text(declarator.at) != "(":
                self.found.globals.append(GlobalDecl(self.rank, declarator))
                self.global_names.add(self.code.text(declarator.name))

    def _members(self, class_name: str, lo: int, hi: int) -> None:
        """Record the data members a class-scope declaration [lo, hi) declares (none for a constexpr one)."""
        code = self.code
        declarators = parse_declaration(code, lo, hi)
        if not declarators or any(code.text(k) == "constexpr" for k in range(lo, declarators[0].name)):
            return
        members = self.found.members.setdefault(class_name, {})
        for declarator in declarators:
            if code.text(declarator.at) != "(":
                members[code.text(declarator.name)] = declarator.types


def class_head(code: Code, i: int, hi: int) -> tuple[str, int] | None:
    """Return (name, index of `{`) when the class key at `i` starts a class definition, else None."""
    j, name = i + 1, ""
    while j < hi and (code.text(j) == "[" and code.text(j + 1) == "[" or code.text(j) == "alignas"):
        j = code.close(j if code.text(j) == "[" else j + 1) + 1
    while j < hi:
        t = code.text(j)
        if t == "final" and code.text(j + 1) in ("{", ":"):
            break
        if code.is_name(j):
            name = t
        elif t == "<":
            end = code.template_end(j)
            if end < 0:
                return None
            j = end
        elif t != "::":
            break
        j += 1
    if code.text(j) == "final":
        j += 1
    if code.is_punct(j, (":",)):
        while j < hi and code.text(j) not in ("{", ";"):
            j = code.close(j) + 1 if code.is_punct(j, ("(", "[")) else j + 1
    return (name, j) if code.is_punct(j, ("{",)) else None


def class_bases(code: Code, i: int, opener: int) -> set[str]:
    """Return the names in the base clause of the class whose key is at `i` and whose body opens at `opener`."""
    j = i + 1
    while j < opener and not code.is_punct(j, (":",)):
        if code.text(j) == "<":
            end = code.template_end(j)
            j = end + 1 if end > j else j + 1
            continue
        j = code.close(j) + 1 if code.is_punct(j, ("(", "[")) else j + 1
    return {code.text(k) for k in range(j + 1, opener) if code.is_name(k)}


def function_name(code: Code, p: int) -> tuple[str, tuple[str, ...], int] | None:
    """Return (name, qualifiers, first index) of the declarator before the `(` at `p`, or None."""
    named = _operator_name(code, p)
    if named is None:
        j = p - 1
        if code.text(j) in (">", ">>"):
            j = code.template_open_back(j) - 1
        if j < 0 or not code.is_name(j):
            return None
        name, start = code.text(j), j
        if code.text(start - 1) == "~":
            name, start = "~" + name, start - 1
    else:
        name, start = named
    qualifiers: list[str] = []
    while code.text(start - 1) == "::":
        if not code.is_name(start - 2):
            start -= 1
            break
        qualifiers.append(code.text(start - 2))
        start -= 2
    qualifiers.reverse()
    return name, tuple(qualifiers), start


def _operator_name(code: Code, p: int) -> tuple[str, int] | None:
    """Return (name, index of `operator`) when an operator function's name ends before `p`, else None."""
    for back in (1, 2, 3):
        if code.text(p - back) == "operator" and code.kind(p - back) == IDENT:
            parts = [code.text(i) for i in range(p - back + 1, p)]
            if parts == ["("]:
                return None
            return "operator" + "".join(parts), p - back
    return None


def _qualifiers_end(code: Code, k: int, hi: int) -> int | None:
    """Return the index after a function's qualifiers and trailing return type starting at `k`.

    None when the trailing return type runs past TEMPLATE_SCAN_LIMIT tokens
    (Code.trailing_type_end): no function is read there.
    """
    while k < hi:
        t = code.text(k)
        if t in ("const", "volatile", "override", "final", "&", "&&", "try"):
            k += 1
        elif t in ("noexcept", "throw", "__attribute__"):
            k = code.close(k + 1) + 1 if code.text(k + 1) == "(" else k + 1
        elif t == "[" and code.text(k + 1) == "[":
            k = code.close(k) + 1
        elif t == "->":
            end = code.trailing_type_end(k + 1, hi, ("{", ";", "="), ("(", "["))
            return None if end < 0 else end
        else:
            return k
    return k


def _initializers_end(code: Code, k: int, hi: int) -> int | None:
    """Return the index of the body `{` after the constructor initializer list at `k` (`:`), or None."""
    j = k + 1
    while j < hi:
        if code.text(j) == "::":
            j += 1
        if not code.is_name(j):
            return None
        j += 1
        while code.text(j) == "::" and code.is_name(j + 1):
            j += 2
        if code.text(j) == "<":
            end = code.template_end(j)
            if end < 0:
                return None
            j = end + 1
        if not code.is_punct(j, ("(", "{")):
            return None
        j = code.close(j) + 1
        if code.text(j) == "...":
            j += 1
        if code.is_punct(j, ("{",)):
            return j
        if not code.is_punct(j, (",",)):
            return None
        j += 1
    return None


# ---------------------------------------------------------------------------
# Symbols and the program


@dataclass(frozen=True, slots=True)
class Symbol:
    """One named thing the flow analysis tracks: name, declared kind, role, declared type names, out-port flag."""

    name: str
    kind: str
    role: str
    types: frozenset[str] = frozenset()
    out: bool = False


class SymbolTable:
    """Every symbol of one program, numbered from 0."""

    def __init__(self) -> None:
        """Start empty."""
        self.items: list[Symbol] = []

    def new(self, symbol: Symbol) -> int:
        """Add `symbol` and return its number."""
        self.items.append(symbol)
        return len(self.items) - 1

    def __getitem__(self, number: int) -> Symbol:
        """Return symbol `number`."""
        return self.items[number]


@dataclass
class Program:
    """The host text as parse_program reads it.

    `paths` and `codes` are indexed by file rank (the sorted order the
    caller gives); `functions` are in (rank, position) order; `globals` maps
    each file-scope variable name to its one symbol; `aliases` merges every
    file's single-identifier macro aliases, the first by rank winning;
    `members` and `bases` are each class's data members (name -> declared
    type names) and host base classes, merged over files; `open_classes`
    are the classes whose base clause names a class outside the host text;
    `defines` are every macro name; `budget` bounds the reading's work.
    """

    paths: tuple[str, ...]
    codes: tuple[Code, ...]
    functions: tuple[Function, ...]
    classes: frozenset[str]
    globals: dict[str, int]
    global_decls: tuple[GlobalDecl, ...]
    aliases: dict[str, str]
    symbols: SymbolTable
    members: dict[str, dict[str, frozenset[str]]] = field(default_factory=dict)
    bases: dict[str, frozenset[str]] = field(default_factory=dict)
    open_classes: frozenset[str] = frozenset()
    defines: frozenset[str] = frozenset()
    budget: Budget | None = None

    def class_view(self, class_name: str) -> tuple[dict[str, frozenset[str]], bool]:
        """Return the data members of `class_name` and its host bases, and whether any of them is open."""
        members: dict[str, frozenset[str]] = {}
        seen, stack, opened = set(), [class_name], False
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            for member, types in self.members.get(name, {}).items():
                members.setdefault(member, types)
            opened = opened or name in self.open_classes
            stack.extend(sorted(self.bases.get(name, ())))
        return members, opened


def parse_program(files: Sequence[Lexed], budget: Budget | None = None) -> Program:
    """Return the Program of `files`, whose order gives each file its rank; `budget` bounds its scans."""
    codes = tuple(Code(item.tokens, item.match, budget) for item in files)
    defines = frozenset(name for item in files for name in item.defines)
    found = [parse_file(code, rank, defines) for rank, code in enumerate(codes)]
    classes = frozenset(name for item in found for name in item.classes)
    functions = []
    for item in found:
        for function in item.functions:
            if function.scope_class is not None:
                owner: str | None = function.scope_class
            else:
                owner = function.qualifiers[-1] if function.qualifiers and function.qualifiers[-1] in classes else None
            functions.append(replace(function, class_name=owner))
    symbols = SymbolTable()
    names: dict[str, int] = {}
    decls = tuple(decl for item in found for decl in item.globals)
    for decl in decls:
        name = codes[decl.rank].text(decl.declarator.name)
        if name not in names:
            kind = decl.declarator.kind
            names[name] = symbols.new(Symbol(name, kind, GLOBAL, decl.declarator.types, out=True))
    aliases: dict[str, str] = {}
    for item in files:
        for alias, target in item.aliases.items():
            aliases.setdefault(alias, target)
    members: dict[str, dict[str, frozenset[str]]] = {}
    base_names: dict[str, set[str]] = {}
    for item in found:
        for class_name, declared in item.members.items():
            members.setdefault(class_name, {}).update(declared)
        for class_name, named in item.base_names.items():
            base_names.setdefault(class_name, set()).update(named)
    bases = {name: frozenset(named & classes) for name, named in base_names.items()}
    open_classes = frozenset(name for name, named in base_names.items() if named - classes)
    paths = tuple(item.path for item in files)
    return Program(
        paths, codes, tuple(functions), classes, names, decls, aliases, symbols, members, bases, open_classes,
        defines, budget,
    )


# ---------------------------------------------------------------------------
# Function bodies


@dataclass(frozen=True)
class Decl:
    """A declared name with an initializer range or none; its value is defined at token `at`.

    `head` is the declared type's last name (Declarator.head); `binding`
    marks a name of a structured binding; `capture` is the `[` index of the
    lambda whose reference init capture the name is, or None.
    """

    symbol: int
    kind: str
    init: tuple[int, int] | None
    at: int
    head: str | None = None
    binding: bool = False
    capture: int | None = None


@dataclass(frozen=True)
class RangeFor:
    """A range-for declaration `for (D X : E)`: X's symbol and kind, E's range, the `:` token, and a binding flag."""

    symbol: int
    kind: str
    range: tuple[int, int]
    at: int
    binding: bool = False


@dataclass(frozen=True)
class Return:
    """A return statement outside a lambda body: its expression range and the `return` token."""

    range: tuple[int, int]
    at: int


@dataclass(frozen=True)
class Expr:
    """An expression range whose assignments and calls the flow analysis reads (lambda bodies excluded)."""

    range: tuple[int, int]


@dataclass
class Body:
    """One function body walked: its symbols, items, and the scopes that resolve names.

    `params` holds each parameter's symbol (None for an unnamed one), `this`
    the THIS symbol of a member function, `ret` the RET symbol, and
    `lambda_params` each lambda's parameter symbols by its `[` index.
    `members` are the data members of a member function's class and its
    host bases (name -> declared type names) and `open_class` says whether
    one of those classes has a base outside the host text; `defines` are the
    host text's macro names. `loops` are the token ranges (keyword, last
    token) of the for, while, and do statements, lambda bodies included;
    `named_lambdas` maps the symbol of each variable initialized with `=`
    and exactly one lambda to that Lambda.
    """

    function: Function
    code: Code
    params: tuple[int | None, ...]
    this: int | None
    ret: int
    items: list[Decl | RangeFor | Return | Expr]
    lambda_params: dict[int, tuple[int | None, ...]]
    blocks: list[tuple[int, int, int]]
    decls: dict[tuple[int, str], tuple[list[int], list[int]]]
    globals: Mapping[str, int]
    table: SymbolTable
    inner: list[int] = field(default_factory=list)
    free: dict[str, int] = field(default_factory=dict)
    members: Mapping[str, frozenset[str]] = field(default_factory=dict)
    open_class: bool = False
    defines: Container[str] = ()
    loops: list[tuple[int, int]] = field(default_factory=list)
    named_lambdas: dict[int, Lambda] = field(default_factory=dict)
    param_symbols: dict[str, int] = field(default_factory=dict)

    def resolve(self, i: int) -> int:
        """Return the symbol identifier token `i` names (see the module docstring, Structure)."""
        name = self.code.text(i)
        local = self.lookup(i)
        if local is not None:
            return local
        if self.this is not None and (name == "this" or name not in self.globals and self.is_member(name)):
            return self.this
        if name in self.globals and name != "this":
            return self.globals[name]
        if name not in self.free:
            self.free[name] = self.table.new(Symbol(name, VALUE, FREE))
        return self.free[name]

    def lookup(self, i: int) -> int | None:
        """Return the symbol of the block declaration or parameter that identifier token `i` names, or None.

        Each block passed spends one step of the budget (Bounds in the module docstring).
        """
        name = self.code.text(i)
        block = self._block_at(i)
        while block >= 0:
            self.code.spend(1)
            entry = self.decls.get((block, name))
            if entry is not None:
                k = bisect_left(entry[0], i) - 1
                if k >= 0:
                    return entry[1][k]
            block = self.blocks[block][2]
        return self.param_symbols.get(name)

    def is_member(self, name: str) -> bool:
        """Return True when `name` is a data member in a member function: recorded, or unknown in an open class."""
        return name in self.members or self.open_class and name not in self.defines

    def _block_at(self, i: int) -> int:
        """Return the innermost block holding token `i` (0 outside the body's recorded range)."""
        offset = i - self.function.body[0]
        return self.inner[offset] if 0 <= offset < len(self.inner) else 0


def walk_body(program: Program, index: int) -> Body:
    """Return the walked body of program.functions[index] (see the module docstring, Structure)."""
    return _Walker(program, program.functions[index]).walk()


class _Walker:
    """The state of one walk_body call: blocks, declarations, items, and a stack of ranges still to walk."""

    def __init__(self, program: Program, function: Function) -> None:
        """Make the function's parameter, THIS, and RET symbols."""
        self.program = program
        self.code = program.codes[function.rank]
        self.function = function
        self.table = program.symbols
        self.globals = program.globals
        self.blocks: list[tuple[int, int, int]] = []
        self.decls: dict[tuple[int, str], list[tuple[int, int]]] = {}
        self.items: list[Decl | RangeFor | Return | Expr] = []
        self.lambda_params: dict[int, tuple[int | None, ...]] = {}
        self.loops: list[tuple[int, int]] = []
        self.named_lambdas: dict[int, Lambda] = {}
        self.declaring: set[int] = set()
        self.ends: dict[int, int] = {}
        self.work: list[tuple[int, int, int, bool]] = []
        self.params = tuple(
            None if param is None or param.name is None
            else self.table.new(Symbol(param.name, param.kind, PARAM, param.types, param.out))
            for param in function.params
        )
        self.this = self.table.new(Symbol("this", VALUE, THIS, out=True)) if function.member else None
        self.ret = self.table.new(Symbol("return", VALUE, RET))

    def walk(self) -> Body:
        """Walk every statement of the body, lambda bodies included, and return the Body."""
        opener, closer = self.function.body
        root = self._block(opener, closer, -1)
        self.work.append((opener + 1, min(closer, self.code.n), root, False))
        while self.work:
            lo, hi, block, lam = self.work.pop()
            i = lo
            while i < hi:
                after = max(self._statement(i, hi, block, lam), i + 1)
                self.code.spend(1)
                i = after
        decls = {key: ([pos for pos, _ in sorted(entries)], [sym for _, sym in sorted(entries)])
                 for key, entries in self.decls.items()}
        body = Body(self.function, self.code, self.params, self.this, self.ret, self.items, self.lambda_params,
                    self.blocks, decls, self.globals, self.table)
        body.inner = self._innermost(opener, min(closer, self.code.n))
        if self.function.class_name is not None:
            body.members, body.open_class = self.program.class_view(self.function.class_name)
        body.defines = self.program.defines
        body.loops = sorted(self.loops)
        body.named_lambdas = self.named_lambdas
        for param, symbol in zip(self.function.params, self.params, strict=True):
            if param is not None and param.name is not None and symbol is not None:
                body.param_symbols.setdefault(param.name, symbol)
        return body

    def _block(self, start: int, end: int, parent: int) -> int:
        """Add a block over tokens [start, end], clipped to its parent's end; return its number."""
        if parent >= 0:
            end = min(end, self.blocks[parent][1])
        self.blocks.append((start, end, parent))
        return len(self.blocks) - 1

    def _innermost(self, opener: int, closer: int) -> list[int]:
        """Return, for each token from `opener` to `closer`, the innermost block holding it."""
        order = sorted(range(len(self.blocks)), key=lambda b: (self.blocks[b][0], -self.blocks[b][1]))
        inner = [0] * max(0, closer - opener + 1)
        stack: list[int] = []
        k = 0
        for offset in range(len(inner)):
            pos = opener + offset
            while k < len(order) and self.blocks[order[k]][0] <= pos:
                while stack and self.blocks[stack[-1]][1] < self.blocks[order[k]][0]:
                    stack.pop()
                stack.append(order[k])
                k += 1
            while stack and self.blocks[stack[-1]][1] < pos:
                stack.pop()
            inner[offset] = stack[-1] if stack else 0
        return inner

    def _statement(self, i: int, hi: int, block: int, lam: bool) -> int:
        """Walk the statement at `i`; return the index after it (after a leading `[[...]]`, which leads a statement)."""
        code = self.code
        t = code.text(i)
        if code.kind(i) == PUNCT:
            if t == "{":
                return self._braced(i, block, lam)
            if t == ";" or t in CLOSERS:
                return i + 1
            if t == "[" and code.is_punct(i + 1, ("[",)):
                return code.close(i) + 1
        elif code.kind(i) == IDENT:
            if t in ("if", "while", "switch", "for", "catch"):
                return self._header(i, hi, block, lam)
            if t == "do":
                self.loops.append((i, self._do_last(i)))
            if t in ("else", "do", "try"):
                return i + 1
            if t in ("return", "co_return"):
                return self._return(i, hi, block, lam)
            if t == "case" or t == "default" and code.text(i + 1) == ":":
                return self._label_end(i, hi)
            if t in SKIPPED_STATEMENTS:
                return code.statement_end(i, hi) + 1
            head = class_head(code, i, hi) if t in CLASS_KEYS else None
            if head is not None:
                return code.statement_end(code.close(head[1]) + 1, hi) + 1
            if code.is_name(i) and code.is_punct(i + 1, (":",)):
                return i + 2
        end = code.statement_end(i, hi)
        self._simple(i, min(end, hi), block, lam)
        return end + 1

    def _braced(self, i: int, block: int, lam: bool) -> int:
        """Queue the block whose `{` is `i`; return the index after it."""
        closer = self.code.close(i)
        inner = self._block(i, closer, block)
        self.work.append((i + 1, min(closer, self.code.n), inner, lam))
        return closer + 1

    def _header(self, i: int, hi: int, block: int, lam: bool) -> int:
        """Walk the parenthesized header of if, while, switch, for, or catch at `i`; return the index after it.

        The header's declarations go into a scope block that starts at the
        keyword. When it declares a name, the scope is widened to the end of
        the statement the header controls (an if's else branch included) and
        that statement is walked inside it (Structure in the module docstring).
        """
        code = self.code
        j = i + 1
        if code.text(i) == "if" and code.text(j) in ("constexpr", "consteval", "!"):
            j += 2 if code.text(j) == "!" else 1
        if not code.is_punct(j, ("(",)):
            return i + 1
        closer = code.close(j)
        if code.text(i) in ("for", "while"):
            self.loops.append((i, self._statement_last(closer + 1)))
        scope = self._block(i, min(closer, hi - 1), block)
        end = min(closer, code.n)
        if code.text(i) == "catch":
            param = parse_param(code, j + 1, end)
            if param is not None and param.name is not None:
                self._declare_name(scope, param.name, param.index, param.kind, param.types)
        elif code.text(i) == "for":
            self._for(j + 1, end, scope, lam)
        else:
            parts = code.split_at(j + 1, end, ";")
            for a, b in parts[:-1]:
                self._simple(a, b, scope, lam)
            self._simple(*parts[-1], scope, lam, condition=True)
        if scope not in self.declaring:
            return closer + 1
        last = self._statement_last(closer + 1)
        if code.text(i) == "if" and code.text(last + 1) == "else":
            last = self._statement_last(last + 2)
        last = min(last, hi - 1, self.blocks[block][1])
        self.blocks[scope] = (i, max(last, i), block)
        if closer + 1 <= last:
            self.work.append((closer + 1, last + 1, scope, lam))
        return max(last, closer) + 1

    def _statement_last(self, k: int) -> int:
        """Return the index of the last token of the statement starting at `k` (a loop's or a header's body).

        A braced block ends at its `}`; if, for, while, and switch headers
        and a leading attribute `[[...]]` are passed to the statement they
        control or lead; an `else` continues an if whose header was passed;
        a try statement ends at its last handler's `}`; any other statement
        ends at its `;`. Each header or attribute passed, each braced block
        (a try block and each handler included), and each token from another
        statement's start to its end spend one step of the budget. The end
        found is kept for every token the search reached with no if waiting
        for an else, where a search from that token would find the same
        end, so a later search that reaches one with no if waiting stops
        there and spends nothing more (an else-if chain is searched once).
        """
        code, n = self.code, self.code.n
        ifs, starts, last = 0, [], n - 1
        while k < n:
            if not ifs:
                if k in self.ends:
                    last = self.ends[k]
                    break
                starts.append(k)
            t = code.text(k)
            j = k + 1
            if t == "if" and code.text(j) in ("constexpr", "consteval", "!"):
                j += 2 if code.text(j) == "!" else 1
            if code.is_punct(k, ("[",)) and code.is_punct(j, ("[",)):
                k = code.close(k) + 1
                code.spend(1)
                continue
            if code.is_punct(k, ("{",)):
                end = code.close(k)
                code.spend(1)
            elif code.kind(k) == IDENT and t == "try" and code.is_punct(j, ("{",)):
                end = self._try_last(j)
            elif code.kind(k) == IDENT and t in ("if", "for", "while", "switch") and code.is_punct(j, ("(",)):
                ifs += t == "if"
                k = code.close(j) + 1
                code.spend(1)
                continue
            elif code.kind(k) == IDENT and t == "do":
                end = self._do_last(k)
                code.spend(max(1, end - k + 1))
            else:
                end = code.statement_end(k, n)
                code.spend(max(1, end - k + 1))
            if ifs and code.text(end + 1) == "else":
                ifs, k = ifs - 1, end + 2
                continue
            last = min(end, n - 1)
            break
        for start in starts:
            self.ends[start] = last
        return last

    def _try_last(self, block: int) -> int:
        """Return the index of the `}` that ends the try statement whose block opens at `block` (its last handler's)."""
        code = self.code
        end = code.close(block)
        code.spend(1)
        while code.text(end + 1) == "catch" and code.is_punct(end + 2, ("(",)):
            handler = code.close(end + 2) + 1
            if not code.is_punct(handler, ("{",)):
                break
            end = code.close(handler)
            code.spend(1)
        return min(end, code.n - 1)

    def _do_last(self, i: int) -> int:
        """Return the index of the last token of the do statement at `i` (its `while (...)` and `;`)."""
        code = self.code
        body = i + 1
        end = code.close(body) if code.is_punct(body, ("{",)) else code.statement_end(body, code.n)
        if code.text(end + 1) == "while" and code.is_punct(end + 2, ("(",)):
            end = code.close(end + 2)
            if code.is_punct(end + 1, (";",)):
                end += 1
        return min(end, code.n - 1)

    def _for(self, lo: int, hi: int, block: int, lam: bool) -> None:
        """Walk a for header: init, condition, and increment, or a range-for after an optional init statement.

        The increment, and anything after it, is an expression.
        """
        code = self.code
        parts = code.split_at(lo, hi, ";")
        colon = self._range_colon(parts[-1][0], hi) if len(parts) <= 2 else None
        if colon is None:
            self._simple(*parts[0], block, lam)
            if len(parts) > 1:
                self._simple(*parts[1], block, lam, condition=True)
            for a, b in parts[2:]:
                self._expr(a, b, block)
            return
        if len(parts) == 2:
            self._simple(*parts[0], block, lam)
        declarators = parse_declaration(code, parts[-1][0], colon, range_for=True)
        if declarators is not None and all(found.init is None and found.at == colon for found in declarators):
            for found in declarators:
                symbol = self._declare(block, found)
                self.items.append(RangeFor(symbol, found.kind, (colon + 1, hi), colon, found.binding))
        self._expr(colon + 1, hi, block)

    def _range_colon(self, lo: int, hi: int) -> int | None:
        """Return the depth-0 `:` of a range-for header [lo, hi), or None (a `?` first means none)."""
        code = self.code
        i = lo
        while i < hi:
            if code.is_punct(i, OPENERS):
                i = code.close(i) + 1
                continue
            if code.is_punct(i, ("?", ";")):
                return None
            if code.is_punct(i, (":",)):
                return i
            i += 1
        return None

    def _return(self, i: int, hi: int, block: int, lam: bool) -> int:
        """Walk a return statement; a lambda's return gives no Return item."""
        end = min(self.code.statement_end(i + 1, hi), hi)
        if end > i + 1:
            if not lam:
                self.items.append(Return((i + 1, end), i))
            self._expr(i + 1, end, block)
        return end + 1

    def _label_end(self, i: int, hi: int) -> int:
        """Return the index after the `:` that ends a case or default label at `i`."""
        j = i + 1
        while j < hi and not self.code.is_punct(j, (":",)):
            j = self.code.close(j) + 1 if self.code.is_punct(j, OPENERS) else j + 1
        return j + 1

    def _simple(self, lo: int, hi: int, block: int, lam: bool, *, condition: bool = False) -> None:
        """Walk a declaration or an expression statement over [lo, hi).

        A `condition` is a declaration only when every name it declares has
        an `=` or braced initializer, as C++ requires there; otherwise it is
        an expression (`x && y`, `x & y`, `x * y`, `ok && f(v)`).
        """
        if lo >= hi:
            return
        declarators = parse_declaration(self.code, lo, hi)
        if declarators is not None and condition:
            if not all(found.init is not None and self.code.text(found.at) in ("=", "{") for found in declarators):
                declarators = None
        if declarators is None:
            self._expr(lo, hi, block)
            return
        walked: set[tuple[int, int]] = set()
        for found in declarators:
            symbol = self._declare(block, found)
            self.items.append(Decl(symbol, found.kind, found.init, found.at, found.head, found.binding))
            if found.init is None or found.init in walked:
                continue
            walked.add(found.init)
            if self.code.text(found.at) == "=":
                lam_found = self.code.lambda_at(found.init[0])
                if lam_found is not None and lam_found.body[1] == found.init[1] - 1:
                    self.named_lambdas[symbol] = lam_found
            self._expr(found.init[0], found.init[1], block)

    def _declare(self, block: int, found: Declarator) -> int:
        """Declare `found` in `block` and return its new symbol."""
        return self._declare_name(block, self.code.text(found.name), found.name, found.kind, found.types)

    def _declare_name(self, block: int, name: str, at: int, kind: str, types: frozenset[str] = frozenset()) -> int:
        """Declare a local `name` of `kind` in `block`, visible after token `at`; return its new symbol."""
        symbol = self.table.new(Symbol(name, kind, LOCAL, types, kind != VALUE))
        self.decls.setdefault((block, name), []).append((at, symbol))
        self.declaring.add(block)
        return symbol

    def _expr(self, lo: int, hi: int, block: int) -> None:
        """Record the expression [lo, hi) and walk each lambda in it as a block of the function."""
        if lo >= hi:
            return
        self.items.append(Expr((lo, hi)))
        code = self.code
        i = lo
        while i < hi:
            found = code.lambda_at(i) if code.is_punct(i, ("[",)) else None
            if found is None:
                i += 1
                continue
            self._lambda(found, block)
            i = max(found.body[1] + 1, i + 1)

    def _lambda(self, found: Lambda, block: int) -> None:
        """Make the block of the lambda `found`, declare its init captures and parameters there, and queue its body.

        An init capture's name is visible after the introducer, so its
        initializer names the enclosing scope's variables. A reference init
        capture gets a Decl of its initializer marked with the lambda
        (lassi.toolchains._cxx_flow links it unless the lambda is enclosed,
        Positions there); a value one gets none, so it has no value.
        """
        code = self.code
        inner = self._block(found.intro[0], found.body[1], block)
        for name, kind, init in init_captures(code, found.intro):
            symbol = self._declare_name(inner, code.text(name), found.intro[1], kind)
            if kind == REFERENCE:
                self.items.append(Decl(symbol, kind, init, found.intro[1], capture=found.intro[0]))
        params: list[int | None] = []
        if found.params is not None:
            for a, b in code.split_commas(found.params[0] + 1, found.params[1], angles=True):
                param = parse_param(code, a, b)
                if param is None or param.name is None:
                    params.append(None)
                    continue
                params.append(self._declare_name(inner, param.name, param.index, param.kind, param.types))
        self.lambda_params[found.intro[0]] = tuple(params)
        self.work.append((found.body[0] + 1, min(found.body[1], code.n), inner, True))
