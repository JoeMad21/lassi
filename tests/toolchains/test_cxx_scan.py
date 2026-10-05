"""Unit tests for the generic C and C++ text reader lassi.toolchains._cxx_scan (task P4.12).

Bible: Harness Contract (the CPU -> TT guard, which reads host text with this
reader), Design Principle 8 (the reader names nothing from tt-metal). Plan:
plans/p4-ttsim.md, P4.12. The reader's behavior is stated in the module
docstring of lassi.toolchains._cxx_scan (Lexing, Includes, Structure,
Expressions, Bounds).

These tests use one small surface of the module, reached only through the
adapters below, so another spelling changes those adapters and nothing
else:

- `lex(text, path)` returns an object with
  - `tokens`: the code tokens in order, each with `kind`, `text`, and `line`
    (the 1-based original line of its first character). Directive lines and
    `#if 0` / `#if false` regions are not among them. `kind` is "LITERAL"
    for a string, char, or raw-string literal (one token each, its prefix
    included) and "OTHER" for a non-ASCII character outside a literal or a
    comment; these tests read no other kind name.
  - `directives`: one entry per directive line outside a dropped region,
    each with `tokens`, the first of which is "#".
  - `aliases`: {A: B} for each `#define A B` whose body is one identifier.
  It raises `Unreadable`, whose `reason` names the problem and the path
  ("an unterminated comment in <path>", "an unterminated literal in
  <path>").
- `include_names(lexed)`: [(name, quoted)] for each `#include` outside a
  dropped region, in order.
- `resolve_include(including, name, quoted, files, harness)`: the
  build-directory path the include reads, or None (Includes).
- `mentions(tokens)`: the identifier tokens of the expression `tokens` that
  count as mentions (Expressions), in order, before name resolution and
  before the call rules that need function summaries (lassi.toolchains.
  _cxx_flow applies those).
- `base(tokens)`: the identifier token that is the expression's base
  (Expressions), or None.
- `METADATA`: the member names whose object an expression does not mention.

Structure, scopes, declarations, and name resolution are tested through the
guard's readings in test_ttmetal_guard.py, where their effect on a reading
is visible. Every text here is SYNTHETIC; nothing is compiled or run. No value
in this module is a measurement.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any

import pytest

SCAN_MODULE = "lassi.toolchains._cxx_scan"
METADATA = frozenset(
    {"name", "dtype", "rank", "dims", "count", "nbytes", "size", "empty", "capacity", "length", "shape"}
)


def scan() -> ModuleType:
    """Return lassi.toolchains._cxx_scan; fail the test clearly while it, or a name these tests use, is missing."""
    try:
        module = importlib.import_module(SCAN_MODULE)
    except ModuleNotFoundError as error:
        pytest.fail(f"{SCAN_MODULE} does not exist ({error}); task P4.12 adds it")
    for name in ("lex", "Unreadable", "include_names", "resolve_include", "mentions", "base", "METADATA"):
        if not hasattr(module, name):
            pytest.fail(f"{SCAN_MODULE} has no {name}; see this module's docstring for the surface these tests use")
    return module


def lex(text: str, path: str = "main.cpp") -> Any:
    """Return the module's lexing of `text` as the file `path`."""
    return scan().lex(text, path)


def texts(tokens: Sequence[Any]) -> list[str]:
    """Return the text of each token."""
    return [token.text for token in tokens]


def code(text: str) -> list[str]:
    """Return the text of each code token of `text`."""
    return texts(lex(text).tokens)


def lines(text: str) -> list[tuple[str, int]]:
    """Return (text, line) of each code token of `text`."""
    return [(token.text, token.line) for token in lex(text).tokens]


def unreadable_reason(text: str, path: str = "main.cpp") -> str:
    """Return the reason the module refuses `text` as unreadable; fail when it reads it."""
    module = scan()
    with pytest.raises(module.Unreadable) as caught:
        module.lex(text, path)
    return str(getattr(caught.value, "reason", caught.value))


def mention_texts(expression: str) -> list[str]:
    """Return the text of each identifier `mentions` counts in `expression`."""
    return texts(scan().mentions(lex(expression).tokens))


def base_text(expression: str) -> str | None:
    """Return the text of `base` of `expression`, or None when it has none."""
    token = scan().base(lex(expression).tokens)
    return None if token is None else token.text


def resolve(including: str, name: str, quoted: bool, files: Mapping[str, str], harness: Mapping[str, str]) -> Any:
    """Return the module's resolution of one include."""
    return scan().resolve_include(including, name, quoted, files, harness)


# ---------------------------------------------------------------------------
# The lexer (Lexing)


def test_identifiers_numbers_and_punctuators_by_longest_match() -> None:
    assert code("int x = a->b >>= c::d ... e <=> f;") == [
        "int", "x", "=", "a", "->", "b", ">>=", "c", "::", "d", "...", "e", "<=>", "f", ";",
    ]


def test_identifiers_may_hold_a_dollar_and_an_underscore() -> None:
    assert code("_a $b c$1 __x") == ["_a", "$b", "c$1", "__x"]


def test_pp_numbers_keep_digit_separators_exponents_and_suffixes() -> None:
    assert code("1'000'000 1.5e-3f 0x1p+4 2'5u 0b1010'0101ULL .5e+2") == [
        "1'000'000", "1.5e-3f", "0x1p+4", "2'5u", "0b1010'0101ULL", ".5e+2",
    ]


def test_comments_are_dropped_and_lines_kept() -> None:
    assert lines("a // one\n/* two\nthree */ b\nc") == [("a", 1), ("b", 3), ("c", 4)]


def test_crlf_and_a_lone_cr_end_a_line() -> None:
    assert lines("a\r\nb\rc\nd") == [("a", 1), ("b", 2), ("c", 3), ("d", 4)]


def test_a_backslash_newline_is_spliced_and_each_token_keeps_its_first_line() -> None:
    assert lines("ab\\\ncd e\nf") == [("abcd", 1), ("e", 2), ("f", 3)]
    assert lines("x // a comment \\\ncontinued\ny") == [("x", 1), ("y", 3)]


@pytest.mark.parametrize(
    "literal",
    [
        '"a \\" b"',
        'u8"bytes"',
        'u"wide"',
        'U"wider"',
        'L"long"',
        "'c'",
        "'\\''",
        "L'w'",
        'R"(plain raw)"',
        'R"d(a )" b)d"',
        'u8R"x(raw " with ) parts)x"',
        'R"0123456789abcdef(sixteen)0123456789abcdef"',
    ],
)
def test_each_literal_is_one_token(literal: str) -> None:
    tokens = lex(f"x = {literal};").tokens
    assert texts(tokens) == ["x", "=", literal, ";"]
    assert tokens[2].kind == "LITERAL"


def test_a_raw_string_can_span_lines_and_the_next_token_keeps_its_line() -> None:
    assert lines('a = R"(one\ntwo)";\nb') == [("a", 1), ("=", 1), ('R"(one\ntwo)"', 1), (";", 2), ("b", 3)]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("a /* never closed", "an unterminated comment in main.cpp"),
        ('a = "never closed;\nb;', "an unterminated literal in main.cpp"),
        ("a = 'x;\nb;", "an unterminated literal in main.cpp"),
        ('a = R"d(never closed', "an unterminated literal in main.cpp"),
        ('a = R"d(closed by another delimiter)e";', "an unterminated literal in main.cpp"),
    ],
    ids=["comment", "string", "char", "raw-string", "raw-delimiter"],
)
def test_unterminated_comments_and_literals_are_unreadable(text: str, reason: str) -> None:
    assert unreadable_reason(text) == reason


def test_the_unreadable_reason_names_the_path_it_was_given() -> None:
    assert unreadable_reason("/* open", "kernels/ops.h") == "an unterminated comment in kernels/ops.h"


def test_a_non_ascii_character_outside_literals_is_an_other_token() -> None:
    tokens = lex('int gr\u00f6\u00dfe = 1; s = "\u00e9t\u00e9";').tokens
    others = [token.text for token in tokens if token.kind == "OTHER"]
    assert others == ["\u00f6", "\u00df"]
    assert '"\u00e9t\u00e9"' in texts(tokens), "a literal keeps its non-ASCII text"
    assert "gr" in texts(tokens) and "e" in texts(tokens)


def test_digraphs_are_not_mapped() -> None:
    assert "{" not in code("<% x %>") and "#" not in code("x %: y")


# ---------------------------------------------------------------------------
# Directives (Lexing)


def test_directive_lines_are_split_from_code() -> None:
    lexed = lex("#include <a.h>\nint x;\n  #  define A B\n/*\n#define C D\n*/ y;\n")
    assert texts(lexed.tokens) == ["int", "x", ";", "y", ";"]
    assert [texts(item.tokens)[0] for item in lexed.directives] == ["#", "#"]
    assert dict(lexed.aliases) == {"A": "B"}, "a # inside a block comment starts no directive"


def test_if_0_and_if_false_regions_are_dropped_and_the_else_branch_kept() -> None:
    text = (
        "a\n#if 0\nb\n#ifdef X\nc\n#endif\nd\n#else\ne\n#endif\n"
        "f\n#if false\ng\n#elif 1\nh\n#endif\n"
        "#ifdef Y\ni\n#else\nj\n#endif\n"
        "#if 1\nk\n#else\nl\n#endif\n"
    )
    assert code(text) == ["a", "e", "f", "h", "i", "j", "k", "l"], "no conditional but #if 0 and #if false is evaluated"


def test_a_quote_in_a_directive_line_or_a_dropped_region_only_ends_the_line() -> None:
    assert code("#error it's not code\nint x;\n#if 0\nthe kernel's prose\n#endif\ny;\n") == ["int", "x", ";", "y", ";"]
    reason = unreadable_reason("#if 0\n/* never closed\n#endif\n")
    assert reason == "an unterminated comment in main.cpp", "an unterminated block comment is unreadable anywhere"


def test_tokens_after_a_dropped_region_keep_their_lines() -> None:
    assert lines("a\n#if 0\nb\nc\n#endif\nd\n") == [("a", 1), ("d", 6)]


def test_macro_aliases_are_single_identifier_bodies_only() -> None:
    text = "#define A B\n#define C(x) x\n#define D 1\n#define E F G\n#define H\n#define I \\\n    J\n"
    assert dict(lex(text).aliases) == {"A": "B", "I": "J"}


# ---------------------------------------------------------------------------
# Includes (Includes)


def test_include_names_in_both_forms_outside_dropped_regions() -> None:
    text = '#include "a/b.h"\n#include <c/d.hpp>\n#  include   "e.h"\n#if 0\n#include "f.h"\n#endif\nint x;\n'
    assert scan().include_names(lex(text)) == [("a/b.h", True), ("c/d.hpp", False), ("e.h", True)]


def test_a_quoted_include_resolves_beside_the_including_file_first() -> None:
    files = {"host/util.h": "", "util.h": ""}
    assert resolve("host/main.cpp", "util.h", True, files, {}) == "host/util.h"
    assert resolve("host/main.cpp", "util.h", False, files, {}) == "util.h"
    assert resolve("host/main.cpp", "../util.h", True, {"util.h": ""}, {}) == "util.h"
    assert resolve("main.cpp", "./util.h", True, {"util.h": ""}, {}) == "util.h"


def test_an_include_resolves_to_a_harness_file_and_otherwise_to_nothing() -> None:
    assert resolve("main.cpp", "lassi_io.h", True, {}, {"lassi_io.h": ""}) == "lassi_io.h"
    assert resolve("main.cpp", "tt-metalium/host_api.hpp", False, {}, {}) is None, "pinned headers are not read"
    assert resolve("main.cpp", "../outside.h", True, {}, {"outside.h": ""}) is None, "it leaves the build directory"


# ---------------------------------------------------------------------------
# mentions and base (Expressions)


def test_the_metadata_names_are_the_documented_set() -> None:
    assert frozenset(scan().METADATA) == METADATA


def test_mentions_skip_callees_members_qualifiers_and_subscripts() -> None:
    assert mention_texts("f(x) + y.z + p->q + ns::w + v[i + j] + t<4>(u) + w::r") == ["x", "y", "p", "v", "u"]


def test_mentions_skip_keywords_but_keep_this() -> None:
    assert mention_texts("true + false + nullptr + this->m + a") == ["this", "a"]


def test_mentions_skip_sizeof_alignof_decltype_typeid_noexcept_and_lambda_introducers() -> None:
    expression = (
        "a + sizeof(b) + alignof(c) + decltype(d)() + typeid(e).hash_code() + noexcept(g) + [h, &k]() { return 0; }"
    )
    assert mention_texts(expression) == ["a"]


def test_mentions_skip_the_object_of_a_metadata_member() -> None:
    expression = "a.count + b->dims + c.size() + d.data + e.nbytes + f.shape + g.name + h.rank + k.empty()"
    assert mention_texts(expression) == ["d"]


def test_mentions_read_the_arguments_of_calls_and_casts() -> None:
    assert mention_texts("g(a, b.c, h<4>(d)) + static_cast<float>(e) + (float)f") == ["a", "b", "d", "e", "f"]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("c", "c"),
        ("(c)", "c"),
        ("c.data()", "c"),
        ("&c[0]", "c"),
        ("c.begin() + k", "c"),
        ("(float*)c.data", "c"),
        ("*(c + i)", "c"),
        ("(*c)[i]", "c"),
        ("static_cast<float*>(p)", "p"),
        ("reinterpret_cast<const char*>(q.data())", "q"),
        ("const_cast<float*>(r)", "r"),
        ("std::bit_cast<float>(z)", "z"),
        ("std::data(v)", "v"),
        ("std::begin(w)", "w"),
        ("std::end(w2)", "w2"),
        ("std::cbegin(w3)", "w3"),
        ("std::addressof(x)", "x"),
        ("this->c[i]", "this"),
        ("a.count", "a"),
        ("(const float*)(a.data)", "a"),
    ],
)
def test_base_strips_casts_address_dereference_parentheses_and_std_helpers(expression: str, expected: str) -> None:
    assert base_text(expression) == expected


@pytest.mark.parametrize("expression", ["new float[4]", "malloc(16)", "calloc(4, 4)", "make_buffer(n)", "1.0f", ""])
def test_allocations_calls_and_constants_have_no_base(expression: str) -> None:
    assert base_text(expression) is None
