"""Tests for the Python-subset frontend's refusals: every construct outside the subset (task P12.4).

Bible: Frontend Rules (anything outside a published subset fails with a readable diagnostic that is counted in
corpus statistics), Component Interfaces (Frontend; diagnostics records), Result Record (Diagnostic);
plans/p12-frontends.md, task P12.4 and its planning decision "Refusals (P12.4)".

The contract these tests fix:

- A source outside the subset raises lassi.core.interfaces.FrontendRefusal, the one refusal every frontend
  raises (tests/core/test_frontend_refusal.py), and nothing else. Its diagnostics hold one Diagnostic per
  construct outside the subset, in source order: stage "parse", severity "error", file the source's path as
  given to raise_source, line the construct's 1-based line, column its ast col_offset plus 1, a code naming
  its refusal class, and a message of one line of printable ASCII that names the construct and says it is
  outside the subset (a syntax error's says why the text does not parse, and its column is the parser's).
- A construct is reported at its own node: a statement at the statement; an expression, a call, or a
  subscript at that node; a name bound with another type at that binding; a scalar that a prange body binds
  and code outside it also uses at its first binding inside the body. A refused statement's parts are not
  read again, so one construct gives one diagnostic.
- The refusal classes and their codes, one test each (the module docstring of lassi.frontends.python_subset
  publishes every code):

  | class | code | example here |
  | --- | --- | --- |
  | syntax | subset-syntax | text that does not parse as Python 3.10 |
  | import | subset-import | an import other than the subset's (import os) |
  | module | subset-module | a module-level statement but a docstring, an import, a function, or the entry guard |
  | statement | subset-statement | a statement kind outside the subset (try) |
  | control | subset-control | break, continue, an early return, a loop else, a loop over anything but range or prange |
  | expression | subset-expression | an expression or operator outside the subset (x ** 2) |
  | call | subset-call | a call of anything but the module's functions and the subset's builtins (print) |
  | type | subset-type | a type the subset cannot settle statically (a name rebound with another type) |
  | array | subset-array | an array use outside the subset (a two-dimensional index) |
  | io | subset-io | a binding call outside the subset's form (a literal output path) |
  | prange | subset-prange | a scalar shared between a prange body and code outside it |

Every source here is SYNTHETIC test data written for this task. No value in this module is a measurement.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, NamedTuple

import pytest
from python_subset_fixtures import frontend_refusal, python_subset_module, raise_text

from lassi.core.record import Diagnostic

PATH = "main.py"


class Refusal(NamedTuple):
    """One refusal class: a source with one construct outside the subset and the diagnostic it must give."""

    source: str
    code: str
    line: int
    column: int | None  # None: the parser's column (a syntax error)
    words: tuple[str, ...]  # at least one names the construct in the message, as a whole word


REFUSALS = {
    "syntax": Refusal(
        '"""A source that does not parse."""\n\n\ndef f(x: float) -> float:\n    return x +\n',
        "subset-syntax", 5, None, ("syntax",),
    ),
    "import": Refusal(
        '"""Imports outside the subset."""\n\nimport os\n\n\ndef f(x: float) -> float:\n    return x\n',
        "subset-import", 3, 1, ("os",),
    ),
    "module": Refusal(
        '"""A module-level statement."""\n\nSIZE = 521\n\n\ndef f(x: float) -> float:\n    return x\n',
        "subset-module", 3, 1, ("module", "SIZE"),
    ),
    "statement": Refusal(
        '"""A try statement."""\n\n\ndef f(x: float) -> float:\n    y = 0.0\n    try:\n        y = 1.0 / x\n'
        "    except ZeroDivisionError:\n        y = 0.0\n    return y\n",
        "subset-statement", 6, 5, ("try",),
    ),
    "control": Refusal(
        '"""A break."""\n\n\ndef f(n: int) -> int:\n    k = 0\n    while k < n:\n        k += 1\n'
        "        if k == 3:\n            break\n    return k\n",
        "subset-control", 9, 13, ("break",),
    ),
    "expression": Refusal(
        '"""A power operator."""\n\n\ndef f(x: float) -> float:\n    return x ** 2\n',
        "subset-expression", 5, 12, ("**", "power"),
    ),
    "call": Refusal(
        '"""A print call."""\n\n\ndef f(x: float) -> float:\n    print(x)\n    return x\n',
        "subset-call", 5, 5, ("print",),
    ),
    "type": Refusal(
        '"""A name rebound with another type."""\n\n\ndef f(n: int) -> float:\n    total = 0\n'
        "    for i in range(n):\n        total = total + 0.5\n    return total\n",
        "subset-type", 7, 9, ("total",),
    ),
    "array": Refusal(
        '"""A two-dimensional index."""\n\nfrom typing import Annotated\n\n\n'
        'def f(x: Annotated[memoryview, "f64"], n: int) -> None:\n    x[n, 0] = 1.0\n',
        "subset-array", 7, 5, ("index", "subscript"),
    ),
    "io": Refusal(
        '"""A literal output path."""\n\nfrom lassi_io import write_array\n\n\ndef main() -> None:\n'
        '    y = memoryview(bytearray(1 * 8)).cast("d")\n    write_array("y.lassiio", "y", "f64", (1,), y)\n',
        "subset-io", 8, 5, ("write_array",),
    ),
    "prange": Refusal(
        '"""A scalar shared across prange iterations."""\n\nfrom typing import Annotated\n\n'
        "from lassi_io import prange\n\n\n"
        'def f(n: int, x: Annotated[memoryview, "f64"]) -> float:\n    total = 0.0\n    for i in prange(n):\n'
        "        total += x[i]\n    return total\n",
        "subset-prange", 11, 9, ("total",),
    ),
}

# Two independent constructs outside the subset: a print call (line 5) and a break (line 10).
TWO_REFUSALS = (
    '"""Two constructs outside the subset."""\n\n\ndef f(n: int) -> int:\n    print(n)\n    k = 0\n'
    "    while k < n:\n        k += 1\n        if k == 3:\n            break\n    return k\n"
)


def refused(text: str) -> tuple[Diagnostic, ...]:
    """Raise `text` as main.py and return the diagnostics of the FrontendRefusal it must raise."""
    refusal = frontend_refusal()
    with pytest.raises(refusal) as caught:
        raise_text(text, PATH)
    diagnostics = caught.value.diagnostics
    assert isinstance(diagnostics, tuple) and diagnostics, diagnostics
    assert all(isinstance(item, Diagnostic) for item in diagnostics), diagnostics
    return diagnostics


def readable(message: Any) -> bool:
    """Return True for one non-empty line of printable ASCII."""
    return isinstance(message, str) and bool(message.strip()) and all(" " <= char <= "~" for char in message)


def names_one_of(message: str, words: tuple[str, ...]) -> bool:
    """Return True when the message holds one of `words`, as a whole word for an identifier."""
    for word in words:
        pattern = rf"(?<![\w.]){re.escape(word)}(?!\w)" if word.isidentifier() else re.escape(word)
        if re.search(pattern, message, flags=re.IGNORECASE):
            return True
    return False


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_each_refusal_class_raises_the_shared_refusal(name: str) -> None:
    case = REFUSALS[name]
    (diagnostic,) = refused(case.source)
    assert (diagnostic.stage, diagnostic.severity, diagnostic.code) == ("parse", "error", case.code), diagnostic
    assert diagnostic.file == PATH, diagnostic
    assert diagnostic.line == case.line, diagnostic
    if case.column is None:
        assert isinstance(diagnostic.column, int) and diagnostic.column >= 1, diagnostic
    else:
        assert diagnostic.column == case.column, diagnostic
    assert readable(diagnostic.message), diagnostic.message
    assert names_one_of(diagnostic.message, case.words), (diagnostic.message, case.words)
    expected = "syntax" if name == "syntax" else "subset"
    assert expected in diagnostic.message.lower(), f"the message says why: {diagnostic.message!r}"


def test_every_construct_outside_the_subset_is_counted() -> None:
    diagnostics = refused(TWO_REFUSALS)
    found = [(item.code, item.line, item.column) for item in diagnostics]
    assert found == [("subset-call", 5, 5), ("subset-control", 10, 13)], found


def test_the_refusal_message_names_each_place() -> None:
    refusal = frontend_refusal()
    with pytest.raises(refusal) as caught:
        raise_text(TWO_REFUSALS, PATH)
    message = str(caught.value)
    assert "main.py:5:5" in message and "main.py:10:13" in message, message


def test_the_refusal_codes_are_distinct_and_published() -> None:
    codes = [case.code for case in REFUSALS.values()]
    assert len(set(codes)) == len(codes), codes
    doc = inspect.getdoc(python_subset_module()) or ""
    missing = [code for code in codes if code not in doc]
    assert not missing, f"the module docstring does not publish {missing}"
