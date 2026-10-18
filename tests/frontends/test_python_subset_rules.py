"""Tests for the Python-subset frontend's edges: its raise contract, parse guards, and subset rules (task P12.4).

Bible: Frontend Rules (the published subset; anything outside it fails with a readable diagnostic that is counted),
Component Interfaces (Frontend); plans/p12-frontends.md, task P12.4 and its planning decisions "Refusals (P12.4)",
"Harness I/O in raised programs", and prange.

The contract these tests fix, beside tests/frontends/test_python_subset.py and test_python_subset_refusals.py:

- raise_source takes no flags (a ValueError naming how many were given, an iterator of flags included), at least
  one file (a ValueError), and str paths and texts (a TypeError); two or more files are one subset-module refusal.
- A logical line of more than MAX_LINE_TOKENS tokens is a subset-syntax refusal at its first line, column 1,
  before the parser reads it; a null byte (ValueError) and a parser MemoryError are subset-syntax refusals at line
  1, column 1; nesting too deep for the frontend's recursion is one subset-expression refusal with no line.
- Every int literal fits in 64 bits: a range step past 2**63 - 1 is subset-type, an argv index past it subset-io.
- One cause is counted once: a later use of a name whose binding was refused, in the function or at module level
  (an import, an assignment, a class, a function the subset cannot take), gives no diagnostic of its own.
- prange: a prange inside a prange body, a prange body's names used outside the loop, a scalar the body reads
  before the same iteration binds it, and write_array in the body are subset-prange; another for loop may reuse
  the name of a prange body's scalar as its own loop variable, unless that loop holds the prange loop.
- NumPy: a call of a module function that takes an array, after an ndarray in a vectorized expression, is
  subset-array, since NumPy reads that ndarray before the call and the hub module after it.
- The entry: the guard calls main, and main takes no argument and returns None (subset-module); no function takes
  a C library name the hub's lowering can call (malloc, free, exp, log, sin, cos, sqrt).
- Messages read as English: the article fits the statement, a comparison operator is quoted, np.sqrt's message
  holds inside and outside z[:] = <expression>, and an alias bound before its array names that use.

Every source here is SYNTHETIC test data written for this task. No value in this module is a measurement.
"""

from __future__ import annotations

import ast
from textwrap import dedent
from typing import Any

import pytest
from python_subset_fixtures import frontend_refusal, python_subset, python_subset_module, raise_text

from lassi.core.record import Diagnostic

PATH = "main.py"
SCALAR = "def f(x: int) -> int:\n    return x\n"
PRANGE_HEAD = 'from typing import Annotated\n\nfrom lassi_io import prange\n\n\n'
PRANGE_DEF = 'def f(n: int, a: Annotated[memoryview, "f64"], out: Annotated[memoryview, "f64"]) -> None:\n'


def refused(text: str) -> tuple[Diagnostic, ...]:
    """Raise `text` as main.py and return the diagnostics of the FrontendRefusal it must raise."""
    with pytest.raises(frontend_refusal()) as caught:
        raise_text(text, PATH)
    return caught.value.diagnostics


def only(text: str) -> Diagnostic:
    """Return the one diagnostic a refused `text` gives, failing when it gives another count."""
    diagnostics = refused(text)
    assert len(diagnostics) == 1, [(item.code, item.line, item.column, item.message) for item in diagnostics]
    return diagnostics[0]


def place(diagnostic: Diagnostic) -> tuple[Any, ...]:
    """Return a diagnostic's code, line, and column."""
    return (diagnostic.code, diagnostic.line, diagnostic.column)


def prange_body(body: str) -> str:
    """Return a function f(n, a, out) whose body (indented four blanks) starts on line 7, after prange's import."""
    return PRANGE_HEAD + PRANGE_DEF + "".join(f"    {line}\n" for line in dedent(body).splitlines())


# ---------------------------------------------------------------------------
# The raise contract


def test_a_flag_is_a_value_error_naming_how_many() -> None:
    frontend = python_subset()
    with pytest.raises(ValueError, match="2 were given"):
        frontend.raise_source({PATH: SCALAR}, iter(["-O2", "-g"]))
    with pytest.raises(ValueError, match="1 was given"):
        frontend.raise_source({PATH: SCALAR}, ["-O2"])


def test_no_file_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="none was given"):
        python_subset().raise_source({}, [])


@pytest.mark.parametrize("files", [{1: SCALAR}, {PATH: SCALAR.encode("ascii")}, {PATH: None}])
def test_a_path_or_text_that_is_not_a_str_is_a_type_error(files: dict[Any, Any]) -> None:
    with pytest.raises(TypeError, match="str"):
        python_subset().raise_source(files, [])


def test_two_files_are_one_module_refusal() -> None:
    with pytest.raises(frontend_refusal()) as caught:
        python_subset().raise_source({"a.py": SCALAR, "b.py": SCALAR}, [])
    (diagnostic,) = caught.value.diagnostics
    assert (diagnostic.stage, diagnostic.code) == ("parse", "subset-module"), diagnostic
    assert "a.py" in diagnostic.message and "b.py" in diagnostic.message, diagnostic.message


# ---------------------------------------------------------------------------
# Parse guards


def test_a_long_logical_line_is_refused_before_the_parser() -> None:
    limit = python_subset_module().MAX_LINE_TOKENS
    terms = " + ".join(["x"] * (limit // 2 + 2))
    diagnostic = only(f"def f(x: int) -> int:\n    return {terms}\n")
    assert place(diagnostic) == ("subset-syntax", 2, 1), diagnostic
    assert str(limit) in diagnostic.message, diagnostic.message


def test_a_line_under_the_limit_raises() -> None:
    terms = " + ".join(["x"] * 50)
    assert raise_text(f"def f(x: int) -> int:\n    return {terms}\n", PATH).tag == "structured"


def test_nesting_too_deep_is_one_expression_refusal() -> None:
    diagnostic = only("def f(x: int) -> int:\n    return " + "-" * 1500 + "x\n")
    assert (diagnostic.code, diagnostic.file, diagnostic.line) == ("subset-expression", PATH, None), diagnostic
    assert "nested too deeply" in diagnostic.message, diagnostic.message


def test_a_null_byte_is_a_syntax_refusal() -> None:
    diagnostic = only("def f(x: int) -> int:\n    return x\x00\n")
    assert place(diagnostic) == ("subset-syntax", 1, 1), diagnostic
    assert all(" " <= char <= "~" for char in diagnostic.message), diagnostic.message


def test_a_parser_memory_error_is_a_syntax_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    def exhausted(*args: object, **kwargs: object) -> ast.Module:
        """Stand in for ast.parse running out of memory."""
        raise MemoryError

    monkeypatch.setattr(python_subset_module().ast, "parse", exhausted)
    diagnostic = only(SCALAR)
    assert place(diagnostic) == ("subset-syntax", 1, 1), diagnostic
    assert "MemoryError" in diagnostic.message, diagnostic.message


# ---------------------------------------------------------------------------
# Int literals fit in 64 bits


def test_a_range_step_past_64_bits_is_a_type_refusal() -> None:
    diagnostic = only("def f(n: int) -> None:\n    for i in range(0, n, 9223372036854775808):\n        pass\n")
    assert place(diagnostic) == ("subset-type", 2, 26), diagnostic


def test_the_largest_64_bit_step_raises() -> None:
    module = raise_text("def f(n: int) -> None:\n    for i in range(0, n, 9223372036854775807):\n        pass\n", PATH)
    assert "arith.constant 9223372036854775807 : index" in module.text


@pytest.mark.parametrize("index", ["9223372036854775808", "99999999999999999999"])
def test_an_argv_index_past_64_bits_is_an_io_refusal(index: str) -> None:
    text = dedent(
        f"""\
        import sys
        from typing import Annotated

        from lassi_io import read_array, write_array


        def main() -> None:
            x: Annotated[memoryview, "f64"] = read_array(sys.argv[{index}]).data
            write_array(sys.argv[{index}], "x", "f64", (1,), x)
        """
    )
    found = [place(item) for item in refused(text)]
    assert found == [("subset-io", 8, 39), ("subset-io", 9, 5)], found


# ---------------------------------------------------------------------------
# One cause, counted once

ONE_CAUSE = {
    "a refused local, then its uses": (
        """\
        def f(x: float) -> float:
            y = x ** 2
            z = y + 1.0
            return z
        """,
        ("subset-expression", 2, 9),
    ),
    "a lambda, then its call": (
        """\
        def f(x: int) -> int:
            g = lambda y: y + 1
            z = g(x)
            return z
        """,
        ("subset-expression", 2, 9),
    ),
    "a nested def, then its call": (
        """\
        def f(x: int) -> int:
            def g(y: int) -> int:
                return y
            return g(x)
        """,
        ("subset-statement", 2, 5),
    ),
    "a module-level class, then its call": (
        """\
        class P:
            pass


        def f() -> None:
            P()
        """,
        ("subset-module", 1, 1),
    ),
    "a module-level constant, then its use": (
        """\
        N = 10


        def f(x: int) -> int:
            return x + N
        """,
        ("subset-module", 1, 1),
    ),
    "import numpy, then two calls through it": (
        """\
        import numpy
        from typing import Annotated


        def f(xs: Annotated[memoryview, "f64"]) -> None:
            x = numpy.asarray(xs)
            y = numpy.asarray(xs)
        """,
        ("subset-import", 1, 1),
    ),
    "from math import sqrt, then its call": (
        """\
        from math import sqrt


        def f(x: float) -> float:
            return sqrt(x)
        """,
        ("subset-import", 1, 1),
    ),
    "a refused binding import, then a read": (
        """\
        import sys
        from typing import Annotated

        from lassi_io import read_array, open_array


        def main() -> None:
            x: Annotated[memoryview, "f64"] = read_array(sys.argv[1]).data
        """,
        ("subset-import", 4, 1),
    ),
    "a refused typing import, then an array annotation": (
        """\
        from typing import Annotated, List


        def f(x: Annotated[memoryview, "f64"]) -> None:
            x[0] = 1.0
        """,
        ("subset-import", 1, 1),
    ),
    "a function name the subset cannot take, then its call": (
        """\
        def lassi_io_read_f64(k: int) -> int:
            return k


        def f() -> int:
            return lassi_io_read_f64(1)
        """,
        ("subset-module", 1, 1),
    ),
}


@pytest.mark.parametrize("name", sorted(ONE_CAUSE))
def test_a_refused_binding_is_counted_once(name: str) -> None:
    text, expected = ONE_CAUSE[name]
    assert place(only(dedent(text))) == expected


# ---------------------------------------------------------------------------
# prange


def test_prange_inside_a_prange_body_is_refused() -> None:
    text = prange_body(
        """\
        for i in prange(n):
            for j in prange(n):
                out[j] = 0.0
        """
    )
    assert place(only(text)) == ("subset-prange", 8, 9)


def test_a_prange_loop_variable_used_after_its_loop_is_refused() -> None:
    text = "from lassi_io import prange\n\n\ndef g(n: int) -> int:\n    for i in prange(n):\n        pass\n"
    assert place(only(text + "    return i\n")) == ("subset-prange", 5, 9)


READ_BEFORE_BINDING = {
    "a binding in one branch, then a read": (
        """\
        for i in prange(n):
            if i == 0:
                s = 0.0
            s = s + a[i]
            out[i] = s
        """,
        ("subset-prange", 10, 13),
    ),
    "a read in a branch, then the binding": (
        """\
        for i in prange(n):
            if i > 0:
                out[i] = s
            s = a[i]
        """,
        ("subset-prange", 9, 22),
    ),
    "a binding in one branch, then an update": (
        """\
        for i in prange(n):
            if i == 0:
                s = 0.0
            s += 1.0
            out[i] = s
        """,
        ("subset-prange", 10, 9),
    ),
    "a binding inside an inner loop, then a read after it": (
        """\
        for i in prange(n):
            for j in range(n):
                t = a[j]
            out[i] = t
        """,
        ("subset-prange", 10, 18),
    ),
}


@pytest.mark.parametrize("name", sorted(READ_BEFORE_BINDING))
def test_a_prange_scalar_read_before_the_iteration_binds_it_is_refused(name: str) -> None:
    body, expected = READ_BEFORE_BINDING[name]
    diagnostic = only(prange_body(body))
    assert place(diagnostic) == expected, diagnostic
    assert "before" in diagnostic.message, diagnostic.message


def test_a_prange_scalar_bound_on_both_branches_raises() -> None:
    text = prange_body(
        """\
        for i in prange(n):
            if i == 0:
                s = 1.0
            else:
                s = 2.0
            out[i] = s
        """
    )
    assert "scf.parallel" in raise_text(text, PATH).text


def test_a_range_loop_after_a_prange_loop_may_reuse_its_variable() -> None:
    text = prange_body(
        """\
        for i in prange(n):
            out[i] = 1.0
        for i in range(n):
            out[i] = out[i] + a[i]
        """
    )
    lines = raise_text(text, PATH).text.split("\n")
    slots = [line.strip() for line in lines if line.strip().endswith("= memref.alloca() : memref<i64>")]
    assert len(slots) == 2, "one slot for the range loop's i at entry, one private slot in the prange body"


def test_a_loop_that_holds_the_prange_loop_does_not_cover_its_name() -> None:
    text = prange_body(
        """\
        for i in range(n):
            for k in prange(n):
                i = k
            out[i] = 1.0
        """
    )
    assert place(only(text)) == ("subset-prange", 9, 13)


def test_write_array_in_a_prange_body_is_refused() -> None:
    text = dedent(
        """\
        import sys
        from typing import Annotated

        from lassi_io import prange, write_array


        def f(n: int, a: Annotated[memoryview, "f64"]) -> None:
            for i in prange(n):
                write_array(sys.argv[1], "a", "f64", (n,), a)
        """
    )
    assert place(only(text)) == ("subset-prange", 9, 9)


# ---------------------------------------------------------------------------
# NumPy evaluation order

VECTOR_HEAD = dedent(
    """\
    from typing import Annotated

    import numpy as np


    def bump(xs: Annotated[memoryview, "f64"]) -> float:
        xs[0] = 100.0
        return 1.0


    def f(xs: Annotated[memoryview, "f64"], zs: Annotated[memoryview, "f64"]) -> None:
        x = np.asarray(xs)
        z = np.asarray(zs)
    """
)


def test_a_call_that_takes_an_array_after_an_ndarray_is_refused() -> None:
    diagnostic = only(VECTOR_HEAD + "    z[:] = x * 2.0 + bump(xs)\n")
    assert place(diagnostic) == ("subset-array", 14, 22), diagnostic
    assert "bump" in diagnostic.message, diagnostic.message


def test_a_call_that_takes_an_array_before_every_ndarray_raises() -> None:
    text = raise_text(VECTOR_HEAD + "    z[:] = bump(xs) + x * 2.0\n", PATH).text
    assert text.index("@bump(%arg0)") < text.index("scf.parallel"), text


# ---------------------------------------------------------------------------
# The entry and the C library's names


def test_the_guard_calls_main() -> None:
    text = 'def run() -> None:\n    pass\n\n\nif __name__ == "__main__":\n    run()\n'
    assert place(only(text)) == ("subset-module", 5, 1)


@pytest.mark.parametrize(
    "text",
    [
        'def main() -> int:\n    return 3\n\n\nif __name__ == "__main__":\n    main()\n',
        "def main(n: int) -> None:\n    pass\n",
    ],
)
def test_main_takes_no_argument_and_returns_none(text: str) -> None:
    diagnostic = only(text)
    assert place(diagnostic) == ("subset-module", 1, 1), diagnostic
    assert "main" in diagnostic.message, diagnostic.message


@pytest.mark.parametrize("name", ["malloc", "free", "exp", "log", "sin", "cos", "sqrt"])
def test_a_function_takes_no_c_library_name_the_lowering_calls(name: str) -> None:
    diagnostic = only(f"def {name}(x: float) -> float:\n    return x\n")
    assert place(diagnostic) == ("subset-module", 1, 1), diagnostic
    assert name in diagnostic.message, diagnostic.message


# ---------------------------------------------------------------------------
# Messages


def test_a_statement_takes_the_article_its_word_needs() -> None:
    message = only("def f(x: int) -> int:\n    assert x > 0\n    return x\n").message
    assert "an assert statement" in message, message


def test_a_comparison_operator_is_quoted() -> None:
    message = only("def f(x: int, y: int) -> bool:\n    return x is y\n").message
    assert "'is'" in message and "is is" not in message, message


def test_np_sqrt_of_a_scalar_inside_a_vectorized_expression_reads_true() -> None:
    text = dedent(
        """\
        from typing import Annotated

        import numpy as np


        def f(xs: Annotated[memoryview, "f64"]) -> None:
            x = np.asarray(xs)
            x[:] = x + np.sqrt(2.0)
        """
    )
    message = only(text).message
    assert "outside z[:]" not in message and "ndarray" in message, message


def test_an_alias_bound_before_its_array_names_that_use() -> None:
    text = 'def f(n: int) -> None:\n    b = a\n    a = memoryview(bytearray(n * 8)).cast("d")\n    b[0] = 1.0\n'
    diagnostic = only(text)
    assert place(diagnostic) == ("subset-array", 2, 9), diagnostic
    assert "before its binding" in diagnostic.message, diagnostic.message
