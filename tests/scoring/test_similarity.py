"""Tests for Sim-T and Sim-L (task P1.8).

Bible: Source Papers (LASSI results table; quirk table, Sim-T row), Evaluation
Protocol (LASSI reproduction row), Repository Layout (`scoring/`).

These tests check `lassi.scoring.similarity`, which provides:

- `sim_t(reference, candidate)`: the faithful Sim-T. It equals upstream's
  `token_similarity(reference, candidate, "tokenize")`, where Python's
  `tokenize` runs over C/C++ text. The upstream quirk is kept (quirk table,
  Sim-T row).
- `sim_l(reference, candidate)`: the faithful Sim-L. It equals upstream's
  `compare_lines_with_reordering(reference, candidate)`.
- `sim_t_c(reference, candidate)`: the C-aware Sim-T, under its own name. It
  is the ratio 2M/T of matched tokens over tokens from `c_tokens`. The lexer's
  rules are in its docstring.
- `c_tokens(text)`: the documented C lexer. It is total, so it never raises.
- `measure(reference, candidate)`: a record with `sim_t`, `sim_l`, `sim_t_c`,
  and `python` (the interpreter version, `platform.python_version()`, as the
  runner records it). Python's tokenize changes between versions, so every
  written value carries the version (`to_dict()` is the written form).

The argument order follows upstream: the reference target first, the candidate
second. Sim-T need not be symmetric; Sim-L is (a multiset overlap of lines).

Oracle: upstream's own functions, taken from the pinned notebook cell with
`ast` and run in memory under the subprocess and socket guard (see
conftest.py). The fast suite checks one ordered pair per app and a set of edge
cases: empty text, and each tokenize error branch upstream catches (a
dedent error, EOF inside brackets or a triple-quoted string, an unknown
coding cookie). A `slow` test checks every ordered pair of the 20 `*_main`
files. Equality is exact, since both sides compute the same ratio of
integers.

The C-aware checks use hand-built inputs whose token lists and ratios follow
from the lexer rules and the 2M/T definition. No value here is a measurement.
Wiring the measures into metrics is P2's work.
"""

from __future__ import annotations

import importlib
import inspect
import json
import platform
import socket
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
APPS = (
    "atomicCost", "bsearch", "colorwheel", "dense-embedding", "entropy",
    "jacobi", "layout", "matrix-rotate", "pathfinder", "randomAccess",
)
LANGS = ("omp", "cuda")
MAIN_KEYS = [(app, lang) for app in APPS for lang in LANGS]


def similarity() -> types.ModuleType:
    """Import and return lassi.scoring.similarity (inside each test, so a missing module fails that test)."""
    return importlib.import_module("lassi.scoring.similarity")


# ---------------------------------------------------------------- edge cases
#
# (id, reference, candidate, upstream takes a tokenize error branch). The
# inputs are made up here and quote no upstream text.

EDGE_CASES = [
    ("both-empty", "", "", False),
    ("empty-reference", "", "int a;\n", False),
    ("empty-candidate", "int a;\n", "", False),
    ("blank-lines", "   \n\n", "\n", False),
    ("identical", "int a = 1;\n", "int a = 1;\n", False),
    ("shifted-one-line", "int a = 1;\n", "\nint a = 1;\n", False),
    ("reordered-lines", "int a;\nint b;\n", "int b;\nint a;\n", False),
    ("duplicate-unbalanced-braces", "}\n}\n}\n", "}\n", True),  # more closers than openers: EOF error
    ("no-final-newline", "int a;", "int a;\n", False),
    ("crlf-form-feed", "int a;\r\n\x0cint b;\r\n", "int a;\nint b;\n", False),
    ("apostrophe-in-comment", "// it's one\nint a;\n", "int a;\n", False),
    ("non-ascii-comment", "/* \u00a9 2024 */\nint a;\n", "int a;\n", False),
    ("utf8-bom", "\ufeffint a;\n", "int a;\n", False),
    ("bad-dedent", "    int a;\n  int b;\n", "int a;\nint b;\n", True),
    ("bad-dedent-candidate", "int a;\nint b;\n", "        x = 1;\n    y = 2;\n", True),
    ("eof-in-braces", "int main() {\n  return 0;\n", "int main() {\n  return 0;\n}\n", True),
    ("eof-in-triple-string", 'const char *s = """x\n', "const char *s;\n", True),
    ("unknown-coding-cookie", "#pragma coding: nonesuch\nint a;\n", "int a;\n", True),
]


@pytest.mark.parametrize(
    ("reference", "candidate", "error_branch"),
    [case[1:] for case in EDGE_CASES],
    ids=[case[0] for case in EDGE_CASES],
)
def test_faithful_sim_t_and_sim_l_match_upstream_on_edge_cases(
    upstream: Any, reference: str, candidate: str, error_branch: bool
) -> None:
    before = upstream.prints()
    want_t = upstream.sim_t(reference, candidate)
    took_error_branch = upstream.prints() > before
    assert took_error_branch == error_branch, "the edge case no longer reaches the tokenize branch it names"
    want_l = upstream.sim_l(reference, candidate)
    sim = similarity()
    assert sim.sim_t(reference, candidate) == want_t
    assert sim.sim_l(reference, candidate) == want_l


# ---------------------------------------------------------------- upstream sources


@pytest.mark.parametrize("app", APPS)
def test_faithful_sim_t_matches_upstream_on_one_pair_per_app(
    upstream: Any, upstream_mains: dict[tuple[str, str], str], app: str
) -> None:
    reference, candidate = upstream_mains[(app, "omp")], upstream_mains[(app, "cuda")]
    want = upstream.sim_t(reference, candidate)
    assert similarity().sim_t(reference, candidate) == want


@pytest.mark.parametrize("app", APPS)
def test_faithful_sim_l_matches_upstream_on_one_pair_per_app(
    upstream: Any, upstream_mains: dict[tuple[str, str], str], app: str
) -> None:
    reference, candidate = upstream_mains[(app, "omp")], upstream_mains[(app, "cuda")]
    want = upstream.sim_l(reference, candidate)
    assert similarity().sim_l(reference, candidate) == want


@pytest.mark.slow
@pytest.mark.parametrize("reference_key", MAIN_KEYS, ids=[f"{app}-{lang}" for app, lang in MAIN_KEYS])
def test_faithful_sim_t_and_sim_l_match_upstream_on_every_ordered_pair(
    upstream: Any, upstream_mains: dict[tuple[str, str], str], reference_key: tuple[str, str]
) -> None:
    sim = similarity()
    reference = upstream_mains[reference_key]
    mismatches: list[tuple[tuple[str, str], tuple[float, float], tuple[float, float]]] = []
    for candidate_key in MAIN_KEYS:
        candidate = upstream_mains[candidate_key]
        want = (upstream.sim_t(reference, candidate), upstream.sim_l(reference, candidate))
        got = (sim.sim_t(reference, candidate), sim.sim_l(reference, candidate))
        if got != want:
            mismatches.append((candidate_key, got, want))
    assert not mismatches, f"{reference_key} against: {mismatches}"


# ---------------------------------------------------------------- C-aware Sim-T

C_TOKEN_CASES = [
    ("declaration", "int a = 1;", ["int", "a", "=", "1", ";"]),
    ("layout", "a\n\t  b\r\n\x0c c", ["a", "b", "c"]),
    ("empty", "", []),
    ("comments", "a // it's\nb /* it's\n x */ c", ["a", "b", "c"]),
    ("operators", "x->y++ <<= 2 && z::w != v", ["x", "->", "y", "++", "<<=", "2", "&&", "z", "::", "w", "!=", "v"]),
    ("numbers", "1.5e-3f + 0x1Fu - 42UL", ["1.5e-3f", "+", "0x1Fu", "-", "42UL"]),
    ("string-literal", 'f("a \\"b\\" c")', ["f", "(", '"a \\"b\\" c"', ")"]),
    ("char-literal", "c = '\\'';", ["c", "=", "'\\''", ";"]),
    ("pragma", "#pragma omp parallel for", ["#", "pragma", "omp", "parallel", "for"]),
]


@pytest.mark.parametrize(("text", "tokens"), [case[1:] for case in C_TOKEN_CASES], ids=[c[0] for c in C_TOKEN_CASES])
def test_c_lexer_yields_the_documented_tokens(text: str, tokens: list[str]) -> None:
    assert similarity().c_tokens(text) == tokens


@pytest.mark.parametrize("text", ["/* open", '"open', "'", "a \\\n b", "\x00", "\u00a9", "}}}", "\"\"\"x\n"])
def test_c_lexer_never_raises(text: str) -> None:
    tokens = similarity().c_tokens(text)
    assert isinstance(tokens, list)
    assert all(isinstance(token, str) and token and not token.isspace() for token in tokens)


def test_c_lexer_is_documented() -> None:
    doc = inspect.getdoc(similarity().c_tokens) or ""
    assert len(doc.splitlines()) >= 3, "c_tokens needs a docstring that states the lexer rules"
    for word in ("comment", "literal", "operator"):
        assert word in doc.lower(), f"the lexer docstring does not say how it treats: {word}"


C_SIM_CASES = [
    ("identical", "int a = 1;", "int a = 1;", 1.0),
    ("one-token-differs", "int a = 1;", "int a = 2;", 0.8),
    ("layout-and-comment", "int a;", "int  a ;  // note\n", 1.0),
    ("shifted-one-line", "int a = 1;\n", "\nint a = 1;\n", 1.0),
    ("disjoint", "a", "b", 0.0),
    ("empty-reference", "", "a", 0.0),
]


@pytest.mark.parametrize(
    ("reference", "candidate", "value"), [case[1:] for case in C_SIM_CASES], ids=[c[0] for c in C_SIM_CASES]
)
def test_c_aware_sim_t_is_the_matched_token_ratio(reference: str, candidate: str, value: float) -> None:
    assert similarity().sim_t_c(reference, candidate) == value


def test_c_aware_sim_t_has_its_own_name() -> None:
    sim = similarity()
    assert callable(sim.sim_t_c)
    assert sim.sim_t_c is not sim.sim_t


@pytest.mark.parametrize("app", APPS)
def test_c_aware_sim_t_is_bounded_on_upstream_sources(upstream_mains: dict[tuple[str, str], str], app: str) -> None:
    sim = similarity()
    reference, candidate = upstream_mains[(app, "omp")], upstream_mains[(app, "cuda")]
    assert 0.0 <= sim.sim_t_c(reference, candidate) <= 1.0
    assert sim.sim_t_c(reference, reference) == 1.0


# ---------------------------------------------------------------- written values


def test_measure_records_every_value_with_the_interpreter_version() -> None:
    sim = similarity()
    reference, candidate = "int a = 1;\nint b;\n", "int b;\nint a = 2;\n"
    record = sim.measure(reference, candidate)
    assert record.sim_t == sim.sim_t(reference, candidate)
    assert record.sim_l == sim.sim_l(reference, candidate)
    assert record.sim_t_c == sim.sim_t_c(reference, candidate)
    assert record.python == platform.python_version()
    written = record.to_dict()
    assert json.loads(json.dumps(written)) == written
    assert written["python"] == platform.python_version()
    assert {"sim_t", "sim_l", "sim_t_c", "python"} <= set(written)
    assert (written["sim_t"], written["sim_l"], written["sim_t_c"]) == (record.sim_t, record.sim_l, record.sim_t_c)


# ---------------------------------------------------------------- re-implementation, not upstream


def scoring_sources() -> list[Path]:
    """Return the Python files of lassi/scoring (the task's shared code)."""
    return sorted((REPO / "lassi" / "scoring").glob("*.py"))


def test_scoring_package_reimplements_and_never_loads_upstream() -> None:
    similarity()  # the package must exist
    for name in ("__init__.py", "similarity.py"):
        assert (REPO / "lassi" / "scoring" / name).is_file(), f"lassi/scoring/{name} is missing"
    for path in scoring_sources():
        text = path.read_text(encoding="utf-8")
        for needle in ("third_party", ".ipynb", "exec(", "eval("):
            assert needle not in text, f"{path.name} must re-implement upstream, not load it: found {needle!r}"
        assert text.isascii(), f"{path.name} is not plain ASCII"


def test_scoring_code_and_tests_quote_no_upstream_text(upstream_texts: list[str]) -> None:
    similarity()  # the package must exist
    files = scoring_sources() + sorted(Path(__file__).parent.glob("*.py"))
    for path in files:
        text = path.read_text(encoding="utf-8")
        windows = {text[i:i + 40] for i in range(len(text) - 39)}
        for source in upstream_texts:
            hits = [source[i:i + 40] for i in range(len(source) - 39) if source[i:i + 40] in windows]
            assert not hits, f"{path.relative_to(REPO).as_posix()} repeats upstream text (OQ-018): {hits[0]!r}"


# ---------------------------------------------------------------- the guard itself


def _start_process() -> None:
    subprocess.run([sys.executable, "-c", "pass"], check=False)


def _open_socket() -> None:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()


def _resolve_name() -> None:
    socket.getaddrinfo("localhost", 80)


@pytest.mark.parametrize("attempt", [_start_process, _open_socket, _resolve_name], ids=["process", "socket", "lookup"])
def test_guard_fails_on_an_attempt_even_when_the_error_is_swallowed(guard: type, attempt: Callable[[], None]) -> None:
    originals = (subprocess.Popen, socket.socket, socket.getaddrinfo)
    with pytest.raises(pytest.fail.Exception, match="tried to start a process or open a socket"):
        with guard():
            try:
                attempt()
            except Exception:
                pass  # upstream's tokenize helper swallows errors this way
    assert (subprocess.Popen, socket.socket, socket.getaddrinfo) == originals, "the guard left a patch behind"


# ---------------------------------------------------------------- the upstream pin


def test_upstream_fixture_follows_the_pin_manifest(upstream_pin: Any, tmp_path: Path) -> None:
    manifest = yaml.safe_load((REPO / "assets" / "upstream" / "lassi.yaml").read_text(encoding="utf-8"))
    assert (upstream_pin.commit, upstream_pin.path) == (manifest["commit"], manifest["path"])
    with pytest.raises(pytest.skip.Exception) as skipped:
        upstream_pin.require(tmp_path / "absent")
    message = skipped.value.msg or ""
    assert "uv run tools/fetch_upstream.py" in message, "the skip reason must name the fetch tool"
    assert manifest["commit"][:7] in message
    assert "submodule" not in message, "the upstream checkout is fetched by the tool, not a submodule"
