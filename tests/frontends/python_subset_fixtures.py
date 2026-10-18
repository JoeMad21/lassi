"""Shared helpers for the Python-subset frontend tests (task P12.4): its constructs, their fixtures, and the frontend.

Each construct of the subset has a small Python source and its golden hub module, written by hand, under
tests/fixtures/frontends/python_subset/ as <construct>.py and <construct>.mlir. The goldens are MLIR in custom
assembly at LLVM 18.0.0git's syntax (Polygeist's LLVM, llvm-project 26eb4285); no pinned MLIR tool runs on
Windows, so their pinned parse and verify is task P12.12's. The lowering scheme the goldens follow is stated in
tests/frontends/test_python_subset.py's docstring.

The helpers load the frontend package and the refusal exception through importlib and fail the calling test
with a plain message while either does not exist, so a missing piece reads as missing behavior, not as an
import error of the test module.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.core.registry import DEFAULT_REGISTRY

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "frontends" / "python_subset"
FRONTEND_NAME = "python-subset"
HUB_LEVEL = "cpu-mlir"
STRUCTURED = "structured"
LOW = "low"
# The dialects a P12.4 hub module may hold: builtin's module, func, arith, math, scf, and memref.
HUB_DIALECTS = frozenset({"builtin", "func", "arith", "math", "scf", "memref"})

# The subset's constructs, one fixture pair each, in the order the test docstring lists them.
CONSTRUCTS = (
    "scalar_function",
    "float_arith",
    "int_arith",
    "compare_if",
    "conditional_expr",
    "for_range",
    "while_loop",
    "prange",
    "array_access",
    "array_alloc",
    "call_arrays",
    "math_calls",
    "numpy_vectorized",
    "harness_io",
    "annotated_scalars",
    "array_alias",
    "write_shapes",
    "prange_bounds",
)

# An op's name where a line of a module starts one: after an optional result list, the first word.
_OP = re.compile(r"^\s*(?:%[\w:#]+(?:, %[\w:#]+)* = )?([A-Za-z_][\w.$]*)")
# Ops the func dialect prints without its prefix where it is the default dialect (directly in a func.func body).
_FUNC_SHORT = frozenset({"call", "return"})


def frontends_package() -> ModuleType:
    """Return the lassi.frontends package, failing the test clearly while it does not exist (task P12.4)."""
    try:
        return importlib.import_module("lassi.frontends")
    except ModuleNotFoundError as error:
        if error.name in ("lassi.frontends", "lassi.frontends.python_subset"):
            pytest.fail(f"lassi.frontends does not exist yet (task P12.4): {error}")
        raise


def python_subset_module() -> ModuleType:
    """Return lassi.frontends.python_subset, failing the test clearly while it does not exist (task P12.4)."""
    frontends_package()
    try:
        return importlib.import_module("lassi.frontends.python_subset")
    except ModuleNotFoundError as error:
        if error.name == "lassi.frontends.python_subset":
            pytest.fail(f"lassi.frontends.python_subset does not exist yet (task P12.4): {error}")
        raise


def frontend_refusal() -> type:
    """Return lassi.core.interfaces.FrontendRefusal, failing the test clearly while it does not exist."""
    module = importlib.import_module("lassi.core.interfaces")
    if not hasattr(module, "FrontendRefusal"):
        pytest.fail("lassi.core.interfaces has no FrontendRefusal yet (task P12.4)")
    return module.FrontendRefusal


def hub_tag_function() -> Any:
    """Return lassi.frontends.hub_tag, failing the test clearly while it does not exist."""
    package = frontends_package()
    if not hasattr(package, "hub_tag"):
        pytest.fail("lassi.frontends has no hub_tag yet (task P12.4)")
    return package.hub_tag


def python_subset() -> Any:
    """Return a new python-subset Frontend, built as the runner builds a component with no config: factory()."""
    frontends_package()
    names = DEFAULT_REGISTRY.names("Frontend")
    if FRONTEND_NAME not in names:
        pytest.fail(f"no Frontend is registered as {FRONTEND_NAME!r} yet (task P12.4); registered: {names}")
    return DEFAULT_REGISTRY.get("Frontend", FRONTEND_NAME).factory()


def read_ascii(path: Path) -> str:
    """Return a fixture file's text, read as bytes and decoded as ASCII, so no newline is translated."""
    return path.read_bytes().decode("ascii")


def source(construct: str) -> str:
    """Return the Python source of `construct`."""
    return read_ascii(FIXTURES / f"{construct}.py")


def golden(construct: str) -> str:
    """Return the golden hub module of `construct`."""
    return read_ascii(FIXTURES / f"{construct}.mlir")


def raise_text(text: str, path: str = "main.py") -> Any:
    """Raise one Python source with no flags through a new python-subset frontend; return its Module."""
    return python_subset().raise_source({path: text}, [])


def op_names(text: str) -> list[str]:
    """Return the name of each op a module text starts, in text order (a line that only closes a region has none)."""
    names = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith(("}", "//")):
            continue
        match = _OP.match(line)
        if match:
            names.append(match.group(1))
    return names


def dialect_of(op: str) -> str:
    """Return the dialect an op name belongs to: its prefix, func for call and return, builtin for module."""
    if op in _FUNC_SHORT:
        return "func"
    if op == "module":
        return "builtin"
    return op.split(".", 1)[0]
