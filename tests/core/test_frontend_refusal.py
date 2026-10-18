"""Tests for the shared frontend refusal: a source outside a frontend's subset (task P12.4).

Bible: Component Interfaces (Frontend; diagnostics records), Result Record (Diagnostic), Frontend Rules
(anything outside a published subset fails with a readable diagnostic that is counted in corpus statistics),
Agent Rule 3; plans/p12-frontends.md, task P12.4 and its planning decision "Refusals (P12.4)".

The contract these tests fix:

- lassi.core.interfaces.FrontendRefusal(diagnostics), defined beside Frontend, is the one exception every
  frontend raises for a source outside its subset. It is built from Diagnostic records (lassi.core.record)
  alone, keeps them in order as a tuple in `diagnostics`, and refuses, with a ValueError, an empty sequence
  and a diagnostic at any stage other than "parse"; an item that is not a Diagnostic is refused too.
- Its message is readable: it holds each diagnostic's message and, for one with a file, a line, and a column,
  `<file>:<line>:<column>`.
- It is generic: Frontend.raise_source's docstring names it, it takes no language or frontend, and its source
  names no project, suite, model, or frontend.
- The Result Record does not change: Diagnostic keeps the bible's fields and DIAGNOSTIC_STAGES its six stages.
  These two guards pass before the task and fail if it changes the record.

Every diagnostic here is SYNTHETIC. No value in this module is a measurement.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect

import pytest

from lassi.core.interfaces import Frontend
from lassi.core.record import DIAGNOSTIC_STAGES, Diagnostic

PROJECT_NAMES = (
    "lassi-repro", "lassi-ee", "lassi-df", "lassi-demo", "hecbench", "xlang", "qwen", "wizardcoder", "a100",
    "mi300x", "gpt-oss", "python-subset", "rust-mir", "csharp-roslyn",
)
FIRST = Diagnostic(
    stage="parse", severity="error", code="subset-call", file="main.py", line=5, column=5,
    message="SYNTHETIC: a call of print is outside the subset",
)
SECOND = Diagnostic(
    stage="parse", severity="error", code="subset-control", file="main.py", line=10, column=13,
    message="SYNTHETIC: break is outside the subset",
)
PLACELESS = Diagnostic(stage="parse", severity="error", message="SYNTHETIC: the text names no place")


def frontend_refusal() -> type:
    """Return lassi.core.interfaces.FrontendRefusal, failing the test clearly while it does not exist."""
    module = importlib.import_module("lassi.core.interfaces")
    if not hasattr(module, "FrontendRefusal"):
        pytest.fail("lassi.core.interfaces has no FrontendRefusal yet (task P12.4)")
    return module.FrontendRefusal


def test_the_refusal_keeps_its_diagnostics_in_order() -> None:
    refusal = frontend_refusal()
    given = [FIRST, SECOND]
    error = refusal(given)
    assert isinstance(error, Exception)
    assert error.diagnostics == (FIRST, SECOND)
    assert isinstance(error.diagnostics, tuple)
    given.append(PLACELESS)
    assert error.diagnostics == (FIRST, SECOND), "the refusal keeps its own copy"


def test_the_message_names_each_place_and_reason() -> None:
    refusal = frontend_refusal()
    message = str(refusal([FIRST, SECOND, PLACELESS]))
    assert "main.py:5:5" in message and "main.py:10:13" in message, message
    for item in (FIRST, SECOND, PLACELESS):
        assert item.message in message, message
    assert message.isascii()


def test_a_refusal_needs_a_diagnostic() -> None:
    refusal = frontend_refusal()
    with pytest.raises(ValueError):
        refusal([])


def test_a_refusal_holds_parse_diagnostics_only() -> None:
    refusal = frontend_refusal()
    for stage in DIAGNOSTIC_STAGES:
        if stage == "parse":
            continue
        other = dataclasses.replace(FIRST, stage=stage)
        with pytest.raises(ValueError, match=stage):
            refusal([FIRST, other])


def test_a_refusal_holds_diagnostics_only() -> None:
    refusal = frontend_refusal()
    with pytest.raises((TypeError, ValueError)):
        refusal([FIRST, "SYNTHETIC: not a Diagnostic"])


def test_the_refusal_is_generic() -> None:
    refusal = frontend_refusal()
    assert refusal.__module__ == "lassi.core.interfaces"
    assert inspect.getdoc(refusal)
    required = [
        name for name, parameter in inspect.signature(refusal).parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    assert required == ["diagnostics"], f"it is built from diagnostics alone: {required}"
    lowered = inspect.getsource(refusal).lower()
    found = [name for name in PROJECT_NAMES if name in lowered]
    assert not found, f"Agent Rule 3: the refusal names {found}"
    assert "FrontendRefusal" in (inspect.getdoc(Frontend.raise_source) or ""), "the contract names the refusal"


def test_the_result_record_does_not_change() -> None:
    names = [item.name for item in dataclasses.fields(Diagnostic)]
    assert names == ["stage", "severity", "code", "file", "line", "column", "message"]
    assert DIAGNOSTIC_STAGES == ("parse", "verify", "lower", "compile", "jit", "run")
