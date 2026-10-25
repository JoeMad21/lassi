"""Tests for the Judge interface's records: JudgeInput, Verdict, and checked_verdict (task P15.1).

Bible: Agents And Judges (the Judge interface: input of candidate, source,
artifacts, and measurements; a rubric; output of verdict, score in [0, 1],
and rationale; Judge rules), Component Interfaces (the Judge row), Agent
Rule 14 (measured beats judged). Plan: plans/p15-judges.md, P15.1.

The contract these tests fix:

- lassi.core.interfaces holds JudgeInput with exactly the fields candidate,
  source, artifacts, and measurements, in that order. candidate and source
  map each file's relative path (exactly a str) to its text (exactly a
  str); artifacts maps a non-empty name (exactly a str) to its text;
  measurements maps a non-empty name to a finite number, an int or a float
  and never a bool, kept as a float. Each is kept as a read-only copy. No
  field holds an oracle outcome, an alignment, or a statistic, so a judge
  never receives one (Agent Rule 14). A wrong type is refused with a
  TypeError and a wrong value (an empty name, a NaN or an infinity) with a
  ValueError, each naming the field as JudgeInput.<field>.
- Verdict has exactly the fields verdict, score, values, and rationale.
  verdict is exactly a str, a short token (a letter, then letters, digits,
  '_' or '-', at most 64 characters); score a finite number in [0, 1],
  never a bool, kept as a float; values maps a non-empty name to a finite
  number, kept as floats in a read-only copy; rationale is plain ASCII
  (printable characters, line feeds, and tabs). Anything else is refused
  naming the field as Verdict.<field>.
- checked_verdict(verdict, verdicts, values) returns a Verdict whose verdict
  is among the rubric's declared verdicts and whose every value name is
  among its declared value names; otherwise it raises ValueError naming
  Verdict.verdict or Verdict.values, and a TypeError for anything that is
  not a Verdict.
- Both classes are frozen and take no subclass, which could add a field.
- The Judge protocol's method is judge(view, rubric) -> Verdict.

Every value here is a SYNTHETIC fixture value, never a measurement.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from collections.abc import Callable
from typing import Any

import pytest

from lassi.core import interfaces

SOURCE = {"main.cpp": "int main() { return 0; }\n"}
CANDIDATE = {"main.cu": "int main() { return 0; }\n"}
ARTIFACTS = {"source_stdout": "SYNTHETIC 1 2 3\n", "candidate_stdout": "SYNTHETIC 1 2 3\n"}
MEASUREMENTS = {"runtime_s": 0.5, "energy_j": 2}
VERDICTS = ("AGREE", "DISAGREE")
VALUE_NAMES = ("delta",)
# Words naming what a judge never receives (Agent Rule 14); no JudgeInput field may hold one.
NEVER_SEEN = ("oracle", "outcome", "alignment", "statistic", "stats", "passed", "pcc", "max_abs", "max_ulp")


def need(name: str) -> Any:
    """Return lassi.core.interfaces' attribute `name`, failing with the task's name when it does not exist yet."""
    assert hasattr(interfaces, name), f"lassi.core.interfaces has no {name} (task P15.1)"
    return getattr(interfaces, name)


def judge_input(**changes: Any) -> Any:
    """Return a SYNTHETIC JudgeInput with `changes` applied to its fields."""
    values: dict[str, Any] = {
        "candidate": dict(CANDIDATE),
        "source": dict(SOURCE),
        "artifacts": dict(ARTIFACTS),
        "measurements": dict(MEASUREMENTS),
    }
    return need("JudgeInput")(**{**values, **changes})


def verdict(**changes: Any) -> Any:
    """Return a SYNTHETIC Verdict with `changes` applied to its fields."""
    values: dict[str, Any] = {"verdict": "AGREE", "score": 0.75, "values": {"delta": -0.25}, "rationale": "Same."}
    return need("Verdict")(**{**values, **changes})


def refused(error: type[Exception], build: Callable[[], Any], field: str) -> str:
    """Expect `build` to raise `error` whose message names `field`; return the message."""
    with pytest.raises(error) as caught:
        build()
    message = str(caught.value)
    assert field in message, message
    return message


# ---------------------------------------------------------------------------
# JudgeInput


def test_judge_input_has_exactly_its_four_fields_and_none_for_an_outcome_or_statistic() -> None:
    cls = need("JudgeInput")
    assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen
    names = tuple(spec.name for spec in dataclasses.fields(cls))
    assert names == ("candidate", "source", "artifacts", "measurements")
    assert not [name for name in names for word in NEVER_SEEN if word in name]


def test_judge_input_keeps_read_only_copies_and_measurements_as_floats() -> None:
    candidate = dict(CANDIDATE)
    view = judge_input(candidate=candidate)
    candidate["extra.cu"] = "int y;\n"
    assert dict(view.candidate) == CANDIDATE, "the input keeps its own copy"
    with pytest.raises(TypeError):
        view.candidate["main.cu"] = "changed"  # type: ignore[index]
    assert dict(view.measurements) == {"runtime_s": 0.5, "energy_j": 2.0}
    assert all(type(value) is float for value in view.measurements.values())
    assert judge_input(artifacts={}, measurements={}).measurements == {}


def test_judge_input_refuses_extra_fields() -> None:
    with pytest.raises(TypeError):
        judge_input(outcome="PASS")


@pytest.mark.parametrize(
    ("changes", "error", "field"),
    [
        ({"candidate": "int main();"}, TypeError, "JudgeInput.candidate"),
        ({"candidate": {"main.cu": b"int main();"}}, TypeError, "JudgeInput.candidate"),
        ({"source": {1: "int main();"}}, TypeError, "JudgeInput.source"),
        ({"source": [("main.cpp", "x")]}, TypeError, "JudgeInput.source"),
        ({"artifacts": {"stdout": 3}}, TypeError, "JudgeInput.artifacts"),
        ({"artifacts": {"": "text"}}, ValueError, "JudgeInput.artifacts"),
        ({"measurements": {"runtime_s": True}}, TypeError, "JudgeInput.measurements"),
        ({"measurements": {"runtime_s": "0.5"}}, TypeError, "JudgeInput.measurements"),
        ({"measurements": {"runtime_s": math.nan}}, ValueError, "JudgeInput.measurements"),
        ({"measurements": {"runtime_s": math.inf}}, ValueError, "JudgeInput.measurements"),
        ({"measurements": {"": 1.0}}, ValueError, "JudgeInput.measurements"),
        ({"measurements": {2: 1.0}}, TypeError, "JudgeInput.measurements"),
        ({"measurements": None}, TypeError, "JudgeInput.measurements"),
    ],
)
def test_judge_input_refuses_a_bad_field_naming_it(changes: dict[str, Any], error: type[Exception], field: str) -> None:
    refused(error, lambda: judge_input(**changes), field)


def test_judge_input_takes_no_subclass() -> None:
    with pytest.raises(TypeError):

        class Wider(need("JudgeInput")):  # type: ignore[misc]
            """A subclass that could add a field."""


# ---------------------------------------------------------------------------
# Verdict


def test_verdict_has_exactly_its_four_fields() -> None:
    cls = need("Verdict")
    assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen
    assert tuple(spec.name for spec in dataclasses.fields(cls)) == ("verdict", "score", "values", "rationale")


def test_verdict_keeps_a_float_score_and_read_only_values() -> None:
    values = {"delta": 1}
    found = verdict(score=1, values=values, rationale="Line one.\n\tLine two.\n")
    values["other"] = 2.0
    assert found.score == 1.0 and type(found.score) is float
    assert dict(found.values) == {"delta": 1.0} and type(found.values["delta"]) is float
    with pytest.raises(TypeError):
        found.values["delta"] = 2.0  # type: ignore[index]
    assert verdict(score=0, values={}).score == 0.0


@pytest.mark.parametrize(
    ("changes", "error", "field"),
    [
        ({"verdict": 1}, TypeError, "Verdict.verdict"),
        ({"verdict": ""}, ValueError, "Verdict.verdict"),
        ({"verdict": "NOT A TOKEN"}, ValueError, "Verdict.verdict"),
        ({"verdict": "V" * 65}, ValueError, "Verdict.verdict"),
        ({"score": True}, TypeError, "Verdict.score"),
        ({"score": "0.5"}, TypeError, "Verdict.score"),
        ({"score": None}, TypeError, "Verdict.score"),
        ({"score": math.nan}, ValueError, "Verdict.score"),
        ({"score": math.inf}, ValueError, "Verdict.score"),
        ({"score": -0.01}, ValueError, "Verdict.score"),
        ({"score": 1.01}, ValueError, "Verdict.score"),
        ({"values": [("delta", 1.0)]}, TypeError, "Verdict.values"),
        ({"values": {"delta": False}}, TypeError, "Verdict.values"),
        ({"values": {"delta": "1"}}, TypeError, "Verdict.values"),
        ({"values": {"delta": math.nan}}, ValueError, "Verdict.values"),
        ({"values": {"delta": -math.inf}}, ValueError, "Verdict.values"),
        ({"values": {"": 1.0}}, ValueError, "Verdict.values"),
        ({"rationale": None}, TypeError, "Verdict.rationale"),
        ({"rationale": b"bytes"}, TypeError, "Verdict.rationale"),
        ({"rationale": "caf\N{LATIN SMALL LETTER E WITH ACUTE}"}, ValueError, "Verdict.rationale"),
        ({"rationale": "a\x00b"}, ValueError, "Verdict.rationale"),
        ({"rationale": "a\r\nb"}, ValueError, "Verdict.rationale"),
    ],
)
def test_verdict_refuses_a_bad_field_naming_it(changes: dict[str, Any], error: type[Exception], field: str) -> None:
    refused(error, lambda: verdict(**changes), field)


def test_verdict_takes_no_subclass() -> None:
    with pytest.raises(TypeError):

        class Wider(need("Verdict")):  # type: ignore[misc]
            """A subclass that could add a field."""


# ---------------------------------------------------------------------------
# checked_verdict: the rubric's declared verdicts and value names


def test_checked_verdict_returns_a_verdict_the_rubric_declares() -> None:
    found = verdict()
    assert need("checked_verdict")(found, VERDICTS, VALUE_NAMES) is found
    assert need("checked_verdict")(verdict(values={}), VERDICTS, VALUE_NAMES).values == {}


def test_checked_verdict_refuses_an_undeclared_verdict_naming_the_field() -> None:
    found = verdict(verdict="VALID")
    message = refused(ValueError, lambda: need("checked_verdict")(found, VERDICTS, ()), "Verdict.verdict")
    assert "AGREE" in message and "DISAGREE" in message, "the refusal lists the declared verdicts"


def test_checked_verdict_refuses_an_undeclared_value_name_naming_the_field() -> None:
    found = verdict(values={"delta": 1.0, "power": 2.0})
    message = refused(ValueError, lambda: need("checked_verdict")(found, VERDICTS, VALUE_NAMES), "Verdict.values")
    assert "power" in message, message


def test_checked_verdict_refuses_what_is_not_a_verdict() -> None:
    refused(TypeError, lambda: need("checked_verdict")({"verdict": "AGREE"}, VERDICTS, VALUE_NAMES), "Verdict")


# ---------------------------------------------------------------------------
# The Judge protocol


def test_the_judge_protocol_judges_an_input_with_a_rubric() -> None:
    method = interfaces.Judge.judge
    assert list(inspect.signature(method).parameters) == ["self", "view", "rubric"]
    hints = method.__annotations__
    assert hints["view"] == "JudgeInput" and hints["rubric"] == "Rubric" and hints["return"] == "Verdict"
    assert inspect.getdoc(method)
