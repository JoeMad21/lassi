"""Tests for the runner's side of the adversary stage: what it now carries out and what it refuses (task P16.5).

Bible: Project Recipes (the agent and judge bindings block; Notes,
executors), Component Interfaces (Stage; Executor; Oracle), Sandbox
(model-generated code), Oracles (binary_io rules), Adversarial Setups
(item 1), Agent Rules 6 and 14. Plan: plans/p16-adversarial.md, P16.5;
PHASE-NOTES P16, where it stands (the runner refuses adversary sections,
lassi/core/runner.py _NOT_CARRIED_OUT).

The contract these tests fix:

- lassi.core.runner._NOT_CARRIED_OUT no longer holds adversary; refine,
  agents, and judges stay refused as sections the runner does not carry
  out, each named in the message, before any directory exists.
- Before any directory exists and before anything runs, the runner refuses
  (RunError, or the loader's RecipeError where the load already refuses
  it), each message naming what is wrong:
  - an adversary section with no adversary stage listed, and the stage
    listed with no adversary section;
  - with the stage listed, an Oracle that does not declare
    aligns_output_files, since a divergence counts only when binary_io
    confirms it;
  - an item of the run with no held-out inputs, which has no input
    contract, naming the item;
  - a target language's executor that declares compile_only, which can run
    no proposal;
  - a target language's executor that runs programs without declaring
    sandboxed, since every proposal runs the candidate, model-generated
    code (Agent Rule 6), whether or not run_loop is listed.

The suite, recipes, and components are SYNTHETIC fakes
(tests/core/adversary_support.py); no value in this module is a
measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from adversary_support import (
    PLAIN,
    SPLIT,
    SUITE,
    World,
    adversary_stage,
    clean_environment,
    recipe_data,
    refused,
    write_suite,
)

from lassi.core import runner as runner_module

STILL_REFUSED = ("refine", "agents", "judges")
NO_ADVERSARY_STAGES = ["baseline", "generate", "compile_loop", "run_loop", "oracle"]


@pytest.fixture(autouse=True)
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no gate variable and a TMPDIR under the test's directory."""
    clean_environment(tmp_path, monkeypatch)


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the SYNTHETIC suite and return its bench root."""
    return write_suite(tmp_path, monkeypatch)


# ---------------------------------------------------------------------------
# What the runner carries out


def test_the_runner_carries_out_adversary_and_still_refuses_the_rest() -> None:
    assert runner_module._NOT_CARRIED_OUT == STILL_REFUSED


@pytest.mark.parametrize(
    ("key", "section"),
    [
        ("refine", {"counter_start": 1, "counter_step": 1, "max_iters": 3}),
        ("agents", {"summarizer": {"model": "SYNTHETIC-model"}}),
        ("judges", {"equivalence": {"model": "SYNTHETIC-model", "rubric": "SYNTHETIC-rubric", "use": "screen"}}),
    ],
)
def test_refine_agents_and_judges_stay_refused(tmp_path: Path, bench: Path, key: str, section: Any) -> None:
    data = recipe_data(stages=NO_ADVERSARY_STAGES, adversary=None, **{key: section})
    message = refused(tmp_path, bench, World(), data)
    assert key in message and "does not carry out" in message, message


# ---------------------------------------------------------------------------
# Refusals before any directory exists


def test_an_adversary_section_without_the_stage_is_refused(tmp_path: Path, bench: Path) -> None:
    adversary_stage()
    message = refused(tmp_path, bench, World(), recipe_data(stages=NO_ADVERSARY_STAGES))
    assert "adversary" in message and "stage" in message, message
    assert "does not carry out" not in message, "the runner carries out the section now; it needs the stage"


def test_the_stage_without_an_adversary_section_is_refused(tmp_path: Path, bench: Path) -> None:
    adversary_stage()
    message = refused(tmp_path, bench, World(), recipe_data(adversary=None))
    assert "adversary" in message, message
    assert "registered" not in message, "the stage is registered; the recipe lacks the section"


def test_an_oracle_that_compares_no_output_files_is_refused(tmp_path: Path, bench: Path) -> None:
    adversary_stage()
    message = refused(tmp_path, bench, World(), recipe_data(oracle={"kind": "masks-only"}))
    assert "aligns_output_files" in message and "adversary" in message, message


def test_an_item_without_held_out_inputs_is_refused_naming_it(tmp_path: Path, bench: Path) -> None:
    adversary_stage()
    data = recipe_data(bench={"suite": SUITE, "split": SPLIT, "items": ["scale", PLAIN]})
    message = refused(tmp_path, bench, World(), data)
    assert PLAIN in message and "held-out" in message and "adversary" in message, message


def test_a_compile_only_target_executor_is_refused(tmp_path: Path, bench: Path) -> None:
    adversary_stage()
    message = refused(tmp_path, bench, World(), recipe_data(executor={"kind": "compile-only"}))
    assert "compile_only" in message and "adversary" in message, message


@pytest.mark.parametrize(
    "stages",
    [
        ["baseline", "generate", "compile_loop", "run_loop", "oracle", "adversary"],
        ["baseline", "generate", "compile_loop", "oracle", "adversary"],
    ],
    ids=["with-run-loop", "without-run-loop"],
)
def test_a_target_executor_that_runs_programs_unsandboxed_is_refused(
    tmp_path: Path, bench: Path, stages: list[str]
) -> None:
    adversary_stage()
    message = refused(tmp_path, bench, World(), recipe_data(stages=stages, executor={"kind": "unsandboxed"}))
    assert "sandboxed" in message and "unsandboxed" in message, message
    if "run_loop" not in stages:
        assert "adversary" in message, "the adversary stage itself runs model-generated code"
