"""Tests for the runner's side of the adversary stage: what it carries out, refuses, and reports (P16.5, P16.6).

Bible: Project Recipes (the agent and judge bindings block; Notes,
executors; Notes, score and metrics), Component Interfaces (Stage;
Executor; Oracle), Sandbox (model-generated code), Oracles (binary_io
rules), Adversarial Setups (item 1), Evaluation Protocol (the Judges and
adversaries row), Readability Standards (Run row), Agent Rules 6 and 14.
Plan: plans/p16-adversarial.md, P16.5 and P16.6; PHASE-NOTES P16, where it
stands (the runner refuses adversary sections, lassi/core/runner.py
_NOT_CARRIED_OUT), and Hand-offs from P16.5 (the P16.6 item).

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
- The refusals the bible states that P16.5's tests left uncovered
  (PHASE-NOTES P16, Hand-offs from P16.5), each before any directory
  exists: the adversary stage listed twice, or listed before the oracle
  stage or with no oracle stage listed; an adversary.budget.inputs that is
  not an int of at least 1 (a bool, 0, a negative, a float, a string, null,
  a missing budget, or a budget that is not a mapping), naming
  adversary.budget.inputs; an Agent that raises ValueError when built
  (naming it and its message) or has no callable propose() (naming
  propose). The stage itself raises ValueError naming adversary.kind when
  it takes a trial and the recipe binds no adversary.
- Task P16.6: with the recipe's metrics naming the adversary metrics
  (lassi.analysis.adversary ADVERSARY_METRICS), run.md's Metrics section
  lists each with its source and shows every line of
  adversary_markdown(adversary_tables(trials), level=3) in order (its
  `### Adversary metrics` heading, then one `####` heading per table naming
  the arm, suite, adversary, and direction), and parquet/ holds them
  (read_adversary_parquet gives adversary_rows of the same tables) beside
  the trials' own adversary table. A run whose stage took no trial still
  finishes, with the names listed, the heading and NO_TABLES under it, and
  no adversary table rows; a run that names no metric has no Metrics
  section and no adversary metric rows.

The suite, recipes, and components are SYNTHETIC fakes
(tests/core/adversary_support.py); no value in this module is a
measurement.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from adversary_support import (
    ADVERSARY,
    CANDIDATE_ROLE,
    DIRECTION,
    MODEL_ID,
    PLAIN,
    SPLIT,
    SUITE,
    World,
    adversary_stage,
    clean_environment,
    constant,
    make_registry,
    observer,
    program,
    proposal,
    recipe_data,
    refused,
    run,
    write_recipe,
    write_suite,
)

from lassi.core import runner as runner_module
from lassi.core.interfaces import Sampling
from lassi.core.parquet import read_run_parquet
from lassi.core.recipe import RecipeError
from lassi.core.record import (
    Alignment,
    Attempt,
    BenchItem,
    ModelInfo,
    OutputStats,
    Provenance,
    RunInfo,
    TextRef,
    Trial,
    make_trial_id,
)
from lassi.core.runner import RunError, RunOptions, run_recipe

STILL_REFUSED = ("refine", "agents", "judges")
NO_ADVERSARY_STAGES = ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
ANALYSIS = "lassi.analysis.adversary"
ADVERSARY_METRICS = ["confirmed_divergence_rate", "invalid_input_rejection_rate", "contract_finding_count"]
# The registry names of the SYNTHETIC Agents that refuse their settings and that have no propose().
REFUSING, SILENT, INERT = "refusing-adversary", "silent-adversary", "inert-adversary"
REFUSAL_TEXT = "SYNTHETIC: the seed must be set"
METRICS_HEADING = "## Metrics"


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


# A judges section the loader accepts since task P15.1: every required choice set, the SYNTHETIC fixture rubric, and a
# model of a family the test's table gives, so the refusal under test is the runner's.
JUDGE_SECTION = {
    "backend": "scripted",
    "model": "SYNTHETIC-model",
    "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64},
    "rubric": "synthetic-v1",
    "use": "screen",
}
JUDGE_RUBRICS = Path(__file__).resolve().parents[1] / "fixtures" / "judges" / "rubrics"


@pytest.mark.parametrize(
    ("key", "section"),
    [
        ("refine", {"counter_start": 1, "counter_step": 1, "max_iters": 3}),
        ("agents", {"summarizer": {"model": "SYNTHETIC-model"}}),
        ("judges", {"equivalence": JUDGE_SECTION}),
    ],
)
def test_refine_agents_and_judges_stay_refused(
    tmp_path: Path, bench: Path, key: str, section: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    families = tmp_path / "families.yaml"
    families.write_bytes(
        f"families: [synthetic-a, synthetic-b]\nreasoning_effort: []\n"
        f"models: {{{MODEL_ID}: synthetic-a, SYNTHETIC-model: synthetic-b}}\n".encode("ascii")
    )
    monkeypatch.setattr(importlib.import_module("lassi.judges.families"), "FAMILIES_FILE", families)
    monkeypatch.setattr(importlib.import_module("lassi.judges.rubric"), "RUBRIC_ROOT", JUDGE_RUBRICS)
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


# ---------------------------------------------------------------------------
# Hand-offs from P16.5: refusals the bible states


class RefusingAdversary:
    """A SYNTHETIC adversary that refuses its settings when built."""

    name = REFUSING
    capabilities = frozenset({"proposes_inputs"})
    config_keys = frozenset({"budget"})

    def __init__(self, *, budget: Mapping[str, Any] | None = None) -> None:
        """Refuse every config."""
        raise ValueError(REFUSAL_TEXT)


class SilentAdversary:
    """A SYNTHETIC adversary that declares proposes_inputs but has no propose()."""

    name = SILENT
    capabilities = frozenset({"proposes_inputs"})
    config_keys = frozenset({"budget"})

    def __init__(self, *, budget: Mapping[str, Any] | None = None) -> None:
        """Keep the budget section as given."""
        self.budget = budget


class InertAdversary(SilentAdversary):
    """A SYNTHETIC adversary whose propose is an attribute that cannot be called."""

    name = INERT
    propose = None


def refused_with(tmp_path: Path, bench: Path, world: World, data: Mapping[str, Any], agents: Mapping[str, type]) -> str:
    """Run `data` with `agents` added to the test registry, expect a refusal before any directory, return it."""
    registry = make_registry(world)
    for name, factory in agents.items():
        registry.register("Agent", name, factory)
    path = write_recipe(tmp_path, data, "recipe-refused-with")
    options = RunOptions(
        runs_root=tmp_path / "runs-root",
        run_id="refused",
        bench_root=bench,
        registry=registry,
        observer=observer(world),
    )
    with pytest.raises((RunError, RecipeError)) as caught:
        run_recipe(path, options)
    assert not (tmp_path / "runs-root").exists(), "a refused run creates no directory"
    assert not any(kind == "run" for kind, _ in world.events), "a refused run runs nothing"
    return str(caught.value)


def test_the_stage_listed_twice_is_refused(tmp_path: Path, bench: Path) -> None:
    adversary_stage()
    stages = [*NO_ADVERSARY_STAGES, "adversary", "adversary"]
    message = refused(tmp_path, bench, World(), recipe_data(stages=stages))
    assert "adversary" in message and ("2 times" in message or "twice" in message), message


@pytest.mark.parametrize(
    "stages",
    [
        ["baseline", "generate", "compile_loop", "run_loop", "adversary", "oracle"],
        ["baseline", "generate", "compile_loop", "run_loop", "adversary"],
    ],
    ids=["before-the-oracle-stage", "no-oracle-stage"],
)
def test_the_stage_listed_before_the_oracle_stage_is_refused(tmp_path: Path, bench: Path, stages: list[str]) -> None:
    adversary_stage()
    message = refused(tmp_path, bench, World(), recipe_data(stages=stages))
    assert "adversary" in message and "oracle" in message and "before" in message, message


@pytest.mark.parametrize(
    "section",
    [
        {"kind": ADVERSARY, "budget": {"inputs": True}},
        {"kind": ADVERSARY, "budget": {"inputs": False}},
        {"kind": ADVERSARY, "budget": {"inputs": 0}},
        {"kind": ADVERSARY, "budget": {"inputs": -2}},
        {"kind": ADVERSARY, "budget": {"inputs": 2.5}},
        {"kind": ADVERSARY, "budget": {"inputs": 4.0}},
        {"kind": ADVERSARY, "budget": {"inputs": "4"}},
        {"kind": ADVERSARY, "budget": {"inputs": None}},
        {"kind": ADVERSARY, "budget": {}},
        {"kind": ADVERSARY},
        {"kind": ADVERSARY, "budget": 4},
    ],
    ids=["true", "false", "zero", "negative", "fraction", "float", "string", "null", "no-inputs", "no-budget", "int"],
)
def test_a_budget_that_is_not_an_int_of_at_least_one_is_refused(
    tmp_path: Path, bench: Path, section: dict[str, Any]
) -> None:
    adversary_stage()
    message = refused(tmp_path, bench, World(), recipe_data(adversary=section))
    assert "adversary.budget.inputs" in message, message


def test_an_agent_that_refuses_its_settings_is_refused(tmp_path: Path, bench: Path) -> None:
    adversary_stage()
    data = recipe_data(adversary={"kind": REFUSING, "budget": {"inputs": 4}})
    message = refused_with(tmp_path, bench, World(), data, {REFUSING: RefusingAdversary})
    assert REFUSING in message and REFUSAL_TEXT in message, message


@pytest.mark.parametrize(
    ("name", "factory"), [(SILENT, SilentAdversary), (INERT, InertAdversary)], ids=["no-propose", "inert-propose"]
)
def test_an_agent_with_no_callable_propose_is_refused(tmp_path: Path, bench: Path, name: str, factory: type) -> None:
    adversary_stage()
    data = recipe_data(adversary={"kind": name, "budget": {"inputs": 4}})
    message = refused_with(tmp_path, bench, World(), data, {name: factory})
    assert name in message and "propose" in message, message


def taken_trial() -> Trial:
    """Return a SYNTHETIC trial whose standing attempt ran clean (S5) and passed binary_io (mean 1.0)."""
    stats = OutputStats(name="y", pcc=1.0, max_abs=0.0, max_ulp=0, passed=True, note=None)
    attempt = Attempt(
        index=0,
        files={"main.cpp": program(CANDIDATE_ROLE, "double")},
        stage_reached="S5",
        run=RunInfo(exit_code=0, hang=False, stdout_ref=TextRef(sha256="e" * 64, path="texts/ee/stdout.txt")),
        alignment=Alignment(per_input=[1.0], mean=1.0, outputs=[stats]),
    )
    return Trial(
        trial_id=make_trial_id("adv-stage", "scripted-fixture", SUITE, "omp-cpp", "scale", 1),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=None, dirty=None, device=None, sdk=None, date="2026-10-19T00:00:00+00:00"),
        bench_item=BenchItem(suite=SUITE, item="scale", split=SPLIT, direction="omp-cpp"),
        model=ModelInfo(backend="scripted", id="scripted-fixture", sampling=Sampling(0.2, 0.9, 8)),
        attempts=[attempt],
    )


def test_the_stage_raises_when_the_recipe_binds_no_adversary() -> None:
    stage = adversary_stage()(context=SimpleNamespace(recipe=SimpleNamespace(bindings=[])))
    with pytest.raises(ValueError, match=r"adversary\.kind"):
        stage(taken_trial())


# ---------------------------------------------------------------------------
# Task P16.6: the adversary metrics in run.md and parquet/


def analysis() -> ModuleType:
    """Import lassi.analysis.adversary; fail clearly while task P16.6 has not added it."""
    try:
        return importlib.import_module(ANALYSIS)
    except ModuleNotFoundError as error:
        if error.name != ANALYSIS:
            raise
        pytest.fail(f"task P16.6 adds {ANALYSIS}: {error}")


def metrics_part(run_dir: Path) -> str | None:
    """Return run.md from its Metrics heading to its end (the last section), or None when it has none."""
    lines = (run_dir / "run.md").read_bytes().decode("ascii").split("\n")
    if METRICS_HEADING not in lines:
        return None
    return "\n".join(lines[lines.index(METRICS_HEADING) :])


def assert_lines_shown(section: str, rendered: str) -> None:
    """Assert that every non-blank line of `rendered`, headings included, appears in `section`, in order."""
    lines = section.split("\n")
    position = 0
    for line in rendered.split("\n"):
        if not line.strip():
            continue
        assert line in lines[position:], f"the Metrics section lacks, or misorders, the line {line!r}"
        position = lines.index(line, position) + 1


def adversary_headings(section: str) -> list[str]:
    """Return the Metrics section's heading lines below `## Metrics`: the adversary metrics' and their tables'."""
    return [line for line in section.split("\n")[1:] if line.startswith("#")]


def assert_sources_listed(section: str) -> None:
    """Assert that the Metrics section lists each adversary metric in a table line that names the adversary."""
    for name in ADVERSARY_METRICS:
        found = [line for line in section.split("\n") if line.startswith(f"| {name} |") and "adversary" in line.lower()]
        assert found, f"the Metrics section lists no source for {name}"


def test_run_md_and_parquet_show_the_adversary_metrics(tmp_path: Path, bench: Path) -> None:
    adversary_stage()
    batch = [proposal(constant(500.0)), proposal(constant(1.0)), proposal(constant(5000.0))]
    world = World(candidate="large-zero", batches=[batch])
    outcome = run(tmp_path, bench, world, recipe_data(metrics=ADVERSARY_METRICS))
    found = outcome.trial.adversary
    assert found is not None, "precondition: the stage took the trial"
    assert (found.proposals, found.refusals, len(found.divergences)) == (3, {"out-of-range": 1}, 1), "precondition"
    module = analysis()
    tables = module.adversary_tables([outcome.trial])
    section = metrics_part(outcome.run_dir)
    assert section is not None, f"run.md has no line {METRICS_HEADING!r}"
    assert_sources_listed(section)
    headings = adversary_headings(section)
    assert headings[0] == "### Adversary metrics", headings
    assert len(headings) == 2 and headings[1].startswith("#### "), f"one table, a level below: {headings}"
    assert all(part in headings[1] for part in (MODEL_ID, SUITE, ADVERSARY, DIRECTION)), headings[1]
    assert_lines_shown(section, module.adversary_markdown(tables, level=3))
    parquet = outcome.run_dir / "parquet"
    assert module.read_adversary_parquet(parquet) == module.adversary_rows(tables)
    trial_rows = read_run_parquet(parquet)["adversary"]
    assert [row["trial_id"] for row in trial_rows] == [outcome.trial.trial_id], "the trials' own table stays"


def test_a_run_whose_stage_took_no_trial_still_lists_the_adversary_metrics(tmp_path: Path, bench: Path) -> None:
    adversary_stage()
    world = World(candidate="triple", batches=[[proposal(constant(500.0))]])
    outcome = run(tmp_path, bench, world, recipe_data(metrics=ADVERSARY_METRICS))
    assert outcome.trial.adversary is None, "precondition: the candidate fails binary_io, so the stage takes nothing"
    section = metrics_part(outcome.run_dir)
    assert section is not None, f"run.md has no line {METRICS_HEADING!r}"
    assert_sources_listed(section)
    assert adversary_headings(section) == ["### Adversary metrics"], "the heading stays, with no table under it"
    assert analysis().NO_TABLES in section.split("\n"), "run.md says the stage took no trial"
    assert analysis().read_adversary_parquet(outcome.run_dir / "parquet") == []


def test_a_run_that_names_no_metric_shows_no_adversary_metrics(tmp_path: Path, bench: Path) -> None:
    adversary_stage()
    world = World(candidate="large-zero", batches=[[proposal(constant(500.0))]])
    outcome = run(tmp_path, bench, world, recipe_data())
    assert outcome.trial.adversary is not None, "precondition: the stage took the trial"
    assert metrics_part(outcome.run_dir) is None
    assert analysis().read_adversary_parquet(outcome.run_dir / "parquet") == []
