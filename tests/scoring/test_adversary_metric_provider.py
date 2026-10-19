"""Tests for the adversary metrics as a third metric provider in `lassi run` (task P16.6).

Bible: Project Recipes (Notes, score and metrics: each metrics name is
offered by exactly one provider, and a name no provider offers, two offer,
or that is listed twice is refused before the run directory exists),
Evaluation Protocol (the Judges and adversaries row; Run Metrics),
Readability Standards (Run row: run.md's Metrics section; parquet/ mirrors
it), Agent Rule 14. Plan: plans/p16-adversarial.md, P16.6.

The contract these tests fix:

- lassi.scoring.run_scoring.plan_scoring(data, registry, bench_root) offers
  the names of lassi.analysis.adversary ADVERSARY_METRICS when the
  recipe's stages list a stage that declares runs_proposals
  (lassi.core.adversary_stage RUNS_PROPOSALS), read from the registry as
  the runner reads it, never from the stage's name; with no such stage
  listed (registered or not), a listed stage named `adversary` that does
  not declare it included, each name is refused as one no provider
  offers. Their sources come from the trial records, so naming only them
  builds no ScoreProfile into the plan (plan.profiles is empty), and each
  source's description names the adversary.
- When a registered ScoreProfile declares one of those names as a trial
  component and such a stage is listed, the name is refused as one more
  than one provider offers, as the other providers are; with no such stage
  listed the profile alone offers it. A name listed twice is refused. The
  registered ScoreProfiles (lassi, df-v0) offer none of the names.
- compute_metrics builds the adversary tables from the trials
  (lassi.analysis.adversary adversary_tables), metrics_section shows them
  (every line of adversary_markdown at heading level 3, in order, its
  heading `### Adversary metrics` and one `####` heading per table) in plain
  ASCII,
  and write_metrics writes them under parquet/ so that
  read_adversary_parquet gives adversary_rows of the same tables. Naming
  only them writes no metric_values table and no run metric table.

Every trial and record here is SYNTHETIC and hand-built; no value in this
module is a measurement.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.analysis.tables import read_metrics_parquet
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.adversary_stage import RUNS_PROPOSALS
from lassi.core.interfaces import Sampling, Score
from lassi.core.record import (
    AdversaryRecord,
    BenchItem,
    ContractFinding,
    Divergence,
    ModelInfo,
    OutputStats,
    Provenance,
    Trial,
    make_trial_id,
)
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.scoring.run_scoring import compute_metrics, metrics_section, plan_scoring, write_metrics

MODULE = "lassi.analysis.adversary"
DIVERGENCE_RATE = "confirmed_divergence_rate"
REJECTION_RATE = "invalid_input_rejection_rate"
FINDING_COUNT = "contract_finding_count"
NAMES = [DIVERGENCE_RATE, REJECTION_RATE, FINDING_COUNT]
NO_ADVERSARY = ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
WITH_ADVERSARY = [*NO_ADVERSARY, "adversary"]
SCORE_PROFILES = ("lassi", "df-v0")
# The registry name of the SYNTHETIC ScoreProfile that declares an adversary metric's name as a trial component.
FAKE_PROFILE = "declaring-fake"
# The registry name under which the real adversary stage is registered for the capability test.
OTHER_NAME = "proposal-runner"
PROJECT, MODEL, SUITE, DIRECTION = "adv-provider", "scripted-fixture", "adv-suite", "omp-cpp"
SHA = "d" * 64
FAILED = OutputStats(name="y", pcc=0.0, max_abs=1000.0, max_ulp=7, passed=False, note="SYNTHETIC: y differs")


def adversary() -> ModuleType:
    """Import lassi.analysis.adversary; fail clearly while task P16.6 has not added it."""
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as error:
        if error.name != MODULE:
            raise
        pytest.fail(f"task P16.6 adds {MODULE}: {error}")


def need(name: str) -> Any:
    """Return the module's attribute `name`, failing with the task's name when it does not exist yet."""
    module = adversary()
    assert hasattr(module, name), f"{MODULE} has no {name} (task P16.6)"
    return getattr(module, name)


def declaring_profile(declared: Sequence[str]) -> type:
    """Return a ScoreProfile class that declares `declared` as its trial_components and never scores a trial."""

    class DeclaringProfile:
        """A SYNTHETIC ScoreProfile, built only to read its trial_components."""

        name = FAKE_PROFILE
        capabilities: frozenset[str] = frozenset()
        trial_components = tuple(declared)

        def score(self, trial: Trial) -> Score:
            """Fail: no test here scores a trial with this profile."""
            raise AssertionError(f"{FAKE_PROFILE} scored {trial.trial_id}")

    return DeclaringProfile


def make_registry(*, profiles: Sequence[str] = (), fake: Sequence[str] | None = None, other: bool = False) -> Registry:
    """Return a Registry with the real stages, the named real ScoreProfiles, and the optional fakes.

    `fake` registers declaring_profile(fake); `other` registers the real
    adversary stage once more under OTHER_NAME.
    """
    registry = Registry()
    for name in WITH_ADVERSARY:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    if other:
        registry.register("Stage", OTHER_NAME, DEFAULT_REGISTRY.get("Stage", "adversary").factory)
    for name in profiles:
        registry.register("ScoreProfile", name, DEFAULT_REGISTRY.get("ScoreProfile", name).factory)
    if fake is not None:
        registry.register("ScoreProfile", FAKE_PROFILE, declaring_profile(fake))
    return registry


def recipe(stages: Sequence[str], metrics: Sequence[str]) -> dict[str, Any]:
    """Return the resolved-recipe keys plan_scoring reads, with an adversary section when a stage runs proposals."""
    data: dict[str, Any] = {"stages": list(stages), "metrics": list(metrics)}
    if "adversary" in stages or OTHER_NAME in stages:
        data["adversary"] = {"kind": "fuzzer", "budget": {"inputs": 4}, "seed": 1}
    return data


# ---------------------------------------------------------------------------
# Which names are offered, and when


def test_the_registered_adversary_stage_declares_runs_proposals() -> None:
    assert RUNS_PROPOSALS in DEFAULT_REGISTRY.get("Stage", "adversary").capabilities, "precondition (task P16.5)"


@pytest.mark.parametrize("metrics", [[name] for name in NAMES] + [NAMES], ids=[*NAMES, "all"])
def test_the_adversary_metrics_are_offered_when_the_stage_is_listed(metrics: list[str]) -> None:
    assert list(need("ADVERSARY_METRICS")) == NAMES
    plan = plan_scoring(recipe(WITH_ADVERSARY, metrics), make_registry(), None)
    assert [source.metric for source in plan.metrics] == metrics
    assert dict(plan.profiles) == {}, "the adversary metrics read trial records; no ScoreProfile is needed for them"
    for source in plan.metrics:
        assert "adversary" in source.describe().lower(), f"{source.metric}: the source does not name the adversary"


def test_any_listed_stage_that_declares_runs_proposals_offers_them() -> None:
    stages = [*NO_ADVERSARY, OTHER_NAME]
    plan = plan_scoring(recipe(stages, NAMES), make_registry(other=True), None)
    assert [source.metric for source in plan.metrics] == NAMES, "the provider reads the capability, not the name"


@pytest.mark.parametrize("name", NAMES)
def test_an_adversary_metric_is_refused_when_no_listed_stage_runs_proposals(name: str) -> None:
    registry = make_registry()
    assert "adversary" in registry.names("Stage"), "precondition: registered, but not listed"
    with pytest.raises(ValueError, match=name):
        plan_scoring(recipe(NO_ADVERSARY, [name]), registry, None)


class NamedOnlyStage:
    """A SYNTHETIC stage registered as `adversary` that declares no capability, runs_proposals included."""

    capabilities: frozenset[str] = frozenset()

    def __init__(self, *, context: object) -> None:
        """Keep the run context; this stage is only registered, never built by these tests."""
        self.context = context

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` unchanged."""
        return trial


@pytest.mark.parametrize("name", NAMES)
def test_a_stage_named_adversary_that_does_not_declare_runs_proposals_offers_nothing(name: str) -> None:
    registry = Registry()
    for stage in NO_ADVERSARY:
        registry.register("Stage", stage, DEFAULT_REGISTRY.get("Stage", stage).factory)
    registry.register("Stage", "adversary", NamedOnlyStage)
    assert RUNS_PROPOSALS not in registry.get("Stage", "adversary").capabilities, "precondition"
    with pytest.raises(ValueError, match="no registered provider") as caught:
        plan_scoring(recipe(WITH_ADVERSARY, [name]), registry, None)
    assert name in str(caught.value)


def test_a_name_another_provider_also_offers_is_refused_when_the_stage_is_listed() -> None:
    registry = make_registry(fake=[DIVERGENCE_RATE])
    with pytest.raises(ValueError, match="more than one provider") as caught:
        plan_scoring(recipe(WITH_ADVERSARY, [DIVERGENCE_RATE]), registry, None)
    assert DIVERGENCE_RATE in str(caught.value)


def test_without_the_stage_the_other_provider_alone_offers_that_name() -> None:
    plan = plan_scoring(recipe(NO_ADVERSARY, [DIVERGENCE_RATE]), make_registry(fake=[DIVERGENCE_RATE]), None)
    (source,) = plan.metrics
    assert (source.metric, source.profile) == (DIVERGENCE_RATE, FAKE_PROFILE)


def test_an_adversary_metric_named_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="twice"):
        plan_scoring(recipe(WITH_ADVERSARY, [REJECTION_RATE, FINDING_COUNT, REJECTION_RATE]), make_registry(), None)


def test_no_registered_score_profile_offers_an_adversary_metric_name(tmp_path: Path) -> None:
    registry = make_registry(profiles=SCORE_PROFILES)
    plan = plan_scoring(recipe(WITH_ADVERSARY, NAMES), registry, tmp_path)
    assert [source.metric for source in plan.metrics] == NAMES
    assert dict(plan.profiles) == {}


# ---------------------------------------------------------------------------
# The run's metrics: run.md's Metrics section and parquet/


def trial(item: str, found: AdversaryRecord | None) -> Trial:
    """Return a SYNTHETIC trial of `item` with `found` as its adversary record."""
    return Trial(
        trial_id=make_trial_id(PROJECT, MODEL, SUITE, DIRECTION, item, 1),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=None, dirty=None, device=None, sdk=None, date="2026-10-19T00:00:00+00:00"),
        bench_item=BenchItem(suite=SUITE, item=item, split="unassigned", direction=DIRECTION),
        model=ModelInfo(backend="scripted", id=MODEL, sampling=Sampling(0.2, 0.9, 8)),
        adversary=found,
    )


def hand_built_trials() -> list[Trial]:
    """Return three SYNTHETIC trials: one diverged with a refusal, one with a contract finding, one not taken."""
    diverged = AdversaryRecord(
        agent="fuzzer", config={"budget": {"inputs": 4}, "seed": 1}, proposals=4, refusals={"out-of-range": 1},
        runs=6, wall_s=2.0, contract_findings=[],
        divergences=[Divergence(proposal=1, inputs={"x": SHA}, stats=[FAILED], candidate_failure=None)],
    )
    finding = AdversaryRecord(
        agent="fuzzer", config={"budget": {"inputs": 4}, "seed": 1}, proposals=2, refusals={}, runs=4, wall_s=1.0,
        divergences=[], contract_findings=[ContractFinding(proposal=0, inputs={"x": SHA}, message="SYNTHETIC")],
    )
    return [trial("item0", diverged), trial("item1", finding), trial("item2", None)]


def assert_lines_shown(section: str, rendered: str) -> None:
    """Assert that every non-blank line of `rendered`, headings included, appears in `section`, in order."""
    lines = section.split("\n")
    position = 0
    for line in rendered.split("\n"):
        if not line.strip():
            continue
        assert line in lines[position:], f"the Metrics section lacks, or misorders, the line {line!r}"
        position = lines.index(line, position) + 1


def test_the_metrics_section_and_parquet_show_the_adversary_tables(tmp_path: Path) -> None:
    trials = hand_built_trials()
    plan = plan_scoring(recipe(WITH_ADVERSARY, NAMES), make_registry(), None)
    metrics = compute_metrics(plan, trials)
    assert metrics is not None
    expected = need("adversary_tables")(trials)
    assert len(expected) == 1, "precondition: one arm, suite, adversary, and direction"
    section = metrics_section(metrics, None)
    assert section.isascii()
    rendered = need("adversary_markdown")(expected, level=3)
    headings = [line for line in rendered.split("\n") if line.startswith("#")]
    assert headings[0] == "### Adversary metrics", headings
    assert len(headings) == 2 and headings[1].startswith("#### "), "one heading per table, a level below"
    assert all(part in headings[1] for part in (MODEL, SUITE, "fuzzer", DIRECTION)), headings[1]
    assert section.split("\n")[0] == "## Metrics", "the adversary tables sit under the Metrics heading"
    assert_lines_shown(section, rendered)
    parquet = tmp_path / "parquet"
    write_metrics(metrics, parquet)
    assert need("read_adversary_parquet")(parquet) == need("adversary_rows")(expected)
    assert not (parquet / "metric_values").exists(), "no trial component was named"
    assert all(rows == [] for rows in read_metrics_parquet(parquet).values()), "no run metric table was named"
