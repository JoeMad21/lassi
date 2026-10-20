"""Tests for the adversary's own log in the trial record: AgentLog and AdversaryRecord.agent_log (task P16.4).

Bible: Result Record (Trial; Storage), Readability Standards (Trial row),
Design Principle 7 (Parquet mirrors the JSON records), Component Interfaces
(capabilities), Agent Rules 1 and 14. Plan: plans/p16-adversarial.md, P16.4
("ContextExceeded ends its turn, recorded"; "a malformed reply gives no
proposal and a counted refusal"), and the design's section 6.

The contract these tests fix:

- lassi.core.record.AGENT_STOPS is ("steps", "tokens", "context-exceeded",
  "no-proposal"). AgentLog is a frozen, keyword-only record of steps,
  prompt_tokens, completion_tokens, malformed_replies, dropped_proposals
  (each an int of at least 0, a bool refused; malformed_replies at most
  steps), and stop (one of AGENT_STOPS, or None); a value outside these
  rules is a ValueError naming the field.
- AdversaryRecord.agent_log, an AgentLog or None, is its last field, None
  by default, so a trial.json written before it loads with it None. It
  round-trips through to_json and from_json and through write_trial and
  read_trial.
- The Parquet adversary table gains agent_log, a string column, last: the
  sorted-key JSON of the log, or null for a record without one (the
  fuzzer's).
- trial.md's Adversary section gains "### Agent log", a Field/Value table of
  the log's fields, only when agent_log is set.
- lassi.core.capabilities.REPORTS_LOG is "reports_log". The adversary
  stage, after asking, records agent.log() for an Agent that declares
  reports_log and None for any other, never calling log() of an Agent that
  does not declare it, whatever methods it has.

The suite, recipes, and components of the stage tests are SYNTHETIC fakes
(tests/core/adversary_support.py); every count and digest here is a
SYNTHETIC fixture value, never a measurement.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from adversary_support import (
    DIRECTION,
    MODEL_ID,
    PROJECT,
    SUITE,
    Outcome,
    World,
    clean_environment,
    constant,
    make_registry,
    observer,
    proposal,
    recipe_data,
    views,
    write_recipe,
    write_suite,
)

from lassi.core import parquet, record
from lassi.core.interfaces import Sampling
from lassi.core.record import BenchItem, ModelInfo, Provenance, Trial, from_dict, from_json, make_trial_id, to_json
from lassi.core.registry import Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir, write_trial
from lassi.core.trial_md import render_trial_md

TRIAL_ID = record.make_trial_id("adv-log", "scripted-fixture", "adv-stage", "omp-cpp", "scale", 1)
OTHER_ID = record.make_trial_id("adv-log", "scripted-fixture", "adv-stage", "omp-cpp", "fragile", 1)
STOPS = ("steps", "tokens", "context-exceeded", "no-proposal")
LOG_FIELDS = ("steps", "prompt_tokens", "completion_tokens", "malformed_replies", "dropped_proposals", "stop")
# A SYNTHETIC log: fixture counts, never measurements.
LOG_VALUES: dict[str, Any] = {
    "steps": 7,
    "prompt_tokens": 1234,
    "completion_tokens": 56,
    "malformed_replies": 2,
    "dropped_proposals": 3,
    "stop": "no-proposal",
}
LOGGING = "logging-adversary"
SILENT = "silent-adversary"


def need(name: str) -> Any:
    """Return lassi.core.record's attribute `name`, failing with the task's name when it does not exist yet."""
    assert hasattr(record, name), f"lassi.core.record has no {name} (task P16.4)"
    return getattr(record, name)


def agent_log(**changes: Any) -> Any:
    """Return the SYNTHETIC AgentLog of LOG_VALUES with `changes`."""
    return need("AgentLog")(**{**LOG_VALUES, **changes})


def base_trial(trial_id: str = TRIAL_ID) -> Trial:
    """Return a SYNTHETIC trial with no attempt and no adversary record."""
    item = record.parse_trial_id(trial_id).item
    return Trial(
        trial_id=trial_id,
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=None, dirty=None, device=None, sdk=None, date="2026-10-20T00:00:00+00:00"),
        bench_item=BenchItem(suite="adv-stage", item=item, split="unassigned", direction="omp-cpp"),
        model=ModelInfo(backend="scripted", id="scripted-fixture", sampling=Sampling(0.2, 0.9, 8)),
    )


def adversary_record(log: Any = None, agent: str = "llm") -> Any:
    """Return a SYNTHETIC AdversaryRecord with no divergence, one refusal, and `log` as its agent_log."""
    return record.AdversaryRecord(
        agent=agent,
        config={"budget": {"inputs": 4, "tokens": 20000, "steps": 6}, "backend": "replay"},
        proposals=2,
        refusals={"out-of-range": 1},
        runs=2,
        wall_s=1.5,
        divergences=[],
        contract_findings=[],
        agent_log=log,
    )


def taken(log: Any = None, trial_id: str = TRIAL_ID, agent: str = "llm") -> Trial:
    """Return base_trial with an adversary record whose agent_log is `log`."""
    return dataclasses.replace(base_trial(trial_id), adversary=adversary_record(log, agent))


def table_rows(text: str) -> list[list[str]]:
    """Return every Markdown table row of `text` as its cells, stripped of blanks and backticks."""
    rows = []
    for line in text.splitlines():
        if line.startswith("|") and line.endswith("|"):
            rows.append([cell.strip().strip("`") for cell in line[1:-1].split("|")])
    return rows


def agent_log_section(page: str) -> str:
    """Return the page's "### Agent log" section, up to the next heading of level 3 or less, or the end."""
    start = page.index("### Agent log\n")
    ends = [index for index in (page.find("\n### ", start + 1), page.find("\n## ", start + 1)) if index != -1]
    return page[start : min(ends)] if ends else page[start:]


# ---------------------------------------------------------------------------
# AgentLog


def test_the_stops_are_the_four_designed_ones() -> None:
    assert need("AGENT_STOPS") == STOPS


def test_agent_log_is_a_frozen_keyword_only_record_of_its_fields() -> None:
    cls = need("AgentLog")
    assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen
    specs = dataclasses.fields(cls)
    assert tuple(spec.name for spec in specs) == LOG_FIELDS
    assert all(spec.kw_only for spec in specs)
    assert all(spec.default is dataclasses.MISSING for spec in specs), "every field is required"


@pytest.mark.parametrize("stop", [*STOPS, None])
def test_agent_log_takes_each_stop_and_none(stop: str | None) -> None:
    assert agent_log(stop=stop).stop == stop


def test_malformed_replies_may_equal_the_steps() -> None:
    assert agent_log(steps=2, malformed_replies=2).malformed_replies == 2
    assert agent_log(steps=0, malformed_replies=0, prompt_tokens=0, completion_tokens=0, dropped_proposals=0).steps == 0


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        pytest.param({"steps": -1, "malformed_replies": 0}, "steps", id="steps-negative"),
        pytest.param({"prompt_tokens": -1}, "prompt_tokens", id="prompt-tokens-negative"),
        pytest.param({"completion_tokens": -1}, "completion_tokens", id="completion-tokens-negative"),
        pytest.param({"malformed_replies": -1}, "malformed_replies", id="malformed-negative"),
        pytest.param({"dropped_proposals": -1}, "dropped_proposals", id="dropped-negative"),
        pytest.param({"steps": True}, "steps", id="steps-bool"),
        pytest.param({"steps": 1.5}, "steps", id="steps-float"),
        pytest.param({"prompt_tokens": "12"}, "prompt_tokens", id="tokens-str"),
        pytest.param({"steps": 2, "malformed_replies": 3}, "malformed_replies", id="malformed-past-steps"),
        pytest.param({"stop": "budget"}, "stop", id="unknown-stop"),
        pytest.param({"stop": ""}, "stop", id="empty-stop"),
        pytest.param({"stop": 0}, "stop", id="stop-not-a-str"),
    ],
)
def test_agent_log_refuses_values_outside_its_rules(changes: dict[str, Any], field: str) -> None:
    with pytest.raises(ValueError) as caught:
        agent_log(**changes)
    assert field in str(caught.value), str(caught.value)


# ---------------------------------------------------------------------------
# AdversaryRecord.agent_log and the trial files


def test_agent_log_is_the_last_field_of_the_adversary_record_and_none_by_default() -> None:
    need("AgentLog")
    names = [spec.name for spec in dataclasses.fields(record.AdversaryRecord)]
    assert names[-1] == "agent_log", names
    assert adversary_record().agent_log is None
    without = record.AdversaryRecord(
        agent="fuzzer", config={}, proposals=0, refusals={}, runs=0, wall_s=0.0, divergences=[], contract_findings=[]
    )
    assert without.agent_log is None


def test_a_record_with_a_log_round_trips_through_json_and_the_trial_files(tmp_path: Path) -> None:
    trial = taken(agent_log())
    assert from_json(Trial, to_json(trial)) == trial
    store = TextStore(tmp_path / "run")
    directory = write_trial(trial, tmp_path / "run", store)
    assert read_trial(directory, store) == trial
    data = json.loads((directory / "trial.json").read_text(encoding="ascii"))
    assert data["adversary"]["agent_log"] == LOG_VALUES


def test_a_trial_json_written_before_the_field_loads_with_it_none(tmp_path: Path) -> None:
    need("AgentLog")
    data = json.loads(to_json(taken()))
    assert data["adversary"]["agent_log"] is None
    del data["adversary"]["agent_log"]
    assert from_dict(Trial, data).adversary.agent_log is None
    store = TextStore(tmp_path / "run")
    directory = write_trial(taken(), tmp_path / "run", store)
    stored = json.loads((directory / "trial.json").read_text(encoding="ascii"))
    del stored["adversary"]["agent_log"]
    (directory / "trial.json").write_bytes(json.dumps(stored, indent=2).encode("ascii") + b"\n")
    assert read_trial(directory, store).adversary.agent_log is None


# ---------------------------------------------------------------------------
# The Parquet adversary table


def test_the_adversary_table_has_an_agent_log_string_column_last() -> None:
    schema = parquet.SCHEMAS["adversary"]
    assert schema.names[-1] == "agent_log", schema.names
    assert schema.field("agent_log").type == pa.string()


def test_the_agent_log_column_holds_the_logs_sorted_key_json_or_null(tmp_path: Path) -> None:
    trials = [taken(agent_log()), taken(None, OTHER_ID, agent="fuzzer")]
    rows = {row["trial_id"]: row for row in parquet.trial_rows(trials)["adversary"]}
    assert rows[TRIAL_ID]["agent_log"] == json.dumps(LOG_VALUES, sort_keys=True, ensure_ascii=True)
    assert rows[OTHER_ID]["agent_log"] is None, "a record without a log (the fuzzer's) has a null agent_log"
    parquet.write_run_parquet(trials, tmp_path / "parquet")
    back = parquet.read_run_parquet(tmp_path / "parquet")["adversary"]
    assert back == parquet.trial_rows(trials)["adversary"]


# ---------------------------------------------------------------------------
# trial.md


def test_trial_md_shows_the_agent_log_only_when_it_is_set(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "store")
    without = render_trial_md(taken(None), store)
    assert "## Adversary\n" in without and "### Agent log" not in without
    page = render_trial_md(taken(agent_log()), store)
    assert page.index("## Adversary\n") < page.index("### Agent log\n"), "the log is part of the Adversary section"
    rows = table_rows(agent_log_section(page))
    assert rows[0] == ["Field", "Value"]
    for name, value in LOG_VALUES.items():
        assert [name, str(value)] in rows, f"no row {name} | {value}: {rows}"


def test_a_page_without_an_adversary_record_is_unchanged(tmp_path: Path) -> None:
    page = render_trial_md(base_trial(), TextStore(tmp_path / "store"))
    assert "## Adversary" not in page and "### Agent log" not in page


# ---------------------------------------------------------------------------
# The stage records agent.log() for an Agent that declares reports_log


def test_reports_log_is_a_capability_name() -> None:
    from lassi.core import capabilities

    assert getattr(capabilities, "REPORTS_LOG", None) == "reports_log"


def logging_adversary(world: World, name: str, capabilities: frozenset[str]) -> type:
    """Return an Agent class that proposes World's batches and keeps a SYNTHETIC log of its propose() calls.

    log() reports one step per propose() call and stop no-proposal once a
    call returned nothing; an Agent registered without reports_log must
    never be asked for it.
    """

    class LoggingAdversary:
        """Records each view, returns the next scripted batch, and logs its calls."""

        config_keys = frozenset({"budget"})

        def __init__(self, *, budget: Mapping[str, Any] | None = None) -> None:
            """Keep the budget section as given; no call made yet."""
            self.budget = budget
            self.calls = 0
            self.stopped = False

        def propose(self, view: Any) -> list[Any]:
            """Record `view` and return the next batch."""
            world.events.append(("view", view))
            self.calls += 1
            batch = world.batches.pop(0) if world.batches else []
            self.stopped = not batch
            return batch

        def log(self) -> Any:
            """Return the SYNTHETIC log of the calls made; fail when the Agent does not declare reports_log."""
            assert "reports_log" in self.capabilities, "log() was called on an Agent that does not declare reports_log"
            return record.AgentLog(
                steps=self.calls,
                prompt_tokens=10 * self.calls,
                completion_tokens=self.calls,
                malformed_replies=0,
                dropped_proposals=0,
                stop="no-proposal" if self.stopped else None,
            )

    LoggingAdversary.name = name  # type: ignore[attr-defined]
    LoggingAdversary.capabilities = capabilities  # type: ignore[attr-defined]
    return LoggingAdversary


def run_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, world: World, data: Mapping[str, Any],
    extra: Callable[[Registry], None],
) -> Outcome:
    """Run the recipe `data` over adv-stage's item scale with the world's registry plus `extra`'s components."""
    clean_environment(tmp_path, monkeypatch)
    bench = write_suite(tmp_path, monkeypatch)
    registry = make_registry(world)
    extra(registry)
    options = RunOptions(
        runs_root=tmp_path / "runs-root", run_id="agent-log", bench_root=bench, registry=registry,
        observer=observer(world),
    )
    run_dir = run_recipe(write_recipe(tmp_path, data, "recipe-agent-log"), options)
    trial_id = make_trial_id(PROJECT, MODEL_ID, SUITE, DIRECTION, "scale", 1)
    return Outcome(run_dir, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir)))


def test_the_stage_records_the_log_of_an_agent_that_declares_reports_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    need("AgentLog")
    world = World(candidate="large-zero", batches=[[proposal(constant(500.0))]])

    def extra(registry: Registry) -> None:
        caps = frozenset({"proposes_inputs", "reports_log"})
        registry.register("Agent", LOGGING, logging_adversary(world, LOGGING, caps))

    data = recipe_data(adversary={"kind": LOGGING, "budget": {"inputs": 4}})
    outcome = run_with(tmp_path, monkeypatch, world, data, extra)
    found = outcome.trial.adversary
    assert found is not None and found.agent == LOGGING
    assert len(views(world)) == 2, "one batch, then an empty reply"
    assert found.agent_log == record.AgentLog(
        steps=2, prompt_tokens=20, completion_tokens=2, malformed_replies=0, dropped_proposals=0, stop="no-proposal"
    ), "the log is read after the stage stopped asking"
    page = (trial_dir(outcome.run_dir, outcome.trial.trial_id) / "trial.md").read_text(encoding="ascii")
    assert "### Agent log" in page


def test_the_stage_records_no_log_for_an_agent_that_does_not_declare_reports_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    need("AgentLog")
    world = World(candidate="large-zero", batches=[[proposal(constant(500.0))]])

    def extra(registry: Registry) -> None:
        registry.register("Agent", SILENT, logging_adversary(world, SILENT, frozenset({"proposes_inputs"})))

    data = recipe_data(adversary={"kind": SILENT, "budget": {"inputs": 4}})
    found = run_with(tmp_path, monkeypatch, world, data, extra).trial.adversary
    assert found is not None and found.agent == SILENT
    assert found.agent_log is None, "the stage reads the capability, never the method"


def test_the_scripted_adversary_records_no_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    need("AgentLog")
    world = World(candidate="large-zero", batches=[[proposal(constant(500.0))]])
    found = run_with(tmp_path, monkeypatch, world, recipe_data(), lambda registry: None).trial.adversary
    assert found is not None and found.agent_log is None
