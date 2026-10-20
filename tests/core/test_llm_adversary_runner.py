"""End-to-end tests of the llm adversary through the runner, on the replay backend's recordings (task P16.4).

Bible: Adversarial Setups (item 1; the adversary rule), Agents And Judges
(the Agent interface), Project Recipes (the bindings block's adversary
line), Result Record (Trial.adversary), Agent Rules 6 and 14. Plan:
plans/p16-adversarial.md, P16.4 ("each parsed proposal goes through
check_inputs"; "its claims about a candidate are never recorded as
findings"; "a recording written to break the contract, which the gate also
runs").

The contract these tests fix: a recipe binds the llm adversary as

    adversary: {kind: llm, budget: {inputs, tokens, steps}, backend: replay,
                model: <id>, sampling: {temperature, top_p, max_tokens},
                prompts: adversary-v0, recording: <path under tests/fixtures/>}

and lassi.core.runner run_recipe carries it out with the adversary stage
over adv-stage's item scale (tests/core/adversary_support.py):

- with in-contract.json and the candidate large-zero, the agent proposes x
  filled with 500 and x of 1, 2, 3, 4, then stops; the stage runs both, and
  binary_io confirms one divergence, proposal 0, whose stored input is x
  filled with 500; the control candidate double gets none;
- with break-contract.json, every one of its eight proposals is refused by
  check_inputs under its reason and nothing runs;
- the trial's AdversaryRecord names the agent llm, copies its config with
  the recording as written and the loader's recording_sha256 (no host
  path), and carries the agent's AgentLog (two steps, stop no-proposal);
- the recordings' prose, which claims a divergence, appears in no file of
  the run: only the oracle's confirmation is recorded (Agent Rule 14).

Every recording, program, and suite here is SYNTHETIC; the replay backend's
token counts are word counts, never measurements. No real model is called.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from adversary_support import (
    DIRECTION,
    MODEL_ID,
    PROJECT,
    SUITE,
    Outcome,
    World,
    adversary_runs,
    clean_environment,
    constant,
    make_registry,
    observer,
    recipe_data,
    stored_array,
    write_recipe,
    write_suite,
)

from lassi import agents  # noqa: F401  (registers the Agents)
from lassi.core.record import make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir

REPO = Path(__file__).resolve().parents[2]
IN_CONTRACT = "tests/fixtures/adversary/recordings/in-contract.json"
BREAK_CONTRACT = "tests/fixtures/adversary/recordings/break-contract.json"
# The prose outside the json block of the recordings (tests/fixtures/adversary/recordings/README.md).
CLAIMS = ("SYNTHETIC claim", "SYNTHETIC text after the block", "SYNTHETIC reply written to break the contract")
BREAK_REASONS = {
    "dtype": 1,
    "shape": 1,
    "out-of-range": 1,
    "nan": 1,
    "extra-input": 1,
    "missing-input": 1,
    "too-many-elements": 1,
    "unreadable": 1,
}


@pytest.fixture(autouse=True)
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no gate variable, a TMPDIR under the test's directory, and its cwd there."""
    clean_environment(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the SYNTHETIC suite and return its bench root."""
    return write_suite(tmp_path, monkeypatch)


def section(recording: str, inputs: int) -> dict[str, Any]:
    """Return the llm adversary section over the replay backend and `recording`, with a budget of `inputs` sets."""
    return {
        "kind": "llm",
        "budget": {"inputs": inputs, "tokens": 20000, "steps": 6},
        "backend": "replay",
        "model": "adversary-replay",
        "sampling": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 2048},
        "prompts": "adversary-v0",
        "recording": recording,
    }


def run_llm(tmp_path: Path, bench: Path, world: World, adversary: Mapping[str, Any]) -> Outcome:
    """Run the test recipe with `adversary` over item scale, the llm Agent joining the world's registry."""
    assert "llm" in DEFAULT_REGISTRY.names("Agent"), "lassi.agents registers no Agent 'llm' (task P16.4)"
    registry = make_registry(world)
    registry.register("Agent", "llm", DEFAULT_REGISTRY.get("Agent", "llm").factory)
    options = RunOptions(
        runs_root=tmp_path / "runs-root", run_id="llm-run", bench_root=bench, registry=registry,
        observer=observer(world),
    )
    run_dir = run_recipe(write_recipe(tmp_path, recipe_data(adversary=dict(adversary)), "recipe-llm"), options)
    trial_id = make_trial_id(PROJECT, MODEL_ID, SUITE, DIRECTION, "scale", 1)
    return Outcome(run_dir, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir)))


def assert_no_claim_recorded(run_dir: Path) -> None:
    """Fail when any file of the run holds the recordings' prose."""
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            for claim in CLAIMS:
                assert claim.encode("ascii") not in raw, f"{path.relative_to(run_dir).as_posix()} records {claim!r}"


def test_the_in_contract_recording_finds_the_divergence_of_a_large_zero_candidate(tmp_path: Path, bench: Path) -> None:
    world = World(candidate="large-zero")
    outcome = run_llm(tmp_path, bench, world, section(IN_CONTRACT, 4))
    found = outcome.trial.adversary
    assert found is not None, "the stage took the trial"
    assert found.agent == "llm"
    assert (found.proposals, found.refusals, found.runs) == (2, {}, 4)
    assert len(adversary_runs(world)) == 4, "each accepted set runs the target reference and the candidate"
    assert [entry.proposal for entry in found.divergences] == [0], "x filled with 500 diverges; 1, 2, 3, 4 agrees"
    assert stored_array(outcome.run_dir, found.divergences[0].inputs["x"], tmp_path / "blobs-read") == constant(500.0)
    assert found.contract_findings == []


def test_the_record_carries_the_config_as_written_and_the_agent_log(tmp_path: Path, bench: Path) -> None:
    world = World(candidate="large-zero")
    outcome = run_llm(tmp_path, bench, world, section(IN_CONTRACT, 4))
    found = outcome.trial.adversary
    assert found is not None
    digest = hashlib.sha256((REPO / IN_CONTRACT).read_bytes()).hexdigest()
    expected = {key: value for key, value in section(IN_CONTRACT, 4).items() if key != "kind"}
    assert found.config == {**expected, "recording_sha256": digest}
    text = json.dumps(found.config, sort_keys=True)
    assert str(REPO) not in text and REPO.as_posix() not in text, "no host path enters the record"
    log = found.agent_log
    assert log is not None, "the llm adversary declares reports_log, so its log is in the record"
    assert (log.steps, log.malformed_replies, log.dropped_proposals, log.stop) == (2, 0, 0, "no-proposal")
    assert log.prompt_tokens > 0 and log.completion_tokens > 0
    assert_no_claim_recorded(outcome.run_dir)


def test_a_control_candidate_gets_no_divergence(tmp_path: Path, bench: Path) -> None:
    world = World(candidate="double")
    found = run_llm(tmp_path, bench, world, section(IN_CONTRACT, 4)).trial.adversary
    assert found is not None
    assert (found.proposals, found.runs, found.divergences, found.contract_findings) == (2, 4, [], [])


def test_every_proposal_of_the_break_contract_recording_is_refused_and_nothing_runs(
    tmp_path: Path, bench: Path
) -> None:
    world = World(candidate="large-zero")
    outcome = run_llm(tmp_path, bench, world, section(BREAK_CONTRACT, 10))
    found = outcome.trial.adversary
    assert found is not None
    assert found.proposals == 8
    assert found.refusals == BREAK_REASONS, "each proposal is refused under its own reason, before any run"
    assert found.runs == 0 and adversary_runs(world) == [], "a refused proposal runs nothing"
    assert found.divergences == [] and found.contract_findings == []
    assert found.agent_log is not None and found.agent_log.stop == "no-proposal"
    assert found.agent_log.steps == 2 and found.agent_log.malformed_replies == 0
    assert_no_claim_recorded(outcome.run_dir)
