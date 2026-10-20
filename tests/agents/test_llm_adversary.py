"""Tests for the llm adversary: a registered Agent that asks an LLMBackend for proposals (task P16.4).

Bible: Agents And Judges (the Agent interface: backend, tools, versioned
prompts, budget), Adversarial Setups (item 1), Harness Contract (held-out
inputs never in a prompt), Component Interfaces (LLMBackend;
ContextExceeded), Model Serving (Serving Rules), Agent Rules 6, 12, and 14.
Plan: plans/p16-adversarial.md, P16.4; its Constraints (Replay recordings).

The contract these tests fix (design sections 3, 4, and 7):

- lassi.agents.llm defines LLMAdversary, registered in DEFAULT_REGISTRY as
  the Agent "llm" when lassi.agents is imported, with name "llm", the
  capabilities proposes_inputs and reports_log (lassi.core.capabilities
  REPORTS_LOG), and the config keys budget, backend, model, sampling,
  prompts, recording, and recording_sha256.
- LLMAdversary(*, budget, backend, model, sampling, prompts, recording,
  recording_sha256) refuses, with a ValueError naming the key
  (adversary.<key>): a budget that is not a mapping of exactly inputs,
  tokens, and steps, each an int of at least 1 (a bool refused; a missing
  or null one "is a required choice with no default"; another key "is not
  an llm budget key"); a backend that is not a registered LLMBackend, and
  the mock, which declares needs_reference; a model that is not a non-empty
  str; a sampling that is not exactly temperature (a finite number >= 0),
  top_p (a number in (0, 1]), and max_tokens (an int >= 1); a prompt set
  without propose.txt and malformed.txt; a recording missing for a backend
  that lists recording (replay), or given for one that does not; a
  recording_sha256 missing with a recording, given without one, or other
  than the sha256 of the bytes the backend read; and a recording path
  outside the recording rule (lassi.core.recipe recording_file). The
  backend is built as factory(model, recording=<file>) or factory(model).
- propose(view) takes exactly an AdversaryView (TypeError otherwise). It
  renders one user message with render_messages(view, prompts, shown) and
  asks the backend, one step per call, with the recipe's Sampling. Before
  each call, steps used >= budget.steps stops it at "steps" and tokens used
  >= budget.tokens at "tokens"; after each call tokens used grows by the
  backend's prompt_tokens and completion_tokens. ContextExceeded counts as
  a step with no tokens and stops it at "context-exceeded"; any other
  backend error (ServingError) propagates. A reply that parse_reply refuses
  counts as a malformed reply, and the agent asks again within the same
  call, the messages gaining the reply as an assistant message and
  malformed.txt (its reason and budget_left) as a user message. A reply of
  no proposals stops it at "no-proposal". A reply of more than
  view.budget_left proposals keeps the first budget_left and counts the
  rest as dropped. Once stopped, propose returns [] and asks nothing.
- log() returns an AgentLog (lassi.core.record) of its steps, token sums,
  malformed replies, dropped proposals, and stop (None until it stops
  itself). Text outside the json block is never recorded: the log holds
  counts only.
- The committed SYNTHETIC recordings under
  tests/fixtures/adversary/recordings/ are replay recordings with reply
  text only: in-contract.json's proposals pass check_inputs against
  adv-stage's contract, every proposal of break-contract.json is refused
  under its reason, and malformed.json's first six replies are refused by
  parse_reply. No JSON under tests/fixtures/adversary/ holding
  "completions" carries "messages" in any completion.

Every recording, reply, and count here is SYNTHETIC (tests/agents/
llm_support.py); the replay backend's token counts are word counts, never
measurements. No real model is called (OQ-068).
"""

# The tests rely on one attribute of LLMAdversary beyond the design's methods: `agent.backend`, the LLMBackend it
# built, which the tests read and replace with a fake (llm_support.with_fake) after building the agent.

from __future__ import annotations

import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from llm_support import (
    BREAK_CONTRACT,
    BUDGET,
    CANDIDATE,
    CONTRACT,
    IN_CONTRACT,
    MALFORMED,
    MALFORMED_REPLY,
    MISSING,
    MODEL,
    PROMPTS,
    REPO,
    SAMPLING,
    SOURCE,
    STOP_REPLY,
    Recorder,
    build,
    config,
    inputs,
    llm_module,
    log_tuple,
    proposal_json,
    recording_texts,
    reply,
    sha256_of,
    view,
    with_fake,
)

from lassi import agents  # noqa: F401  (registers the Agents)
from lassi.core import record
from lassi.core.adversary_stage import staged_inputs
from lassi.core.interfaces import AdversaryView, ContextExceeded, Message, Proposal, ProposalOutcome, Sampling
from lassi.core.registry import DEFAULT_REGISTRY, Binding, RegistryError, check_bindings
from lassi.harness import InputContractError, LassiArray, check_inputs
from lassi.llm import ReplayBackend, ServingError
from lassi.prompts import render

KIND = "llm"
CAPABILITIES = frozenset({"proposes_inputs", "reports_log"})
CONFIG_KEYS = frozenset({"budget", "backend", "model", "sampling", "prompts", "recording", "recording_sha256"})
# The reasons check_inputs gives break-contract.json's eight proposals, in order (its README).
BREAK_REASONS = (
    "dtype",
    "shape",
    "out-of-range",
    "nan",
    "extra-input",
    "missing-input",
    "too-many-elements",
    "unreadable",
)
FIXTURES = REPO / "tests" / "fixtures" / "adversary"
X_500 = [inputs(fill=500)]
X_VALUES = [inputs(values=[1, 2, 3, 4])]
X_ONE = [inputs(fill=1)]


@pytest.fixture(autouse=True)
def away_from_the_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Work from tmp_path, so a recording read against the working directory, not the repository root, is missed."""
    monkeypatch.chdir(tmp_path)


def agent_log_class() -> type:
    """Return lassi.core.record.AgentLog, failing with the task's name when it does not exist yet."""
    assert hasattr(record, "AgentLog"), "lassi.core.record has no AgentLog (task P16.4)"
    return record.AgentLog


def names_key(message: str, key: str) -> bool:
    """Return True when `message` names `key` itself (or a key under it), not a longer key such as key_sha256."""
    return re.search(re.escape(key) + r"(?!\w)", message) is not None


def x_array(*values: float) -> LassiArray:
    """Return the f32 array x of `values`."""
    return LassiArray("x", "f32", (len(values),), struct.pack(f"<{len(values)}f", *values))


def reason_of(found: Proposal, directory: Path) -> str:
    """Return the check_inputs reason a proposal is refused under, staged as the adversary stage stages it."""
    with pytest.raises(InputContractError) as info:
        check_inputs(CONTRACT, staged_inputs(found.arrays, directory))
    return info.value.reason


# ---------------------------------------------------------------------------
# Registration


def test_the_llm_agent_is_registered_with_its_capabilities_and_config_keys() -> None:
    module = llm_module()
    assert KIND in DEFAULT_REGISTRY.names("Agent"), "no Agent is registered as 'llm' (task P16.4)"
    entry = DEFAULT_REGISTRY.get("Agent", KIND)
    assert entry.factory is module.LLMAdversary
    assert module.LLMAdversary.name == KIND
    assert entry.capabilities == CAPABILITIES
    assert entry.config_keys == CONFIG_KEYS


def test_reports_log_is_a_capability_name() -> None:
    from lassi.core import capabilities

    assert getattr(capabilities, "REPORTS_LOG", None) == "reports_log"


def test_importing_lassi_agents_registers_the_llm_agent() -> None:
    code = (
        "import lassi.agents\n"
        "from lassi.core.registry import DEFAULT_REGISTRY\n"
        "print(sorted(DEFAULT_REGISTRY.names('Agent')))\n"
    )
    found = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    assert "'llm'" in found.stdout, found.stdout + found.stderr


def test_the_adversary_section_binds_it_with_its_keys_and_refuses_another_key() -> None:
    llm_module()
    settings = {key: "SYNTHETIC" for key in CONFIG_KEYS}
    good = Binding("Agent", KIND, "adversary.kind", settings, needs=frozenset({"proposes_inputs"}))
    check_bindings([good], DEFAULT_REGISTRY)
    bad = Binding("Agent", KIND, "adversary.kind", {**settings, "seed": 1}, needs=frozenset({"proposes_inputs"}))
    with pytest.raises(RegistryError) as info:
        check_bindings([bad], DEFAULT_REGISTRY)
    assert "seed" in str(info.value) and "adversary" in str(info.value)


# ---------------------------------------------------------------------------
# Building: the backend and every refusal


def test_a_valid_config_builds_the_replay_backend_over_the_recording() -> None:
    agent = build()
    assert isinstance(agent.backend, ReplayBackend)
    assert agent.backend.model_id == MODEL
    assert agent.backend.recording_sha256 == sha256_of(IN_CONTRACT)
    assert isinstance(agent.log(), agent_log_class())
    assert log_tuple(agent.log()) == (0, 0, 0, 0, 0, None), "a new agent has made no call and never stopped"


@pytest.mark.parametrize("backend", ["openai_compat", "ollama", "hf_local"])
def test_a_served_or_in_process_backend_is_refused_naming_the_key(backend: str) -> None:
    # Scripted backends only (OQ-068): a backend that replays no recording is refused when built, before any request,
    # so no default server address is ever used; a served or in-process model is task P16.11.
    module = llm_module()
    with pytest.raises(ValueError) as info:
        module.LLMAdversary(**config(backend=backend, recording=MISSING, recording_sha256=MISSING))
    message = str(info.value)
    assert names_key(message, "adversary.backend") and "P16.11" in message, message


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"budget": {"inputs": 1, "tokens": 1, "steps": 1}}, id="budget-of-ones"),
        pytest.param({"sampling": {"temperature": 0, "top_p": 1, "max_tokens": 1}}, id="sampling-edges-as-ints"),
        pytest.param({"sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1}}, id="sampling-edges"),
        pytest.param({"sampling": {"temperature": 2, "top_p": 0.01, "max_tokens": 8192}}, id="sampling-wide"),
    ],
)
def test_edge_values_are_accepted(changes: dict[str, Any]) -> None:
    build(**changes)


BUDGET_REFUSALS = [
    pytest.param({"budget": MISSING}, "adversary.budget", id="no-budget"),
    pytest.param({"budget": None}, "adversary.budget", id="null-budget"),
    pytest.param({"budget": [4, 20000, 6]}, "adversary.budget", id="budget-not-a-mapping"),
    pytest.param({"budget": {"inputs": 4, "steps": 6}}, "adversary.budget.tokens", id="no-tokens"),
    pytest.param({"budget": {"inputs": 4, "tokens": 20000}}, "adversary.budget.steps", id="no-steps"),
    pytest.param({"budget": {"tokens": 20000, "steps": 6}}, "adversary.budget.inputs", id="no-inputs"),
    pytest.param({"budget": {**BUDGET, "tokens": None}}, "adversary.budget.tokens", id="null-tokens"),
    pytest.param({"budget": {**BUDGET, "seed": 1}}, "adversary.budget.seed", id="extra-key"),
    pytest.param({"budget": {**BUDGET, "inputs": 0}}, "adversary.budget.inputs", id="inputs-zero"),
    pytest.param({"budget": {**BUDGET, "inputs": -1}}, "adversary.budget.inputs", id="inputs-negative"),
    pytest.param({"budget": {**BUDGET, "inputs": True}}, "adversary.budget.inputs", id="inputs-bool"),
    pytest.param({"budget": {**BUDGET, "inputs": 1.5}}, "adversary.budget.inputs", id="inputs-float"),
    pytest.param({"budget": {**BUDGET, "inputs": "4"}}, "adversary.budget.inputs", id="inputs-str"),
    pytest.param({"budget": {**BUDGET, "tokens": 0}}, "adversary.budget.tokens", id="tokens-zero"),
    pytest.param({"budget": {**BUDGET, "tokens": 2.0}}, "adversary.budget.tokens", id="tokens-float"),
    pytest.param({"budget": {**BUDGET, "steps": True}}, "adversary.budget.steps", id="steps-bool"),
    pytest.param({"budget": {**BUDGET, "steps": 0}}, "adversary.budget.steps", id="steps-zero"),
]

SETTING_REFUSALS = [
    pytest.param({"backend": MISSING}, "adversary.backend", id="no-backend"),
    pytest.param({"backend": ""}, "adversary.backend", id="empty-backend"),
    pytest.param({"backend": 5}, "adversary.backend", id="backend-not-a-str"),
    pytest.param({"backend": "no-such-backend"}, "adversary.backend", id="unknown-backend"),
    pytest.param(
        {"backend": "mock", "recording": MISSING, "recording_sha256": MISSING}, "adversary.backend", id="the-mock"
    ),
    pytest.param({"model": MISSING}, "adversary.model", id="no-model"),
    pytest.param({"model": ""}, "adversary.model", id="empty-model"),
    pytest.param({"model": 5}, "adversary.model", id="model-not-a-str"),
    pytest.param({"sampling": MISSING}, "adversary.sampling", id="no-sampling"),
    pytest.param({"sampling": [0.2, 0.9, 2048]}, "adversary.sampling", id="sampling-not-a-mapping"),
    pytest.param({"sampling": {"top_p": 0.9, "max_tokens": 2048}}, "adversary.sampling.temperature", id="no-temp"),
    pytest.param({"sampling": {"temperature": 0.2, "max_tokens": 2048}}, "adversary.sampling.top_p", id="no-top-p"),
    pytest.param({"sampling": {"temperature": 0.2, "top_p": 0.9}}, "adversary.sampling.max_tokens", id="no-max"),
    pytest.param({"sampling": {**SAMPLING, "seed": 1}}, "adversary.sampling.seed", id="sampling-extra-key"),
    pytest.param({"sampling": {**SAMPLING, "temperature": -0.1}}, "adversary.sampling.temperature", id="temp-neg"),
    pytest.param({"sampling": {**SAMPLING, "temperature": math.nan}}, "adversary.sampling.temperature", id="temp-nan"),
    pytest.param({"sampling": {**SAMPLING, "temperature": math.inf}}, "adversary.sampling.temperature", id="temp-inf"),
    pytest.param({"sampling": {**SAMPLING, "temperature": True}}, "adversary.sampling.temperature", id="temp-bool"),
    pytest.param({"sampling": {**SAMPLING, "temperature": "0.2"}}, "adversary.sampling.temperature", id="temp-str"),
    pytest.param({"sampling": {**SAMPLING, "top_p": 0}}, "adversary.sampling.top_p", id="top-p-zero"),
    pytest.param({"sampling": {**SAMPLING, "top_p": -0.5}}, "adversary.sampling.top_p", id="top-p-negative"),
    pytest.param({"sampling": {**SAMPLING, "top_p": 1.5}}, "adversary.sampling.top_p", id="top-p-past-one"),
    pytest.param({"sampling": {**SAMPLING, "top_p": True}}, "adversary.sampling.top_p", id="top-p-bool"),
    pytest.param({"sampling": {**SAMPLING, "top_p": math.nan}}, "adversary.sampling.top_p", id="top-p-nan"),
    pytest.param({"sampling": {**SAMPLING, "max_tokens": 0}}, "adversary.sampling.max_tokens", id="max-zero"),
    pytest.param({"sampling": {**SAMPLING, "max_tokens": 1.5}}, "adversary.sampling.max_tokens", id="max-float"),
    pytest.param({"sampling": {**SAMPLING, "max_tokens": True}}, "adversary.sampling.max_tokens", id="max-bool"),
    pytest.param({"prompts": MISSING}, "adversary.prompts", id="no-prompts"),
    pytest.param({"prompts": ""}, "adversary.prompts", id="empty-prompts"),
    pytest.param({"prompts": 5}, "adversary.prompts", id="prompts-not-a-str"),
    pytest.param({"prompts": "no-such-set"}, "adversary.prompts", id="unknown-set"),
    pytest.param({"prompts": "p0-smoke"}, "adversary.prompts", id="set-without-the-two-prompts"),
    pytest.param({"prompts": "../adversary-v0"}, "adversary.prompts", id="prompts-path"),
]

RECORDING_REFUSALS = [
    pytest.param({"recording": MISSING, "recording_sha256": MISSING}, "adversary.recording", id="replay-no-recording"),
    pytest.param({"recording_sha256": MISSING}, "adversary.recording_sha256", id="recording-without-sha256"),
    pytest.param({"recording_sha256": sha256_of(MALFORMED)}, "adversary.recording_sha256", id="sha256-mismatch"),
    pytest.param({"recording_sha256": "0" * 64}, "adversary.recording_sha256", id="sha256-of-nothing"),
    pytest.param(
        {"recording": str(REPO / IN_CONTRACT), "recording_sha256": sha256_of(IN_CONTRACT)},
        "adversary.recording",
        id="absolute-path",
    ),
    pytest.param(
        {"recording": IN_CONTRACT.replace("/", "\\"), "recording_sha256": sha256_of(IN_CONTRACT)},
        "adversary.recording",
        id="backslashes",
    ),
    pytest.param(
        {"recording": "pyproject.toml", "recording_sha256": sha256_of("pyproject.toml")},
        "adversary.recording",
        id="outside-tests-fixtures",
    ),
    pytest.param(
        {"recording": "tests/fixtures/adversary/recordings/no-such.json", "recording_sha256": "0" * 64},
        "adversary.recording",
        id="missing-file",
    ),
    pytest.param(
        {"recording": "tests/fixtures/adversary/../adversary/recordings/in-contract.json",
         "recording_sha256": sha256_of(IN_CONTRACT)},
        "adversary.recording",
        id="dot-dot",
    ),
    pytest.param({"recording": 5, "recording_sha256": "0" * 64}, "adversary.recording", id="not-a-string"),
    pytest.param({"backend": "openai_compat"}, "adversary.backend", id="a-backend-without-a-recording"),
    pytest.param({"recording": MISSING, "recording_sha256": "0" * 64}, "adversary.recording_sha256",
                 id="sha256-without-recording"),
]


@pytest.mark.parametrize(("changes", "key"), BUDGET_REFUSALS + SETTING_REFUSALS + RECORDING_REFUSALS)
def test_a_bad_setting_is_refused_with_a_value_error_naming_its_key(changes: dict[str, Any], key: str) -> None:
    module = llm_module()
    with pytest.raises(ValueError) as info:
        module.LLMAdversary(**config(**changes))
    assert names_key(str(info.value), key), f"the refusal does not name {key}: {info.value}"


@pytest.mark.parametrize("key", ["inputs", "tokens", "steps"])
def test_a_missing_budget_key_is_a_required_choice_with_no_default(key: str) -> None:
    module = llm_module()
    budget = {name: value for name, value in BUDGET.items() if name != key}
    with pytest.raises(ValueError) as info:
        module.LLMAdversary(**config(budget=budget))
    assert f"adversary.budget.{key} is a required choice with no default" in str(info.value), str(info.value)


def test_another_budget_key_is_not_an_llm_budget_key() -> None:
    module = llm_module()
    with pytest.raises(ValueError) as info:
        module.LLMAdversary(**config(budget={**BUDGET, "seed": 1}))
    assert "adversary.budget.seed is not an llm budget key" in str(info.value), str(info.value)


# ---------------------------------------------------------------------------
# propose(): the replay recordings


def test_propose_takes_only_an_adversary_view() -> None:
    agent = build()
    fake = with_fake(agent, [STOP_REPLY])
    for wrong in (SimpleNamespace(**vars(view())), {"budget_left": 4}, None):
        with pytest.raises(TypeError):
            agent.propose(wrong)
    assert fake.calls == [], "a refused view asks no model"


def test_the_in_contract_recording_proposes_its_sets_then_stops() -> None:
    agent = build()
    recorder = Recorder(agent.backend)
    agent.backend = recorder
    first = agent.propose(view(4))
    assert type(first) is list
    assert first == [Proposal((x_array(500, 500, 500, 500),)), Proposal((x_array(1, 2, 3, 4),))]
    assert agent.propose(view(2, first)) == []
    log = agent.log()
    assert (log.steps, log.malformed_replies, log.dropped_proposals, log.stop) == (2, 0, 0, "no-proposal")
    assert log.prompt_tokens == sum(item.prompt_tokens for item in recorder.completions) > 0
    assert log.completion_tokens == sum(item.completion_tokens for item in recorder.completions) > 0
    assert [call.sampling for call in recorder.calls] == [Sampling(0.2, 0.9, 2048)] * 2
    assert agent.propose(view(2, first)) == [] and len(recorder.calls) == 2, "a stopped agent asks nothing"


def test_the_in_contract_recording_proposes_only_in_contract_sets(tmp_path: Path) -> None:
    found = llm_module().parse_reply(recording_texts(IN_CONTRACT)[0])
    for index, item in enumerate(found):
        check_inputs(CONTRACT, staged_inputs(item.arrays, tmp_path / f"p{index}"))
    assert llm_module().parse_reply(recording_texts(IN_CONTRACT)[1]) == ()


def test_every_proposal_of_the_break_contract_recording_is_refused_under_its_reason(tmp_path: Path) -> None:
    found = llm_module().parse_reply(recording_texts(BREAK_CONTRACT)[0])
    reasons = tuple(reason_of(item, tmp_path / f"p{index}") for index, item in enumerate(found))
    assert reasons == BREAK_REASONS
    assert llm_module().parse_reply(recording_texts(BREAK_CONTRACT)[1]) == ()


def test_the_break_contract_recording_gives_every_proposal_to_the_stage() -> None:
    agent = build(BREAK_CONTRACT, budget={**BUDGET, "inputs": 10})
    found = agent.propose(view(10))
    assert len(found) == len(BREAK_REASONS), "a proposal outside the contract is data; the stage refuses it"
    outcomes = [ProposalOutcome(item, reason) for item, reason in zip(found, BREAK_REASONS, strict=True)]
    later = AdversaryView(contract=CONTRACT, source=SOURCE, candidate=CANDIDATE, budget_left=2, outcomes=outcomes)
    assert agent.propose(later) == []
    assert log_tuple(agent.log())[3:] == (0, 0, "no-proposal")


def test_each_malformed_reply_of_the_recording_is_refused_by_parse_reply() -> None:
    module = llm_module()
    texts = recording_texts(MALFORMED)
    assert len(texts) == 7
    for text in texts[:6]:
        with pytest.raises(module.ReplyError):
            module.parse_reply(text)
    assert module.parse_reply(texts[6]) == ()


def test_the_malformed_recording_gives_no_proposal_and_counts_each_reply() -> None:
    agent = build(MALFORMED, budget={**BUDGET, "steps": 7})
    assert agent.propose(view(4)) == [], "a malformed reply gives no proposal; the agent asks again in the same call"
    log = agent.log()
    assert (log.steps, log.malformed_replies, log.dropped_proposals, log.stop) == (7, 6, 0, "no-proposal")


def test_malformed_replies_that_use_up_the_steps_stop_at_steps() -> None:
    agent = build(MALFORMED, budget={**BUDGET, "steps": 3})
    assert agent.propose(view(4)) == []
    log = agent.log()
    assert (log.steps, log.malformed_replies, log.stop) == (3, 3, "steps")


# ---------------------------------------------------------------------------
# propose(): the messages


def test_the_first_call_sends_the_rendered_view_alone() -> None:
    agent = build()
    fake = with_fake(agent, [STOP_REPLY])
    shown = view(4)
    assert agent.propose(shown) == []
    (call,) = fake.calls
    assert list(call.messages) == list(llm_module().render_messages(shown, PROMPTS, []))
    assert call.sampling == Sampling(0.2, 0.9, 2048)


def test_a_malformed_reply_is_answered_with_the_malformed_prompt_in_the_same_call() -> None:
    module = llm_module()
    bad_json = "```json\n{\"proposals\": [\n```"
    agent = build()
    fake = with_fake(agent, [MALFORMED_REPLY, bad_json, reply(X_500)])
    found = agent.propose(view(3))
    assert found == [Proposal((x_array(500, 500, 500, 500),))]
    assert len(fake.calls) == 3
    for index, text in enumerate((MALFORMED_REPLY, bad_json)):
        with pytest.raises(module.ReplyError) as info:
            module.parse_reply(text)
        answer = render(PROMPTS, "malformed", {"reason": str(info.value), "budget_left": "3"})
        expected = (*fake.calls[index].messages, Message("assistant", text), Message("user", answer))
        assert fake.calls[index + 1].messages == expected
    assert log_tuple(agent.log()) == (3, 300, 30, 2, 0, None)


def test_the_next_view_shows_the_agents_own_proposals_as_compact_json() -> None:
    module = llm_module()
    agent = build()
    fake = with_fake(agent, [reply(X_500, X_VALUES), STOP_REPLY])
    first = agent.propose(view(4))
    later = view(2, first, code="diverged")
    assert agent.propose(later) == []
    shown = [(first[0], proposal_json(X_500)), (first[1], proposal_json(X_VALUES))]
    assert list(fake.calls[1].messages) == list(module.render_messages(later, PROMPTS, shown))
    text = fake.calls[1].messages[0].content
    assert f"- proposal 0: diverged\n  proposed: {proposal_json(X_500)}" in text
    assert f"- proposal 1: diverged\n  proposed: {proposal_json(X_VALUES)}" in text


def test_an_outcome_whose_proposal_is_not_the_agents_shows_no_json() -> None:
    module = llm_module()
    agent = build()
    fake = with_fake(agent, [reply(X_500), STOP_REPLY])
    agent.propose(view(4))
    other = Proposal((x_array(7, 7, 7, 7),))
    later = view(3, [other])
    assert agent.propose(later) == []
    text = fake.calls[1].messages[0].content
    assert "- proposal 0: agreed" in text.splitlines()
    assert list(fake.calls[1].messages) == list(module.render_messages(later, PROMPTS, [])), (
        "an outcome whose proposal is not the agent's own renders as if the agent had shown nothing"
    )


# ---------------------------------------------------------------------------
# propose(): budgets and stops


def test_zero_proposals_stop_at_no_proposal_and_nothing_more_is_asked() -> None:
    agent = build()
    fake = with_fake(agent, [STOP_REPLY])
    assert agent.propose(view(4)) == []
    assert log_tuple(agent.log()) == (1, 100, 10, 0, 0, "no-proposal")
    assert agent.propose(view(4)) == [] and len(fake.calls) == 1


def test_proposals_past_the_budget_left_are_dropped_and_counted() -> None:
    agent = build()
    with_fake(agent, [reply(X_500, X_VALUES, X_ONE), reply(X_ONE, X_500, X_VALUES)])
    first = agent.propose(view(2))
    assert first == [Proposal((x_array(500, 500, 500, 500),)), Proposal((x_array(1, 2, 3, 4),))]
    assert agent.log().dropped_proposals == 1
    second = agent.propose(view(1, first))
    assert second == [Proposal((x_array(1, 1, 1, 1),))]
    assert log_tuple(agent.log()) == (2, 200, 20, 0, 3, None)


def test_the_step_budget_stops_the_agent_before_a_call_past_it() -> None:
    agent = build(budget={**BUDGET, "steps": 2})
    fake = with_fake(agent, [reply(X_500), reply(X_VALUES)])
    first = agent.propose(view(4))
    second = agent.propose(view(3, first))
    assert agent.propose(view(2, first + second)) == []
    assert len(fake.calls) == 2
    assert log_tuple(agent.log()) == (2, 200, 20, 0, 0, "steps")


def test_the_token_budget_stops_the_agent_once_the_tokens_used_reach_it() -> None:
    agent = build(budget={**BUDGET, "tokens": 150})
    fake = with_fake(agent, [reply(X_500), reply(X_VALUES)])
    first = agent.propose(view(4))
    assert first, "110 tokens used is below the budget of 150: the second call goes ahead"
    second = agent.propose(view(3, first))
    assert second, "the total may pass the budget by the last call's tokens (a known case)"
    assert agent.propose(view(2, first + second)) == []
    assert len(fake.calls) == 2
    assert log_tuple(agent.log()) == (2, 200, 20, 0, 0, "tokens")


def test_tokens_used_equal_to_the_budget_stop_the_agent() -> None:
    agent = build(budget={**BUDGET, "tokens": 110})
    fake = with_fake(agent, [reply(X_500)])
    first = agent.propose(view(4))
    assert agent.propose(view(3, first)) == []
    assert len(fake.calls) == 1
    assert log_tuple(agent.log()) == (1, 100, 10, 0, 0, "tokens")


def test_context_exceeded_ends_the_turn_as_a_step_with_no_tokens() -> None:
    agent = build()
    fake = with_fake(agent, [ContextExceeded(5000, 2048, 4096)])
    assert agent.propose(view(4)) == []
    assert log_tuple(agent.log()) == (1, 0, 0, 0, 0, "context-exceeded")
    assert agent.propose(view(4)) == [] and len(fake.calls) == 1, "a stopped agent asks nothing"


def test_context_exceeded_after_a_malformed_reply_keeps_the_earlier_counts() -> None:
    agent = build()
    with_fake(agent, [MALFORMED_REPLY, ContextExceeded(5000, 2048, 4096)])
    assert agent.propose(view(4)) == []
    assert log_tuple(agent.log()) == (2, 100, 10, 1, 0, "context-exceeded")


def test_another_backend_error_propagates() -> None:
    agent = build()
    with_fake(agent, [ServingError("SYNTHETIC: the recording is used up")])
    with pytest.raises(ServingError):
        agent.propose(view(4))


def test_the_replay_backend_running_out_propagates() -> None:
    agent = build()
    first = agent.propose(view(4))
    agent.backend = ReplayBackend(MODEL, recording=REPO / IN_CONTRACT)
    agent.backend.complete([], Sampling(0.2, 0.9, 2048))
    agent.backend.complete([], Sampling(0.2, 0.9, 2048))
    with pytest.raises(ServingError):
        agent.propose(view(2, first))


# ---------------------------------------------------------------------------
# Claims and the log


def test_text_outside_the_block_changes_nothing_and_is_never_recorded() -> None:
    claim = "SYNTHETIC claim: x = 500 breaks the candidate; this is a confirmed bug."
    plain, claimed = build(), build()
    with_fake(plain, [reply(X_500)])
    with_fake(claimed, [reply(X_500, before=claim + "\n", after=claim)])
    assert plain.propose(view(4)) == claimed.propose(view(4))
    assert plain.log() == claimed.log()
    assert claim not in repr(claimed.log()) and "SYNTHETIC claim" not in json.dumps(vars(claimed.log()))


def test_the_log_holds_counts_and_a_stop_only() -> None:
    import dataclasses

    names = tuple(spec.name for spec in dataclasses.fields(agent_log_class()))
    assert names == ("steps", "prompt_tokens", "completion_tokens", "malformed_replies", "dropped_proposals", "stop")


# ---------------------------------------------------------------------------
# The fixtures


def test_the_recordings_are_synthetic_replay_recordings_of_reply_text_only() -> None:
    for relative in (IN_CONTRACT, BREAK_CONTRACT, MALFORMED):
        raw = (REPO / relative).read_bytes()
        assert raw.isascii() and b"\r" not in raw, f"{relative} is not plain ASCII with LF line ends"
        data = json.loads(raw)
        assert data["synthetic"] is True
        assert all(set(entry) == {"text"} for entry in data["completions"]), f"{relative} holds more than text"
        ReplayBackend(MODEL, recording=REPO / relative)


def test_the_recordings_readme_labels_them_synthetic() -> None:
    text = (FIXTURES / "recordings" / "README.md").read_text(encoding="ascii")
    assert "SYNTHETIC" in text and "not model output" in text


def test_no_recording_under_the_adversary_fixtures_carries_messages() -> None:
    # The P16 plan's Constraints, Replay recordings line: reply text only, since prompts may show upstream text.
    found = []
    for path in sorted(FIXTURES.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "completions" in data:
            found.append(path)
            for index, entry in enumerate(data["completions"]):
                assert "messages" not in entry, f"{path.relative_to(REPO).as_posix()} completion {index} has messages"
    names = {path.name for path in found}
    assert {"in-contract.json", "break-contract.json", "malformed.json"} <= names, names
