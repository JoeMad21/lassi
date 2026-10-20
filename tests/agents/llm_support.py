"""Shared helpers for the tests of the llm adversary (task P16.4); not a test module.

- llm_module imports lassi.agents.llm inside each test, so a missing module
  fails each test on its own instead of the whole collection.
- CONTRACT is adv-stage's input contract (tests/core/adversary_support.py):
  one input x, f32, shape [4], each element in [-1000, 1000), no NaN and no
  infinity. view builds the AdversaryView a stage would give the agent
  (lassi.core.interfaces), with SYNTHETIC source and candidate texts.
- config gives the keyword arguments of a valid LLMAdversary bound to the
  replay backend over a committed recording (in-contract.json by default);
  a key set to MISSING is left out.
- FakeBackend stands in for the agent's backend (the attribute
  `agent.backend`, which the tests replace after building the agent): it
  returns scripted replies with fixed token counts, raises a scripted
  exception in place of a reply, and records each call's messages and
  sampling. Its counts are fixture values, never measurements.
- Recorder wraps a real backend and records each call and its completion.
- reply renders a reply holding one json block of proposal objects; inputs
  builds an input object of the reply form.

Every program text, reply, and count here is SYNTHETIC. No value in this
module is a measurement.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from lassi.core.interfaces import AdversaryView, Completion, Message, Proposal, ProposalOutcome, Sampling
from lassi.harness import ContractInput, InputContract

MODULE = "lassi.agents.llm"
REPO = Path(__file__).resolve().parents[2]
RECORDINGS_DIR = "tests/fixtures/adversary/recordings"
IN_CONTRACT = f"{RECORDINGS_DIR}/in-contract.json"
BREAK_CONTRACT = f"{RECORDINGS_DIR}/break-contract.json"
MALFORMED = f"{RECORDINGS_DIR}/malformed.json"
PROMPTS = "adversary-v0"
MODEL = "adversary-replay"
BUDGET = {"inputs": 4, "tokens": 20000, "steps": 6}
SAMPLING = {"temperature": 0.2, "top_p": 0.9, "max_tokens": 2048}
# The outcome code the driven views give an earlier proposal that ran: a token of the outcome form (task P16.2).
CODE = "agreed"
# The contract of adv-stage's x (tests/core/adversary_support.py: CONTRACT_LO, CONTRACT_HI).
CONTRACT = InputContract((ContractInput("x", "f32", (4,), -1000.0, 1000.0),))
SOURCE = {"main.cpp": "// SYNTHETIC source program\n// SYNTHETIC-PROGRAM double\n"}
CANDIDATE = {"main.cpp": "// SYNTHETIC candidate\n// SYNTHETIC-PROGRAM large-zero\n"}
# Left out of config(): the keyword is not passed at all.
MISSING = object()


def llm_module() -> ModuleType:
    """Import and return lassi.agents.llm, the module under test."""
    return importlib.import_module(MODULE)


def sha256_of(relative: str) -> str:
    """Return the sha256 of the committed file at `relative`, a path from the repository root."""
    return hashlib.sha256((REPO / relative).read_bytes()).hexdigest()


def recording_texts(relative: str) -> list[str]:
    """Return the reply texts of the committed recording at `relative`, in order."""
    data = json.loads((REPO / relative).read_text(encoding="ascii"))
    return [entry["text"] for entry in data["completions"]]


def config(recording: str = IN_CONTRACT, **changes: Any) -> dict[str, Any]:
    """Return the keyword arguments of a valid LLMAdversary over the replay backend, with `changes` applied.

    The recording's sha256 is the committed file's, as the loader enters
    it, unless `changes` states one. A change to MISSING leaves that
    keyword out.
    """
    found: dict[str, Any] = {
        "budget": dict(BUDGET),
        "backend": "replay",
        "model": MODEL,
        "sampling": dict(SAMPLING),
        "prompts": PROMPTS,
        "recording": recording,
    }
    if "recording_sha256" not in changes and recording is not MISSING:
        found["recording_sha256"] = sha256_of(recording)
    found.update(changes)
    return {key: value for key, value in found.items() if value is not MISSING}


def build(recording: str = IN_CONTRACT, **changes: Any) -> Any:
    """Return an LLMAdversary built from config(recording, **changes)."""
    return llm_module().LLMAdversary(**config(recording, **changes))


def view(
    budget_left: int = 4,
    earlier: Sequence[Proposal] = (),
    *,
    code: str = CODE,
    contract: InputContract = CONTRACT,
    source: Mapping[str, str] = SOURCE,
    candidate: Mapping[str, str] = CANDIDATE,
) -> AdversaryView:
    """Return the AdversaryView of one trial: the contract, the programs, the budget left, the earlier outcomes."""
    outcomes = tuple(ProposalOutcome(proposal, code) for proposal in earlier)
    return AdversaryView(
        contract=contract, source=source, candidate=candidate, budget_left=budget_left, outcomes=outcomes
    )


def inputs(name: str = "x", dtype: str = "f32", shape: Sequence[int] = (4,), **kind: Any) -> dict[str, Any]:
    """Return one input object of the reply form; `kind` holds values, fill, or pattern."""
    return {"name": name, "dtype": dtype, "shape": list(shape), **kind}


def reply(*proposals: Sequence[Mapping[str, Any]], before: str = "", after: str = "") -> str:
    """Return a reply whose one json block proposes `proposals`, each a list of input objects."""
    data = {"proposals": [{"inputs": list(items)} for items in proposals]}
    return f"{before}```json\n{json.dumps(data)}\n```\n{after}"


def proposal_json(items: Sequence[Mapping[str, Any]]) -> str:
    """Return the compact JSON the agent keeps for a proposal: its {"inputs": [...]} object, keys sorted."""
    return json.dumps({"inputs": list(items)}, sort_keys=True, separators=(",", ":"))


STOP_REPLY = reply()
MALFORMED_REPLY = "SYNTHETIC reply with no block."


@dataclass
class Call:
    """One call a fake or recorded backend received: the messages and the sampling."""

    messages: tuple[Message, ...]
    sampling: Sampling


@dataclass
class FakeBackend:
    """A SYNTHETIC backend: each call returns the next scripted reply, or raises it when it is an exception.

    Every completion carries the fixed counts prompt_tokens and
    completion_tokens, fixture values chosen so the budget tests are exact.
    A call past the script fails the test.
    """

    replies: list[Any]
    prompt_tokens: int = 100
    completion_tokens: int = 10
    calls: list[Call] = field(default_factory=list)
    name: str = "fake"
    capabilities: frozenset[str] = frozenset({"chat"})

    def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
        """Record the call and return (or raise) the next scripted reply."""
        self.calls.append(Call(tuple(messages), sampling))
        assert self.replies, f"the agent made call {len(self.calls)}, past the {len(self.calls) - 1} scripted"
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        return Completion(text=item, prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens)


@dataclass
class Recorder:
    """Wraps a real backend: forwards each call and records it with the completion it returned."""

    inner: Any
    calls: list[Call] = field(default_factory=list)
    completions: list[Completion] = field(default_factory=list)

    @property
    def model_id(self) -> str:
        """Return the wrapped backend's model id."""
        return self.inner.model_id

    def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
        """Forward the call, record it and its completion, and return the completion."""
        self.calls.append(Call(tuple(messages), sampling))
        completion = self.inner.complete(messages, sampling)
        self.completions.append(completion)
        return completion


def with_fake(agent: Any, replies: list[Any], **counts: int) -> FakeBackend:
    """Replace `agent.backend` with a FakeBackend of `replies` and return the fake."""
    assert hasattr(agent, "backend"), "the llm adversary keeps its LLMBackend as agent.backend (task P16.4)"
    fake = FakeBackend(list(replies), **counts)
    agent.backend = fake
    return fake


def log_tuple(log: Any) -> tuple[Any, ...]:
    """Return an AgentLog's fields in order."""
    return (
        log.steps,
        log.prompt_tokens,
        log.completion_tokens,
        log.malformed_replies,
        log.dropped_proposals,
        log.stop,
    )
