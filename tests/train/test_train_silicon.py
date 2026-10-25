"""Tests that training never reaches Tenstorrent silicon (task P17.16).

Bible: Agent Rule 9 ("RL rewards never execute on silicon; silicon runs are
evaluation runs a human starts"); Training Module, Safeguards ("Rewards
execute on ttsim or native only, inside the sandbox; never on Tenstorrent
silicon"); OQ-075 (6a), approved: training methods and reward stages refuse
tt_silicon outright; plans/p17-portable.md, task P17.16, Human start.

The contract these tests fix (lassi.train.run):

- `lassi train` takes no --silicon flag (tests/core/test_runner_silicon.py)
  and TrainOptions has no silicon field (tests/train/test_train_run.py pins
  its fields), so no train run carries the human start.
- A train recipe whose reward.executor names an Executor that declares
  `silicon` (lassi.core.capabilities SILICON; the real tt_silicon executor,
  or any other that declares it) is refused before any directory exists and
  before the Trainer is built, and the refusal names that executor and
  Agent Rule 9. Today the layer refuses every reward section as a key it
  does not carry out (NOT_CARRIED_OUT, for P7 and P8); the silicon refusal
  must name the rule itself, so it stands once the reward is carried out.
  The check reads the Executor's registry entry, never its name (Agent
  Rule 3): an executor that does not declare the capability gets only the
  not-carried-out refusal.

Every component is the train layer's fake Trainer (train_fakes) in a test
Registry, beside the real tt_silicon executor class, which is never built.
No card, node, or process is touched. No value here is a measurement.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from train_fakes import Log, Setup, fake_registry, recipe_data

from lassi.core.registry import Registry
from lassi.core.runner import RunError

# The capability literal, so this module collects before lassi.core.capabilities defines SILICON.
SILICON = "silicon"


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    """Return the test's directories and log."""
    return Setup(tmp_path)


def tt_silicon_module() -> ModuleType:
    """Import lassi.executors.tt_silicon, failing the test clearly while it does not exist."""
    try:
        return importlib.import_module("lassi.executors.tt_silicon")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.executors.tt_silicon does not exist yet (task P17.16): {error}")


def never_built(name: str, capabilities: frozenset[str]) -> type:
    """Return an Executor class `name` with `capabilities`; building it fails the test."""

    def refuse(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"Executor {name} was built by the train layer")

    namespace = {"__doc__": f"Fake Executor {name}; never built.", "name": name, "capabilities": capabilities}
    return type(f"Fake_{name}", (), {**namespace, "config_keys": frozenset(), "__init__": refuse})


def registry_with(log: Log, *executors: tuple[str, type]) -> Registry:
    """Return the fake Trainer's registry with `executors` registered as Executors."""
    registry = fake_registry(log)
    for name, cls in executors:
        registry.register("Executor", name, cls)
    return registry


def reward(executor: str) -> dict[str, Any]:
    """Return a reward section that names `executor`."""
    return {"profile": "df-v0", "executor": executor, "cache": True}


def refusal(setup: Setup, registry: Registry, executor: str) -> str:
    """Run a train recipe whose reward runs on `executor`; assert it is refused before anything; return why."""
    with pytest.raises(RunError) as refused:
        setup.run(recipe_data(reward=reward(executor)), registry=registry)
    assert not setup.runs_root.exists(), "the refusal comes before any directory exists"
    assert "build" not in setup.log.events and "train" not in setup.log.events, setup.log.events
    return str(refused.value)


def test_a_reward_on_the_tt_silicon_executor_is_refused_naming_agent_rule_9(setup: Setup) -> None:
    module = tt_silicon_module()
    registry = registry_with(setup.log, ("tt_silicon", module.TtSiliconExecutor))
    message = refusal(setup, registry, "tt_silicon")
    assert "tt_silicon" in message and "Agent Rule 9" in message, message


def test_the_train_refusal_reads_the_capability_never_the_name(setup: Setup) -> None:
    registry = registry_with(
        setup.log,
        ("renamed", never_built("renamed", frozenset({"runs_code", "sandboxed", SILICON}))),
        ("tt_silicon", never_built("tt_silicon", frozenset({"runs_code", "sandboxed"}))),
    )
    message = refusal(setup, registry, "renamed")
    assert "renamed" in message and "Agent Rule 9" in message, message
    plain = refusal(setup, registry, "tt_silicon")
    assert "reward" in plain and "Agent Rule 9" not in plain, "no capability, only the not-carried-out refusal"
