"""Tests that a reward on a silicon executor meets no silicon refusal (tasks P17.16 and P17.53).

Bible: Training Module, Safeguards; Component Interfaces, the Executor
bullet. The owner, 2026-10-29, directing removed the rule that RL rewards never
execute on silicon (Agent Rule 9's current text stands until the owner's
revision, OQ-099): "OF COURSE silicon runs should be included in RL
rewards. Remove that rule that says RL rewards must never come from
silicon. It's fine, they have to pass the simulator and compiler first."
plans/p17-portable.md, task P17.53.

The contract these tests fix (lassi.train.run):

- A train recipe whose reward.executor names an Executor that declares
  `silicon` (lassi.core.capabilities SILICON; the real tt_silicon executor,
  or any other that declares it) is treated as a reward on any executor:
  today the layer refuses every reward section as a key it does not carry
  out (NOT_CARRIED_OUT, for P7 and P8), before any directory exists and
  before the Trainer is built, and that refusal names neither silicon nor
  Agent Rule 9. Nothing reads the capability for a reward any more.

Every component is the train layer's fake Trainer (train_fakes) in a test
Registry, beside the real tt_silicon executor class, which is never built.
No card, node, or process is touched. No value here is a measurement.
"""

from __future__ import annotations

import importlib
from pathlib import Path
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


def test_a_reward_on_the_tt_silicon_executor_meets_only_the_not_carried_out_refusal(setup: Setup) -> None:
    module = importlib.import_module("lassi.executors.tt_silicon")
    registry = registry_with(setup.log, ("tt_silicon", module.TtSiliconExecutor))
    message = refusal(setup, registry, "tt_silicon")
    assert "reward" in message and "Agent Rule 9" not in message and "runs programs on silicon" not in message, message


def test_a_reward_on_any_executor_meets_the_same_refusal(setup: Setup) -> None:
    registry = registry_with(
        setup.log,
        ("renamed", never_built("renamed", frozenset({"runs_code", "sandboxed", SILICON}))),
        ("plain", never_built("plain", frozenset({"runs_code", "sandboxed"}))),
    )
    on_silicon, plain = refusal(setup, registry, "renamed"), refusal(setup, registry, "plain")
    assert on_silicon == plain, (on_silicon, plain)
    assert not hasattr(importlib.import_module("lassi.train.run"), "_check_reward_executor"), "the refusal is gone"
