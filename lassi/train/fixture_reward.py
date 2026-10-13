"""The grpo smoke's fixture reward and its per-step record (task P17.9; bible Training Module, Safeguards).

fixture_reward(target, completion) scores one completion against the
target text of its SYNTHETIC record (tests/fixtures/train/grpo-smoke.jsonl):
the value is the fraction of the completion's characters that occur in the
target, 0.0 for an empty completion by rule, and its one component,
target_fraction, is that same fraction. A null target is a reward that
cannot be scored: the value and the component are None, and nothing stands
in for them (Agent Rule 1). It reads text only and runs nothing (Agent Rule
6).

step_rewards(rewards) is one step's reward record: the values in order with
each null kept, the count of scored values, the nulls skipped, and the mean
over the scored values only (None when none is scored), with each
component's count and mean on the same rule. A null is never counted as 0
(PHASE-NOTES P7 and P8, from the P4 notes).

This module imports only the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# The fixture reward's one component, and the rule a checkpoint records it by.
COMPONENT = "target_fraction"


@dataclass(frozen=True)
class FixtureReward:
    """One completion's fixture reward: its value and its components by name; None where it cannot be scored."""

    value: float | None
    components: Mapping[str, float | None] = field(default_factory=dict)


def fixture_reward(target: str | None, completion: str) -> FixtureReward:
    """Return the fraction of the completion's characters that occur in `target`; None for a null target.

    An empty completion scores 0.0 by rule. The component target_fraction
    holds the same value.
    """
    if target is None:
        return FixtureReward(None, {COMPONENT: None})
    value = sum(char in target for char in completion) / len(completion) if completion else 0.0
    return FixtureReward(value, {COMPONENT: value})


def _scored_mean(values: Sequence[float | None]) -> tuple[int, float | None]:
    """Return the count of the values that are not None and their mean, None when there are none."""
    scored = [value for value in values if value is not None]
    return len(scored), (sum(scored) / len(scored) if scored else None)


def step_rewards(rewards: Sequence[FixtureReward]) -> dict[str, Any]:
    """Return one step's reward record: values, count, skipped_null, mean, and components; a null is never 0.

    `components` maps each component name to {"count", "mean"}, taken over
    the scored values of that component only.
    """
    values = [reward.value for reward in rewards]
    count, mean = _scored_mean(values)
    components: dict[str, dict[str, Any]] = {}
    for name in sorted({name for reward in rewards for name in reward.components}):
        found, average = _scored_mean([reward.components.get(name) for reward in rewards])
        components[name] = {"count": found, "mean": average}
    return {"values": values, "count": count, "skipped_null": len(values) - count, "mean": mean,
            "components": components}
