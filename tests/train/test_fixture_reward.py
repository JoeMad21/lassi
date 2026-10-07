"""Tests for the grpo smoke's fixture reward and its per-step record (task P17.9; lassi.train.fixture_reward).

Bible: Training Module (Safeguards), Agent Rules 1 and 6; PHASE-NOTES P7 and
P8, "From the P4 notes" (P4.11: the trainer skips a null reward and never
reads it as 0); plans/p17-portable.md, task P17.9 ("grpo, with in-process
generation and a fixture reward, executing nothing"; "per-step losses and
reward components").

The contract these tests fix:

- fixture_reward(target, completion) -> FixtureReward, with `value` and
  `components` {"target_fraction": f}: f is the fraction of the
  completion's characters that occur in `target`, 0.0 for an empty
  completion by rule. A null target gives value None and the component
  None: the reward cannot be scored, and nothing stands in for it (Agent
  Rule 1). It reads text only and runs nothing (Agent Rule 6).
- step_rewards(rewards) -> {"values", "count", "skipped_null", "mean",
  "components"}: values in order with each null kept, count the scored
  ones, skipped_null the nulls, mean the mean over the scored values only
  (null when none is scored), and per component its count and mean on the
  same rule. A null is never counted as 0. The record is JSON without NaN.

The module imports only the standard library and lassi (tests/train/
test_train_package.py). Every text is SYNTHETIC; no value in this module is
a measurement.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

import pytest
from trl_smoke import TARGET, train_module


def reward_module() -> Any:
    """Import lassi.train.fixture_reward."""
    return train_module("fixture_reward")


def score(target: str | None, completion: str) -> Any:
    """Return fixture_reward(target, completion)."""
    return reward_module().fixture_reward(target, completion)


@pytest.mark.parametrize(
    ("target", "completion", "expected"),
    [
        ("abc", "abc", 1.0),
        ("abc", "aab", 1.0),
        ("abc", "axyz", 0.25),
        ("abc", "xyz", 0.0),
        ("", "abc", 0.0),
        (TARGET, "SYNTHETIC", 0.0),
        (TARGET, "ab\N{REPLACEMENT CHARACTER}\N{REPLACEMENT CHARACTER}", 0.5),
    ],
    ids=["all", "repeats", "a-quarter", "none", "empty-target", "upper-case", "undecodable-bytes"],
)
def test_the_value_is_the_fraction_of_characters_in_the_target(
    target: str, completion: str, expected: float
) -> None:
    found = score(target, completion)
    assert found.value == pytest.approx(expected, abs=1e-12)
    assert dict(found.components) == {"target_fraction": found.value}


def test_an_empty_completion_scores_zero_by_rule() -> None:
    found = score(TARGET, "")
    assert found.value == 0.0 and dict(found.components) == {"target_fraction": 0.0}


@pytest.mark.parametrize("completion", ["abc", ""], ids=["text", "empty"])
def test_a_null_target_gives_a_null_reward_never_zero(completion: str) -> None:
    found = score(None, completion)
    assert found.value is None
    assert dict(found.components) == {"target_fraction": None}


def test_the_reward_runs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the fixture reward started a process (Agent Rule 6)")

    monkeypatch.setattr(subprocess.Popen, "__init__", refuse)
    monkeypatch.setattr(os, "system", refuse)
    assert score(TARGET, "abc").value == 1.0


def test_step_rewards_take_the_mean_over_scored_values_only() -> None:
    record = reward_module().step_rewards([score("abc", "abc"), score(None, "abc")])
    assert record == {
        "values": [1.0, None],
        "count": 1,
        "skipped_null": 1,
        "mean": 1.0,
        "components": {"target_fraction": {"count": 1, "mean": 1.0}},
    }


def test_a_null_is_never_counted_as_zero_in_the_mean() -> None:
    rewards = [score("abc", "abc"), score("abc", "axyz"), score(None, "xyz"), score(None, "")]
    record = reward_module().step_rewards(rewards)
    assert record["values"] == [1.0, 0.25, None, None]
    assert (record["count"], record["skipped_null"]) == (2, 2)
    assert record["mean"] == pytest.approx(0.625, abs=1e-12), "a null read as 0 would give 0.3125"
    assert record["components"]["target_fraction"]["count"] == 2
    assert record["components"]["target_fraction"]["mean"] == pytest.approx(0.625, abs=1e-12)


def test_an_all_null_step_has_a_null_mean() -> None:
    record = reward_module().step_rewards([score(None, "abc"), score(None, "def")])
    assert record == {
        "values": [None, None],
        "count": 0,
        "skipped_null": 2,
        "mean": None,
        "components": {"target_fraction": {"count": 0, "mean": None}},
    }


def test_the_step_record_is_json_without_nan() -> None:
    record = reward_module().step_rewards([score(TARGET, "abc"), score(None, "abc"), score(TARGET, "")])
    text = json.dumps(record, allow_nan=False)
    assert json.loads(text) == record
