"""Tests for binary_io's from_item choice: each output judged by the item's declared tolerance (OQ-115).

Bible: Oracles (binary_io row), Bench Suites (an item's tolerance), Agent
Rule 3 (one generic recipe value behind the Oracle interface).

The contract these tests fix:

- `metric: from_item` and `threshold: from_item` go together; either alone
  is refused naming both keys. tolerance_setting is "oracle.threshold
  from_item" then, else None; agreement_setting stays None.
- An unbound from_item oracle refuses compare and threshold_for;
  with_tolerance binds the item's metric and threshold and keeps the shape
  rule. describe() names from_item.
- lassi.core.stages agreement_bound binds a from_item oracle to the item's
  tolerance, and refuses one with no tolerance naming the trial.

Every value here is SYNTHETIC.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.stages import agreement_bound
from lassi.core.tolerance import Tolerance
from lassi.oracles.binary_io import FROM_ITEM, BinaryIOOracle


def item_oracle(**config: Any) -> BinaryIOOracle:
    """Return a from_item oracle with `config` added."""
    return BinaryIOOracle(metric=FROM_ITEM, threshold=FROM_ITEM, **config)


@pytest.mark.parametrize(("metric", "threshold"), [("from_item", 0.01), ("max_abs", "from_item"),
                                                   ("from_item", "from_baseline")])
def test_from_item_goes_with_from_item(metric: str, threshold: Any) -> None:
    with pytest.raises(ValueError, match="go together"):
        BinaryIOOracle(metric=metric, threshold=threshold)


def test_the_settings_say_which_binding_the_oracle_needs() -> None:
    oracle = item_oracle()
    assert oracle.tolerance_setting == "oracle.threshold from_item" and oracle.agreement_setting is None
    assert BinaryIOOracle(metric="pcc", threshold=0.9).tolerance_setting is None
    assert "from_item" in oracle.describe()


def test_an_unbound_from_item_oracle_refuses_to_judge(tmp_path: Path) -> None:
    oracle = item_oracle()
    with pytest.raises(ValueError, match="with_tolerance"):
        oracle.compare({}, {})
    with pytest.raises(ValueError, match="with_tolerance"):
        oracle.threshold_for("c")


def test_with_tolerance_binds_the_items_metric_and_threshold_and_keeps_the_shape_rule() -> None:
    bound = item_oracle(shape="elements").with_tolerance(Tolerance(metric="pcc", threshold=0.97))
    assert (bound.metric, bound.threshold_for("c"), bound.shape) == ("pcc", 0.97, "elements")
    assert bound.tolerance_setting is None


def test_agreement_bound_binds_the_items_tolerance_and_refuses_none() -> None:
    trial: Any = SimpleNamespace(trial_id="trial-x", reference_agreement=None, baseline_diagnostics=[])
    bound = agreement_bound(item_oracle(), trial, Tolerance(metric="max_abs", threshold=0.05))
    assert (bound.metric, bound.threshold_for("c")) == ("max_abs", 0.05)
    with pytest.raises(ValueError, match="trial-x.*from_item"):
        agreement_bound(item_oracle(), trial, None)
