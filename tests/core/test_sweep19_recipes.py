"""Tests for sweep19: sweep17's ttsim arms with simulator gaps fed back as corrections (task P17.61).

Bible: Project Recipes (the loop opt-ins, correct_on_sim_gap), Execution
Backends (ttmlir-ttsim), Agent Rules 2 and 5.

What they fix for projects/lassi-demo/sweep19/gpt-oss-120b/: three recipes,
ttir-min-gap, ttir-full-gap, and d2m-min-gap, each sweep17's arm of the same
name but for loop.correct_on_sim_gap true and its project name, labelled as
running on ttsim with the output check and exploratory in its header.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.recipe import load_recipe

REPO = Path(__file__).resolve().parents[2]
RECIPES = REPO / "projects" / "lassi-demo" / "sweep19" / "gpt-oss-120b"
SWEEP17 = REPO / "projects" / "lassi-demo" / "sweep17" / "gpt-oss-120b"
ARMS = ["d2m-min", "ttir-full", "ttir-min"]


def test_the_directory_holds_exactly_the_three_arms() -> None:
    assert sorted(path.stem for path in RECIPES.glob("*.yaml")) == [f"{arm}-gap" for arm in ARMS]


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_is_sweep17s_with_simulator_gaps_fed_back(arm: str) -> None:
    ours = load_recipe(RECIPES / f"{arm}-gap.yaml").data
    theirs = load_recipe(SWEEP17 / f"{arm}.yaml").data
    changed = {key for key in set(ours) | set(theirs) if key != "project" and ours.get(key) != theirs.get(key)}
    assert changed == {"loop"}
    assert ours["loop"] == {**theirs["loop"], "correct_on_sim_gap": True}
    assert ours["project"] == f"{arm}-gap"


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_says_ttsim_and_exploratory(arm: str) -> None:
    text = (RECIPES / f"{arm}-gap.yaml").read_text(encoding="ascii")
    header = " ".join(line.lstrip("# ") for line in text.split("\nextends:")[0].split("\n"))
    assert "RUN ON TTSIM WITH THE OUTPUT CHECK" in header and "a simulator" in header
    assert "exploratory" in header.lower()
