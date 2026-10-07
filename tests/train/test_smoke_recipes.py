"""The committed train smoke recipes of gate part (a) are the CI smoke runs' recipes (task P17.11).

Bible: Build Roadmap (P17 row, Gate part (a): `lassi train` runs sft, dpo,
and grpo briefly on the CPU with a tiny model on synthetic data), Training
Module, Project Recipes (train.yaml), Agent Rule 5; plans/p17-portable.md,
task P17.11 and PHASE-NOTES P17 (the smoke recipes name data.synthetic
files under tests/fixtures/train/ and a trainer section with device {kind:
cpu}).

Each tests/fixtures/recipes/p17-train-<method>.yaml loads with
lassi.core.recipe load_train_recipe and lassi's own registry, and its data
equals trl_smoke.smoke_recipe(<method>, <weights>), the recipe the slow
tests in test_trl_runs.py train, so the gate's runs on alpha01 and the CI
runs describe the same training: sft with full weights, dpo and grpo with
lora. The fixtures are SYNTHETIC; no framework is imported here and no
value is a measurement.
"""

from __future__ import annotations

import pytest
from train_fakes import REPO
from trl_smoke import SYNTHETIC_DIR, smoke_recipe

import lassi.train  # noqa: F401  (registers the trl trainer without a framework)
from lassi.core.recipe import load_train_recipe

RECIPES = REPO / "tests" / "fixtures" / "recipes"
# The weight mode each committed smoke recipe trains with.
WEIGHTS = {"sft": "full", "dpo": "lora", "grpo": "lora"}


@pytest.mark.parametrize("method", sorted(WEIGHTS))
def test_each_smoke_recipe_is_the_ci_smoke_recipe(method: str) -> None:
    recipe = load_train_recipe(RECIPES / f"p17-train-{method}.yaml")
    assert recipe.data == smoke_recipe(method, WEIGHTS[method])


@pytest.mark.parametrize("method", sorted(WEIGHTS))
def test_each_smoke_recipe_trains_on_the_cpu_from_a_synthetic_fixture(method: str) -> None:
    data = load_train_recipe(RECIPES / f"p17-train-{method}.yaml").data
    assert data["trainer"]["device"] == {"kind": "cpu"}
    assert "bench" not in data, "training data is synthetic, never a bench item (Agent Rule 5)"
    assert (SYNTHETIC_DIR / data["data"]["synthetic"]).is_file()
    assert "runs_root" not in data, "lassi train takes $LASSI_RUNS_ROOT"
