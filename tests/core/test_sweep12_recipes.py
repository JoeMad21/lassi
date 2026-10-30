"""Tests for sweep12, the temperature arms of gpt-oss-120b at high reasoning effort (tasks P17.51 and P17.52).

Bible: Project Recipes (llm.sampling; the reasoning effort; the loop opt-ins),
Agent Rule 1. plans/p17-portable.md, P17.51 and P17.52 (the owner's directions
of 2026-10-29).

What they fix: projects/lassi-demo/sweep12-t02, -t05, -t07, and -t10 each hold
the same eight recipes, kh-allsk (sweep9), bale-cpp and bale-omp (sweep10), and
mlir-rag (sweep10), in both halves, named <combo>-<arm>-<half>.yaml; each loads
and resolves to the recipe it extends but for llm.sampling (its arm's
temperature, top_p 1.0, reasoning effort high, max_tokens 65536) and the loop's
restarts (restart_after 3, restart_on_repeat true); the four arms differ in the
temperature alone.

No value here is a measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lassi.core import runner  # noqa: F401  (importing the runner registers every component)
from lassi.core.recipe import load_recipe

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "projects" / "lassi-demo"
ARMS = {"t02": 0.2, "t05": 0.5, "t07": 0.7, "t10": 1.0}
BASES = {
    ("kh-allsk", "maxabs"): "sweep9/gpt-oss-120b/kh-allsk-maxabs.yaml",
    ("kh-allsk", "pcc"): "sweep9/gpt-oss-120b/kh-allsk-pcc.yaml",
    ("bale-cpp", "a"): "sweep10/gpt-oss-120b/bale-cpp-a.yaml",
    ("bale-cpp", "b"): "sweep10/gpt-oss-120b/bale-cpp-b.yaml",
    ("bale-omp", "a"): "sweep10/gpt-oss-120b/bale-omp-a.yaml",
    ("bale-omp", "b"): "sweep10/gpt-oss-120b/bale-omp-b.yaml",
    ("mlir-rag", "maxabs"): "sweep10/gpt-oss-120b/mlir-rag-maxabs.yaml",
    ("mlir-rag", "pcc"): "sweep10/gpt-oss-120b/mlir-rag-pcc.yaml",
}
CASES = [(arm, combo, half) for arm in ARMS for combo, half in BASES]


def path_of(arm: str, combo: str, half: str) -> Path:
    """Return the sweep12 recipe of one arm, combination, and half."""
    return DEMO / f"sweep12-{arm}" / "gpt-oss-120b" / f"{combo}-{arm}-{half}.yaml"


def without(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Return `data` less the named top-level keys."""
    return {key: value for key, value in data.items() if key not in keys}


def test_each_arm_holds_the_eight_recipes_and_nothing_else() -> None:
    for arm in ARMS:
        names = sorted(path.name for path in (DEMO / f"sweep12-{arm}").rglob("*.yaml"))
        assert names == sorted(f"{combo}-{arm}-{half}.yaml" for combo, half in BASES)


@pytest.mark.parametrize(("arm", "combo", "half"), CASES)
def test_a_recipe_is_its_base_but_for_the_sampling_and_the_restarts(arm: str, combo: str, half: str) -> None:
    data = load_recipe(path_of(arm, combo, half)).data
    base = load_recipe(DEMO / BASES[(combo, half)]).data
    sampling = data["llm"]["sampling"]
    assert (sampling["temperature"], sampling["top_p"]) == (ARMS[arm], 1.0)
    assert (sampling["reasoning_effort"], sampling["max_tokens"]) == ("high", 65536)
    assert {key: value for key, value in sampling.items() if key not in ("temperature", "top_p", "reasoning_effort",
            "max_tokens")} == {key: value for key, value in base["llm"]["sampling"].items() if key not in (
            "temperature", "top_p", "reasoning_effort", "max_tokens")}
    assert data["loop"] == {**base["loop"], "restart_after": 3, "restart_on_repeat": True}
    assert without(data, "llm", "loop", "project") == without(base, "llm", "loop", "project")


@pytest.mark.parametrize(("combo", "half"), sorted(BASES))
def test_the_arms_differ_in_the_temperature_alone(combo: str, half: str) -> None:
    loaded = {arm: load_recipe(path_of(arm, combo, half)).data for arm in ARMS}
    reference = loaded["t02"]
    for arm, data in loaded.items():
        assert data["llm"]["sampling"]["temperature"] == ARMS[arm]
        same = {**data, "llm": {**data["llm"], "sampling": {**data["llm"]["sampling"], "temperature": 0.2}}}
        assert without(same, "project") == without(reference, "project")
