"""Tests for sweep18: the TTIR feature ladder and the D2M arm on ttsim for Llama-3.3-70B (task P17.61).

Bible: Project Recipes (template sets, the context, retrieval, and
source_mlir sections, the loop opt-ins), Execution Backends (ttmlir-ttsim),
Oracles (binary_io, from_item), Agent Rules 2 and 5.

What they fix for projects/lassi-demo/sweep18/llama33-70b/:

- Five recipes: ttir-l0 to ttir-l3 and d2m-min, each running on ttsim with
  the output check and labelled so in its header.
- Each ladder rung adds exactly one feature to the rung below: l1 the API
  reference (with verbatim_context), l2 retrieval, l3 the cgeist affine
  source; l0 is the bare loop.
- l1 and d2m-min are sweep17's gpt-oss-120b ttir-min and d2m-min but for the
  model and its sampling, so the two models compare arm for arm.
- The model is Llama-3.3-70B-Instruct-FP8-dynamic with the sampling of its own
  generation_config.json (temperature 0.6, top_p 0.9), 8192 tokens, and no
  reasoning setting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.recipe import load_recipe

REPO = Path(__file__).resolve().parents[2]
RECIPES = REPO / "projects" / "lassi-demo" / "sweep18" / "llama33-70b"
SWEEP17 = REPO / "projects" / "lassi-demo" / "sweep17" / "gpt-oss-120b"
NAMES = ["d2m-min", "ttir-l0", "ttir-l1", "ttir-l2", "ttir-l3"]
ADDED = {1: {"context", "stages"}, 2: {"retrieval"}, 3: {"source_mlir"}}
MODEL = "furiosa-ai/Llama-3.3-70B-Instruct-FP8-dynamic"


def load(path: Path) -> dict[str, Any]:
    """Return the resolved data of one recipe."""
    return load_recipe(path).data


def changed(upper: dict[str, Any], lower: dict[str, Any], ignore: tuple[str, ...] = ("project",)) -> set[str]:
    """Return the top-level keys whose values differ between two resolved recipes, the ignored keys aside."""
    return {key for key in set(upper) | set(lower) if key not in ignore and upper.get(key) != lower.get(key)}


def test_the_directory_holds_exactly_the_five_recipes() -> None:
    assert sorted(path.stem for path in RECIPES.glob("*.yaml")) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_each_recipe_runs_on_ttsim_with_the_item_tolerance_and_says_so(name: str) -> None:
    data = load(RECIPES / f"{name}.yaml")
    language = name.split("-")[0]
    assert data["executor"][language]["kind"] == "ttmlir-ttsim"
    assert data["oracle"] == {"kind": "binary_io", "metric": "from_item", "threshold": "from_item", "shape": "elements"}
    assert data["stages"][-3:] == ["compile_loop", "run_loop", "oracle"] and data["stages"][0] == "baseline"
    assert data["loop"] == {"max_corrections": 8, "correct_on_mismatch": True} and data["trials"] == {"n": 2}
    text = (RECIPES / f"{name}.yaml").read_text(encoding="ascii")
    header = " ".join(line.lstrip("# ") for line in text.split("\nextends:")[0].split("\n"))
    assert "RUN ON TTSIM WITH THE OUTPUT CHECK" in header and "a simulator" in header
    assert "exploratory" in header.lower()


@pytest.mark.parametrize("name", NAMES)
def test_each_recipe_uses_llamas_own_sampling_and_no_reasoning_setting(name: str) -> None:
    data = load(RECIPES / f"{name}.yaml")
    assert data["model"]["id"] == MODEL and data["model"]["timeout_s"] == 1800
    assert data["llm"]["sampling"] == {"temperature": 0.6, "top_p": 0.9, "max_tokens": 8192}


@pytest.mark.parametrize("rung", sorted(ADDED))
def test_each_rung_adds_exactly_one_feature(rung: int) -> None:
    assert changed(load(RECIPES / f"ttir-l{rung}.yaml"), load(RECIPES / f"ttir-l{rung - 1}.yaml")) == ADDED[rung]


def test_the_bottom_rung_is_the_bare_loop() -> None:
    data = load(RECIPES / "ttir-l0.yaml")
    assert data["prompts"] == "tt-ttir-bare-v1" and "verbatim_context" not in data["stages"]
    for key in ("context", "retrieval", "source_mlir", "dialect_lookup", "examples"):
        assert key not in data, key


PAIRS = [("ttir-l1", "ttir-min"), ("ttir-l3", "ttir-full"), ("d2m-min", "d2m-min")]


@pytest.mark.parametrize(("ours", "theirs"), PAIRS)
def test_the_arms_match_sweep17_but_for_the_model(ours: str, theirs: str) -> None:
    assert changed(load(RECIPES / f"{ours}.yaml"), load(SWEEP17 / f"{theirs}.yaml")) == {"model", "llm"}
