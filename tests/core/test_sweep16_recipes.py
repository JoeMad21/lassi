"""Tests for sweep16's TTIR recipe on the fewest features that reach the correction loop (task P17.61).

Bible: Project Recipes (template sets, the context section, the loop
opt-ins), Agent Rules 2 and 5.

What they fix for projects/lassi-demo/sweep16/gpt-oss-120b/: one recipe,
ttir-min, sweep15's rung l1 unchanged (the bare template set with the API
reference alone: no retrieval, cgeist source, guidance set, or restarts;
temperature 1.0) but for its project name, labelled compile only and
exploratory in its header.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.recipe import load_recipe

REPO = Path(__file__).resolve().parents[2]
RECIPES = REPO / "projects" / "lassi-demo" / "sweep16" / "gpt-oss-120b"
RUNG = REPO / "projects" / "lassi-demo" / "sweep15" / "gpt-oss-120b" / "ttir-l1.yaml"
NAMES = ["ttir-min"]


def test_the_directory_holds_exactly_the_one_recipe() -> None:
    assert sorted(path.stem for path in RECIPES.glob("*.yaml")) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_the_recipe_is_rung_l1_but_for_its_project(name: str) -> None:
    data = load_recipe(RECIPES / f"{name}.yaml").data
    rung = load_recipe(RUNG).data
    assert {key: value for key, value in data.items() if key != "project"} == {
        key: value for key, value in rung.items() if key != "project"}
    assert data["project"] == name and data["llm"]["sampling"]["temperature"] == 1.0
    assert data["prompts"] == "tt-ttir-bare-v0" and data["context"] == ["ttir-ref-5f396fd6"]
    for key in ("retrieval", "source_mlir"):
        assert key not in data, key
    assert data["loop"] == {"max_corrections": 8}
    assert data["executor"] == {"kind": "none"} and "oracle" not in data


@pytest.mark.parametrize("name", NAMES)
def test_the_recipe_says_compile_only(name: str) -> None:
    text = (RECIPES / f"{name}.yaml").read_text(encoding="ascii")
    header = " ".join(line.lstrip("# ") for line in text.split("\nextends:")[0].split("\n"))
    assert "COMPILE ONLY" in header and "no output is checked" in header.lower()
    assert "exploratory" in header.lower()
