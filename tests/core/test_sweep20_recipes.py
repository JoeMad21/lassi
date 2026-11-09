"""Tests for sweep20: sweep18's TTIR ladder and D2M arm on ttsim for the 30B-class RNGD models (task P17.61).

Bible: Project Recipes (template sets, the context, retrieval, and
source_mlir sections, the loop opt-ins), Execution Backends (ttmlir-ttsim),
Oracles (binary_io, from_item), Agent Rules 2 and 5.

What they fix for projects/lassi-demo/sweep20/<slug>/, one directory per row
of plans/runs/suite/models-30b.tsv (the owner's window of 2026-11-09):

- Five recipes per model: ttir-l0 to ttir-l3 and d2m-min, each running on
  ttsim with the output check and labelled so in its header.
- Each ladder rung adds exactly one feature to the rung below.
- Each recipe is sweep18's Llama-3.3-70B recipe of the same name but for the
  model and its sampling, and l1, l3, and d2m-min are sweep19's gpt-oss-120b
  arms but for the same two sections, so every arm compares with both.
- Each model's id and sampling: its own generation_config.json at v2026.4
  (EXAONE-4.0's sets none, so projects/base.yaml's LASSI values), with
  16384 tokens for the models that think or may think and 8192 otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.recipe import load_recipe

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "projects" / "lassi-demo" / "sweep20"
SWEEP18 = REPO / "projects" / "lassi-demo" / "sweep18" / "llama33-70b"
SWEEP19 = REPO / "projects" / "lassi-demo" / "sweep19" / "gpt-oss-120b"
TABLE = REPO / "plans" / "runs" / "suite" / "models-30b.tsv"
NAMES = ["d2m-min", "ttir-l0", "ttir-l1", "ttir-l2", "ttir-l3"]
ADDED = {1: {"context", "stages"}, 2: {"retrieval"}, 3: {"source_mlir"}}
# slug -> (model id, temperature, top_p, max_tokens), in serving order
MODELS = {
    "qwen3-coder-30b": ("furiosa-ai/Qwen3-Coder-30B-A3B-Instruct-FP8", 0.7, 0.8, 8192),
    "qwen3-30b-instruct": ("furiosa-ai/Qwen3-30B-A3B-Instruct-2507-FP8", 0.7, 0.8, 8192),
    "qwen3-32b": ("furiosa-ai/Qwen3-32B-FP8", 0.6, 0.95, 16384),
    "gemma4-31b": ("furiosa-ai/gemma-4-31B-it-FP8-block", 1.0, 0.95, 16384),
    "exaone4-32b": ("furiosa-ai/EXAONE-4.0-32B-FP8", 0.2, 0.9, 8192),
    "exaone45-33b": ("furiosa-ai/EXAONE-4.5-33B-FP8", 1.0, 0.95, 16384),
}
PAIRS = [("ttir-l1", "ttir-min-gap"), ("ttir-l3", "ttir-full-gap"), ("d2m-min", "d2m-min-gap")]


def load(path: Path) -> dict[str, Any]:
    """Return the resolved data of one recipe."""
    return load_recipe(path).data


def changed(upper: dict[str, Any], lower: dict[str, Any], ignore: tuple[str, ...] = ("project",)) -> set[str]:
    """Return the top-level keys whose values differ between two resolved recipes, the ignored keys aside."""
    return {key for key in set(upper) | set(lower) if key not in ignore and upper.get(key) != lower.get(key)}


def table_rows() -> list[list[str]]:
    """Return the model table's rows, comment lines left out."""
    lines = TABLE.read_text(encoding="ascii").splitlines()
    return [line.split("\t") for line in lines if line and not line.startswith("#")]


def test_the_table_lists_the_six_models_in_serving_order_with_their_sizes() -> None:
    rows = table_rows()
    assert [row[0] for row in rows] == list(MODELS)
    for row in rows:
        assert len(row) == 7 and row[1] == MODELS[row[0]][0]
        assert row[2:6] == ["v2026.4", "4", "-", "furiosa-venv-2026.4"] and row[6].isdigit()


def test_each_model_directory_holds_exactly_the_five_recipes() -> None:
    assert sorted(path.name for path in SWEEP.iterdir()) == sorted(MODELS)
    for slug in MODELS:
        assert sorted(path.stem for path in (SWEEP / slug).glob("*.yaml")) == NAMES, slug


@pytest.mark.parametrize("slug", list(MODELS))
@pytest.mark.parametrize("name", NAMES)
def test_each_recipe_runs_on_ttsim_with_the_item_tolerance_and_says_so(slug: str, name: str) -> None:
    data = load(SWEEP / slug / f"{name}.yaml")
    language = name.split("-")[0]
    assert data["executor"][language]["kind"] == "ttmlir-ttsim"
    assert data["oracle"] == {"kind": "binary_io", "metric": "from_item", "threshold": "from_item", "shape": "elements"}
    assert data["loop"] == {"max_corrections": 8, "correct_on_mismatch": True, "correct_on_sim_gap": True}
    assert data["bench"]["split"] == "unassigned"
    text = (SWEEP / slug / f"{name}.yaml").read_text(encoding="ascii")
    header = " ".join(line.lstrip("# ") for line in text.split("\nextends:")[0].split("\n"))
    assert "RUN ON TTSIM WITH THE OUTPUT CHECK" in header and "a simulator" in header
    assert "exploratory" in header.lower()
    assert max(len(line) for line in text.splitlines()) <= 120


@pytest.mark.parametrize("slug", list(MODELS))
@pytest.mark.parametrize("name", NAMES)
def test_each_recipe_is_sweep18s_but_for_the_model_and_its_sampling(slug: str, name: str) -> None:
    data = load(SWEEP / slug / f"{name}.yaml")
    model, temperature, top_p, tokens = MODELS[slug]
    assert data["model"] == {**load(SWEEP18 / f"{name}.yaml")["model"], "id": model}
    assert data["llm"]["sampling"] == {"temperature": temperature, "top_p": top_p, "max_tokens": tokens}
    assert changed(data, load(SWEEP18 / f"{name}.yaml")) == {"model", "llm"}


@pytest.mark.parametrize("slug", list(MODELS))
@pytest.mark.parametrize("rung", sorted(ADDED))
def test_each_rung_adds_exactly_one_feature(slug: str, rung: int) -> None:
    upper, lower = load(SWEEP / slug / f"ttir-l{rung}.yaml"), load(SWEEP / slug / f"ttir-l{rung - 1}.yaml")
    assert changed(upper, lower) == ADDED[rung]


@pytest.mark.parametrize("slug", list(MODELS))
@pytest.mark.parametrize(("ours", "theirs"), PAIRS)
def test_the_arms_match_sweep19_but_for_the_model(slug: str, ours: str, theirs: str) -> None:
    assert changed(load(SWEEP / slug / f"{ours}.yaml"), load(SWEEP19 / f"{theirs}.yaml")) == {"model", "llm"}
