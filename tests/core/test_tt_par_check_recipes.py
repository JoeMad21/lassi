"""Tests for the tt-par-v0 check recipes, which build and run each reference on ttsim (task P17.76).

Bible: Benchmark Suites (tt-par-v0), Oracles (binary_io, the item's own tolerance), Execution Backends (ttsim),
Agent Rules 2 and 5. The recipes ask no model: the mock answers with each item's TT reference, so a run checks that
the C++ program and the TT reference build, run, and agree, on the host CPU and on ttsim, one trial per item, for the
development items and for the held-out items (evaluation is a purpose both splits are given for).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.recipe import load_recipe

REPO = Path(__file__).resolve().parents[2]
CHECKS = REPO / "projects" / "lassi-demo" / "tt-par-check"
SPLITS = ("dev", "eval")


@pytest.mark.parametrize("split", SPLITS)
def test_each_check_recipe_loads_and_reads_its_split(split: str) -> None:
    """Each recipe loads and reads its split of tt-par-v0, cpp -> tt, with the reference mock."""
    path = CHECKS / f"{split}-cpp2tt.yaml"
    assert path.is_file(), f"{path.relative_to(REPO).as_posix()} does not exist (task P17.76)"
    data = load_recipe(path).data
    assert data["bench"]["suite"] == "tt-par-v0" and data["bench"]["split"] == split
    assert data["model"]["backend"] == "mock" and data["model"]["id"] == "mock-reference"
    assert data["directions"] == [{"source": "cpp", "target": "tt"}]


@pytest.mark.parametrize("split", SPLITS)
def test_each_check_recipe_runs_both_references_and_judges_by_the_items_tolerance(split: str) -> None:
    """Both references build and run (C++ on the host CPU, TT on ttsim); the oracle takes each item's tolerance."""
    data = load_recipe(CHECKS / f"{split}-cpp2tt.yaml").data
    assert data["fixes"]["baseline_both"] is True
    assert data["stages"] == ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
    assert data["executor"]["cpp"]["kind"] == "native"
    assert data["executor"]["tt"]["kind"] == "ttsim"
    assert data["oracle"]["metric"] == "from_item" and data["oracle"]["threshold"] == "from_item"
    assert data["trials"]["n"] == 1 and data["loop"]["max_corrections"] == 0


def test_the_check_recipes_are_the_only_files_there() -> None:
    """The directory holds the two check recipes and nothing else."""
    assert sorted(path.name for path in CHECKS.iterdir()) == [f"{split}-cpp2tt.yaml" for split in SPLITS]
