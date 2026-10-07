"""The demo scripts under tools/demo/ (lassi-live and lassi-showcase) write recipes the loader accepts.

lassi-live writes one recipe per segment next to the demo recipes, extending projects/lassi-demo: the CPU
segment extends rngd-cpu.yaml (or rngd-compile.yaml), and the Tenstorrent segment extends the Tier A recipe
that judges its items by their declared metric, with one trial per item and direction and a short correction
cap. Nothing here serves a model or runs a program; the scripts are loaded as modules and their recipe
writers called with the repository as the demo slot.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

import lassi.core.runner  # noqa: F401  (registers the components the recipes bind)
from lassi.core.recipe import load_recipe

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str) -> ModuleType:
    """Load tools/demo/<name>, a script without a .py suffix, as a module."""
    path = ROOT / "tools" / "demo" / name
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_"), str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture()
def live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Return lassi-live with the repository as its slot and a scratch directory for the recipes it writes."""
    module = load_script("lassi-live")
    monkeypatch.setattr(module, "SLOT", ROOT)
    monkeypatch.setattr(module, "RECIPES", tmp_path / "recipes")
    return module


def resolved(path: Path) -> dict:
    """Return the resolved recipe of `path` as a mapping."""
    recipe = load_recipe(str(path))
    return recipe.resolved if hasattr(recipe, "resolved") else recipe.data


def test_cpu_segment_recipe_loads(live: ModuleType) -> None:
    data = resolved(live.write_recipe("demo-test-cpu", ["matrix-rotate"], compile_only=False))
    assert data["bench"]["items"] == ["matrix-rotate"]
    assert data["directions"] == [{"source": "cuda", "target": "omp"}]


@pytest.mark.parametrize("apps, metric", [(["eltwise_binary"], "max_abs"), (["matmul_single_core"], "pcc")])
def test_tt_segment_recipe_loads_with_one_trial_and_a_short_cap(live: ModuleType, apps: list[str],
                                                                  metric: str) -> None:
    data = resolved(live.write_tt_recipe("demo-test-tt", apps))
    assert data["bench"]["items"] == apps
    assert data["bench"]["suite"] == "tt-pairs-v0"
    assert data["trials"]["n"] == 1
    assert data["loop"]["max_corrections"] == live.TT_CORRECTIONS == 3
    assert data["prompts"] == "tt-host-v0"
    assert data["oracle"]["metric"] == metric
    assert data["executor"]["tt"]["kind"] == "ttsim"
    assert {(d["source"], d["target"]) for d in data["directions"]} == {("cpp", "tt"), ("tt", "cpp")}


def test_tt_segment_refuses_mixed_metrics_and_unknown_items(live: ModuleType) -> None:
    with pytest.raises(SystemExit, match="not both"):
        live.tt_parent(["eltwise_binary", "matmul_multi_core"])
    with pytest.raises(SystemExit, match="not a tt-pairs-v0 Tier A item"):
        live.tt_parent(["layout"])


def test_tt_directions_are_named_and_labeled_simulator(live: ModuleType) -> None:
    assert live.NAMES["tt"] == "TT (ttsim)"
    assert live.NAMES["cpp"] == "C++"
    assert "simulator, not silicon" in live.describe("tt", ["eltwise_binary"], compile_only=False)


def test_showcase_runs_from_the_demo_slot() -> None:
    showcase = load_script("lassi-showcase")
    assert showcase.WORKTREE.name == "demo-live"
