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


@pytest.mark.parametrize("segment", ["cpu", "tt"])
def test_mock_recipes_bind_the_mock_model_and_one_trial(live: ModuleType, segment: str) -> None:
    path = (live.write_tt_recipe("demo-test-mock-tt", ["eltwise_binary"], mock=True) if segment == "tt" else
            live.write_recipe("demo-test-mock-cpu", ["matrix-rotate"], compile_only=False, mock=True))
    data = resolved(path)
    assert data["model"]["backend"] == "mock"
    assert data["model"]["id"] == "mock-reference"
    assert data["trials"]["n"] == 1


# `furiosa-smi status` on alpha01 at 2026-10-22T21:00:11-07:00 (rx 20261022-210011-exec-0726), npu1 to npu7
# trimmed to two rows and npu2's memory edited to show a card in use.
STATUS_2026_10_22 = """\
+------+--------+---------------------------+-------------+
| Arch | Device | Memory                    | Utilization |
+------+--------+---------------------------+-------------+
| RNGD | npu0   |  0.00/47.50 GiB   (0.00%) |      -      |
+------+--------+---------------------------+-------------+
| RNGD | npu1   |  0.00/47.50 GiB   (0.00%) |      -      |
+------+--------+---------------------------+-------------+
| RNGD | npu2   | 12.30/47.50 GiB  (25.89%) |      -      |
+------+--------+---------------------------+-------------+
"""


def test_card_status_reads_the_table_without_a_state_column(live: ModuleType) -> None:
    cards = live.parse_status(STATUS_2026_10_22)
    assert cards == {"npu0": ("-", 0.0, 47.5), "npu1": ("-", 0.0, 47.5), "npu2": ("-", 12.3, 47.5)}


def test_card_status_still_reads_the_table_with_a_state_column(live: ModuleType) -> None:
    old = "| rngd | npu3 | alive | 0.00/47.50 GiB | 0.00% |\n| rngd | npu4 | alive | 3.10/47.50 GiB | 9.00% |\n"
    assert live.parse_status(old) == {"npu3": ("alive", 0.0, 47.5), "npu4": ("alive", 3.1, 47.5)}


def test_pick_card_takes_npu0_when_its_memory_reads_zero(live: ModuleType) -> None:
    cards = live.parse_status(STATUS_2026_10_22)
    assert live.pick_card(cards, None) == "npu0"
    assert live.pick_card(cards, "npu0") == "npu0"


def test_pick_card_refuses_a_card_in_use(live: ModuleType) -> None:
    cards = live.parse_status(STATUS_2026_10_22)
    with pytest.raises(SystemExit, match="no free card among npu2"):
        live.pick_card(cards, "npu2")
