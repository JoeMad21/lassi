"""Tests for sweep9, the first C++ -> TT recipes in which the model writes the kernels too (task P17.39).

Bible: Project Recipes (the examples, skeleton, context, lookup, retrieval,
and loop sections), Benchmark Suites (tt-pairs-v0-kh; a development set,
OQ-083 (b)), Agent Rules 2, 5, and 6. The owner's direction of 2026-10-26:
"I want to start having the model write kernels as well as host side code."

What they fix for projects/lassi-demo/sweep9/gpt-oss-120b/:

- Six recipes: kh-ex, kh-exsk, and kh-allsk, each in the maxabs and pcc
  halves, and nothing else; each loads, and each half covers its items once
  by their declared metric.
- Each binds cpp -> ttkh on tt-pairs-v0-kh (split unassigned) with the
  tt-kh-v0 set, ttmetal-host for ttkh and for the tt reference, ttsim for
  both (a simulator), sweep8's gpt-oss-120b model settings, 2 samples, a
  correction cap of 8, the mismatch report and simulator-gap feedback on,
  the oracle from the references' agreement, and the stages baseline,
  generate, compile_loop, run_loop, and oracle (no adversary, whose
  reference build reads no reference language; no summarize_context, which
  would send a whole pack).
- kh-ex carries tt-kh-examples-v0 at generation and compile corrections;
  kh-exsk adds tt-kh-skeleton-v0 (with its sha256); kh-allsk adds the
  kernels pack with the lookup (host and kernel headers at generation,
  nearest names at corrections), retrieval over the tt-metal docs, and a
  restart after 3 failed corrections, as sweep7's all does for cpp -> tt.
- With stand-in fetched kernels, the runner's checks before any directory
  pass (skipped when a generated pack, gitignored, is not built here).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lassi.bench import load_suite
from lassi.core import runner as runner_module
from lassi.core.recipe import load_recipe
from lassi.prompts import skeletons

REPO = Path(__file__).resolve().parents[2]
RECIPES = REPO / "projects" / "lassi-demo" / "sweep9" / "gpt-oss-120b"
SWEEP8 = REPO / "projects" / "lassi-demo" / "sweep8" / "gpt-oss-120b"
SUITE = load_suite(REPO / "assets" / "bench" / "tt-pairs-v0-kh.yaml")
COMBOS = ("kh-ex", "kh-exsk", "kh-allsk")
HALVES = {"maxabs": "max_abs", "pcc": "pcc"}
NAMES = [f"{combo}-{half}" for combo in COMBOS for half in HALVES]
STAGES = ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
SKELETON = "tt-kh-skeleton-v0"
KH_PACK = "tt-metalium-5280a9cf-kh"
GENERATED = (
    REPO / "assets" / "examples" / "tt-kh-examples-v0" / "hello_world_compute_kernel" / "kernels" / "compute"
    / "void_compute_kernel.cpp",
    REPO / "assets" / "context" / KH_PACK / "kernel_api.ttkh.txt",
)


def load(name: str) -> dict[str, Any]:
    """Return the resolved data of one sweep9 recipe."""
    return load_recipe(RECIPES / f"{name}.yaml").data


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a sources root with a SYNTHETIC stand-in at every upstream support path of the suite."""
    root = tmp_path_factory.mktemp("kh-sources")
    for found in SUITE.items.values():
        for spec in found.languages.values():
            for entry in spec.support.values():
                if not entry.tracked:
                    (root / entry.path).parent.mkdir(parents=True, exist_ok=True)
                    (root / entry.path).write_bytes(f"// SYNTHETIC stand-in for {entry.path}\n".encode("ascii"))
    return root


def test_sweep9_holds_the_six_recipes() -> None:
    assert sorted(path.stem for path in RECIPES.glob("*.yaml")) == sorted(NAMES)
    assert sorted(path.name for path in RECIPES.parent.iterdir()) == ["gpt-oss-120b"]


@pytest.mark.parametrize("name", NAMES)
def test_each_recipe_binds_cpp_to_ttkh_with_sweep8s_model_settings(name: str) -> None:
    data = load(name)
    half = name.rsplit("-", 1)[1]
    sweep8 = load_recipe(SWEEP8 / f"v1ex-{half}.yaml").data
    for key in ("model", "llm", "extract", "fixes", "sandbox", "faithful"):
        assert data[key] == sweep8[key], key
    assert data["directions"] == [{"source": "cpp", "target": "ttkh"}]
    assert (data["bench"]["suite"], data["bench"]["split"], data["prompts"]) == (SUITE.name, "unassigned", "tt-kh-v0")
    assert data["toolchain"] == {"cpp": "gcc-native", "tt": "ttmetal-host", "ttkh": "ttmetal-host"}
    assert {language: spec["kind"] for language, spec in data["executor"].items()} == {
        "cpp": "native", "tt": "ttsim", "ttkh": "ttsim"}
    assert data["trials"] == {"n": 2}
    assert data["loop"]["max_corrections"] == 8
    assert data["loop"]["correct_on_mismatch"] is True and data["loop"]["correct_on_sim_gap"] is True
    assert data["stages"] == STAGES
    assert data["oracle"] == {"kind": "binary_io", "metric": HALVES[half], "threshold": "from_baseline"}
    assert data["project"] == name


@pytest.mark.parametrize("half", sorted(HALVES))
def test_each_half_covers_its_items_once_by_their_metric(half: str) -> None:
    for combo in COMBOS:
        items = load(f"{combo}-{half}")["bench"]["items"]
        assert all(SUITE.items[item].tolerance.metric == HALVES[half] for item in items), items
    both = {item for half_name in HALVES for item in load(f"kh-ex-{half_name}")["bench"]["items"]}
    assert both == set(SUITE.items)


@pytest.mark.parametrize("half", sorted(HALVES))
def test_the_combinations_add_examples_then_the_skeleton_then_the_rest(half: str) -> None:
    ex, exsk, allsk = (load(f"{combo}-{half}") for combo in COMBOS)
    examples = {"pack": "tt-kh-examples-v0", "max_chars": 32000, "prompts": ["generate", "correct"]}
    assert ex["examples"] == exsk["examples"] == allsk["examples"] == examples
    assert "skeleton" not in ex and "context" not in ex and "lookup" not in ex and "retrieval" not in ex
    digest = skeletons.load_skeleton(skeletons.skeleton_dir(SKELETON)).sha256
    section = {"name": SKELETON, "prompts": ["generate", "correct"], "sha256": digest}
    assert exsk["skeleton"] == allsk["skeleton"] == section
    assert "context" not in exsk and "lookup" not in exsk and "restart_after" not in exsk["loop"]
    assert allsk["context"] == [KH_PACK]
    assert allsk["lookup"]["prompts"] == ["generate", "correct"] and allsk["lookup"]["nearest"] == 3
    assert allsk["lookup"]["max_chars"] == 32000
    core = allsk["lookup"]["core"]
    assert "tt-metalium/host_api.hpp" in core and "api/dataflow/dataflow_api.h" in core
    assert "api/compute/tile_move_copy.h" in core
    assert allsk["retrieval"] == {"index": "tt-metal-docs", "k": 4, "max_chars": 6000,
                                  "prompts": ["generate", "correct"]}
    assert allsk["loop"]["restart_after"] == 3


@pytest.mark.parametrize("name", NAMES)
def test_the_runners_checks_before_any_directory_pass(name: str, sources: Path) -> None:
    missing = [str(path) for path in GENERATED if not path.is_file()]
    if missing:
        pytest.skip(f"a generated pack is not built here ({missing[0]}); run tools/build_tt_examples.py and "
                    "tools/build_tt_context.py --kernels")
    options = runner_module.RunOptions(bench_root=sources)
    registry = runner_module.DEFAULT_REGISTRY
    recipe = runner_module._load(RECIPES / f"{name}.yaml", options, registry)
    settings = runner_module._settings(recipe)
    bench = runner_module._bench(recipe, settings, options)
    runner_module._check_plan(recipe, registry, settings, bench)
    assert runner_module._examples_pack(recipe)
    runner_module._check_oracle(recipe, registry, bench)


def test_every_core_declaration_of_kh_allsk_fits_its_max_chars() -> None:
    if not GENERATED[1].is_file():
        pytest.skip("the kernels pack is not built here; build it with uv run tools/build_tt_context.py --kernels")
    from types import SimpleNamespace

    from lassi.bench import Direction
    from lassi.core.stages import core_declarations, pack_declarations
    from lassi.prompts.assets import load_tree, pack_text

    pack = pack_text(KH_PACK, load_tree(REPO / "assets" / "context" / KH_PACK))
    for half in HALVES:
        data = load(f"kh-allsk-{half}")
        context: Any = SimpleNamespace(recipe=SimpleNamespace(data=data, path=RECIPES), packs={"ttkh": pack},
                                       fragments={}, direction=Direction("cpp", "ttkh"))
        carried = core_declarations(context)
        wanted = [item for item in pack_declarations(pack) if item.header in set(data["lookup"]["core"])]
        assert carried == wanted, "every declaration of the core headers is carried whole"
        assert {item.header for item in carried} == set(data["lookup"]["core"]), "each core header declares some"
