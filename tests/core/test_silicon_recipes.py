"""Tests for sweep10-si, the ttkernel path with the Wormhole cards as the loop's final test (task P17.56).

Bible: Project Recipes (the opt-in final_executor section, task P17.55),
Execution Backends (tt_silicon), Agent Rules 2, 6, and 10. The owner's
answers of 2026-10-29 to OQ-078 and OQ-099 and the MLIR-only direction of
that day. plans/p17-portable.md, P17.56; plans/runs/silicon-launch.md.

What they fix: projects/lassi-demo/sweep10-si/gpt-oss-120b holds the four
recipes mlir-{ref,rag}-si-{maxabs,pcc}.yaml; each loads, passes the loader's
binding checks and the runner's final-executor check, and resolves to the
sweep10 recipe it extends but for the final_executor section (tt_silicon,
private-per-run hugepages, the reset filter, the device profiler at kernel)
and sandbox.mem_gb 32 (PROJECTED); every direction targets ttkernel, the
MLIR path; the trial's own executors are the base's; and the final
executor declares silicon, so the launcher's plan names each recipe.

Nothing is built or run here, and no card is read. No value is a
measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lassi.core import runner
from lassi.core.recipe import final_executor_binding, load_recipe, run_executor_bindings
from lassi.core.registry import DEFAULT_REGISTRY

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "projects" / "lassi-demo"
SET = DEMO / "sweep10-si" / "gpt-oss-120b"
CASES = [(combo, half) for combo in ("ref", "rag") for half in ("maxabs", "pcc")]
FINAL = {"kind": "tt_silicon", "hugepages": "private-per-run", "reset_filter": True, "device_profiler": "kernel"}


def without(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Return `data` less the named top-level keys."""
    return {key: value for key, value in data.items() if key not in keys}


def test_the_set_holds_the_four_mlir_recipes_and_nothing_else() -> None:
    names = sorted(path.name for path in (DEMO / "sweep10-si").rglob("*.yaml"))
    assert names == sorted(f"mlir-{combo}-si-{half}.yaml" for combo, half in CASES)


@pytest.mark.parametrize(("combo", "half"), CASES)
def test_a_recipe_is_its_base_with_silicon_as_the_final_test(combo: str, half: str) -> None:
    recipe = load_recipe(SET / f"mlir-{combo}-si-{half}.yaml")
    base = load_recipe(DEMO / "sweep10" / "gpt-oss-120b" / f"mlir-{combo}-{half}.yaml")
    assert recipe.data["final_executor"] == FINAL
    assert recipe.data["sandbox"] == {**base.data["sandbox"], "mem_gb": 32}
    assert without(recipe.data, "final_executor", "sandbox", "project") == without(base.data, "sandbox", "project")
    assert recipe.data["loop"]["correct_on_mismatch"] is True
    assert {direction["target"] for direction in recipe.data["directions"]} == {"ttkernel"}, "the MLIR path only"
    ours = [(binding.where, binding.name) for binding in run_executor_bindings(recipe.bindings)]
    theirs = [(binding.where, binding.name) for binding in run_executor_bindings(base.bindings)]
    assert ours == theirs, "the trial's own executors stay the base's (ttsim for ttkernel)"


@pytest.mark.parametrize(("combo", "half"), CASES)
def test_a_recipe_passes_the_runners_final_executor_check_and_names_silicon(combo: str, half: str) -> None:
    recipe = load_recipe(SET / f"mlir-{combo}-si-{half}.yaml")
    runner._check_final_executor(recipe, DEFAULT_REGISTRY)
    binding = final_executor_binding(recipe.bindings)
    assert binding is not None and binding.name == "tt_silicon"
    capabilities = DEFAULT_REGISTRY.get("Executor", binding.name).capabilities
    assert {"runs_code", "sandboxed", "silicon"} <= capabilities
    assert "simulator" not in capabilities, "a silicon run never names a simulator (Agent Rule 2)"
