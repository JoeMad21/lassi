"""Tests for the faithful check of toolchains and executors at load (task P16.2; the faithful-check gap).

Bible: Design Principle 4 and Agent Rule 4 (faithful mode reproduces
upstream behavior, quirks included), Project Recipes (Notes: the faithful
toolchain rule; recipes are validated against component capabilities at
load time, before any model call), Execution Backends (the native row), and
PHASE-NOTES All Phases (gcc-native is a host g++ build upstream never used). Plan:
plans/p16-adversarial.md, P16.2, which closes the gap PHASE-NOTES All
Phases states (the P4.15 finding, plans/spikes/p4-p2-review.md, question 1).

The gap: lassi/core/recipe.py refused a faithful recipe only for a
toolchain that declares openmp_multicore, so a faithful child of
projects/lassi-demo/rngd-cpu.yaml that binds gcc-native, or nvcpp-cc80 with
the native executor, loaded. The contract these tests fix ([DESIGN], task
P16.2): load_recipe refuses, with a RecipeError, a recipe with faithful:
true that binds

- gcc-native, whatever its executor; the message names the binding
  (toolchain.<language>), the toolchain, and faithful;
- nvcpp-cc80 together with the native executor, in the single executor
  form or as the executor of nvcpp-cc80's language in the per-language
  form; the message names the toolchain's binding and the executor's
  binding (executor.kind, or executor.<language>), and faithful.

The rule goes by capability (lassi.core.recipe OFFLOAD_CAPABILITIES): an
offload build (openmp_offload) with an executor that runs programs
(runs_code) and takes no device section (takes_device). So ttsim, which
runs programs on no GPU, is refused with nvcpp-cc80 as native is.

What stays: a recipe that is not faithful binds either; a faithful recipe
binds nvcpp-cc80 with the none executor (lassi-repro's tier 1) or the gpu
executor (the bible's lassi-repro block); and every recipe under projects/
still loads, the faithful ones among them (lassi-repro, which binds no model
yet, through a child that names only the mock).

Known cases, not complete: the check covers the two cases PHASE-NOTES
states and no more. A faithful recipe still loads that runs a CUDA build
(nvcc-sm80) or a HIP build (hipcc-gfx942) on an executor with no GPU, or
binds nvcpp-cc80, an NVIDIA offload build, with the gpu executor on a rocm
device. A test below pins each, so closing one changes it and the limits
text in lassi.core.recipe together.

lassi.core.recipe reads the capability runs_code from
lassi.core.capabilities RUNS_CODE; lassi.core.stages keeps its own
RUNS_CODE, which must stay equal to it (test).

Recipes here are written per test and extend projects/base.yaml through the
default roots; the components are the registered ones, which loading never
builds. No value in this module is a measurement.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.core import capabilities, recipe, stages
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.recipe import Recipe, RecipeError, load_recipe, resolved_data
from lassi.core.registry import DEFAULT_REGISTRY

REPO = Path(__file__).resolve().parents[2]
PROJECTS = REPO / "projects"
CPU_DEMO = PROJECTS / "lassi-demo" / "rngd-cpu.yaml"
REPRO = PROJECTS / "lassi-repro" / "recipe.yaml"
GCC = "gcc-native"
NVCPP = "nvcpp-cc80"
NVCC = "nvcc-sm80"
HIPCC = "hipcc-gfx942"
NATIVE = "native"
TTSIM = "ttsim"
CUDA_0 = {"kind": "cuda", "indices": [0]}
ROCM_0 = {"kind": "rocm", "indices": [0]}


def write(directory: Path, data: Mapping[str, Any], name: str = "faithful-under-test.yaml") -> Path:
    """Write `data` as a YAML recipe under `directory` and return its path."""
    path = directory / name
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def cpp_recipe(*, faithful: bool, executor: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recipe over base.yaml that binds gcc-native for cpp, with `executor`."""
    return {
        "extends": "base",
        "faithful": faithful,
        "model": {"backend": "mock", "id": "mock-reference"},
        "bench": {"suite": "tt-pairs-v0", "split": "unassigned"},
        "directions": [{"source": "tt", "target": "cpp"}],
        "toolchain": {"cpp": GCC},
        "stages": ["baseline", "generate", "compile_loop", "run_loop"],
        "executor": dict(executor),
    }


def omp_recipe(*, faithful: bool, executor: Mapping[str, Any], omp: str = NVCPP) -> dict[str, Any]:
    """Return a recipe over base.yaml that binds nvcc-sm80 for cuda and `omp` (nvcpp-cc80) for omp, with `executor`."""
    return {
        "extends": "base",
        "faithful": faithful,
        "model": {"backend": "mock", "id": "mock-reference"},
        "bench": {"suite": "lassi-hecbench-10", "split": "eval"},
        "directions": [{"source": "cuda", "target": "omp"}],
        "toolchain": {"cuda": NVCC, "omp": omp},
        "stages": ["baseline", "generate", "compile_loop", "run_loop"],
        "executor": dict(executor),
    }


def refusal(path: Path) -> str:
    """Return the message of the RecipeError that loading `path` raises."""
    with pytest.raises(RecipeError) as info:
        load_recipe(path)
    return str(info.value)


def assert_names(message: str, *needles: str) -> None:
    """Assert the refusal names faithful and every needle."""
    for needle in ("faithful", *needles):
        assert needle in message, f"the refusal does not name {needle!r}: {message}"


# ---------------------------------------------------------------------------
# Refused at load


@pytest.mark.parametrize(
    "executor",
    [{"kind": "none"}, {"kind": NATIVE}, {"cpp": NATIVE}, {"cpp": {"kind": NATIVE}}],
    ids=["none", "native", "native-per-language", "native-kind-section"],
)
def test_a_faithful_recipe_binding_gcc_native_is_refused_at_load(tmp_path: Path, executor: dict[str, Any]) -> None:
    message = refusal(write(tmp_path, cpp_recipe(faithful=True, executor=executor)))
    assert_names(message, "toolchain.cpp", GCC)


@pytest.mark.parametrize(
    ("executor", "where"),
    [
        pytest.param({"kind": NATIVE}, "executor.kind", id="single-form"),
        pytest.param({"cuda": "none", "omp": NATIVE}, "executor.omp", id="per-language-name"),
        pytest.param({"cuda": "none", "omp": {"kind": NATIVE}}, "executor.omp", id="per-language-kind-section"),
    ],
)
def test_a_faithful_recipe_binding_nvcpp_cc80_with_the_native_executor_is_refused_at_load(
    tmp_path: Path, executor: dict[str, Any], where: str
) -> None:
    message = refusal(write(tmp_path, omp_recipe(faithful=True, executor=executor)))
    assert_names(message, "toolchain.omp", NVCPP, where, NATIVE)


@pytest.mark.parametrize(
    "executor",
    [{"cuda": "none", "omp": TTSIM}, {"cuda": "none", "omp": {"kind": TTSIM}}],
    ids=["per-language-name", "per-language-kind-section"],
)
def test_the_rule_goes_by_capability_so_ttsim_is_refused_with_nvcpp_cc80_as_native_is(
    tmp_path: Path, executor: dict[str, Any]
) -> None:
    # ttsim declares runs_code and not takes_device, as native does: its runs see no GPU either.
    declared = DEFAULT_REGISTRY.get("Executor", TTSIM).capabilities
    assert "runs_code" in declared and "takes_device" not in declared, declared
    message = refusal(write(tmp_path, omp_recipe(faithful=True, executor=executor)))
    assert_names(message, "toolchain.omp", NVCPP, "executor.omp", TTSIM)


def test_the_gap_as_stated_a_faithful_child_of_the_cpu_demo_binding_nvcpp_cc80_is_refused(tmp_path: Path) -> None:
    # rngd-cpu.yaml binds the native executor and the proxy nvcpp-multicore; the child binds nvcpp-cc80 instead, so
    # the proxy rule no longer applies. The roots find rngd-cpu by name, then base.yaml for rngd-compile.yaml.
    child = {"extends": CPU_DEMO.stem, "faithful": True, "toolchain": {"omp": NVCPP}}
    path = write(tmp_path, child)
    with pytest.raises(RecipeError) as info:
        load_recipe(path, roots=[CPU_DEMO.parent, PROJECTS])
    assert_names(str(info.value), "toolchain.omp", NVCPP, "executor.kind", NATIVE)


def test_the_cpu_demo_child_loads_when_it_is_not_faithful(tmp_path: Path) -> None:
    child = {"extends": CPU_DEMO.stem, "toolchain": {"omp": NVCPP}}
    recipe = load_recipe(write(tmp_path, child), roots=[CPU_DEMO.parent, PROJECTS])
    assert recipe.data["faithful"] is False


# ---------------------------------------------------------------------------
# What the check leaves alone


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(cpp_recipe(faithful=False, executor={"kind": NATIVE}), id="not-faithful-gcc-native"),
        pytest.param(omp_recipe(faithful=False, executor={"kind": NATIVE}), id="not-faithful-nvcpp-native"),
        pytest.param(omp_recipe(faithful=True, executor={"kind": "none"}), id="faithful-nvcpp-compile-only"),
        pytest.param(omp_recipe(faithful=True, executor={"kind": "gpu", "device": CUDA_0}), id="faithful-nvcpp-gpu"),
    ],
)
def test_recipes_outside_the_check_still_load(tmp_path: Path, data: dict[str, Any]) -> None:
    recipe = load_recipe(write(tmp_path, data))
    assert recipe.data["faithful"] is data["faithful"]


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(omp_recipe(faithful=True, executor={"cuda": NATIVE, "omp": "none"}), id="nvcc-sm80-on-native"),
        pytest.param(
            omp_recipe(faithful=True, executor={"cuda": "none", "omp": NATIVE}, omp=HIPCC), id="hipcc-gfx942-on-native"
        ),
        pytest.param(omp_recipe(faithful=True, executor={"kind": "gpu", "device": ROCM_0}), id="nvcpp-cc80-on-rocm"),
    ],
)
def test_known_cases_not_complete_these_faithful_pairs_still_load(tmp_path: Path, data: dict[str, Any]) -> None:
    # Known cases, not complete: the check closes only the gap PHASE-NOTES states. Each pair below runs a GPU build
    # where upstream's GPU is not, yet loads; closing one changes this test and lassi.core.recipe's limits text.
    assert load_recipe(write(tmp_path, data)).data["faithful"] is True


def test_the_recipe_check_reads_runs_code_from_capabilities_and_stages_agrees() -> None:
    assert capabilities.RUNS_CODE == "runs_code"
    assert recipe.RUNS_CODE is capabilities.RUNS_CODE
    assert stages.RUNS_CODE == capabilities.RUNS_CODE


def committed_recipes() -> list[Path]:
    """Return every run recipe under projects/ but base.yaml, the shared base that states no bench or stages.

    The train recipes under projects/train/ are train recipes, which
    load_train_recipe loads (tests/train/test_train_recipes.py), not run
    recipes, so they are left out (2026-10-23).
    """
    train = PROJECTS / "train"
    found = PROJECTS.rglob("*.yaml")
    return sorted(path for path in found if path != PROJECTS / "base.yaml" and train not in path.parents)


def load_committed(path: Path, directory: Path) -> Recipe:
    """Load a committed recipe as it is, or, when it binds no model, through a child that names only the mock.

    lassi-repro binds no model until P3 (its comments; bible Project Recipes,
    Notes), so it loads through such a child, as tests/core/
    test_single_executor_golden.py loads it.
    """
    if "model" in resolved_data(path):
        return load_recipe(path)
    name = path.parent.name if path.name == "recipe.yaml" else path.stem
    child = write(directory, {"extends": name, "model": {"backend": "mock", "id": "mock-reference"}}, "mock-child.yaml")
    return load_recipe(child, roots=[path.parent, PROJECTS])


@pytest.mark.parametrize("path", committed_recipes(), ids=lambda path: path.relative_to(PROJECTS).as_posix())
def test_every_committed_project_recipe_still_loads(tmp_path: Path, path: Path) -> None:
    load_committed(path, tmp_path)


def test_the_committed_faithful_recipes_still_load(tmp_path: Path) -> None:
    faithful = []
    for index, path in enumerate(committed_recipes()):
        directory = tmp_path / str(index)
        directory.mkdir()
        if load_committed(path, directory).data["faithful"] is True:
            faithful.append(path)
    assert REPRO in faithful, f"faithful recipes under projects/: {faithful}"
