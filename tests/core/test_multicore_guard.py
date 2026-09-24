"""Tests for the faithful guard against the multicore proxy toolchain (task P1.10).

Bible: Design Principle 4 and Agent Rule 4 (faithful mode reproduces
upstream behavior), Harness Contract (the `-mp=multicore` proxy checks
outputs, never runtime), Component Interfaces (capability rule), Project
Recipes (Notes: recipes are validated against component capabilities before
any model call).

The contract these tests fix (the P1.6 commit audit, carried out under
P1.10): upstream builds OpenMP for GPU offload, so a recipe with
`faithful: true` that binds a toolchain whose capabilities include
`openmp_multicore` (the proxy preset nvcpp-multicore declares it) is refused
at load or by the runner, before anything is written and before any build,
with a message that names the capability or the toolchain and the faithful
flag. The guard reads the capability, not the preset's name, and holds for a
toolchain bound for either language of a direction. A recipe with
`faithful: false` may still bind the proxy (the CPU demo recipe does), and a
faithful recipe binding a toolchain without the capability still runs.

Runs use the p0-smoke template prompt set with the real mock backend, the
real baseline, generate, compile_loop, and run_loop stages, the none
executor, and fake toolchains that compile nothing. Bench sources are small
SYNTHETIC files, not HeCBench sources. No value in this module is a
measurement.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import load_suite
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.interfaces import BuildResult
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.executors.none import NoneExecutor
from lassi.llm import MockBackend

REPO = Path(__file__).resolve().parents[2]
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
CPU_DEMO = REPO / "projects" / "lassi-demo" / "rngd-cpu.yaml"
MULTICORE = "openmp_multicore"
PROXY_PRESET = "nvcpp-multicore"
# Stages that together reproduce every named fix, so faithful: true runs with the template prompt set.
STAGES = ["baseline", "generate", "compile_loop", "run_loop"]
# Fake toolchain names: one declaring the proxy capability under a name that is not the preset's, and two without.
PROXY = "proxy-cc"
PLAIN_OMP = "plain-cc"
PLAIN_CUDA = "plain-cuda"

# SYNTHETIC bench sources for the one item the runs translate.
SOURCES = {
    "omp": "#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  std::printf(\"done\\n\");\n}\n",
    "cuda": "#include <cstdio>\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n",
}


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """Return a bench root holding the SYNTHETIC layout sources where the suite manifest lays them out."""
    root = tmp_path / "bench"
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, text in SOURCES.items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def fake_toolchain(capabilities: frozenset[str], builds: list[str]) -> type:
    """Return a Toolchain class without PIN declaring `capabilities`; it logs each build and compiles nothing."""

    class FakeToolchain:
        """Logs the build directory and reports a PLACEHOLDER artifact; nothing is written or run."""

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Log `workdir` and return a PLACEHOLDER artifact path with no diagnostics."""
            builds.append(str(workdir))
            return BuildResult(artifact=Path(workdir) / "PLACEHOLDER-artifact", diagnostics=[])

    FakeToolchain.capabilities = capabilities
    return FakeToolchain


def make_registry(builds: list[str]) -> Registry:
    """Return a test Registry: the real mock, stages, and none executor; a proxy and two plain fake toolchains."""
    registry = Registry()
    registry.register("LLMBackend", "mock", MockBackend)
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Toolchain", PROXY, fake_toolchain(frozenset({MULTICORE, "diagnostics"}), builds))
    registry.register("Toolchain", PLAIN_OMP, fake_toolchain(frozenset({"diagnostics"}), builds))
    registry.register("Toolchain", PLAIN_CUDA, fake_toolchain(frozenset({"diagnostics"}), builds))
    for name in STAGES:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    return registry


def recipe_data(*, faithful: bool, omp: str, cuda: str = PLAIN_CUDA, source: str = "cuda") -> dict[str, Any]:
    """Return a one-trial recipe binding `omp` and `cuda`; `source` picks the direction's source language."""
    target = "omp" if source == "cuda" else "cuda"
    return {
        "extends": "base",
        "faithful": faithful,
        "model": {"backend": "mock", "id": "mock-reference"},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": source, "target": target}],
        "prompts": "p0-smoke",
        "toolchain": {"cuda": cuda, "omp": omp},
        "stages": list(STAGES),
        "executor": {"kind": "none"},
        "trials": {"n": 1},
    }


def write_recipe(path: Path, data: Mapping[str, Any]) -> Path:
    """Write `data` as YAML at `path` and return the path."""
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def assert_refused(error: BaseException, toolchain: str) -> None:
    """Assert the refusal names the proxy (its capability or the toolchain) and the faithful flag."""
    message = str(error)
    assert MULTICORE in message or toolchain in message, message
    assert "faithful" in message, message


# ---------------------------------------------------------------------------
# The guard


@pytest.mark.parametrize("source", ["cuda", "omp"], ids=["bound-for-the-target", "bound-for-the-source"])
def test_a_faithful_recipe_binding_a_multicore_toolchain_is_refused_before_anything_is_written(
    tmp_path: Path, bench: Path, source: str
) -> None:
    builds: list[str] = []
    runs_root = tmp_path / "runs"
    recipe = write_recipe(tmp_path / "faithful-proxy.yaml", recipe_data(faithful=True, omp=PROXY, source=source))
    options = RunOptions(runs_root=runs_root, run_id="test-run", bench_root=bench, registry=make_registry(builds))
    with pytest.raises((RecipeError, RunError)) as info:
        run_recipe(recipe, options)
    assert_refused(info.value, PROXY)
    assert not runs_root.exists(), "nothing may be written before the refusal"
    assert builds == [], "nothing may be built before the refusal"


def test_the_proxy_preset_is_refused_in_a_faithful_recipe(tmp_path: Path, bench: Path) -> None:
    runs_root = tmp_path / "runs"
    data = recipe_data(faithful=True, omp=PROXY_PRESET, cuda="nvcc-sm80")
    recipe = write_recipe(tmp_path / "faithful-preset.yaml", data)
    with pytest.raises((RecipeError, RunError)) as info:
        run_recipe(recipe, RunOptions(runs_root=runs_root, run_id="test-run", bench_root=bench))
    assert_refused(info.value, PROXY_PRESET)
    assert not runs_root.exists(), "nothing may be written before the refusal"


# ---------------------------------------------------------------------------
# What the guard leaves alone


@pytest.mark.parametrize(
    ("faithful", "omp"), [(False, PROXY), (True, PLAIN_OMP)], ids=["not-faithful-proxy", "faithful-plain"]
)
def test_recipes_outside_the_guard_still_run(tmp_path: Path, bench: Path, faithful: bool, omp: str) -> None:
    builds: list[str] = []
    recipe = write_recipe(tmp_path / "allowed.yaml", recipe_data(faithful=faithful, omp=omp))
    registry = make_registry(builds)
    options = RunOptions(runs_root=tmp_path / "runs", run_id="test-run", bench_root=bench, registry=registry)
    run_dir = run_recipe(recipe, options)
    assert len(list(run_dir.rglob("trial.json"))) == 1
    assert builds, "the run built the reference and attempt 0"


def test_the_cpu_demo_recipe_still_loads() -> None:
    recipe = load_recipe(CPU_DEMO)
    assert recipe.data["faithful"] is False
    assert recipe.data["toolchain"]["omp"] == PROXY_PRESET
