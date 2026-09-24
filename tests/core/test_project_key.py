"""Tests for the recipe's explicit `project` key (task P1.10, first criterion).

Bible: Project Recipes (Notes), Design Principles 1 and 5, Readability
Standards (Naming: `<project>/<arm>/<bench>/<direction>/<item>/run<NN>`).

The contract these tests fix, from the P1.10 acceptance criteria
(plans/p1-faithful.md) and the planning decision "the resolved recipe
carries its project name in an explicit `project` key":

- A run recipe may set the top-level key `project`, a string. It merges as
  any scalar does: inherited through `extends`, the nearest file that sets
  it wins.
- Without it anywhere in the chain, the project is the recipe's name (the
  file stem, or the directory name for a file called recipe.yaml) of the
  file being loaded, never an ancestor's.
- The resolved mapping always holds `project`, so the canonical YAML and
  the recipe hash cover it: two recipes that differ only in project hash
  differently, and a recipe saved as recipe.resolved.yaml reloads with its
  project and hash.
- The project is the first segment of every trial id the runner writes, so
  a rerun from a run's recipe.resolved.yaml writes the same trial ids.
- A project that is not one trial-id segment, or that names an entry of the
  run tree itself (texts, parquet, ...), is refused before anything is
  written.

Runs use the p0-smoke template prompt set with the real mock backend, the
real generate and compile_loop stages, the compile-only executor, and a fake
toolchain that compiles nothing. Bench sources are small SYNTHETIC files, not
HeCBench sources. No value in this module is a measurement.
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
from lassi.core.recipe import RecipeError, load_recipe, resolved_yaml
from lassi.core.registry import Registry
from lassi.core.runner import RESOLVED_RECIPE, RunError, RunOptions, run_recipe
from lassi.core.stages import CompileLoopStage, GenerateStage
from lassi.executors.none import NoneExecutor
from lassi.llm import MockBackend

REPO = Path(__file__).resolve().parents[2]
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
MOCK_ID = "mock-reference"
COMPILED = "S4"

# SYNTHETIC bench sources for the one item the runs translate.
SOURCES = {
    "omp": "#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  std::printf(\"done\\n\");\n}\n",
    "cuda": "#include <cstdio>\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n",
}

# Names the run tree uses for its own entries (lassi.core.runner): a project with one of them would put trial
# directories among them.
RUN_TREE_NAMES = ("texts", "parquet")
# Values that are not one trial-id segment: a separator, a leading dash, and the empty string.
NOT_ONE_SEGMENT = ("a/b", "-lead", "")


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


# ---------------------------------------------------------------------------
# Fakes, recipes, and runs


class FakeToolchain:
    """A Toolchain without PIN, registered as nvcc-sm80: writes files and a PLACEHOLDER artifact; compiles nothing."""

    name = "nvcc-sm80"
    capabilities = frozenset({"diagnostics", "emits_warnings"})

    def build(self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None) -> BuildResult:
        """Write every file under `workdir` and report a build with no diagnostics."""
        workdir = Path(workdir)
        for path, text in [*files.items(), *(harness or {}).items()]:
            target = workdir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
        artifact = workdir / "main"
        artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
        return BuildResult(artifact=artifact, diagnostics=[])


def make_registry() -> Registry:
    """Return a test Registry: the real mock backend, generate and compile_loop, the none executor, a fake nvcc."""
    registry = Registry()
    registry.register("LLMBackend", "mock", MockBackend)
    registry.register("Stage", "generate", GenerateStage)
    registry.register("Stage", "compile_loop", CompileLoopStage)
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Toolchain", "nvcc-sm80", FakeToolchain)
    return registry


def recipe_data(**changes: Any) -> dict[str, Any]:
    """Return a runnable p0-smoke-style recipe for layout, omp to cuda, one trial; `changes` add or replace keys."""
    data: dict[str, Any] = {
        "extends": "base",
        "model": {"backend": "mock", "id": MOCK_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": "omp", "target": "cuda"}],
        "prompts": "p0-smoke",
        "toolchain": {"cuda": "nvcc-sm80"},
        "stages": ["generate", "compile_loop"],
        "executor": {"kind": "none"},
        "trials": {"n": 1},
    }
    data.update(changes)
    return data


def write_recipe(path: Path, data: Mapping[str, Any]) -> Path:
    """Write `data` as YAML at `path`, creating directories, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


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


def run(recipe: Path, runs_root: Path, bench_root: Path, run_id: str = "test-run") -> Path:
    """Run `recipe` with the test registry; return the run directory."""
    options = RunOptions(runs_root=runs_root, run_id=run_id, bench_root=bench_root, registry=make_registry())
    return run_recipe(recipe, options)


def trial_ids(run_dir: Path) -> list[str]:
    """Return the trial ids of a run tree, sorted: the path of each trial.json's directory under the run directory."""
    return sorted(path.parent.relative_to(run_dir).as_posix() for path in run_dir.rglob("trial.json"))


def load(path: Path) -> Any:
    """Load a recipe with the test registry and the default roots (projects/base.yaml is found by name)."""
    return load_recipe(path, registry=make_registry())


def project_of(recipe: Any) -> Any:
    """Return the resolved recipe's project; fail clearly while the loader does not materialize it."""
    if "project" not in recipe.data:
        pytest.fail(f"the resolved recipe {recipe.name!r} has no project key; task P1.10 adds it")
    return recipe.data["project"]


# ---------------------------------------------------------------------------
# The loader: the key, its default, inheritance, and the hash


def test_an_explicit_project_key_sets_the_project(tmp_path: Path) -> None:
    recipe = load(write_recipe(tmp_path / "some-file.yaml", recipe_data(project="demo-project")))
    assert project_of(recipe) == "demo-project"


def test_without_a_project_key_the_recipe_name_is_the_project(tmp_path: Path) -> None:
    assert project_of(load(write_recipe(tmp_path / "plain-name.yaml", recipe_data()))) == "plain-name"
    assert project_of(load(write_recipe(tmp_path / "proj" / "recipe.yaml", recipe_data()))) == "proj"


def test_the_default_is_the_loaded_files_name_not_an_ancestors(tmp_path: Path) -> None:
    write_recipe(tmp_path / "ancestor.yaml", recipe_data())
    leaf = write_recipe(tmp_path / "leaf.yaml", {"extends": "ancestor.yaml"})
    assert project_of(load(leaf)) == "leaf"


def test_the_project_key_is_inherited_and_the_nearest_file_wins(tmp_path: Path) -> None:
    write_recipe(tmp_path / "grand.yaml", recipe_data(project="from-grand"))
    write_recipe(tmp_path / "mid.yaml", {"extends": "grand.yaml", "project": "from-mid"})
    write_recipe(tmp_path / "mid-plain.yaml", {"extends": "grand.yaml"})
    assert project_of(load(write_recipe(tmp_path / "leaf.yaml", {"extends": "mid.yaml"}))) == "from-mid"
    own = write_recipe(tmp_path / "own.yaml", {"extends": "mid.yaml", "project": "from-own"})
    assert project_of(load(own)) == "from-own"
    assert project_of(load(write_recipe(tmp_path / "deep.yaml", {"extends": "mid-plain.yaml"}))) == "from-grand"


@pytest.mark.parametrize("value", [3, ["a"], {"name": "a"}, True], ids=["int", "list", "mapping", "bool"])
def test_a_project_that_is_not_a_string_is_refused_at_load(tmp_path: Path, value: Any) -> None:
    path = write_recipe(tmp_path / "typed.yaml", recipe_data(project=value))
    with pytest.raises(RecipeError) as info:
        load(path)
    message = str(info.value)
    assert "typed.yaml" in message and "project" in message, message
    assert "unknown key" not in message, f"the loader does not know the project key yet: {message}"


def test_a_null_project_is_a_required_choice_with_no_value(tmp_path: Path) -> None:
    path = tmp_path / "null-project.yaml"
    path.write_bytes(b"extends: plain.yaml\nproject: null\n")
    write_recipe(tmp_path / "plain.yaml", recipe_data())
    with pytest.raises(RecipeError) as info:
        load(path)
    message = str(info.value)
    assert "project" in message and "no value" in message, message


def test_the_resolved_recipe_always_writes_project(tmp_path: Path) -> None:
    for name, data, expected in (
        ("keyed.yaml", recipe_data(project="demo-project"), "demo-project"),
        ("unkeyed.yaml", recipe_data(), "unkeyed"),
    ):
        recipe = load(write_recipe(tmp_path / name, data))
        body = resolved_yaml(recipe).split("\n", 2)[2]
        assert yaml.safe_load(body).get("project") == expected, body
        assert f"\nproject: {expected}\n" in "\n" + recipe.canonical_yaml, recipe.canonical_yaml


def test_the_project_enters_the_recipe_hash(tmp_path: Path) -> None:
    first = load(write_recipe(tmp_path / "first.yaml", recipe_data(project="one")))
    other = load(write_recipe(tmp_path / "other.yaml", recipe_data(project="two")))
    renamed = load(write_recipe(tmp_path / "renamed.yaml", recipe_data(project="one")))
    assert project_of(first) == "one" and project_of(other) == "two"
    assert first.recipe_hash != other.recipe_hash, "recipes that differ only in project must hash differently"
    assert first.recipe_hash == renamed.recipe_hash, "the file name enters the hash only through the project"


@pytest.mark.parametrize("keyed", [False, True], ids=["from-name", "explicit"])
def test_a_saved_resolved_recipe_reloads_with_its_project_and_hash(tmp_path: Path, keyed: bool) -> None:
    data = recipe_data(project="demo-project") if keyed else recipe_data()
    recipe = load(write_recipe(tmp_path / "orig-name.yaml", data))
    saved = tmp_path / "saved" / RESOLVED_RECIPE
    saved.parent.mkdir()
    saved.write_bytes(resolved_yaml(recipe).encode("ascii"))
    back = load(saved)
    assert project_of(back) == ("demo-project" if keyed else "orig-name")
    assert back.recipe_hash == recipe.recipe_hash


# ---------------------------------------------------------------------------
# The runner: the first trial id segment and reruns


def test_the_project_key_names_the_first_trial_id_segment(tmp_path: Path, bench: Path) -> None:
    run_dir = run(write_recipe(tmp_path / "file-name.yaml", recipe_data(project="demo-project")), tmp_path / "runs",
                  bench)
    ids = trial_ids(run_dir)
    assert ids == [f"demo-project/{MOCK_ID}/{SUITE}/omp-cuda/{ITEM}/run01"], ids
    assert not (run_dir / "file-name").exists()


def test_without_a_project_key_the_recipe_name_is_the_first_segment(tmp_path: Path, bench: Path) -> None:
    run_dir = run(write_recipe(tmp_path / "plain-name.yaml", recipe_data()), tmp_path / "runs", bench)
    assert trial_ids(run_dir) == [f"plain-name/{MOCK_ID}/{SUITE}/omp-cuda/{ITEM}/run01"]


def test_an_inherited_project_names_the_trial_ids_of_a_child(tmp_path: Path, bench: Path) -> None:
    write_recipe(tmp_path / "parent-file.yaml", recipe_data(project="parent-project"))
    child = write_recipe(tmp_path / "child-file.yaml", {"extends": "parent-file.yaml", "trials": {"n": 1}})
    run_dir = run(child, tmp_path / "runs", bench)
    assert trial_ids(run_dir) == [f"parent-project/{MOCK_ID}/{SUITE}/omp-cuda/{ITEM}/run01"]


@pytest.mark.parametrize("keyed", [False, True], ids=["from-name", "explicit"])
def test_a_rerun_from_recipe_resolved_yaml_keeps_the_trial_ids(tmp_path: Path, bench: Path, keyed: bool) -> None:
    data = recipe_data(project="demo-project") if keyed else recipe_data()
    first = run(write_recipe(tmp_path / "orig-name.yaml", data), tmp_path / "runs", bench, run_id="first")
    saved = first / RESOLVED_RECIPE
    again = run(saved, tmp_path / "runs", bench, run_id="rerun")
    expected = f"{'demo-project' if keyed else 'orig-name'}/{MOCK_ID}/{SUITE}/omp-cuda/{ITEM}/run01"
    assert trial_ids(first) == [expected]
    assert trial_ids(again) == [expected], "a rerun from recipe.resolved.yaml must keep the trial ids"
    assert (again / RESOLVED_RECIPE).read_bytes().split(b"\n", 2)[2] == saved.read_bytes().split(b"\n", 2)[2]


@pytest.mark.parametrize("project", RUN_TREE_NAMES)
def test_a_project_named_like_a_run_tree_entry_is_refused_before_anything_is_written(
    tmp_path: Path, bench: Path, project: str
) -> None:
    runs_root = tmp_path / "runs"
    with pytest.raises((RecipeError, RunError)) as info:
        run(write_recipe(tmp_path / "ordinary-name.yaml", recipe_data(project=project)), runs_root, bench)
    message = str(info.value)
    assert "unknown key" not in message, f"the loader does not know the project key yet: {message}"
    assert repr(project) in message or f" {project} " in message or f" {project}," in message, message
    assert not runs_root.exists(), "nothing may be written before the refusal"


@pytest.mark.parametrize("project", NOT_ONE_SEGMENT, ids=["slash", "leading-dash", "empty"])
def test_a_project_that_is_not_one_trial_id_segment_is_refused_before_anything_is_written(
    tmp_path: Path, bench: Path, project: str
) -> None:
    runs_root = tmp_path / "runs"
    recipe = write_recipe(tmp_path / "ordinary-name.yaml", recipe_data(project=project))
    with pytest.raises((RecipeError, RunError)) as info:
        run(recipe, runs_root, bench)
    assert "unknown key" not in str(info.value), f"the loader does not know the project key yet: {info.value}"
    assert not runs_root.exists(), "nothing may be written before the refusal"
