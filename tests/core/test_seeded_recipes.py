"""Tests for the runner's side of the seeded set and the recipes under projects/adversary-seeded/ (task P16.8).

Bible: Project Recipes (the bindings block; Notes, executors), Benchmark
Suites (the sources rule; Item tolerance), Oracles (binary_io rules),
Execution Backends (the native row), Toolchain Pins (gcc), Agent Rules 3, 5,
and 6. Plan: plans/p16-adversarial.md, P16.8, and its planning decisions
"Gate subjects (P16.8)" and "Gate reading (P16.G)".

The contract these tests fix:

- lassi.core.runner refuses binary_io's threshold from_baseline when a
  direction's source and target are one language, since the baseline then
  builds one reference and compares it with itself, an agreement that would
  make from_baseline an exact match (lassi/core/stages.py,
  BaselineStage.__call__): a RunError naming the direction and
  oracle.threshold, before any directory exists. The same recipe with a
  two-language direction runs.
- For a tracked suite (lassi.bench.registry: no repo, no commit, every
  language tracked) the runner needs no bench root and no $LASSI_SCRATCH:
  every file the run reads is in the repository (TRACKED_ROOT). A bench
  root given for a tracked suite is a RunError naming the suite and saying
  it is a tracked suite that takes no bench root, before any directory
  exists, so no run silently reads elsewhere.
- The three recipes under projects/adversary-seeded/ (seeded-fuzzer,
  control-fuzzer, seeded-llm) load, bind what the plan gives them (direction
  cpp -> cpp, gcc-native, native, binary_io max_abs 0.001, the adversary
  stage after the oracle stage, the fuzzer or the replay-driven llm
  adversary, no correction), and pass every check of the runner up to the
  first build, with no bench root and no $LASSI_SCRATCH: the first build is
  pair_max's reference with lassi_io.h beside it. Their trials.n matches the
  recording: two completions per item for the replay recipes, one trial per
  item for the mock control.

The runner checks run with the real stages, oracle, adversaries, and replay
and mock backends; the toolchain and executor are SYNTHETIC fakes (the
toolchain stops the run at its first build, and the executor runs nothing),
and the suites of the first two parts are SYNTHETIC
(tests/core/adversary_support.py). No compiler runs and no program runs. No
value in this module is a measurement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from adversary_support import (
    CONTRACT_HI,
    CONTRACT_LO,
    HELD_OUT,
    ITEM_SEED,
    MODEL_ID,
    PROJECT,
    REFERENCE_ROLE,
    SPLIT,
    SUPPORT,
    SUPPORT_TEXT,
    TOLERANCE,
    World,
    clean_environment,
    make_registry,
    observer,
    program,
    recipe_data,
    refused,
    run,
    write_recipe,
    write_suite,
)

from lassi import agents  # noqa: F401  (registers the Agents the recipes bind)
from lassi.bench import registry as bench_registry
from lassi.core import runner as runner_module
from lassi.core.interfaces import BuildResult, Limits, RunResult
from lassi.core.recipe import load_recipe
from lassi.core.record import make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.llm import MockBackend, ReplayBackend

REPO = Path(__file__).resolve().parents[2]
NO_ADVERSARY_STAGES = ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
FROM_BASELINE = {"kind": "binary_io", "metric": "max_abs", "threshold": "from_baseline"}
# The SYNTHETIC tracked suite of the bench-root tests: one cpp item whose files lie under TRACKED_ROOT.
TRACKED_SUITE = "adv-tracked"
TRACKED_ITEM = "scale"
# The seeded set's suite, recipes, and recordings.
SEEDED_SUITE = "adv-seeded-v0"
SEEDED_ITEMS = ("pair_max", "piecewise_gate", "piecewise_ops", "prefix_sum", "scale_i64", "signed_root")
PROJECTS = REPO / "projects" / "adversary-seeded"
RECIPES = ("seeded-fuzzer", "control-fuzzer", "seeded-llm")
RECORDING = "tests/fixtures/adversary/seeded/seeded.json"
BREAK_CONTRACT = "tests/fixtures/adversary/seeded/break-contract.json"
HEADER = REPO / "assets" / "harness" / "c" / "lassi_io.h"
FUZZER = {"kind": "fuzzer", "budget": {"inputs": 32}, "seed": 1}
LLM = {
    "kind": "llm",
    "budget": {"inputs": 8, "tokens": 200000, "steps": 4},
    "backend": "replay",
    "model": "adv-seeded-break-contract",
    "sampling": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 2048},
    "prompts": "adversary-v0",
    "recording": BREAK_CONTRACT,
}
# What the plan gives each recipe: its model section (without the loader's sha256), trials.n, and adversary.
PLANNED: dict[str, tuple[dict[str, Any], int, dict[str, Any]]] = {
    "seeded-fuzzer": ({"backend": "replay", "id": "adv-seeded-replay", "recording": RECORDING}, 2, FUZZER),
    "control-fuzzer": ({"backend": "mock", "id": "mock-reference"}, 1, FUZZER),
    "seeded-llm": ({"backend": "replay", "id": "adv-seeded-replay", "recording": RECORDING}, 2, LLM),
}
ADVERSARY_STAGES = ["baseline", "generate", "compile_loop", "run_loop", "oracle", "adversary"]
SHA256_KEY = "recording_sha256"


@pytest.fixture(autouse=True)
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no gate variable, a TMPDIR under the test's directory, and its cwd there."""
    clean_environment(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# from_baseline on a one-language direction


@pytest.mark.parametrize(
    ("directions", "toolchain"),
    [
        ([{"source": "cpp", "target": "cpp"}], {"cpp": "fake-cpp"}),
        ([{"source": "omp", "target": "cpp"}, {"source": "cpp", "target": "cpp"}], None),
    ],
    ids=["cpp-cpp", "omp-cpp-then-cpp-cpp"],
)
def test_from_baseline_is_refused_on_a_one_language_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directions: list[dict[str, str]], toolchain: Any
) -> None:
    bench = write_suite(tmp_path, monkeypatch)
    changes: dict[str, Any] = {"stages": NO_ADVERSARY_STAGES, "adversary": None, "oracle": dict(FROM_BASELINE)}
    changes["directions"] = directions
    if toolchain is not None:
        changes["toolchain"] = toolchain
    message = refused(tmp_path, bench, World(), recipe_data(**changes))
    assert "oracle.threshold" in message, message
    assert "cpp-cpp" in message or "cpp -> cpp" in message, f"the refusal does not name the direction: {message}"


def test_from_baseline_on_a_two_language_direction_still_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = write_suite(tmp_path, monkeypatch)
    data = recipe_data(stages=NO_ADVERSARY_STAGES, adversary=None, oracle=dict(FROM_BASELINE))
    outcome = run(tmp_path, bench, World(), data)
    assert outcome.trial.final.end_reason is None, outcome.trial.final


# ---------------------------------------------------------------------------
# A tracked suite needs no bench root and refuses one


def write_tracked_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the SYNTHETIC tracked suite and its files; point the runner and the registry at them; return the root."""
    root = tmp_path / "repo"
    base = f"assets/{TRACKED_SUITE}"
    for path, text in {
        f"{base}/{TRACKED_ITEM}/cpp/main.cpp": program(REFERENCE_ROLE, "double"),
        f"{base}/{SUPPORT}": SUPPORT_TEXT,
    }.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(text.encode("ascii"))
    monkeypatch.setattr(bench_registry, "TRACKED_ROOT", root)
    item = {
        "split": SPLIT,
        "inputs": [dict(HELD_OUT)],
        "outputs": ["y"],
        "seed": ITEM_SEED,
        "contract": [{"name": "x", "dtype": "f32", "lo": CONTRACT_LO, "hi": CONTRACT_HI}],
        "tolerance": dict(TOLERANCE),
        "languages": {
            "cpp": {
                "dir": f"{base}/{TRACKED_ITEM}/cpp",
                "tracked": True,
                "files": ["main.cpp"],
                "support": {SUPPORT: {"tracked": f"{base}/{SUPPORT}"}},
            }
        },
    }
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    data = {"suite": TRACKED_SUITE, "items": {TRACKED_ITEM: item}}
    (manifests / f"{TRACKED_SUITE}.yaml").write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    monkeypatch.setattr(runner_module, "BENCH_DIR", manifests)
    return root


def tracked_recipe() -> dict[str, Any]:
    """Return the test recipe over the tracked suite, direction cpp -> cpp."""
    return recipe_data(
        bench={"suite": TRACKED_SUITE, "split": SPLIT},
        directions=[{"source": "cpp", "target": "cpp"}],
        toolchain={"cpp": "fake-cpp"},
    )


def tracked_options(tmp_path: Path, world: World, bench_root: Path | None) -> RunOptions:
    """Return run options over the world's registry with `bench_root` (None: no bench root)."""
    return RunOptions(
        runs_root=tmp_path / "runs-root",
        run_id="tracked",
        bench_root=bench_root,
        registry=make_registry(world),
        observer=observer(world),
    )


def test_a_tracked_suite_runs_with_no_bench_root_and_no_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_tracked_suite(tmp_path, monkeypatch)
    world = World(candidate="double")
    path = write_recipe(tmp_path, tracked_recipe(), "recipe-tracked")
    run_dir = run_recipe(path, tracked_options(tmp_path, world, None))
    trial_id = make_trial_id(PROJECT, MODEL_ID, TRACKED_SUITE, "cpp-cpp", TRACKED_ITEM, 1)
    trial = read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))
    assert trial.reference_run is not None and trial.reference_run.exit_code == 0, "the tracked reference ran"
    assert trial.final.end_reason is None, trial.final


def test_a_bench_root_given_for_a_tracked_suite_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_tracked_suite(tmp_path, monkeypatch)
    world = World(candidate="double")
    elsewhere = tmp_path / "bench"
    elsewhere.mkdir()
    path = write_recipe(tmp_path, tracked_recipe(), "recipe-tracked-root")
    with pytest.raises(RunError) as caught:
        run_recipe(path, tracked_options(tmp_path, world, elsewhere))
    message = str(caught.value)
    assert TRACKED_SUITE in message and "tracked suite" in message and "bench root" in message, message
    assert not (tmp_path / "runs-root").exists(), "a refused run creates no directory"
    assert not any(kind == "run" for kind, _ in world.events), "a refused run runs nothing"


# ---------------------------------------------------------------------------
# The recipes under projects/adversary-seeded/


class FirstBuild(Exception):
    """Raised by the SYNTHETIC toolchain at the run's first build, carrying what it was asked to build."""

    def __init__(self, files: Mapping[str, str], harness: Mapping[str, str]) -> None:
        """Keep the files and the harness files of the build."""
        super().__init__("SYNTHETIC: the first build")
        self.files = dict(files)
        self.harness = dict(harness)


class FirstBuildToolchain:
    """A SYNTHETIC stand-in for gcc-native, built as factory() (no PIN); its first build stops the run."""

    name = "gcc-native"
    capabilities = frozenset({"diagnostics", "emits_warnings"})

    def build(
        self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
    ) -> BuildResult:
        """Raise FirstBuild with what the build was given; nothing is compiled."""
        raise FirstBuild(files, harness or {})


class InertNative:
    """A SYNTHETIC stand-in for the native executor: it declares runs_code and sandboxed and runs nothing."""

    name = "native"
    capabilities = frozenset({"runs_code", "sandboxed"})

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Fail the test: no program may run before the first build."""
        raise AssertionError("a program ran before the first build")


def seeded_registry() -> Registry:
    """Return a Registry of the real components the recipes bind, with the SYNTHETIC toolchain and executor."""
    found = Registry()
    found.register("LLMBackend", "replay", ReplayBackend)
    found.register("LLMBackend", "mock", MockBackend)
    found.register("Toolchain", "gcc-native", FirstBuildToolchain)
    found.register("Executor", "native", InertNative)
    found.register("Oracle", "binary_io", DEFAULT_REGISTRY.get("Oracle", "binary_io").factory)
    for name in ("fuzzer", "llm"):
        found.register("Agent", name, DEFAULT_REGISTRY.get("Agent", name).factory)
    for name in ADVERSARY_STAGES:
        found.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    return found


def recipe_path(name: str) -> Path:
    """Return the path of a recipe under projects/adversary-seeded/."""
    return PROJECTS / f"{name}.yaml"


def seeded_completions() -> list[Any]:
    """Return the completions of the seeded recording."""
    return json.loads((REPO / RECORDING).read_bytes().decode("ascii"))["completions"]


@pytest.mark.parametrize("name", RECIPES)
def test_each_recipe_binds_what_the_plan_gives_it(name: str) -> None:
    data = load_recipe(recipe_path(name)).data
    model, trials, adversary = PLANNED[name]
    assert data["faithful"] is False
    assert {key: value for key, value in data["model"].items() if key != SHA256_KEY} == model
    assert data["trials"]["n"] == trials
    assert {key: value for key, value in data["adversary"].items() if key != SHA256_KEY} == adversary
    assert (data["bench"]["suite"], data["bench"]["split"]) == (SEEDED_SUITE, "unassigned")
    assert data["directions"] == [{"source": "cpp", "target": "cpp"}]
    assert data["prompts"] == "p0-smoke"
    assert data["toolchain"] == {"cpp": "gcc-native"}
    assert data["executor"]["kind"] == "native"
    assert data["stages"] == ADVERSARY_STAGES
    assert data["loop"]["max_corrections"] == 0, "each trial takes exactly one completion"
    assert data["llm"]["sampling"]["max_tokens"] == 4096
    oracle = data["oracle"]
    assert (oracle["kind"], oracle["metric"], oracle["threshold"]) == ("binary_io", "max_abs", 0.001)


@pytest.mark.parametrize("name", RECIPES)
def test_each_recipe_passes_the_runners_checks_up_to_the_first_build(tmp_path: Path, name: str) -> None:
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id=f"seeded-{name}", registry=seeded_registry())
    with pytest.raises(FirstBuild) as caught:
        run_recipe(recipe_path(name), options)
    reference = REPO / "assets" / "bench" / SEEDED_SUITE / "pair_max" / "cpp" / "pair_max.cpp"
    assert caught.value.files == {"pair_max.cpp": reference.read_bytes().decode("ascii")}
    assert caught.value.harness == {"lassi_io.h": HEADER.read_bytes().decode("ascii")}


@pytest.mark.parametrize("name", ["seeded-fuzzer", "seeded-llm"])
def test_trials_n_matches_the_recording(name: str) -> None:
    data = load_recipe(recipe_path(name)).data
    assert data["model"]["recording"] == RECORDING
    count = data["trials"]["n"] * len(SEEDED_ITEMS) * len(data["directions"])
    assert len(seeded_completions()) == count, "one completion per trial"
    assert data["trials"]["n"] == 2, "two completions per item"


def test_the_control_takes_one_trial_per_item_from_the_mock() -> None:
    data = load_recipe(recipe_path("control-fuzzer")).data
    assert data["model"]["backend"] == "mock" and "recording" not in data["model"]
    assert data["trials"]["n"] == 1
