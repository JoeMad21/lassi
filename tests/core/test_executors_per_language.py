"""Tests for executors bound per language and the device each run records (task P4.5).

Bible: Project Recipes (the recipe blocks and Notes: validation at load,
extends and kind sections), Component Interfaces (Executor; capability
rule), Execution Backends (native and none rows), Sandbox (model-generated
code runs only in the sandbox), Result Record (provenance; Storage: the run
manifest is authoritative and each trial carries a copy), Readability
Standards (Run row), Agent Rules 1, 6, and 10. Planning decision "Executors
per language (P4.5)" in plans/p4-ttsim.md: a baseline runs the C++
reference natively and the TT reference on ttsim.

The contract these tests fix:

- `executor` takes a second form beside today's kind section: a mapping
  from language to executor, like `toolchain`. Each entry is a registered
  Executor name (`cuda: exec-cuda`, no config) or a kind section with that
  executor's config (`omp: {kind: exec-omp, host: box1}`). The loader binds
  one Executor per language, its binding's recipe path naming the
  language (executor.<language>), checks each entry's config keys and
  registration, and refuses an entry that is neither form, naming
  executor.<language>. A child that switches between the two forms
  replaces the inherited executor section, as a child naming another kind
  does. The single form `executor: {kind: ...}` is unchanged, and so are
  the resolved recipe and hash of every recipe that uses it
  (test_single_executor_golden.py).
- Before any directory exists: a direction's target language with no
  executor is refused, and so is its source language when a listed stage
  builds and runs the source reference (baseline under baseline_both);
  each message names the key to set, executor.<language>. The sandboxed
  check runs for each bound executor (Agent Rule 6), and the sandbox
  limits are checked when any bound executor runs programs.
- In a run, every attempt runs on its direction's target-language
  executor, and the baseline runs each reference on its own language's
  executor; a compile-only executor for the source language runs nothing.
- Every executor names its device (device(), test_executor_devices.py).
  provenance.json of a per-language run records `executor` as the mapping
  language -> executor name and `devices` as language -> device, run.md
  shows each language's device, and each trial's provenance.device is its
  target language's device. A single-executor run keeps the `device`
  field: the bound executor's device (the native executor now names the
  host CPU where it recorded null), and each trial carries it.

Every component here is a fake in a test Registry, beside the real stages
(baseline, generate, compile_loop, run_loop) and the real NativeExecutor,
which these runs never ask to run anything. The fake executors start no
process and answer with SYNTHETIC results; the fake toolchains write a
PLACEHOLDER artifact and compile nothing; the scripted backend answers with
SYNTHETIC replies; bench sources are small SYNTHETIC files. Device names are
SYNTHETIC. No value in this module is a measurement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import Direction, load_suite
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.recipe import RecipeError, executor_languages, load_recipe, resolved_yaml
from lassi.core.record import Trial, make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import BaselineStage, RunContext
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.executors import NativeExecutor
from lassi.executors import native as native_module

REPO = Path(__file__).resolve().parents[2]
PROJECTS = REPO / "projects"
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
MODEL_ID = "scripted-fixture"
PROMPTS = "p0-smoke"
OMP_TO_CUDA = Direction("omp", "cuda")
CUDA_TO_OMP = Direction("cuda", "omp")
DIRECTIONS = [OMP_TO_CUDA, CUDA_TO_OMP]
DIRECTION_IDS = ["omp-cuda", "cuda-omp"]
TOOLCHAINS = {"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"}
RUN_STAGES = ["baseline", "generate", "compile_loop", "run_loop"]
COMPILE_STAGES = ["generate", "compile_loop"]
RAN_CLEAN = "S5"

# The per-language executors: one fake per language, each running programs in the sandbox.
LABEL = {"cuda": "exec-cuda", "omp": "exec-omp"}
PER_LANGUAGE = dict(LABEL)
SANDBOXED_RUNNER = frozenset({"runs_code", "sandboxed"})
# SYNTHETIC device names; none holds a language name, so a run.md line that shows one names its language itself.
DEVICE = {
    "exec-cuda": "SYNTHETIC device A",
    "exec-omp": "SYNTHETIC device B",
    "exec-open": "SYNTHETIC device C",
    "none": "none (compile only)",
}
HOST_CPU = "host CPU (native)"
SYNTHETIC_MODEL = "SYNTHETIC CPU model 9000"

# SYNTHETIC bench sources and replies; the fake toolchain refuses any file holding an `#error` line.
OMP_SOURCE = '#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  std::printf("done\\n");\n}\n'
CUDA_SOURCE = "#include <cstdio>\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n"
SOURCES = {"omp": OMP_SOURCE, "cuda": CUDA_SOURCE}
GOOD_SOURCE = "int main() { return 0; }\n"
GOOD_REPLIES = {
    "cuda": render_file_blocks({"main.cu": GOOD_SOURCE}),
    "omp": render_file_blocks({"main.cpp": GOOD_SOURCE}),
}
# The wall time every fake run reports: a fixture value, not a measurement.
SYNTHETIC_WALL_S = 1.25


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate and compile variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


# ---------------------------------------------------------------------------
# Fake components, all writing to one Log so the order of builds, runs, and model calls shows


@dataclass
class Build:
    """One build a fake toolchain was asked for."""

    toolchain: str
    workdir: Path
    artifact: Path | None


@dataclass
class Run:
    """One run a fake executor was asked for."""

    executor: str
    artifact: Path


@dataclass
class Log:
    """What the fake components saw: events in order, builds, runs, model requests, and the replies left."""

    replies: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    builds: list[Build] = field(default_factory=list)
    runs: list[Run] = field(default_factory=list)
    requests: list[list[Message]] = field(default_factory=list)


def fake_toolchain(registered_as: str, log: Log) -> type:
    """Return a Toolchain class without PIN that writes the files and a PLACEHOLDER artifact; it compiles nothing."""

    class FakeToolchain:
        """Writes every file under the workdir and reports a build, or one error for an `#error` line."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Record the build, write every file, and return the artifact or one compile error."""
            workdir = Path(workdir)
            for path, text in [*files.items(), *(harness or {}).items()]:
                target = workdir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            failed = any("#error" in text for text in files.values())
            artifact = None if failed else workdir / "main"
            if artifact is not None:
                artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            log.events.append(f"build {registered_as}")
            log.builds.append(Build(registered_as, workdir, artifact))
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


def scripted_executor(label: str, log: Log, capabilities: frozenset[str]) -> type:
    """Return an Executor class registered as `label` that names its SYNTHETIC device and runs nothing."""

    class ScriptedExecutor:
        """Records each run and returns a clean SYNTHETIC RunResult whose stdout names this executor."""

        name = label
        config_keys = frozenset({"host"})

        def __init__(self, *, host: str | None = None) -> None:
            """Keep the recipe config, as a kind section passes it."""
            self.host = host

        def device(self) -> str:
            """Return this fake's SYNTHETIC device name."""
            return DEVICE[label]

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Record the run and return a clean SYNTHETIC result."""
            log.events.append(f"run {label}")
            log.runs.append(Run(label, Path(artifact)))
            stdout = f"SYNTHETIC stdout of a run on {label}\n"
            return RunResult(exit_code=0, hang=False, stdout=stdout, stderr="", wall_s=SYNTHETIC_WALL_S)

    ScriptedExecutor.capabilities = capabilities
    return ScriptedExecutor


class CompileOnlyExecutor:
    """A compile-only Executor registered as "none"; being asked to run anything fails the test."""

    name = "none"
    capabilities = frozenset({"compile_only"})

    def device(self) -> str:
        """Return the compile-only device name runs record."""
        return DEVICE["none"]

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Fail the test: a compile-only executor is never asked to run a program."""
        raise AssertionError(f"a compile-only executor was asked to run {artifact}")


def scripted_backend(log: Log) -> type:
    """Return an LLMBackend class, registered as "scripted", that answers from `log.replies` in order."""

    class ScriptedBackend:
        """Records each request and answers with the next scripted reply."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id, as every backend does."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return the next scripted reply."""
            log.events.append("request")
            log.requests.append(list(messages))
            assert log.replies, "the backend was asked for more replies than the script holds"
            return Completion(text=log.replies.pop(0), prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def make_registry(log: Log) -> Registry:
    """Return a test Registry: the scripted backend, fake toolchains and executors, native, and the real stages."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(log))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, fake_toolchain(name, log))
    for label in LABEL.values():
        registry.register("Executor", label, scripted_executor(label, log, SANDBOXED_RUNNER))
    registry.register("Executor", "exec-open", scripted_executor("exec-open", log, frozenset({"runs_code"})))
    registry.register("Executor", "none", CompileOnlyExecutor)
    registry.register("Executor", "native", NativeExecutor)
    for name in RUN_STAGES:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    return registry


# ---------------------------------------------------------------------------
# Recipes, bench sources, and runs


def recipe_data(
    directions: Sequence[Direction],
    executor: Mapping[str, Any],
    *,
    stages: Sequence[str] = RUN_STAGES,
    both: bool = True,
    **changes: Any,
) -> dict[str, Any]:
    """Return a template-set recipe for the layout item over `directions` with `executor` as given.

    `both` False turns fixes.baseline_both off (the baseline builds only the
    target reference); `changes` add or replace top-level keys.
    """
    data: dict[str, Any] = {
        "extends": "base",
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": direction.source, "target": direction.target} for direction in directions],
        "prompts": PROMPTS,
        "toolchain": dict(TOOLCHAINS),
        "stages": list(stages),
        "executor": json.loads(json.dumps(executor)),
        "trials": {"n": 1},
    }
    if not both:
        data["fixes"] = {"baseline_both": False}
    data.update(changes)
    return data


def write_recipe(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe `<directory>/<name>.yaml` and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def write_bench(root: Path) -> Path:
    """Write the layout item's SYNTHETIC source per language where the suite manifest lays it out; return `root`."""
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, text in SOURCES.items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    for relative in spec.support.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"// SYNTHETIC support file\n")
    return root


def options(tmp_path: Path, log: Log) -> RunOptions:
    """Return the run options: a runs root under tmp_path, a fixed run id, the synthetic bench, the test registry."""
    bench = tmp_path / "bench"
    if not bench.exists():
        write_bench(bench)
    registry = make_registry(log)
    return RunOptions(runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench, registry=registry)


@dataclass
class Outcome:
    """One finished run: its directory, the recipe name, provenance.json, run.md, and the log."""

    run_dir: Path
    name: str
    provenance: dict[str, Any]
    run_md: str
    log: Log

    def trial(self, direction: Direction) -> Trial:
        """Return the one trial of `direction`, read back from the run tree."""
        trial_id = make_trial_id(self.name, MODEL_ID, SUITE, f"{direction.source}-{direction.target}", ITEM, 1)
        return read_trial(trial_dir(self.run_dir, trial_id), TextStore(self.run_dir))


def run_all(tmp_path: Path, name: str, data: Mapping[str, Any], log: Log) -> Outcome:
    """Run the recipe with the fakes answering from `log`; return the outcome."""
    run_dir = run_recipe(write_recipe(tmp_path, name, data), options(tmp_path, log))
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="ascii"))
    return Outcome(run_dir, name, provenance, (run_dir / "run.md").read_text(encoding="ascii"), log)


def load(tmp_path: Path, name: str, data: Mapping[str, Any], log: Log | None = None) -> Any:
    """Load the recipe with the test registry; construct nothing."""
    path = write_recipe(tmp_path, name, data)
    return load_recipe(path, roots=(tmp_path, PROJECTS), registry=make_registry(log or Log()))


def refused(tmp_path: Path, name: str, data: Mapping[str, Any], log: Log) -> str:
    """Run a recipe that must be refused before any directory exists; return the refusal's message."""
    with pytest.raises((RecipeError, RunError)) as caught:
        run_recipe(write_recipe(tmp_path, name, data), options(tmp_path, log))
    assert not (tmp_path / "runs-root").exists(), "a refusal comes before any directory is created"
    assert log.events == [], "a refusal comes before anything is built, run, or asked"
    return str(caught.value)


def executor_bindings(recipe: Any) -> list[Any]:
    """Return the recipe's Executor bindings."""
    return [binding for binding in recipe.bindings if binding.interface == "Executor"]


def builds_before_first_request(log: Log) -> list[Build]:
    """Return the builds made before the first model request, which are the baseline's."""
    assert "request" in log.events, "the trial never asked the model"
    count = sum(1 for event in log.events[: log.events.index("request")] if event.startswith("build "))
    return log.builds[:count]


# ---------------------------------------------------------------------------
# The recipe form


def test_a_recipe_binds_one_executor_per_language_by_name(tmp_path: Path) -> None:
    recipe = load(tmp_path, "per-language", recipe_data(DIRECTIONS, PER_LANGUAGE))
    bindings = executor_bindings(recipe)
    assert sorted((binding.name, dict(binding.config)) for binding in bindings) == [
        ("exec-cuda", {}),
        ("exec-omp", {}),
    ]
    for binding in bindings:
        language = next(language for language, label in LABEL.items() if label == binding.name)
        assert f"executor.{language}" in binding.where, "a binding's recipe path names its language"


def test_a_language_entry_may_be_a_kind_section_with_its_executors_config(tmp_path: Path) -> None:
    executor = {"cuda": "exec-cuda", "omp": {"kind": "exec-omp", "host": "box1"}}
    recipe = load(tmp_path, "kind-entry", recipe_data(DIRECTIONS, executor))
    configs = {binding.name: dict(binding.config) for binding in executor_bindings(recipe)}
    assert configs == {"exec-cuda": {}, "exec-omp": {"host": "box1"}}


def test_a_config_key_the_languages_executor_does_not_accept_is_refused(tmp_path: Path) -> None:
    executor = {"cuda": "exec-cuda", "omp": {"kind": "exec-omp", "hots": "box1"}}
    with pytest.raises(RecipeError) as caught:
        load(tmp_path, "bad-config", recipe_data(DIRECTIONS, executor))
    assert "executor.omp" in str(caught.value) and "hots" in str(caught.value)


def test_an_unregistered_executor_for_a_language_is_refused(tmp_path: Path) -> None:
    executor = {"cuda": "exec-cuda", "omp": "no-such-executor"}
    with pytest.raises(RecipeError) as caught:
        load(tmp_path, "unregistered", recipe_data(DIRECTIONS, executor))
    assert "executor.omp" in str(caught.value) and "no-such-executor" in str(caught.value)


@pytest.mark.parametrize("entry", [5, ["exec-omp"], {"host": "box1"}], ids=["int", "list", "mapping-without-kind"])
def test_a_language_entry_that_is_neither_a_name_nor_a_kind_section_is_refused(tmp_path: Path, entry: Any) -> None:
    with pytest.raises(RecipeError) as caught:
        load(tmp_path, "bad-entry", recipe_data(DIRECTIONS, {"cuda": "exec-cuda", "omp": entry}))
    assert "executor.omp" in str(caught.value)


def test_the_single_form_still_binds_one_executor_at_executor_kind(tmp_path: Path) -> None:
    recipe = load(tmp_path, "single", recipe_data(DIRECTIONS, {"kind": "exec-cuda", "host": "box1"}))
    assert [(binding.name, binding.where, dict(binding.config)) for binding in executor_bindings(recipe)] == [
        ("exec-cuda", "executor.kind", {"host": "box1"})
    ]
    assert recipe.data["executor"] == {"kind": "exec-cuda", "host": "box1"}


@pytest.mark.parametrize(
    ("parent", "child"),
    [({"kind": "none"}, PER_LANGUAGE), (PER_LANGUAGE, {"kind": "none"})],
    ids=["single-to-per-language", "per-language-to-single"],
)
def test_a_child_that_switches_forms_replaces_the_inherited_executor_section(
    tmp_path: Path, parent: Mapping[str, Any], child: Mapping[str, Any]
) -> None:
    write_recipe(tmp_path, "form-parent", recipe_data(DIRECTIONS, parent))
    recipe = load(tmp_path, "form-child", {"extends": "form-parent", "executor": dict(child)})
    assert recipe.data["executor"] == dict(child), "nothing of the other form is inherited"


def test_a_per_language_recipe_reloads_from_its_resolved_file_to_the_same_hash(tmp_path: Path) -> None:
    executor = {"cuda": "exec-cuda", "omp": {"kind": "exec-omp", "host": "box1"}}
    recipe = load(tmp_path, "resolved-source", recipe_data(DIRECTIONS, executor))
    saved = tmp_path / "saved"
    saved.mkdir()
    path = saved / "resolved-source.resolved.yaml"
    path.write_bytes(resolved_yaml(recipe).encode("ascii"))
    again = load_recipe(path, roots=(saved, PROJECTS), registry=make_registry(Log()))
    assert (again.data, again.recipe_hash) == (recipe.data, recipe.recipe_hash)


# ---------------------------------------------------------------------------
# Refusals before any directory exists


def test_an_executor_missing_for_a_target_language_is_refused_before_any_directory(tmp_path: Path) -> None:
    log = Log()
    message = refused(tmp_path, "no-target-executor", recipe_data([OMP_TO_CUDA], {"omp": "exec-omp"}), log)
    assert "executor.cuda" in message, "the refusal names the key to set"


def test_an_executor_missing_for_a_source_the_baseline_runs_is_refused_before_any_directory(tmp_path: Path) -> None:
    log = Log()
    message = refused(tmp_path, "no-source-executor", recipe_data([OMP_TO_CUDA], {"cuda": "exec-cuda"}), log)
    assert "executor.omp" in message, "the refusal names the key to set"


@pytest.mark.parametrize("direction", DIRECTIONS, ids=DIRECTION_IDS)
def test_the_sandboxed_check_runs_for_each_executor_before_any_directory(tmp_path: Path, direction: Direction) -> None:
    executor = {direction.target: "exec-open", direction.source: LABEL[direction.source]}
    log = Log()
    with pytest.raises(RunError) as caught:
        run_recipe(write_recipe(tmp_path, "unsandboxed", recipe_data([direction], executor)), options(tmp_path, log))
    assert "exec-open" in str(caught.value) and "sandboxed" in str(caught.value)
    assert not (tmp_path / "runs-root").exists() and log.events == []


def test_the_sandbox_limits_are_checked_when_any_bound_executor_runs_programs(tmp_path: Path) -> None:
    data = recipe_data([CUDA_TO_OMP], {"cuda": "none", "omp": "exec-omp"}, sandbox={"wall_s": "soon"})
    log = Log()
    with pytest.raises(RunError) as caught:
        run_recipe(write_recipe(tmp_path, "bad-limits", data), options(tmp_path, log))
    assert "sandbox.wall_s" in str(caught.value)
    assert not (tmp_path / "runs-root").exists() and log.events == []


# ---------------------------------------------------------------------------
# Which executor runs what


@pytest.mark.parametrize("direction", DIRECTIONS, ids=DIRECTION_IDS)
def test_attempts_run_on_the_target_executor_and_each_reference_on_its_languages(
    tmp_path: Path, direction: Direction
) -> None:
    log = Log(replies=[GOOD_REPLIES[direction.target]])
    outcome = run_all(tmp_path, "routing", recipe_data([direction], PER_LANGUAGE), log)
    baseline = builds_before_first_request(log)
    (target_reference,) = [build for build in baseline if build.toolchain == TOOLCHAINS[direction.target]]
    (source_reference,) = [build for build in baseline if build.toolchain == TOOLCHAINS[direction.source]]
    (attempt,) = log.builds[len(baseline) :]
    assert [(run.executor, run.artifact) for run in log.runs] == [
        (LABEL[direction.target], target_reference.artifact),
        (LABEL[direction.source], source_reference.artifact),
        (LABEL[direction.target], attempt.artifact),
    ], "each reference runs on its own language's executor, and the attempt on the target's"
    trial = outcome.trial(direction)
    assert trial.final.end_reason is None
    assert [item.stage_reached for item in trial.attempts] == [RAN_CLEAN]
    reference = trial.reference_run
    assert reference.stdout_ref is not None
    expected = f"SYNTHETIC stdout of a run on {LABEL[direction.target]}\n"
    assert TextStore(outcome.run_dir).get(reference.stdout_ref) == expected, "the target reference ran on its executor"


def test_a_compile_only_executor_for_the_source_language_runs_nothing(tmp_path: Path) -> None:
    log = Log(replies=[GOOD_REPLIES["omp"]])
    outcome = run_all(tmp_path, "no-gpu", recipe_data([CUDA_TO_OMP], {"cuda": "none", "omp": "exec-omp"}), log)
    baseline = builds_before_first_request(log)
    assert sorted(build.toolchain for build in baseline) == sorted(TOOLCHAINS.values()), "both references build"
    (target_reference,) = [build for build in baseline if build.toolchain == TOOLCHAINS["omp"]]
    (attempt,) = log.builds[len(baseline) :]
    assert [(run.executor, run.artifact) for run in log.runs] == [
        ("exec-omp", target_reference.artifact),
        ("exec-omp", attempt.artifact),
    ], "the compile-only source executor runs nothing"
    trial = outcome.trial(CUDA_TO_OMP)
    assert trial.final.end_reason is None
    assert [item.stage_reached for item in trial.attempts] == [RAN_CLEAN]


def test_the_single_form_runs_everything_on_its_one_executor(tmp_path: Path) -> None:
    log = Log(replies=[GOOD_REPLIES["cuda"]])
    run_all(tmp_path, "single-routing", recipe_data([OMP_TO_CUDA], {"kind": "exec-cuda"}), log)
    assert [run.executor for run in log.runs] == ["exec-cuda"] * 3, "two references and the attempt"


# ---------------------------------------------------------------------------
# The device each run records


def test_a_per_language_run_records_each_languages_executor_and_device(tmp_path: Path) -> None:
    log = Log(replies=[GOOD_REPLIES[direction.target] for direction in DIRECTIONS])
    outcome = run_all(tmp_path, "devices", recipe_data(DIRECTIONS, PER_LANGUAGE), log)
    assert outcome.provenance["executor"] == PER_LANGUAGE
    assert outcome.provenance["devices"] == {language: DEVICE[label] for language, label in LABEL.items()}


def test_each_trial_carries_its_target_languages_device(tmp_path: Path) -> None:
    log = Log(replies=[GOOD_REPLIES[direction.target] for direction in DIRECTIONS])
    outcome = run_all(tmp_path, "trial-devices", recipe_data(DIRECTIONS, PER_LANGUAGE), log)
    for direction in DIRECTIONS:
        trial = outcome.trial(direction)
        assert trial.provenance.device == DEVICE[LABEL[direction.target]], direction
        assert trial.provenance.device == outcome.provenance["devices"][direction.target]


def test_run_md_shows_the_device_of_each_language(tmp_path: Path) -> None:
    log = Log(replies=[GOOD_REPLIES[direction.target] for direction in DIRECTIONS])
    outcome = run_all(tmp_path, "run-md-devices", recipe_data(DIRECTIONS, PER_LANGUAGE), log)
    lines = outcome.run_md.splitlines()
    for language, label in LABEL.items():
        assert any(language in line and DEVICE[label] in line for line in lines), (
            f"run.md shows {language} with its device {DEVICE[label]!r}"
        )


def test_a_single_executor_run_keeps_the_device_field_with_its_executors_device(tmp_path: Path) -> None:
    log = Log(replies=[GOOD_REPLIES["cuda"]])
    outcome = run_all(tmp_path, "single-device", recipe_data([OMP_TO_CUDA], {"kind": "exec-cuda"}), log)
    assert outcome.provenance["executor"] == "exec-cuda"
    assert outcome.provenance["device"] == DEVICE["exec-cuda"], "an executor that names its device is recorded"
    assert outcome.trial(OMP_TO_CUDA).provenance.device == DEVICE["exec-cuda"]
    assert any(line.startswith("| Device |") and DEVICE["exec-cuda"] in line for line in outcome.run_md.splitlines())


def test_a_native_run_names_the_host_cpu_where_it_recorded_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_bytes(f"processor\t: 0\nmodel name\t: {SYNTHETIC_MODEL}\n\n".encode("ascii"))
    monkeypatch.setattr(native_module, "CPUINFO", cpuinfo, raising=False)
    log = Log(replies=[GOOD_REPLIES["cuda"]])
    data = recipe_data([OMP_TO_CUDA], {"kind": "native"}, stages=COMPILE_STAGES)
    outcome = run_all(tmp_path, "native-device", data, log)
    device = f"{HOST_CPU}: {SYNTHETIC_MODEL}"
    assert outcome.provenance["device"] == device, "PHASE-NOTES P2: the native executor names the host CPU"
    assert outcome.trial(OMP_TO_CUDA).provenance.device == device
    assert any(line.startswith("| Device |") and device in line for line in outcome.run_md.splitlines())


def test_a_compile_only_run_keeps_its_device(tmp_path: Path) -> None:
    log = Log(replies=[GOOD_REPLIES["cuda"]])
    data = recipe_data([OMP_TO_CUDA], {"kind": "none"}, stages=COMPILE_STAGES)
    outcome = run_all(tmp_path, "none-device", data, log)
    assert outcome.provenance["device"] == DEVICE["none"]
    assert outcome.trial(OMP_TO_CUDA).provenance.device == DEVICE["none"]


# ---------------------------------------------------------------------------
# Merge rules of the per-language form, a baseline without the source reference, and device() checks
# (commit audit of P4.5: each claim here was covered only by the auditor's probes)


@pytest.mark.parametrize(
    ("parent", "child", "merged"),
    [
        pytest.param(
            {"cuda": "exec-cuda", "omp": "exec-omp"},
            {"omp": {"kind": "exec-omp", "host": "box1"}},
            {"cuda": "exec-cuda", "omp": {"kind": "exec-omp", "host": "box1"}},
            id="entries-merge-per-language",
        ),
        pytest.param(
            {"cuda": "exec-cuda", "omp": {"kind": "exec-omp", "host": "box1"}},
            {"omp": {"kind": "exec-cuda"}},
            {"cuda": "exec-cuda", "omp": {"kind": "exec-cuda"}},
            id="another-kind-replaces-the-entry",
        ),
        pytest.param(
            {"cuda": "exec-cuda", "omp": {"kind": "exec-omp", "host": "box1"}},
            {"omp": {"host": "box2"}},
            {"cuda": "exec-cuda", "omp": {"kind": "exec-omp", "host": "box2"}},
            id="a-mapping-without-kind-merges-into-the-inherited-kind-section",
        ),
        pytest.param(
            {"cuda": "exec-cuda", "omp": {"kind": "exec-omp", "host": "box1"}},
            {"omp": "exec-omp"},
            {"cuda": "exec-cuda", "omp": "exec-omp"},
            id="a-name-replaces-a-kind-section",
        ),
    ],
)
def test_two_per_language_sections_merge_entry_by_entry(
    tmp_path: Path, parent: Mapping[str, Any], child: Mapping[str, Any], merged: Mapping[str, Any]
) -> None:
    write_recipe(tmp_path, "merge-parent", recipe_data(DIRECTIONS, parent))
    recipe = load(tmp_path, "merge-child", {"extends": "merge-parent", "executor": dict(child)})
    assert recipe.data["executor"] == merged


def test_the_source_language_needs_no_executor_when_the_baseline_builds_only_the_target(tmp_path: Path) -> None:
    log = Log(replies=[GOOD_REPLIES["cuda"]])
    outcome = run_all(tmp_path, "target-only", recipe_data([OMP_TO_CUDA], {"cuda": "exec-cuda"}, both=False), log)
    assert [run.executor for run in log.runs] == ["exec-cuda", "exec-cuda"], "the target reference and the attempt"
    assert [build.toolchain for build in builds_before_first_request(log)] == [TOOLCHAINS["cuda"]]
    assert outcome.provenance["executor"] == {"cuda": "exec-cuda"}
    assert outcome.provenance["devices"] == {"cuda": DEVICE["exec-cuda"]}


def executor_naming(device: Any) -> type:
    """Return a sandboxed Executor class, registered as "exec-odd" by the tests, whose device() returns `device`."""

    class OddDeviceExecutor:
        """Names the given device and runs nothing; being asked to run fails the test."""

        name = "exec-odd"
        capabilities = SANDBOXED_RUNNER

        def device(self) -> Any:
            """Return the device the test gave."""
            return device

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Fail the test: a refused run runs nothing."""
            raise AssertionError(f"exec-odd was asked to run {artifact}")

    return OddDeviceExecutor


@pytest.mark.parametrize(
    "device",
    [" leading blank", "trailing blank ", "tab\tinside", "two\nlines", "", "caf\N{LATIN SMALL LETTER E WITH ACUTE}", 7],
    ids=["leading-blank", "trailing-blank", "tab", "two-lines", "empty", "not-ascii", "not-a-string"],
)
@pytest.mark.parametrize("form", ["single", "per-language"])
def test_a_device_that_is_not_one_line_of_printable_ascii_is_refused_before_any_directory(
    tmp_path: Path, device: Any, form: str
) -> None:
    log = Log()
    run_options = options(tmp_path, log)
    assert run_options.registry is not None
    run_options.registry.register("Executor", "exec-odd", executor_naming(device))
    executor = {"kind": "exec-odd"} if form == "single" else {"cuda": "exec-odd", "omp": "exec-omp"}
    with pytest.raises(RunError) as caught:
        run_recipe(write_recipe(tmp_path, "odd-device", recipe_data([OMP_TO_CUDA], executor)), run_options)
    assert "exec-odd" in str(caught.value) and "device()" in str(caught.value)
    assert not (tmp_path / "runs-root").exists() and log.events == []


def test_a_device_of_none_records_no_device(tmp_path: Path) -> None:
    log = Log(replies=[GOOD_REPLIES["cuda"]])
    run_options = options(tmp_path, log)
    assert run_options.registry is not None
    run_options.registry.register("Executor", "exec-odd", executor_naming(None))
    data = recipe_data([OMP_TO_CUDA], {"cuda": "exec-odd", "omp": "exec-omp"}, stages=COMPILE_STAGES)
    run_dir = run_recipe(write_recipe(tmp_path, "none-named", data), run_options)
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="ascii"))
    assert provenance["devices"] == {"cuda": None, "omp": DEVICE["exec-omp"]}
    trial_id = make_trial_id("none-named", MODEL_ID, SUITE, "omp-cuda", ITEM, 1)
    assert read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir)).provenance.device is None


def baseline_description(tmp_path: Path, executor: Mapping[str, Any], direction: Direction, *, both: bool) -> str:
    """Return BaselineStage.describe() for a loaded recipe, with each bound executor built as the runner builds it.

    The single form has no executor languages, so its one executor is the context's executor and executors is {}.
    """
    registry = make_registry(Log())
    recipe = load_recipe(
        write_recipe(tmp_path, "describe", recipe_data([direction], executor, both=both)),
        roots=(tmp_path, PROJECTS),
        registry=registry,
    )
    bindings = [binding for binding in recipe.bindings if binding.interface == "Executor"]
    built = [registry.get("Executor", binding.name).factory(**binding.config) for binding in bindings]
    languages = executor_languages(recipe.data)
    by_language = dict(zip(languages, built, strict=True)) if languages else {}
    context = RunContext(
        recipe=recipe,
        backend=None,  # type: ignore[arg-type]
        sampling=Sampling(temperature=0.0, top_p=1.0, max_tokens=1),
        toolchains={},
        executor=by_language.get(direction.target, built[0]),
        store=TextStore(tmp_path / "store"),
        suite=load_suite(SUITE_MANIFEST),
        sources_root=tmp_path,
        item=ITEM,
        direction=direction,
        build_root=tmp_path,
        prompts=PROMPTS,
        max_corrections=0,
        executors=by_language,
    )
    return BaselineStage(context=context).describe()


def test_the_baseline_describes_each_reference_by_its_own_languages_executor(tmp_path: Path) -> None:
    text = baseline_description(tmp_path, {"cuda": "none", "omp": "exec-omp"}, CUDA_TO_OMP, both=True)
    assert text == (
        "baseline: build and run the omp reference program and build the cuda reference program before any model call"
    )


def test_the_baseline_description_with_baseline_both_off_names_only_the_target(tmp_path: Path) -> None:
    text = baseline_description(tmp_path, {"kind": "exec-cuda"}, OMP_TO_CUDA, both=False)
    assert text == "baseline: build and run only the cuda reference program before any model call"
