"""Tests for the baseline stage, final.end_reason, and Trial.reference_run (task P1.5, first half).

Bible: Source Papers (LASSI pipeline step 1; quirk table), Component
Interfaces (Stage row: Baseline; Toolchain and Executor contract rules),
Harness Contract, Result Record, Design Principle 4.

The contract these tests fix, from the P1.5 acceptance criteria
(plans/p1-faithful.md) and the P1.5 Decision Log entries (baseline stage,
Trial.reference_run, final.end_reason; faithful correction prompt):

- lassi.core.record gains END_REASONS, the fixed end codes (at least
  `baseline-compile`, `baseline-run`, and `correction-cap`), and the record
  EndReason(code, message): the code is one of END_REASONS and the message a
  non-empty string, else ValueError. Final gains `end_reason: EndReason |
  None = None` (None: the trial ended normally). Trial gains `reference_run:
  RunInfo`, the target reference's baseline run, all None by default and for
  a compile-only executor. Both round-trip through JSON, trial.md shows the
  end reason's code and message, and the Parquet trials table carries the
  code as `final_end_reason_code` (nested fields become `<field>_<key>`).
- The stage "baseline" is registered in DEFAULT_REGISTRY once
  lassi.core.runner is imported; it names the fix `baseline_both` in
  `reproduces`. `baseline_both` is a named fix in lassi.core.recipe.FIXES, on
  by default and off under `faithful: true`.
- With `baseline_both` off (upstream): the stage builds only the target
  reference (Suite.reference_target) with the target language's toolchain,
  in a fresh build directory under the trial's directory that is no
  attempt's build directory, before any model call. With it on: it builds
  the source reference too, with the source language's toolchain; the runner
  refuses such a recipe before anything runs when no toolchain is bound for
  the source language. Every baseline build gets the item's support files
  as harness files (`harness=`; entropy needs reference.h).
- When the executor runs programs (capability `runs_code`), each reference
  built is run once with the item's run arguments (jacobi's one empty
  argument included), and the target's run is kept in Trial.reference_run:
  exit_code, hang, wall_s, and stdout_ref (the stdout in the text store). A
  compile-only executor is never asked to run anything and reference_run
  stays all None (its placeholder values never enter the record).
- A reference that does not build ends the trial with final.end_reason code
  `baseline-compile`; one whose run exits nonzero or hangs ends it with
  `baseline-run`. Either way no model is asked, no attempt is appended,
  final.stage_reached is None and final.corrections 0, and the runner keeps
  the end reason in trial.json, trial.md, and Parquet. A trial whose baseline
  and attempts go through has no end reason.

The runs use the p0-smoke template prompt set, so no upstream text is
needed; faithful runs over the lassi-2024 fragments are in
test_correction_loop.py. Fake toolchains write the files and a PLACEHOLDER
artifact and compile nothing; the scripted executor answers with SYNTHETIC
run results and runs nothing; the scripted backend answers with SYNTHETIC
replies. Bench sources are small SYNTHETIC files, not HeCBench sources. No
value in this module is a measurement: SYNTHETIC_WALL_S is a fixture value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import Direction, load_suite
from lassi.core import record as record_module
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.parquet import read_run_parquet
from lassi.core.recipe import FIXES, load_recipe
from lassi.core.record import Final, ModelInfo, Provenance, RunInfo, Trial, from_json, make_trial_id, to_json
from lassi.core.registry import DEFAULT_REGISTRY, Registry, RegistryError
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.core.trial_md import render_trial_md
from lassi.executors.workdir import build_dir

REPO = Path(__file__).resolve().parents[2]
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
MODEL_ID = "scripted-fixture"
PROMPTS = "p0-smoke"
OMP_TO_CUDA = Direction("omp", "cuda")
CUDA_TO_OMP = Direction("cuda", "omp")
DIRECTIONS = [OMP_TO_CUDA, CUDA_TO_OMP]
DIRECTION_IDS = ["omp-cuda", "cuda-omp"]
TOOLCHAINS = {"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"}
STAGES = ["baseline", "generate", "compile_loop"]
COMPILED = "S4"
BASELINE_FIX = "baseline_both"
END_CODES = ("baseline-compile", "baseline-run", "correction-cap")
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
# A commit id for synthetic provenance; not a commit of this repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"

# SYNTHETIC bench sources; the fake toolchain refuses any file holding an `#error` line.
OMP_SOURCE = '#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  std::printf("done\\n");\n}\n'
CUDA_SOURCE = "#include <cstdio>\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n"
SOURCES = {"omp": OMP_SOURCE, "cuda": CUDA_SOURCE}
BROKEN_SOURCE = "#error SYNTHETIC reference that does not build\nint main() { return 0; }\n"
REFERENCE_H = "// SYNTHETIC stand-in for the support header of the entropy item\n#define SYNTHETIC_REFERENCE 1\n"
GOOD_SOURCE = "int main() { return 0; }\n"
GOOD_REPLIES = {
    "cuda": render_file_blocks({"main.cu": GOOD_SOURCE}),
    "omp": render_file_blocks({"main.cpp": GOOD_SOURCE}),
}

# SYNTHETIC run results the scripted executor gives, per language of the reference it runs; not measurements.
SYNTHETIC_WALL_S = 1.25
TARGET_STDOUT = {"cuda": "SYNTHETIC cuda reference stdout\nPASS\n", "omp": "SYNTHETIC omp reference stdout\nPASS\n"}


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate and compile variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CPATH", raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


# ---------------------------------------------------------------------------
# Names this task adds, looked up so a missing one fails its test with a clear message


def record_name(name: str) -> Any:
    """Return lassi.core.record.<name>; fail the test clearly while it is missing."""
    value = getattr(record_module, name, None)
    if value is None:
        pytest.fail(f"lassi.core.record has no {name}; task P1.5 adds it")
    return value


def end_reason_of(trial: Trial) -> Any:
    """Return trial.final.end_reason; fail the test clearly while Final has no such field."""
    if not hasattr(trial.final, "end_reason"):
        pytest.fail("Final has no end_reason field; task P1.5 adds final.end_reason")
    return trial.final.end_reason


def reference_run_of(trial: Trial) -> RunInfo:
    """Return trial.reference_run; fail the test clearly while Trial has no such field."""
    if not hasattr(trial, "reference_run"):
        pytest.fail("Trial has no reference_run field; task P1.5 adds the reference run to the Result Record")
    return trial.reference_run


def require_fixes(*names: str) -> None:
    """Fail the test clearly when a named fix is not in lassi.core.recipe.FIXES yet."""
    missing = [name for name in names if name not in FIXES]
    if missing:
        pytest.fail(f"lassi.core.recipe.FIXES has no {', '.join(missing)}; task P1.5 adds it")


def registered_stage(name: str) -> type:
    """Return the Stage class registered as `name` in DEFAULT_REGISTRY; fail clearly while it is missing."""
    try:
        return DEFAULT_REGISTRY.get("Stage", name).factory
    except RegistryError as error:
        pytest.fail(f"no stage is registered as {name!r} ({error}); task P1.5 adds the baseline stage")


# ---------------------------------------------------------------------------
# Bench sources


def write_bench(root: Path, item: str = "layout", sources: Mapping[str, str] | None = None) -> Path:
    """Write the item's SYNTHETIC sources per language, and its support files, as the manifest lays them out."""
    spec = load_suite(SUITE_MANIFEST).items[item]
    for language, text in (sources or SOURCES).items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    for relative in spec.support.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(REFERENCE_H.encode("ascii"))
    return root


def reference_files(item: str, language: str, sources: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the reference program of `item` in `language` as the build sees it: file name -> text."""
    name = load_suite(SUITE_MANIFEST).items[item].languages[language].files[0]
    return {name: (sources or SOURCES)[language]}


# ---------------------------------------------------------------------------
# Fake components, all writing to one Log so the order of builds, runs, and model calls shows


@dataclass
class Build:
    """One build a fake toolchain was asked for."""

    toolchain: str
    workdir: Path
    files: dict[str, str]
    harness: dict[str, str]
    artifact: Path | None


@dataclass
class Run:
    """One run the scripted executor was asked for."""

    artifact: Path
    inputs: list[str]
    limits: Limits


@dataclass
class Log:
    """What the fake components saw: events in order, builds, runs, model requests, and the replies left."""

    replies: list[str] = field(default_factory=list)
    run_results: dict[str, RunResult] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    builds: list[Build] = field(default_factory=list)
    runs: list[Run] = field(default_factory=list)
    requests: list[list[Message]] = field(default_factory=list)


def fake_toolchain(registered_as: str, log: Log) -> type:
    """Return a Toolchain class without PIN, registered as `registered_as`, that records each build in `log`.

    A build whose files hold an `#error` line fails with one compile error;
    any other build writes a PLACEHOLDER artifact `main`. It runs nothing.
    """

    class FakeToolchain:
        """Writes the files and harness files under a fresh workdir and reports a build; compiles nothing."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Record the build, write every file, and return the artifact or one compile error."""
            workdir = Path(workdir)
            assert workdir.is_dir() and not any(workdir.iterdir()), f"{workdir} is not a fresh build directory"
            for path, text in [*files.items(), *(harness or {}).items()]:
                target = workdir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            failed = any("#error" in text for text in files.values())
            artifact = None if failed else workdir / "main"
            if artifact is not None:
                artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            log.events.append(f"build {registered_as}")
            log.builds.append(Build(registered_as, workdir, dict(files), dict(harness or {}), artifact))
            if failed:
                error = record_module.Diagnostic(
                    stage="compile", severity="error", code="fake-error", message="SYNTHETIC: an #error line"
                )
                return BuildResult(artifact=None, diagnostics=[error])
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


def scripted_executor(log: Log) -> type:
    """Return an Executor class that runs programs (`runs_code`) and answers from `log.run_results` by language.

    The language is the one whose reference file lies beside the artifact
    (main.cu for cuda, main.cpp for omp). It starts no process.
    """

    class ScriptedExecutor:
        """Records each run and returns the SYNTHETIC RunResult of the artifact's language."""

        name = "scripted"
        capabilities = frozenset({"runs_code"})

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Record the run and return the scripted result."""
            artifact = Path(artifact)
            log.events.append("run")
            log.runs.append(Run(artifact, list(inputs), limits))
            language = "cuda" if (artifact.parent / "main.cu").is_file() else "omp"
            return log.run_results[language]

    return ScriptedExecutor


class CompileOnlyExecutor:
    """A compile-only Executor registered as "none"; being asked to run anything fails the test."""

    name = "none"
    capabilities = frozenset({"compile_only"})

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
    """Return a test Registry: the scripted backend, both executors, fake toolchains, and the real stages."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(log))
    registry.register("Executor", "none", CompileOnlyExecutor)
    registry.register("Executor", "scripted", scripted_executor(log))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, fake_toolchain(name, log))
    for name in ("baseline", "generate", "compile_loop"):
        registry.register("Stage", name, registered_stage(name))
    return registry


# ---------------------------------------------------------------------------
# Recipes and runs


def recipe_data(
    direction: Direction = OMP_TO_CUDA,
    item: str = "layout",
    *,
    both: bool | None = None,
    executor: str = "none",
    languages: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a template-set recipe for one item and direction; `both` sets fixes.baseline_both when not None.

    `languages` names the languages bound to a toolchain (default: the
    target alone, plus the source when `both` is not False).
    """
    if languages is None:
        languages = [direction.target] if both is False else [direction.target, direction.source]
    data: dict[str, Any] = {
        "extends": "base",
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [item]},
        "directions": [{"source": direction.source, "target": direction.target}],
        "prompts": PROMPTS,
        "toolchain": {language: TOOLCHAINS[language] for language in languages},
        "stages": list(STAGES),
        "executor": {"kind": executor},
        "trials": {"n": 1},
    }
    if both is not None:
        data["fixes"] = {BASELINE_FIX: both}
    return data


def write_recipe(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe `<directory>/<name>.yaml` and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


@dataclass
class Outcome:
    """One finished run: its directory, its one trial read back from the run tree, the trial id, and the log."""

    run_dir: Path
    trial_id: str
    trial: Trial
    log: Log

    def stored(self, ref: Any) -> str:
        """Return a text of the run's text store."""
        return TextStore(self.run_dir).get(ref)


def run_one(
    tmp_path: Path, bench_root: Path, name: str, data: Mapping[str, Any], log: Log
) -> Outcome:
    """Run a one-trial recipe with the fake components answering from `log`; return the outcome."""
    if "fixes" in data:
        require_fixes(*data["fixes"])
    registry = make_registry(log)
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench_root, registry=registry)
    run_dir = run_recipe(write_recipe(tmp_path, name, data), options)
    direction = data["directions"][0]
    item = data["bench"]["items"][0]
    trial_id = make_trial_id(name, MODEL_ID, SUITE, f"{direction['source']}-{direction['target']}", item, 1)
    trial = read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))
    return Outcome(run_dir, trial_id, trial, log)


def ran(stdout: str, exit_code: int | None = 0, hang: bool = False) -> RunResult:
    """Return a SYNTHETIC RunResult; its wall time is a fixture value, not a measurement."""
    return RunResult(exit_code=exit_code, hang=hang, stdout=stdout, stderr="", wall_s=SYNTHETIC_WALL_S)


def clean_runs() -> dict[str, RunResult]:
    """Return a clean SYNTHETIC run of each language's reference."""
    return {language: ran(stdout) for language, stdout in TARGET_STDOUT.items()}


def attempt_dirs(outcome: Outcome) -> list[Path]:
    """Return the resolved build directory of every attempt of the outcome's trial."""
    return [build_dir(outcome.run_dir, outcome.trial_id, attempt.index).resolve() for attempt in outcome.trial.attempts]


def builds_before_first_request(log: Log) -> list[Build]:
    """Return the builds made before the first model request, which are the baseline's."""
    assert "request" in log.events, "the trial never asked the model"
    count = sum(1 for event in log.events[: log.events.index("request")] if event.startswith("build "))
    return log.builds[:count]


# ---------------------------------------------------------------------------
# The Result Record: EndReason, Final.end_reason, Trial.reference_run


def synthetic_trial(**changes: Any) -> Trial:
    """Return a SYNTHETIC trial of layout (omp to cuda) with no attempts; `changes` replace fields."""
    suite = load_suite(SUITE_MANIFEST)
    fields: dict[str, Any] = {
        "trial_id": make_trial_id("baseline-test", MODEL_ID, SUITE, "omp-cuda", "layout", 1),
        "recipe_hash": "0123456789abcdef" * 4,
        "provenance": Provenance(
            commit=FAKE_COMMIT, dirty=False, device="none (compile only)", sdk=None, date="2026-09-24T00:00:00+00:00"
        ),
        "bench_item": suite.bench_item("layout", OMP_TO_CUDA),
        "model": ModelInfo(backend="scripted", id=MODEL_ID, sampling=SAMPLING),
    }
    fields.update(changes)
    return Trial(**fields)


def test_the_end_codes_are_fixed_and_an_end_reason_holds_a_code_and_a_message() -> None:
    codes = record_name("END_REASONS")
    end_reason = record_name("EndReason")
    assert set(END_CODES) <= set(codes), f"END_REASONS must hold {END_CODES}, got {codes}"
    assert all(isinstance(code, str) for code in codes)
    reason = end_reason(code="baseline-compile", message="SYNTHETIC: the cuda reference did not build")
    assert (reason.code, reason.message) == ("baseline-compile", "SYNTHETIC: the cuda reference did not build")
    assert end_reason_of(synthetic_trial()) is None, "a trial ends with no end reason unless a stage sets one"
    final = Final(stage_reached=None, corrections=0, end_reason=reason)
    assert final.end_reason == reason


@pytest.mark.parametrize(
    ("code", "message"),
    [("no-such-code", "SYNTHETIC message"), ("baseline-run", ""), ("", "SYNTHETIC message")],
    ids=["unknown-code", "empty-message", "empty-code"],
)
def test_an_end_reason_refuses_an_unknown_code_and_an_empty_message(code: str, message: str) -> None:
    end_reason = record_name("EndReason")
    with pytest.raises(ValueError, match="EndReason"):
        end_reason(code=code, message=message)


def test_reference_run_defaults_to_not_run_and_both_new_fields_round_trip_through_json(tmp_path: Path) -> None:
    assert reference_run_of(synthetic_trial()) == RunInfo(), "a trial whose reference never ran keeps all None"
    store = TextStore(tmp_path / "store")
    reference = RunInfo(exit_code=0, hang=False, wall_s=SYNTHETIC_WALL_S, stdout_ref=store.put(TARGET_STDOUT["cuda"]))
    reason = record_name("EndReason")(code="correction-cap", message="SYNTHETIC: the cap of 2 corrections was hit")
    trial = synthetic_trial(reference_run=reference, final=Final(stage_reached="S1", corrections=2, end_reason=reason))
    again = from_json(Trial, to_json(trial))
    assert again == trial
    assert again.reference_run.stdout_ref == reference.stdout_ref
    assert again.final.end_reason == reason


def test_trial_md_shows_the_end_reason_code_and_message(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "store")
    message = "SYNTHETIC: the omp reference program exited with status 3"
    reason = record_name("EndReason")(code="baseline-run", message=message)
    page = render_trial_md(synthetic_trial(final=Final(corrections=0, end_reason=reason)), store)
    assert "baseline-run" in page
    assert message in page


# ---------------------------------------------------------------------------
# The fix and the stage


def test_baseline_both_is_a_named_fix_on_by_default_off_under_faithful_and_reproduced_by_the_baseline(
    tmp_path: Path,
) -> None:
    require_fixes(BASELINE_FIX)
    assert isinstance(FIXES[BASELINE_FIX], str) and FIXES[BASELINE_FIX].strip(), "the fix needs a description"
    assert BASELINE_FIX in getattr(registered_stage("baseline"), "reproduces", ()), (
        "the baseline stage reproduces upstream's target-only baseline, so it names the fix in `reproduces`"
    )
    plain = recipe_data()
    faithful = {**plain, "faithful": True}
    registry = make_registry(Log())
    assert load_recipe(write_recipe(tmp_path, "plain", plain), registry=registry).data["fixes"][BASELINE_FIX] is True
    faithful_fixes = load_recipe(write_recipe(tmp_path, "faithful", faithful), registry=registry).data["fixes"]
    assert faithful_fixes[BASELINE_FIX] is False


@pytest.mark.parametrize("direction", DIRECTIONS, ids=DIRECTION_IDS)
def test_with_the_fix_off_the_baseline_builds_only_the_target_reference_before_any_model_call(
    tmp_path: Path, direction: Direction
) -> None:
    log = Log(replies=[GOOD_REPLIES[direction.target]])
    bench = write_bench(tmp_path / "bench")
    outcome = run_one(tmp_path, bench, "target-only", recipe_data(direction, both=False), log)
    assert log.events[:2] == [f"build {TOOLCHAINS[direction.target]}", "request"], (
        "the target reference is built once, before the first model call"
    )
    baseline = log.builds[0]
    assert baseline.files == reference_files("layout", direction.target)
    assert [build.toolchain for build in log.builds] == [TOOLCHAINS[direction.target]] * 2, (
        "the baseline builds the target reference and attempt 0 is built next; the source is never built"
    )
    trial_root = trial_dir(outcome.run_dir, outcome.trial_id).resolve()
    assert trial_root in baseline.workdir.resolve().parents, "the baseline builds under the trial's directory"
    assert baseline.workdir.resolve() not in attempt_dirs(outcome), "the baseline never uses an attempt's directory"
    assert log.builds[1].workdir.resolve() == attempt_dirs(outcome)[0]
    assert reference_run_of(outcome.trial) == RunInfo(), "a compile-only executor leaves the reference run all None"
    assert outcome.trial.final.stage_reached == COMPILED
    assert end_reason_of(outcome.trial) is None


@pytest.mark.parametrize("direction", DIRECTIONS, ids=DIRECTION_IDS)
def test_with_the_fix_on_the_baseline_builds_both_references(tmp_path: Path, direction: Direction) -> None:
    log = Log(replies=[GOOD_REPLIES[direction.target]])
    bench = write_bench(tmp_path / "bench")
    run_one(tmp_path, bench, "both-built", recipe_data(direction, both=True), log)
    baseline = builds_before_first_request(log)
    assert len(baseline) == 2
    built = {build.toolchain: build.files for build in baseline}
    assert built == {
        TOOLCHAINS[direction.source]: reference_files("layout", direction.source),
        TOOLCHAINS[direction.target]: reference_files("layout", direction.target),
    }, "with baseline_both on, both reference programs are built, each by its language's toolchain"


def test_the_fix_is_on_by_default_so_a_recipe_without_it_builds_both(tmp_path: Path) -> None:
    require_fixes(BASELINE_FIX)
    log = Log(replies=[GOOD_REPLIES["cuda"]])
    bench = write_bench(tmp_path / "bench")
    run_one(tmp_path, bench, "default-fix", recipe_data(OMP_TO_CUDA, languages=["cuda", "omp"]), log)
    baseline = builds_before_first_request(log)
    assert sorted(build.toolchain for build in baseline) == sorted(TOOLCHAINS.values())


def test_with_the_fix_on_a_recipe_without_a_source_toolchain_is_refused_before_anything_runs(tmp_path: Path) -> None:
    require_fixes(BASELINE_FIX)
    log = Log(replies=[GOOD_REPLIES["cuda"]])
    bench = write_bench(tmp_path / "bench")
    data = recipe_data(OMP_TO_CUDA, both=True, languages=["cuda"])
    registry = make_registry(log)
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench, registry=registry)
    with pytest.raises(RunError, match=r"toolchain.*\bomp\b|\bomp\b.*toolchain"):
        run_recipe(write_recipe(tmp_path, "no-source-toolchain", data), options)
    assert not (tmp_path / "runs-root").exists(), "a refused run creates no directory"
    assert log.events == [], "a refused run builds nothing and asks no model"


@pytest.mark.parametrize(
    "stages",
    [["generate", "baseline", "compile_loop"], ["generate", "compile_loop", "baseline"]],
    ids=["after-generate", "after-compile-loop"],
)
def test_a_baseline_listed_after_a_stage_that_asks_the_model_is_refused_before_anything_runs(
    tmp_path: Path, stages: list[str]
) -> None:
    log = Log(replies=[GOOD_REPLIES["cuda"]])
    bench = write_bench(tmp_path / "bench")
    data = {**recipe_data(OMP_TO_CUDA, both=False), "stages": stages}
    registry = make_registry(log)
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench, registry=registry)
    with pytest.raises(RunError, match=r"'baseline'.*before"):
        run_recipe(write_recipe(tmp_path, "late-baseline", data), options)
    assert not (tmp_path / "runs-root").exists(), "a refused run creates no directory"
    assert log.events == [], "a refused run builds nothing and asks no model"


@pytest.mark.parametrize("both", [False, True], ids=["target-only", "both"])
@pytest.mark.parametrize("direction", DIRECTIONS, ids=DIRECTION_IDS)
def test_every_baseline_build_gets_the_items_support_files_as_harness_files(
    tmp_path: Path, direction: Direction, both: bool
) -> None:
    log = Log(replies=[GOOD_REPLIES[direction.target]])
    bench = write_bench(tmp_path / "bench", item="entropy")
    run_one(tmp_path, bench, "entropy-harness", recipe_data(direction, "entropy", both=both), log)
    baseline = builds_before_first_request(log)
    assert len(baseline) == (2 if both else 1)
    for build in baseline:
        assert build.harness == {"reference.h": REFERENCE_H}, f"{build.toolchain} got harness {build.harness}"
        assert "reference.h" not in build.files, "a support file is a harness file, never a model file"


@pytest.mark.parametrize("item", ["layout", "jacobi"])
def test_with_an_executor_that_runs_programs_the_target_reference_runs_once_and_is_recorded(
    tmp_path: Path, item: str
) -> None:
    log = Log(replies=[GOOD_REPLIES["cuda"]], run_results=clean_runs())
    bench = write_bench(tmp_path / "bench", item=item)
    outcome = run_one(tmp_path, bench, "reference-run", recipe_data(OMP_TO_CUDA, item, both=False,
                                                                      executor="scripted"), log)
    assert log.events[:3] == [f"build {TOOLCHAINS['cuda']}", "run", "request"], (
        "the target reference is built and run before the first model call"
    )
    (run,) = log.runs
    assert run.artifact == log.builds[0].artifact
    assert run.inputs == list(load_suite(SUITE_MANIFEST).items[item].run_args), (
        "the reference runs with the item's run arguments exactly (jacobi: one empty argument)"
    )
    reference = reference_run_of(outcome.trial)
    assert (reference.exit_code, reference.hang, reference.wall_s) == (0, False, SYNTHETIC_WALL_S)
    assert reference.stdout_ref is not None, "the reference stdout is kept in the text store"
    assert outcome.stored(reference.stdout_ref) == TARGET_STDOUT["cuda"]
    assert end_reason_of(outcome.trial) is None


@pytest.mark.parametrize("direction", DIRECTIONS, ids=DIRECTION_IDS)
def test_with_the_fix_on_both_references_run_and_the_target_run_is_the_reference(
    tmp_path: Path, direction: Direction
) -> None:
    log = Log(replies=[GOOD_REPLIES[direction.target]], run_results=clean_runs())
    bench = write_bench(tmp_path / "bench")
    outcome = run_one(tmp_path, bench, "both-run", recipe_data(direction, both=True, executor="scripted"), log)
    assert len(log.runs) == 2, "with baseline_both on, both reference programs run"
    assert log.events.index("request") > max(index for index, event in enumerate(log.events) if event == "run")
    reference = reference_run_of(outcome.trial)
    assert reference.stdout_ref is not None
    assert outcome.stored(reference.stdout_ref) == TARGET_STDOUT[direction.target]


# ---------------------------------------------------------------------------
# Baseline failures end the trial before any model call


def assert_ended_before_any_model_call(outcome: Outcome, code: str) -> None:
    """Assert the trial ended with `code` and a message, no model call, no attempt, in the record and its mirrors."""
    reason = end_reason_of(outcome.trial)
    assert reason is not None, "a baseline failure sets final.end_reason"
    assert reason.code == code
    assert isinstance(reason.message, str) and reason.message.strip()
    assert outcome.log.requests == [], "a baseline failure ends the trial before any model call"
    assert outcome.trial.attempts == []
    assert (outcome.trial.final.stage_reached, outcome.trial.final.corrections) == (None, 0)
    page = (trial_dir(outcome.run_dir, outcome.trial_id) / "trial.md").read_text(encoding="ascii")
    assert code in page and reason.message in page, "trial.md shows the end reason"
    (row,) = read_run_parquet(outcome.run_dir / "parquet")["trials"]
    assert row.get("final_end_reason_code") == code, "the Parquet trials table carries final_end_reason_code"


@pytest.mark.parametrize(
    ("both", "broken"),
    [(False, "target"), (True, "target"), (True, "source")],
    ids=["target-only-target-broken", "both-target-broken", "both-source-broken"],
)
def test_a_reference_that_does_not_build_ends_the_trial_with_baseline_compile(
    tmp_path: Path, both: bool, broken: str
) -> None:
    direction = OMP_TO_CUDA
    language = direction.target if broken == "target" else direction.source
    sources = {**SOURCES, language: BROKEN_SOURCE}
    log = Log(replies=[GOOD_REPLIES["cuda"]], run_results=clean_runs())
    bench = write_bench(tmp_path / "bench", sources=sources)
    data = recipe_data(direction, both=both, executor="scripted")
    outcome = run_one(tmp_path, bench, "baseline-compile", data, log)
    assert_ended_before_any_model_call(outcome, "baseline-compile")
    built = {(build.toolchain, build.artifact is None) for build in log.builds}
    assert (TOOLCHAINS[language], True) in built, "the broken reference was built and failed"


@pytest.mark.parametrize(
    ("both", "results"),
    [
        (False, {"cuda": ran("SYNTHETIC partial\n", exit_code=3), "omp": ran(TARGET_STDOUT["omp"])}),
        (False, {"cuda": ran("", exit_code=None, hang=True), "omp": ran(TARGET_STDOUT["omp"])}),
        (True, {"cuda": ran(TARGET_STDOUT["cuda"]), "omp": ran("SYNTHETIC partial\n", exit_code=134)}),
    ],
    ids=["target-exits-nonzero", "target-hangs", "both-source-exits-nonzero"],
)
def test_a_reference_run_that_fails_ends_the_trial_with_baseline_run(
    tmp_path: Path, both: bool, results: dict[str, RunResult]
) -> None:
    log = Log(replies=[GOOD_REPLIES["cuda"]], run_results=results)
    bench = write_bench(tmp_path / "bench")
    outcome = run_one(tmp_path, bench, "baseline-run", recipe_data(OMP_TO_CUDA, both=both, executor="scripted"), log)
    assert_ended_before_any_model_call(outcome, "baseline-run")


def test_a_failed_target_reference_run_is_kept_in_the_reference_run(tmp_path: Path) -> None:
    results = {"cuda": ran("SYNTHETIC partial\n", exit_code=3), "omp": ran(TARGET_STDOUT["omp"])}
    log = Log(replies=[], run_results=results)
    bench = write_bench(tmp_path / "bench")
    outcome = run_one(tmp_path, bench, "failed-reference", recipe_data(OMP_TO_CUDA, both=False,
                                                                        executor="scripted"), log)
    reference = reference_run_of(outcome.trial)
    assert (reference.exit_code, reference.hang) == (3, False), "the record says how the reference run ended"
