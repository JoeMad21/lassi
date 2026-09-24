"""Tests for the oracle stage and its place in the runner (task P1.7).

Bible: Oracles, Component Interfaces (Oracle and Stage rows; Stage contract
rules: stages are pure over the trial record), Result Record
(Attempt.alignment, Attempt.run), Project Recipes (lassi-repro block:
`oracle: {kind: stdout_mask, passfail: true}` and the `oracle` stage).

What these tests expect:

- Importing lassi.core.runner registers Stage "oracle" and Oracle
  "stdout_mask" in DEFAULT_REGISTRY; the oracle accepts the config key
  `passfail`, so the lassi-repro oracle section binds.
- `oracle` is no longer on the runner's not-carried-out list: a recipe with
  an oracle section runs. A recipe that lists the oracle stage but binds no
  oracle is refused before anything runs (no oracle is picked silently).
  So is a recipe that sets an oracle section when no listed stage uses an
  Oracle, and one whose oracle section leaves `passfail` unset; each is
  refused before any directory is created.
- The stage is built as `factory(context=<RunContext>)`. Called on a trial,
  it leaves every attempt that holds no run stdout as it is (alignment
  None: per_input [] and mean None); a compile-only trial comes back equal.
- `stage.align_runs(trial, reference_stdout)` returns a new trial in which
  every attempt whose `run.stdout_ref` is set (read through the context's
  text store) carries Alignment(per_input=[v], mean=v), where v is
  passfail_score when the recipe's `oracle.passfail` is true, reading
  PASS/FAIL only when the direction's target language is in the item's
  manifest `passfail` list, and stdout_mask_score otherwise. Every other
  field and every other attempt is unchanged. That includes the stale case
  of P1.6: the attempt that ran is aligned, and later attempts that were
  never run stay None.

Where the reference stdout comes from (task P1.5): the baseline stage keeps
the target reference's run in `Trial.reference_run` (a RunInfo whose
stdout_ref names the reference stdout in the text store). Called as a
stage, the oracle stage aligns every attempt whose run holds stdout against
that stdout, exactly as `align_runs(trial, <reference stdout>)` does; a
trial whose attempts ran but that records no reference stdout still raises
rather than go unaligned. The other tests pass the reference stdout to
`align_runs` directly.

SYNTHETIC fixtures: every stdout, reply, and source here is written by hand
in the print formats of the pinned HeCBench sources. The values are
invented, not program output, and nothing here is executed.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import Direction, load_suite
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.interfaces import BuildResult, Sampling
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.record import (
    Alignment,
    Attempt,
    Diagnostic,
    ModelInfo,
    Provenance,
    RunInfo,
    Trial,
    make_trial_id,
)
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import CompileLoopStage, GenerateStage, RunContext
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.executors import NoneExecutor
from lassi.llm import MockBackend

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "tests" / "fixtures" / "recipes" / "p0-smoke.yaml"
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
OMP_TO_CUDA = Direction("omp", "cuda")
CUDA_TO_OMP = Direction("cuda", "omp")
ORACLE = {"kind": "stdout_mask", "passfail": True}
STAGES = ["generate", "compile_loop", "oracle"]
# A commit id for synthetic provenance; not a commit of this repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"

# SYNTHETIC: layout's stdout, twice with different timings; values invented.
LAYOUT = "Average kernel execution time (AoS): 1.5 (us)\nPASS\nAverage kernel execution time (SoA): 2.5 (us)\nPASS\n"
LAYOUT_OTHER_TIMES = LAYOUT.replace("1.5 (us)", "3.75 (us)").replace("2.5 (us)", "0.5 (us)")
# SYNTHETIC: randomAccess's last line reads PASS or FAIL in its CUDA version and passed or failed in OpenMP.
RANDOM_ACCESS = (
    "Table size = 1024\nAverage kernel execution time: 1.000000 (s)\nFound 9 errors in 1024 locations ({}).\n"
)
# SYNTHETIC: colorwheel's output when its results disagree: no PASS, and colorwheel never prints FAIL.
COLORWHEEL_NO_PASS = (
    "Start execution on a device\n"
    "Average kernel execution time : 1.000000 (ms)\n"
    "Maximum error between host and device results: 3\n"
)
# SYNTHETIC: the sources the mock backend reads as the layout references (never compiled or run).
FAKE_SOURCES = {
    "src/layout-omp/main.cpp": "int main() {\n#pragma omp target\n  { }\n  return 0;\n}\n",
    "src/layout-cuda/main.cu": "__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n",
}


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
# Recipes, contexts, and trials


def recipe_data(item: str = "layout", **changes: Any) -> dict[str, Any]:
    """Return the p0-smoke recipe for `item` with top-level keys replaced by `changes`; None drops a key."""
    data = copy.deepcopy(yaml.safe_load(SMOKE.read_text(encoding="utf-8")))
    data["bench"]["items"] = [item]
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def write_recipe(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as `<directory>/<name>.yaml` and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def oracle_stage(tmp_path: Path, item: str, direction: Direction, passfail: bool = True) -> tuple[Any, RunContext]:
    """Return the registered oracle stage built on a RunContext for `item` and `direction`, and the context."""
    oracle = {"kind": "stdout_mask", "passfail": passfail}
    recipe = load_recipe(write_recipe(tmp_path, "oracle-test", recipe_data(item, oracle=oracle, stages=STAGES)))
    context = RunContext(
        recipe=recipe,
        backend=MockBackend("mock-reference"),
        sampling=SAMPLING,
        toolchains={},
        executor=NoneExecutor(),
        store=TextStore(tmp_path / "store"),
        suite=load_suite(SUITE_MANIFEST),
        sources_root=tmp_path / "bench",
        item=item,
        direction=direction,
        build_root=tmp_path / "builds",
        prompts="p0-smoke",
        max_corrections=10,
    )
    return DEFAULT_REGISTRY.get("Stage", "oracle").factory(context=context), context


def target_file(direction: Direction) -> str:
    """Return the file name of the direction's target language version."""
    return {"cuda": "main.cu", "omp": "main.cpp"}[direction.target]


def compiled(index: int, direction: Direction, run: RunInfo | None = None, **fields: Any) -> Attempt:
    """Return a SYNTHETIC attempt that compiled (S4), with `run` (default: never run)."""
    files = {target_file(direction): "int main() { return 0; }\n"}
    run = run or RunInfo()
    return Attempt(index=index, response_text="SYNTHETIC reply", files=files, stage_reached="S4", run=run, **fields)


def compile_failed(index: int, direction: Direction) -> Attempt:
    """Return a SYNTHETIC attempt that parsed (S1) but did not compile."""
    error = Diagnostic(stage="compile", severity="error", code="fake-error", message="SYNTHETIC compile error")
    files = {target_file(direction): "#error SYNTHETIC\n"}
    return Attempt(index=index, response_text="SYNTHETIC reply", files=files, stage_reached="S1", diagnostics=[error])


def ran(store: TextStore, stdout: str, exit_code: int = 0) -> RunInfo:
    """Return a SYNTHETIC RunInfo whose stdout is kept in `store`; the wall time is not a measurement."""
    return RunInfo(exit_code=exit_code, hang=False, wall_s=1.0, stdout_ref=store.put(stdout))


def trial_of(context: RunContext, attempts: Sequence[Attempt]) -> Trial:
    """Return a SYNTHETIC trial of the context's item and direction holding `attempts`."""
    trial_id = make_trial_id("oracle-test", "mock-reference", SUITE, context.direction.name, context.item, 1)
    return Trial(
        trial_id=trial_id,
        recipe_hash=context.recipe.recipe_hash,
        provenance=Provenance(
            commit=FAKE_COMMIT, dirty=False, device="none (compile only)", sdk=None, date="2026-09-24T00:00:00+00:00"
        ),
        bench_item=context.suite.bench_item(context.item, context.direction),
        model=ModelInfo(backend="mock", id="mock-reference", sampling=SAMPLING),
        attempts=list(attempts),
    )


def with_reference_run(trial: Trial, store: TextStore, reference_stdout: str) -> Trial:
    """Return `trial` with a SYNTHETIC reference run whose stdout is kept in `store`; the wall time is no measurement.

    Fails the test clearly while Trial has no `reference_run` field (task P1.5 adds it).
    """
    if "reference_run" not in {spec.name for spec in dataclasses.fields(Trial)}:
        pytest.fail("Trial has no reference_run field; task P1.5 adds the reference run to the Result Record")
    reference = RunInfo(exit_code=0, hang=False, wall_s=1.0, stdout_ref=store.put(reference_stdout))
    return dataclasses.replace(trial, reference_run=reference)


def assert_only_alignment_changed(before: Trial, after: Trial) -> None:
    """Assert that `after` differs from `before` in attempt alignments at most."""
    assert dataclasses.replace(after, attempts=[]) == dataclasses.replace(before, attempts=[])
    assert len(after.attempts) == len(before.attempts)
    for old, new in zip(before.attempts, after.attempts, strict=True):
        assert dataclasses.replace(new, alignment=Alignment()) == dataclasses.replace(old, alignment=Alignment())


# ---------------------------------------------------------------------------
# Registration and recipes


def test_importing_the_runner_registers_the_oracle_stage_and_the_stdout_mask_oracle() -> None:
    oracle = DEFAULT_REGISTRY.get("Oracle", "stdout_mask")
    assert "passfail" in oracle.config_keys
    for passfail in (True, False):
        assert callable(getattr(oracle.factory(passfail=passfail), "align", None))
    DEFAULT_REGISTRY.get("Stage", "oracle")


def test_a_recipe_with_the_oracle_section_and_stage_loads_with_the_default_registry(tmp_path: Path) -> None:
    recipe = load_recipe(write_recipe(tmp_path, "with-oracle", recipe_data(oracle=ORACLE, stages=STAGES)))
    bound = {(binding.interface, binding.name): dict(binding.config) for binding in recipe.bindings}
    assert bound[("Oracle", "stdout_mask")] == {"passfail": True}
    assert ("Stage", "oracle") in bound


def test_the_stage_describes_itself_in_one_line(tmp_path: Path) -> None:
    stage, _ = oracle_stage(tmp_path, "layout", OMP_TO_CUDA)
    text = stage.describe()
    assert isinstance(text, str) and text and "\n" not in text


# ---------------------------------------------------------------------------
# The stage on trials


def test_compile_only_attempts_keep_alignment_none(tmp_path: Path) -> None:
    stage, context = oracle_stage(tmp_path, "layout", OMP_TO_CUDA)
    trial = trial_of(context, [compile_failed(0, OMP_TO_CUDA), compiled(1, OMP_TO_CUDA)])
    result = stage(trial)
    assert result == trial
    assert [attempt.alignment for attempt in result.attempts] == [Alignment(), Alignment()]


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [(LAYOUT_OTHER_TIMES, 1.0), (LAYOUT.replace("PASS", "FAIL"), 0.0), ("", 0.0)],
    ids=["same-but-timing", "fail", "empty"],
)
def test_an_attempt_that_ran_gets_its_alignment(tmp_path: Path, candidate: str, expected: float) -> None:
    stage, context = oracle_stage(tmp_path, "layout", OMP_TO_CUDA)
    run = ran(context.store, candidate)
    trial = trial_of(context, [compile_failed(0, OMP_TO_CUDA), compiled(1, OMP_TO_CUDA, run)])
    result = stage.align_runs(trial, LAYOUT)
    assert result.attempts[0].alignment == Alignment()
    assert result.attempts[1].alignment == Alignment(per_input=[expected], mean=expected)
    assert_only_alignment_changed(trial, result)


def test_the_stale_output_attempt_is_aligned_and_later_unrun_attempts_stay_none(tmp_path: Path) -> None:
    """P1.6: attempt 0 ran, 1 to 8 failed to compile, 9 compiled after 8 corrections and was never run."""
    stage, context = oracle_stage(tmp_path, "layout", OMP_TO_CUDA)
    partial = LAYOUT.split("\n", 1)[0] + "\n"
    stale = Diagnostic(stage="run", severity="warning", code="stale-output", message="SYNTHETIC: attempt 0 stands")
    attempts = [
        compiled(0, OMP_TO_CUDA, ran(context.store, partial, exit_code=1)),
        *(compile_failed(index, OMP_TO_CUDA) for index in range(1, 9)),
        compiled(9, OMP_TO_CUDA, diagnostics=[stale]),
    ]
    trial = trial_of(context, attempts)
    result = stage.align_runs(trial, LAYOUT)
    assert result.attempts[0].alignment == Alignment(per_input=[0.0], mean=0.0)
    assert [attempt.alignment for attempt in result.attempts[1:]] == [Alignment()] * 9
    assert_only_alignment_changed(trial, result)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [(LAYOUT_OTHER_TIMES, 1.0), (LAYOUT.replace("PASS", "FAIL"), 0.0)],
    ids=["same-but-timing", "fail"],
)
def test_as_a_stage_it_aligns_runs_against_the_reference_run_the_baseline_recorded(
    tmp_path: Path, candidate: str, expected: float
) -> None:
    stage, context = oracle_stage(tmp_path, "layout", OMP_TO_CUDA)
    run = ran(context.store, candidate)
    attempts = [compile_failed(0, OMP_TO_CUDA), compiled(1, OMP_TO_CUDA, run)]
    trial = with_reference_run(trial_of(context, attempts), context.store, LAYOUT)
    result = stage(trial)
    assert result == stage.align_runs(trial, LAYOUT)
    assert result.attempts[0].alignment == Alignment()
    assert result.attempts[1].alignment == Alignment(per_input=[expected], mean=expected)
    assert_only_alignment_changed(trial, result)


def test_as_a_stage_it_refuses_run_stdout_when_no_reference_stdout_was_recorded(tmp_path: Path) -> None:
    stage, context = oracle_stage(tmp_path, "layout", OMP_TO_CUDA)
    trial = trial_of(context, [compiled(0, OMP_TO_CUDA, ran(context.store, LAYOUT))])
    with pytest.raises(ValueError, match="reference"):
        stage(trial)


@pytest.mark.parametrize(
    ("direction", "verdict", "expected"),
    [(OMP_TO_CUDA, "FAIL", 0.0), (OMP_TO_CUDA, "PASS", 1.0), (CUDA_TO_OMP, "failed", 1.0)],
    ids=["cuda-target-reads-fail", "cuda-target-reads-pass", "omp-target-reads-no-token"],
)
def test_passfail_is_read_only_for_a_target_language_the_manifest_lists(
    tmp_path: Path, direction: Direction, verdict: str, expected: float
) -> None:
    """randomAccess prints PASS/FAIL only in CUDA; the OpenMP output has no token, which is no fail there."""
    stage, context = oracle_stage(tmp_path, "randomAccess", direction)
    stdout = RANDOM_ACCESS.format(verdict)
    trial = trial_of(context, [compiled(0, direction, ran(context.store, stdout.replace("1.000000", "2.0")))])
    result = stage.align_runs(trial, stdout)
    assert result.attempts[0].alignment == Alignment(per_input=[expected], mean=expected)


@pytest.mark.parametrize(("passfail", "expected"), [(True, 0.0), (False, 1.0)], ids=["passfail", "mask-only"])
def test_the_recipe_passfail_choice_is_honored(tmp_path: Path, passfail: bool, expected: float) -> None:
    """colorwheel without PASS: a fail under passfail, a match under stdout_mask alone."""
    stage, context = oracle_stage(tmp_path, "colorwheel", OMP_TO_CUDA, passfail=passfail)
    trial = trial_of(context, [compiled(0, OMP_TO_CUDA, ran(context.store, COLORWHEEL_NO_PASS))])
    result = stage.align_runs(trial, COLORWHEEL_NO_PASS)
    assert result.attempts[0].alignment == Alignment(per_input=[expected], mean=expected)


# ---------------------------------------------------------------------------
# The runner


def write_bench(root: Path) -> Path:
    """Write the SYNTHETIC layout sources under `root` as the manifest lays them out, and return `root`."""
    for relative, text in FAKE_SOURCES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def fake_toolchain(registered_as: str) -> type:
    """Return a Toolchain class without PIN that writes the files and a PLACEHOLDER artifact; it runs nothing."""

    class FakeToolchain:
        """Writes the files under the fresh workdir and reports a built artifact; compiles nothing."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Write `files` and a placeholder artifact under `workdir` and return it with no diagnostics."""
            for path, text in files.items():
                target = Path(workdir) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            artifact = Path(workdir) / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


class StandInOracle:
    """An Oracle bound as stdout_mask for the runner test; nothing runs in a compile-only trial, so it never aligns."""

    name = "stdout_mask"
    capabilities = frozenset({"aligns"})
    config_keys = frozenset({"passfail"})

    def __init__(self, *, passfail: bool) -> None:
        """Keep the passfail choice."""
        self.passfail = passfail

    def align(self, reference: Any, candidate: Any) -> float:
        """Fail the test: a compile-only trial has no run to align."""
        raise AssertionError("a compile-only trial has no run to align")


def stand_in_stage(calls: list[str]) -> type:
    """Return a Stage class registered as "oracle" that records each trial it gets and returns it unchanged."""

    class StandInOracleStage:
        """Records the trial id and returns the trial."""

        name = "oracle"
        capabilities = frozenset({"aligns"})
        requires = {"Oracle": frozenset()}

        def __init__(self, *, context: RunContext) -> None:
            """Keep the context."""
            self.context = context

        def __call__(self, trial: Trial) -> Trial:
            """Record the trial id and return the trial unchanged."""
            calls.append(trial.trial_id)
            return trial

        def describe(self) -> str:
            """Return a one-line description."""
            return "oracle: stand-in for the runner test"

    return StandInOracleStage


def runner_registry(oracle_stage_class: type, oracle_class: type) -> Registry:
    """Return a Registry with the mock backend, the P0 stages, the none executor, fake toolchains, and the oracle."""
    registry = Registry()
    registry.register("LLMBackend", "mock", MockBackend)
    registry.register("Stage", "generate", GenerateStage)
    registry.register("Stage", "compile_loop", CompileLoopStage)
    registry.register("Stage", "oracle", oracle_stage_class)
    registry.register("Oracle", "stdout_mask", oracle_class)
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Toolchain", "nvcc-sm80", fake_toolchain("nvcc-sm80"))
    registry.register("Toolchain", "nvcpp-cc80", fake_toolchain("nvcpp-cc80"))
    return registry


def run_oracle_recipe(tmp_path: Path, registry: Registry, data: Mapping[str, Any]) -> Path:
    """Run the recipe `data` as oracle-run.yaml on the SYNTHETIC layout sources and return the run directory."""
    bench_root = write_bench(tmp_path / "bench")
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench_root, registry=registry)
    return run_recipe(write_recipe(tmp_path, "oracle-run", data), options)


TRIAL_ID = f"oracle-run/mock-reference/{SUITE}/omp-cuda/layout/run01"


def test_the_runner_carries_out_a_recipe_that_sets_an_oracle(tmp_path: Path) -> None:
    calls: list[str] = []
    registry = runner_registry(stand_in_stage(calls), StandInOracle)
    run_oracle_recipe(tmp_path, registry, recipe_data(oracle=ORACLE, stages=STAGES))
    assert calls == [TRIAL_ID]


def test_the_runner_refuses_the_oracle_stage_when_no_oracle_is_bound(tmp_path: Path) -> None:
    registry = runner_registry(DEFAULT_REGISTRY.get("Stage", "oracle").factory, StandInOracle)
    with pytest.raises((RecipeError, RunError), match="[Oo]racle"):
        run_oracle_recipe(tmp_path, registry, recipe_data(stages=STAGES))
    assert not (tmp_path / "runs-root").exists(), "a refused run creates no directory"


def test_the_runner_refuses_an_oracle_that_no_listed_stage_uses(tmp_path: Path) -> None:
    stage_class = DEFAULT_REGISTRY.get("Stage", "oracle").factory
    oracle_class = DEFAULT_REGISTRY.get("Oracle", "stdout_mask").factory
    registry = runner_registry(stage_class, oracle_class)
    with pytest.raises(RunError, match="sets oracle, but no listed stage uses an Oracle"):
        run_oracle_recipe(tmp_path, registry, recipe_data(oracle=ORACLE, stages=["generate", "compile_loop"]))
    assert not (tmp_path / "runs-root").exists(), "a refused run creates no directory"


def test_the_runner_refuses_an_oracle_section_without_passfail_before_any_directory(tmp_path: Path) -> None:
    stage_class = DEFAULT_REGISTRY.get("Stage", "oracle").factory
    oracle_class = DEFAULT_REGISTRY.get("Oracle", "stdout_mask").factory
    registry = runner_registry(stage_class, oracle_class)
    with pytest.raises(RunError, match="oracle.passfail is a required choice"):
        run_oracle_recipe(tmp_path, registry, recipe_data(oracle={"kind": "stdout_mask"}, stages=STAGES))
    assert not (tmp_path / "runs-root").exists(), "a refused run creates no directory"


def test_a_compile_only_run_with_the_oracle_stage_records_no_alignment(tmp_path: Path) -> None:
    stage_class = DEFAULT_REGISTRY.get("Stage", "oracle").factory
    oracle_class = DEFAULT_REGISTRY.get("Oracle", "stdout_mask").factory
    registry = runner_registry(stage_class, oracle_class)
    run_dir = run_oracle_recipe(tmp_path, registry, recipe_data(oracle=ORACLE, stages=STAGES))
    trial = read_trial(trial_dir(run_dir, TRIAL_ID), TextStore(run_dir))
    assert trial.attempts, "the mock's reply is one attempt"
    assert [attempt.alignment for attempt in trial.attempts] == [Alignment()] * len(trial.attempts)
    assert trial.final.alignment is None
