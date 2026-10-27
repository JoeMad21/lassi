"""Tests for the human start of a silicon run: `lassi run --silicon` (task P17.16).

Bible: Agent Rules 3, 9, and 10 (silicon runs are evaluation runs a human
starts; RL rewards never execute on silicon); Execution Backends (the
tt_silicon row); Training Module, Safeguards (rewards never execute on
Tenstorrent silicon); Result Record (provenance). OQ-075 (6a), approved:
"a command-line flag a recipe cannot set, never passed by an unattended
session"; plans/p17-portable.md, task P17.16, Human start.

The contract these tests fix (lassi.core.runner and lassi.cli):

- RunOptions gains `silicon`, a bool that defaults to False. Only the
  command line sets it: `lassi run <recipe> --silicon`. No other command
  takes the flag but `lassi train` (task P17.38, OQ-100 (a): a training
  run on a card; tests/train/test_train_silicon_guard.py), no environment
  variable stands in for it, and a recipe
  cannot set it: a `silicon` key at the top level or in an executor section
  is refused at load as an unknown key.
- The runner reads capabilities, never names (Agent Rule 3). An executor
  that declares `silicon` (lassi.core.capabilities SILICON, proposed in
  tests/executors/test_tt_silicon.py), bound in the single form or for any
  language, is refused with a RunError naming --silicon and Agent Rule 9
  unless RunOptions.silicon is True. A listed stage that declares
  `computes_reward` (lassi.core.capabilities COMPUTES_REWARD, a name
  proposed here for the reward stages of later phases) refuses such an
  executor outright, with the flag too, naming the stage and Agent Rule 9.
  Both refusals come before the executor is built and before any directory
  exists, so nothing reads the cards' files in a run a human did not start.
  An executor named tt_silicon that does not declare the capability is not
  refused.
- Until OQ-078 is answered (its recommendation (c): under the cards'
  identity IOMMU domain the sandbox does not isolate host memory from a
  card's DMA, so silicon runs only tracked references a human starts;
  plans/spikes/tt-card-dma.md), such an executor is refused, with the flag
  too, beside any listed stage that runs model-generated code (it declares
  `runs_model_code`, as run_loop does) or an adversary's proposals (it
  declares `runs_proposals`, lassi.core.adversary_stage RUNS_PROPOSALS),
  naming OQ-078 and the stage, at the same point and before anything is
  built. The runner reads the attribute and the capability, never a stage's
  name: a stage named run_loop without the attribute is not refused, and a
  reference-only recipe (baseline, generate, compile_loop) runs its
  reference on the silicon executor with the flag.
- With the flag, the run goes ahead, and the executor's device(), for
  tt_silicon "Tenstorrent Wormhole n300 x <N> (silicon)" with every card's
  serial and firmware and the driver, is provenance.json's device and every
  trial's provenance.device, as for any executor; run.md writes no
  simulator note for it.

Every component is a fake in a test Registry beside the real generate and
compile_loop stages (tests/core/test_runner_devices.py's fakes), except in
the tests that bind the real tt_silicon executor, built on a SYNTHETIC host
tree (tests/executors/tt_silicon_fakes.py) with an exploding sandbox: the
stages here compile only, so no program runs. No card, node, or process is
touched. No value in this module is a measurement.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from test_runner_devices import (
    CUDA_SOURCE,
    FAKE_COMMIT,
    OMP_SOURCE,
    Log,
    compile_only_executor,
    fake_toolchain,
    manifest,
    plain_backend,
    read_ascii,
    recipe_data,
    trial_id,
    write_recipe,
)

from lassi import cli
from lassi.core import runner as runner_module
from lassi.core.interfaces import Limits, RunResult
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.record import Trial
from lassi.core.registry import Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import BaselineStage, CompileLoopStage, GenerateStage, RunLoopStage
from lassi.core.store import TextStore, read_trial, trial_dir

EXECUTORS_TESTS = Path(__file__).resolve().parents[1] / "executors"
if str(EXECUTORS_TESTS) not in sys.path:
    sys.path.insert(0, str(EXECUTORS_TESTS))
from tt_silicon_fakes import (  # noqa: E402
    OWN_UID,
    PRIVATE_PER_RUN,
    Access,
    Clock,
    FakeFlock,
    Owners,
    serial,
    write_host,
)

# The capability literals, so this module collects before lassi.core.capabilities defines them.
SILICON = "silicon"
COMPUTES_REWARD = "computes_reward"
RUNS_PROPOSALS = "runs_proposals"
SILICON_DEVICE = "SYNTHETIC Tenstorrent Wormhole n300 x 8 (silicon)"
# A fixture wall time, not a measurement.
REFERENCE_WALL_S = 1.25
REAL_SECTION = {"kind": "tt_silicon", "hugepages": PRIVATE_PER_RUN, "reset_filter": True, "device_profiler": "off"}


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make tmp_path the scratch root, point TMPDIR inside it, unset the other gate variables, and fake git."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "LASSI_SILICON"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path))
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setenv("LASSI_GRAPHICS", "off")
    monkeypatch.setenv("LASSI_CONFIG_DIR", str(tmp_path / "config"))
    answers = {"rev-parse": FAKE_COMMIT + "\n", "status": ""}
    monkeypatch.setattr(runner_module, "_git", lambda *args: answers[args[0]])


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """Write the SYNTHETIC layout sources where the suite manifest lays them out; return the bench root."""
    root = tmp_path / "bench"
    for relative, text in (("src/layout-omp/main.cpp", OMP_SOURCE), ("src/layout-cuda/main.cu", CUDA_SOURCE)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


# ---------------------------------------------------------------------------
# Fakes


def silicon_executor(
    log: Log, name: str = "silexec", capabilities: Sequence[str] = ("runs_code", "sandboxed", SILICON)
) -> type:
    """Return an Executor class `name` with `capabilities`; it records its build and runs only reference programs."""

    class SiliconFake:
        """A fake executor that names a SYNTHETIC silicon device; only a baseline reference runs on it."""

        def __init__(self, **config: Any) -> None:
            """Record the build."""
            log.events.append(f"build {name}")

        def device(self) -> str:
            """Return the SYNTHETIC device line."""
            return SILICON_DEVICE

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Return a clean SYNTHETIC run of a reference (built under baseline-<language>); fail on anything else."""
            if not Path(artifact).parent.parent.name.startswith("baseline-"):
                raise AssertionError(f"a program that is no reference ran on the silicon fake: {artifact}")
            log.events.append("run reference")
            return RunResult(exit_code=0, hang=False, stdout="SYNTHETIC\n", stderr="", wall_s=REFERENCE_WALL_S)

    SiliconFake.name = name  # type: ignore[attr-defined]
    SiliconFake.capabilities = frozenset(capabilities)  # type: ignore[attr-defined]
    SiliconFake.config_keys = frozenset()  # type: ignore[attr-defined]
    return SiliconFake


def reward_stage(log: Log) -> type:
    """Return a Stage class "reward_fixture" that declares computes_reward; building it fails the test."""

    class RewardFixture:
        """A fake reward stage of a later phase; the runner refuses before any stage is built."""

        name = "reward_fixture"
        capabilities = frozenset({COMPUTES_REWARD})
        config_keys: frozenset[str] = frozenset()

        def __init__(self, **kwargs: Any) -> None:
            """Fail the test: a refused run builds no stage."""
            log.events.append("build reward_fixture")
            raise AssertionError("the reward stage was built")

    return RewardFixture


def refused_stage(log: Log, name: str, capabilities: Sequence[str] = (), runs_model_code: bool = False) -> type:
    """Return a Stage class `name` with `capabilities` and, when asked, runs_model_code; building it fails the test."""

    class RefusedFixture:
        """A fake stage that runs model output (attempts or proposals); the runner refuses before it is built."""

        config_keys: frozenset[str] = frozenset()

        def __init__(self, **kwargs: Any) -> None:
            """Fail the test: a refused run builds no stage."""
            log.events.append(f"build {name}")
            raise AssertionError(f"the stage {name} was built")

    RefusedFixture.name = name  # type: ignore[attr-defined]
    RefusedFixture.capabilities = frozenset(capabilities)  # type: ignore[attr-defined]
    if runs_model_code:
        RefusedFixture.runs_model_code = True  # type: ignore[attr-defined]
    return RefusedFixture


def passing_stage(log: Log, name: str) -> type:
    """Return a Stage class `name` that declares nothing and returns each trial unchanged."""

    class PassingFixture:
        """A fake stage with a model-code stage's name but no runs_model_code; it changes no trial."""

        capabilities: frozenset[str] = frozenset()
        config_keys: frozenset[str] = frozenset()

        def __init__(self, *, context: Any) -> None:
            """Record the build."""
            log.events.append(f"build {name}")

        def __call__(self, trial: Trial) -> Trial:
            """Return the trial unchanged."""
            return trial

        def describe(self) -> str:
            """Return the run.md line."""
            return f"{name}: a SYNTHETIC stage that changes nothing"

    PassingFixture.name = name  # type: ignore[attr-defined]
    return PassingFixture


def make_registry(log: Log, *extra: tuple[str, str, type]) -> Registry:
    """Return a test Registry: the plain backend, the fake toolchain, the fake executors, the stages, and `extra`."""
    registry = Registry()
    registry.register("LLMBackend", "plainllm", plain_backend(log))
    registry.register("Toolchain", "nvcc-sm80", fake_toolchain(log))
    registry.register("Executor", "none", compile_only_executor(log, takes_device=False))
    registry.register("Executor", "silexec", silicon_executor(log))
    registry.register("Executor", "tt_silicon", silicon_executor(log, "tt_silicon", ("runs_code", "sandboxed")))
    registry.register("Stage", "baseline", BaselineStage)
    registry.register("Stage", "generate", GenerateStage)
    registry.register("Stage", "compile_loop", CompileLoopStage)
    registry.register("Stage", "reward_fixture", reward_stage(log))
    registry.register("Stage", "model_code_fixture", refused_stage(log, "model_code_fixture", runs_model_code=True))
    registry.register("Stage", "proposals_fixture", refused_stage(log, "proposals_fixture", [RUNS_PROPOSALS]))
    for interface, name, cls in extra:
        registry.register(interface, name, cls)
    return registry


def data(executor: Mapping[str, Any], stages: Sequence[str] = ("generate", "compile_loop")) -> dict[str, Any]:
    """Return a compile-only recipe for the layout item, omp to cuda, binding `executor` and listing `stages`."""
    found = recipe_data(executor=executor, backend="plainllm")
    found["stages"] = list(stages)
    return found


def options(tmp_path: Path, bench: Path, registry: Registry, **changes: Any) -> RunOptions:
    """Return RunOptions for a test run under tmp_path/runs-root, changed by `changes`."""
    return RunOptions(
        runs_root=tmp_path / "runs-root", run_id="silicon-run", bench_root=bench, registry=registry, **changes
    )


def refused(tmp_path: Path, recipe: Path, run_options: RunOptions) -> str:
    """Run a recipe the runner must refuse; assert no directory exists under the runs root; return the message."""
    with pytest.raises(RunError) as info:
        run_recipe(recipe, run_options)
    assert not (tmp_path / "runs-root").exists(), "a refusal comes before any directory exists"
    return str(info.value)


def tt_silicon_module() -> ModuleType:
    """Import lassi.executors.tt_silicon, failing the test clearly while it does not exist."""
    try:
        return importlib.import_module("lassi.executors.tt_silicon")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.executors.tt_silicon does not exist yet (task P17.16): {error}")


class ExplodingSandbox:
    """A stand-in sandbox: any use of it fails the test."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the sandbox was used ({name})")


def seams(root: Path) -> dict[str, Any]:
    """Return the keyword seams that put a TtSiliconExecutor on the SYNTHETIC host tree at `root`."""
    clock = Clock()
    return {
        "root": root, "access": Access(root), "owner": Owners(root), "uid": OWN_UID, "clock": clock,
        "sleep": clock.sleep, "flock": FakeFlock(), "sandbox": ExplodingSandbox(),
    }  # fmt: skip


def on_the_fake_host(module: ModuleType, root: Path, log: Log) -> type:
    """Return a subclass of the real TtSiliconExecutor that the runner builds on the SYNTHETIC host at `root`."""
    base = module.TtSiliconExecutor

    class OnTheFakeHost(base):  # type: ignore[misc, valid-type]
        """The real executor, built on the SYNTHETIC host tree; it records its build."""

        def __init__(self, **config: Any) -> None:
            """Record the build, then build the real executor with the recipe's config and the test's seams."""
            log.events.append("build tt_silicon")
            super().__init__(**config, **seams(root))

    return OnTheFakeHost


# ---------------------------------------------------------------------------
# The option, the capability names, and the documentation


def test_run_options_carry_a_silicon_flag_that_defaults_to_false() -> None:
    fields = {item.name: item for item in dataclasses.fields(RunOptions)}
    assert "silicon" in fields, "RunOptions needs `silicon` (task P17.16)"
    assert RunOptions().silicon is False


def test_the_capability_names_live_in_lassi_core_capabilities() -> None:
    from lassi.core import capabilities

    assert getattr(capabilities, "SILICON", None) == SILICON
    assert getattr(capabilities, "COMPUTES_REWARD", None) == COMPUTES_REWARD


def test_the_runner_and_the_cli_document_the_human_start() -> None:
    for module in (runner_module, cli):
        doc = module.__doc__ or ""
        assert "--silicon" in doc and "Agent Rule 9" in doc, module.__name__


# ---------------------------------------------------------------------------
# The runner: refused without the flag, by capability, before anything is built


def test_a_silicon_executor_is_refused_without_the_flag_before_it_is_built(tmp_path: Path, bench: Path) -> None:
    log = Log()
    recipe = write_recipe(tmp_path, "unstarted", data({"kind": "silexec"}))
    message = refused(tmp_path, recipe, options(tmp_path, bench, make_registry(log)))
    assert "--silicon" in message and "Agent Rule 9" in message, message
    assert "build silexec" not in log.events, "nothing reads the cards' files in a run a human did not start"


def test_a_silicon_executor_bound_for_one_language_is_refused_too(tmp_path: Path, bench: Path) -> None:
    log = Log()
    recipe = write_recipe(tmp_path, "per-language", data({"cuda": "silexec", "omp": "none"}))
    message = refused(tmp_path, recipe, options(tmp_path, bench, make_registry(log)))
    assert "executor.cuda" in message and "--silicon" in message, message
    assert "build silexec" not in log.events


def test_the_refusal_reads_the_capability_never_the_name(tmp_path: Path, bench: Path) -> None:
    log = Log()
    registry = make_registry(log, ("Executor", "anyname", silicon_executor(log, "anyname")))
    renamed = write_recipe(tmp_path, "renamed", data({"kind": "anyname"}))
    message = refused(tmp_path, renamed, options(tmp_path, bench, registry))
    assert "--silicon" in message, message
    named = write_recipe(tmp_path, "named", data({"kind": "tt_silicon"}))
    run_dir = run_recipe(named, options(tmp_path, bench, make_registry(Log())))
    assert manifest(run_dir)["executor"] == "tt_silicon", "a name without the capability is not refused (Agent Rule 3)"


def test_with_the_flag_the_run_goes_ahead_and_records_the_device(tmp_path: Path, bench: Path) -> None:
    log = Log()
    recipe = write_recipe(tmp_path, "started", data({"kind": "silexec"}))
    run_dir = run_recipe(recipe, options(tmp_path, bench, make_registry(log), silicon=True))
    assert "build silexec" in log.events
    assert manifest(run_dir)["device"] == SILICON_DEVICE
    trial = read_trial(trial_dir(run_dir, trial_id("started")), TextStore(run_dir))
    assert trial.provenance.device == SILICON_DEVICE


def test_a_reward_stage_refuses_a_silicon_executor_even_with_the_flag(tmp_path: Path, bench: Path) -> None:
    log = Log()
    stages = ("generate", "compile_loop", "reward_fixture")
    recipe = write_recipe(tmp_path, "rewarded", data({"kind": "silexec"}, stages))
    message = refused(tmp_path, recipe, options(tmp_path, bench, make_registry(log), silicon=True))
    assert "reward_fixture" in message and "Agent Rule 9" in message, message
    assert "build silexec" not in log.events and "build reward_fixture" not in log.events


# ---------------------------------------------------------------------------
# OQ-078: no model output on silicon until the owner answers, with the flag too


@pytest.mark.parametrize("stage", ["model_code_fixture", "proposals_fixture", "run_loop"])
@pytest.mark.parametrize("started", [True, False], ids=["with-flag", "without-flag"])
def test_a_stage_that_runs_model_output_refuses_a_silicon_executor_naming_oq_078(
    tmp_path: Path, bench: Path, stage: str, started: bool
) -> None:
    log = Log()
    registry = make_registry(log, ("Stage", "run_loop", RunLoopStage))
    stages = ("generate", "compile_loop", stage)
    recipe = write_recipe(tmp_path, f"model-output-{stage}", data({"kind": "silexec"}, stages))
    message = refused(tmp_path, recipe, options(tmp_path, bench, registry, silicon=started))
    assert "OQ-078" in message and repr(stage) in message, message
    assert "build silexec" not in log.events and f"build {stage}" not in log.events, log.events


def test_the_oq_078_refusal_covers_a_silicon_executor_bound_for_one_language(tmp_path: Path, bench: Path) -> None:
    log = Log()
    registry = make_registry(log, ("Stage", "run_loop", RunLoopStage))
    stages = ("generate", "compile_loop", "run_loop")
    recipe = write_recipe(tmp_path, "model-output-per-language", data({"cuda": "silexec", "omp": "none"}, stages))
    message = refused(tmp_path, recipe, options(tmp_path, bench, registry, silicon=True))
    assert "OQ-078" in message and "executor.cuda" in message, message
    assert "build silexec" not in log.events


def test_the_oq_078_refusal_reads_the_attribute_never_the_stage_name(tmp_path: Path, bench: Path) -> None:
    log = Log()
    registry = make_registry(log, ("Stage", "run_loop", passing_stage(log, "run_loop")))
    stages = ("generate", "compile_loop", "run_loop")
    recipe = write_recipe(tmp_path, "named-run-loop", data({"kind": "silexec"}, stages))
    run_dir = run_recipe(recipe, options(tmp_path, bench, registry, silicon=True))
    assert manifest(run_dir)["device"] == SILICON_DEVICE
    assert "build run_loop" in log.events, "a stage named run_loop that runs no model code is not refused"


def test_a_reference_only_recipe_runs_its_reference_on_silicon_with_the_flag(tmp_path: Path, bench: Path) -> None:
    log = Log()
    found = data({"kind": "silexec"}, ("baseline", "generate", "compile_loop"))
    found["fixes"] = {"baseline_both": False}  # the target reference only: the fake toolchain builds cuda alone
    recipe = write_recipe(tmp_path, "references", found)
    run_dir = run_recipe(recipe, options(tmp_path, bench, make_registry(log), silicon=True))
    assert log.events.count("run reference") == 1, log.events
    trial = read_trial(trial_dir(run_dir, trial_id("references")), TextStore(run_dir))
    assert trial.reference_run.exit_code == 0, trial.reference_run
    assert trial.provenance.device == SILICON_DEVICE


def test_the_flag_never_comes_from_the_environment(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("LASSI_SILICON", "LASSI_SILICON_TESTS", "LASSI_RUN_SILICON"):
        monkeypatch.setenv(name, "1")
    recipe = write_recipe(tmp_path, "environment", data({"kind": "silexec"}))
    assert "--silicon" in refused(tmp_path, recipe, options(tmp_path, bench, make_registry(Log())))


def test_a_recipe_cannot_set_the_flag(tmp_path: Path) -> None:
    module = tt_silicon_module()
    registry = make_registry(Log(), ("Executor", "tt_silicon_real", module.TtSiliconExecutor))
    top = data({"kind": "silexec"})
    top["silicon"] = True
    with pytest.raises(RecipeError) as caught:
        load_recipe(write_recipe(tmp_path, "top-key", top), registry=registry)
    assert "silicon" in str(caught.value), str(caught.value)
    section = data({**REAL_SECTION, "kind": "tt_silicon_real", "silicon": True})
    with pytest.raises(RecipeError) as caught:
        load_recipe(write_recipe(tmp_path, "section-key", section), registry=registry)
    assert "executor.silicon" in str(caught.value), str(caught.value)


# ---------------------------------------------------------------------------
# The real tt_silicon executor through the runner, on a SYNTHETIC host


def test_the_tt_silicon_executor_is_refused_without_the_flag_and_never_built(tmp_path: Path, bench: Path) -> None:
    module, log = tt_silicon_module(), Log()
    host = write_host(tmp_path / "host")
    registry = make_registry(log, ("Executor", "tt_silicon_real", on_the_fake_host(module, host.root, log)))
    recipe = write_recipe(tmp_path, "real-unstarted", data({**REAL_SECTION, "kind": "tt_silicon_real"}))
    message = refused(tmp_path, recipe, options(tmp_path, bench, registry))
    assert "--silicon" in message, message
    assert "build tt_silicon" not in log.events


def test_the_tt_silicon_executor_records_every_card_in_provenance_and_each_trial(tmp_path: Path, bench: Path) -> None:
    module, log = tt_silicon_module(), Log()
    host = write_host(tmp_path / "host")
    registry = make_registry(log, ("Executor", "tt_silicon_real", on_the_fake_host(module, host.root, log)))
    recipe = write_recipe(tmp_path, "real-started", data({**REAL_SECTION, "kind": "tt_silicon_real"}))
    run_dir = run_recipe(recipe, options(tmp_path, bench, registry, silicon=True))
    expected = module.TtSiliconExecutor(
        hugepages=PRIVATE_PER_RUN, reset_filter=True, device_profiler="off", **seams(host.root)
    ).device()
    written = manifest(run_dir)
    assert written["device"] == expected
    assert expected.startswith("Tenstorrent Wormhole n300 x 8 (silicon)")
    assert all(serial(card) in expected for card in host.cards)
    trial = read_trial(trial_dir(run_dir, trial_id("real-started")), TextStore(run_dir))
    assert trial.provenance.device == expected
    raw = json.loads(read_ascii(trial_dir(run_dir, trial_id("real-started")) / "trial.json"))
    assert raw["provenance"]["device"] == expected
    page = read_ascii(run_dir / "run.md")
    assert expected in page and "simulator wall time" not in page, "silicon is never labeled a simulator"


# ---------------------------------------------------------------------------
# The command line: `lassi run --silicon`, and no other way


def test_the_run_command_takes_the_silicon_flag() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["run", "recipe.yaml", "--silicon"]).silicon is True
    assert parser.parse_args(["run", "recipe.yaml"]).silicon is False


@pytest.mark.parametrize("given", [True, False], ids=["with-flag", "without-flag"])
def test_the_command_line_passes_the_flag_to_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, given: bool
) -> None:
    monkeypatch.setenv("LASSI_SILICON", "1")
    seen: list[RunOptions] = []
    monkeypatch.setattr(cli, "run_recipe", lambda path, run_options: seen.append(run_options))
    argv = ["--graphics", "off", "run", str(tmp_path / "recipe.yaml"), *(["--silicon"] if given else [])]
    assert cli.main(argv) == 0
    (passed,) = seen
    assert passed.silicon is given, "only the flag sets it; no variable stands in for it"


@pytest.mark.parametrize("command", [["score", "run-dir", "--profile", "lassi"], ["corpus", "summary", "dir"]])
def test_no_other_command_takes_the_flag(command: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    # `lassi train` takes it too since task P17.38 (OQ-100 (a)): tests/train/test_train_silicon_guard.py.
    with pytest.raises(SystemExit) as exit_:
        cli.main(["--graphics", "off", *command, "--silicon"])
    assert exit_.value.code == 2
    assert "--silicon" in capsys.readouterr().err


def _through_the_cli(monkeypatch: pytest.MonkeyPatch, registry: Registry, bench: Path) -> list[RunOptions]:
    """Make cli.run_recipe the real runner with the test's registry and bench root; return the options it saw."""
    seen: list[RunOptions] = []

    def wrapped(path: Path, run_options: RunOptions) -> Path:
        seen.append(run_options)
        return run_recipe(path, dataclasses.replace(run_options, registry=registry, bench_root=bench))

    monkeypatch.setattr(cli, "run_recipe", wrapped)
    return seen


def test_lassi_run_without_the_flag_exits_two_naming_it(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    log = Log()
    _through_the_cli(monkeypatch, make_registry(log), bench)
    recipe = write_recipe(tmp_path, "cli-unstarted", data({"kind": "silexec"}))
    argv = ["--graphics", "off", "run", str(recipe), "--runs-root", str(tmp_path / "runs-root"), "--run-id", "cli"]
    assert cli.main(argv) == 2
    assert "--silicon" in capsys.readouterr().err
    assert not (tmp_path / "runs-root").exists() and "build silexec" not in log.events


def test_lassi_run_with_the_flag_runs_and_records_the_device(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _through_the_cli(monkeypatch, make_registry(Log()), bench)
    recipe = write_recipe(tmp_path, "cli-started", data({"kind": "silexec"}))
    runs_root = tmp_path / "runs-root"
    argv = ["--graphics", "off", "run", str(recipe), "--runs-root", str(runs_root), "--run-id", "cli", "--silicon"]
    assert cli.main(argv) == 0
    assert manifest(runs_root / "runs" / "cli")["device"] == SILICON_DEVICE
