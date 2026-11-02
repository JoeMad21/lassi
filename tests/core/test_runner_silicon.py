"""Tests that a silicon executor runs like any other: no start flag, no refusal (tasks P17.16 and P17.53).

Bible: Execution Backends (the tt_silicon row); Component Interfaces, the
Executor bullet; Agent Rules 3 and 10; Result Record (provenance). The
owner's answers of 2026-10-29: to OQ-078, "Don't gate the DMA IOMMU feature,
remove all gates. I'm okay with you not allowing the runs, but you cannot
enforce gates on my software."; to OQ-099, "It should be like TT-SIM.
Silicon is there as a final test gate, no human's required."
plans/p17-portable.md, task P17.53.

The contract these tests fix (lassi.core.runner and lassi.cli):

- RunOptions has no `silicon` field: no human start exists. An executor
  that declares `silicon` (lassi.core.capabilities SILICON, the tt_silicon
  executor's), bound in the single form or for one language, is built and
  run like any other executor, without any flag, beside a stage that runs
  model-generated code (it declares `runs_model_code`, as run_loop does) and
  beside a stage that declares `computes_reward` (lassi.core.capabilities
  COMPUTES_REWARD, kept as the name of the reward stages of later phases).
  A stage that runs an adversary's proposals meets only the refusals every
  executor meets (here, the missing adversary section), none naming
  silicon, OQ-078, or Agent Rule 9.
- The executor's device(), for tt_silicon "Tenstorrent Wormhole n300 x <N>
  (silicon)" with every card's serial, firmware, IOMMU group and domain
  type, and the driver, is provenance.json's device and every trial's
  provenance.device, as for any executor; run.md writes no simulator note
  for it.
- No command takes --silicon: the flag is gone from `lassi run` and
  `lassi train`, so the argument parser refuses it as an unknown argument
  (exit 2), as it does for every other command, and a silicon recipe runs
  through `lassi run` without it. A `silicon` key at a recipe's top level
  or in an executor section is still refused at load as an unknown key,
  as any unknown key is.

Every component is a fake in a test Registry beside the real generate and
compile_loop stages (tests/core/test_runner_devices.py's fakes), except in
the test that binds the real tt_silicon executor, built on a SYNTHETIC host
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
from lassi.core.stages import BaselineStage, CompileLoopStage, GenerateStage
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
    Links,
    Owners,
    group,
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


def passing_stage(log: Log, name: str, capabilities: Sequence[str] = (), runs_model_code: bool = False) -> type:
    """Return a Stage class `name` with `capabilities` (and runs_model_code when asked) that changes no trial."""

    class PassingFixture:
        """A fake stage that records its build and each trial it sees, and returns each trial unchanged."""

        config_keys: frozenset[str] = frozenset()

        def __init__(self, *, context: Any) -> None:
            """Record the build."""
            log.events.append(f"build {name}")

        def __call__(self, trial: Trial) -> Trial:
            """Record the call and return the trial unchanged."""
            log.events.append(f"call {name}")
            return trial

        def describe(self) -> str:
            """Return the run.md line."""
            return f"{name}: a SYNTHETIC stage that changes nothing"

    PassingFixture.name = name  # type: ignore[attr-defined]
    PassingFixture.capabilities = frozenset(capabilities)  # type: ignore[attr-defined]
    if runs_model_code:
        PassingFixture.runs_model_code = True  # type: ignore[attr-defined]
    return PassingFixture


def make_registry(log: Log, *extra: tuple[str, str, type]) -> Registry:
    """Return a test Registry: the plain backend, the fake toolchain, the fake executors, the stages, and `extra`."""
    registry = Registry()
    registry.register("LLMBackend", "plainllm", plain_backend(log))
    registry.register("Toolchain", "nvcc-sm80", fake_toolchain(log))
    registry.register("Executor", "none", compile_only_executor(log, takes_device=False))
    registry.register("Executor", "silexec", silicon_executor(log))
    registry.register("Stage", "baseline", BaselineStage)
    registry.register("Stage", "generate", GenerateStage)
    registry.register("Stage", "compile_loop", CompileLoopStage)
    registry.register("Stage", "reward_fixture", passing_stage(log, "reward_fixture", [COMPUTES_REWARD]))
    registry.register("Stage", "model_code_fixture", passing_stage(log, "model_code_fixture", runs_model_code=True))
    registry.register("Stage", "proposals_fixture", passing_stage(log, "proposals_fixture", [RUNS_PROPOSALS]))
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


def tt_silicon_module() -> ModuleType:
    """Import lassi.executors.tt_silicon."""
    return importlib.import_module("lassi.executors.tt_silicon")


class ExplodingSandbox:
    """A stand-in sandbox: any use of it fails the test."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the sandbox was used ({name})")


def seams(root: Path, host: Any) -> dict[str, Any]:
    """Return the keyword seams that put a TtSiliconExecutor on the SYNTHETIC host tree at `root`."""
    clock = Clock()
    return {
        "root": root, "access": Access(root), "owner": Owners(root), "link": Links(host), "uid": OWN_UID,
        "clock": clock, "sleep": clock.sleep, "flock": FakeFlock(), "sandbox": ExplodingSandbox(),
    }  # fmt: skip


def on_the_fake_host(module: ModuleType, host: Any, log: Log) -> type:
    """Return a subclass of the real TtSiliconExecutor that the runner builds on the SYNTHETIC host `host`."""
    base = module.TtSiliconExecutor

    class OnTheFakeHost(base):  # type: ignore[misc, valid-type]
        """The real executor, built on the SYNTHETIC host tree; it records its build."""

        def __init__(self, **config: Any) -> None:
            """Record the build, then build the real executor with the recipe's config and the test's seams."""
            log.events.append("build tt_silicon")
            super().__init__(**config, **seams(host.root, host))

    return OnTheFakeHost


# ---------------------------------------------------------------------------
# No human start: no option, no flag, no refusal


def test_run_options_carry_no_silicon_flag() -> None:
    assert "silicon" not in {item.name for item in dataclasses.fields(RunOptions)}, "no human start (task P17.53)"


def test_the_capability_names_live_in_lassi_core_capabilities() -> None:
    from lassi.core import capabilities

    assert getattr(capabilities, "SILICON", None) == SILICON
    assert getattr(capabilities, "COMPUTES_REWARD", None) == COMPUTES_REWARD


def test_the_runner_and_the_cli_document_that_silicon_runs_like_any_executor() -> None:
    runner_doc, cli_doc = runner_module.__doc__ or "", cli.__doc__ or ""
    assert "P17.53" in runner_doc and "OQ-078" in runner_doc and "OQ-099" in runner_doc
    assert "--silicon" in cli_doc and "is gone" in cli_doc, "the flag is documented as removed"
    assert not hasattr(runner_module, "_check_silicon"), "the refusal is gone"


def test_a_silicon_executor_runs_without_any_flag_and_records_the_device(tmp_path: Path, bench: Path) -> None:
    log = Log()
    recipe = write_recipe(tmp_path, "started", data({"kind": "silexec"}))
    run_dir = run_recipe(recipe, options(tmp_path, bench, make_registry(log)))
    assert "build silexec" in log.events
    assert manifest(run_dir)["device"] == SILICON_DEVICE
    trial = read_trial(trial_dir(run_dir, trial_id("started")), TextStore(run_dir))
    assert trial.provenance.device == SILICON_DEVICE


def test_a_silicon_executor_bound_for_one_language_runs_too(tmp_path: Path, bench: Path) -> None:
    log = Log()
    recipe = write_recipe(tmp_path, "per-language", data({"cuda": "silexec", "omp": "none"}))
    run_recipe(recipe, options(tmp_path, bench, make_registry(log)))
    assert "build silexec" in log.events


@pytest.mark.parametrize("stage", ["model_code_fixture", "reward_fixture"])
def test_a_stage_that_runs_model_code_or_computes_a_reward_runs_beside_a_silicon_executor(
    tmp_path: Path, bench: Path, stage: str
) -> None:
    log = Log()
    recipe = write_recipe(tmp_path, f"beside-{stage}", data({"kind": "silexec"}, ("generate", "compile_loop", stage)))
    run_dir = run_recipe(recipe, options(tmp_path, bench, make_registry(log)))
    assert manifest(run_dir)["device"] == SILICON_DEVICE
    assert "build silexec" in log.events and f"call {stage}" in log.events, log.events


def test_a_proposals_stage_meets_only_the_refusals_every_executor_meets(tmp_path: Path, bench: Path) -> None:
    log = Log()
    stages = ("generate", "compile_loop", "proposals_fixture")
    recipe = write_recipe(tmp_path, "proposals", data({"kind": "silexec"}, stages))
    with pytest.raises(RunError) as caught:
        run_recipe(recipe, options(tmp_path, bench, make_registry(log)))
    message = str(caught.value)
    assert "adversary section" in message, message
    for word in ("silicon", "OQ-078", "Agent Rule 9", "human"):
        assert word not in message, (word, message)


def test_a_reference_only_recipe_runs_its_reference_on_silicon(tmp_path: Path, bench: Path) -> None:
    log = Log()
    found = data({"kind": "silexec"}, ("baseline", "generate", "compile_loop"))
    found["fixes"] = {"baseline_both": False}  # the target reference only: the fake toolchain builds cuda alone
    recipe = write_recipe(tmp_path, "references", found)
    run_dir = run_recipe(recipe, options(tmp_path, bench, make_registry(log)))
    assert log.events.count("run reference") == 1, log.events
    trial = read_trial(trial_dir(run_dir, trial_id("references")), TextStore(run_dir))
    assert trial.reference_run.exit_code == 0, trial.reference_run
    assert trial.provenance.device == SILICON_DEVICE


def test_a_recipe_key_named_silicon_is_still_an_unknown_key(tmp_path: Path) -> None:
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


def test_the_tt_silicon_executor_records_every_card_and_its_iommu_group_in_provenance_and_each_trial(
    tmp_path: Path, bench: Path
) -> None:
    module, log = tt_silicon_module(), Log()
    host = write_host(tmp_path / "host", iommu={0: "DMA-FQ"})
    registry = make_registry(log, ("Executor", "tt_silicon_real", on_the_fake_host(module, host, log)))
    stages = ("generate", "compile_loop", "model_code_fixture")
    recipe = write_recipe(tmp_path, "real", data({**REAL_SECTION, "kind": "tt_silicon_real"}, stages))
    run_dir = run_recipe(recipe, options(tmp_path, bench, registry))
    expected = module.TtSiliconExecutor(
        hugepages=PRIVATE_PER_RUN, reset_filter=True, device_profiler="off", **seams(host.root, host)
    ).device()
    written = manifest(run_dir)
    assert written["device"] == expected
    assert expected.startswith("Tenstorrent Wormhole n300 x 8 (silicon)")
    assert all(serial(card) in expected for card in host.cards)
    assert f"IOMMU group {group(0)} DMA-FQ" in expected and f"IOMMU group {group(1)} identity" in expected
    trial = read_trial(trial_dir(run_dir, trial_id("real")), TextStore(run_dir))
    assert trial.provenance.device == expected
    raw = json.loads(read_ascii(trial_dir(run_dir, trial_id("real")) / "trial.json"))
    assert raw["provenance"]["device"] == expected
    page = read_ascii(run_dir / "run.md")
    assert expected in page and "simulator wall time" not in page, "silicon is never labeled a simulator"


# ---------------------------------------------------------------------------
# The command line: --silicon is gone


def test_the_run_command_parses_without_any_silicon_flag() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["run", "recipe.yaml"]).command == "run"
    assert "silicon" not in vars(parser.parse_args(["run", "recipe.yaml"]))


def test_the_runner_sees_plain_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[RunOptions] = []
    monkeypatch.setattr(cli, "run_recipe", lambda path, run_options: seen.append(run_options))
    argv = ["--graphics", "off", "run", str(tmp_path / "recipe.yaml")]
    assert cli.main(argv) == 0
    (passed,) = seen
    plain = dataclasses.replace(passed, observer=None)
    assert plain == RunOptions(runs_root=None, run_id=None, bench_root=None), plain


@pytest.mark.parametrize(
    "command",
    [["run", "recipe.yaml"], ["train", "train.yaml"], ["score", "run-dir", "--profile", "lassi"],
     ["corpus", "summary", "dir"]],
)
def test_no_command_takes_the_flag(command: list[str], capsys: pytest.CaptureFixture[str]) -> None:
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


def test_lassi_run_runs_a_silicon_recipe_and_records_the_device(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _through_the_cli(monkeypatch, make_registry(Log()), bench)
    recipe = write_recipe(tmp_path, "cli", data({"kind": "silexec"}))
    runs_root = tmp_path / "runs-root"
    argv = ["--graphics", "off", "run", str(recipe), "--runs-root", str(runs_root), "--run-id", "cli"]
    assert cli.main(argv) == 0
    assert manifest(runs_root / "runs" / "cli")["device"] == SILICON_DEVICE
