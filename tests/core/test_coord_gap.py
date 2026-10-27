"""Tests for task P17.41's loop.coord_gap_as_error: ttsim's coord_to_tile gap read as an addressing error.

Bible: ttsim Facts (UnimplementedFunctionality is a simulator gap), Project
Recipes (Notes, the Loop opt-ins bullet, lassi.core.recipe LOOP_OPT_INS),
Design Principle 4 and Agent Rules 2 and 4. plans/spikes/mlir-issues.md,
Half 1, finding 2: every built ttkernel attempt of the window run addressed
DRAM at the NOC coordinate (0, 0), and ttsim stopped it with
`UnimplementedFunctionality: coord_to_tile: coord (0,0)`, which OQ-098's gap
rule read as a simulator gap, "not a model failure", though the cause was the
model's addressing.

The contract these tests fix:

1. loop.coord_gap_as_error is a bool, off when absent (the loader
   materializes nothing, so earlier recipes keep their hashes), refused at
   load set true under faithful: true, and refused by the runner set true
   with a fragment prompt set.
2. Off or absent, a coord_to_tile finding is a gap as before: the trial
   ends at sim-gap, or, under loop.correct_on_sim_gap, the gap is fed back
   with the gap text.
3. On, a run that stopped at a gap whose findings hold ttsim's
   coord_to_tile finding on a coordinate is a failed run: the attempt stays
   S4 with a run-error naming the coordinate as an addressing error, carries
   no sim-gap note, and run_loop asks for a correction with the addressing
   text (never the gap text), under the one count and cap; left at the cap
   the trial ends at correction-cap, not sim-gap. A gap of another function
   stays a gap.

The findings, replies, and runs are SYNTHETIC (tests/core/test_template_context.py's
fakes and test_correct_on_sim_gap.py's simulator fake); nothing here is a
simulator result.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from test_correct_on_sim_gap import (
    FRAGMENT_SET,
    FRAGMENTS,
    GAP_STATUS,
    GAP_STDERR,
    RUN_LOOP,
    SIM_EXECUTOR,
    gap_recipe,
    stages_of,
)
from test_template_context import (
    CPP,
    ITEM,
    MODEL_ID,
    PROMPTS,
    RECIPE_HASHES,
    REPO,
    SUITE,
    TT,
    Script,
    Values,
    doubled,
    load,
    make_assets,
    make_registry,
    program,
    recipe_data,
    refused,
    runs_root,
    scripted_executor,
    stage_context,
    user_prompts,
    write_recipe,
    write_suite,
    write_tree,
)

from lassi.core.interfaces import Limits, RunResult
from lassi.core.recipe import LOOP_OPT_INS, RecipeError, load_recipe
from lassi.core.record import Diagnostic, Trial, make_trial_id
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.stages import RunLoopStage, addressing_text, coord_finding, sim_gap_text
from lassi.core.store import TextStore, read_trial, trial_dir

NAME = "coord_gap_as_error"
KEY = f"loop.{NAME}"
GAP_CLASS = "UnimplementedFunctionality"
COORD = f"{GAP_CLASS}: coord_to_tile: coord (0,0)"
OTHER = f"{GAP_CLASS}: SYNTHETIC_function: SYNTHETIC construct the fake simulator does not implement"
GENERATE = "generate"


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "scratch" / "tmp").mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch" / "tmp"))


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """Return the directory every run of a test writes under."""
    return tmp_path / "scratch"


@pytest.fixture
def assets(base: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return the temporary assets root with a copy of tt-host-v0 (test_template_context.make_assets)."""
    return make_assets(base, monkeypatch)


@pytest.fixture
def bench(base: Path, monkeypatch: pytest.MonkeyPatch, assets: Path) -> Path:
    """Return the bench root of the SYNTHETIC suite, with the assets root in place."""
    return write_suite(base, monkeypatch)


@dataclass(frozen=True)
class CoordGap:
    """A SYNTHETIC attempt run that stops at a gap whose finding is `message` (ttsim's coord_to_tile by default)."""

    message: str = COORD

    def __call__(self, values: Values) -> tuple[int, str, dict[str, Values]]:
        """Return the run's exit status, its stderr, and no output file."""
        return GAP_STATUS, GAP_STDERR, {}


def finding(message: str) -> Diagnostic:
    """Return the run-stage finding a simulator executor reports with `message`, coded with its class."""
    return Diagnostic(stage="run", severity="error", code=message.split(":")[0], message=message)


def coord_registry(script: Script) -> Any:
    """Return test_template_context's registry plus SIM_EXECUTOR, a simulator fake that reports each CoordGap's gap.

    It is test_correct_on_sim_gap's gap_registry with the Gap behavior
    replaced by CoordGap; it starts no process.
    """
    registry = make_registry(script)
    plain = scripted_executor(script)

    class CoordExecutor(plain):  # type: ignore[misc, valid-type]
        """The SYNTHETIC simulator, reporting a CoordGap behavior's finding as a gap of its class."""

        name = SIM_EXECUTOR
        capabilities = frozenset({"runs_code", "sandboxed", "simulator"})

        def run(self, artifact: Path, inputs: Any, limits: Limits) -> RunResult:
            """Run the scripted behavior; a CoordGap's run carries its class and its finding, twice."""
            reference = Path(artifact).parent.parent.name.startswith("baseline-")
            behavior = None if reference else script.attempts[0]
            result = super().run(artifact, inputs, limits)
            if not isinstance(behavior, CoordGap):
                return result
            gap = behavior.message.split(":")[0]
            gapped = dataclasses.replace(result, sim_gap=gap, diagnostics=[finding(behavior.message)] * 2)
            script.runs[-1] = gapped
            return gapped

    registry.register("Executor", SIM_EXECUTOR, CoordExecutor)
    return registry


def run_coord(base: Path, bench: Path, script: Script, data: Mapping[str, Any], name: str) -> Trial:
    """Run recipe `data` as <base>/<name>.yaml on coord_registry; return trial 1."""
    options = RunOptions(runs_root=runs_root(base), run_id=name, bench_root=bench, registry=coord_registry(script))
    run_dir = run_recipe(write_recipe(base, data, name), options)
    trial_id = make_trial_id(name, MODEL_ID, SUITE, f"{CPP}-{TT}", ITEM, 1)
    return read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))


def codes(trial: Trial, index: int) -> list[str | None]:
    """Return the code of every diagnostic of attempt `index`, in order."""
    return [item.code for item in trial.attempts[index].diagnostics]


def a_run(message: str = COORD) -> RunResult:
    """Return a SYNTHETIC run stopped at a gap with `message` as its finding."""
    return RunResult(exit_code=GAP_STATUS, hang=False, stdout="", stderr=GAP_STDERR, output_files={}, wall_s=1.0,
                     sim_gap=message.split(":")[0], diagnostics=[finding(message), finding(message)])


# ---------------------------------------------------------------------------
# The recipe key (contract point 1)


def test_the_key_is_a_loop_opt_in() -> None:
    assert NAME in LOOP_OPT_INS and "coord_to_tile" in LOOP_OPT_INS[NAME]


def test_absent_the_key_adds_nothing(base: Path) -> None:
    recipe = load(base, recipe_data(loop={}))
    assert NAME not in recipe.canonical_yaml


@pytest.mark.parametrize("path", sorted(RECIPE_HASHES))
def test_a_committed_recipe_without_the_key_keeps_its_recipe_hash(path: str) -> None:
    recipe = load_recipe(REPO / path)
    assert NAME not in recipe.data["loop"]
    assert recipe.recipe_hash == RECIPE_HASHES[path]


def test_set_the_key_enters_the_hash(base: Path) -> None:
    hashes = set()
    for label, loop in (("plain", {}), ("on", {NAME: True}), ("off", {NAME: False})):
        (base / label).mkdir()
        hashes.add(load(base / label, recipe_data(loop=loop)).recipe_hash)
    assert len(hashes) == 3


@pytest.mark.parametrize("value", [None, 1, "true"], ids=repr)
def test_a_value_that_is_not_a_bool_is_refused(base: Path, value: Any) -> None:
    with pytest.raises(RecipeError, match=KEY.replace(".", r"\.")):
        load(base, recipe_data(loop={NAME: value}))


def test_the_key_is_refused_under_faithful(base: Path) -> None:
    with pytest.raises(RecipeError) as caught:
        load(base, recipe_data(loop={NAME: True}, faithful=True))
    assert KEY in str(caught.value) and "faithful: true" in str(caught.value)


def test_the_runner_refuses_the_key_with_a_fragment_prompt_set(base: Path, bench: Path, assets: Path) -> None:
    write_tree(assets / "prompts" / FRAGMENT_SET, FRAGMENTS)
    data = recipe_data(loop={NAME: True}, prompts=FRAGMENT_SET, stages=[GENERATE, "compile_loop"])
    data.pop("oracle")
    message = refused(base, bench, data)
    assert KEY in message and "fragment" in message, message


# ---------------------------------------------------------------------------
# The reading (contract points 2 and 3)


def test_coord_finding_reads_only_the_coord_to_tile_gap() -> None:
    found = coord_finding(a_run())
    assert found is not None and found.message == COORD
    assert coord_finding(a_run(OTHER)) is None, "another function's gap stays a gap"
    clean = dataclasses.replace(a_run(), sim_gap=None)
    assert coord_finding(clean) is None, "no gap, no addressing error"


def test_the_addressing_text_names_the_coordinate_and_the_bank_rule() -> None:
    text = addressing_text(a_run())
    for words in ("(0,0)", "addressing error", "no tile", "DRAM", "bank", "NOC coordinate", COORD):
        assert words in text, f"{words!r} missing from:\n{text}"
    assert "a limit of the simulator, not" in text or "not a limit of the simulator" in text, text
    assert text.count(COORD) == 1 and text.endswith(GAP_STDERR)
    assert text != sim_gap_text(a_run()) and all(ord(char) < 128 for char in text)


@pytest.mark.parametrize("loop", [{}, {NAME: False}], ids=["absent", "false"])
def test_off_a_coord_gap_ends_the_trial_at_sim_gap(base: Path, bench: Path, loop: Mapping[str, Any]) -> None:
    script = Script(replies=[program(0)], attempts=[CoordGap()])
    trial = run_coord(base, bench, script, gap_recipe(loop), "off")
    assert stages_of(trial) == [GENERATE]
    reason = trial.final.end_reason
    assert reason is not None and reason.code == "sim-gap", reason


def test_off_with_the_gap_correction_it_is_fed_back_as_a_gap(base: Path, bench: Path) -> None:
    script = Script(replies=[program(0), program(1)], attempts=[CoordGap(), doubled])
    trial = run_coord(base, bench, script, gap_recipe({"correct_on_sim_gap": True}), "gap")
    assert stages_of(trial) == [GENERATE, RUN_LOOP]
    assert "sim-gap" in codes(trial, 0) and "run-error" not in codes(trial, 0)
    assert sim_gap_text(script.runs[0]) in user_prompts(script)[1]


@pytest.mark.parametrize("loop", [{NAME: True}, {NAME: True, "correct_on_sim_gap": True}], ids=["alone", "with-gap"])
def test_on_a_coord_gap_is_a_failed_run_fed_back_as_an_addressing_error(
    base: Path, bench: Path, loop: Mapping[str, Any]
) -> None:
    script = Script(replies=[program(0), program(1)], attempts=[CoordGap(), doubled])
    trial = run_coord(base, bench, script, gap_recipe(loop), "on")
    assert stages_of(trial) == [GENERATE, RUN_LOOP]
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S4", "S5"]
    assert trial.final.end_reason is None and trial.final.corrections == 1
    assert "sim-gap" not in codes(trial, 0), "no gap note: the attempt failed"
    errors = [item for item in trial.attempts[0].diagnostics if item.code == "run-error"]
    assert len(errors) == 1 and "(0,0)" in errors[0].message and "addressing" in errors[0].message, errors
    prompt = user_prompts(script)[1]
    assert addressing_text(script.runs[0]) in prompt
    assert sim_gap_text(script.runs[0]) not in prompt


def test_on_a_gap_of_another_function_stays_a_gap(base: Path, bench: Path) -> None:
    script = Script(replies=[program(0)], attempts=[CoordGap(OTHER)])
    trial = run_coord(base, bench, script, gap_recipe({NAME: True}), "other")
    reason = trial.final.end_reason
    assert reason is not None and reason.code == "sim-gap", reason


def test_on_an_addressing_error_left_at_the_cap_ends_at_correction_cap(base: Path, bench: Path) -> None:
    script = Script(replies=[program(0), program(1)], attempts=[CoordGap(), CoordGap()])
    trial = run_coord(base, bench, script, gap_recipe({NAME: True, "max_corrections": 1}), "cap")
    reason = trial.final.end_reason
    assert reason is not None and reason.code == "correction-cap" and "run error" in reason.message, reason
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S4", "S4"]


def test_run_loop_describes_the_key(base: Path) -> None:
    executor = coord_registry(Script(replies=[], attempts=[])).get("Executor", SIM_EXECUTOR).factory()
    on = load(base, recipe_data(loop={NAME: True}), "on")
    text = RunLoopStage(context=stage_context(on, PROMPTS, executor=executor)).describe()
    assert KEY in text, text
