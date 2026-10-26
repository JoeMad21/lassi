"""Tests for task P17.35: simulator gaps reported as not verifiable on ttsim, and fed back (loop.correct_on_sim_gap).

Bible: ttsim Facts (UnimplementedFunctionality and UnsupportedFunctionality
are simulator gaps, not model errors), Result Record (the end reason sim-gap,
Simulator runs, Attempt.run, Attempt.diagnostics, Trial.requests), Project
Recipes (Notes, the Loop opt-ins bullet, lassi.core.recipe LOOP_OPT_INS),
Evaluation Protocol (Reporting Rules), Design Principle 4 and Agent Rules 1,
2, 4, and 9. plans/p17-portable.md, task P17.35; OQ-098, simulator gaps,
option (a), the owner's answer of 2026-10-25: a gap is reported as built but
not verifiable on ttsim, never a fail, listed for a later human-started
hardware check, and, as a named opt-in, fed back under the same cap.

The contract these tests fix:

1. loop.correct_on_sim_gap is a bool. Absent, the loader materializes
   nothing, so a recipe without it keeps its resolved recipe and recipe hash
   (the committed recipes of tests/core/test_template_context.py, hashed
   before the key existed); set, it enters the hash as written. A value that
   is not a bool, and a null, are refused at load naming the key. Set true
   under faithful: true, in the recipe or inherited, it is refused at load
   naming the key and faithful: true; set false it loads.
2. The runner refuses it set true, before any directory exists, with a
   fragment prompt set: its feedback is this project's wording, which only
   a template set's correct.txt carries.
3. Off or absent, nothing changes: a run that stops at a simulator gap with
   no kernel JIT error ends the trial at sim-gap right after that attempt,
   with no correction, the message as before; the attempt stays S4 with its
   run recorded, no run-error, and no gap note.
4. On, the gap attempt keeps its record: S4, Attempt.run as the run gave
   it, no run-error, and one run-stage note, code sim-gap, naming the gap's
   class (the record's mark of an attempt built but not verifiable on the
   simulator). While loop.max_corrections leaves room, run_loop asks for a
   correction (Request stage run_loop) with correct.txt, whose diagnostics
   field is the gap text: the gap's class, that the simulator cannot run a
   construct the program uses, that this is a limit of the simulator and
   not necessarily an error in the program, and that an equivalent
   construct the simulator runs should be used if one exists; then each of
   the run's findings of that class once, then the stderr. Its reply is
   built, run, and corrected as any other, under the one count and cap; a
   gap left at the cap ends the trial at sim-gap (not correction-cap), the
   message naming the gap's class and the cap.
5. A kernel JIT error in the same run is still read first, with the key on.
6. The report (plans/runs/rngd-sweep/tabulate.py) keeps every existing
   column of trials.tsv and passes.tsv as it was and adds, last, a count of
   the attempts that hit a gap, apart from any failure; not_verifiable.tsv
   lists each such attempt (trial id, attempt index, gap class, stage), and
   hardware_check.tsv lists them for a later hardware check that a human
   starts (Agent Rule 9: the script runs nothing). An attempt is read as a
   gap attempt from its sim-gap note, or as the last attempt of a trial that
   ended at sim-gap; a failed run and a baseline gap are not listed.
7. The score profiles read the gap attempt run_loop records under the key
   as holding no evidence either way, though a later attempt follows it:
   df-v0 gives it no R, and the lassi profile leaves correct null with the
   gap note while its output stands (tests/scoring/test_sim_gap_scoring.py
   covers the readings).

The suite, programs, replies, findings, and arrays are SYNTHETIC, from
tests/core/test_template_context.py's fakes: the fake toolchain compiles
nothing, and the scripted executors start no process and open no device.
Wall times are PLACEHOLDER fixture values. No value here is a measurement,
and nothing here is a simulator result.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from test_template_context import (
    BROKEN,
    CPP,
    ITEM,
    MODEL_ID,
    PROMPTS,
    RECIPE_HASHES,
    REPO,
    RUN_STATUS,
    RUN_STDERR,
    SUITE,
    TT,
    Ran,
    Script,
    Values,
    crashed,
    doubled,
    golden,
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

from lassi.bench import Direction
from lassi.core import fragments
from lassi.core.interfaces import Limits, RunResult
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.record import Diagnostic, Trial, make_trial_id, to_dict
from lassi.core.registry import Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.stages import RunLoopStage, sim_gap_text
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.scoring.df_v0 import DfV0Profile
from lassi.scoring.lassi_profile import PROFILE_FILE, load_profile, outcome_components

KEY = "loop.correct_on_sim_gap"
NAME = "correct_on_sim_gap"
GENERATE, COMPILE_LOOP, RUN_LOOP = "generate", "compile_loop", "run_loop"
SIM_GAP = "sim-gap"
# The executor name of the SYNTHETIC simulator fake (gap_registry).
SIM_EXECUTOR = "scripted-sim"
GAP_CLASS = "UnimplementedFunctionality"
OTHER_GAP = "UnsupportedFunctionality"
GAP_STATUS = 134
GAP_STDERR = "SYNTHETIC TT_THROW: the run was stopped\n"
FINDING = f"{GAP_CLASS}: SYNTHETIC_function: SYNTHETIC construct the fake simulator does not implement"
JIT_ERROR = Diagnostic(stage="jit", severity="error", code="synthetic", message="SYNTHETIC kernel JIT error")
# The words the gap text must hold (contract point 4).
GAP_WORDS = (
    "simulator gap",
    "the simulator cannot run",
    "a limit of the simulator",
    "not necessarily an error in the program",
    "equivalent construct",
)
# The end message of a gap with the key off: the text run_loop gave before this task (contract point 3).
OFF_MESSAGE = (
    f"the run of attempt 0 stopped at a simulator gap, {GAP_CLASS}: the simulator cannot run the program, which is "
    "not a model error, so the trial stops with no correction"
)
# The run error text of a crashed run, as the golden run-correction prompt holds it.
CRASH_TEXT = f"the program exited with status {RUN_STATUS}; its standard error follows:\n{RUN_STDERR}"
# A SYNTHETIC fragment set holding every key generate and compile_loop read for cpp -> tt.
FRAGMENT_SET = "syn-gap-fragments"
FRAGMENTS = {
    fragments.fragment_key(key, Direction(CPP, TT)): f"<{key}>"
    for key in (*fragments.GENERATE_KEYS, *fragments.CORRECT_KEYS)
}
TABULATE = REPO / "plans" / "runs" / "rngd-sweep" / "tabulate.py"
# trials.tsv's columns before this task, which stay as they were (contract point 6).
OLD_TRIAL_COLUMNS = ("pass", "slug", "half", "direction", "item", "run", "stage", "alignment", "corrections", "end",
                     "attempts", "unfenced_warnings", "adv_proposals", "adv_refused", "adv_runs", "adv_divergences",
                     "combo")
OLD_PASS_COLUMNS = ("pass", "slug", "trials", "s5", "alignment_1", "with_divergence", "combo")


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
class Gap:
    """A SYNTHETIC attempt run that stops at a simulator gap of class `cls`, with exit status `status` and no output.

    The fake simulator (gap_registry) reports the gap in RunResult.sim_gap and
    the finding twice among RunResult.diagnostics, as a simulator may print
    one finding more than once; with `jit` it also reports a kernel JIT error.
    """

    cls: str = GAP_CLASS
    status: int = GAP_STATUS
    jit: bool = False

    def __call__(self, values: Values) -> Ran:
        """Return the run's exit status, its stderr, and no output file."""
        return self.status, GAP_STDERR, {}


def finding(cls: str) -> Diagnostic:
    """Return the SYNTHETIC run-stage finding a simulator executor reports for a gap of class `cls`."""
    return Diagnostic(stage="run", severity="error", code=cls, message=FINDING.replace(GAP_CLASS, cls))


def gap_registry(script: Script) -> Registry:
    """Return test_template_context's registry plus SIM_EXECUTOR: its scripted executor, declaring simulator.

    A Gap behavior's run comes back with sim_gap set to its class and its
    findings (finding, twice, and JIT_ERROR with `jit`); every other run is
    the scripted executor's own. It starts no process.
    """
    registry = make_registry(script)
    plain = scripted_executor(script)

    class GapExecutor(plain):  # type: ignore[misc, valid-type]
        """The scripted executor as a SYNTHETIC simulator that reports a gap for each Gap behavior."""

        name = SIM_EXECUTOR
        capabilities = frozenset({"runs_code", "sandboxed", "simulator"})

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Run the scripted behavior; a Gap's run carries its class and findings."""
            reference = Path(artifact).parent.parent.name.startswith("baseline-")
            behavior = None if reference else script.attempts[0]
            result = super().run(artifact, inputs, limits)
            if not isinstance(behavior, Gap):
                return result
            found = [finding(behavior.cls), finding(behavior.cls), *([JIT_ERROR] if behavior.jit else [])]
            gapped = dataclasses.replace(result, sim_gap=behavior.cls, diagnostics=found)
            script.runs[-1] = gapped
            return gapped

    registry.register("Executor", SIM_EXECUTOR, GapExecutor)
    return registry


def gap_recipe(loop: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    """Return test_template_context's cpp -> tt recipe on the fake simulator, with `loop` as its loop section."""
    return recipe_data(executor={"kind": SIM_EXECUTOR}, loop=dict(loop), **changes)


def run_gap(base: Path, bench: Path, script: Script, data: Mapping[str, Any], name: str) -> tuple[Path, Trial]:
    """Run recipe `data` as <base>/<name>.yaml on gap_registry; return the run directory and trial 1."""
    options = RunOptions(runs_root=runs_root(base), run_id=name, bench_root=bench, registry=gap_registry(script))
    run_dir = run_recipe(write_recipe(base, data, name), options)
    trial_id = make_trial_id(name, MODEL_ID, SUITE, f"{CPP}-{TT}", ITEM, 1)
    return run_dir, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))


def stages_of(trial: Trial) -> list[str]:
    """Return the stage name of every request of `trial`, in order."""
    return [request.stage for request in trial.requests or []]


def codes(trial: Trial, index: int) -> list[str | None]:
    """Return the code of every diagnostic of attempt `index`, in order."""
    return [item.code for item in trial.attempts[index].diagnostics]


def gap_notes(trial: Trial, index: int) -> list[Diagnostic]:
    """Return the sim-gap notes of attempt `index`."""
    return [item for item in trial.attempts[index].diagnostics if item.code == SIM_GAP]


def load_refusal(base: Path, data: Mapping[str, Any], name: str = "loaded") -> str:
    """Load `data`, expect a RecipeError from a loader that knows the key, and return its message."""
    with pytest.raises(RecipeError) as caught:
        load(base, data, name)
    message = str(caught.value)
    assert f"unknown key {KEY}" not in message, f"the loader does not know the key: {message}"
    return message


# ---------------------------------------------------------------------------
# The recipe key (contract point 1)


def test_absent_the_key_adds_nothing_to_the_resolved_recipe(base: Path) -> None:
    recipe = load(base, recipe_data(loop={}))
    assert recipe.data["loop"] == {"max_corrections": 10}
    assert NAME not in recipe.canonical_yaml


@pytest.mark.parametrize("path", sorted(RECIPE_HASHES))
def test_a_committed_recipe_without_the_key_keeps_its_recipe_hash(path: str) -> None:
    recipe = load_recipe(REPO / path)
    assert NAME not in recipe.data["loop"]
    assert recipe.recipe_hash == RECIPE_HASHES[path]


def test_set_the_key_loads_as_written_and_enters_the_hash(base: Path) -> None:
    hashes = {}
    for label, loop in (("plain", {}), ("on", {NAME: True}), ("off", {NAME: False})):
        (base / label).mkdir()
        recipe = load(base / label, recipe_data(loop=loop))
        assert recipe.data["loop"] == {"max_corrections": 10, **loop}
        hashes[label] = recipe.recipe_hash
    assert len(set(hashes.values())) == 3, "a key that is set enters the hash as written"


@pytest.mark.parametrize("value", [None, 1, 0, "true", [True], {"on": True}], ids=repr)
def test_a_value_that_is_not_a_bool_is_refused_naming_the_key(base: Path, value: Any) -> None:
    message = load_refusal(base, recipe_data(loop={NAME: value}))
    assert KEY in message and "loaded.yaml" in message, message


def test_the_key_is_refused_under_faithful(base: Path) -> None:
    message = load_refusal(base, recipe_data(loop={NAME: True}, faithful=True))
    assert KEY in message and "faithful: true" in message and "loaded.yaml" in message, message
    off = load(base, recipe_data(loop={NAME: False}, faithful=True), "off")
    assert off.data["loop"] == {"max_corrections": "uncapped", NAME: False}, "set false it loads"


def test_an_inherited_key_is_refused_when_the_child_turns_faithful_on(base: Path) -> None:
    load(base, recipe_data(loop={NAME: True}), "gap-parent")
    message = load_refusal(base, {"extends": "gap-parent.yaml", "faithful": True}, "child")
    assert KEY in message and "faithful: true" in message, message


# ---------------------------------------------------------------------------
# The runner's refusal with a fragment set (contract point 2)


def fragment_recipe(loop: Mapping[str, Any]) -> dict[str, Any]:
    """Return a generate and compile_loop recipe on the SYNTHETIC fragment set, with `loop`."""
    data = recipe_data(loop=dict(loop), prompts=FRAGMENT_SET, stages=[GENERATE, COMPILE_LOOP])
    data.pop("oracle")
    return data


def test_the_runner_refuses_the_key_with_a_fragment_prompt_set(base: Path, bench: Path, assets: Path) -> None:
    write_tree(assets / "prompts" / FRAGMENT_SET, FRAGMENTS)
    message = refused(base, bench, fragment_recipe({NAME: True}))
    assert KEY in message and "fragment" in message, message
    script = Script(replies=[program(0)], attempts=[])
    _, trial = run_gap(base, bench, script, fragment_recipe({NAME: False}), "upstream")
    assert stages_of(trial) == [GENERATE], "set false, the same recipe runs"


def test_run_loop_refuses_the_key_with_a_fragment_set_itself(base: Path) -> None:
    recipe = load(base, recipe_data(loop={NAME: True}))
    context = stage_context(recipe, FRAGMENT_SET, fragments=dict(FRAGMENTS))
    with pytest.raises(ValueError, match=KEY):
        RunLoopStage(context=context)


# ---------------------------------------------------------------------------
# Off or absent: nothing changes (contract point 3)


@pytest.mark.parametrize("loop", [{}, {NAME: False}], ids=["absent", "false"])
def test_without_the_key_a_gap_ends_the_trial_with_no_correction(
    base: Path, bench: Path, loop: Mapping[str, Any]
) -> None:
    script = Script(replies=[program(0)], attempts=[Gap()])
    _, trial = run_gap(base, bench, script, gap_recipe(loop), "off")
    assert stages_of(trial) == [GENERATE], "no correction is asked"
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S4"]
    reason = trial.final.end_reason
    assert reason is not None and reason.code == SIM_GAP and reason.message == OFF_MESSAGE
    assert codes(trial, 0) == [GAP_CLASS, GAP_CLASS], "the executor's findings only: no run-error and no gap note"
    assert trial.attempts[0].run.exit_code == GAP_STATUS
    assert script.attempts == [] and script.replies == []


def test_absent_and_false_record_the_same_attempts(base: Path, bench: Path) -> None:
    records = []
    for label, loop in (("absent", {}), ("false", {NAME: False})):
        (base / label).mkdir()
        script = Script(replies=[program(0)], attempts=[Gap()])
        _, trial = run_gap(base / label, bench, script, gap_recipe(loop), label)
        records.append(([to_dict(attempt) for attempt in trial.attempts], trial.final.end_reason))
    assert records[0] == records[1]


def test_without_the_key_run_loop_describes_itself_as_before(base: Path) -> None:
    executor = gap_registry(Script(replies=[], attempts=[])).get("Executor", SIM_EXECUTOR).factory()
    plain = load(base, recipe_data(loop={}), "plain")
    text = RunLoopStage(context=stage_context(plain, PROMPTS, executor=executor)).describe()
    assert NAME not in text and "gap" not in text, text
    on = load(base, recipe_data(loop={NAME: True}), "on")
    text_on = RunLoopStage(context=stage_context(on, PROMPTS, executor=executor)).describe()
    assert KEY in text_on and text_on.startswith(text.split(", sharing")[0]), text_on


# ---------------------------------------------------------------------------
# On: the record kept and the gap fed back (contract point 4)


def test_a_gap_is_fed_back_and_a_clean_run_follows(base: Path, bench: Path) -> None:
    script = Script(replies=[program(0), program(1)], attempts=[Gap(), doubled])
    _, trial = run_gap(base, bench, script, gap_recipe({NAME: True}), "fed")
    assert stages_of(trial) == [GENERATE, RUN_LOOP]
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S4", "S5"]
    assert trial.final.end_reason is None and trial.final.corrections == 1 and trial.final.alignment == 1.0
    assert script.attempts == [] and script.replies == []
    gap = trial.attempts[0]
    assert gap.run.exit_code == GAP_STATUS and gap.run.stdout_ref is not None, "the gap attempt keeps its run"
    assert "run-error" not in codes(trial, 0), "a gap is not a failed run"
    notes = gap_notes(trial, 0)
    assert len(notes) == 1 and notes[0].stage == "run" and notes[0].severity == "note", notes
    assert GAP_CLASS in notes[0].message and "not verifiable" in notes[0].message, notes[0].message
    assert gap_notes(trial, 1) == [], "the clean attempt carries no gap note"
    prompt = user_prompts(script)[1]
    text = sim_gap_text(script.runs[0])
    assert text in prompt and prompt.startswith("Your tt program failed."), prompt
    assert (trial.requests or [])[1].attempt_index == 1


def test_the_gap_text_names_the_class_and_the_simulator_limit() -> None:
    run = RunResult(exit_code=GAP_STATUS, hang=False, stdout="", stderr=GAP_STDERR, output_files={}, wall_s=1.0,
                    sim_gap=GAP_CLASS, diagnostics=[finding(GAP_CLASS), finding(GAP_CLASS), finding(OTHER_GAP)])
    text = sim_gap_text(run)
    assert GAP_CLASS in text
    for words in GAP_WORDS:
        assert words in text, f"{words!r} missing from:\n{text}"
    assert text.count(FINDING) == 1, "each finding of the gap's class once"
    assert FINDING.replace(GAP_CLASS, OTHER_GAP) not in text, "only the gap's own class"
    assert text.endswith(GAP_STDERR) and text.index(FINDING) < text.index(GAP_STDERR)
    bare = sim_gap_text(dataclasses.replace(run, stderr="", diagnostics=[]))
    assert GAP_CLASS in bare and "standard error" not in bare and FINDING not in bare, bare
    assert all(ord(char) < 128 for char in text)


def test_the_gap_correction_is_correct_txt_with_the_gap_text(base: Path, bench: Path) -> None:
    # The golden run-correction prompt is correct.txt for attempt 1 (program 1) after a crash; with a gap at the same
    # attempt the prompt is the same, the gap text in place of the crash's run error text.
    script = Script(replies=[BROKEN, program(1), program(2)], attempts=[Gap(), doubled])
    _, trial = run_gap(base, bench, script, gap_recipe({NAME: True}), "golden")
    assert stages_of(trial) == [GENERATE, COMPILE_LOOP, RUN_LOOP]
    expected = golden("run-correction").replace(CRASH_TEXT, sim_gap_text(script.runs[0]))
    assert CRASH_TEXT in golden("run-correction")
    assert user_prompts(script)[2] == expected


def test_gaps_and_failed_runs_share_one_count(base: Path, bench: Path) -> None:
    script = Script(replies=[program(0), program(1), program(2)], attempts=[Gap(), crashed, doubled])
    _, trial = run_gap(base, bench, script, gap_recipe({NAME: True, "max_corrections": 2}), "shared")
    assert stages_of(trial) == [GENERATE, RUN_LOOP, RUN_LOOP]
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S4", "S4", "S5"]
    assert "run-error" in codes(trial, 1) and gap_notes(trial, 1) == []
    assert trial.final.end_reason is None and trial.final.corrections == 2


def test_a_gap_attempt_counts_for_restarts_as_a_correction_that_did_not_reach_a_clean_run(
    base: Path, bench: Path
) -> None:
    # Correction 1 stops at a gap again: it did not reach a clean run, so with restart_after 1 the next request is a
    # restart, under the same count and cap.
    script = Script(replies=[program(0), program(1), program(2)], attempts=[Gap(), Gap(), doubled])
    loop = {NAME: True, "restart_after": 1, "max_corrections": 3}
    _, trial = run_gap(base, bench, script, gap_recipe(loop), "restart")
    assert stages_of(trial) == [GENERATE, RUN_LOOP, "restart"]
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S4", "S4", "S5"]
    assert trial.final.end_reason is None and trial.final.corrections == 2


@pytest.mark.parametrize("cap", [0, 1, 2])
def test_a_gap_left_at_the_cap_ends_the_trial_at_sim_gap(base: Path, bench: Path, cap: int) -> None:
    script = Script(replies=[program(number) for number in range(cap + 1)], attempts=[Gap()] * (cap + 1))
    _, trial = run_gap(base, bench, script, gap_recipe({NAME: True, "max_corrections": cap}), f"cap{cap}")
    assert stages_of(trial) == [GENERATE] + [RUN_LOOP] * cap
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S4"] * (cap + 1)
    assert all(len(gap_notes(trial, index)) == 1 for index in range(cap + 1)), "every gap attempt keeps its note"
    reason = trial.final.end_reason
    assert reason is not None and reason.code == SIM_GAP, reason
    assert GAP_CLASS in reason.message and "loop.max_corrections" in reason.message, reason.message
    assert f"attempt {cap}" in reason.message and reason.message != OFF_MESSAGE
    assert trial.final.corrections == cap
    assert script.attempts == [] and script.replies == []


def test_the_profiles_read_a_fed_back_gap_attempt_as_no_evidence(base: Path, bench: Path) -> None:
    # A gap, then a reply that does not build, at cap 1: compile_loop ends the trial at correction-cap with the gap
    # run's output standing. The profiles read the record run_loop wrote: the gap attempt holds no evidence either way.
    script = Script(replies=[program(0), BROKEN], attempts=[Gap()])
    _, trial = run_gap(base, bench, script, gap_recipe({NAME: True, "max_corrections": 1}), "scored")
    reason = trial.final.end_reason
    assert reason is not None and reason.code == "correction-cap", reason
    assert len(gap_notes(trial, 0)) == 1 and trial.attempts[1].stage_reached == "S1"
    profile = DfV0Profile()
    rs = [score.scalar for score in profile.score_attempts(trial)]
    assert rs[0] is None and rs[1] is not None, rs
    assert profile.score(trial).components["single_turn"] is None
    notes = dict(load_profile(PROFILE_FILE).notes)
    values, written = outcome_components(trial, notes)
    assert (values["correct"], values["first_try"]) == (None, None) and written["correct"] == notes["gap"], values


def test_a_kernel_jit_error_in_a_gap_run_is_still_corrected_first(base: Path, bench: Path) -> None:
    script = Script(replies=[program(0), program(1)], attempts=[Gap(jit=True), doubled])
    _, trial = run_gap(base, bench, script, gap_recipe({NAME: True}), "jit")
    assert stages_of(trial) == [GENERATE, COMPILE_LOOP]
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S1", "S5"]
    assert gap_notes(trial, 0) == [], "a JIT failure is read first, so the attempt is no gap attempt"
    assert "simulator gap" not in user_prompts(script)[1]


# ---------------------------------------------------------------------------
# The report (contract point 6)


def tabulate() -> ModuleType:
    """Return plans/runs/rngd-sweep/tabulate.py, loaded by path (it is a script, not a package module)."""
    spec = importlib.util.spec_from_file_location("rngd_sweep_tabulate", TABULATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tsv(path: Path) -> list[list[str]]:
    """Return the rows of the TSV file at `path`, its header first."""
    return [line.split("\t") for line in path.read_text(encoding="ascii").splitlines()]


def attempt(index: int, stage: str, *diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Return a SYNTHETIC attempt record as trial.json holds one, with only the fields the report reads."""
    return {"index": index, "stage_reached": stage, "diagnostics": [dict(item) for item in diagnostics]}


def note(cls: str) -> dict[str, Any]:
    """Return a SYNTHETIC sim-gap note naming `cls`, in run_loop's words under the key."""
    message = f"the run stopped at a simulator gap, {cls}: built, but not verifiable on the simulator"
    return {"stage": "run", "severity": "note", "code": SIM_GAP, "message": message}


RUN_ERROR = {"stage": "run", "severity": "error", "code": "run-error", "message": "the program exited with status 3"}


def trial_record(name: str, attempts: list[dict[str, Any]], end: tuple[str, str] | None) -> dict[str, Any]:
    """Return a SYNTHETIC trial record named `name` with `attempts` and the end reason (code, message) or none."""
    reason = None if end is None else {"code": end[0], "message": end[1]}
    stage = attempts[-1]["stage_reached"] if attempts else None
    return {"trial_id": f"syn/arm/{SUITE}/cpp-tt/{name}/run01", "attempts": attempts,
            "final": {"stage_reached": stage, "alignment": None, "corrections": max(len(attempts) - 1, 0),
                      "end_reason": reason}}


SYNTHETIC_TRIALS = {
    "off-gap": trial_record("off-gap", [attempt(0, "S1"), attempt(1, "S4")],
                            (SIM_GAP, OFF_MESSAGE.replace("attempt 0", "attempt 1"))),
    "fed-gap": trial_record("fed-gap", [attempt(0, "S4", note(OTHER_GAP)), attempt(1, "S5")], None),
    "cap-gap": trial_record("cap-gap", [attempt(0, "S4", note(GAP_CLASS)), attempt(1, "S4", note(GAP_CLASS))],
                            (SIM_GAP, f"the run of attempt 1 stopped at a simulator gap, {GAP_CLASS}, after 1 "
                                      "correction(s)")),
    "failed": trial_record("failed", [attempt(0, "S4", RUN_ERROR), attempt(1, "S4", RUN_ERROR)],
                           ("correction-cap", "a run error remained after 1 correction(s)")),
    "baseline-gap": trial_record("baseline-gap", [],
                                 (SIM_GAP, f"the tt reference run stopped at a simulator gap, {GAP_CLASS}: no model")),
}
# The gap attempts the report lists for SYNTHETIC_TRIALS: (item, attempt index, gap class, stage).
LISTED = [("cap-gap", "0", GAP_CLASS, "S4"), ("cap-gap", "1", GAP_CLASS, "S4"), ("fed-gap", "0", OTHER_GAP, "S4"),
          ("off-gap", "1", GAP_CLASS, "S4")]


def write_records(root: Path, tree: str, records: Mapping[str, Mapping[str, Any]]) -> None:
    """Write each record as <root>/runs/<tree>/<name>/trial.json."""
    for name, record in records.items():
        path = root / "runs" / tree / name / "trial.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record), encoding="ascii")


def test_the_report_lists_gap_attempts_apart_from_failures(tmp_path: Path) -> None:
    root, out = tmp_path / "root", tmp_path / "out"
    write_records(root, "sweep-20261025-2000-syn-maxabs", SYNTHETIC_TRIALS)
    assert tabulate().main([str(root), str(out)]) == 0
    trials = tsv(out / "trials.tsv")
    assert tuple(trials[0][:len(OLD_TRIAL_COLUMNS)]) == OLD_TRIAL_COLUMNS, "the existing columns stay as they were"
    assert trials[0][len(OLD_TRIAL_COLUMNS):] == ["sim_gap_attempts"]
    counts = {row[4]: row[-1] for row in trials[1:]}
    assert counts == {"baseline-gap": "0", "cap-gap": "2", "failed": "0", "fed-gap": "1", "off-gap": "1"}
    ends = {row[4]: row[9] for row in trials[1:]}
    assert ends["failed"] == "correction-cap" and ends["fed-gap"] == "-" and ends["off-gap"] == SIM_GAP
    passes = tsv(out / "passes.tsv")
    assert tuple(passes[0][:len(OLD_PASS_COLUMNS)]) == OLD_PASS_COLUMNS
    assert passes[0][len(OLD_PASS_COLUMNS):] == ["sim_gap_attempts"]
    assert passes[1] == ["20261025-2000", "syn", "5", "1", "0", "0", "base", "4"]
    listed = tsv(out / "not_verifiable.tsv")
    assert listed[0] == ["pass", "slug", "half", "combo", "trial_id", "attempt", "gap_class", "stage"]
    assert [(row[4].split("/")[4], row[5], row[6], row[7]) for row in listed[1:]] == LISTED
    assert all(row[:4] == ["20261025-2000", "syn", "maxabs", "base"] for row in listed[1:])
    checks = tsv(out / "hardware_check.tsv")
    assert checks[0] == ["trial_id", "attempt", "gap_class", "record"]
    assert [(row[0], row[1], row[2]) for row in checks[1:]] == [(row[4], row[5], row[6]) for row in listed[1:]]
    assert checks[1][3] == "sweep-20261025-2000-syn-maxabs/cap-gap/trial.json"


def test_the_report_reads_the_records_run_loop_writes(base: Path, bench: Path) -> None:
    on = Script(replies=[program(0), program(1)], attempts=[Gap(), doubled])
    run_gap(base, bench, on, gap_recipe({NAME: True}), "sweep-20261025-2100-syn-maxabs")
    off = Script(replies=[program(0)], attempts=[Gap(OTHER_GAP)])
    run_gap(base, bench, off, gap_recipe({}), "sweep-20261025-2100-syn-pcc")
    out = base / "tables"
    assert tabulate().main([str(runs_root(base)), str(out)]) == 0
    listed = tsv(out / "not_verifiable.tsv")[1:]
    assert [(row[2], row[5], row[6], row[7]) for row in listed] == [("maxabs", "0", GAP_CLASS, "S4"),
                                                                   ("pcc", "0", OTHER_GAP, "S4")]
