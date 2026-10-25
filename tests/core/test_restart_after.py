"""Tests for task P17.25: the opt-in loop key loop.restart_after, a fresh generation after N failed corrections.

Bible: Component Interfaces (the Stage row: generate, compile_loop, run_loop,
and their one correction count), Project Recipes (the loop opt-ins,
lassi.core.recipe LOOP_OPT_INS), Result Record (Trial.requests,
Final.corrections, the end reasons), Design Principle 4 and Agent Rules 1
and 4 (a departure from upstream is a named toggle, off when absent and
refused under faithful: true). plans/p17-portable.md, task P17.25, added on
2026-10-25 from the passes of 2026-10-23 and 2026-10-24: every pass came at
correction 0 to 4, and no trial stuck at a build error recovered later.

The contract these tests fix:

1. loop.restart_after is an int from 1 to the resolved loop.max_corrections,
   both included. A bool, a null, a string, a float, a list, zero, a
   negative int, and an int above the cap are refused at load, the message
   naming loop.restart_after, and so is every value when max_corrections is
   0. Under faithful: true, set in the recipe or inherited, it is refused at
   load, the message naming the key and faithful: true. Absent, the loader
   materializes nothing, so a recipe without it keeps its resolved recipe,
   its recipe hash, and every prompt (the golden prompts and recipe hashes
   of tests/core/test_template_context.py, recorded before the key existed).
   Set, it enters the hash.
2. The runner refuses it, before any directory exists, with a fragment
   prompt set: a restart sends generate.txt, which only a template set
   holds.
3. A correction is a request after attempt 0 that is not a restart. A
   correction fails when the attempt its reply became does not reach a clean
   run: stage_reached S5. An attempt that ran clean but whose output files
   failed the oracle under loop.correct_on_mismatch did reach one. Counting
   from attempt 0 or the last restart, when the last N requests were
   corrections that all failed and the loop would ask for another, it asks
   for a restart instead: a fresh generation whose one user message is byte
   for byte attempt 0's prompt (generate.txt as attempt 0 had it, with no
   previous files, diagnostics, or run report), and whose reply becomes the
   next attempt, built, run, and corrected like any other. A restart is not
   a correction and does not count toward N, so after a restart the count
   of consecutive failed corrections starts again from 0; a correction that
   reaches a clean run starts it again too.
4. Corrections and restarts share one count and the cap: Final.corrections
   counts every attempt after the first, restarts included, and
   loop.max_corrections ends the trial at correction-cap as before, with no
   restart past it.
5. The record: a restart's Request has the stage name "restart", its own,
   never generate, compile_loop, or run_loop, whichever loop asked for it
   (compile_loop after a build error, run_loop after a failed run); its
   attempt_index names the attempt its reply became, whose prompt_ref is
   attempt 0's.

The suite, programs, replies, and arrays are SYNTHETIC, from
tests/core/test_template_context.py's fakes: the fake toolchain compiles
nothing and the scripted executor starts no process. No value here is a
measurement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from test_template_context import (
    CPP,
    ERROR,
    PROMPT_KINDS,
    RECIPE_HASHES,
    REPO,
    REQUEST_STAGES,
    TT,
    Script,
    crashed,
    doubled,
    full_script,
    golden,
    load,
    make_assets,
    program,
    recipe_data,
    refused,
    run,
    user_prompts,
    write_suite,
    write_tree,
    wrong,
)

from lassi.bench import Direction
from lassi.core import fragments
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.record import Trial
from lassi.core.store import TRIAL_JSON, trial_dir

KEY = "loop.restart_after"
# The stage name a restart's Request carries (contract point 5).
RESTART = "restart"
GENERATE, COMPILE_LOOP, RUN_LOOP = "generate", "compile_loop", "run_loop"
G, C, R = GENERATE, COMPILE_LOOP, RESTART

# The request stages of a trial whose every reply fails to build, by (restart_after, max_corrections).
COMPILE_SEQUENCES = {
    (1, 5): [G, C, R, C, R, C],
    (2, 5): [G, C, C, R, C, C],
    (2, 8): [G, C, C, R, C, C, R, C, C],
    (3, 5): [G, C, C, C, R, C],
    (5, 5): [G, C, C, C, C, C],
}

# A SYNTHETIC fragment set holding every key generate and compile_loop read for cpp -> tt.
FRAGMENT_SET = "syn-restart-fragments"
FRAGMENTS = {
    fragments.fragment_key(key, Direction(CPP, TT)): f"<{key}>"
    for key in (*fragments.GENERATE_KEYS, *fragments.CORRECT_KEYS)
}


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


def broken(number: int) -> str:
    """Return a SYNTHETIC reply that does not build: its one file holds an error directive, numbered."""
    return program(number, f"{ERROR}SYNTHETIC build error {number}")


def restart_recipe(loop: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    """Return test_template_context's cpp -> tt recipe with `loop` as its loop section, over base.yaml's cap."""
    return recipe_data(loop=dict(loop), **changes)


def stages_of(trial: Trial) -> list[str]:
    """Return the stage name of every request of `trial`, in order."""
    return [request.stage for request in trial.requests or []]


def check_restarts(script: Script, trial: Trial) -> None:
    """Check that every restart sent attempt 0's prompt, and that no correction did.

    Each request's one user message is compared with the first request's,
    generate's, and each restart's attempt must carry attempt 0's prompt_ref.
    """
    prompts = user_prompts(script)
    requests = trial.requests or []
    assert len(prompts) == len(requests) == len(trial.attempts)
    for request, prompt in zip(requests, prompts, strict=True):
        assert request.attempt_index == request.index
        if request.stage == RESTART:
            assert prompt == prompts[0], f"restart {request.index} does not send generate's prompt:\n{prompt}"
            assert trial.attempts[request.index].prompt_ref == trial.attempts[0].prompt_ref
        elif request.index:
            assert prompt != prompts[0], f"correction {request.index} sends generate's prompt"


def load_refusal(base: Path, data: Mapping[str, Any], name: str = "loaded") -> str:
    """Load `data`, expect a RecipeError from a loader that knows the key, and return its message."""
    with pytest.raises(RecipeError) as caught:
        load(base, data, name)
    message = str(caught.value)
    assert f"unknown key {KEY}" not in message, f"the loader does not know the key: {message}"
    return message


# ---------------------------------------------------------------------------
# Absent: every recipe, hash, and prompt as before (contract point 1)


def test_absent_the_key_adds_nothing_to_the_resolved_recipe(base: Path) -> None:
    recipe = load(base, restart_recipe({}))
    assert recipe.data["loop"] == {"max_corrections": 10}
    assert "restart_after" not in recipe.canonical_yaml


@pytest.mark.parametrize("path", sorted(RECIPE_HASHES))
def test_a_committed_recipe_without_the_key_keeps_its_recipe_hash(path: str) -> None:
    recipe = load_recipe(REPO / path)
    assert "restart_after" not in recipe.data["loop"]
    assert recipe.recipe_hash == RECIPE_HASHES[path]


def test_absent_every_prompt_matches_the_goldens(base: Path, bench: Path) -> None:
    script = full_script()
    _, trial = run(base, bench, script, recipe_data(), "plain")
    assert stages_of(trial) == REQUEST_STAGES
    for kind, prompt in zip(PROMPT_KINDS, user_prompts(script), strict=True):
        assert prompt == golden(kind), f"the {kind} prompt changed without the key"


# ---------------------------------------------------------------------------
# The recipe key (contract point 1)


def test_set_the_key_loads_as_written_and_enters_the_hash(base: Path) -> None:
    hashes = {}
    for label, loop in (("plain", {}), ("two", {"restart_after": 2}), ("three", {"restart_after": 3})):
        (base / label).mkdir()
        recipe = load(base / label, restart_recipe(loop))
        assert recipe.data["loop"] == {"max_corrections": 10, **loop}
        hashes[label] = recipe.recipe_hash
    assert len(set(hashes.values())) == 3, "a key that is set enters the hash"


def cap_id(cap: Mapping[str, Any], value: Any) -> str:
    """Return a parameter id: the resolved cap (base.yaml's 10 when the loop does not set one) and the value."""
    return f"cap{cap.get('max_corrections', 10)}-{value!r}"


GOOD = [({}, 1), ({}, 10), ({"max_corrections": 3}, 1), ({"max_corrections": 3}, 3), ({"max_corrections": 15}, 15)]


@pytest.mark.parametrize(("cap", "after"), GOOD, ids=[cap_id(cap, after) for cap, after in GOOD])
def test_an_int_from_one_to_the_cap_loads(base: Path, cap: Mapping[str, Any], after: int) -> None:
    assert load(base, restart_recipe({**cap, "restart_after": after})).data["loop"]["restart_after"] == after


BAD = [
    ({}, True), ({}, False), ({}, None), ({}, 0), ({}, -1), ({}, 11), ({}, "2"), ({}, 2.0), ({}, [2]),
    ({"max_corrections": 3}, 4), ({"max_corrections": 0}, 1),
]


@pytest.mark.parametrize(("cap", "value"), BAD, ids=[cap_id(cap, value) for cap, value in BAD])
def test_a_bad_value_is_refused_at_load_naming_the_key(base: Path, cap: Mapping[str, Any], value: Any) -> None:
    message = load_refusal(base, restart_recipe({**cap, "restart_after": value}))
    assert KEY in message and "loaded.yaml" in message, message


def test_the_key_is_refused_under_faithful(base: Path) -> None:
    message = load_refusal(base, restart_recipe({"restart_after": 2}, faithful=True))
    assert KEY in message and "faithful: true" in message and "loaded.yaml" in message, message
    assert load(base, restart_recipe({}, faithful=True), "control").data["loop"] == {"max_corrections": "uncapped"}


def test_an_inherited_key_is_refused_when_the_child_turns_faithful_on(base: Path) -> None:
    load(base, restart_recipe({"restart_after": 2}), "restart-parent")
    message = load_refusal(base, {"extends": "restart-parent.yaml", "faithful": True}, "child")
    assert KEY in message and "faithful: true" in message, message


# ---------------------------------------------------------------------------
# The runner's refusal with a fragment set (contract point 2)


def fragment_recipe(loop: Mapping[str, Any]) -> dict[str, Any]:
    """Return a generate and compile_loop recipe on the SYNTHETIC fragment set, with `loop`."""
    data = restart_recipe(loop, prompts=FRAGMENT_SET, stages=[GENERATE, COMPILE_LOOP])
    data.pop("oracle")
    return data


def test_the_runner_refuses_the_key_with_a_fragment_prompt_set(base: Path, bench: Path, assets: Path) -> None:
    write_tree(assets / "prompts" / FRAGMENT_SET, FRAGMENTS)
    message = refused(base, bench, fragment_recipe({"restart_after": 2}))
    assert KEY in message and "fragment" in message, message
    script = Script(replies=[program(0)], attempts=[])
    _, trial = run(base, bench, script, fragment_recipe({}), "upstream")
    assert stages_of(trial) == [GENERATE], "without the key the same recipe runs"


# ---------------------------------------------------------------------------
# Restarts (contract points 3 to 5)


SEQUENCE_IDS = [f"n{after}-cap{cap}" for after, cap in sorted(COMPILE_SEQUENCES)]


@pytest.mark.parametrize(("after", "cap"), sorted(COMPILE_SEQUENCES), ids=SEQUENCE_IDS)
def test_a_restart_comes_exactly_after_n_failed_compile_corrections(
    base: Path, bench: Path, after: int, cap: int
) -> None:
    # Every reply fails to build, so every correction fails: a restart comes after each run of N corrections, the
    # count starts again after it, and the cap ends the trial with restarts counted among its corrections.
    script = Script(replies=[broken(number) for number in range(cap + 1)], attempts=[])
    name = f"compile-n{after}-cap{cap}"
    run_dir, trial = run(base, bench, script, restart_recipe({"max_corrections": cap, "restart_after": after}), name)
    assert stages_of(trial) == COMPILE_SEQUENCES[(after, cap)]
    assert script.replies == [] and len(trial.attempts) == cap + 1
    assert all(attempt.stage_reached == "S1" for attempt in trial.attempts)
    reason = trial.final.end_reason
    assert reason is not None and reason.code == "correction-cap" and "loop.max_corrections" in reason.message
    assert trial.final.corrections == cap, "restarts and corrections share the one count"
    check_restarts(script, trial)
    record = json.loads((trial_dir(run_dir, trial.trial_id) / TRIAL_JSON).read_text(encoding="ascii"))
    assert [request["stage"] for request in record["requests"]] == COMPILE_SEQUENCES[(after, cap)]


def test_a_restart_comes_after_n_failed_run_corrections_under_the_same_stage_name(base: Path, bench: Path) -> None:
    # Every reply builds and every run crashes, so run_loop asks each correction, and the restart still reads as one.
    script = Script(replies=[program(number) for number in range(5)], attempts=[crashed] * 5)
    _, trial = run(base, bench, script, restart_recipe({"max_corrections": 4, "restart_after": 2}), "runs-n2")
    assert stages_of(trial) == [GENERATE, RUN_LOOP, RUN_LOOP, RESTART, RUN_LOOP]
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S4"] * 5
    assert script.attempts == [] and script.replies == []
    reason = trial.final.end_reason
    assert reason is not None and reason.code == "correction-cap"
    assert trial.final.corrections == 4
    check_restarts(script, trial)


def test_compile_and_run_failures_count_together_and_a_restarted_attempt_can_pass(base: Path, bench: Path) -> None:
    # Attempt 0 does not build; correction 1 builds and crashes; correction 2 does not build. Two corrections in a
    # row failed, so attempt 3 is a restart, and it builds and passes.
    script = Script(replies=[broken(0), program(1), broken(2), program(3)], attempts=[crashed, doubled])
    _, trial = run(base, bench, script, restart_recipe({"max_corrections": 3, "restart_after": 2}), "mixed")
    assert stages_of(trial) == [GENERATE, COMPILE_LOOP, RUN_LOOP, RESTART]
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S1", "S4", "S1", "S5"]
    assert script.attempts == [] and script.replies == []
    assert trial.final.end_reason is None and trial.final.corrections == 3 and trial.final.alignment == 1.0
    check_restarts(script, trial)


# ---------------------------------------------------------------------------
# A clean run before N means no restart (contract point 3)


def test_a_clean_run_before_n_failures_asks_no_restart_and_changes_no_prompt(base: Path, bench: Path) -> None:
    replies = [broken(0), broken(1), program(2)]
    plain = Script(replies=list(replies), attempts=[doubled])
    run(base, bench, plain, restart_recipe({}), "without")
    script = Script(replies=list(replies), attempts=[doubled])
    _, trial = run(base, bench, script, restart_recipe({"restart_after": 3}), "with")
    assert stages_of(trial) == [GENERATE, COMPILE_LOOP, COMPILE_LOOP]
    assert trial.final.end_reason is None and trial.final.corrections == 2
    assert user_prompts(script) == user_prompts(plain), "a key that never fires changes no prompt"


def test_a_correction_that_runs_clean_starts_the_count_again(base: Path, bench: Path) -> None:
    # Under loop.correct_on_mismatch, correction 1 runs clean but misses the tolerance: it reached a clean run, so
    # the count starts again there. Corrections 2 and 3 fail to build, so attempt 4, not attempt 3, is the restart.
    script = Script(replies=[broken(0), program(1), broken(2), broken(3), program(4)], attempts=[wrong, doubled])
    loop = {"correct_on_mismatch": True, "restart_after": 2}
    _, trial = run(base, bench, script, restart_recipe(loop), "reset")
    assert stages_of(trial) == [GENERATE, COMPILE_LOOP, RUN_LOOP, COMPILE_LOOP, RESTART]
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S1", "S5", "S1", "S1", "S5"]
    assert trial.attempts[1].alignment.mean == 0.0, "correction 1 ran clean and failed the oracle"
    assert script.attempts == [] and script.replies == []
    assert trial.final.end_reason is None and trial.final.corrections == 4 and trial.final.alignment == 1.0
    check_restarts(script, trial)


def test_set_but_never_reached_every_prompt_matches_the_goldens(base: Path, bench: Path) -> None:
    # The four-prompt script: one failed correction (a crash), then a clean run that misses the tolerance, which
    # starts the count again, then a pass; with restart_after 2 no restart comes, and every prompt is the golden one.
    script = full_script()
    loop = {"correct_on_mismatch": True, "restart_after": 2}
    _, trial = run(base, bench, script, restart_recipe(loop), "unreached")
    assert stages_of(trial) == REQUEST_STAGES
    for kind, prompt in zip(PROMPT_KINDS, user_prompts(script), strict=True):
        assert prompt == golden(kind), f"the {kind} prompt changed with the key set but never reached"
