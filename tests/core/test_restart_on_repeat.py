"""Tests for task P17.50: the opt-in loop key loop.restart_on_repeat, a fresh generation in place of a stuck correction.

Bible: Project Recipes (the loop opt-ins, lassi.core.recipe LOOP_OPT_INS),
Component Interfaces (compile_loop, run_loop, and their one correction
count), Design Principle 4 and Agent Rules 1 and 4 (a departure from upstream
is a named toggle, off when absent and refused under faithful: true).
plans/p17-portable.md, task P17.50, added on 2026-10-29: in the window of
2026-10-26 a trial's attempts 5 and 7, and 6 and 8, had byte-identical
prompts, the model swinging between two files until the cap.

The contract these tests fix:
1. loop.restart_on_repeat is a bool, off when absent (the loader
   materializes nothing), refused at load under faithful: true, and refused
   by the runner with a fragment prompt set, as loop.restart_after is.
2. A correction chain starts at attempt 0 or at the last restart's attempt.
   When the last attempt's error set (each error-severity diagnostic's file
   and message, line and column left out), not empty, equals that of an
   earlier attempt of its chain, the loop asks for a restart in place of the
   correction it would ask: attempt 0's prompt again, recorded under the
   stage name "restart", as loop.restart_after's restarts are. An error set
   seen only before the last restart does not count.
3. Restarts share the one count and cap with corrections.

The suite, programs, replies, and arrays are SYNTHETIC, from
tests/core/test_template_context.py's fakes, as tests/core/test_restart_after.py
uses them. No value here is a measurement.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_restart_after import (
    COMPILE_LOOP,
    FRAGMENT_SET,
    FRAGMENTS,
    GENERATE,
    RESTART,
    broken,
    check_restarts,
    fragment_recipe,
    load_refusal,
    restart_recipe,
    stages_of,
)
from test_template_context import (
    Script,
    load,
    make_assets,
    program,
    refused,
    run,
    write_suite,
    write_tree,
)

KEY = "loop.restart_on_repeat"
G, C, R = GENERATE, COMPILE_LOOP, RESTART


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


def replies(*numbers: int) -> list[str]:
    """Return SYNTHETIC replies that do not build, each with the numbered error; a repeated number repeats the error."""
    return [broken(number) for number in numbers]


@pytest.mark.parametrize(
    ("numbers", "cap", "stages"),
    [
        ((1, 1, 2, 3), 3, [G, C, R, C]),  # the same error twice in a row
        ((1, 2, 1, 3, 4), 4, [G, C, C, R, C]),  # an error that comes back after another
        ((1, 1, 1, 1, 2), 4, [G, C, R, C, R]),  # after a restart, only the new chain counts
        ((1, 2, 3, 4), 3, [G, C, C, C]),  # no error repeats: only corrections
    ],
    ids=["twice", "returns", "new-chain", "distinct"],
)
def test_a_repeated_error_set_asks_a_restart_in_place_of_the_correction(
    base: Path, bench: Path, numbers: tuple[int, ...], cap: int, stages: list[str]
) -> None:
    script = Script(replies=replies(*numbers), attempts=[])
    name = "repeat-" + "-".join(str(number) for number in numbers)
    _, trial = run(base, bench, script, restart_recipe({"max_corrections": cap, "restart_on_repeat": True}), name)
    assert stages_of(trial) == stages
    assert len(trial.attempts) == cap + 1 and trial.final.corrections == cap, "restarts share the one count and cap"
    reason = trial.final.end_reason
    assert reason is not None and reason.code == "correction-cap"
    check_restarts(script, trial)


def test_without_the_key_a_repeated_error_is_corrected_as_before(base: Path, bench: Path) -> None:
    script = Script(replies=replies(1, 1, 1, 1), attempts=[])
    _, trial = run(base, bench, script, restart_recipe({"max_corrections": 3}), "repeat-off")
    assert stages_of(trial) == [G, C, C, C]


def test_it_works_beside_restart_after(base: Path, bench: Path) -> None:
    # restart_after 3 alone would restart at attempt 4; the repeat at attempt 1 restarts first.
    script = Script(replies=replies(1, 1, 2, 3, 4, 5), attempts=[])
    loop = {"max_corrections": 5, "restart_on_repeat": True, "restart_after": 3}
    _, trial = run(base, bench, script, restart_recipe(loop), "repeat-and-after")
    assert stages_of(trial) == [G, C, R, C, C, C]
    check_restarts(script, trial)


@pytest.mark.parametrize("value", [None, "true", 1, 0.5, ["x"]])
def test_a_value_other_than_a_bool_is_refused_at_load(base: Path, value: object) -> None:
    message = load_refusal(base, restart_recipe({"restart_on_repeat": value}))
    assert KEY in message and "unknown key" not in message, message


def test_it_is_refused_under_faithful_true_and_absent_materializes_nothing(base: Path) -> None:
    message = load_refusal(base, restart_recipe({"restart_on_repeat": True}, faithful=True))
    assert KEY in message and "faithful: true" in message and "unknown key" not in message, message
    assert "restart_on_repeat" not in load(base, restart_recipe({}), "control").data["loop"]


def test_the_runner_refuses_the_key_with_a_fragment_prompt_set(base: Path, bench: Path, assets: Path) -> None:
    write_tree(assets / "prompts" / FRAGMENT_SET, FRAGMENTS)
    message = refused(base, bench, fragment_recipe({"restart_on_repeat": True}))
    assert KEY in message and "fragment" in message, message
    script = Script(replies=[program(0)], attempts=[])
    _, trial = run(base, bench, script, fragment_recipe({}), "upstream")
    assert stages_of(trial) == [GENERATE], "without the key the same recipe runs"
