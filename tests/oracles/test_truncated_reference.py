"""Tests for the oracle stage under a reference stdout marked truncated (task P2.2).

Bible: Oracles (stdout_mask rules), Result Record (reference_run,
Attempt.run, Attempt.alignment, Diagnostic), Sandbox (output caps).

The contract these tests fix, from the P2.2 acceptance criteria
(plans/p2-scoring.md) and the P2 phase note "Reference run flags":

- An executor keeps at most its output cap of a program's stdout and marks
  the cut with RunResult.stdout_truncated. P2.2 records the flag of the
  target reference's baseline run in Trial.reference_run.stdout_truncated
  (RunInfo, `bool | None`; None means not recorded).
- Called as a stage, the oracle stage never aligns against a reference stdout
  marked truncated (reference_run.stdout_truncated is True). Every attempt
  keeps its alignment unset (Alignment(): per_input [] and mean None), even
  a candidate whose stdout equals the kept part of the reference exactly.
  Each affected attempt, that is each attempt whose run holds stdout (the
  attempts the stage would align otherwise, the faithful loop's stale output
  included), gains exactly one Diagnostic with stage "run", severity
  "warning", and code REFERENCE_TRUNCATED ("reference-stdout-truncated"),
  whose message names the cut: it says that the reference run's stdout was
  truncated. The attempt's other diagnostics stay as they were, in order;
  an attempt that never ran is unchanged, and nothing else in the trial
  changes.
- A reference stdout kept whole (stdout_truncated False), or whose flag was
  not recorded (None, as in a trial.json from before P2.2), is aligned
  exactly as align_runs aligns it, with no such warning. Only the stdout
  flag bears on a stdout oracle: a reference run whose stderr was cut or
  whose workdir came back incomplete is still aligned.

The whole path, from the baseline's reference run through trial.json, is
tested in tests/core/test_run_flags.py.

SYNTHETIC fixtures: every stdout, reply, and source here is written by hand
in the print formats of the pinned HeCBench sources. The values are
invented, not program output, and nothing here is executed. Wall times are
PLACEHOLDER fixture values, not measurements.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import Direction, load_suite
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.interfaces import Sampling
from lassi.core.recipe import load_recipe
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
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.core.stages import RunContext
from lassi.core.store import TextStore
from lassi.executors import NoneExecutor
from lassi.llm import MockBackend

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "tests" / "fixtures" / "recipes" / "p0-smoke.yaml"
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
OMP_TO_CUDA = Direction("omp", "cuda")
STAGES = ["generate", "compile_loop", "oracle"]
# A commit id for synthetic provenance; not a commit of this repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
PLACEHOLDER_WALL_S = 1.0

FLAGS = ("stdout_truncated", "stderr_truncated", "workdir_incomplete")
# The code of the run-stage warning the oracle stage gives an attempt it does not align because the reference
# stdout was cut.
REFERENCE_TRUNCATED = "reference-stdout-truncated"

# SYNTHETIC: layout's stdout, and the same with different timings (a match under the masks); values invented.
LAYOUT = "Average kernel execution time (AoS): 1.5 (us)\nPASS\nAverage kernel execution time (SoA): 2.5 (us)\nPASS\n"
LAYOUT_OTHER_TIMES = LAYOUT.replace("1.5 (us)", "3.75 (us)").replace("2.5 (us)", "0.5 (us)")
# SYNTHETIC: the part of layout's stdout a cut at an output cap could keep: its first timing line and PASS.
LAYOUT_CUT = "".join(LAYOUT.splitlines(keepends=True)[:2])


# ---------------------------------------------------------------------------
# The contract's names, looked up so a missing one fails with a clear message


def require_flag_fields() -> None:
    """Fail the test clearly while RunInfo lacks any of the three run flags."""
    names = {spec.name for spec in dataclasses.fields(RunInfo)}
    missing = [flag for flag in FLAGS if flag not in names]
    if missing:
        pytest.fail(f"RunInfo has no {', '.join(missing)}; task P2.2 records the run flags in the Result Record")


# ---------------------------------------------------------------------------
# The stage, trials, and attempts


def recipe_data(item: str) -> dict[str, Any]:
    """Return the p0-smoke recipe for `item` with the stdout_mask oracle (passfail on) and the oracle stage."""
    data = copy.deepcopy(yaml.safe_load(SMOKE.read_text(encoding="utf-8")))
    data["bench"]["items"] = [item]
    data["oracle"] = {"kind": "stdout_mask", "passfail": True}
    data["stages"] = list(STAGES)
    return data


def oracle_stage(tmp_path: Path, item: str = "layout", direction: Direction = OMP_TO_CUDA) -> tuple[Any, RunContext]:
    """Return the registered oracle stage built on a RunContext for `item` and `direction`, and the context."""
    path = tmp_path / "truncated-reference.yaml"
    path.write_bytes(yaml.safe_dump(recipe_data(item), sort_keys=False).encode("ascii"))
    context = RunContext(
        recipe=load_recipe(path),
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


def compiled(index: int, run: RunInfo | None = None, **fields: Any) -> Attempt:
    """Return a SYNTHETIC attempt that compiled (S4), with `run` (default: never run)."""
    files = {target_file(OMP_TO_CUDA): "int main() { return 0; }\n"}
    return Attempt(
        index=index, response_text="SYNTHETIC reply", files=files, stage_reached="S4", run=run or RunInfo(), **fields
    )


def compile_failed(index: int) -> Attempt:
    """Return a SYNTHETIC attempt that parsed (S1) but did not compile."""
    error = Diagnostic(stage="compile", severity="error", code="fake-error", message="SYNTHETIC compile error")
    files = {target_file(OMP_TO_CUDA): "#error SYNTHETIC\n"}
    return Attempt(index=index, response_text="SYNTHETIC reply", files=files, stage_reached="S1", diagnostics=[error])


def ran(store: TextStore, stdout: str, exit_code: int = 0) -> RunInfo:
    """Return a SYNTHETIC RunInfo of an attempt run whose stdout is kept in `store`, every flag recorded False."""
    require_flag_fields()
    return RunInfo(
        exit_code=exit_code, hang=False, wall_s=PLACEHOLDER_WALL_S, stdout_ref=store.put(stdout),
        **dict.fromkeys(FLAGS, False),
    )


def trial_of(context: RunContext, attempts: Sequence[Attempt], reference: RunInfo) -> Trial:
    """Return a SYNTHETIC trial of the context's item and direction with `reference` as its reference run."""
    trial_id = make_trial_id("oracle-test", "mock-reference", SUITE, context.direction.name, context.item, 1)
    return Trial(
        trial_id=trial_id,
        recipe_hash=context.recipe.recipe_hash,
        provenance=Provenance(
            commit=FAKE_COMMIT, dirty=False, device="scripted (SYNTHETIC)", sdk=None, date="2026-09-24T00:00:00+00:00"
        ),
        bench_item=context.suite.bench_item(context.item, context.direction),
        model=ModelInfo(backend="mock", id="mock-reference", sampling=SAMPLING),
        reference_run=reference,
        attempts=list(attempts),
    )


def reference_run(store: TextStore, stdout: str, **flags: bool | None) -> RunInfo:
    """Return a SYNTHETIC clean reference run whose stdout is kept in `store`, with the run flags `flags`."""
    require_flag_fields()
    return RunInfo(exit_code=0, hang=False, wall_s=PLACEHOLDER_WALL_S, stdout_ref=store.put(stdout), **flags)


def cut_reference(store: TextStore, stdout: str = LAYOUT_CUT) -> RunInfo:
    """Return a SYNTHETIC reference run whose stdout the executor marked truncated (only stdout was cut)."""
    return reference_run(store, stdout, stdout_truncated=True, stderr_truncated=False, workdir_incomplete=False)


def cut_warnings(attempt: Attempt) -> list[Diagnostic]:
    """Return the attempt's diagnostics with code REFERENCE_TRUNCATED."""
    return [item for item in attempt.diagnostics if item.code == REFERENCE_TRUNCATED]


def assert_warned_and_unaligned(before: Attempt, after: Attempt) -> None:
    """Assert that `after` is `before` with its alignment unset and exactly one REFERENCE_TRUNCATED warning added."""
    assert after.alignment == Alignment(), f"attempt {after.index}: never aligned against a cut reference stdout"
    (warning,) = cut_warnings(after)
    assert (warning.stage, warning.severity) == ("run", "warning")
    assert [item for item in after.diagnostics if item.code != REFERENCE_TRUNCATED] == before.diagnostics, (
        "the attempt's other diagnostics stay, in order"
    )
    assert dataclasses.replace(after, diagnostics=before.diagnostics) == before, "nothing else in the attempt changes"


def assert_only_diagnostics_changed(before: Trial, after: Trial) -> None:
    """Assert that `after` differs from `before` in attempt diagnostics at most."""
    assert dataclasses.replace(after, attempts=[]) == dataclasses.replace(before, attempts=[])
    assert len(after.attempts) == len(before.attempts)
    for old, new in zip(before.attempts, after.attempts, strict=True):
        assert dataclasses.replace(new, diagnostics=old.diagnostics) == old


# ---------------------------------------------------------------------------
# A reference stdout marked truncated


def test_no_attempt_is_aligned_against_a_reference_stdout_marked_truncated(tmp_path: Path) -> None:
    stage, context = oracle_stage(tmp_path)
    store = context.store
    attempts = [compile_failed(0), compiled(1, ran(store, LAYOUT_OTHER_TIMES)), compiled(2, ran(store, LAYOUT_CUT))]
    trial = trial_of(context, attempts, cut_reference(store))
    result = stage(trial)
    assert result.attempts[0] == trial.attempts[0], "an attempt that never ran is unchanged"
    for index in (1, 2):
        assert_warned_and_unaligned(trial.attempts[index], result.attempts[index])
    assert_only_diagnostics_changed(trial, result)


def test_a_candidate_equal_to_the_kept_reference_text_is_still_not_aligned(tmp_path: Path) -> None:
    """Without the flag this stdout would align at 1.0: it equals what the reference kept, byte for byte."""
    stage, context = oracle_stage(tmp_path)
    store = context.store
    trial = trial_of(context, [compiled(0, ran(store, LAYOUT_CUT))], cut_reference(store))
    assert stage.align_runs(trial, LAYOUT_CUT).attempts[0].alignment == Alignment(per_input=[1.0], mean=1.0)
    assert_warned_and_unaligned(trial.attempts[0], stage(trial).attempts[0])


def test_the_warning_names_the_cut(tmp_path: Path) -> None:
    stage, context = oracle_stage(tmp_path)
    store = context.store
    trial = trial_of(context, [compiled(0, ran(store, LAYOUT_OTHER_TIMES))], cut_reference(store))
    (warning,) = cut_warnings(stage(trial).attempts[0])
    message = warning.message
    assert isinstance(message, str) and message.strip()
    for word in ("reference", "stdout", "truncat"):
        assert re.search(word, message, re.IGNORECASE), f"the message names the cut ({word!r}): {message!r}"


def test_the_stale_output_attempt_gets_the_warning_and_later_unrun_attempts_do_not(tmp_path: Path) -> None:
    """P1.6: attempt 0 ran, 1 to 8 failed to compile, 9 compiled after 8 corrections and was never run."""
    stage, context = oracle_stage(tmp_path)
    store = context.store
    stale = Diagnostic(stage="run", severity="warning", code="stale-output", message="SYNTHETIC: attempt 0 stands")
    attempts = [
        compiled(0, ran(store, LAYOUT_CUT, exit_code=1)),
        *(compile_failed(index) for index in range(1, 9)),
        compiled(9, diagnostics=[stale]),
    ]
    trial = trial_of(context, attempts, cut_reference(store))
    result = stage(trial)
    assert_warned_and_unaligned(trial.attempts[0], result.attempts[0])
    assert result.attempts[1:] == trial.attempts[1:], "attempts that never ran are unchanged, the stale one included"


def test_a_compile_only_trial_is_unchanged_under_a_cut_reference(tmp_path: Path) -> None:
    stage, context = oracle_stage(tmp_path)
    trial = trial_of(context, [compile_failed(0), compiled(1)], cut_reference(context.store))
    assert stage(trial) == trial


# ---------------------------------------------------------------------------
# A reference stdout kept whole, or whose flag was not recorded


@pytest.mark.parametrize("recorded", [False, None], ids=["kept-whole", "not-recorded"])
def test_a_reference_stdout_not_marked_truncated_is_aligned_as_before(tmp_path: Path, recorded: bool | None) -> None:
    stage, context = oracle_stage(tmp_path)
    store = context.store
    flags: Mapping[str, bool | None] = dict.fromkeys(FLAGS, recorded)
    attempts = [compile_failed(0), compiled(1, ran(store, LAYOUT_OTHER_TIMES))]
    trial = trial_of(context, attempts, reference_run(store, LAYOUT, **flags))
    result = stage(trial)
    assert result == stage.align_runs(trial, LAYOUT)
    assert result.attempts[1].alignment == Alignment(per_input=[1.0], mean=1.0)
    assert all(cut_warnings(attempt) == [] for attempt in result.attempts)


def test_a_reference_run_whose_stderr_or_workdir_was_cut_is_still_aligned(tmp_path: Path) -> None:
    stage, context = oracle_stage(tmp_path)
    store = context.store
    reference = reference_run(store, LAYOUT, stdout_truncated=False, stderr_truncated=True, workdir_incomplete=True)
    trial = trial_of(context, [compiled(0, ran(store, LAYOUT_OTHER_TIMES))], reference)
    result = stage(trial)
    assert result == stage.align_runs(trial, LAYOUT), "only a cut stdout bears on a stdout oracle"
    assert result.attempts[0].alignment == Alignment(per_input=[1.0], mean=1.0)
    assert cut_warnings(result.attempts[0]) == []
