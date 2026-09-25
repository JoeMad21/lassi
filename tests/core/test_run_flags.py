"""Tests for the run flags of Trial.reference_run and Attempt.run (task P2.2).

Bible: Result Record (reference_run, Attempt.run), Sandbox (output caps,
attempt run limits), Oracles (stdout_mask rules).

The contract these tests fix, from the P2.2 acceptance criteria
(plans/p2-scoring.md) and the P2 phase notes ("Reference run flags" and the
projects/base.yaml sandbox.wall_s comment):

- RunInfo (lassi.core.record) gains three fields, `stdout_truncated`,
  `stderr_truncated`, and `workdir_incomplete`, each `bool | None` with
  default None, named as the RunResult flags they record. None means not
  recorded: a run that did not happen, or a trial.json written before the
  fields existed. A bool means the run happened and the executor's flag of
  the same name was read, so a run whose output was kept whole records
  False, never None. Any other value is refused with ValueError naming the
  field, as for every record field.
- The baseline stage fills them for Trial.reference_run from the target
  reference's RunResult; the source reference's run (fixes.baseline_both)
  never reaches the record. run_loop fills them for Attempt.run from each
  attempt's RunResult and keeps the run-stage warnings of P1.6
  (stdout-truncated, stderr-truncated, workdir-incomplete). A compile-only
  executor runs nothing, so every flag stays None.
- With the oracle stage listed, a reference stdout marked truncated is never
  aligned against: the alignments stay unset and each attempt that ran gets
  one run-stage warning, code `reference-stdout-truncated` (the stage-level
  contract is in tests/oracles/test_truncated_reference.py; here the whole
  path runs, baseline to trial.json).
- trial.md shows the three flags in the Reference run table and in each
  attempt's Run table: a recorded flag reads true or false, and a flag not
  recorded reads neither. The Parquet trials table carries them as
  reference_run_stdout_truncated, reference_run_stderr_truncated, and
  reference_run_workdir_incomplete, and the attempts table as
  run_stdout_truncated, run_stderr_truncated, and run_workdir_incomplete,
  all bool columns, null where not recorded.
- A trial.json written before this change (no flag keys in reference_run or
  in any Attempt.run) loads with every flag None and every other field as
  written.
- projects/base.yaml's sandbox.wall_s comment names the 30 s floor
  (RUN_WALL_FLOOR_S) and the case where no reference ran.

The runs use the p0-smoke template prompt set, so no upstream text is
needed. Fake toolchains write the files and a PLACEHOLDER artifact and
compile nothing; the scripted executor answers with SYNTHETIC run results and
runs nothing; the scripted backend answers with SYNTHETIC replies. Bench
sources and program output are SYNTHETIC texts in layout's print format, and
wall times are PLACEHOLDER fixture values. No value in this module is a
measurement.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import yaml

from lassi.bench import Direction, load_suite
from lassi.core import parquet
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.record import (
    Alignment,
    Attempt,
    Diagnostic,
    Final,
    ModelInfo,
    Provenance,
    RunInfo,
    Trial,
    from_json,
    json_text,
    make_trial_id,
    to_json,
)
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.stages import RUN_WALL_FLOOR_S
from lassi.core.store import TextStore, read_trial, trial_dir, write_trial
from lassi.core.trial_md import render_trial_md

REPO = Path(__file__).resolve().parents[2]
BASE_RECIPE = REPO / "projects" / "base.yaml"
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
MODEL_ID = "scripted-fixture"
TEMPLATE_SET = "p0-smoke"
OMP_TO_CUDA = Direction("omp", "cuda")
TARGET, SOURCE = OMP_TO_CUDA.target, OMP_TO_CUDA.source
TOOLCHAINS = {"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"}
RUN_STAGES = ["baseline", "generate", "compile_loop", "run_loop"]
ORACLE_STAGES = [*RUN_STAGES, "oracle"]
ORACLE = {"kind": "stdout_mask", "passfail": True}
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
# A commit id for synthetic provenance; not a commit of this repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"

# The three run flags, named as RunResult names them, and the code of the run-stage warning run_loop gives each (P1.6).
FLAGS = ("stdout_truncated", "stderr_truncated", "workdir_incomplete")
WARNING_CODES = {
    "stdout_truncated": "stdout-truncated",
    "stderr_truncated": "stderr-truncated",
    "workdir_incomplete": "workdir-incomplete",
}
# The Parquet column of each flag: trials table (Trial.reference_run) and attempts table (Attempt.run).
REFERENCE_RUN_COLUMNS = {flag: f"reference_run_{flag}" for flag in FLAGS}
ATTEMPT_RUN_COLUMNS = {flag: f"run_{flag}" for flag in FLAGS}
# The code of the warning the oracle stage gives an attempt it does not align because the reference stdout was cut.
REFERENCE_TRUNCATED = "reference-stdout-truncated"
NOT_RECORDED = dict.fromkeys(FLAGS)
RECORDED_CLEAR = dict.fromkeys(FLAGS, False)

# SYNTHETIC bench sources; the fake toolchains refuse any file holding an `#error` line.
OMP_SOURCE = '#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  std::printf("done\\n");\n}\n'
CUDA_SOURCE = "#include <cstdio>\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n"
SOURCES = {"omp": OMP_SOURCE, "cuda": CUDA_SOURCE}
BAD_CODE = "#error SYNTHETIC not translated yet\nint main() { return 1; }\n"
GOOD_CODE = "int main() { return 0; }\n"
COMPILE_ERROR = Diagnostic(stage="compile", severity="error", code="synthetic", message="SYNTHETIC compile error")

# SYNTHETIC program output in layout's print format (values invented), and PLACEHOLDER wall times in seconds.
LAYOUT = "Average kernel execution time (AoS): 1.5 (us)\nPASS\nAverage kernel execution time (SoA): 2.5 (us)\nPASS\n"
LAYOUT_OTHER_TIMES = LAYOUT.replace("1.5 (us)", "3.75 (us)").replace("2.5 (us)", "0.5 (us)")
LAYOUT_FAIL = LAYOUT_OTHER_TIMES.replace("PASS", "FAIL")
SOURCE_REFERENCE_STDOUT = "SYNTHETIC omp reference stdout\nPASS\n"
PLACEHOLDER_REFERENCE_WALL_S = 1.25
PLACEHOLDER_ATTEMPT_WALL_S = 0.5
RUN_ERROR_EXIT = 3


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
# The contract's names, looked up so a missing one fails with a clear message


def require_flag_fields() -> None:
    """Fail the test clearly while RunInfo lacks any of the three run flags."""
    names = {spec.name for spec in dataclasses.fields(RunInfo)}
    missing = [flag for flag in FLAGS if flag not in names]
    if missing:
        pytest.fail(f"RunInfo has no {', '.join(missing)}; task P2.2 records the run flags in the Result Record")


def flags_of(run: RunInfo) -> dict[str, bool | None]:
    """Return the three run flags a RunInfo records, by name."""
    require_flag_fields()
    return {flag: getattr(run, flag) for flag in FLAGS}


def only(flag: str | None) -> dict[str, bool]:
    """Return the recorded flags of a run whose RunResult set `flag` alone (None: no flag set)."""
    return {name: name == flag for name in FLAGS}


def codes(attempt: Attempt, code: str) -> list[tuple[str, str]]:
    """Return (stage, severity) of each of the attempt's diagnostics with `code`."""
    return [(item.stage, item.severity) for item in attempt.diagnostics if item.code == code]


# ---------------------------------------------------------------------------
# Fake components


@dataclass
class Log:
    """The script the fakes follow and what they saw.

    `references` maps a language to the RunResult every baseline run of that
    language's reference returns; `attempt_runs` are the results of the runs
    of attempts, in order. `reference_runs` lists the languages of the
    reference runs made, in order.
    """

    replies: list[str] = field(default_factory=list)
    references: dict[str, RunResult] = field(default_factory=dict)
    attempt_runs: list[RunResult] = field(default_factory=list)
    reference_runs: list[str] = field(default_factory=list)


def fake_toolchain(registered_as: str) -> type:
    """Return a Toolchain class without PIN: a file holding `#error` fails, anything else builds a PLACEHOLDER."""

    class FakeToolchain:
        """Writes the files and reports a build; compiles nothing."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Write every file under `workdir` and return a failed build or a PLACEHOLDER artifact."""
            workdir = Path(workdir)
            for path, text in [*files.items(), *(harness or {}).items()]:
                target = workdir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            if any("#error" in text for text in files.values()):
                return BuildResult(artifact=None, diagnostics=[COMPILE_ERROR])
            artifact = workdir / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


def scripted_executor(log: Log) -> type:
    """Return a sandboxed Executor class that runs programs: references by language, attempts from the script."""

    class ScriptedExecutor:
        """Records each run and returns its SYNTHETIC RunResult; runs nothing."""

        name = "scripted"
        capabilities = frozenset({"runs_code", "sandboxed"})

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Return the reference result of the artifact's language, or the next scripted attempt result."""
            folder = Path(artifact).parent.parent.name
            if folder.startswith("baseline-"):
                language = folder[len("baseline-") :]
                log.reference_runs.append(language)
                return log.references[language]
            assert log.attempt_runs, f"the executor was asked to run {artifact}, but the script holds no more runs"
            return log.attempt_runs.pop(0)

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
        """Answers with the next scripted reply."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id, as every backend does."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Return the next scripted reply."""
            assert log.replies, "the backend was asked for more replies than the script holds"
            return Completion(text=log.replies.pop(0), prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def make_registry(log: Log, stages: Sequence[str]) -> Registry:
    """Return a test Registry: the fakes, the real stdout_mask oracle, and the named real stages."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(log))
    registry.register("Executor", "none", CompileOnlyExecutor)
    registry.register("Executor", "scripted", scripted_executor(log))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, fake_toolchain(name))
    registry.register("Oracle", ORACLE["kind"], DEFAULT_REGISTRY.get("Oracle", ORACLE["kind"]).factory)
    for name in stages:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    return registry


# ---------------------------------------------------------------------------
# Recipes, runs, and SYNTHETIC run results


def write_bench(root: Path) -> Path:
    """Write the item's SYNTHETIC source per language where the suite manifest lays it out; return `root`."""
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, text in SOURCES.items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def target_file() -> str:
    """Return the item's one target file name, from the suite manifest."""
    return load_suite(SUITE_MANIFEST).items[ITEM].languages[TARGET].files[0]


def blocks(code: str) -> str:
    """Return a SYNTHETIC reply holding `code` as the target's one FILE block."""
    return render_file_blocks({target_file(): code})


def recipe_data(*, executor: str = "scripted", stages: Sequence[str] = RUN_STAGES, **changes: Any) -> dict[str, Any]:
    """Return a one-trial template-set recipe (fixes on, so both references run) for layout, omp to cuda."""
    data: dict[str, Any] = {
        "extends": "base",
        "faithful": False,
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": SOURCE, "target": TARGET}],
        "prompts": TEMPLATE_SET,
        "toolchain": dict(TOOLCHAINS),
        "stages": list(stages),
        "executor": {"kind": executor},
        "trials": {"n": 1},
    }
    data.update(changes)
    return data


@dataclass
class Outcome:
    """One finished run: its directory, its one trial's id, the trial read back from trial.json, and the log."""

    run_dir: Path
    trial_id: str
    trial: Trial
    log: Log

    @property
    def store(self) -> TextStore:
        """Return the run's text store."""
        return TextStore(self.run_dir)


def run_one(tmp_path: Path, name: str, data: Mapping[str, Any], log: Log) -> Outcome:
    """Run the one-trial recipe `data`, saved as `<name>.yaml`, with the fakes following `log`; return the outcome."""
    registry = make_registry(log, data["stages"])
    bench = write_bench(tmp_path / f"{name}-bench")
    options = RunOptions(runs_root=tmp_path / name / "runs-root", run_id="test-run", bench_root=bench,
                         registry=registry)
    path = tmp_path / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    run_dir = run_recipe(path, options)
    trial_id = make_trial_id(name, MODEL_ID, SUITE, OMP_TO_CUDA.name, ITEM, 1)
    return Outcome(run_dir, trial_id, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir)), log)


def reference_result(stdout: str = LAYOUT, **flags: bool) -> RunResult:
    """Return a SYNTHETIC clean reference run with `flags` set on the RunResult."""
    return RunResult(exit_code=0, hang=False, stdout=stdout, stderr="", wall_s=PLACEHOLDER_REFERENCE_WALL_S, **flags)


def references(**target_flags: bool) -> dict[str, RunResult]:
    """Return the reference runs by language: the target's with `target_flags`, the source's clean."""
    return {TARGET: reference_result(**target_flags), SOURCE: reference_result(SOURCE_REFERENCE_STDOUT)}


def clean_run(stdout: str = LAYOUT_OTHER_TIMES, **flags: bool) -> RunResult:
    """Return a SYNTHETIC attempt run that exited 0, with `flags` set on the RunResult."""
    return RunResult(exit_code=0, hang=False, stdout=stdout, stderr="", wall_s=PLACEHOLDER_ATTEMPT_WALL_S, **flags)


def failed_run(stdout: str = LAYOUT_FAIL, **flags: bool) -> RunResult:
    """Return a SYNTHETIC attempt run that exited nonzero, with `flags` set on the RunResult."""
    return RunResult(
        exit_code=RUN_ERROR_EXIT, hang=False, stdout=stdout, stderr="SYNTHETIC run error\n",
        wall_s=PLACEHOLDER_ATTEMPT_WALL_S, **flags,
    )


# ---------------------------------------------------------------------------
# SYNTHETIC records built by hand (trial.md, Parquet, older trial.json)


def run_info(store: TextStore, stdout: str, flags: Mapping[str, bool | None] | None = None) -> RunInfo:
    """Return a SYNTHETIC RunInfo of a clean run with its stdout in `store`; `flags` sets the run flags when given."""
    return RunInfo(exit_code=0, hang=False, wall_s=PLACEHOLDER_ATTEMPT_WALL_S, stdout_ref=store.put(stdout),
                   **(flags or {}))


def hand_trial(store: TextStore, reference: RunInfo, attempt_runs: Sequence[RunInfo]) -> Trial:
    """Return a SYNTHETIC trial of layout with `reference` as its reference run and one S5 attempt per run."""
    attempts = [
        Attempt(index=index, prompt_ref=store.put(f"SYNTHETIC prompt {index}\n"), response_text=blocks(GOOD_CODE),
                files={target_file(): GOOD_CODE}, stage_reached="S5", run=run)
        for index, run in enumerate(attempt_runs)
    ]
    return Trial(
        trial_id=make_trial_id("run-flags", MODEL_ID, SUITE, OMP_TO_CUDA.name, ITEM, 1),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=FAKE_COMMIT, dirty=False, device="scripted (SYNTHETIC)", sdk=None,
                              date="2026-09-24T00:00:00+00:00"),
        bench_item=load_suite(SUITE_MANIFEST).bench_item(ITEM, OMP_TO_CUDA),
        model=ModelInfo(backend="scripted", id=MODEL_ID, sampling=SAMPLING),
        reference_run=reference,
        attempts=attempts,
        final=Final(stage_reached="S5" if attempts else None, corrections=max(len(attempts) - 1, 0)),
    )


def table_rows(page: str, heading: str, after: str | None = None) -> dict[str, str]:
    """Return the Field/Value rows of the table under the line `heading` (the first after the line `after`)."""
    lines = page.split("\n")
    start = lines.index(heading, lines.index(after) if after is not None else 0)
    rows: dict[str, str] = {}
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            break
        if line.startswith("| ") and not line.startswith(("| ---", "| Field |")):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            rows[cells[0]] = cells[1]
    return rows


# Mixed SYNTHETIC flag values for the hand-built records: the reference and two attempt runs.
REFERENCE_FLAGS = {"stdout_truncated": True, "stderr_truncated": False, "workdir_incomplete": True}
ATTEMPT_FLAGS = (
    {"stdout_truncated": False, "stderr_truncated": True, "workdir_incomplete": False},
    {"stdout_truncated": False, "stderr_truncated": False, "workdir_incomplete": False},
)


def flagged_trial(store: TextStore) -> Trial:
    """Return a SYNTHETIC trial whose reference run and two attempt runs record REFERENCE_FLAGS and ATTEMPT_FLAGS."""
    require_flag_fields()
    reference = run_info(store, LAYOUT, REFERENCE_FLAGS)
    return hand_trial(store, reference, [run_info(store, LAYOUT_OTHER_TIMES, flags) for flags in ATTEMPT_FLAGS])


def md(value: bool) -> str:
    """Return a bool as trial.md writes it."""
    return "true" if value else "false"


# ---------------------------------------------------------------------------
# The record: RunInfo's three flags


def test_run_info_has_the_three_flags_each_none_when_not_recorded() -> None:
    assert flags_of(RunInfo()) == NOT_RECORDED, "a RunInfo that records no run holds no flag"
    for flag in FLAGS:
        for value in (True, False, None):
            assert getattr(RunInfo(**{flag: value}), flag) is value
    assert flags_of(Attempt(index=0, stage_reached="S0").run) == NOT_RECORDED, "an attempt that never ran"


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("value", [1, 0, "true", 1.0], ids=["int-1", "int-0", "str", "float"])
def test_a_run_flag_is_a_bool_or_none_and_nothing_else(flag: str, value: Any) -> None:
    require_flag_fields()
    with pytest.raises(ValueError, match=flag):
        RunInfo(**{flag: value})


def test_the_flags_round_trip_through_json_and_trial_json(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    store = TextStore(run_root)
    trial = flagged_trial(store)
    assert from_json(RunInfo, to_json(trial.reference_run)) == trial.reference_run
    out = write_trial(trial, run_root, store)
    data = json.loads((out / "trial.json").read_text(encoding="utf-8"))
    assert {flag: data["reference_run"][flag] for flag in FLAGS} == REFERENCE_FLAGS
    assert [{flag: attempt["run"][flag] for flag in FLAGS} for attempt in data["attempts"]] == list(ATTEMPT_FLAGS)
    assert read_trial(out, store) == trial


# ---------------------------------------------------------------------------
# The baseline fills Trial.reference_run's flags


@pytest.mark.parametrize("flag", [None, *FLAGS], ids=["no-flag", *FLAGS])
def test_the_baseline_records_the_target_reference_runs_flags(tmp_path: Path, flag: str | None) -> None:
    target_flags = {} if flag is None else {flag: True}
    log = Log(replies=[blocks(GOOD_CODE)], references=references(**target_flags), attempt_runs=[clean_run()])
    outcome = run_one(tmp_path, "reference-flags", recipe_data(), log)
    reference = outcome.trial.reference_run
    assert flags_of(reference) == only(flag), (
        "each flag as the target reference's RunResult set it; a flag the run did not set is recorded False"
    )
    assert reference.stdout_ref is not None and outcome.store.get(reference.stdout_ref) == LAYOUT
    assert outcome.trial.final.end_reason is None, "a reference run whose output was cut still lets the trial go on"
    assert outcome.trial.attempts[-1].stage_reached == "S5"


def test_the_source_reference_runs_flags_never_reach_the_reference_run(tmp_path: Path) -> None:
    every_flag = dict.fromkeys(FLAGS, True)
    log = Log(replies=[blocks(GOOD_CODE)], attempt_runs=[clean_run()],
              references={TARGET: reference_result(), SOURCE: reference_result(SOURCE_REFERENCE_STDOUT, **every_flag)})
    outcome = run_one(tmp_path, "source-flags", recipe_data(), log)
    assert log.reference_runs == [TARGET, SOURCE], "fixes.baseline_both on: the target reference runs, then the source"
    assert flags_of(outcome.trial.reference_run) == RECORDED_CLEAR, "Trial.reference_run is the target's run alone"


def test_with_a_compile_only_executor_every_flag_stays_none(tmp_path: Path) -> None:
    log = Log(replies=[blocks(BAD_CODE), blocks(GOOD_CODE)])
    outcome = run_one(tmp_path, "compile-only-flags", recipe_data(executor="none"), log)
    assert [attempt.stage_reached for attempt in outcome.trial.attempts] == ["S1", "S4"]
    assert flags_of(outcome.trial.reference_run) == NOT_RECORDED, "no reference ran"
    for attempt in outcome.trial.attempts:
        assert flags_of(attempt.run) == NOT_RECORDED, f"attempt {attempt.index} never ran"


# ---------------------------------------------------------------------------
# run_loop fills Attempt.run's flags and keeps the warnings of P1.6


@pytest.mark.parametrize("flag", [None, *FLAGS], ids=["no-flag", *FLAGS])
def test_run_loop_records_the_attempt_runs_flags_and_keeps_its_warning(tmp_path: Path, flag: str | None) -> None:
    run_flags = {} if flag is None else {flag: True}
    log = Log(replies=[blocks(GOOD_CODE)], references=references(), attempt_runs=[clean_run(**run_flags)])
    outcome = run_one(tmp_path, "attempt-flags", recipe_data(), log)
    (attempt,) = outcome.trial.attempts
    assert attempt.stage_reached == "S5"
    assert flags_of(attempt.run) == only(flag), "each flag as the attempt's RunResult set it; unset ones read False"
    for name, code in WARNING_CODES.items():
        expected = [("run", "warning")] if name == flag else []
        assert codes(attempt, code) == expected, f"the P1.6 run-stage warning {code} stays, once, only for {name}"


def test_each_attempt_records_its_own_runs_flags_and_one_that_never_ran_keeps_none(tmp_path: Path) -> None:
    runs = [failed_run(workdir_incomplete=True), clean_run(stderr_truncated=True)]
    log = Log(replies=[blocks(BAD_CODE), blocks(GOOD_CODE), blocks(GOOD_CODE)], references=references(),
              attempt_runs=runs)
    outcome = run_one(tmp_path, "mixed-flags", recipe_data(), log)
    attempts = outcome.trial.attempts
    assert [attempt.stage_reached for attempt in attempts] == ["S1", "S4", "S5"]
    assert flags_of(attempts[0].run) == NOT_RECORDED, "attempt 0 did not compile, so it never ran"
    assert flags_of(attempts[1].run) == only("workdir_incomplete"), "a failed run records its flags too"
    assert flags_of(attempts[2].run) == only("stderr_truncated")
    assert flags_of(outcome.trial.reference_run) == RECORDED_CLEAR


# ---------------------------------------------------------------------------
# The oracle stage, end to end: a cut reference stdout is never aligned against


@pytest.mark.parametrize("truncated", [True, False], ids=["reference-cut", "reference-whole"])
def test_end_to_end_no_run_is_aligned_against_a_reference_stdout_marked_truncated(
    tmp_path: Path, truncated: bool
) -> None:
    log = Log(replies=[blocks(GOOD_CODE), blocks(GOOD_CODE)], references=references(stdout_truncated=truncated),
              attempt_runs=[failed_run(), clean_run()])
    outcome = run_one(tmp_path, "oracle-cut", recipe_data(stages=ORACLE_STAGES, oracle=ORACLE), log)
    trial = outcome.trial
    assert flags_of(trial.reference_run) == only("stdout_truncated" if truncated else None)
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S4", "S5"]
    warnings = [codes(attempt, REFERENCE_TRUNCATED) for attempt in trial.attempts]
    if truncated:
        assert [attempt.alignment for attempt in trial.attempts] == [Alignment(), Alignment()], "alignments unset"
        assert warnings == [[("run", "warning")]] * 2, "one run-stage warning on each attempt that ran"
    else:
        assert [attempt.alignment.mean for attempt in trial.attempts] == [0.0, 1.0], "aligned as before"
        assert warnings == [[], []]


# ---------------------------------------------------------------------------
# trial.md and the Parquet mirror show the flags


def test_trial_md_shows_the_flags_in_the_reference_run_and_in_each_attempts_run(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "run")
    page = render_trial_md(flagged_trial(store), store)
    reference = table_rows(page, "## Reference run")
    assert {flag: reference.get(flag) for flag in FLAGS} == {flag: md(REFERENCE_FLAGS[flag]) for flag in FLAGS}
    for index, flags in enumerate(ATTEMPT_FLAGS):
        run = table_rows(page, "### Run", after=f"## Attempt {index}")
        assert {flag: run.get(flag) for flag in FLAGS} == {flag: md(flags[flag]) for flag in FLAGS}, f"attempt {index}"


def test_trial_md_never_shows_a_flag_that_was_not_recorded_as_true_or_false(tmp_path: Path) -> None:
    require_flag_fields()
    store = TextStore(tmp_path / "run")
    trial = hand_trial(store, run_info(store, LAYOUT), [run_info(store, LAYOUT_OTHER_TIMES)])
    page = render_trial_md(trial, store)
    tables = {"reference run": table_rows(page, "## Reference run"),
              "attempt 0 run": table_rows(page, "### Run", after="## Attempt 0")}
    for where, rows in tables.items():
        for flag in FLAGS:
            assert flag in rows, f"trial.md shows {flag} in the {where} table"
            assert rows[flag] not in ("true", "false"), f"{where}: a flag not recorded never reads as {rows[flag]}"


def test_the_parquet_flag_columns_are_bool_columns_named_for_their_field() -> None:
    for table, columns in (("trials", REFERENCE_RUN_COLUMNS), ("attempts", ATTEMPT_RUN_COLUMNS)):
        schema = parquet.SCHEMAS[table]
        for column in columns.values():
            assert column in schema.names, f"the Parquet {table} table has no {column} column"
            assert schema.field(column).type == pa.bool_(), f"{table}.{column} is a bool column"


def test_the_parquet_rows_carry_each_flag_and_read_back(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "run")
    trial = flagged_trial(store)
    rows = parquet.trial_rows([trial])
    (trial_row,) = rows["trials"]
    assert {flag: trial_row.get(column) for flag, column in REFERENCE_RUN_COLUMNS.items()} == REFERENCE_FLAGS
    attempt_rows = sorted(rows["attempts"], key=lambda row: row["index"])
    got = [{flag: row.get(column) for flag, column in ATTEMPT_RUN_COLUMNS.items()} for row in attempt_rows]
    assert got == list(ATTEMPT_FLAGS)
    out = tmp_path / "parquet"
    parquet.write_run_parquet([trial], out)
    read = parquet.read_run_parquet(out)
    assert (read["trials"], read["attempts"]) == (rows["trials"], rows["attempts"]), "the flags read back as written"


# ---------------------------------------------------------------------------
# A trial.json written before the flags existed


def test_a_trial_json_written_before_the_flags_existed_loads_with_every_flag_none(tmp_path: Path) -> None:
    require_flag_fields()
    run_root = tmp_path / "run"
    store = TextStore(run_root)
    trial = hand_trial(store, run_info(store, LAYOUT), [run_info(store, LAYOUT_FAIL), run_info(store, LAYOUT)])
    out = write_trial(trial, run_root, store)
    data = json.loads((out / "trial.json").read_text(encoding="utf-8"))
    for run in [data["reference_run"], *(attempt["run"] for attempt in data["attempts"])]:
        assert set(FLAGS) <= set(run), "trial.json holds every RunInfo field"
        for flag in FLAGS:
            del run[flag]
    (out / "trial.json").write_bytes(json_text(data).encode("ascii"))
    loaded = read_trial(out, store)
    assert flags_of(loaded.reference_run) == NOT_RECORDED
    assert [flags_of(attempt.run) for attempt in loaded.attempts] == [NOT_RECORDED, NOT_RECORDED]
    assert loaded == trial, "every other field loads as written"
    rows = parquet.trial_rows([loaded])
    assert all(rows["trials"][0][column] is None for column in REFERENCE_RUN_COLUMNS.values())
    assert all(row[column] is None for row in rows["attempts"] for column in ATTEMPT_RUN_COLUMNS.values())
    assert render_trial_md(loaded, store).isascii(), "trial.md still renders an older record"


# ---------------------------------------------------------------------------
# projects/base.yaml: the sandbox.wall_s comment


def test_the_base_recipe_wall_s_comment_names_the_floor_and_the_no_reference_case() -> None:
    text = BASE_RECIPE.read_text(encoding="utf-8")
    assert yaml.safe_load(text)["sandbox"]["wall_s"] == "baseline_x10"
    lines = text.splitlines()
    wall = next(line for line in lines[lines.index("sandbox:") + 1 :] if line.strip().startswith("wall_s:"))
    assert "#" in wall, "sandbox.wall_s carries a comment"
    comment = wall.split("#", 1)[1]
    floor = f"{RUN_WALL_FLOOR_S:g} s"
    assert floor in comment, f"the comment names the floor of {floor} (RUN_WALL_FLOOR_S): {comment!r}"
    assert re.search(r"\bno reference\b", comment), f"the comment names the case where no reference ran: {comment!r}"
