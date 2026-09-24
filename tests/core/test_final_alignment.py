"""Tests for the output that stands and final.alignment (task P2.3).

Bible: Result Record (Attempt.alignment, Final: stage_reached, alignment,
end_reason), Oracles (the stale output rule: the oracle stage aligns every
attempt whose run holds stdout, stale output included), Source Papers (LASSI
quirk table, loop row: past the execution gate the last attempt that ran
stands as the trial's output; with none, upstream-crash).

The contract these tests fix, from the P2.3 acceptance criteria
(plans/p2-scoring.md), the planning decision "The output that stands is the
last attempt that ran, stale output included, as the P1.9 replay reads it;
final.alignment is its alignment mean", and the P2 phase note "Final
alignment":

- One shared helper, `standing_attempt(trial)` in lassi.core.record, returns
  the Attempt whose output stands: the last attempt that ran (its
  Attempt.run recorded a run, run.stdout_ref set, as the replay and the
  stale-output warning read it), whether or not later attempts exist that
  did not run. It returns None when no attempt ran: a compile-only trial, an
  upstream-crash, or a trial with no attempts.
- The runner sets final.alignment to that attempt's alignment mean (None
  when the helper returns None or the attempt was never aligned).
- trial.md's summary Alignment row and the Parquet trials column
  final_alignment show that value.

Five cases, each checked on hand-built records (the helper) and end to end
through the runner (the helper and final.alignment): a clean finish, stale
output past the execution gate, upstream-crash, a run error at the
correction cap, and a compile-only trial.

Under the runner's stage rule, a stage that sets final.end_reason ends the
trial, so the oracle stage does not run after a correction-cap end; the run
error at the cap is therefore checked as "final.alignment equals the
standing attempt's alignment mean", whatever the oracle left there.

The runs use the p0-smoke template prompt set, so no upstream text is
needed. Fake toolchains write the files and a PLACEHOLDER artifact and
compile nothing; the scripted executor answers with SYNTHETIC run results and
runs nothing; the scripted backend answers with SYNTHETIC replies. Program
output is SYNTHETIC text in layout's print format, and wall times are
PLACEHOLDER fixture values. The alignments are the real stdout_mask oracle's
outputs on those SYNTHETIC texts. No value in this module is a measurement.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import Direction, load_suite
from lassi.core import runner as runner_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.parquet import read_run_parquet
from lassi.core.record import (
    Alignment,
    Attempt,
    Diagnostic,
    EndReason,
    Final,
    ModelInfo,
    Provenance,
    RunInfo,
    Trial,
    make_trial_id,
)
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.store import TRIAL_MD, TextStore, read_trial, trial_dir
from lassi.core.trial_md import PLACEHOLDER

HELPER_MODULE = "lassi.core.record"
HELPER_NAME = "standing_attempt"

REPO = Path(__file__).resolve().parents[2]
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
MODEL_ID = "scripted-fixture"
TEMPLATE_SET = "p0-smoke"
OMP_TO_CUDA = Direction("omp", "cuda")
TARGET, SOURCE = OMP_TO_CUDA.target, OMP_TO_CUDA.source
TOOLCHAINS = {"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"}
STAGES = ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
ORACLE = {"kind": "stdout_mask", "passfail": True}
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
# A commit id for synthetic provenance; not a commit of this repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
STALE_OUTPUT = "stale-output"
UPSTREAM_CRASH = "upstream-crash"
CORRECTION_CAP = "correction-cap"

# SYNTHETIC bench sources and replies; the fake toolchains refuse any file holding an `#error` line.
OMP_SOURCE = '#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  std::printf("done\\n");\n}\n'
CUDA_SOURCE = "#include <cstdio>\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n"
SOURCES = {"omp": OMP_SOURCE, "cuda": CUDA_SOURCE}
BAD_CODE = "#error SYNTHETIC not translated yet\nint main() { return 1; }\n"
GOOD_CODE = "int main() { return 0; }\n"
COMPILE_ERROR = Diagnostic(stage="compile", severity="error", code="synthetic", message="SYNTHETIC compile error")

# SYNTHETIC program output in layout's print format (values invented), and PLACEHOLDER wall times in seconds.
# Against LAYOUT, the stdout_mask oracle with passfail gives LAYOUT_OTHER_TIMES 1.0 and LAYOUT_FAIL 0.0.
LAYOUT = "Average kernel execution time (AoS): 1.5 (us)\nPASS\nAverage kernel execution time (SoA): 2.5 (us)\nPASS\n"
LAYOUT_OTHER_TIMES = LAYOUT.replace("1.5 (us)", "3.75 (us)").replace("2.5 (us)", "0.5 (us)")
LAYOUT_FAIL = LAYOUT_OTHER_TIMES.replace("PASS", "FAIL")
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
# The helper, looked up so that a missing one fails each test with a clear message


def standing_attempt(trial: Trial) -> Attempt | None:
    """Call the shared helper on `trial`; fail the test clearly while it does not exist."""
    helper = getattr(importlib.import_module(HELPER_MODULE), HELPER_NAME, None)
    if helper is None:
        pytest.fail(
            f"{HELPER_MODULE} has no {HELPER_NAME}; task P2.3 adds the one shared helper that names the attempt "
            "whose output stands"
        )
    return helper(trial)


def assert_stands(trial: Trial, index: int | None) -> Attempt | None:
    """Assert that the helper names attempt `index` of `trial` (None: no attempt), and return what it named."""
    named = standing_attempt(trial)
    if index is None:
        assert named is None, f"no attempt's output stands, but the helper named {named!r}"
        return None
    assert isinstance(named, Attempt), f"the helper returns the Attempt whose output stands, got {named!r}"
    assert named.index == index and named == trial.attempts[index], f"attempt {index}'s output stands"
    return named


# ---------------------------------------------------------------------------
# Hand-built SYNTHETIC records


def ran(store: TextStore, stdout: str, exit_code: int = 0) -> RunInfo:
    """Return a SYNTHETIC RunInfo of a run that happened, its stdout kept in `store`."""
    return RunInfo(exit_code=exit_code, hang=False, wall_s=PLACEHOLDER_ATTEMPT_WALL_S, stdout_ref=store.put(stdout),
                   stdout_truncated=False, stderr_truncated=False, workdir_incomplete=False)


def aligned(value: float | None) -> Alignment:
    """Return the Alignment the oracle stage gives one run of value `value` (unset for None)."""
    return Alignment() if value is None else Alignment(per_input=[value], mean=value)


def compiled(index: int, run: RunInfo | None = None, value: float | None = None, **fields: Any) -> Attempt:
    """Return a SYNTHETIC attempt that compiled: S5 after a clean run, else S4, with its alignment."""
    run = run or RunInfo()
    stage = "S5" if run.exit_code == 0 and run.hang is False else "S4"
    return Attempt(index=index, response_text="SYNTHETIC reply", files={"main.cu": GOOD_CODE}, stage_reached=stage,
                   run=run, alignment=aligned(value), **fields)


def compile_failed(index: int) -> Attempt:
    """Return a SYNTHETIC attempt that parsed (S1) but did not compile, so it never ran."""
    return Attempt(index=index, response_text="SYNTHETIC reply", files={"main.cu": BAD_CODE}, stage_reached="S1",
                   diagnostics=[COMPILE_ERROR])


def stale_warning(index: int) -> Diagnostic:
    """Return a SYNTHETIC stale-output warning naming attempt `index`."""
    return Diagnostic(stage="run", severity="warning", code=STALE_OUTPUT,
                      message=f"SYNTHETIC: the stdout of attempt {index} stands as the trial's output")


def hand_trial(attempts: Sequence[Attempt], end: str | None = None) -> Trial:
    """Return a SYNTHETIC trial of layout holding `attempts`, ended with the code `end` when given."""
    reason = None if end is None else EndReason(code=end, message=f"SYNTHETIC end: {end}")
    return Trial(
        trial_id=make_trial_id("final-alignment", MODEL_ID, SUITE, OMP_TO_CUDA.name, ITEM, 1),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=FAKE_COMMIT, dirty=False, device="scripted (SYNTHETIC)", sdk=None,
                              date="2026-09-24T00:00:00+00:00"),
        bench_item=load_suite(SUITE_MANIFEST).bench_item(ITEM, OMP_TO_CUDA),
        model=ModelInfo(backend="scripted", id=MODEL_ID, sampling=SAMPLING),
        attempts=list(attempts),
        final=Final(stage_reached=attempts[-1].stage_reached if attempts else None,
                    corrections=max(len(attempts) - 1, 0), end_reason=reason),
    )


def test_the_helper_names_the_last_attempt_after_a_clean_finish(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "store")
    trial = hand_trial([
        compile_failed(0),
        compiled(1, ran(store, LAYOUT_FAIL, exit_code=RUN_ERROR_EXIT), 0.0),
        compiled(2, ran(store, LAYOUT_OTHER_TIMES), 1.0),
    ])
    assert_stands(trial, 2)


def test_the_helper_names_the_last_attempt_that_ran_when_stale_output_stands(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "store")
    attempts = [
        compiled(0, ran(store, LAYOUT_OTHER_TIMES, exit_code=RUN_ERROR_EXIT), 1.0),
        compiled(1, ran(store, LAYOUT_FAIL, exit_code=RUN_ERROR_EXIT), 0.0),
        *(compile_failed(index) for index in range(2, 8)),
        compiled(8, diagnostics=[stale_warning(1)]),
    ]
    assert_stands(hand_trial(attempts), 1)


def test_the_helper_names_no_attempt_after_upstream_crash() -> None:
    trial = hand_trial([*(compile_failed(index) for index in range(8)), compiled(8)], end=UPSTREAM_CRASH)
    assert_stands(trial, None)


def test_the_helper_names_the_last_attempt_after_a_run_error_at_the_cap(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "store")
    trial = hand_trial([
        compiled(0, ran(store, LAYOUT_FAIL, exit_code=RUN_ERROR_EXIT)),
        compiled(1, ran(store, LAYOUT_FAIL, exit_code=RUN_ERROR_EXIT)),
    ], end=CORRECTION_CAP)
    assert_stands(trial, 1)


def test_the_helper_names_no_attempt_in_a_compile_only_trial() -> None:
    assert_stands(hand_trial([compile_failed(0), compiled(1)]), None)


def test_the_helper_names_no_attempt_when_the_trial_has_none() -> None:
    assert_stands(hand_trial([], end="baseline-compile"), None)


# ---------------------------------------------------------------------------
# Fake components for the runner


@dataclass
class Log:
    """The script the fakes follow: replies, the reference run per language, and the attempt runs in order."""

    replies: list[str] = field(default_factory=list)
    references: dict[str, RunResult] = field(default_factory=dict)
    attempt_runs: list[RunResult] = field(default_factory=list)


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
        """Returns the SYNTHETIC RunResult of each run; runs nothing."""

        name = "scripted"
        capabilities = frozenset({"runs_code", "sandboxed"})

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Return the reference result of the artifact's language, or the next scripted attempt result."""
            folder = Path(artifact).parent.parent.name
            if folder.startswith("baseline-"):
                return log.references[folder[len("baseline-") :]]
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


def make_registry(log: Log) -> Registry:
    """Return a test Registry: the fakes, the real stdout_mask oracle, and the real stages."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(log))
    registry.register("Executor", "none", CompileOnlyExecutor)
    registry.register("Executor", "scripted", scripted_executor(log))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, fake_toolchain(name))
    registry.register("Oracle", ORACLE["kind"], DEFAULT_REGISTRY.get("Oracle", ORACLE["kind"]).factory)
    for name in STAGES:
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


def blocks(code: str) -> str:
    """Return a SYNTHETIC reply holding `code` as the target's one FILE block."""
    target = load_suite(SUITE_MANIFEST).items[ITEM].languages[TARGET].files[0]
    return render_file_blocks({target: code})


def recipe_data(*, executor: str = "scripted", **changes: Any) -> dict[str, Any]:
    """Return a one-trial template-set recipe for layout, omp to cuda, with the oracle stage last."""
    data: dict[str, Any] = {
        "extends": "base",
        "faithful": False,
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": SOURCE, "target": TARGET}],
        "prompts": TEMPLATE_SET,
        "toolchain": dict(TOOLCHAINS),
        "stages": list(STAGES),
        "executor": {"kind": executor},
        "oracle": dict(ORACLE),
        "trials": {"n": 1},
    }
    data.update(changes)
    return data


def references() -> dict[str, RunResult]:
    """Return the SYNTHETIC clean reference run of each language; the target's prints LAYOUT."""
    return {
        language: RunResult(exit_code=0, hang=False, stdout=LAYOUT if language == TARGET else "SYNTHETIC\n",
                            stderr="", wall_s=PLACEHOLDER_REFERENCE_WALL_S)
        for language in (TARGET, SOURCE)
    }


def clean_run(stdout: str = LAYOUT_OTHER_TIMES) -> RunResult:
    """Return a SYNTHETIC attempt run that exited 0."""
    return RunResult(exit_code=0, hang=False, stdout=stdout, stderr="", wall_s=PLACEHOLDER_ATTEMPT_WALL_S)


def failed_run(stdout: str = LAYOUT_FAIL) -> RunResult:
    """Return a SYNTHETIC attempt run that exited nonzero."""
    return RunResult(exit_code=RUN_ERROR_EXIT, hang=False, stdout=stdout, stderr="SYNTHETIC run error\n",
                     wall_s=PLACEHOLDER_ATTEMPT_WALL_S)


@dataclass
class Outcome:
    """One finished run: its directory, its one trial's id, and the trial read back from trial.json."""

    run_dir: Path
    trial_id: str
    trial: Trial

    def summary_alignment(self) -> str:
        """Return the value of the Alignment row in trial.md's summary table."""
        page = (trial_dir(self.run_dir, self.trial_id) / TRIAL_MD).read_text(encoding="utf-8")
        match = re.search(r"^\| Alignment \| (.*) \|$", page, flags=re.MULTILINE)
        assert match is not None, "trial.md's summary table has an Alignment row"
        return match.group(1)

    def parquet_final_alignment(self) -> Any:
        """Return the final_alignment value of the trial's row in the Parquet trials table."""
        rows = read_run_parquet(self.run_dir / runner_module.PARQUET_DIR)["trials"]
        (row,) = [row for row in rows if row["trial_id"] == self.trial_id]
        return row["final_alignment"]


def run_one(tmp_path: Path, name: str, data: Mapping[str, Any], log: Log) -> Outcome:
    """Run the one-trial recipe `data`, saved as `<name>.yaml`, with the fakes following `log`; return the outcome."""
    bench = write_bench(tmp_path / f"{name}-bench")
    options = RunOptions(runs_root=tmp_path / name / "runs-root", run_id="test-run", bench_root=bench,
                         registry=make_registry(log))
    path = tmp_path / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    run_dir = run_recipe(path, options)
    trial_id = make_trial_id(name, MODEL_ID, SUITE, OMP_TO_CUDA.name, ITEM, 1)
    return Outcome(run_dir, trial_id, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir)))


def end_code(trial: Trial) -> str | None:
    """Return final.end_reason's code, or None when the trial ended normally."""
    return None if trial.final.end_reason is None else trial.final.end_reason.code


def names_attempt(message: str, index: int) -> bool:
    """Return True when `message` names attempt `index`: the phrase "attempt <index>", the index a whole number."""
    return re.search(rf"\battempt {index}(?![0-9])", message, re.IGNORECASE) is not None


# ---------------------------------------------------------------------------
# The runner sets final.alignment from the attempt the helper names


def test_a_clean_finish_gives_the_last_attempts_alignment(tmp_path: Path) -> None:
    log = Log(replies=[blocks(GOOD_CODE)] * 2, references=references(), attempt_runs=[failed_run(), clean_run()])
    outcome = run_one(tmp_path, "clean-finish", recipe_data(), log)
    trial = outcome.trial
    assert [(attempt.stage_reached, attempt.alignment.mean) for attempt in trial.attempts] == [("S4", 0.0), ("S5", 1.0)]
    assert end_code(trial) is None
    assert_stands(trial, 1)
    assert trial.final.alignment == 1.0, "final.alignment is the alignment mean of the attempt whose output stands"
    assert outcome.summary_alignment() == "1.0", "trial.md's summary shows it"
    assert outcome.parquet_final_alignment() == 1.0, "the Parquet final_alignment column shows it"


def test_stale_output_past_the_gate_gives_the_alignment_of_the_last_attempt_that_ran(tmp_path: Path) -> None:
    # Attempt 0's run fails but prints the reference's lines (1.0); attempts 1 to 7 fail and print FAIL (0.0);
    # attempt 8 compiles past the execution gate and is not run, so attempt 7's output stands.
    runs = [failed_run(LAYOUT_OTHER_TIMES), *(failed_run() for _ in range(7))]
    log = Log(replies=[blocks(GOOD_CODE)] * 9, references=references(), attempt_runs=runs)
    outcome = run_one(tmp_path, "stale-output", recipe_data(fixes={"execution_gate": False}), log)
    trial = outcome.trial
    means = [attempt.alignment.mean for attempt in trial.attempts]
    assert means == [1.0, *[0.0] * 7, None], "the oracle aligned every run; the unexecuted attempt has none"
    assert end_code(trial) is None
    (stale,) = [item for item in trial.attempts[8].diagnostics if item.code == STALE_OUTPUT]
    named = assert_stands(trial, 7)
    assert named is not None and names_attempt(stale.message, named.index), (
        "the helper and the stale-output warning name the same attempt"
    )
    assert trial.final.alignment == 0.0, "the last attempt that ran stands: neither the unexecuted attempt nor the best"
    assert outcome.summary_alignment() == "0.0"
    assert outcome.parquet_final_alignment() == 0.0


def test_upstream_crash_has_no_output_and_no_final_alignment(tmp_path: Path) -> None:
    log = Log(replies=[*[blocks(BAD_CODE)] * 8, blocks(GOOD_CODE)], references=references())
    outcome = run_one(tmp_path, "upstream-crash", recipe_data(fixes={"execution_gate": False}), log)
    trial = outcome.trial
    assert end_code(trial) == UPSTREAM_CRASH
    assert_stands(trial, None)
    assert trial.final.alignment is None
    assert outcome.summary_alignment() == PLACEHOLDER
    assert outcome.parquet_final_alignment() is None


def test_a_run_error_at_the_cap_gives_the_alignment_of_the_last_attempt(tmp_path: Path) -> None:
    log = Log(replies=[blocks(GOOD_CODE)] * 2, references=references(), attempt_runs=[failed_run(), failed_run()])
    outcome = run_one(tmp_path, "cap-run-error", recipe_data(loop={"max_corrections": 1}), log)
    trial = outcome.trial
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S4", "S4"]
    assert end_code(trial) == CORRECTION_CAP
    named = assert_stands(trial, 1)
    assert named is not None and trial.final.alignment == named.alignment.mean, (
        "final.alignment is the last attempt's alignment mean, as the oracle left it"
    )
    assert trial.final.alignment is None, "a trial that ends at correction-cap never reaches the oracle ([DESIGN])"
    assert outcome.summary_alignment() == PLACEHOLDER
    assert outcome.parquet_final_alignment() is None


def test_a_compile_only_trial_has_no_output_and_no_final_alignment(tmp_path: Path) -> None:
    log = Log(replies=[blocks(BAD_CODE), blocks(GOOD_CODE)])
    outcome = run_one(tmp_path, "compile-only", recipe_data(executor="none"), log)
    trial = outcome.trial
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S1", "S4"]
    assert end_code(trial) is None
    assert_stands(trial, None)
    assert trial.final.alignment is None
    assert outcome.summary_alignment() == PLACEHOLDER
    assert outcome.parquet_final_alignment() is None
