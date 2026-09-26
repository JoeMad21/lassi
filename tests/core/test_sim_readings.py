"""Tests for the simulator readings of the baseline and run_loop (task P4.6).

Bible: ttsim Facts (UndefinedBehavior is real feedback for the model;
UnimplementedFunctionality and UnsupportedFunctionality are simulator gaps,
not model errors, so the trial stops with the end reason sim-gap), Harness Contract
(kernel JIT errors surface during execution and the executor reclassifies
them as stage jit; a timeout returns a hang diagnostic naming a likely
circular-buffer or semaphore deadlock), Result Record (RunInfo.sim_ub,
final.end_reason), Reward Function (stage table: S1 parses, S4 compiles with
host and kernel JIT, S5 runs clean with no crash, UB, or hang), Component
Interfaces (Executor, RunResult), Agent Rule 2. Plan: plans/p4-ttsim.md,
P4.6, and its planning decision "Simulator readings (P4.6)".

The contract these tests fix:

- RunResult (lassi.core.interfaces) carries the simulator's findings in
  three fields: `sim_ub`, True when the simulator reported
  UndefinedBehavior; `sim_gap`, the class of a simulator gap
  ("UnimplementedFunctionality" or "UnsupportedFunctionality"), else None;
  and `diagnostics`, the Diagnostics the executor parsed from the run, kernel
  JIT messages among them with stage "jit". A RunResult built without them,
  as every executor before this task builds one, reads as no finding.
- lassi.core.record END_REASONS gains `sim-gap`, spelled with a hyphen as
  every end reason code is; the underscore spelling `sim_gap`, the name of
  the RunResult field, is not a record code and EndReason refuses it.
- baseline: a reference run with UB ends the trial with `baseline-run`, its
  message naming the undefined behavior, even when the run exited 0, and
  Trial.reference_run.sim_ub records True for the target reference; a
  reference run with a gap ends the trial with `sim-gap`, its message naming
  the gap's class. Either way no model is asked.
- run_loop:
  - a kernel JIT failure (a jit-stage error among RunResult.diagnostics)
    leaves the attempt at S1 with those errors and no run-error, and it is
    corrected as a compile error is: compile_loop asks for the correction
    (Request.stage compile_loop) with the jit error in its prompt, under the
    same correction cap;
  - jit-stage warnings of a clean run stay on the attempt, which is S5;
  - UB is a failed run fed back to the model: the attempt stays S4, even
    after exit status 0, with a run-stage error naming the undefined
    behavior, Attempt.run.sim_ub is True, and run_loop asks for the
    correction with that error in its prompt;
  - a gap ends the trial with `sim-gap` (its message naming the class): the
    attempt stays S4 and no correction is asked;
  - a hang carries the Harness Contract's hang diagnostic: a run-stage error
    naming a likely circular-buffer or semaphore deadlock, which the
    correction prompt carries;
  - a clean simulator run records sim_ub False, and a run on an executor
    that reports no finding keeps sim_ub None (not recorded), as before.
- Records: a trial that ended at sim-gap round-trips through trial.json, and
  a trial.json written before this task (without any key it adds) loads
  unchanged. With `score: df-v0`, a trial that ended at sim-gap after an
  attempt ran gets final.score null and the gap attempt's R null, while the
  attempt before it keeps its own R (tests/scoring/test_sim_gap_scoring.py
  pins that reading on hand-built records).

The fake simulator executor declares runs_code, sandboxed, and simulator; it
starts no process, opens no device, and returns SYNTHETIC RunResults whose
stderr names no finding, so any wording about a finding comes from the
stages. The suite manifest, sources, replies, diagnostics, and program output
are SYNTHETIC; fake toolchains compile nothing. Wall times are PLACEHOLDER
fixture values. No value in this module is a measurement.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.core import record as record_module
from lassi.core import runner as runner_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.record import (
    Attempt,
    BenchItem,
    Diagnostic,
    EndReason,
    Final,
    ModelInfo,
    Provenance,
    RunInfo,
    Trial,
    make_trial_id,
    to_dict,
)
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.stages import run_error_text
from lassi.core.store import TextStore, read_trial, trial_dir, write_trial
from lassi.core.trial_md import render_trial_md
from lassi.scoring.df_v0 import WEIGHTS_FILE, load_weights

SUITE = "simfix"
ITEM = "vadd"
MODEL_ID = "scripted-fixture"
SOURCE, TARGET = "cpu", "tt"
DIRECTION = f"{SOURCE}-{TARGET}"
TARGET_FILE = "host.cpp"
TOOLCHAINS = {SOURCE: "fake-cpu", TARGET: "fake-tt"}
STAGES = ["baseline", "generate", "compile_loop", "run_loop"]
SIM_EXECUTOR, PLAIN_EXECUTOR = "fake-sim", "scripted"
SIMULATOR = "simulator"
SIM_GAP = "sim-gap"
BASELINE_RUN = "baseline-run"
CORRECTION_CAP = "correction-cap"
GAP_CLASSES = ("UnimplementedFunctionality", "UnsupportedFunctionality")
FINDINGS = ("sim_ub", "sim_gap", "diagnostics")
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
# A commit id for the synthetic manifest and provenance; not a commit of this repository or of any source.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"

# What a message about a finding must say, in any letter case and spacing.
UB_WORDS = re.compile(r"undefined[ _-]?behaviou?r", re.IGNORECASE)
HANG_WORDS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (r"circular[ _-]?buffer", "semaphore", "deadlock"))

# SYNTHETIC sources and replies; the fakes never compile or run them.
SOURCES = {
    SOURCE: "// SYNTHETIC cpu reference of a made-up vector add\nint main() { return 0; }\n",
    TARGET: "// SYNTHETIC tt reference host program of a made-up vector add\nint main() { return 0; }\n",
}
REPLY = render_file_blocks({TARGET_FILE: "// SYNTHETIC candidate host program\nint main() { return 0; }\n"})
FIXED_REPLY = render_file_blocks({TARGET_FILE: "// SYNTHETIC corrected host program\nint main() { return 0; }\n"})

# SYNTHETIC diagnostics an executor could parse from a kernel JIT build.
JIT_ERROR = Diagnostic(
    stage="jit", severity="error", code="SYNTHETIC-jit-error", file="kernels/compute.cpp", line=12, column=5,
    message="SYNTHETIC kernel JIT error: 'tile_count' was not declared in this scope",
)
JIT_WARNING = Diagnostic(
    stage="jit", severity="warning", code="SYNTHETIC-jit-warning", file="kernels/compute.cpp", line=3, column=1,
    message="SYNTHETIC kernel JIT warning: unused variable 'scratch'",
)
# SYNTHETIC program output. The stderr names no finding, so a message about one must come from the stages.
REFERENCE_STDOUT = "SYNTHETIC reference stdout\n"
ATTEMPT_STDOUT = "SYNTHETIC attempt stdout\n"
NEUTRAL_STDERR = "SYNTHETIC simulator stderr\n"
PLACEHOLDER_REFERENCE_WALL_S = 1.25
PLACEHOLDER_ATTEMPT_WALL_S = 0.5
GAP_EXIT = 1


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
# Names this task adds, looked up so a missing one fails its test with a clear message


def require_findings(*names: str) -> None:
    """Fail the test clearly while RunResult lacks any of the named finding fields."""
    have = {spec.name for spec in dataclasses.fields(RunResult)}
    missing = [name for name in names if name not in have]
    if missing:
        pytest.fail(f"RunResult has no {', '.join(missing)}; task P4.6 makes it carry the simulator's findings")


def require_sim_gap() -> None:
    """Fail the test clearly while END_REASONS lacks sim-gap."""
    if SIM_GAP not in record_module.END_REASONS:
        pytest.fail(f"lassi.core.record.END_REASONS has no {SIM_GAP!r}; task P4.6 adds it")


def sim_run(
    *, exit_code: int | None = 0, hang: bool = False, stdout: str = ATTEMPT_STDOUT, stderr: str = NEUTRAL_STDERR,
    wall_s: float = PLACEHOLDER_ATTEMPT_WALL_S, **findings: Any,
) -> RunResult:
    """Return a SYNTHETIC RunResult of a simulator run with `findings` (sim_ub, sim_gap, diagnostics) set."""
    require_findings(*findings)
    return RunResult(exit_code=exit_code, hang=hang, stdout=stdout, stderr=stderr, wall_s=wall_s, **findings)


def clean_sim_run(**findings: Any) -> RunResult:
    """Return a SYNTHETIC clean simulator run of an attempt: exit status 0, no UB, no gap."""
    return sim_run(sim_ub=False, **findings)


def clean_sim_reference() -> RunResult:
    """Return a SYNTHETIC clean simulator run of a reference program."""
    return sim_run(stdout=REFERENCE_STDOUT, wall_s=PLACEHOLDER_REFERENCE_WALL_S, sim_ub=False)


def plain_run(stdout: str = ATTEMPT_STDOUT, wall_s: float = PLACEHOLDER_ATTEMPT_WALL_S) -> RunResult:
    """Return a SYNTHETIC clean run built without any finding field, as executors before this task build one."""
    return RunResult(exit_code=0, hang=False, stdout=stdout, stderr="", wall_s=wall_s)


def plain_reference() -> RunResult:
    """Return a SYNTHETIC clean reference run built without any finding field."""
    return plain_run(REFERENCE_STDOUT, PLACEHOLDER_REFERENCE_WALL_S)


# ---------------------------------------------------------------------------
# Fake components


@dataclass
class Script:
    """What the fakes answer and what they saw.

    `references` maps a language to the RunResult of its reference run, and
    `attempt_runs` holds the results of the attempt runs in order; a run past
    the script fails the test. `calls` counts the replies the backend gave.
    """

    replies: list[str] = field(default_factory=list)
    references: dict[str, RunResult] = field(default_factory=dict)
    attempt_runs: list[RunResult] = field(default_factory=list)
    calls: int = 0


def fake_toolchain(registered_as: str) -> type:
    """Return a Toolchain class without PIN that writes the files and a PLACEHOLDER artifact; it compiles nothing."""

    class FakeToolchain:
        """Writes the files under the fresh workdir and reports a built artifact."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Write `files` and a placeholder artifact, and return it with no diagnostics."""
            for path, text in [*files.items(), *(harness or {}).items()]:
                target = Path(workdir) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            artifact = Path(workdir) / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


def executor_class(registered_as: str, declared: frozenset[str], script: Script) -> type:
    """Return an Executor class that runs nothing: references by language, attempts from the script, in order."""

    class FakeExecutor:
        """Returns the scripted SYNTHETIC RunResult of each run; starts no process and opens no device."""

        name = registered_as
        capabilities = declared

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Return the reference result of the artifact's language, or the next scripted attempt result."""
            folder = Path(artifact).parent.parent.name
            if folder.startswith("baseline-"):
                return script.references[folder[len("baseline-"):]]
            assert script.attempt_runs, f"the executor was asked to run {artifact} past the script"
            return script.attempt_runs.pop(0)

        def device(self) -> str:
            """Return a SYNTHETIC device name."""
            return f"SYNTHETIC {registered_as} device"

    return FakeExecutor


def backend_class(script: Script) -> type:
    """Return an LLMBackend class, registered as "scripted", that answers from script.replies in order."""

    class ScriptedBackend:
        """Counts each request and answers with the next scripted reply."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id, as every backend does."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Return the next scripted reply."""
            assert script.replies, "the backend was asked for more replies than the script holds"
            script.calls += 1
            return Completion(text=script.replies.pop(0), prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def make_registry(script: Script) -> Registry:
    """Return a test Registry: the fakes, the real stages, and the real df-v0 profile."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", backend_class(script))
    sim = frozenset({"runs_code", "sandboxed", SIMULATOR})
    registry.register("Executor", SIM_EXECUTOR, executor_class(SIM_EXECUTOR, sim, script))
    plain = frozenset({"runs_code", "sandboxed"})
    registry.register("Executor", PLAIN_EXECUTOR, executor_class(PLAIN_EXECUTOR, plain, script))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, fake_toolchain(name))
    for name in STAGES:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    registry.register("ScoreProfile", "df-v0", DEFAULT_REGISTRY.get("ScoreProfile", "df-v0").factory)
    return registry


# ---------------------------------------------------------------------------
# The SYNTHETIC suite, recipes, and runs


def write_manifest(directory: Path) -> Path:
    """Write the SYNTHETIC simfix manifest (one item, cpu and tt) under `directory` and return the directory."""
    languages = {
        SOURCE: {"dir": "src/vadd-cpu", "files": ["main.cpp"]},
        TARGET: {"dir": "src/vadd-tt", "files": [TARGET_FILE]},
    }
    data = {
        "suite": SUITE, "repo": "https://example.invalid/simfix", "commit": FAKE_COMMIT,
        "items": {ITEM: {"split": "eval", "languages": languages}},
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{SUITE}.yaml").write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    for language, layout in languages.items():
        path = directory.parent / "bench" / layout["dir"] / layout["files"][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(SOURCES[language].encode("ascii"))
    return directory


def recipe_data(executor: str = SIM_EXECUTOR, **changes: Any) -> dict[str, Any]:
    """Return a one-trial template-set recipe for the item, cpu to tt, with only the target reference built."""
    data: dict[str, Any] = {
        "extends": "base",
        "faithful": False,
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": SOURCE, "target": TARGET}],
        "prompts": "p0-smoke",
        "toolchain": dict(TOOLCHAINS),
        "stages": list(STAGES),
        "executor": {"kind": executor},
        "fixes": {"baseline_both": False},
        "trials": {"n": 1},
    }
    data.update(changes)
    return data


@dataclass
class Outcome:
    """One finished run: its directory, its one trial read back from trial.json, and the script."""

    run_dir: Path
    trial: Trial
    script: Script

    def prompt(self, index: int) -> str:
        """Return the prompt text of attempt `index` from the run's text store."""
        ref = self.trial.attempts[index].prompt_ref
        assert ref is not None, f"attempt {index} kept no prompt"
        return TextStore(self.run_dir).get(ref)


def run_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: Script, data: Mapping[str, Any], direction: str = DIRECTION
) -> Outcome:
    """Run the one-trial recipe `data` on the SYNTHETIC suite with the fakes following `script`; read `direction`'s."""
    manifests = write_manifest(tmp_path / "manifests")
    monkeypatch.setattr(runner_module, "BENCH_DIR", manifests)
    options = RunOptions(
        runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=tmp_path / "bench",
        registry=make_registry(script),
    )
    path = tmp_path / "sim-run.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    run_dir = run_recipe(path, options)
    trial_id = make_trial_id("sim-run", MODEL_ID, SUITE, direction, ITEM, 1)
    return Outcome(run_dir, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir)), script)


def errors_of(attempt: Attempt, stage: str) -> list[Diagnostic]:
    """Return the attempt's error Diagnostics of `stage`."""
    return [item for item in attempt.diagnostics if item.stage == stage and item.severity == "error"]


def stages_of(trial: Trial) -> list[str]:
    """Return the stage each attempt reached, in order."""
    return [attempt.stage_reached for attempt in trial.attempts]


# ---------------------------------------------------------------------------
# RunResult carries the simulator's findings


def test_run_result_carries_the_simulators_findings() -> None:
    require_findings(*FINDINGS)
    run = RunResult(
        exit_code=1, hang=False, stdout="", stderr="", sim_ub=True, sim_gap=GAP_CLASSES[1], diagnostics=[JIT_ERROR]
    )
    assert run.sim_ub is True
    assert run.sim_gap == "UnsupportedFunctionality"
    assert list(run.diagnostics) == [JIT_ERROR]


def test_a_run_result_built_without_findings_reads_as_no_finding() -> None:
    require_findings(*FINDINGS)
    run = RunResult(exit_code=0, hang=False, stdout="", stderr="")
    assert run.sim_ub in (None, False), "an executor that checks no UB reports none"
    assert run.sim_gap is None
    assert list(run.diagnostics) == []


# ---------------------------------------------------------------------------
# END_REASONS and the record


def test_end_reasons_gain_sim_gap_spelled_with_a_hyphen() -> None:
    require_sim_gap()
    reason = EndReason(code=SIM_GAP, message="SYNTHETIC: the simulator reported UnimplementedFunctionality")
    assert reason.code == "sim-gap"


def test_the_underscore_spelling_is_not_a_record_code() -> None:
    assert "sim_gap" not in record_module.END_REASONS
    with pytest.raises(ValueError, match="code"):
        EndReason(code="sim_gap", message="SYNTHETIC: the RunResult field's spelling")


def hand_trial(store: TextStore, end: EndReason | None = None) -> Trial:
    """Return a SYNTHETIC trial of the item: a clean reference run, a failed attempt, and a clean one."""
    stdout = store.put(ATTEMPT_STDOUT)
    flags = {"stdout_truncated": False, "stderr_truncated": False, "workdir_incomplete": False}
    failed = RunInfo(exit_code=3, hang=False, wall_s=PLACEHOLDER_ATTEMPT_WALL_S, stdout_ref=stdout, **flags)
    clean = RunInfo(exit_code=0, hang=False, wall_s=PLACEHOLDER_ATTEMPT_WALL_S, stdout_ref=stdout, **flags)
    run_error = Diagnostic(stage="run", severity="error", code="run-error", message="SYNTHETIC run error")
    attempts = [
        Attempt(index=0, prompt_ref=store.put("SYNTHETIC prompt 0\n"), response_text=REPLY, stage_reached="S4",
                diagnostics=[run_error], run=failed),
        Attempt(index=1, prompt_ref=store.put("SYNTHETIC prompt 1\n"), response_text=FIXED_REPLY, stage_reached="S5",
                run=clean),
    ]
    reference = RunInfo(
        exit_code=0, hang=False, wall_s=PLACEHOLDER_REFERENCE_WALL_S, stdout_ref=store.put(REFERENCE_STDOUT), **flags
    )
    return Trial(
        trial_id=make_trial_id("sim-run", MODEL_ID, SUITE, DIRECTION, ITEM, 1),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=FAKE_COMMIT, dirty=False, device="SYNTHETIC device", sdk=None,
                              date="2026-09-25T00:00:00+00:00"),
        bench_item=BenchItem(suite=SUITE, item=ITEM, split="eval", direction=DIRECTION),
        model=ModelInfo(backend="scripted", id=MODEL_ID, sampling=SAMPLING),
        reference_run=reference,
        requests=[],
        attempts=attempts,
        final=Final(stage_reached=attempts[-1].stage_reached, corrections=1, end_reason=end),
    )


def test_a_trial_ended_at_sim_gap_round_trips_through_trial_json(tmp_path: Path) -> None:
    require_sim_gap()
    store = TextStore(tmp_path / "run")
    trial = hand_trial(store, EndReason(code=SIM_GAP, message="SYNTHETIC: UnsupportedFunctionality"))
    out = write_trial(trial, tmp_path / "run", store)
    data = json.loads((out / "trial.json").read_text(encoding="ascii"))
    assert data["final"]["end_reason"]["code"] == "sim-gap"
    assert read_trial(out, store) == trial


# The keys each record class held before this task (the working tree P4.6 starts from), by class.
OLD_KEYS: dict[str, tuple[str, ...]] = {
    "Trial": ("trial_id", "recipe_hash", "toolchain_pins", "provenance", "bench_item", "model", "reference_run",
              "reference_agreement", "baseline_diagnostics", "context", "requests", "attempts", "final"),
    "RunInfo": ("exit_code", "hang", "sim_ub", "wall_s", "stdout_ref", "outputs_ref", "stdout_truncated",
                "stderr_truncated", "workdir_incomplete", "outputs"),
    "Attempt": ("index", "prompt_ref", "response_text", "files", "diff_from_previous", "stage_reached",
                "diagnostics", "run", "alignment", "profile", "guards", "score"),
    "Final": ("stage_reached", "alignment", "score", "corrections", "wall_s", "end_reason"),
    "EndReason": ("code", "message"),
    "Diagnostic": ("stage", "severity", "code", "file", "line", "column", "message"),
}


def _keep_old(data: dict[str, Any], cls: str, where: str, removed: list[tuple[str, Any]]) -> None:
    """Drop from `data` every key class `cls` did not hold before this task, noting each in `removed`."""
    for key in [key for key in data if key not in OLD_KEYS[cls]]:
        removed.append((f"{where}.{key}", data.pop(key)))


def older_form(trial_data: Mapping[str, Any]) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
    """Return trial data as a trial.json written before this task holds it, and the (path, value) of each drop."""
    data = copy.deepcopy(dict(trial_data))
    removed: list[tuple[str, Any]] = []
    _keep_old(data, "Trial", "Trial", removed)
    _keep_old(data["reference_run"], "RunInfo", "Trial.reference_run", removed)
    _keep_old(data["final"], "Final", "Trial.final", removed)
    if data["final"]["end_reason"] is not None:
        _keep_old(data["final"]["end_reason"], "EndReason", "Trial.final.end_reason", removed)
    for position, note in enumerate(data["baseline_diagnostics"]):
        _keep_old(note, "Diagnostic", f"Trial.baseline_diagnostics[{position}]", removed)
    for position, attempt in enumerate(data["attempts"]):
        where = f"Trial.attempts[{position}]"
        _keep_old(attempt, "Attempt", where, removed)
        _keep_old(attempt["run"], "RunInfo", f"{where}.run", removed)
        for number, diagnostic in enumerate(attempt["diagnostics"]):
            _keep_old(diagnostic, "Diagnostic", f"{where}.diagnostics[{number}]", removed)
    return data, removed


def test_a_trial_json_written_before_this_task_loads_unchanged(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    store = TextStore(run_root)
    trial = hand_trial(store, EndReason(code=CORRECTION_CAP, message="SYNTHETIC: an error remained at the cap"))
    out = write_trial(trial, run_root, store)
    written = json.loads((out / "trial.json").read_text(encoding="ascii"))
    older, _ = older_form(written)
    (out / "trial.json").write_bytes((json.dumps(older, indent=2) + "\n").encode("ascii"))
    loaded = read_trial(out, store)
    kept, added = older_form(to_dict(loaded))
    assert kept == older_form(to_dict(trial))[0], "every field an older trial.json holds loads as written"
    assert all(value is None or value == [] for _, value in added), f"a key the file lacks is not recorded: {added}"
    page = render_trial_md(loaded, store)
    assert "| wall_s | 1.25 |" in page and page.count("| wall_s | 0.5 |") == 2, "wall times print as before"


# ---------------------------------------------------------------------------
# The baseline reads the findings of a reference run


def test_a_target_reference_run_with_undefined_behavior_ends_at_baseline_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ub = sim_run(stdout=REFERENCE_STDOUT, wall_s=PLACEHOLDER_REFERENCE_WALL_S, sim_ub=True)
    script = Script(replies=[REPLY], references={TARGET: ub}, attempt_runs=[clean_sim_run()])
    outcome = run_one(tmp_path, monkeypatch, script, recipe_data())
    trial = outcome.trial
    reason = trial.final.end_reason
    assert reason is not None and reason.code == BASELINE_RUN, (
        f"UB in a reference run ends the trial at the baseline, even after exit status 0; got {reason}"
    )
    assert UB_WORDS.search(reason.message), f"the message names the undefined behavior: {reason.message!r}"
    assert trial.reference_run.sim_ub is True, "Trial.reference_run records the target reference's UB"
    assert (script.calls, trial.attempts, trial.requests) == (0, [], []), "no model is asked"
    assert len(script.attempt_runs) == 1, "no attempt ran"


def test_a_source_reference_run_with_undefined_behavior_ends_at_baseline_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references = {TARGET: clean_sim_reference(), SOURCE: sim_run(stdout=REFERENCE_STDOUT, sim_ub=True)}
    script = Script(replies=[REPLY], references=references, attempt_runs=[clean_sim_run()])
    outcome = run_one(tmp_path, monkeypatch, script, recipe_data(fixes={"baseline_both": True}))
    reason = outcome.trial.final.end_reason
    assert reason is not None and reason.code == BASELINE_RUN, f"UB in the source reference run ends it: {reason}"
    assert UB_WORDS.search(reason.message), f"the message names the undefined behavior: {reason.message!r}"
    assert outcome.trial.reference_run.sim_ub is False, "the target's clean simulator run records no UB"
    assert script.calls == 0 and len(script.attempt_runs) == 1, "no model is asked and no attempt runs"


@pytest.mark.parametrize("which", [TARGET, SOURCE], ids=["target-reference", "source-reference"])
@pytest.mark.parametrize("gap", GAP_CLASSES)
def test_a_reference_run_with_a_gap_ends_at_sim_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gap: str, which: str
) -> None:
    references = {TARGET: clean_sim_reference(), SOURCE: clean_sim_reference()}
    references[which] = sim_run(exit_code=GAP_EXIT, stdout=REFERENCE_STDOUT, sim_ub=False, sim_gap=gap)
    script = Script(replies=[REPLY], references=references)
    outcome = run_one(tmp_path, monkeypatch, script, recipe_data(fixes={"baseline_both": True}))
    reason = outcome.trial.final.end_reason
    assert reason is not None and reason.code == SIM_GAP, f"a gap is a simulator gap, not a failed run: {reason}"
    assert gap in reason.message, f"the message names the gap's class: {reason.message!r}"
    assert (script.calls, outcome.trial.attempts, outcome.trial.requests) == (0, [], []), "no model is asked"
    assert outcome.trial.final.stage_reached is None, "no attempt was made"


# ---------------------------------------------------------------------------
# run_loop reads the findings of an attempt run


def test_a_clean_simulator_run_records_sim_ub_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Script(replies=[REPLY], references={TARGET: clean_sim_reference()}, attempt_runs=[clean_sim_run()])
    trial = run_one(tmp_path, monkeypatch, script, recipe_data()).trial
    assert stages_of(trial) == ["S5"] and trial.final.end_reason is None, "a clean simulator run is S5"
    recorded = (trial.reference_run.sim_ub, trial.attempts[0].run.sim_ub)
    assert recorded == (False, False), f"a simulator run with no UB records sim_ub False, got {recorded}"


def test_an_executor_that_reports_no_finding_keeps_sim_ub_not_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script(replies=[REPLY], references={TARGET: plain_reference()}, attempt_runs=[plain_run()])
    trial = run_one(tmp_path, monkeypatch, script, recipe_data(executor=PLAIN_EXECUTOR)).trial
    assert stages_of(trial) == ["S5"]
    assert trial.reference_run.sim_ub is None and trial.attempts[0].run.sim_ub is None, "sim_ub stays not recorded"


def test_a_kernel_jit_failure_leaves_the_attempt_at_s1_and_is_corrected_as_a_compile_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jit_failed = sim_run(exit_code=1, stdout="", sim_ub=False, diagnostics=[JIT_ERROR])
    script = Script(
        replies=[REPLY, FIXED_REPLY], references={TARGET: clean_sim_reference()},
        attempt_runs=[jit_failed, clean_sim_run()],
    )
    outcome = run_one(tmp_path, monkeypatch, script, recipe_data())
    trial = outcome.trial
    assert stages_of(trial) == ["S1", "S5"], "S4 needs host and kernel JIT, so a JIT failure stays at S1"
    first = trial.attempts[0]
    assert [item.message for item in errors_of(first, "jit")] == [JIT_ERROR.message]
    assert not [item for item in errors_of(first, "run") if item.code == "run-error"], "a JIT failure is no run error"
    assert trial.requests is not None and [request.stage for request in trial.requests] == ["generate", "compile_loop"]
    assert JIT_ERROR.message in outcome.prompt(1), "the correction prompt carries the jit error as a compile error"
    assert (trial.final.corrections, trial.final.end_reason) == (1, None)


def test_a_kernel_jit_failure_at_the_cap_ends_at_correction_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jit_failed = sim_run(exit_code=1, stdout="", sim_ub=False, diagnostics=[JIT_ERROR])
    script = Script(replies=[REPLY, FIXED_REPLY], references={TARGET: clean_sim_reference()}, attempt_runs=[jit_failed])
    trial = run_one(tmp_path, monkeypatch, script, recipe_data(loop={"max_corrections": 0})).trial
    assert stages_of(trial) == ["S1"], f"the JIT failure leaves attempt 0 at S1, got {stages_of(trial)}"
    reason = trial.final.end_reason
    assert reason is not None and reason.code == CORRECTION_CAP, f"a compile error at the cap ends there: {reason}"
    assert script.calls == 1, "no correction past the cap"


def test_jit_warnings_of_a_clean_run_stay_on_the_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Script(
        replies=[REPLY], references={TARGET: clean_sim_reference()},
        attempt_runs=[clean_sim_run(diagnostics=[JIT_WARNING])],
    )
    trial = run_one(tmp_path, monkeypatch, script, recipe_data()).trial
    assert stages_of(trial) == ["S5"], "a JIT warning does not fail the run"
    warnings = [item for item in trial.attempts[0].diagnostics if item.stage == "jit" and item.severity == "warning"]
    found = [(item.code, item.message) for item in warnings]
    assert found == [(JIT_WARNING.code, JIT_WARNING.message)], f"the executor's jit warning stays: {found}"


def test_undefined_behavior_is_a_failed_run_fed_back_to_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script(
        replies=[REPLY, FIXED_REPLY], references={TARGET: clean_sim_reference()},
        attempt_runs=[sim_run(exit_code=0, sim_ub=True), clean_sim_run()],
    )
    outcome = run_one(tmp_path, monkeypatch, script, recipe_data())
    trial = outcome.trial
    assert stages_of(trial) == ["S4", "S5"], "a run with UB is not clean, even after exit status 0"
    first = trial.attempts[0]
    assert first.run.sim_ub is True, "Attempt.run records the UB"
    messages = [item.message for item in errors_of(first, "run")]
    assert any(UB_WORDS.search(message) for message in messages), f"no run-stage error names the UB: {messages}"
    assert trial.requests is not None and trial.requests[1].stage == "run_loop", "a failed run's correction"
    assert UB_WORDS.search(outcome.prompt(1)), "the correction prompt tells the model about the undefined behavior"
    assert trial.final.end_reason is None


@pytest.mark.parametrize("gap", GAP_CLASSES)
def test_a_simulator_gap_in_an_attempt_run_ends_the_trial_at_sim_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gap: str
) -> None:
    script = Script(
        replies=[REPLY, FIXED_REPLY], references={TARGET: clean_sim_reference()},
        attempt_runs=[sim_run(exit_code=GAP_EXIT, sim_ub=False, sim_gap=gap), clean_sim_run()],
    )
    trial = run_one(tmp_path, monkeypatch, script, recipe_data()).trial
    reason = trial.final.end_reason
    assert reason is not None and reason.code == SIM_GAP, f"a gap is not a model error, so the trial stops: {reason}"
    assert gap in reason.message, f"the message names the gap's class: {reason.message!r}"
    assert stages_of(trial) == ["S4"] and trial.final.stage_reached == "S4", "the gap attempt compiled, not ran clean"
    assert script.calls == 1 and script.replies == [FIXED_REPLY], "no correction is asked after a gap"
    assert len(script.attempt_runs) == 1, "no later attempt runs"


def test_a_simulator_hang_carries_the_harness_contracts_hang_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hung = RunResult(exit_code=None, hang=True, stdout="", stderr="", wall_s=PLACEHOLDER_ATTEMPT_WALL_S)
    script = Script(
        replies=[REPLY, FIXED_REPLY], references={TARGET: plain_reference()}, attempt_runs=[hung, plain_run()]
    )
    outcome = run_one(tmp_path, monkeypatch, script, recipe_data())
    trial = outcome.trial
    assert stages_of(trial) == ["S4", "S5"], "a hang is a failed run, corrected like one"
    messages = [item.message for item in errors_of(trial.attempts[0], "run")]
    named = [message for message in messages if all(words.search(message) for words in HANG_WORDS)]
    assert named, f"no run-stage error names a likely circular-buffer or semaphore deadlock: {messages}"
    assert HANG_WORDS[2].search(outcome.prompt(1)), "the correction prompt carries the hang diagnostic"


# ---------------------------------------------------------------------------
# Scoring a run that ended at sim-gap


def test_with_df_v0_a_trial_ended_at_sim_gap_has_no_score_and_its_gap_attempt_no_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gap = sim_run(exit_code=GAP_EXIT, sim_ub=False, sim_gap=GAP_CLASSES[0])
    script = Script(
        replies=[REPLY, FIXED_REPLY], references={TARGET: clean_sim_reference()},
        attempt_runs=[sim_run(exit_code=0, sim_ub=True), gap],
    )
    trial = run_one(tmp_path, monkeypatch, script, recipe_data(score="df-v0")).trial
    reason = trial.final.end_reason
    assert reason is not None and reason.code == SIM_GAP, f"the gap in attempt 1 ends the trial: {reason}"
    assert stages_of(trial) == ["S4", "S4"], f"UB, then a gap: {stages_of(trial)}"
    s4_base = load_weights(WEIGHTS_FILE).stage_base["S4"]
    assert trial.attempts[0].score.scalar == pytest.approx(s4_base), "the UB attempt scores as the failed run it was"
    assert trial.attempts[1].score.scalar is None, "a simulator gap is not a model error, so it earns no reward"
    assert trial.final.score is None, "R_final is the gap attempt's, so the trial has no score"


# ---------------------------------------------------------------------------
# The order of the readings, and the runs that get no hang diagnostic (the P4.6 commit audit)


def test_a_jit_error_is_read_before_a_gap_in_the_same_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    both = sim_run(exit_code=GAP_EXIT, stdout="", sim_ub=False, sim_gap=GAP_CLASSES[0], diagnostics=[JIT_ERROR])
    script = Script(
        replies=[REPLY, FIXED_REPLY], references={TARGET: clean_sim_reference()}, attempt_runs=[both, clean_sim_run()]
    )
    trial = run_one(tmp_path, monkeypatch, script, recipe_data()).trial
    assert stages_of(trial) == ["S1", "S5"], "a kernel JIT failure is corrected first, whatever else the run reported"
    assert trial.requests is not None and [request.stage for request in trial.requests] == ["generate", "compile_loop"]
    assert trial.final.end_reason is None


def test_a_gap_is_read_before_undefined_behavior_in_the_same_attempt_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gap_ub = sim_run(exit_code=GAP_EXIT, sim_ub=True, sim_gap=GAP_CLASSES[1])
    script = Script(
        replies=[REPLY, FIXED_REPLY], references={TARGET: clean_sim_reference()}, attempt_runs=[gap_ub, clean_sim_run()]
    )
    trial = run_one(tmp_path, monkeypatch, script, recipe_data(score="df-v0")).trial
    reason = trial.final.end_reason
    assert reason is not None and reason.code == SIM_GAP, f"the gap ends the trial, UB or not: {reason}"
    first = trial.attempts[0]
    assert stages_of(trial) == ["S4"] and first.run.sim_ub is True, "the record keeps the UB the run reported"
    assert not errors_of(first, "run") and script.calls == 1, "the gap attempt gets no run-error and no correction"
    assert first.score.scalar is None and trial.final.score is None, "the gap reading wins: R is not computed"


def test_a_gap_is_read_before_undefined_behavior_in_a_reference_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = sim_run(exit_code=GAP_EXIT, stdout=REFERENCE_STDOUT, sim_ub=True, sim_gap=GAP_CLASSES[0])
    script = Script(replies=[REPLY], references={TARGET: reference})
    reason = run_one(tmp_path, monkeypatch, script, recipe_data()).trial.final.end_reason
    assert reason is not None and reason.code == SIM_GAP and GAP_CLASSES[0] in reason.message, reason


def hang_run(wall_s: float = PLACEHOLDER_ATTEMPT_WALL_S) -> RunResult:
    """Return a SYNTHETIC run that hung, built without any finding field."""
    return RunResult(exit_code=None, hang=True, stdout="", stderr="", wall_s=wall_s)


def assert_no_hang_diagnostic(outcome: Outcome) -> None:
    """Assert that attempt 0's run-error and the correction prompt are as before task P4.6: no deadlock hint."""
    messages = [item.message for item in errors_of(outcome.trial.attempts[0], "run")]
    assert messages == ["the program was stopped at its wall limit of 30 s"], f"no hang diagnostic: {messages}"
    prompt = outcome.prompt(1)
    assert not any(words.search(prompt) for words in HANG_WORDS), "the correction prompt names no deadlock"


def test_a_hang_on_an_executor_without_the_simulator_capability_gets_no_hang_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script(
        replies=[REPLY, FIXED_REPLY], references={TARGET: plain_reference()}, attempt_runs=[hang_run(), plain_run()]
    )
    assert_no_hang_diagnostic(run_one(tmp_path, monkeypatch, script, recipe_data(executor=PLAIN_EXECUTOR)))


def test_with_executors_per_language_the_target_executor_decides_the_hang_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpu_reply = render_file_blocks({"main.cpp": "// SYNTHETIC candidate cpu program\nint main() { return 0; }\n"})
    script = Script(
        replies=[cpu_reply, cpu_reply], references={SOURCE: plain_reference()}, attempt_runs=[hang_run(), plain_run()]
    )
    data = recipe_data(directions=[{"source": TARGET, "target": SOURCE}])
    data["executor"] = {SOURCE: PLAIN_EXECUTOR, TARGET: SIM_EXECUTOR}
    assert_no_hang_diagnostic(run_one(tmp_path, monkeypatch, script, data, direction=f"{TARGET}-{SOURCE}"))


def test_a_reference_run_that_hangs_on_a_simulator_keeps_its_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script(replies=[REPLY], references={TARGET: hang_run(PLACEHOLDER_REFERENCE_WALL_S)})
    reason = run_one(tmp_path, monkeypatch, script, recipe_data()).trial.final.end_reason
    assert reason == EndReason(code=BASELINE_RUN, message="the tt reference run hung past its wall limit of 600.0 s")


# SYNTHETIC stand-ins for upstream's execute.* fragments; not upstream's text.
FRAGMENTS = {
    "execute.exit_lead": "SYNTHETIC exit lead ",
    "execute.segfault": "SYNTHETIC segfault note ",
    "execute.stderr_lead": "SYNTHETIC stderr lead ",
}
LIMITS = Limits(wall_s=30.0, memory_mb=1024, cpus=1)


def test_under_a_fragment_set_the_findings_follow_upstreams_report_after_one_line_feed() -> None:
    ub = RunResult(exit_code=0, hang=False, stdout="", stderr="SYNTHETIC stderr\n", sim_ub=True)
    report = run_error_text(dataclasses.replace(ub, sim_ub=None), LIMITS, FRAGMENTS, simulator=True)
    assert report == "SYNTHETIC exit lead 0 SYNTHETIC stderr lead SYNTHETIC stderr\n", "no finding: the report alone"
    text = run_error_text(ub, LIMITS, FRAGMENTS, simulator=True)
    assert text.startswith(report + "\n") and UB_WORDS.search(text[len(report) + 1:]), text
    hung = hang_run()
    assert run_error_text(hung, LIMITS, FRAGMENTS) == "SYNTHETIC exit lead None ", "no hint off a simulator"
    hint = run_error_text(hung, LIMITS, FRAGMENTS, simulator=True)
    assert hint.startswith("SYNTHETIC exit lead None \n")
    assert all(words.search(hint) for words in HANG_WORDS), hint
