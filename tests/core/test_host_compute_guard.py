"""Tests for the host-compute guard's seam and its wiring into compile_loop (task P4.12).

Bible: Harness Contract (the CPU -> TT guard: flagged host compute,
data-movement-only programs tagged, not failed), Component Interfaces
(Toolchain row; capability rule; a correction prompt carries the parsed
diagnostics), Result Record (Attempt.guards: True, False, or None for not
checked; Diagnostic stages and severities), Reward Function (a guard
violation sets R = -1; OQ-028 3(b): a violation on any attempt sets
multi_turn), Design Principle 8, Agent Rules 3 and 4. Plan:
plans/p4-ttsim.md, P4.12. The seam is lassi.core.capabilities
(HOST_COMPUTE_GUARD, HostComputeReading, host_compute_reading), the wiring
is lassi.core.stages (compile_loop, prompt_diagnostics), and the real
reader is lassi.toolchains.ttmetal_guard; their docstrings state the
contract below.

The contract these tests fix:

- lassi.core.capabilities gains HOST_COMPUTE_GUARD ("host_compute_guard"),
  GUARD_CODE_PREFIX ("guard-"), GUARD_STAGE ("parse"), GUARD_SEVERITIES
  ({"warning", "note"}), the frozen record HostComputeReading(host_compute,
  diagnostics=()), and host_compute_reading(toolchain, files, harness):
  None (not checked) for a toolchain that does not declare the capability,
  whatever methods it has, and otherwise its host_compute_guard(files,
  harness). A ValueError naming the toolchain (its `name`, else its class
  name) and the capability refuses a declared capability without a callable
  method and any reading outside the contract: not a HostComputeReading,
  host_compute not a bool or None, diagnostics not a tuple, or an item that
  is not a Diagnostic, is not parse-stage, is not a warning or a note, or
  whose code does not start with "guard-".
- lassi.core.stages gains prompt_diagnostics(diagnostics): the list without
  the diagnostics whose code starts with GUARD_CODE_PREFIX.
- compile_loop, after every build that gave an artifact, asks the target
  language's toolchain for a reading by capability (never by name), with
  the attempt's files and the item's support files. The reading goes into
  Attempt.guards.host_compute and its diagnostics after the build's, before
  the attempt event; the attempt keeps it through run_loop, a kernel JIT
  failure, a simulator gap, and the execution gate. A build with no
  artifact, a toolchain without the capability, and the TT -> CPU
  direction get no reading; the baseline never asks for one. A guard
  diagnostic never changes a stage, never asks for a correction, counts in
  no df-v0 term, and never enters a correction prompt, under a template set
  or a fragment set.
- compile_loop's describe() adds " (host_compute_guard after each build)"
  after the builder's name only when the target toolchain declares the
  capability, so every existing text holds.
- Nothing under lassi/scoring changes: df-v0 already reads a True guard as
  R = guard_violation and a violation on any attempt as
  multi_turn = guard_violation - correction_penalty x corrections.

Task P17.14 (OQ-028, the owner's condition of 2026-10-05) adds why
host_compute is null, Attempt.guards.host_compute_not_checked, one code of
lassi.core.record.HOST_COMPUTE_NOT_CHECKED, set by the stages from what they
know:

- `no-guard`: the target language's toolchain does not declare the
  capability (found by capability, never by name). It wins over the other
  three codes, so every attempt of such a target records it, built or not.
- `not-built`: generate and each correction set it when they append the
  attempt (an S0 reply under fixes.fence_tag on, an S1 reply with a
  missing-file error, or a recipe without compile_loop), so the attempt
  event after the reply carries it.
- `no-program`: compile_loop replaces not-built with it after a build that
  gave no program (a build error or no artifact).
- `guard-not-checked`: the guard ran and gave a null reading; the guard's
  own guard-not-checked note on the attempt still names the cause.

A true or false reading records no code. A kernel JIT failure, a simulator
gap, the execution gate's stale-output copy, and a compile-only executor
keep the code as they keep host_compute. No correction prompt holds the
field's name or a code (OQ-041). The trial's trial.json, trial.md Guards
table, and Parquet attempts table show the code, and run.md gains a
"## Guard coverage" section after the Trials section, with one row for the
run and one per target language: attempts, checked, not checked, the
not-checked rate, and the count of each code and of `not recorded`.

The fake toolchains compile nothing and return SYNTHETIC readings (the last
test uses the real reader on the hand-written fixtures); the fake executors
start no process and open no device. The suite manifest, sources, replies,
diagnostics, and program output are SYNTHETIC. Wall times are PLACEHOLDER
fixture values. No value in this module is a measurement.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from lassi.bench import Direction, load_suite
from lassi.core import capabilities as capabilities_module
from lassi.core import runner as runner_module
from lassi.core import stages as stages_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.parquet import read_run_parquet
from lassi.core.progress import ATTEMPT, ProgressEvent
from lassi.core.record import (
    Attempt,
    BenchItem,
    Diagnostic,
    Final,
    Guards,
    ModelInfo,
    Provenance,
    Trial,
    make_trial_id,
    to_dict,
)
from lassi.core.registry import DEFAULT_REGISTRY, Registry, RegistryError
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import CompileLoopStage, RunContext, diagnostics_text
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.scoring.df_v0 import WEIGHTS_FILE, load_weights

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "toolchains" / "fixtures" / "host_compute"
HARNESS_HEADER = REPO / "assets" / "harness" / "c" / "lassi_io.h"
SUITE = "guardfix"
ITEM = "vadd"
MODEL_ID = "scripted-fixture"
PROJECT = "guard-run"
SOURCE, TARGET = "cpu", "tt"
DIRECTION = f"{SOURCE}-{TARGET}"
TARGET_FILE = "main.cpp"
SUPPORT, SUPPORT_PATH = "lassi_io.h", "src/harness/lassi_io.h"
SUPPORT_TEXT = "/* SYNTHETIC stand-in for the harness header */\n"
GUARDED, PLAIN_TT, PLAIN_CPU, NO_METHOD = "fake-tt-guarded", "fake-tt-plain", "fake-cpu", "fake-tt-no-method"
REAL_READER = "fake-tt-real-reader"
SIM_EXECUTOR, COMPILE_ONLY = "fake-sim", "fake-compile-only"
STAGES = ["baseline", "generate", "compile_loop", "run_loop"]
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
# A commit id for the synthetic manifest and provenance; not a commit of this repository or of any source.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
GUARD_DESCRIPTION = " (host_compute_guard after each build)"
CAPABILITY = "host_compute_guard"
TT_NAMES = (
    "ttmetal", "TtMetal", "CreateKernel", "KernelDescriptor", "ProgramDescriptor", "ComputeConfig",
    "EnqueueReadMeshBuffer", "ReadShard", "ReadFromBuffer", "ReadFromDeviceL1",
)

# SYNTHETIC sources and replies; the fakes never compile or run them.
SOURCES = {
    SOURCE: "// SYNTHETIC cpu reference of a made-up vector add\nint main() { return 0; }\n",
    TARGET: "// SYNTHETIC tt reference host program of a made-up vector add\nint main() { return 0; }\n",
}
REPLY = render_file_blocks(
    {
        TARGET_FILE: "// SYNTHETIC candidate host program\nint main() { return 0; }\n",
        "kernels/reader.cpp": "// SYNTHETIC kernel placeholder\nvoid kernel_main() {}\n",
    }
)
FIXED_REPLY = render_file_blocks({TARGET_FILE: "// SYNTHETIC corrected host program\nint main() { return 0; }\n"})
CPU_REPLY = render_file_blocks({"main.cpp": "// SYNTHETIC candidate cpu program\nint main() { return 0; }\n"})

# SYNTHETIC diagnostics: two a guard may give, and the ones a build, a kernel JIT, and the reply parse could give.
TAG = Diagnostic(
    stage="parse", severity="warning", code="guard-data-movement-only", file=TARGET_FILE, line=7,
    message="SYNTHETIC tag: kernels are created but no compute config is named",
)
GUARD_NOTE = Diagnostic(
    stage="parse", severity="note", code="guard-host-compute", file=TARGET_FILE, line=9,
    message="SYNTHETIC guard note: values read from the inputs reach the output through host code",
)
COMPILE_WARNING = Diagnostic(
    stage="compile", severity="warning", code="-Wunused-variable", file=TARGET_FILE, line=3, column=9,
    message="SYNTHETIC compile warning: unused variable 'spare'",
)
COMPILE_ERROR = Diagnostic(
    stage="compile", severity="error", code=None, file=TARGET_FILE, line=4, column=5,
    message="SYNTHETIC compile error: use of undeclared identifier 'undeclared'",
)
JIT_ERROR = Diagnostic(
    stage="jit", severity="error", code="SYNTHETIC-jit-error", file="kernels/reader.cpp", line=2, column=5,
    message="SYNTHETIC kernel JIT error: 'tile_count' was not declared in this scope",
)
PARSE_WARNING = Diagnostic(
    stage="parse", severity="warning", code="invalid-text", message="SYNTHETIC parse warning of the reply"
)
PLACEHOLDER_REFERENCE_WALL_S = 1.25
PLACEHOLDER_ATTEMPT_WALL_S = 0.5


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate and compile variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


# ---------------------------------------------------------------------------
# Names this task adds, looked up so a missing one fails its test with a clear message


def capability(name: str) -> Any:
    """Return lassi.core.capabilities.<name>; fail the test clearly while it is missing."""
    value = getattr(capabilities_module, name, None)
    if value is None:
        pytest.fail(f"lassi.core.capabilities has no {name}; task P4.12 adds it")
    return value


def prompt_diagnostics() -> Callable[[Sequence[Diagnostic]], list[Diagnostic]]:
    """Return lassi.core.stages.prompt_diagnostics; fail the test clearly while it is missing."""
    function = getattr(stages_module, "prompt_diagnostics", None)
    if function is None:
        pytest.fail("lassi.core.stages has no prompt_diagnostics; task P4.12 adds it")
    return function


def reading(host_compute: Any, *diagnostics: Any) -> Any:
    """Return a HostComputeReading with `host_compute` and `diagnostics` as given (no check here)."""
    return capability("HostComputeReading")(host_compute, tuple(diagnostics))


def real_guard() -> Callable[[Mapping[str, str], Mapping[str, str]], Any]:
    """Return TtMetalHost.host_compute_guard, the real reader; fail the test clearly while it is missing."""
    try:
        cls = DEFAULT_REGISTRY.get("Toolchain", "ttmetal-host").factory
    except RegistryError as error:
        pytest.fail(f"no Toolchain is registered as 'ttmetal-host' ({error})")
    method = getattr(cls, CAPABILITY, None)
    if method is None:
        pytest.fail("TtMetalHost has no static host_compute_guard; task P4.12 adds it")
    return method


# ---------------------------------------------------------------------------
# Fake components


@dataclass
class World:
    """What the fakes answer and what they saw.

    `builds` holds the kind of each attempt build in order ("ok", "warn":
    an artifact and a compile warning, "fail": no artifact and a compile
    error, "none": no artifact and no diagnostic), "ok" past the script; a
    reference build is always "ok".
    `readings` holds each guard answer in order; a call past the script
    fails the test. `guard_calls` records each call's (files, harness).
    """

    replies: list[str] = field(default_factory=list)
    references: dict[str, RunResult] = field(default_factory=dict)
    attempt_runs: list[RunResult] = field(default_factory=list)
    builds: list[str] = field(default_factory=list)
    readings: list[Any] = field(default_factory=list)
    guard_calls: list[tuple[dict[str, str], dict[str, str]]] = field(default_factory=list)
    requests: list[list[Message]] = field(default_factory=list)


def toolchain_class(registered_as: str, declared: frozenset[str], world: World, *, method: bool = True) -> type:
    """Return a Toolchain class without PIN that compiles nothing; with `method`, it has a scripted guard."""

    class FakeToolchain:
        """Writes the files and reports the scripted build; reference builds always give an artifact."""

        name = registered_as
        capabilities = declared

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Write `files` and `harness`, then return the next scripted build (see World)."""
            for path, text in [*files.items(), *(harness or {}).items()]:
                target = Path(workdir) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            reference = Path(workdir).parent.name.startswith("baseline-")
            kind = "ok" if reference or not world.builds else world.builds.pop(0)
            if kind == "fail":
                return BuildResult(artifact=None, diagnostics=[COMPILE_ERROR])
            if kind == "none":
                return BuildResult(artifact=None, diagnostics=[])
            artifact = Path(workdir) / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[COMPILE_WARNING] if kind == "warn" else [])

    if method:

        def host_compute_guard(self: Any, files: Mapping[str, str], harness: Mapping[str, str]) -> Any:
            """Record the call and return the next scripted reading."""
            world.guard_calls.append((dict(files), dict(harness)))
            assert world.readings, "the guard was asked for more readings than the script holds"
            return world.readings.pop(0)

        FakeToolchain.host_compute_guard = host_compute_guard  # type: ignore[attr-defined]
    return FakeToolchain


def real_reader_class(world: World) -> type:
    """Return a guarded Toolchain class whose host_compute_guard is TtMetalHost's, the real reader.

    The method is looked up when the guard is called, so a registry that
    holds this class can be built before the real reader exists.
    """
    cls = toolchain_class(REAL_READER, frozenset({"diagnostics", CAPABILITY}), world, method=False)

    def host_compute_guard(self: Any, files: Mapping[str, str], harness: Mapping[str, str]) -> Any:
        """Return TtMetalHost.host_compute_guard's reading of `files` with `harness`."""
        world.guard_calls.append((dict(files), dict(harness)))
        return real_guard()(files, harness)

    cls.host_compute_guard = host_compute_guard  # type: ignore[attr-defined]
    return cls


def executor_class(registered_as: str, declared: frozenset[str], world: World) -> type:
    """Return an Executor class that runs nothing: references by language, attempts from the script, in order."""

    class FakeExecutor:
        """Returns the scripted SYNTHETIC RunResult of each run; starts no process and opens no device."""

        name = registered_as
        capabilities = declared

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Return the reference result of the artifact's language, or the next scripted attempt result."""
            folder = Path(artifact).parent.parent.name
            if folder.startswith("baseline-"):
                return world.references[folder[len("baseline-"):]]
            assert world.attempt_runs, f"the executor was asked to run {artifact} past the script"
            return world.attempt_runs.pop(0)

        def device(self) -> str:
            """Return a SYNTHETIC device name."""
            return f"SYNTHETIC {registered_as} device"

    return FakeExecutor


def backend_class(world: World) -> type:
    """Return an LLMBackend class, registered as "scripted", that answers from world.replies in order."""

    class ScriptedBackend:
        """Records each request's messages and answers with the next scripted reply."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id, as every backend does."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Return the next scripted reply."""
            assert world.replies, "the backend was asked for more replies than the script holds"
            world.requests.append(list(messages))
            return Completion(text=world.replies.pop(0), prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def make_registry(world: World) -> Registry:
    """Return a test Registry: the fakes, the real stages, and the real df-v0 profile."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", backend_class(world))
    sim = frozenset({"runs_code", "sandboxed", "simulator"})
    registry.register("Executor", SIM_EXECUTOR, executor_class(SIM_EXECUTOR, sim, world))
    registry.register("Executor", COMPILE_ONLY, executor_class(COMPILE_ONLY, frozenset(), world))
    assert capability("HOST_COMPUTE_GUARD") == CAPABILITY, "the fakes declare the capability by its name"
    declared = frozenset({"diagnostics", CAPABILITY})
    registry.register("Toolchain", GUARDED, toolchain_class(GUARDED, declared, world))
    registry.register("Toolchain", NO_METHOD, toolchain_class(NO_METHOD, declared, world, method=False))
    registry.register("Toolchain", PLAIN_TT, toolchain_class(PLAIN_TT, frozenset({"diagnostics"}), world))
    registry.register("Toolchain", PLAIN_CPU, toolchain_class(PLAIN_CPU, frozenset({"diagnostics"}), world))
    registry.register("Toolchain", REAL_READER, real_reader_class(world))
    for name in STAGES:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    registry.register("ScoreProfile", "df-v0", DEFAULT_REGISTRY.get("ScoreProfile", "df-v0").factory)
    return registry


# ---------------------------------------------------------------------------
# The SYNTHETIC suite, recipes, and runs


def write_manifest(directory: Path, support_text: str = SUPPORT_TEXT) -> Path:
    """Write the SYNTHETIC guardfix manifest (one item, cpu and tt, one support file) and return its path."""
    languages = {
        SOURCE: {"dir": "src/vadd-cpu", "files": ["main.cpp"]},
        TARGET: {"dir": "src/vadd-tt", "files": [TARGET_FILE]},
    }
    item = {"split": "eval", "languages": languages, "support": {SUPPORT: SUPPORT_PATH}}
    data = {"suite": SUITE, "repo": "https://example.invalid/guardfix", "commit": FAKE_COMMIT, "items": {ITEM: item}}
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / f"{SUITE}.yaml"
    manifest.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    bench = directory.parent / "bench"
    for language, layout in languages.items():
        path = bench / layout["dir"] / layout["files"][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(SOURCES[language].encode("ascii"))
    (bench / SUPPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (bench / SUPPORT_PATH).write_bytes(support_text.encode("utf-8"))
    return manifest


def recipe_data(tt: str = GUARDED, executor: str = SIM_EXECUTOR, **changes: Any) -> dict[str, Any]:
    """Return a one-trial template-set recipe for the item, cpu to tt, with only the target reference built."""
    data: dict[str, Any] = {
        "extends": "base",
        "faithful": False,
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": SOURCE, "target": TARGET}],
        "prompts": "p0-smoke",
        "toolchain": {SOURCE: PLAIN_CPU, TARGET: tt},
        "stages": list(STAGES),
        "executor": {"kind": executor},
        "fixes": {"baseline_both": False},
        "trials": {"n": 1},
    }
    data.update(changes)
    return data


@dataclass
class Outcome:
    """One finished run: its directory, its one trial read back from trial.json, and the world."""

    run_dir: Path
    trial: Trial
    world: World

    @property
    def trial_path(self) -> Path:
        """Return the trial's directory."""
        return trial_dir(self.run_dir, self.trial.trial_id)

    def text(self, ref: Any) -> str:
        """Return a text of the run's text store."""
        return TextStore(self.run_dir).get(ref)

    def request_prompt(self, index: int) -> str:
        """Return the user message of request `index`, as stored."""
        requests = self.trial.requests or []
        return self.text(requests[index].messages[-1].ref)


def run_one(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    world: World,
    data: Mapping[str, Any],
    *,
    direction: str = DIRECTION,
    support_text: str = SUPPORT_TEXT,
    observer: Callable[[ProgressEvent], None] | None = None,
) -> Outcome:
    """Run the one-trial recipe `data` under `root` on the SYNTHETIC suite with the fakes following `world`."""
    manifest = write_manifest(root / "manifests", support_text)
    monkeypatch.setattr(runner_module, "BENCH_DIR", manifest.parent)
    options = RunOptions(
        runs_root=root / "runs-root", run_id="test-run", bench_root=root / "bench", registry=make_registry(world),
        observer=observer,
    )
    path = root / f"{PROJECT}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    run_dir = run_recipe(path, options)
    trial_id = make_trial_id(PROJECT, MODEL_ID, SUITE, direction, ITEM, 1)
    return Outcome(run_dir, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir)), world)


def clean_run(stdout: str = "SYNTHETIC attempt stdout\n") -> RunResult:
    """Return a SYNTHETIC clean simulator run of an attempt: exit status 0, no UB, no gap."""
    return RunResult(exit_code=0, hang=False, stdout=stdout, stderr="", wall_s=PLACEHOLDER_ATTEMPT_WALL_S, sim_ub=False)


def failed_run() -> RunResult:
    """Return a SYNTHETIC failed simulator run of an attempt: exit status 1."""
    return RunResult(exit_code=1, hang=False, stdout="", stderr="SYNTHETIC failure\n", wall_s=0.5, sim_ub=False)


def references() -> dict[str, RunResult]:
    """Return the SYNTHETIC clean reference runs of both languages."""
    run = RunResult(
        exit_code=0, hang=False, stdout="SYNTHETIC reference stdout\n", stderr="", wall_s=PLACEHOLDER_REFERENCE_WALL_S,
        sim_ub=False,
    )
    return {SOURCE: run, TARGET: run}


def weights() -> Any:
    """Return the df-v0 weights a recipe's `score: df-v0` reads."""
    return load_weights(WEIGHTS_FILE)


def codes(attempt: Attempt) -> list[str | None]:
    """Return the code of each of the attempt's diagnostics, in order."""
    return [item.code for item in attempt.diagnostics]


# ---------------------------------------------------------------------------
# The capability and its seam (lassi.core.capabilities)


class Declaring:
    """A SYNTHETIC toolchain that declares the capability and answers with `answer`."""

    name = "synthetic-declaring"
    capabilities = frozenset({"diagnostics", CAPABILITY})

    def __init__(self, answer: Any) -> None:
        """Keep the answer and an empty call log."""
        self.answer = answer
        self.calls: list[tuple[Any, Any]] = []

    def host_compute_guard(self, files: Mapping[str, str], harness: Mapping[str, str]) -> Any:
        """Record the call and return the answer."""
        self.calls.append((files, harness))
        return self.answer


class Undeclaring(Declaring):
    """The same toolchain without the capability: its method is never called."""

    name = "synthetic-undeclaring"
    capabilities = frozenset({"diagnostics"})


def test_the_capability_names_are_the_documented_values() -> None:
    assert capability("HOST_COMPUTE_GUARD") == CAPABILITY
    assert capability("GUARD_CODE_PREFIX") == "guard-"
    assert capability("GUARD_STAGE") == "parse"
    assert frozenset(capability("GUARD_SEVERITIES")) == frozenset({"warning", "note"})


def test_a_reading_is_a_frozen_record_with_no_diagnostics_by_default() -> None:
    record = capability("HostComputeReading")(True)
    assert (record.host_compute, record.diagnostics) == (True, ())
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.host_compute = False


def test_a_toolchain_without_the_capability_is_not_checked() -> None:
    seam = capability("host_compute_reading")
    toolchain = Undeclaring(reading(True, GUARD_NOTE))
    assert seam(toolchain, {"main.cpp": ""}, {}) is None
    assert toolchain.calls == [], "the method of a toolchain that does not declare the capability is never called"
    assert seam(SimpleNamespace(), {"main.cpp": ""}, {}) is None, "no capabilities at all counts as none"


def test_a_declaring_toolchain_gives_its_reading() -> None:
    seam = capability("host_compute_reading")
    answer = reading(False, TAG)
    toolchain = Declaring(answer)
    files, harness = {"main.cpp": "int main() { return 0; }\n"}, {SUPPORT: SUPPORT_TEXT}
    assert seam(toolchain, files, harness) is answer
    ((seen_files, seen_harness),) = toolchain.calls
    assert (dict(seen_files), dict(seen_harness)) == (files, harness)
    for value in (True, None):
        assert seam(Declaring(reading(value)), files, harness).host_compute is value


# Each answer outside the contract, built from the HostComputeReading class.
BAD_ANSWERS: dict[str, Callable[[type], Any]] = {
    "not-a-reading": lambda cls: (True, ()),
    "host-compute-yes": lambda cls: cls("yes", ()),
    "host-compute-one": lambda cls: cls(1, ()),
    "diagnostics-list": lambda cls: cls(True, [GUARD_NOTE]),
    "item-not-a-diagnostic": lambda cls: cls(True, ("SYNTHETIC text",)),
    "error-severity": lambda cls: cls(True, (dataclasses.replace(GUARD_NOTE, severity="error"),)),
    "compile-stage": lambda cls: cls(False, (dataclasses.replace(TAG, stage="compile"),)),
    "run-stage": lambda cls: cls(False, (dataclasses.replace(TAG, stage="run"),)),
    "code-without-prefix": lambda cls: cls(True, (dataclasses.replace(GUARD_NOTE, code="host-compute"),)),
    "code-none": lambda cls: cls(None, (dataclasses.replace(GUARD_NOTE, code=None),)),
}


@pytest.mark.parametrize("case", sorted(BAD_ANSWERS))
def test_a_reading_outside_the_contract_is_refused_naming_the_toolchain(case: str) -> None:
    seam = capability("host_compute_reading")
    toolchain = Declaring(BAD_ANSWERS[case](capability("HostComputeReading")))
    with pytest.raises(ValueError) as caught:
        seam(toolchain, {"main.cpp": ""}, {})
    assert Declaring.name in str(caught.value) and CAPABILITY in str(caught.value), str(caught.value)


@pytest.mark.parametrize("attribute", ["missing", "not-callable"])
def test_a_declared_capability_without_a_callable_method_is_refused(attribute: str) -> None:
    seam = capability("host_compute_reading")
    toolchain: Any = SimpleNamespace(name="synthetic-no-method", capabilities=frozenset({CAPABILITY}))
    if attribute == "not-callable":
        toolchain.host_compute_guard = 3
    with pytest.raises(ValueError) as caught:
        seam(toolchain, {"main.cpp": ""}, {})
    assert "synthetic-no-method" in str(caught.value) and CAPABILITY in str(caught.value)


def test_a_toolchain_without_a_name_is_named_by_its_class() -> None:
    class NamelessToolchain:
        """A SYNTHETIC toolchain with no name attribute."""

        capabilities = frozenset({CAPABILITY})

        def host_compute_guard(self, files: Mapping[str, str], harness: Mapping[str, str]) -> Any:
            """Return an answer outside the contract."""
            return (True, ())

    with pytest.raises(ValueError, match="NamelessToolchain"):
        capability("host_compute_reading")(NamelessToolchain(), {}, {})


def test_prompt_diagnostics_drops_only_guard_codes() -> None:
    near = Diagnostic(stage="parse", severity="note", code="guard", message="SYNTHETIC code without the hyphen")
    plural = Diagnostic(stage="parse", severity="note", code="guards-x", message="SYNTHETIC code guards-x")
    no_code = Diagnostic(stage="run", severity="error", message="SYNTHETIC run error with no code")
    items = [COMPILE_WARNING, TAG, near, GUARD_NOTE, plural, no_code, JIT_ERROR]
    before = list(items)
    kept = prompt_diagnostics()(items)
    assert kept == [COMPILE_WARNING, near, plural, no_code, JIT_ERROR] and isinstance(kept, list)
    assert items == before, "the list given is left unchanged"
    assert "guard-" not in diagnostics_text(kept)
    assert prompt_diagnostics()([]) == []


def test_the_seam_in_core_names_no_tt_concept() -> None:
    capability("host_compute_reading")
    prompt_diagnostics()
    for module in (stages_module, capabilities_module):
        source = inspect.getsource(module)
        named = [name for name in TT_NAMES if name in source]
        assert named == [], f"{module.__name__} names {named}; core names no TT concept (Design Principle 8)"


# ---------------------------------------------------------------------------
# compile_loop records the reading (lassi.core.stages, CompileLoopStage._guarded)


def test_reading_is_recorded_after_the_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(
        replies=[REPLY], references=references(), attempt_runs=[clean_run()], builds=["warn"],
        readings=[reading(True, GUARD_NOTE)],
    )
    outcome = run_one(tmp_path, monkeypatch, world, recipe_data(score="df-v0"))
    trial = outcome.trial
    (attempt,) = trial.attempts
    assert attempt.stage_reached == "S5" and trial.final.end_reason is None, "a guard never changes a stage"
    assert attempt.guards == Guards(host_compute=True), "harness_tamper and oracle_access stay not checked"
    assert attempt.diagnostics == [COMPILE_WARNING, GUARD_NOTE], "the guard's diagnostics follow the build's"
    assert [request.stage for request in trial.requests or []] == ["generate"], "a violation asks no correction"
    assert len(world.guard_calls) == 1, "one reading per built attempt; the baseline asks for none"
    data = json.loads((outcome.trial_path / "trial.json").read_bytes().decode("utf-8"))
    assert data["attempts"][0]["guards"]["host_compute"] is True
    page = (outcome.trial_path / "trial.md").read_bytes().decode("utf-8")
    guards_table = page.split("### Guards\n", 1)[1].split("###", 1)[0]
    assert "| host_compute | true |" in guards_table
    (row,) = read_run_parquet(outcome.run_dir / "parquet")["attempts"]
    assert row["guards_host_compute"] is True
    rules = weights()
    assert attempt.score.components["guard"] == 1.0 and attempt.score.scalar == pytest.approx(rules.guard_violation)
    assert trial.final.score == pytest.approx(rules.guard_violation - rules.correction_penalty * 0)


def test_the_attempt_event_after_the_build_carries_the_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[ProgressEvent] = []
    world = World(
        replies=[REPLY], references=references(), attempt_runs=[clean_run()], readings=[reading(True, GUARD_NOTE)]
    )
    run_one(tmp_path, monkeypatch, world, recipe_data(), observer=events.append)
    built = [event for event in events if event.kind == ATTEMPT and event.stage == "compile_loop"]
    assert built, "compile_loop sends an attempt event after the build"
    last = built[0].trial.attempts[-1]
    assert (last.stage_reached, last.guards.host_compute) == ("S4", True)
    assert last.diagnostics[-1] == GUARD_NOTE


def test_a_violation_after_corrections_scores_the_penalty_on_top(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(
        replies=[REPLY, FIXED_REPLY], references=references(), attempt_runs=[clean_run()], builds=["fail", "ok"],
        readings=[reading(True, GUARD_NOTE)],
    )
    trial = run_one(tmp_path, monkeypatch, world, recipe_data(score="df-v0")).trial
    first, second = trial.attempts
    assert (first.stage_reached, second.stage_reached) == ("S1", "S5")
    assert "guard-host-compute" not in codes(first), "a failed build is not checked"
    assert first.guards == Guards(host_compute_not_checked="no-program"), "P17.14: the build gave no program"
    assert second.guards.host_compute is True and len(world.guard_calls) == 1
    rules = weights()
    assert first.score.scalar == pytest.approx(rules.stage_base["S1"])
    assert trial.final.corrections == 1
    assert trial.final.score == pytest.approx(rules.guard_violation - rules.correction_penalty * 1)


def test_tag_changes_no_stage_warning_count_or_correction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = {}
    for name, answer in (("tagged", reading(False, TAG)), ("untagged", reading(False))):
        world = World(
            replies=[REPLY], references=references(), attempt_runs=[clean_run()], builds=["warn"], readings=[answer]
        )
        outcomes[name] = run_one(tmp_path / name, monkeypatch, world, recipe_data(score="df-v0")).trial
    tagged, untagged = outcomes["tagged"].attempts[0], outcomes["untagged"].attempts[0]
    assert tagged.stage_reached == untagged.stage_reached == "S5"
    assert tagged.diagnostics == [COMPILE_WARNING, TAG] and untagged.diagnostics == [COMPILE_WARNING]
    assert not [item for item in tagged.diagnostics if item.code == "run-error"]
    assert tagged.guards.host_compute is False
    assert tagged.score.components["warning_count"] == 1.0, "W counts compile and jit warnings, never a parse tag"
    assert tagged.score.components["guard"] is None, "harness_tamper and oracle_access are not checked"
    assert tagged.score == untagged.score, "the tag changes no df-v0 term"
    finals = [dataclasses.replace(outcomes[name].final, wall_s=None) for name in ("tagged", "untagged")]
    assert finals[0] == finals[1], "nor anything of the trial's final block but its measured pipeline time"
    assert [request.stage for request in outcomes["tagged"].requests or []] == ["generate"]


def test_no_reading_without_the_capability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(replies=[REPLY], references=references(), attempt_runs=[clean_run()], readings=[reading(True)])
    trial = run_one(tmp_path, monkeypatch, world, recipe_data(tt=PLAIN_TT)).trial
    assert world.guard_calls == [], "found by capability, never by method"
    assert trial.attempts[0].guards == Guards(host_compute_not_checked="no-guard"), "P17.14: no guard declared"


def test_no_reading_in_the_tt_to_cpu_direction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(replies=[CPU_REPLY], references=references(), attempt_runs=[clean_run()], readings=[reading(True)])
    data = recipe_data(directions=[{"source": TARGET, "target": SOURCE}])
    trial = run_one(tmp_path, monkeypatch, world, data, direction=f"{TARGET}-{SOURCE}").trial
    assert trial.attempts[0].stage_reached == "S5"
    assert world.guard_calls == [], "the cpu toolchain declares no guard"
    assert trial.attempts[0].guards == Guards(host_compute_not_checked="no-guard"), "P17.14: no guard declared"


def test_guard_receives_attempt_files_and_support_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(replies=[REPLY], references=references(), attempt_runs=[clean_run()], readings=[reading(False)])
    trial = run_one(tmp_path, monkeypatch, world, recipe_data()).trial
    assert world.guard_calls == [(trial.attempts[0].files, {SUPPORT: SUPPORT_TEXT})]
    assert set(trial.attempts[0].files) == {TARGET_FILE, "kernels/reader.cpp"}


def test_a_compile_only_executor_attempt_carries_the_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(replies=[REPLY], readings=[reading(True, GUARD_NOTE)])
    trial = run_one(tmp_path, monkeypatch, world, recipe_data(executor=COMPILE_ONLY)).trial
    (attempt,) = trial.attempts
    assert (attempt.stage_reached, attempt.guards.host_compute) == ("S4", True), "built, never run, still read"


def test_an_attempt_past_the_execution_gate_keeps_its_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gate = 7
    world = World(
        replies=[REPLY] + [FIXED_REPLY] * (gate + 1),
        references=references(),
        attempt_runs=[failed_run() for _ in range(gate + 1)],
        readings=[reading(False)] * (gate + 1) + [reading(True, GUARD_NOTE)],
    )
    data = recipe_data(fixes={"baseline_both": False, "execution_gate": False}, loop={"max_corrections": 10})
    trial = run_one(tmp_path, monkeypatch, world, data).trial
    last = trial.attempts[-1]
    assert last.index == gate + 1 and last.stage_reached == "S4"
    assert last.guards.host_compute is True, "the stale-output copy keeps the reading"
    assert last.diagnostics[-1].code == "stale-output" and GUARD_NOTE in last.diagnostics


# ---------------------------------------------------------------------------
# A kernel JIT failure, the correction prompt, and the reading (lassi.core.stages prompt_diagnostics)


def test_jit_failure_keeps_reading_and_prompt_has_no_guard_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jit_failed = RunResult(exit_code=1, hang=False, stdout="", stderr="", sim_ub=False, diagnostics=[JIT_ERROR])
    world = World(
        replies=[REPLY, FIXED_REPLY], references=references(), attempt_runs=[jit_failed, clean_run()],
        builds=["warn", "ok"], readings=[reading(True, TAG, GUARD_NOTE), reading(False)],
    )
    outcome = run_one(tmp_path, monkeypatch, world, recipe_data(score="df-v0"))
    trial = outcome.trial
    first, second = trial.attempts
    assert (first.stage_reached, second.stage_reached) == ("S1", "S5")
    assert first.guards.host_compute is True, "the reading taken at build stands on a kernel JIT failure"
    assert {TAG, GUARD_NOTE, COMPILE_WARNING, JIT_ERROR} <= set(first.diagnostics)
    assert [request.stage for request in trial.requests or []] == ["generate", "compile_loop"]
    prompt = outcome.request_prompt(1)
    assert JIT_ERROR.message in prompt and COMPILE_WARNING.message in prompt, "the non-guard diagnostics are fed back"
    assert "guard-" not in prompt and TAG.message not in prompt and GUARD_NOTE.message not in prompt
    assert second.guards.host_compute is False and len(world.guard_calls) == 2, "the next build is read again"
    rules = weights()
    assert trial.final.score == pytest.approx(rules.guard_violation - rules.correction_penalty * 1), "OQ-028 3(b)"


# SYNTHETIC stand-ins for a fragment set's correction fragments for cpu to tt; not upstream's text.
FRAGMENTS = {
    "system_prompt_dict.CPU_to_TT": "SYNTHETIC system prompt for cpu to tt",
    "correct.compile_error_head": "\nSYNTHETIC compile error head\n",
    "setup.tt.compiler": "SYNTHETIC-compiler",
    "setup.tt.flags": "SYNTHETIC-flags",
    "correct.compile_error_tail": "\nSYNTHETIC compile error tail\n",
    "correct.outro": "\nSYNTHETIC outro\n",
}


def stage_context(tmp_path: Path, world: World, tt: type, fragments: Mapping[str, str] | None = None) -> RunContext:
    """Return a RunContext for the cpu to tt direction of the SYNTHETIC item, with the toolchain class `tt`."""
    manifest = write_manifest(tmp_path / "manifests")
    executor = executor_class(COMPILE_ONLY, frozenset(), world)()
    return RunContext(
        recipe=SimpleNamespace(data={"fixes": {}}, bindings=()),  # type: ignore[arg-type]
        backend=backend_class(world)(MODEL_ID),
        sampling=SAMPLING,
        toolchains={TARGET: tt()},
        executor=executor,
        store=TextStore(tmp_path / "store"),
        suite=load_suite(manifest),
        sources_root=manifest.parent.parent / "bench",
        item=ITEM,
        direction=Direction(SOURCE, TARGET),
        build_root=tmp_path / "builds",
        prompts="p0-smoke",
        max_corrections=3,
        fragments=dict(fragments or {}),
    )


def one_attempt_trial(context: RunContext, attempt: Attempt) -> Trial:
    """Return a SYNTHETIC trial of the item holding `attempt` alone."""
    return Trial(
        trial_id=make_trial_id("stage-test", MODEL_ID, SUITE, DIRECTION, ITEM, 1),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(
            commit=FAKE_COMMIT, dirty=False, device="SYNTHETIC device", sdk=None, date="2026-10-01T00:00:00+00:00"
        ),
        bench_item=BenchItem(suite=SUITE, item=ITEM, split="eval", direction=DIRECTION),
        model=ModelInfo(backend="scripted", id=MODEL_ID, sampling=SAMPLING),
        requests=[],
        attempts=[attempt],
        final=Final(stage_reached=attempt.stage_reached),
    )


@pytest.mark.parametrize("prompt_set", ["template", "fragment"])
def test_the_correction_prompt_has_no_guard_text_under_both_prompt_sets(tmp_path: Path, prompt_set: str) -> None:
    world = World(replies=[FIXED_REPLY], readings=[reading(False)])
    tt = toolchain_class(GUARDED, frozenset({"diagnostics", CAPABILITY}), world)
    context = stage_context(tmp_path, world, tt, FRAGMENTS if prompt_set == "fragment" else None)
    jit_failed = Attempt(
        index=0, prompt_ref=context.store.put("SYNTHETIC prompt 0\n"), response_text=REPLY,
        files={TARGET_FILE: "// SYNTHETIC host program\nint main() { return 0; }\n"}, stage_reached="S1",
        diagnostics=[COMPILE_WARNING, TAG, GUARD_NOTE, JIT_ERROR], guards=Guards(host_compute=True),
    )
    trial = CompileLoopStage(context=context)(one_attempt_trial(context, jit_failed))
    first, second = trial.attempts
    assert first == jit_failed, "the corrected attempt is kept as it was, reading included"
    ((*system, user),) = world.requests
    prompt = user.content
    assert prompt == context.store.get(second.prompt_ref)
    assert JIT_ERROR.message in prompt and COMPILE_WARNING.message in prompt
    assert "guard-" not in prompt and TAG.message not in prompt and GUARD_NOTE.message not in prompt
    if prompt_set == "fragment":
        assert [message.content for message in system] == [FRAGMENTS["system_prompt_dict.CPU_to_TT"]]
    assert (second.stage_reached, second.guards.host_compute) == ("S4", False), "the correction is read again"


def test_compile_loop_stays_pure_and_orders_parse_build_guard_diagnostics(tmp_path: Path) -> None:
    world = World(builds=["warn"], readings=[reading(True, TAG, GUARD_NOTE)])
    tt = toolchain_class(GUARDED, frozenset({"diagnostics", CAPABILITY}), world)
    context = stage_context(tmp_path, world, tt)
    files = {TARGET_FILE: "// SYNTHETIC host program\nint main() { return 0; }\n"}
    parsed = Attempt(
        index=0, prompt_ref=context.store.put("SYNTHETIC prompt 0\n"), response_text=REPLY, files=files,
        stage_reached="S1", diagnostics=[PARSE_WARNING],
    )
    trial = one_attempt_trial(context, parsed)
    before = to_dict(trial)
    (built,) = CompileLoopStage(context=context)(trial).attempts
    assert to_dict(trial) == before, "the trial given is never changed"
    assert built.diagnostics == [PARSE_WARNING, COMPILE_WARNING, TAG, GUARD_NOTE], "parse, then build, then guard"
    assert (built.stage_reached, built.guards) == ("S4", Guards(host_compute=True))
    assert world.guard_calls == [(files, {SUPPORT: SUPPORT_TEXT})]
    assert 0 in context.artifacts


def test_describe_names_guard_only_when_declared(tmp_path: Path) -> None:
    world = World()
    tail = (
        ", then correct from p0-smoke/correct.txt with parsed diagnostics while an error remains "
        "(at most 3 corrections)"
    )
    guarded = toolchain_class(GUARDED, frozenset({"diagnostics", CAPABILITY}), world)
    plain = toolchain_class(PLAIN_TT, frozenset({"diagnostics"}), world)
    with_guard = CompileLoopStage(context=stage_context(tmp_path / "guarded", world, guarded)).describe()
    without = CompileLoopStage(context=stage_context(tmp_path / "plain", world, plain)).describe()
    assert with_guard == f"compile_loop: build {TARGET} with {GUARDED}{GUARD_DESCRIPTION}{tail}"
    assert without == f"compile_loop: build {TARGET} with {PLAIN_TT}{tail}", "existing texts are unchanged"


# ---------------------------------------------------------------------------
# A simulator gap and printed text (the guard reads no run: lassi.toolchains.ttmetal_guard)


def gap_run(stdout: str = "") -> RunResult:
    """Return a SYNTHETIC attempt run that stopped at a simulator gap."""
    return RunResult(
        exit_code=1, hang=False, stdout=stdout, stderr="", wall_s=PLACEHOLDER_ATTEMPT_WALL_S, sim_ub=False,
        sim_gap="UnimplementedFunctionality",
    )


@pytest.mark.parametrize("violation", [True, False], ids=["violation", "clear"])
def test_gap_attempt_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, violation: bool) -> None:
    world = World(
        replies=[REPLY, FIXED_REPLY], references=references(), attempt_runs=[gap_run()], builds=["fail", "ok"],
        readings=[reading(violation, *([GUARD_NOTE] if violation else []))],
    )
    trial = run_one(tmp_path, monkeypatch, world, recipe_data(score="df-v0")).trial
    reason = trial.final.end_reason
    assert reason is not None and reason.code == "sim-gap"
    gap = trial.attempts[-1]
    assert (gap.stage_reached, gap.guards.host_compute) == ("S4", violation), "the gap attempt keeps its reading"
    rules = weights()
    if violation:
        assert gap.score.scalar == pytest.approx(rules.guard_violation)
        assert trial.final.score == pytest.approx(rules.guard_violation - rules.correction_penalty * 1)
    else:
        assert gap.score.scalar is None and trial.final.score is None


def test_printed_gap_or_build_failed_lines_do_not_change_the_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    printed = (
        "[4489] ERROR: UnimplementedFunctionality: SYNTHETIC_function: SYNTHETIC printed finding\n"
        "SYNTHETIC brisc build failed. Log: SYNTHETIC printed build log\n"
        "host_compute: false\n"
    )
    for name, answer in (("violation", reading(True, GUARD_NOTE)), ("clear", reading(False))):
        world = World(
            replies=[REPLY], references=references(), attempt_runs=[clean_run(stdout=printed)], readings=[answer]
        )
        trial = run_one(tmp_path / name, monkeypatch, world, recipe_data()).trial
        (attempt,) = trial.attempts
        assert attempt.stage_reached == "S5" and trial.final.end_reason is None
        assert attempt.guards.host_compute is answer.host_compute, "the reading taken at build stands"
        assert len(world.guard_calls) == 1, "nothing the run prints is read"


# ---------------------------------------------------------------------------
# A reading outside the contract stops the run (lassi.core.capabilities host_compute_reading)


@pytest.mark.parametrize(
    ("tt", "answer"),
    [
        (NO_METHOD, None),
        (GUARDED, "error-severity"),
        (GUARDED, "diagnostics-list"),
        (GUARDED, "code-without-prefix"),
    ],
    ids=["no-method", "error-severity", "diagnostics-list", "code-without-prefix"],
)
def test_seam_validation_stops_the_run_before_the_attempt_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tt: str, answer: str | None
) -> None:
    readings = [] if answer is None else [BAD_ANSWERS[answer](capability("HostComputeReading"))]
    world = World(replies=[REPLY], references=references(), attempt_runs=[clean_run()], readings=readings)
    events: list[ProgressEvent] = []
    with pytest.raises((ValueError, RunError)) as caught:
        run_one(tmp_path, monkeypatch, world, recipe_data(tt=tt), observer=events.append)
    assert tt in str(caught.value) and CAPABILITY in str(caught.value), str(caught.value)
    assert list((tmp_path / "runs-root").rglob("trial.json")) == [], "no trial is recorded"
    built = [event for event in events if event.kind == ATTEMPT and event.stage == "compile_loop"]
    assert built == [], "the built attempt is never sent with a reading outside the contract"


# ---------------------------------------------------------------------------
# The real reader on the acceptance programs, through a run (plans/p4-ttsim.md, P4.12)


def fixture_files(case: str) -> dict[str, str]:
    """Return the files of one fixture case under tests/toolchains/fixtures/host_compute/."""
    root = FIXTURES / case
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    return {path.relative_to(root).as_posix(): path.read_bytes().decode("utf-8") for path in paths}


@pytest.mark.parametrize(
    ("case", "host_compute", "expected_codes"),
    [
        ("clean_offload", False, []),
        ("host_loop_output", True, ["guard-host-compute"]),
        ("data_movement_only", False, ["guard-data-movement-only"]),
        ("compute_beside_host_loop", False, []),
    ],
)
def test_end_to_end_with_real_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, host_compute: bool, expected_codes: list[str]
) -> None:
    real_guard()
    reply = render_file_blocks(fixture_files(case))
    world = World(replies=[reply], references=references(), attempt_runs=[clean_run()])
    header = HARNESS_HEADER.read_bytes().decode("utf-8")
    outcome = run_one(tmp_path, monkeypatch, world, recipe_data(tt=REAL_READER), support_text=header)
    data = json.loads((outcome.trial_path / "trial.json").read_bytes().decode("utf-8"))
    (attempt,) = data["attempts"]
    assert attempt["guards"]["host_compute"] is host_compute
    guard_codes = [item["code"] for item in attempt["diagnostics"] if str(item["code"]).startswith("guard-")]
    assert guard_codes == expected_codes
    assert attempt["stage_reached"] == "S5", "a guard reading never changes the stage"


# ---------------------------------------------------------------------------
# Why host_compute is null (task P17.14; OQ-028, the owner's condition of 2026-10-05)

REASON = "host_compute_not_checked"
NOT_BUILT, NO_PROGRAM, NO_GUARD, NOT_CHECKED = "not-built", "no-program", "no-guard", "guard-not-checked"
NOT_CHECKED_CODES = (NOT_BUILT, NO_PROGRAM, NO_GUARD, NOT_CHECKED)
# A SYNTHETIC note as the real reader gives it with a null reading (lassi.toolchains.ttmetal_guard).
NOT_CHECKED_NOTE = Diagnostic(
    stage="parse", severity="note", code="guard-not-checked", file=TARGET_FILE,
    message="SYNTHETIC guard note: the host text could not be read",
)
# SYNTHETIC replies that leave out the target's main.cpp: an S1 reply with a missing-file error, and an S0 reply.
PARTIAL_REPLY = render_file_blocks({"kernels/reader.cpp": "// SYNTHETIC kernel placeholder\nvoid kernel_main() {}\n"})
NO_BLOCK_REPLY = "SYNTHETIC reply that holds no FILE block\n"
CPU_PARTIAL_REPLY = render_file_blocks({"helper.h": "// SYNTHETIC helper header\n"})
# The run.md section and the columns of its table, by header cell.
COVERAGE_HEADING = "## Guard coverage"
COVERAGE_COLUMNS = (
    "Scope", "Attempts", "Checked", "Not checked", "Not-checked rate", *NOT_CHECKED_CODES, "not recorded",
)


def reason(code: str | None) -> Guards:
    """Return the Guards of an attempt whose host_compute is null for `code`."""
    return Guards(host_compute_not_checked=code)


def reasons(trial: Trial) -> list[str | None]:
    """Return each attempt's recorded code, in order."""
    return [attempt.guards.host_compute_not_checked for attempt in trial.attempts]


def jit_failed_run() -> RunResult:
    """Return a SYNTHETIC attempt run whose kernel JIT failed."""
    return RunResult(exit_code=1, hang=False, stdout="", stderr="", sim_ub=False, diagnostics=[JIT_ERROR])


@pytest.mark.parametrize("value", [True, False], ids=["violation", "clear"])
def test_a_set_reading_records_no_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    world = World(replies=[REPLY], references=references(), attempt_runs=[clean_run()], readings=[reading(value)])
    (attempt,) = run_one(tmp_path, monkeypatch, world, recipe_data()).trial.attempts
    assert attempt.guards == Guards(host_compute=value)
    assert attempt.guards.host_compute_not_checked is None, "a reading of true or false records no code"


def test_a_null_reading_records_guard_not_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(
        replies=[REPLY], references=references(), attempt_runs=[clean_run()], readings=[reading(None, NOT_CHECKED_NOTE)]
    )
    trial = run_one(tmp_path, monkeypatch, world, recipe_data(score="df-v0")).trial
    (attempt,) = trial.attempts
    assert attempt.stage_reached == "S5" and len(world.guard_calls) == 1, "precondition: built, read, and run"
    assert attempt.diagnostics[-1] == NOT_CHECKED_NOTE, "the guard's note still names the cause"
    assert attempt.guards == reason(NOT_CHECKED)
    assert attempt.score.components["guard"] is None, "not checked, as before the code existed"


@pytest.mark.parametrize("reply", [PARTIAL_REPLY, NO_BLOCK_REPLY], ids=["missing-file", "no-block"])
def test_an_unbuilt_attempt_records_not_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply: str) -> None:
    world = World(
        replies=[reply, FIXED_REPLY], references=references(), attempt_runs=[clean_run()], readings=[reading(False)]
    )
    trial = run_one(tmp_path, monkeypatch, world, recipe_data()).trial
    first, second = trial.attempts
    assert first.stage_reached in ("S0", "S1") and "missing-file" in codes(first), "precondition: never built"
    assert len(world.guard_calls) == 1, "only the corrected attempt is built and read"
    assert first.guards == reason(NOT_BUILT)
    assert second.guards == Guards(host_compute=False)


def test_an_attempt_of_a_recipe_without_compile_loop_records_not_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(replies=[REPLY], references=references())
    trial = run_one(tmp_path, monkeypatch, world, recipe_data(stages=["baseline", "generate"])).trial
    (attempt,) = trial.attempts
    assert attempt.stage_reached == "S1" and world.guard_calls == [], "precondition: parsed, never built"
    assert attempt.guards == reason(NOT_BUILT)


@pytest.mark.parametrize("build", ["fail", "none"], ids=["build-error", "no-artifact"])
def test_a_build_with_no_program_records_no_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, build: str
) -> None:
    world = World(
        replies=[REPLY, FIXED_REPLY], references=references(), attempt_runs=[clean_run()], builds=[build, "ok"],
        readings=[reading(True, GUARD_NOTE)],
    )
    trial = run_one(tmp_path, monkeypatch, world, recipe_data()).trial
    first, second = trial.attempts
    assert first.stage_reached == "S1" and len(world.guard_calls) == 1, "precondition: built with no program"
    assert ("no-artifact" in codes(first)) is (build == "none")
    assert first.guards == reason(NO_PROGRAM)
    assert second.guards == Guards(host_compute=True)


def no_guard_case(case: str) -> tuple[World, dict[str, Any], str]:
    """Return the world, recipe, and direction of a target whose toolchain declares no guard.

    Each run gives three attempts: a reply that leaves out main.cpp (never
    built), a build with no program, and a clean build and run.
    """
    if case == "tt-to-cpu":
        replies = [CPU_PARTIAL_REPLY, CPU_REPLY, CPU_REPLY]
        data = recipe_data(directions=[{"source": TARGET, "target": SOURCE}])
        direction = f"{TARGET}-{SOURCE}"
    else:
        replies, data, direction = [PARTIAL_REPLY, REPLY, FIXED_REPLY], recipe_data(tt=PLAIN_TT), DIRECTION
    world = World(
        replies=replies, references=references(), attempt_runs=[clean_run()], builds=["fail", "ok"],
        readings=[reading(True)],
    )
    return world, data, direction


@pytest.mark.parametrize("case", ["tt-to-cpu", "cpu-to-plain-tt"])
def test_every_attempt_of_a_target_without_a_guard_records_no_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    world, data, direction = no_guard_case(case)
    events: list[ProgressEvent] = []
    trial = run_one(tmp_path, monkeypatch, world, data, direction=direction, observer=events.append).trial
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S1", "S1", "S5"], "precondition"
    assert world.guard_calls == [], "no guard is ever asked"
    assert [attempt.guards for attempt in trial.attempts] == [reason(NO_GUARD)] * 3, "no-guard wins over the rest"
    sent = [event.trial.attempts[-1].guards for event in events if event.kind == ATTEMPT and event.trial.attempts]
    assert sent and all(guards == reason(NO_GUARD) for guards in sent), "every attempt event carries no-guard"


def test_the_attempt_events_carry_the_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[ProgressEvent] = []
    world = World(
        replies=[REPLY, FIXED_REPLY], references=references(), attempt_runs=[clean_run()], builds=["fail", "ok"],
        readings=[reading(None, NOT_CHECKED_NOTE)],
    )
    run_one(tmp_path, monkeypatch, world, recipe_data(), observer=events.append)
    attempt_events = [event for event in events if event.kind == ATTEMPT]
    generated = [event.trial.attempts[-1] for event in attempt_events if event.stage == "generate"]
    assert [(attempt.index, attempt.guards) for attempt in generated] == [(0, reason(NOT_BUILT))]
    looped = [event.trial.attempts[-1] for event in attempt_events if event.stage == "compile_loop"]
    assert [(attempt.index, attempt.guards) for attempt in looped] == [
        (0, reason(NO_PROGRAM)),  # the build of attempt 0 gave no program
        (1, reason(NOT_BUILT)),  # the correction's reply, appended
        (1, reason(NOT_CHECKED)),  # its build, read as null
    ]


def kept_case(case: str) -> tuple[World, dict[str, Any]]:
    """Return the world and recipe of a run whose attempt read null is later copied or left unrun."""
    null = reading(None, NOT_CHECKED_NOTE)
    if case == "jit":
        world = World(
            replies=[REPLY, FIXED_REPLY], references=references(), attempt_runs=[jit_failed_run(), clean_run()],
            readings=[null, reading(False)],
        )
        return world, recipe_data()
    if case == "gap":
        world = World(replies=[REPLY], references=references(), attempt_runs=[gap_run()], readings=[null])
        return world, recipe_data()
    if case == "compile-only":
        return World(replies=[REPLY], readings=[null]), recipe_data(executor=COMPILE_ONLY)
    gate = 7
    world = World(
        replies=[REPLY] + [FIXED_REPLY] * (gate + 1), references=references(),
        attempt_runs=[failed_run() for _ in range(gate + 1)], readings=[reading(False)] * (gate + 1) + [null],
    )
    return world, recipe_data(fixes={"baseline_both": False, "execution_gate": False}, loop={"max_corrections": 10})


@pytest.mark.parametrize("case", ["jit", "gap", "compile-only", "stale-output"])
def test_a_null_reading_keeps_its_reason_on_every_later_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    world, data = kept_case(case)
    trial = run_one(tmp_path, monkeypatch, world, data).trial
    attempt = trial.attempts[0] if case == "jit" else trial.attempts[-1]
    expected = {"jit": "S1", "gap": "S4", "compile-only": "S4", "stale-output": "S4"}[case]
    assert attempt.stage_reached == expected and NOT_CHECKED_NOTE in attempt.diagnostics, "precondition"
    assert attempt.guards == reason(NOT_CHECKED)


def test_a_run_with_every_reason_sends_none_of_them_to_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(
        replies=[PARTIAL_REPLY, REPLY, REPLY, FIXED_REPLY], references=references(),
        attempt_runs=[jit_failed_run(), clean_run()], builds=["fail", "ok", "ok"],
        readings=[reading(None, NOT_CHECKED_NOTE), reading(False)],
    )
    outcome = run_one(tmp_path, monkeypatch, world, recipe_data())
    assert reasons(outcome.trial) == [NOT_BUILT, NO_PROGRAM, NOT_CHECKED, None], "precondition: each code recorded"
    requests = outcome.trial.requests or []
    texts = [outcome.text(message.ref) for request in requests for message in request.messages]
    assert len(requests) == 4 and texts, "precondition: generate and three corrections"
    for text in texts:
        held = [name for name in (REASON, *NOT_CHECKED_CODES) if name in text]
        assert held == [], f"a request message holds {held} (OQ-041)"
    sent = [message.content for messages in world.requests for message in messages]
    assert not [name for text in sent for name in (REASON, *NOT_CHECKED_CODES) if name in text], "OQ-041"


@pytest.mark.parametrize("prompt_set", ["template", "fragment"])
@pytest.mark.parametrize("code", NOT_CHECKED_CODES)
def test_no_correction_prompt_holds_the_reason_under_both_prompt_sets(
    tmp_path: Path, prompt_set: str, code: str
) -> None:
    world = World(replies=[FIXED_REPLY], readings=[reading(False)])
    tt = toolchain_class(GUARDED, frozenset({"diagnostics", CAPABILITY}), world)
    context = stage_context(tmp_path, world, tt, FRAGMENTS if prompt_set == "fragment" else None)
    failed = Attempt(
        index=0, prompt_ref=context.store.put("SYNTHETIC prompt 0\n"), response_text=REPLY,
        files={TARGET_FILE: "// SYNTHETIC host program\nint main() { return 0; }\n"}, stage_reached="S1",
        diagnostics=[COMPILE_ERROR], guards=reason(code),
    )
    trial = CompileLoopStage(context=context)(one_attempt_trial(context, failed))
    first, second = trial.attempts
    assert first == failed, "the corrected attempt is kept as it was, its code included"
    ((*system, user),) = world.requests
    assert COMPILE_ERROR.message in user.content, "precondition: the compile error is fed back"
    sent = "\n".join(message.content for message in [*system, user])
    assert [name for name in (REASON, *NOT_CHECKED_CODES) if name in sent] == [], "OQ-041"
    assert second.guards == Guards(host_compute=False)


def test_the_reason_reaches_trial_json_trial_md_and_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(
        replies=[REPLY], references=references(), attempt_runs=[clean_run()], readings=[reading(None, NOT_CHECKED_NOTE)]
    )
    outcome = run_one(tmp_path, monkeypatch, world, recipe_data())
    data = json.loads((outcome.trial_path / "trial.json").read_bytes().decode("utf-8"))
    assert data["attempts"][0]["guards"] == {
        "host_compute": None, "harness_tamper": None, "oracle_access": None, REASON: NOT_CHECKED,
    }
    page = (outcome.trial_path / "trial.md").read_bytes().decode("utf-8")
    guards_table = page.split("### Guards\n", 1)[1].split("###", 1)[0]
    assert f"| oracle_access | PLACEHOLDER |\n| {REASON} | {NOT_CHECKED} |\n" in guards_table
    (row,) = read_run_parquet(outcome.run_dir / "parquet")["attempts"]
    assert row["guards_host_compute"] is None and row["guards_host_compute_not_checked"] == NOT_CHECKED


def coverage_rows(run_md: str) -> dict[str, dict[str, str]]:
    """Return run.md's Guard coverage table as {scope: {column: cell}} without the Scope cell.

    The section must appear once, after the Trials section; it ends at the
    next second-level heading or at the end of run.md.
    """
    lines = run_md.split("\n")
    assert lines.count(COVERAGE_HEADING) == 1, f"run.md has {lines.count(COVERAGE_HEADING)} {COVERAGE_HEADING!r} lines"
    start = lines.index(COVERAGE_HEADING)
    assert lines.index("## Trials") < start, "the Guard coverage section follows the Trials section"
    end = next((index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")), len(lines))
    table = [line.strip() for line in lines[start:end] if line.startswith("|")]
    cells = [[cell.strip() for cell in line[1:-1].split("|")] for line in table]
    header, rows = cells[0], cells[2:]
    assert tuple(header) == COVERAGE_COLUMNS
    return {row[0]: dict(zip(header[1:], row[1:], strict=True)) for row in rows}


def coverage(attempts: int, checked: int, rate: str, **counts: int) -> dict[str, str]:
    """Return the expected cells of one Guard coverage row, without its scope; `counts` are keyed by code.

    A code's key is the code with each hyphen or space as an underscore
    (not_built, no_program, no_guard, guard_not_checked, not_recorded); a
    code not given counts 0.
    """
    cells = {
        "Attempts": str(attempts), "Checked": str(checked), "Not checked": str(attempts - checked),
        "Not-checked rate": rate,
    }
    for name in (*NOT_CHECKED_CODES, "not recorded"):
        cells[name] = str(counts.get(name.replace("-", "_").replace(" ", "_"), 0))
    return cells


def run_md_of(outcome: Outcome) -> str:
    """Return the run's run.md as ASCII text."""
    return (outcome.run_dir / "run.md").read_bytes().decode("ascii")


def test_run_md_reports_guard_coverage_per_run_and_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # cpu -> tt: never built, no program, read null then a kernel JIT failure, and read false: 1 checked of 4.
    world = World(
        replies=[PARTIAL_REPLY, REPLY, REPLY, FIXED_REPLY], references=references(),
        attempt_runs=[jit_failed_run(), clean_run()], builds=["fail", "ok", "ok"],
        readings=[reading(None, NOT_CHECKED_NOTE), reading(False)],
    )
    outcome = run_one(tmp_path, monkeypatch, world, recipe_data())
    assert reasons(outcome.trial) == [NOT_BUILT, NO_PROGRAM, NOT_CHECKED, None], "precondition"
    rows = coverage_rows(run_md_of(outcome))
    # Hand counts: 4 attempts, 1 checked, 3 not checked (one each of not-built, no-program, guard-not-checked).
    want = coverage(4, 1, "0.750", not_built=1, no_program=1, guard_not_checked=1)
    assert rows == {"run": want, f"target {TARGET}": want}
    assert list(rows) == ["run", f"target {TARGET}"], "the run row, then each target"


def test_run_md_reports_every_attempt_of_a_target_without_a_guard_as_no_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, data, direction = no_guard_case("tt-to-cpu")
    outcome = run_one(tmp_path, monkeypatch, world, data, direction=direction)
    rows = coverage_rows(run_md_of(outcome))
    # Hand counts: 3 attempts, none checked, all no-guard.
    want = coverage(3, 0, "1.000", no_guard=3)
    assert rows == {"run": want, f"target {SOURCE}": want}


def test_run_md_reports_a_run_with_no_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failed_reference = RunResult(
        exit_code=1, hang=False, stdout="", stderr="SYNTHETIC failure\n", wall_s=0.5, sim_ub=False
    )
    world = World(references={SOURCE: references()[SOURCE], TARGET: failed_reference})
    outcome = run_one(tmp_path, monkeypatch, world, recipe_data())
    assert outcome.trial.attempts == [] and outcome.trial.final.end_reason is not None, "precondition"
    rows = coverage_rows(run_md_of(outcome))
    assert rows["run"] == coverage(0, 0, "-"), "a run with no attempt has no rate"
