"""Remote tests of the adversary stage on fixture programs outside adv-seeded-v0 (task P16.10).

Bible: Adversarial Setups (item 1; the adversary rule; Input contract),
Component Interfaces (Stage; Executor; Oracle), Sandbox (native program runs;
attempt run limits; model-generated code), Execution Backends (the native
row), Oracles (binary_io), Toolchain Pins (gcc), Agent Rules 5, 6, and 7;
AGENTS.md Local Execution. Plan: plans/p16-adversarial.md, P16.10, the third
acceptance bullet's first half: remote tests of the stage on fixture programs
outside adv-seeded-v0, whose proposals run only in the sandbox, a refused one
starting no process. tests/core/test_adversary_stage.py fixes the same stage
with fakes on any host; here the real thing runs on Linux.

The world: a one-item suite, adv-stage-native, whose manifest each test
writes under its own temp directory (not under assets/bench, so no suite
lists it). Its item, scale, has one f32 input x of shape [4], held out in
[0, 10) with seed 11, one output y, a contract that widens x to
[-1000, 1000), and a tolerance of max_abs 0.001; its cpp language reads
tests/fixtures/adversary/stage/ as the bench root, with
assets/harness/c/lassi_io.h as its support file. The reference,
scale/cpp/scale.cpp, writes y = 2x; the faulty candidate,
candidates/scale/large-zero.cpp, writes 0 where |x| >= 100, so it agrees on
the held-out inputs and diverges inside the contract; the control is the
reference handed back as its own candidate. Every program there is
SYNTHETIC, written by hand (its README.md); no held-out input of an eval or
unassigned item comes from it (Agent Rule 5).

Each test runs one recipe through lassi.core.runner run_recipe: the stages
baseline, generate, compile_loop, run_loop, oracle, and adversary, direction
cpp -> cpp, toolchain gcc-native (the pinned host g++ in the compile sandbox),
executor native, oracle binary_io at max_abs 0.001, one trial, no
correction. The run's registry holds those real components as
DEFAULT_REGISTRY registers them, plus two fakes: an LLMBackend that answers
every request with the candidate's file as one FILE block, and the scripted
adversary of tests/core/adversary_support.py, which proposes the test's
batches in turn and records each view it gets. No model is called or loaded.
Three spies observe without changing anything: Sandbox.run (the spec, the argv
it runs inside, and sandbox_command's command for them), subprocess.Popen
(each process started), and NativeExecutor.run (the executor, the artifact,
the arguments, the @inputs file it finds, and the result). The window the
tests read runs from the adversary stage's stage-start event to the
trial-end event.

1. test_every_accepted_proposal_runs_only_through_the_native_executor_in_its_sandbox:
   in the window every process started is the command of the Sandbox.run
   call just before it (the stage's two builds included), every program run
   is a run of the real NativeExecutor with its default Sandbox and runner,
   reached through one Sandbox.run call on its own artifact and arguments,
   and each accepted proposal runs the reference, then the candidate, once
   each, in <trial>/adversary/proposal<NN>/<role>, on that proposal's input.
2. test_a_refused_proposal_starts_no_process_and_is_counted_as_refused: a
   proposal check_inputs refuses (out-of-range, dtype, missing-input,
   unreadable, too-many-elements, repeated-input, one each) starts no
   process and no Sandbox.run or executor run between the view that
   proposed it and the next, leaves only its staged arrays in its folder,
   and Trial.adversary counts it by reason.
3. test_a_faulty_candidate_diverges_on_an_in_contract_input: large-zero
   diverges on x = 500 and x = -1000 and agrees on 1 and 0; each divergence
   records its input in the binary store, binary_io's statistics, and no
   candidate failure; the programs' own outputs are 2x and 0 there.
4. test_a_control_finds_no_divergence: the reference as its own candidate
   agrees on every proposal, with byte-identical outputs.

The tests are marked `remote` and skip unless the host can run them (Linux,
the sandbox tools on PATH, a reachable user systemd manager, $LASSI_SCRATCH,
$LASSI_RUNS_ROOT, and $LASSI_TOOLCHAINS set, TMPDIR inside $LASSI_SCRATCH,
and the pinned host g++ of toolchains/gcc.pin present). With
LASSI_REQUIRE_SANDBOX=1 such a host fails them instead, so a silent skip
never passes for evidence. Run them from the task's clean commit in the LASSI
distro's clone with the host profile sourced (AGENTS.md, Local Execution), or
on the build host, only with the owner's approval in the working session
(Agent Rule 17), with `uv run tools/rx.py run -- '<command>'`, the command
being

    LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/core/test_adversary_stage_remote.py

Every build and run lies in a temp directory under $LASSI_RUNS_ROOT, removed
afterwards (Agent Rule 7). Expected statistics come from binary_io itself
over arrays written here. No value here is a measurement.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from adversary_support import (
    ADVERSARY,
    AGREED,
    DIVERGED,
    FAKE_COMMIT,
    World,
    codes,
    constant,
    proposal,
    scripted_adversary,
    stored_array,
    values,
    views,
    write_recipe,
)

from lassi.core import runner as runner_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import Completion, Limits, Message, Proposal, RunResult, Sampling
from lassi.core.progress import STAGE_START, TRIAL_END, ProgressEvent
from lassi.core.record import Trial, make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.executors.native import NativeExecutor
from lassi.executors.sandbox import Sandbox, SandboxSpec, sandbox_command
from lassi.harness import LassiArray, read_array, write_array
from lassi.harness.contract import MAX_INPUT_ELEMENTS
from lassi.oracles.binary_io import BinaryIOOracle
from lassi.toolchains import capped_runner
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "adversary" / "stage"
SUITE = PROJECT = "adv-stage-native"
ITEM = "scale"
PROGRAM = f"{ITEM}.cpp"
REFERENCE = FIXTURES / ITEM / "cpp" / PROGRAM
FAULTY = FIXTURES / "candidates" / ITEM / "large-zero.cpp"
LASSI_IO = "assets/harness/c/lassi_io.h"
SPLIT = "unassigned"
DIRECTION = "cpp-cpp"
BACKEND = "fixture-reply"
MODEL_ID = "adv-stage-fixture"
STAGES = ["baseline", "generate", "compile_loop", "run_loop", "oracle", "adversary"]
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk", "env", "sh")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
# The item's held-out input, its seed, its contract's range for x, and the threshold of its tolerance and the recipe.
HELD_OUT = {"name": "x", "dtype": "f32", "shape": [4], "dist": "uniform", "lo": 0.0, "hi": 10.0}
SEED = 11
CONTRACT_LO, CONTRACT_HI = -1000.0, 1000.0
THRESHOLD = 0.001
INPUT_FILE = "@inputs/x.lassiio"
OUTPUT_FILE = "y.lassiio"
# The in-contract values the tests propose, each as a constant x, and large-zero's outcome on each: it agrees only
# where |x| < 100.
SETS = (1.0, 500.0, CONTRACT_LO, 0.0)
FAULT_CODES = [AGREED, DIVERGED, DIVERGED, AGREED]
# The spied events: a Sandbox.run call, a process start, and a NativeExecutor run.
SPIED = ("sandbox", "process", "run")


def host_problem() -> str:
    """Return why this host cannot run the tests, or "" when it can."""
    if not sys.platform.startswith("linux"):
        return f"the sandbox needs Linux, not {sys.platform}; run it in the LASSI distro or through tools/rx.py"
    missing = [tool for tool in TOOLS if shutil.which(tool) is None]
    if missing:
        return f"not on PATH: {', '.join(missing)}"
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if not runtime or not (Path(runtime) / "bus").exists():
        return "no user systemd manager: $XDG_RUNTIME_DIR/bus does not exist"
    for name in ("LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS", "TMPDIR"):
        if not os.environ.get(name):
            return f"{name} is not set"
    scratch, tmpdir = Path(os.environ["LASSI_SCRATCH"]).resolve(), Path(os.environ["TMPDIR"]).resolve()
    if scratch not in tmpdir.parents:
        return "TMPDIR does not lie inside $LASSI_SCRATCH"
    executable = pins_module.read_pin("gcc").get("EXECUTABLE", "")
    if not executable or not Path(executable).is_file():
        return f"the pinned host compiler {executable!r} (toolchains/gcc.pin) is not on this host"
    return ""


PROBLEM = host_problem()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can run native programs"),
]


@pytest.fixture(autouse=True)
def host_ready() -> None:
    """Fail, rather than skip, when LASSI_REQUIRE_SANDBOX=1 asks for a real sandbox and the host cannot run one."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")


@pytest.fixture
def base() -> Iterator[Path]:
    """Create a temp directory under $LASSI_RUNS_ROOT for the manifest, recipe, and run; remove it afterwards."""
    path = Path(tempfile.mkdtemp(prefix="lassi-adv-stage.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        yield path
    finally:
        shutil.rmtree(path)


# ---------------------------------------------------------------------------
# What the spies see


@dataclass(frozen=True)
class SandboxCall:
    """One Sandbox.run call: its spec, the argv it runs inside the sandbox, and sandbox_command's command for them."""

    spec: SandboxSpec
    argv: tuple[str, ...]
    command: tuple[str, ...]


@dataclass(frozen=True)
class ProgramRun:
    """One NativeExecutor.run call: the executor, the artifact, its arguments, the @inputs file found, the result."""

    executor: Any
    artifact: Path
    args: tuple[str, ...]
    inputs: bytes | None
    result: RunResult


@dataclass(frozen=True)
class Sandboxed:
    """A program run and the Sandbox.run call that ran it."""

    call: SandboxCall
    run: ProgramRun


def spy(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """Record each Sandbox.run call, process start, and NativeExecutor run in World.events; each still runs."""
    sandbox_run, native_run, popen = Sandbox.run, NativeExecutor.run, subprocess.Popen

    def recording_sandbox_run(self: Sandbox, spec: SandboxSpec, argv: Sequence[str], limits: Limits) -> Any:
        """Record the call and its command, then run it as Sandbox.run does."""
        command = tuple(sandbox_command(spec, argv, limits))
        world.events.append(("sandbox", SandboxCall(spec=spec, argv=tuple(argv), command=command)))
        return sandbox_run(self, spec, argv, limits)

    def recording_native_run(self: NativeExecutor, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Run as NativeExecutor.run does, then record the run with the @inputs file it found."""
        staged = Path(artifact).parent / INPUT_FILE
        found = staged.read_bytes() if staged.is_file() else None
        result = native_run(self, artifact, inputs, limits)
        world.events.append(("run", ProgramRun(self, Path(artifact), tuple(inputs), found, result)))
        return result

    class RecordingPopen(popen):  # type: ignore[misc, valid-type]
        """Records the argv of each process before it starts."""

        def __init__(self, args: Any, *rest: Any, **keywords: Any) -> None:
            """Record `args`, then start the process as subprocess.Popen does."""
            started = (args,) if isinstance(args, (str, bytes, os.PathLike)) else tuple(args)
            world.events.append(("process", tuple(str(part) for part in started)))
            super().__init__(args, *rest, **keywords)

    monkeypatch.setattr(Sandbox, "run", recording_sandbox_run)
    monkeypatch.setattr(NativeExecutor, "run", recording_native_run)
    monkeypatch.setattr(subprocess, "Popen", RecordingPopen)


def adversary_window(world: World) -> list[tuple[str, Any]]:
    """Return the events after the adversary stage's stage-start and before the trial-end event."""
    starts = [index for index, (kind, value) in enumerate(world.events) if kind == "stage" and value == "adversary"]
    assert len(starts) == 1, "the adversary stage starts once"
    end = next(index for index in range(starts[0], len(world.events)) if world.events[index][0] == "trial-end")
    return world.events[starts[0] + 1 : end]


def sandboxed_runs(events: Sequence[tuple[str, Any]]) -> list[Sandboxed]:
    """Return the program runs among `events`, after asserting that nothing among them ran outside the sandbox.

    Every process started must be the command of the Sandbox.run call just
    before it, one process per call, and every NativeExecutor run must come
    right after the call and the process that ran its own artifact and
    arguments in its own workdir. A compile (a call and its process with no
    program run after them) passes too.
    """
    spied = [(kind, value) for kind, value in events if kind in SPIED]
    found: list[Sandboxed] = []
    index = 0
    while index < len(spied):
        kind, call = spied[index]
        assert kind == "sandbox", f"a {kind} event with no Sandbox.run call before it: {call}"
        assert index + 1 < len(spied) and spied[index + 1][0] == "process", f"Sandbox.run started no process: {call}"
        assert spied[index + 1][1] == call.command, "the process started is not the sandbox's command for the call"
        index += 2
        if index < len(spied) and spied[index][0] == "run":
            run = spied[index][1]
            assert call.argv == (str(run.artifact), *run.args), f"the sandbox ran {call.argv}, not {run.artifact}"
            assert call.spec.workdir == run.artifact.parent.resolve(), "the run's workdir is its artifact's directory"
            found.append(Sandboxed(call=call, run=run))
            index += 1
    return found


# ---------------------------------------------------------------------------
# The suite, the registry, and the run


def write_manifest(base: Path) -> Path:
    """Write the one-item SYNTHETIC manifest (the module docstring) under `base`; return its directory."""
    cpp = {"dir": f"{ITEM}/cpp", "files": [PROGRAM], "support": {"lassi_io.h": {"tracked": LASSI_IO}}}
    item = {
        "split": SPLIT,
        "seed": SEED,
        "inputs": [dict(HELD_OUT)],
        "outputs": ["y"],
        "tolerance": {"metric": "max_abs", "threshold": THRESHOLD},
        "contract": [{"name": "x", "dtype": "f32", "lo": CONTRACT_LO, "hi": CONTRACT_HI}],
        "languages": {"cpp": cpp},
    }
    data = {"suite": SUITE, "repo": f"https://example.invalid/{SUITE}", "commit": FAKE_COMMIT, "items": {ITEM: item}}
    manifests = base / "manifests"
    manifests.mkdir()
    (manifests / f"{SUITE}.yaml").write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    return manifests


def fixture_backend(reply: str) -> type:
    """Return an LLMBackend class that answers every request with `reply`."""

    class FixtureBackend:
        """Answers every request with the candidate's FILE block; no model is called or loaded."""

        name = BACKEND
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Return the reply."""
            return Completion(text=reply, prompt_tokens=0, completion_tokens=0)

    return FixtureBackend


def stage_registry(world: World, candidate: str) -> Registry:
    """Return the run's registry: the real stages, gcc-native, native, and binary_io, and the two fakes."""
    registry = Registry()
    registry.register("LLMBackend", BACKEND, fixture_backend(render_file_blocks({PROGRAM: candidate})))
    registry.register("Agent", ADVERSARY, scripted_adversary(world))
    real = [("Toolchain", "gcc-native"), ("Executor", "native"), ("Oracle", "binary_io")]
    for interface, name in [*real, *(("Stage", stage) for stage in STAGES)]:
        registry.register(interface, name, DEFAULT_REGISTRY.get(interface, name).factory)
    return registry


def recipe_data(budget: int) -> dict[str, Any]:
    """Return the test recipe, its adversary the scripted one with budget.inputs `budget`."""
    return {
        "extends": "base",
        "project": PROJECT,
        "faithful": False,
        "model": {"backend": BACKEND, "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "loop": {"max_corrections": 0},
        "bench": {"suite": SUITE, "split": SPLIT, "items": [ITEM]},
        "directions": [{"source": "cpp", "target": "cpp"}],
        "prompts": "p0-smoke",
        "toolchain": {"cpp": "gcc-native"},
        "stages": list(STAGES),
        "executor": {"kind": "native"},
        "oracle": {"kind": "binary_io", "metric": "max_abs", "threshold": THRESHOLD},
        "adversary": {"kind": ADVERSARY, "budget": {"inputs": budget}},
        "trials": {"n": 1},
        "sandbox": {"wall_s": "baseline_x10", "mem_gb": 2},
    }


def observer(world: World) -> Callable[[ProgressEvent], None]:
    """Return an observer that records each stage-start and trial-end event in World.events."""

    def observe(event: ProgressEvent) -> None:
        """Record a stage-start as ("stage", name) and a trial-end as ("trial-end", None)."""
        if event.kind == STAGE_START:
            world.events.append(("stage", event.stage))
        elif event.kind == TRIAL_END:
            world.events.append(("trial-end", None))

    return observe


@dataclass(frozen=True)
class Outcome:
    """One finished run: its run directory, its one trial, and the adversary stage's directory in the trial."""

    run_dir: Path
    trial: Trial
    adversary_dir: Path

    @property
    def record(self) -> Any:
        """Return Trial.adversary, failing with the standing attempt's state when the stage did not take the trial."""
        found = self.trial.adversary
        if found is None:
            standing = self.trial.attempts[-1] if self.trial.attempts else None
            state = "no attempt" if standing is None else f"{standing.stage_reached}, mean {standing.alignment.mean}"
            pytest.fail(f"the adversary stage did not take the trial (standing attempt: {state})")
        return found


def run_stage(base: Path, monkeypatch: pytest.MonkeyPatch, world: World, candidate: Path, budget: int) -> Outcome:
    """Run the recipe with `candidate` as the model's reply, under the spies; return the run and its one trial."""
    monkeypatch.setattr(runner_module, "BENCH_DIR", write_manifest(base))
    spy(world, monkeypatch)
    options = RunOptions(
        runs_root=base / "runs-root",
        run_id="adv-stage",
        bench_root=FIXTURES,
        registry=stage_registry(world, candidate.read_bytes().decode("ascii")),
        observer=observer(world),
    )
    run_dir = run_recipe(write_recipe(base, recipe_data(budget), "recipe"), options)
    home = trial_dir(run_dir, make_trial_id(PROJECT, MODEL_ID, SUITE, DIRECTION, ITEM, 1))
    return Outcome(run_dir=run_dir, trial=read_trial(home, TextStore(run_dir)), adversary_dir=home / "adversary")


# ---------------------------------------------------------------------------
# Reading the runs


def batches_of(*sets: Proposal) -> list[list[Proposal]]:
    """Return one batch per proposal, then an empty reply, so the last view holds every outcome."""
    return [*([item] for item in sets), []]


def lassi_bytes(array: LassiArray, path: Path) -> bytes:
    """Return the bytes of `array` written as a lassi_io file at the new `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_array(path, array.name, array.dtype, array.shape, array.data)
    return path.read_bytes()


def clean(run: ProgramRun) -> None:
    """Assert a clean run: exit status 0, no hang, and a complete workdir."""
    result = run.result
    found = (result.exit_code, result.hang, result.workdir_incomplete)
    tail = f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    assert found == (0, False, False), f"{run.artifact}: exit, hang, incomplete {found}\n{tail}"


def output_values(run: ProgramRun) -> tuple[float, ...]:
    """Return the elements of the run's one output file, y."""
    assert sorted(run.result.output_files) == [OUTPUT_FILE], f"{run.artifact}: {sorted(run.result.output_files)}"
    array = read_array(Path(run.result.output_files[OUTPUT_FILE]))
    assert array.name == "y" and array.dtype == "f32", array
    return values(array)


def expected_stats(scratch: Path, reference: float, candidate: float) -> list[tuple[Any, ...]]:
    """Return binary_io's (name, pcc, max_abs, max_ulp, passed) at THRESHOLD for constant y arrays written here."""
    sides = {}
    for side, value in (("reference", reference), ("candidate", candidate)):
        path = scratch / side / OUTPUT_FILE
        lassi_bytes(constant(value, name="y"), path)
        sides[side] = {OUTPUT_FILE: path}
    oracle = BinaryIOOracle(metric="max_abs", threshold=THRESHOLD)
    return stat_tuples(oracle.compare(sides["reference"], sides["candidate"]))


def stat_tuples(stats: Sequence[Any]) -> list[tuple[Any, ...]]:
    """Return each OutputStats as (name, pcc, max_abs, max_ulp, passed)."""
    return [(entry.name, entry.pcc, entry.max_abs, entry.max_ulp, entry.passed) for entry in stats]


def refused_proposals() -> list[tuple[str, Proposal]]:
    """Return one proposal per check_inputs refusal reason, each with the reason it must get."""
    return [
        ("out-of-range", proposal(constant(5000.0))),
        ("dtype", proposal(LassiArray("x", "f64", (4,), b"\x00" * 32))),
        ("missing-input", proposal()),
        ("unreadable", proposal(LassiArray("x", "f32", (4,), b"\x00" * 12))),  # three elements under a shape of four
        ("too-many-elements", proposal(LassiArray("x", "f32", (MAX_INPUT_ELEMENTS + 1,), b""))),
        ("repeated-input", proposal(constant(1.0), constant(2.0))),
    ]


# ---------------------------------------------------------------------------
# The tests


def test_every_accepted_proposal_runs_only_through_the_native_executor_in_its_sandbox(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert DEFAULT_REGISTRY.get("Executor", "native").factory is NativeExecutor
    world = World(batches=batches_of(*(proposal(constant(value)) for value in SETS)))
    outcome = run_stage(base, monkeypatch, world, FAULTY, len(SETS) + 1)
    runs = sandboxed_runs(adversary_window(world))
    assert len(runs) == outcome.record.runs == 2 * len(SETS), "two runs per accepted proposal, each one sandboxed"
    hidden = Path(os.environ["LASSI_RUNS_ROOT"]).resolve()
    for index, value in enumerate(SETS):
        folder = (outcome.adversary_dir / f"proposal{index:02d}").resolve()
        expected = lassi_bytes(constant(value), base / "expected" / f"{index:02d}.lassiio")
        pair = runs[2 * index : 2 * index + 2]
        assert [entry.run.artifact.parent.resolve() for entry in pair] == [folder / "reference", folder / "candidate"]
        for entry in pair:
            assert type(entry.run.executor) is NativeExecutor, type(entry.run.executor)
            assert type(entry.run.executor.sandbox) is Sandbox and entry.run.executor.sandbox.runner is capped_runner
            roots = entry.call.spec.hidden_roots
            assert any(root == hidden or root in hidden.parents for root in roots), "the sandbox hides the runs root"
            assert entry.run.args == (INPUT_FILE, OUTPUT_FILE)
            assert entry.run.inputs == expected, f"proposal {index}'s run found another input"
            clean(entry.run)


def test_a_refused_proposal_starts_no_process_and_is_counted_as_refused(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused = refused_proposals()
    batches = batches_of(proposal(constant(1.0)), *(item for _, item in refused), proposal(constant(500.0)))
    asks = len(batches)  # the scripted adversary pops World.batches as it answers
    world = World(batches=batches)
    outcome = run_stage(base, monkeypatch, world, FAULTY, asks)
    seen = views(world)
    assert len(seen) == asks
    assert codes(seen[-1]) == [AGREED, *(reason for reason, _ in refused), DIVERGED]
    record = outcome.record
    assert record.proposals == len(refused) + 2 and record.runs == 4
    assert record.refusals == {reason: 1 for reason, _ in refused}
    positions = [index for index, (kind, _) in enumerate(world.events) if kind == "view"]
    for index in range(len(refused) + 2):
        between = world.events[positions[index] + 1 : positions[index + 1]]
        folder = outcome.adversary_dir / f"proposal{index:02d}"
        if 1 <= index <= len(refused):
            started = [kind for kind, _ in between if kind in SPIED]
            assert started == [], f"refused proposal {index} ({refused[index - 1][0]}) started {started}"
            assert sorted(path.name for path in folder.iterdir()) == ["proposed"], "no workdir for a refused one"
        else:
            roles = [entry.run.artifact.parent.name for entry in sandboxed_runs(between)]
            assert roles == ["reference", "candidate"], f"accepted proposal {index} ran {roles}"


def test_a_faulty_candidate_diverges_on_an_in_contract_input(base: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(batches=batches_of(*(proposal(constant(value)) for value in SETS)))
    outcome = run_stage(base, monkeypatch, world, FAULTY, len(SETS) + 1)
    assert codes(views(world)[-1]) == FAULT_CODES
    record = outcome.record
    assert record.contract_findings == [] and record.refusals == {}
    assert [entry.proposal for entry in record.divergences] == [1, 2]
    for divergence in record.divergences:
        value = SETS[divergence.proposal]
        assert set(divergence.inputs) == {"x"}
        assert stored_array(outcome.run_dir, divergence.inputs["x"], base / "blobs-read") == constant(value)
        expected = expected_stats(base / f"expected{divergence.proposal:02d}", 2 * value, 0.0)
        assert stat_tuples(divergence.stats) == expected and expected[0][4] is False
        assert divergence.candidate_failure is None, "the candidate ran clean; only its output differs"
    runs = [entry.run for entry in sandboxed_runs(adversary_window(world))]
    assert len(runs) == record.runs == 2 * len(SETS)
    for index, value in enumerate(SETS):
        reference, candidate = runs[2 * index : 2 * index + 2]
        clean(reference)
        clean(candidate)
        assert output_values(reference) == (2 * value,) * 4
        assert output_values(candidate) == ((2 * value,) if abs(value) < 100 else (0.0,)) * 4


def test_a_control_finds_no_divergence(base: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(batches=batches_of(*(proposal(constant(value)) for value in SETS)))
    outcome = run_stage(base, monkeypatch, world, REFERENCE, len(SETS) + 1)
    assert codes(views(world)[-1]) == [AGREED] * len(SETS)
    record = outcome.record
    assert record.divergences == [] and record.contract_findings == [] and record.refusals == {}
    assert record.proposals == len(SETS)
    runs = [entry.run for entry in sandboxed_runs(adversary_window(world))]
    assert len(runs) == record.runs == 2 * len(SETS)
    for index in range(len(SETS)):
        reference, candidate = runs[2 * index : 2 * index + 2]
        clean(reference)
        clean(candidate)
        assert {relative: Path(path).read_bytes() for relative, path in reference.result.output_files.items()} == {
            relative: Path(path).read_bytes() for relative, path in candidate.result.output_files.items()
        }, f"proposal {index}: the control's outputs differ from the reference's"
