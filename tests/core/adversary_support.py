"""Shared fakes for the tests of the adversary stage and the runner's adversary checks (task P16.5); not a test module.

Bible: Component Interfaces (Stage; Executor; Oracle; the adversary
bullet), Result Record (Trial; Storage), Project Recipes (the bindings
block's adversary line), Oracles (binary_io rules), Sandbox (attempt run
limits), Adversarial Setups (item 1). Plan: plans/p16-adversarial.md,
P16.5, and its planning decision "Confirmed divergence (P16.5)".

The world these fakes make, every part of it SYNTHETIC:

- A suite manifest, adv-stage, written under the test's directory, with
  three items in the unassigned split, each with an omp source program and
  a cpp target reference, one file each (main.cpp), read from a bench root
  under the test's directory. scale and fragile have held-out inputs (one
  input x, f32, shape [4], drawn in [0, 10) with seed 11), one output y, a
  declared contract that widens x to [-1000, 1000), a tolerance of max_abs
  1.0, and an item-level support file table.txt. plain has run_args and no
  held-out inputs.
- A program is a text that names its behavior on a line
  `// SYNTHETIC-PROGRAM <behavior>`; FakeToolchain builds nothing and writes
  an artifact, prog.bin, that holds the program's text, so a copy of the
  artifact is the same program. scripted_executor runs nothing: it reads the
  artifact's behavior, reads x from the first argument and writes y to the
  second, both relative to the artifact's directory, the workdir, and lists
  as output files only the regular files that did not exist there before
  the run, as the native executor does (lassi/executors/native.py:260-263).
  A program finds its support file in its workdir or fails with exit status
  2, writing nothing. The behaviors (evaluate): double (y = 2x), triple
  (y = 3x), large-zero, large-half, and large-two (y = 2x where |x| < 100,
  else 0, 2x + 0.5, and 2x + 2), crash-negative (exit status 139 and no
  output when any x is negative, else y = 2x), gap-large (a simulator gap,
  exit status 1, and no output when any |x| >= 100, else y = 2x), and
  large-zero-rewrites (large-zero, and after a clean run the program
  rewrites its own artifact in its workdir to double, as a candidate that
  changes its built file would). Every candidate here but triple equals
  double on the held-out inputs, which lie in [0, 10).
- scripted_backend replies with one FILE block holding the candidate
  program; scripted_adversary is an Agent that declares proposes_inputs,
  takes the config key budget, records each AdversaryView it gets, and
  returns the scripted batches in turn (or World.propose's answer).
- The observer the runs get records each stage-start event, so the tests
  tell the adversary stage's runs from the runs before it.

No value in this module is a measurement: every wall time, array, and
digest is a fixture value.
"""

from __future__ import annotations

import dataclasses
import re
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.core import runner as runner_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, Proposal, RunResult, Sampling
from lassi.core.progress import STAGE_START, ProgressEvent
from lassi.core.recipe import RecipeError
from lassi.core.record import Trial, make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.store import BlobStore, TextStore, read_trial, trial_dir
from lassi.harness.lassi_io import LassiArray, read_array, write_array

SUITE = "adv-stage"
PROJECT = "adv-stage"
SPLIT = "unassigned"
SOURCE, TARGET = "omp", "cpp"
DIRECTION = f"{SOURCE}-{TARGET}"
TOOLCHAINS = {SOURCE: "fake-omp", TARGET: "fake-cpp"}
MODEL_ID = "scripted-fixture"
ADVERSARY = "scripted-adversary"
# A commit id for the synthetic manifest; not a commit of this repository or of any source.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
# Every stage a recipe here may list, in recipe order; the adversary stage is the one task P16.5 adds.
STAGES = ["baseline", "generate", "compile_loop", "run_loop", "oracle", "adversary"]
ARTIFACT = "prog.bin"
SUPPORT = "table.txt"
SUPPORT_TEXT = "SYNTHETIC support table\n"
INPUT_FILE = "@inputs/x.lassiio"
OUTPUT_FILE = "y.lassiio"
RUNNER = frozenset({"runs_code", "sandboxed"})
# Every fake run's wall time, in seconds: a fixture value, not a measurement.
WALL_S = 5.0
# The declared contract's range for x, and the held-out input spec it widens.
CONTRACT_LO, CONTRACT_HI = -1000.0, 1000.0
HELD_OUT = {"name": "x", "dtype": "f32", "shape": [4], "dist": "uniform", "lo": 0.0, "hi": 10.0}
ITEM_SEED = 11
TOLERANCE = {"metric": "max_abs", "threshold": 1.0}
# Each item with held-out inputs and the behavior of its cpp target reference; the omp source is double.
REFERENCES = {"scale": "double", "fragile": "crash-negative"}
# The behavior that rewrites its own artifact after a clean run, and the simulator gap gap-large reports.
REWRITES = "large-zero-rewrites"
GAP = "SYNTHETIC unsupported-op"
PLAIN = "plain"
# The outcome codes the adversary stage gives a proposal that ran (task P16.5).
AGREED, DIVERGED, CONTRACT_FINDING = "agreed", "diverged", "contract-finding"
# The roles a program's first line names, so a run can be told apart even when two programs behave alike.
SOURCE_ROLE, REFERENCE_ROLE, CANDIDATE_ROLE = "source program", "target reference", "candidate"

_BEHAVIOR = re.compile(rb"SYNTHETIC-PROGRAM (\S+)")
_ROLE = re.compile(rb"// SYNTHETIC (source program|target reference|candidate)\n")


def program(role: str, behavior: str) -> str:
    """Return the text of a SYNTHETIC program: its role line, then the line naming its behavior."""
    return f"// SYNTHETIC {role}\n// SYNTHETIC-PROGRAM {behavior}\n"


def f32(*values: float, name: str = "x") -> LassiArray:
    """Return a SYNTHETIC f32 array of `values`."""
    return LassiArray(name, "f32", (len(values),), struct.pack(f"<{len(values)}f", *values))


def constant(value: float, name: str = "x") -> LassiArray:
    """Return the f32 array of shape (4,) that holds `value` four times."""
    return f32(value, value, value, value, name=name)


def proposal(*arrays: LassiArray) -> Proposal:
    """Return a Proposal of `arrays`."""
    return Proposal(arrays=tuple(arrays))


def values(array: LassiArray) -> tuple[float, ...]:
    """Return an f32 array's elements."""
    return struct.unpack(f"<{len(array.data) // 4}f", array.data)


def evaluate(behavior: str, xs: Sequence[float]) -> tuple[float, ...] | None:
    """Return y for `xs` under a SYNTHETIC behavior, or None when the program crashes (see the module docstring)."""
    if behavior == "double":
        return tuple(2 * x for x in xs)
    if behavior == "triple":
        return tuple(3 * x for x in xs)
    offsets = {"large-zero": None, REWRITES: None, "large-half": 0.5, "large-two": 2.0}
    if behavior in offsets:
        offset = offsets[behavior]
        return tuple(2 * x if abs(x) < 100 else (0.0 if offset is None else 2 * x + offset) for x in xs)
    if behavior == "crash-negative":
        return None if any(x < 0 for x in xs) else tuple(2 * x for x in xs)
    if behavior == "gap-large":
        return None if any(abs(x) >= 100 for x in xs) else tuple(2 * x for x in xs)
    raise AssertionError(f"unknown SYNTHETIC behavior {behavior!r}")


# ---------------------------------------------------------------------------
# What the fakes see and answer


@dataclass(frozen=True)
class RunSeen:
    """One run a fake executor made: who ran it, what it ran, where, with what, and what it returned."""

    executor: str
    artifact: Path
    workdir: Path
    args: tuple[str, ...]
    limits: Limits
    before: frozenset[str]
    behavior: str
    role: str
    inputs: LassiArray | None
    outputs: frozenset[str]
    exit_code: int


@dataclass
class World:
    """The script the fakes follow and the log they keep.

    `candidate` is the behavior of the model's reply. Each propose() call
    returns the next of `batches`, or [] when none is left, unless `propose`
    is set, whose answer it returns instead. `events` holds ("stage", name)
    for each stage-start, ("run", RunSeen) for each run, and ("view",
    AdversaryView) for each propose() call, in order.
    """

    candidate: str = "double"
    batches: list[list[Proposal]] = field(default_factory=list)
    propose: Callable[[Any], Sequence[Any]] | None = None
    events: list[tuple[str, Any]] = field(default_factory=list)


def _files(workdir: Path) -> dict[str, Path]:
    """Return the regular files under `workdir`, relative POSIX path -> path."""
    return {
        path.relative_to(workdir).as_posix(): path
        for path in sorted(workdir.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


class FakeToolchain:
    """Writes the files, the harness files, and an artifact holding the program's text; compiles nothing."""

    name = "fake"
    capabilities = frozenset({"diagnostics"})

    def build(
        self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
    ) -> BuildResult:
        """Write `files` and `harness` under `workdir` and the artifact ARTIFACT; return it with no diagnostics."""
        for path, text in {**dict(harness or {}), **files}.items():
            (Path(workdir) / path).parent.mkdir(parents=True, exist_ok=True)
            (Path(workdir) / path).write_bytes(text.encode("utf-8"))
        artifact = Path(workdir) / ARTIFACT
        artifact.write_bytes("".join(files[path] for path in sorted(files)).encode("utf-8"))
        return BuildResult(artifact=artifact, diagnostics=[])


def _read_input(workdir: Path, args: Sequence[str]) -> LassiArray | None:
    """Return the array the first argument names, read relative to `workdir`, or None when it cannot be read."""
    try:
        return read_array(workdir / args[0])
    except (IndexError, OSError, ValueError):
        return None


def scripted_executor(world: World, name: str, capabilities: frozenset[str] = RUNNER) -> type:
    """Return an Executor class registered as `name` that runs a SYNTHETIC program as the module docstring says."""

    class ScriptedExecutor:
        """Runs nothing: computes the program's output from its behavior, writes it, and lists only new files."""

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Run the SYNTHETIC program at `artifact` on `inputs` (see the module docstring)."""
            artifact = Path(artifact)
            if not artifact.is_absolute():
                raise ValueError(f"the artifact must be an absolute path, got {str(artifact)!r}")
            if isinstance(inputs, str):
                raise ValueError(f"inputs must be a sequence of strings, got {inputs!r}")
            workdir = artifact.parent
            before = _files(workdir)
            text = artifact.read_bytes()
            found = _BEHAVIOR.search(text)
            role = _ROLE.search(text)
            behavior = found.group(1).decode("ascii") if found else "unknown"
            x = _read_input(workdir, inputs)
            gap = GAP if behavior == "gap-large" and x is not None and evaluate(behavior, values(x)) is None else None
            exit_code = 1 if gap else self._program(workdir, inputs, behavior, x)
            if behavior == REWRITES and exit_code == 0:
                artifact.write_bytes(text.replace(REWRITES.encode("ascii"), b"double"))
            after = _files(workdir)
            outputs = {relative: after[relative] for relative in sorted(after) if relative not in before}
            seen = RunSeen(
                executor=name, artifact=artifact, workdir=workdir, args=tuple(inputs), limits=limits,
                before=frozenset(before), behavior=behavior, role=role.group(1).decode("ascii") if role else "",
                inputs=x, outputs=frozenset(outputs), exit_code=exit_code,
            )
            world.events.append(("run", seen))
            return RunResult(
                exit_code=exit_code, hang=False, stdout="SYNTHETIC\n", stderr="", output_files=outputs, wall_s=WALL_S,
                sim_gap=gap,
            )

        @staticmethod
        def _program(workdir: Path, inputs: Sequence[str], behavior: str, x: LassiArray | None) -> int:
            """Write y for x as the behavior says and return the exit status; write nothing on a failure."""
            if not (workdir / SUPPORT).is_file() or x is None or len(inputs) < 2:
                return 2
            ys = evaluate(behavior, values(x))
            if ys is None:
                return 139
            write_array(workdir / inputs[1], "y", "f32", x.shape, struct.pack(f"<{len(ys)}f", *ys))
            return 0

    ScriptedExecutor.name = name  # type: ignore[attr-defined]
    ScriptedExecutor.capabilities = capabilities  # type: ignore[attr-defined]
    return ScriptedExecutor


class CompileOnlyExecutor:
    """A SYNTHETIC executor that runs no program."""

    name = "compile-only"
    capabilities = frozenset({"compile_only"})

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Fail the test: a compile-only executor is never asked to run a program."""
        raise AssertionError("a compile-only executor ran a program")


def scripted_backend(world: World) -> type:
    """Return an LLMBackend class that answers every request with the candidate program in one FILE block."""

    class ScriptedBackend:
        """Answers every request with World.candidate's program."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Return the candidate program as main.cpp."""
            reply = render_file_blocks({"main.cpp": program(CANDIDATE_ROLE, world.candidate)})
            return Completion(text=reply, prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def scripted_adversary(world: World) -> type:
    """Return an Agent class that declares proposes_inputs and proposes World's batches in turn."""

    class ScriptedAdversary:
        """Records each view it gets and returns the next scripted batch (World.propose when set)."""

        name = ADVERSARY
        capabilities = frozenset({"proposes_inputs"})
        config_keys = frozenset({"budget"})

        def __init__(self, *, budget: Mapping[str, Any] | None = None) -> None:
            """Keep the budget section as given."""
            self.budget = budget

        def propose(self, view: Any) -> list[Proposal]:
            """Record `view` and return the next batch."""
            world.events.append(("view", view))
            if world.propose is not None:
                return list(world.propose(view))
            return world.batches.pop(0) if world.batches else []

    return ScriptedAdversary


class MasksOnlyOracle:
    """A SYNTHETIC Oracle that aligns stdout and compares no output file."""

    name = "masks-only"
    capabilities = frozenset({"masks_stdout"})

    def __init__(self) -> None:
        """Take no setting."""


class SummarizerAgent:
    """A SYNTHETIC Agent for a role under agents; never built."""

    name = "summarizer"
    capabilities: frozenset[str] = frozenset()


class EquivalenceJudge:
    """A SYNTHETIC Judge under judges; never built."""

    name = "equivalence"
    capabilities: frozenset[str] = frozenset()


def make_registry(world: World) -> Registry:
    """Return a test Registry: the fakes above, the real binary_io and df-v0, and every real stage of STAGES.

    A stage the default registry lacks is left out, so a recipe that lists
    it fails at load naming it. The real fuzzer joins when lassi.agents
    registered it, and the real ScoreProfile df-v0 (a recipe's score) when
    lassi.scoring registered it.
    """
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(world))
    for name in ("scripted", "scripted-omp", "scripted-cpp"):
        registry.register("Executor", name, scripted_executor(world, name))
    registry.register("Executor", "unsandboxed", scripted_executor(world, "unsandboxed", frozenset({"runs_code"})))
    registry.register("Executor", "compile-only", CompileOnlyExecutor)
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, FakeToolchain)
    registry.register("Oracle", "binary_io", DEFAULT_REGISTRY.get("Oracle", "binary_io").factory)
    registry.register("Oracle", "masks-only", MasksOnlyOracle)
    registry.register("Agent", ADVERSARY, scripted_adversary(world))
    registry.register("Agent", "summarizer", SummarizerAgent)
    registry.register("Judge", "equivalence", EquivalenceJudge)
    if "fuzzer" in DEFAULT_REGISTRY.names("Agent"):
        registry.register("Agent", "fuzzer", DEFAULT_REGISTRY.get("Agent", "fuzzer").factory)
    if "df-v0" in DEFAULT_REGISTRY.names("ScoreProfile"):
        registry.register("ScoreProfile", "df-v0", DEFAULT_REGISTRY.get("ScoreProfile", "df-v0").factory)
    for name in STAGES:
        if name in DEFAULT_REGISTRY.names("Stage"):
            registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    return registry


def adversary_stage() -> type:
    """Return the registered Stage `adversary`, failing with the task's name when it is not registered yet."""
    assert "adversary" in DEFAULT_REGISTRY.names("Stage"), "no Stage is registered as 'adversary' (task P16.5)"
    return DEFAULT_REGISTRY.get("Stage", "adversary").factory


# ---------------------------------------------------------------------------
# The suite, the recipe, and the runs


def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables a test could inherit and point TMPDIR at a directory under the test's own."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "compile-tmp").mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_path / "compile-tmp"))


def _item(name: str) -> dict[str, Any]:
    """Return the manifest entry of one SYNTHETIC item (see the module docstring)."""
    languages = {language: {"dir": f"src/{name}/{language}", "files": ["main.cpp"]} for language in (SOURCE, TARGET)}
    entry: dict[str, Any] = {"split": SPLIT, "languages": languages}
    if name == PLAIN:
        entry["run_args"] = ["4"]
        return entry
    entry["support"] = {SUPPORT: f"src/{name}/{SUPPORT}"}
    entry["tolerance"] = dict(TOLERANCE)
    entry["inputs"] = [dict(HELD_OUT)]
    entry["outputs"] = ["y"]
    entry["seed"] = ITEM_SEED
    entry["contract"] = [{"name": "x", "dtype": "f32", "lo": CONTRACT_LO, "hi": CONTRACT_HI}]
    return entry


def write_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the SYNTHETIC manifest and its sources under `tmp_path`; point the runner at it; return the bench root."""
    names = [*REFERENCES, PLAIN]
    data = {
        "suite": SUITE,
        "repo": "https://example.invalid/adv-stage",
        "commit": FAKE_COMMIT,
        "items": {name: _item(name) for name in names},
    }
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / f"{SUITE}.yaml").write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    monkeypatch.setattr(runner_module, "BENCH_DIR", manifests)
    root = tmp_path / "bench"
    for name in names:
        texts = {
            f"src/{name}/{SOURCE}/main.cpp": program(SOURCE_ROLE, "double"),
            f"src/{name}/{TARGET}/main.cpp": program(REFERENCE_ROLE, REFERENCES.get(name, "double")),
            f"src/{name}/{SUPPORT}": SUPPORT_TEXT,
        }
        for path, text in texts.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_bytes(text.encode("ascii"))
    return root


def recipe_data(**changes: Any) -> dict[str, Any]:
    """Return the test recipe with each top-level key in `changes` set, or removed when its value is None."""
    data: dict[str, Any] = {
        "extends": "base",
        "project": PROJECT,
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": SPLIT, "items": ["scale"]},
        "directions": [{"source": SOURCE, "target": TARGET}],
        "prompts": "p0-smoke",
        "toolchain": dict(TOOLCHAINS),
        "stages": list(STAGES),
        "executor": {"kind": "scripted"},
        "oracle": {"kind": "binary_io", "metric": "max_abs", "threshold": 0.25},
        "adversary": {"kind": ADVERSARY, "budget": {"inputs": 4}},
        "trials": {"n": 1},
        "sandbox": {"wall_s": 45, "mem_gb": 2},
    }
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def observer(world: World) -> Callable[[ProgressEvent], None]:
    """Return an observer that records each stage-start event in World.events."""

    def observe(event: ProgressEvent) -> None:
        if event.kind == STAGE_START:
            world.events.append(("stage", event.stage))

    return observe


def _options(tmp_path: Path, bench: Path, world: World, run_id: str) -> RunOptions:
    """Return the run options of one test run, with the world's registry and observer."""
    return RunOptions(
        runs_root=tmp_path / "runs-root",
        run_id=run_id,
        bench_root=bench,
        registry=make_registry(world),
        observer=observer(world),
    )


def write_recipe(tmp_path: Path, data: Mapping[str, Any], name: str) -> Path:
    """Write `data` as a recipe file under `tmp_path` and return its path."""
    path = tmp_path / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


@dataclass(frozen=True)
class Outcome:
    """One finished test run: its run directory and its one trial."""

    run_dir: Path
    trial: Trial


def run(
    tmp_path: Path, bench: Path, world: World, data: Mapping[str, Any], *, run_id: str = "test-run", item: str = "scale"
) -> Outcome:
    """Run the recipe `data` over `item` and return the run directory and its one trial."""
    path = write_recipe(tmp_path, data, f"recipe-{run_id}")
    run_dir = run_recipe(path, _options(tmp_path, bench, world, run_id))
    trial_id = make_trial_id(PROJECT, MODEL_ID, SUITE, DIRECTION, item, 1)
    return Outcome(run_dir, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir)))


def refused(tmp_path: Path, bench: Path, world: World, data: Mapping[str, Any]) -> str:
    """Run the recipe `data`, expect a refusal before any directory exists, and return its message."""
    path = write_recipe(tmp_path, data, "recipe-refused")
    with pytest.raises((RunError, RecipeError)) as caught:
        run_recipe(path, _options(tmp_path, bench, world, "refused"))
    assert not (tmp_path / "runs-root").exists(), "a refused run creates no directory"
    assert not any(kind == "run" for kind, _ in world.events), "a refused run runs nothing"
    return str(caught.value)


# ---------------------------------------------------------------------------
# Reading the log


def _adversary_start(world: World) -> int | None:
    """Return the index in World.events of the adversary stage's stage-start, or None when it never started."""
    return next((index for index, event in enumerate(world.events) if event == ("stage", "adversary")), None)


def adversary_runs(world: World) -> list[RunSeen]:
    """Return the runs made after the adversary stage started, in order; [] when it never started."""
    start = _adversary_start(world)
    if start is None:
        return []
    return [seen for kind, seen in world.events[start:] if kind == "run"]


def earlier_runs(world: World) -> list[RunSeen]:
    """Return the runs made before the adversary stage started (the baseline's and run_loop's), in order."""
    start = _adversary_start(world)
    events = world.events if start is None else world.events[:start]
    return [seen for kind, seen in events if kind == "run"]


def views(world: World) -> list[Any]:
    """Return every AdversaryView the adversary was given, in order."""
    return [view for kind, view in world.events if kind == "view"]


def codes(view: Any) -> list[str]:
    """Return the outcome codes a view holds, in order."""
    return [outcome.code for outcome in view.outcomes]


def stored_array(run_dir: Path, digest: str, scratch: Path) -> LassiArray:
    """Return the array of the lassi_io file stored in the run's binary store under `digest`."""
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / f"{digest}.lassiio"
    path.write_bytes(BlobStore(run_dir).get(digest))
    return read_array(path)


def without_wall_time(trial: Trial) -> Trial:
    """Return `trial` with its final wall time cleared, the one value two runs of one recipe may differ in."""
    return dataclasses.replace(trial, final=dataclasses.replace(trial.final, wall_s=None))
