"""Tests for profilers in `lassi run` (task P17.7): the runner binds `profiler`; run_loop profiles every attempt run.

Bible: Component Interfaces (Profiler; supports_power; contract rules),
Result Record (Attempt.profile; provenance), Project Recipes (the profiler
section; Notes), Agent Rules 1, 2, and 6, Design Principle 9;
plans/p17-portable.md, P17.7 (the second acceptance line).

The contract these tests fix:

- lassi.core.capabilities SUPPORTS_POWER is "supports_power", and
  Profiler.stop() returns a lassi.core.record.Profile. `profiler` is no
  longer among the sections the runner refuses as not carried out
  (_NOT_CARRIED_OUT); adversary, refine, agents, and judges still are.
- RunContext has a `profiler`, None by default. lassi.core.stages
  profiled_run(context, artifact, inputs, limits) runs context.executor
  alone and returns (run, Profile()) without a profiler; with one it calls
  start(), then executor.run, and stop() in a finally block, so stop() runs
  when the executor raises and the error propagates.
  checked_profile(profiler, value) refuses, with a ValueError naming the
  profiler, a stop() result that is not a Profile, and a Profile with power
  values from a profiler that does not declare supports_power.
- run_loop profiles every attempt run, whatever its outcome (a failed run,
  a hang, a kernel JIT failure, a clean run): start() right after the
  backend's unload, then the run, then stop(); Attempt.profile is stop()'s
  Profile. An attempt that never ran keeps Profile() (all null), and the
  baseline's reference runs are not profiled. trial.md's Profile table and
  the Parquet attempts table's profile_* columns show the values.
- The runner builds the profiler once per run as factory(**config), its
  section's keys other than kind, before any directory exists and before
  the first trial. A profiler that declares takes_device has its device
  section probed with the others, in binding order (after the executor's),
  and its record (key profiler.device) reaches provenance.json, every
  trial's provenance, and trial.md's Device records table.
- Refused before any directory exists and before the profiler is built: a
  profiler with nothing to profile (no listed stage runs attempts, or no
  target language's executor runs programs); a profiler whose profiled
  executor declares simulator (Agent Rule 2); a profiler that declares
  supports_power without takes_device; a power profiler beside a profiled
  executor (the single one, or with executors per language each target
  language's that runs programs) whose device is not the profiler's, the
  same kind and indices as written, naming profiler.device and that
  executor's key; and a profiler device the host lacks (its probe), naming
  profiler.device. Refused when built, still before any directory: a
  setting the profiler refuses (ValueError), naming the profiler and the
  setting; and missing telemetry (TelemetryUnavailable, a DeviceUnavailable),
  naming profiler.device and the run's lack of telemetry.
- Recipes without a profiler keep their resolved recipes and hashes (the
  loader adds no profiler; tests/core/test_single_executor_golden.py and
  tests/core/test_recipe.py pin the hashes) and record null profiles.

Every component is a fake in a test Registry beside the real stages
(baseline, generate, compile_loop, run_loop); the fake executors start no
process and return SYNTHETIC RunResults; the fake toolchains compile
nothing; the fake profilers return SYNTHETIC Profiles; the registered
rocm_smi profiler reads a constant SYNTHETIC source or a SYNTHETIC host
root; the probes answer with SYNTHETIC facts. No GPU, driver library, or SMI
tool is touched, and no value in this module is a measurement.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

from lassi.bench import Direction, load_suite
from lassi.core import capabilities as capabilities_module
from lassi.core import interfaces as interfaces_module
from lassi.core import runner as runner_module
from lassi.core import stages as stages_module
from lassi.core.devices import DeviceUnavailable, HostFacts
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.parquet import read_run_parquet
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.record import Diagnostic, Profile, Trial, make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import RunContext
from lassi.core.store import TextStore, read_trial, trial_dir

REPO = Path(__file__).resolve().parents[2]
PROJECTS = REPO / "projects"
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
MODEL_ID = "scripted-fixture"
OMP_TO_CUDA = Direction("omp", "cuda")
CUDA_TO_OMP = Direction("cuda", "omp")
TOOLCHAINS = {"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"}
RUN_STAGES = ["baseline", "generate", "compile_loop", "run_loop"]
COMPILE_STAGES = ["generate", "compile_loop"]
NOT_CARRIED_OUT = ("adversary", "refine", "agents", "judges")
SUPPORTS_POWER = "supports_power"
TAKES_DEVICE = "takes_device"
RUNNER = frozenset({"runs_code", "sandboxed"})
ROCM_0 = {"kind": "rocm", "indices": [0]}
ROCM_1 = {"kind": "rocm", "indices": [1]}
ROCM_01 = {"kind": "rocm", "indices": [0, 1]}
CUDA_0 = {"kind": "cuda", "indices": [0]}
SAMPLER_THREAD = "lassi-power-sampler"

# SYNTHETIC bench sources and replies; the fake toolchain refuses any file holding an `#error` line.
SOURCES = {
    "omp": '#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  std::printf("done\\n");\n}\n',
    "cuda": "#include <cstdio>\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n",
}
TARGET_FILE = {"cuda": "main.cu", "omp": "main.cpp"}
GOOD = {language: render_file_blocks({name: "int main() { return 0; }\n"}) for language, name in TARGET_FILE.items()}
BAD = {language: render_file_blocks({name: "#error SYNTHETIC\n"}) for language, name in TARGET_FILE.items()}
# Fixture wall times, not measurements.
REFERENCE_WALL_S = 1.25
ATTEMPT_WALL_S = 0.5
EXECUTOR_FAILURE = "SYNTHETIC: the executor failed"
# Not carried out yet: the runner's blanket refusal, which a profiler refusal must no longer be.
BLANKET = "does not carry out"
NO_TELEMETRY = "SYNTHETIC: no power telemetry on this host"
REFUSED_SETTING = "SYNTHETIC: profiler.window_s must be a number of seconds above 0"
ROCM_DRIVER = "SYNTHETIC-amdgpu-6.8.5"
CUDA_DRIVER = "SYNTHETIC-nvidia-999.88"

# The attempt runs that are not clean: each is profiled like a clean one.
OUTCOMES = {
    "failed": RunResult(exit_code=1, hang=False, stdout="", stderr="SYNTHETIC failure\n", wall_s=ATTEMPT_WALL_S),
    "hung": RunResult(exit_code=None, hang=True, stdout="", stderr="", wall_s=ATTEMPT_WALL_S),
    "jit": RunResult(
        exit_code=1,
        hang=False,
        stdout="",
        stderr="",
        wall_s=ATTEMPT_WALL_S,
        diagnostics=[Diagnostic(stage="jit", severity="error", code="SYNTHETIC-jit", message="SYNTHETIC JIT error")],
    ),
}


def timing_profile(number: int) -> Profile:
    """Return the SYNTHETIC Profile the fake timing profiler's `number`-th stop() gives."""
    return Profile(runtime_s=0.25 * number)


def power_profile(number: int) -> Profile:
    """Return the SYNTHETIC Profile the fake power profiler's `number`-th stop() gives."""
    return Profile(runtime_s=2.0 * number, avg_power_w=150.0, energy_j=300.0 * number)


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate and compile variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


def profilers(name: str = "") -> ModuleType:
    """Import lassi.profilers or lassi.profilers.<name>, failing the test clearly while it does not exist."""
    full = "lassi.profilers" + (f".{name}" if name else "")
    try:
        return importlib.import_module(full)
    except ModuleNotFoundError as error:
        if error.name is not None and (full == error.name or full.startswith(f"{error.name}.")):
            pytest.fail(f"{full} does not exist yet (task P17.7): {error}")
        raise


def profiled(module: str, name: str) -> Any:
    """Return lassi.profilers.<module>.<name>, failing the test clearly while task P17.7 has not added it."""
    found = profilers(module)
    if not hasattr(found, name):
        pytest.fail(f"{found.__name__} has no {name} yet (task P17.7)")
    return getattr(found, name)


def stages_attribute(name: str) -> Any:
    """Return lassi.core.stages.<name>, failing the test clearly while task P17.7 has not added it."""
    if not hasattr(stages_module, name):
        pytest.fail(f"lassi.core.stages has no {name} yet (task P17.7)")
    return getattr(stages_module, name)


# ---------------------------------------------------------------------------
# Fake components, all writing to one Log


@dataclass
class Log:
    """What the fakes saw, in order, and how they answer.

    `replies` are the backend's replies; `outcomes` the RunResults of the
    attempt runs in order (a clean run once they run out); `stops` counts
    the fake profilers' stop() calls; `sections` holds each fake profiler's
    config as the runner gave it.
    """

    replies: list[str] = field(default_factory=list)
    outcomes: list[RunResult] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    sections: dict[str, Any] = field(default_factory=dict)
    stops: int = 0
    raise_on_attempt: bool = False
    attempt_sleep_s: float = 0.0


class ExecutorFailure(Exception):
    """The SYNTHETIC error a fake executor raises during an attempt run."""


def clean_run() -> RunResult:
    """Return a clean SYNTHETIC attempt run."""
    return RunResult(exit_code=0, hang=False, stdout="SYNTHETIC stdout\n", stderr="", wall_s=ATTEMPT_WALL_S)


def scripted_backend(log: Log) -> type:
    """Return an LLMBackend class "scripted" that unloads before each run and answers from log.replies."""

    class ScriptedBackend:
        """Records each request and unload, and answers with the next scripted reply."""

        name = "scripted"
        capabilities = frozenset({"chat", "unload_before_run"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id, as every backend does."""
            self.model_id = model_id

        def unload(self) -> None:
            """Record the unload."""
            log.events.append("unload")

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return the next scripted reply."""
            log.events.append("request")
            assert log.replies, "the backend was asked for more replies than the script holds"
            return Completion(text=log.replies.pop(0), prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def fake_toolchain(registered_as: str, log: Log) -> type:
    """Return a Toolchain class without PIN that writes the files and a PLACEHOLDER artifact; it compiles nothing."""

    class FakeToolchain:
        """Writes every file under the workdir and reports a build, or no artifact for an `#error` line."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Record the build, write every file, and return the artifact, or none for an `#error` line."""
            workdir = Path(workdir)
            for path, text in [*files.items(), *(harness or {}).items()]:
                target = workdir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            log.events.append(f"compile {registered_as}")
            if any("#error" in text for text in files.values()):
                return BuildResult(artifact=None, diagnostics=[])
            artifact = workdir / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


def fake_executor(label: str, declared: frozenset[str], log: Log) -> type:
    """Return an Executor class `label` that runs nothing: references run clean, attempts answer from the Log."""

    class FakeExecutor:
        """Records each run as a reference or an attempt run and returns a SYNTHETIC RunResult."""

        name = label
        capabilities = declared
        config_keys = frozenset({"host"})

        def __init__(self, *, device: Mapping[str, Any] | None = None, host: str | None = None) -> None:
            """Record the build; keep the device section a takes_device executor is given."""
            log.events.append(f"executor {label}")
            self.section = device

        def device(self) -> str:
            """Return this fake's SYNTHETIC device name."""
            return f"SYNTHETIC {label} device"

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Return a clean reference run, or the next scripted attempt run (or raise when the Log says so)."""
            if Path(artifact).parent.parent.name.startswith("baseline-"):
                log.events.append("run reference")
                return RunResult(exit_code=0, hang=False, stdout="SYNTHETIC\n", stderr="", wall_s=REFERENCE_WALL_S)
            log.events.append("run attempt")
            if log.raise_on_attempt:
                raise ExecutorFailure(EXECUTOR_FAILURE)
            if log.attempt_sleep_s:
                time.sleep(log.attempt_sleep_s)
            return log.outcomes.pop(0) if log.outcomes else clean_run()

    return FakeExecutor


class CompileOnlyExecutor:
    """A compile-only Executor registered as "none"; being asked to run anything fails the test."""

    name = "none"
    capabilities = frozenset({"compile_only"})

    def device(self) -> str:
        """Return the compile-only device name runs record."""
        return "none (compile only)"

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Fail the test: a compile-only executor is never asked to run a program."""
        raise AssertionError(f"a compile-only executor was asked to run {artifact}")


def fake_profiler(
    label: str, declared: frozenset[str], log: Log, result: Callable[[int], Any], keys: frozenset[str] = frozenset()
) -> type:
    """Return a Profiler class `label` whose n-th stop() returns result(n); it records its build, starts, and stops."""

    class FakeProfiler:
        """Records start() and stop() and answers each stop() with a SYNTHETIC result."""

        name = label
        capabilities = declared
        config_keys = keys

        def __init__(self, **config: Any) -> None:
            """Record the build and the config the runner gave."""
            log.events.append(f"profiler {label}")
            log.sections[label] = config

        def start(self) -> None:
            """Record the start."""
            log.events.append("start")

        def stop(self) -> Any:
            """Record the stop and return the next SYNTHETIC result."""
            log.events.append("stop")
            log.stops += 1
            return result(log.stops)

    return FakeProfiler


def raising_profiler(label: str, declared: frozenset[str], log: Log, error: Callable[[], BaseException]) -> type:
    """Return a Profiler class `label` whose construction records itself and raises error()."""

    class RaisingProfiler:
        """Refuses to be built, as a profiler refuses a setting or finds no telemetry."""

        name = label
        capabilities = declared
        config_keys = frozenset({"interval_ms"})

        def __init__(self, **config: Any) -> None:
            """Record the build attempt and raise."""
            log.events.append(f"profiler {label}")
            raise error()

    return RaisingProfiler


def no_telemetry() -> BaseException:
    """Return the TelemetryUnavailable a power profiler raises when the host lacks its telemetry."""
    return profiled("power", "TelemetryUnavailable")(NO_TELEMETRY)


def with_keywords(cls: type, **fixed: Any) -> type:
    """Return a subclass of a registered profiler class that the runner builds as factory(**config) plus `fixed`."""

    class Fixed(cls):  # type: ignore[misc, valid-type]
        """The registered profiler with keywords the recipe cannot hold (a source or a host root) added."""

        def __init__(self, **config: Any) -> None:
            """Build the registered profiler with the recipe's config and the fixed keywords."""
            super().__init__(**fixed, **config)

    return Fixed


def make_registry(log: Log) -> Registry:
    """Return a test Registry: the scripted backend, fake toolchains, executors, profilers, and the real stages."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(log))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, fake_toolchain(name, log))
    registry.register("Executor", "devexec", fake_executor("devexec", RUNNER | {TAKES_DEVICE}, log))
    registry.register("Executor", "plainexec", fake_executor("plainexec", RUNNER, log))
    registry.register("Executor", "simexec", fake_executor("simexec", RUNNER | {"simulator"}, log))
    registry.register("Executor", "none", CompileOnlyExecutor)
    power = frozenset({SUPPORTS_POWER, TAKES_DEVICE})
    interval = frozenset({"interval_ms"})
    registry.register("Profiler", "faketime", fake_profiler("faketime", frozenset(), log, timing_profile))
    registry.register("Profiler", "fakepower", fake_profiler("fakepower", power, log, power_profile, interval))
    nodevice = fake_profiler("nodevpower", frozenset({SUPPORTS_POWER}), log, power_profile, interval)
    registry.register("Profiler", "nodevpower", nodevice)
    registry.register("Profiler", "dictstop", fake_profiler("dictstop", frozenset(), log, lambda n: {"runtime_s": 1.0}))
    registry.register("Profiler", "powerless", fake_profiler("powerless", frozenset(), log, power_profile))
    registry.register("Profiler", "notelemetry", raising_profiler("notelemetry", power, log, no_telemetry))
    refusing = raising_profiler("refusing", frozenset(), log, lambda: ValueError(REFUSED_SETTING))
    registry.register("Profiler", "refusing", refusing)
    for name in RUN_STAGES:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    return registry


@dataclass
class FakeProbe:
    """A probe of one kind with SYNTHETIC facts (two devices), or a refusal."""

    kind: str
    refuse: str | None = None

    def probe(self) -> HostFacts:
        """Raise DeviceUnavailable with `refuse`, or return the SYNTHETIC facts."""
        if self.refuse is not None:
            raise DeviceUnavailable(self.refuse)
        driver = {"rocm": ROCM_DRIVER, "cuda": CUDA_DRIVER}.get(self.kind)
        return HostFacts(count=2, name=None, memory_bytes=None, driver=driver, runtime=None)


def fake_probes(**refusals: str) -> dict[str, FakeProbe]:
    """Return SYNTHETIC cpu, rocm, and cuda probes; `refusals` maps a kind to its refusal."""
    return {kind: FakeProbe(kind, refusals.get(kind)) for kind in ("cpu", "rocm", "cuda")}


# ---------------------------------------------------------------------------
# Recipes, bench sources, and runs


def recipe_data(
    directions: Sequence[Direction],
    executor: Mapping[str, Any],
    *,
    profiler: Mapping[str, Any] | None = None,
    stages: Sequence[str] = RUN_STAGES,
    **changes: Any,
) -> dict[str, Any]:
    """Return a template-set recipe for the layout item over `directions`, with `executor` and `profiler` as given."""
    data: dict[str, Any] = {
        "extends": "base",
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": direction.source, "target": direction.target} for direction in directions],
        "prompts": "p0-smoke",
        "toolchain": dict(TOOLCHAINS),
        "stages": list(stages),
        "executor": copy.deepcopy(dict(executor)),
        "trials": {"n": 1},
    }
    if profiler is not None:
        data["profiler"] = copy.deepcopy(dict(profiler))
    data.update(copy.deepcopy(changes))
    return data


def write_recipe(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe <directory>/<name>.yaml (ASCII, LF) and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def write_bench(root: Path) -> Path:
    """Write the layout item's SYNTHETIC sources per language where the suite manifest lays them out; return `root`."""
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, text in SOURCES.items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    for relative in spec.support.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"// SYNTHETIC support file\n")
    return root


def options(tmp_path: Path, registry: Registry, root: str = "runs-root", **refusals: str) -> RunOptions:
    """Return the run options: a runs root under tmp_path, a fixed run id, the SYNTHETIC bench and probes."""
    bench = tmp_path / "bench"
    if not bench.exists():
        write_bench(bench)
    return RunOptions(
        runs_root=tmp_path / root, run_id="profile-run", bench_root=bench, registry=registry,
        probes=fake_probes(**refusals),
    )


@dataclass
class Outcome:
    """One finished run: its directory, the recipe name, and provenance.json."""

    run_dir: Path
    name: str
    provenance: dict[str, Any]

    def trial_id(self, direction: Direction) -> str:
        """Return the id of the one trial of `direction`."""
        return make_trial_id(self.name, MODEL_ID, SUITE, f"{direction.source}-{direction.target}", ITEM, 1)

    def trial(self, direction: Direction) -> Trial:
        """Return the one trial of `direction`, read back from the run tree."""
        return read_trial(trial_dir(self.run_dir, self.trial_id(direction)), TextStore(self.run_dir))

    def trial_md(self, direction: Direction) -> str:
        """Return the trial.md of the one trial of `direction`."""
        return (trial_dir(self.run_dir, self.trial_id(direction)) / "trial.md").read_bytes().decode("ascii")


def run_all(
    tmp_path: Path, name: str, data: Mapping[str, Any], registry: Registry, root: str = "runs-root"
) -> Outcome:
    """Run the recipe with `registry`; return the outcome."""
    run_dir = run_recipe(write_recipe(tmp_path, name, data), options(tmp_path, registry, root))
    provenance = json.loads((run_dir / "provenance.json").read_bytes().decode("ascii"))
    return Outcome(run_dir, name, provenance)


def refused(
    tmp_path: Path, name: str, data: Mapping[str, Any], log: Log, registry: Registry | None = None, **refusals: str
) -> str:
    """Run a recipe that must be refused before any directory exists and before anything runs; return the message.

    The test directory's path is replaced by <tmp> in the message returned,
    so a word in a test's own name never stands in for the refusal's reason.
    """
    with pytest.raises((RunError, RecipeError)) as caught:
        run_recipe(write_recipe(tmp_path, name, data), options(tmp_path, registry or make_registry(log), **refusals))
    assert not (tmp_path / "runs-root").exists(), "a refusal comes before any directory is created"
    ran = [event for event in log.events if event.startswith("run ") or event in ("start", "stop", "request")]
    assert ran == [], f"a refusal comes before anything runs or is asked: {ran}"
    return str(caught.value).replace(str(tmp_path), "<tmp>")


def control_run(tmp_path: Path, name: str, data: Mapping[str, Any], log: Log) -> None:
    """Run `data` without its profiler under another runs root, showing the recipe is fine but for the profiler."""
    plain = {key: value for key, value in data.items() if key != "profiler"}
    run_all(tmp_path, f"{name}-control", plain, make_registry(log), root="control-root")


def attempt_rows(outcome: Outcome, direction: Direction) -> list[dict[str, Any]]:
    """Return the Parquet attempts rows of the trial of `direction`, in attempt order."""
    rows = read_run_parquet(outcome.run_dir / "parquet")["attempts"]
    found = [row for row in rows if row["trial_id"] == outcome.trial_id(direction)]
    return sorted(found, key=lambda row: row["index"])


def sampler_threads() -> list[threading.Thread]:
    """Return the live threads named as a power profiler's sampler."""
    return [thread for thread in threading.enumerate() if thread.name == SAMPLER_THREAD and thread.is_alive()]


# ---------------------------------------------------------------------------
# The seams: the capability, the interface, RunContext, profiled_run, and checked_profile


def test_supports_power_is_a_declared_capability() -> None:
    assert getattr(capabilities_module, "SUPPORTS_POWER", None) == SUPPORTS_POWER


def test_profiler_stop_returns_a_profile() -> None:
    annotation = interfaces_module.Profiler.stop.__annotations__.get("return")
    assert annotation in ("Profile", Profile), f"Profiler.stop() is annotated {annotation!r}, not Profile"


def test_the_runner_carries_out_profiler_and_still_refuses_the_rest() -> None:
    assert runner_module._NOT_CARRIED_OUT == NOT_CARRIED_OUT


def test_the_runner_still_refuses_a_section_it_does_not_carry_out(tmp_path: Path) -> None:
    log = Log()
    data = recipe_data([OMP_TO_CUDA], {"kind": "plainexec"}, refine={"max_iters": 3})
    message = refused(tmp_path, "refine", data, log)
    assert "refine" in message and "does not carry out" in message, message


def test_run_context_takes_a_profiler_none_by_default() -> None:
    specs = {spec.name: spec for spec in dataclasses.fields(RunContext)}
    assert "profiler" in specs, "RunContext has no profiler field (task P17.7)"
    assert specs["profiler"].default is None


def stand_in(name: str, declared: frozenset[str], events: list[str], result: Any) -> SimpleNamespace:
    """Return a stand-in profiler object that records start() and stop() in `events` and stops with `result`."""
    return SimpleNamespace(
        name=name,
        capabilities=declared,
        start=lambda: events.append("start"),
        stop=lambda: events.append("stop") or result,
    )


LIMITS = Limits(wall_s=30.0, memory_mb=1024, cpus=1)
ARGS = ["SYNTHETIC-argument"]


def test_profiled_run_without_a_profiler_runs_the_executor_alone() -> None:
    profiled_run = stages_attribute("profiled_run")
    seen: list[Any] = []
    run = clean_run()
    executor = SimpleNamespace(run=lambda artifact, inputs, limits: seen.append((artifact, inputs, limits)) or run)
    found, profile = profiled_run(SimpleNamespace(profiler=None, executor=executor), Path("main"), ARGS, LIMITS)
    assert found is run and seen == [(Path("main"), ARGS, LIMITS)]
    assert profile == Profile(), "without a profiler nothing is measured"


def test_profiled_run_brackets_the_run_with_start_and_stop() -> None:
    profiled_run = stages_attribute("profiled_run")
    events: list[str] = []
    run = clean_run()
    executor = SimpleNamespace(run=lambda artifact, inputs, limits: events.append("run") or run)
    profiler = stand_in("faketime", frozenset(), events, timing_profile(1))
    found, profile = profiled_run(SimpleNamespace(profiler=profiler, executor=executor), Path("main"), ARGS, LIMITS)
    assert (found, profile) == (run, timing_profile(1))
    assert events == ["start", "run", "stop"]


def test_profiled_run_stops_when_the_executor_raises() -> None:
    profiled_run = stages_attribute("profiled_run")
    events: list[str] = []

    def broken(artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        events.append("run")
        raise RuntimeError(EXECUTOR_FAILURE)

    profiler = stand_in("faketime", frozenset(), events, timing_profile(1))
    context = SimpleNamespace(profiler=profiler, executor=SimpleNamespace(run=broken))
    with pytest.raises(RuntimeError, match=EXECUTOR_FAILURE):
        profiled_run(context, Path("main"), ARGS, LIMITS)
    assert events == ["start", "run", "stop"], "stop() runs in a finally block, so no sampling outlives the run"


def test_checked_profile_keeps_a_profile_inside_the_contract() -> None:
    checked = stages_attribute("checked_profile")
    timing = SimpleNamespace(name="faketime", capabilities=frozenset())
    meter = SimpleNamespace(name="fakepower", capabilities=frozenset({SUPPORTS_POWER, TAKES_DEVICE}))
    assert checked(timing, Profile(runtime_s=1.5)) == Profile(runtime_s=1.5)
    assert checked(timing, Profile()) == Profile()
    assert checked(meter, power_profile(1)) == power_profile(1)
    assert checked(meter, Profile(runtime_s=1.0)) == Profile(runtime_s=1.0), "a power profiler may leave power null"


@pytest.mark.parametrize("value", [{"runtime_s": 1.0}, None, 1.0], ids=["mapping", "none", "float"])
def test_checked_profile_refuses_a_result_that_is_not_a_profile(value: Any) -> None:
    timing = SimpleNamespace(name="faketime", capabilities=frozenset())
    with pytest.raises(ValueError, match="faketime"):
        stages_attribute("checked_profile")(timing, value)


@pytest.mark.parametrize(
    "profile",
    [Profile(runtime_s=1.0, avg_power_w=5.0), Profile(runtime_s=1.0, energy_j=5.0)],
    ids=["avg-power", "energy"],
)
def test_checked_profile_refuses_power_from_a_profiler_without_supports_power(profile: Profile) -> None:
    timing = SimpleNamespace(name="faketime", capabilities=frozenset())
    with pytest.raises(ValueError, match=SUPPORTS_POWER) as caught:
        stages_attribute("checked_profile")(timing, profile)
    assert "faketime" in str(caught.value)


# ---------------------------------------------------------------------------
# Runs: every attempt run is profiled, and nothing else


def test_a_recipe_without_a_profiler_keeps_its_hash_and_null_profiles(tmp_path: Path) -> None:
    log = Log(replies=[GOOD["cuda"]])
    data = recipe_data([OMP_TO_CUDA], {"kind": "plainexec"})
    outcome = run_all(tmp_path, "no-profiler", data, make_registry(log))
    loaded = load_recipe(tmp_path / "no-profiler.yaml", roots=(tmp_path, PROJECTS), registry=make_registry(Log()))
    assert "profiler" not in loaded.data, "the loader adds no profiler"
    resolved = yaml.safe_load((outcome.run_dir / "recipe.resolved.yaml").read_bytes().decode("ascii"))
    assert "profiler" not in resolved
    trial = outcome.trial(OMP_TO_CUDA)
    assert trial.recipe_hash == loaded.recipe_hash
    assert [attempt.profile for attempt in trial.attempts] == [Profile()]
    assert "start" not in log.events and "stop" not in log.events


@pytest.mark.parametrize("kind", sorted(OUTCOMES))
def test_timing_fills_the_profile_of_every_attempt_run(tmp_path: Path, kind: str) -> None:
    log = Log(replies=[BAD["cuda"], GOOD["cuda"], GOOD["cuda"]], outcomes=[OUTCOMES[kind]])
    data = recipe_data([OMP_TO_CUDA], {"kind": "plainexec"}, profiler={"kind": "faketime"})
    trial = run_all(tmp_path, f"timing-{kind}", data, make_registry(log)).trial(OMP_TO_CUDA)
    assert len(trial.attempts) == 3 and trial.attempts[-1].stage_reached == "S5", "the third attempt runs clean"
    assert trial.attempts[0].run.exit_code is None, "the first attempt never built, so it never ran"
    assert [attempt.profile for attempt in trial.attempts] == [Profile(), timing_profile(1), timing_profile(2)], (
        f"the {kind} run and the clean run are profiled alike; the attempt that never ran keeps a null profile"
    )
    assert log.events.count("run attempt") == log.events.count("start") == log.events.count("stop") == 2


def test_start_and_stop_bracket_each_attempt_run_after_the_unload(tmp_path: Path) -> None:
    log = Log(replies=[GOOD["cuda"], GOOD["cuda"]], outcomes=[OUTCOMES["failed"]])
    data = recipe_data([OMP_TO_CUDA], {"kind": "plainexec"}, profiler={"kind": "faketime"})
    run_all(tmp_path, "bracket", data, make_registry(log))
    events = log.events
    attempts = [index for index, event in enumerate(events) if event == "run attempt"]
    assert len(attempts) == 2, events
    for index in attempts:
        assert events[index - 2 : index + 2] == ["unload", "start", "run attempt", "stop"], events
    references = [index for index, event in enumerate(events) if event == "run reference"]
    assert len(references) == 2, "both references ran under baseline_both"
    for index in references:
        assert events[index - 1] != "start" and events[index + 1] != "stop", "a reference run is not profiled"
    assert events.count("profiler faketime") == 1, "the profiler is built once per run"
    assert events.index("profiler faketime") < events.index("unload"), "it is built before the first trial"


def test_stop_runs_when_the_executor_raises(tmp_path: Path) -> None:
    log = Log(replies=[GOOD["cuda"]], raise_on_attempt=True)
    data = recipe_data([OMP_TO_CUDA], {"kind": "plainexec"}, profiler={"kind": "faketime"})
    with pytest.raises(ExecutorFailure, match=EXECUTOR_FAILURE):
        run_recipe(write_recipe(tmp_path, "raises", data), options(tmp_path, make_registry(log)))
    assert log.events[-3:] == ["start", "run attempt", "stop"], log.events
    manifest = tmp_path / "runs-root" / "runs" / "profile-run" / "provenance.json"
    assert json.loads(manifest.read_bytes().decode("ascii"))["status"] == "failed"


def test_the_registered_timing_profiler_records_each_attempt_window(tmp_path: Path) -> None:
    profilers()
    log = Log(replies=[GOOD["cuda"]], attempt_sleep_s=0.05)
    registry = make_registry(log)
    registry.register("Profiler", "timing", DEFAULT_REGISTRY.get("Profiler", "timing").factory)
    data = recipe_data([OMP_TO_CUDA], {"kind": "plainexec"}, profiler={"kind": "timing"})
    (attempt,) = run_all(tmp_path, "timing", data, registry).trial(OMP_TO_CUDA).attempts
    profile = attempt.profile
    assert isinstance(profile.runtime_s, float) and profile.runtime_s >= 0.02, "the window covers the 50 ms run"
    assert (profile.avg_power_w, profile.energy_j) == (None, None)


def test_a_power_profiler_on_the_executor_gpu_fills_power(tmp_path: Path) -> None:
    log = Log(replies=[GOOD["cuda"]])
    section = {"kind": "fakepower", "device": ROCM_0, "interval_ms": 10}
    data = recipe_data([OMP_TO_CUDA], {"kind": "devexec", "device": ROCM_0}, profiler=section)
    outcome = run_all(tmp_path, "power", data, make_registry(log))
    assert log.sections["fakepower"] == {"device": ROCM_0, "interval_ms": 10}, "built as factory(**config)"
    assert [attempt.profile for attempt in outcome.trial(OMP_TO_CUDA).attempts] == [power_profile(1)]
    page = outcome.trial_md(OMP_TO_CUDA)
    for row in ("| runtime_s | 2.0 |", "| avg_power_w | 150.0 |", "| energy_j | 300.0 |"):
        assert row in page, f"trial.md's Profile table lacks {row!r}"
    (row,) = attempt_rows(outcome, OMP_TO_CUDA)
    assert (row["profile_runtime_s"], row["profile_avg_power_w"], row["profile_energy_j"]) == (2.0, 150.0, 300.0)


class ConstantSource:
    """A PowerSource that always reads `watts` and counts its reads."""

    def __init__(self, watts: float) -> None:
        """Keep the reading."""
        self.watts = watts
        self.reads = 0

    def read_w(self) -> float:
        """Return the SYNTHETIC reading."""
        self.reads += 1
        return self.watts


def test_the_registered_rocm_smi_profiler_fills_power_from_its_source(tmp_path: Path) -> None:
    source = ConstantSource(100.0)
    cls = profiled("rocm_smi", "RocmSmiProfiler")
    log = Log(replies=[GOOD["cuda"]], attempt_sleep_s=0.05)
    registry = make_registry(log)
    registry.register("Profiler", "rocm_smi", with_keywords(cls, source=source))
    section = {"kind": "rocm_smi", "device": ROCM_0, "interval_ms": 10}
    data = recipe_data([OMP_TO_CUDA], {"kind": "devexec", "device": ROCM_0}, profiler=section)
    (attempt,) = run_all(tmp_path, "rocm-smi", data, registry).trial(OMP_TO_CUDA).attempts
    profile = attempt.profile
    assert profile.runtime_s is not None and profile.runtime_s > 0
    assert profile.avg_power_w == pytest.approx(100.0)
    assert profile.energy_j == pytest.approx(100.0 * profile.runtime_s)
    assert source.reads >= 3, "the build read, the start sample, and the stop sample"
    assert sampler_threads() == [], "no sampler thread outlives the attempt"


def test_the_profiler_device_is_probed_and_recorded_in_provenance(tmp_path: Path) -> None:
    log = Log(replies=[GOOD["cuda"]])
    section = {"kind": "fakepower", "device": ROCM_0, "interval_ms": 10}
    data = recipe_data([OMP_TO_CUDA], {"kind": "devexec", "device": ROCM_0}, profiler=section)
    outcome = run_all(tmp_path, "probed", data, make_registry(log))
    records = outcome.provenance["device_records"]
    assert [record["key"] for record in records] == ["executor.device", "profiler.device"], "binding order"
    profiler_record = records[1]
    assert [profiler_record[key] for key in ("kind", "indices", "driver")] == ["rocm", [0], ROCM_DRIVER]
    trial = outcome.trial(OMP_TO_CUDA)
    assert [record.key for record in trial.provenance.device_records] == ["executor.device", "profiler.device"]
    page = outcome.trial_md(OMP_TO_CUDA)
    assert any(line.startswith("| profiler.device |") for line in page.splitlines()), "trial.md's Device records"


def test_a_profiler_device_the_host_lacks_is_refused_before_it_is_built(tmp_path: Path) -> None:
    log = Log(replies=[GOOD["cuda"]])
    section = {"kind": "fakepower", "device": ROCM_0, "interval_ms": 10}
    data = recipe_data([OMP_TO_CUDA], {"kind": "devexec", "device": CUDA_0}, profiler=section)
    message = refused(tmp_path, "no-gpu", data, log, rocm="SYNTHETIC: this host has no kfd node")
    assert "profiler.device" in message and "no kfd node" in message, message
    assert "profiler fakepower" not in log.events


# ---------------------------------------------------------------------------
# Refusals before any directory: the profiled executors and their devices


@pytest.mark.parametrize(
    ("profiled", "executed"),
    [(ROCM_1, ROCM_0), (CUDA_0, ROCM_0), (ROCM_0, ROCM_01)],
    ids=["another-index", "another-kind", "executor-on-two-gpus"],
)
def test_a_power_profiler_on_another_gpu_is_refused_before_any_directory(
    tmp_path: Path, profiled: dict[str, Any], executed: dict[str, Any]
) -> None:
    log = Log()
    section = {"kind": "fakepower", "device": profiled, "interval_ms": 10}
    data = recipe_data([OMP_TO_CUDA], {"kind": "devexec", "device": executed}, profiler=section)
    message = refused(tmp_path, "other-gpu", data, log)
    assert "profiler.device" in message and "executor.device" in message, message
    assert "profiler fakepower" not in log.events, "refused before the profiler is built"


def test_a_power_profiler_beside_an_executor_without_a_device_is_refused(tmp_path: Path) -> None:
    log = Log()
    section = {"kind": "fakepower", "device": ROCM_0, "interval_ms": 10}
    message = refused(tmp_path, "no-device", recipe_data([OMP_TO_CUDA], {"kind": "plainexec"}, profiler=section), log)
    assert "profiler.device" in message and "executor.kind" in message, message


def test_per_language_every_profiled_executor_must_run_on_the_profiler_gpu(tmp_path: Path) -> None:
    log = Log()
    section = {"kind": "fakepower", "device": ROCM_0, "interval_ms": 10}
    executor = {"cuda": {"kind": "devexec", "device": ROCM_0}, "omp": "plainexec"}
    data = recipe_data([OMP_TO_CUDA, CUDA_TO_OMP], executor, profiler=section)
    message = refused(tmp_path, "per-language", data, log)
    assert "profiler.device" in message and "executor.omp" in message, (
        f"omp is a target whose attempts run off the profiler's GPU: {message}"
    )


def test_per_language_a_compile_only_target_needs_no_device(tmp_path: Path) -> None:
    log = Log(replies=[GOOD["cuda"], GOOD["omp"]])
    section = {"kind": "fakepower", "device": ROCM_0, "interval_ms": 10}
    executor = {"cuda": {"kind": "devexec", "device": ROCM_0}, "omp": "none"}
    data = recipe_data([OMP_TO_CUDA, CUDA_TO_OMP], executor, profiler=section)
    outcome = run_all(tmp_path, "compile-only-omp", data, make_registry(log))
    assert [attempt.profile for attempt in outcome.trial(OMP_TO_CUDA).attempts] == [power_profile(1)]
    assert [attempt.profile for attempt in outcome.trial(CUDA_TO_OMP).attempts] == [Profile()], "never ran"


def test_per_language_a_source_only_language_needs_no_device(tmp_path: Path) -> None:
    log = Log(replies=[GOOD["cuda"]])
    section = {"kind": "fakepower", "device": ROCM_0, "interval_ms": 10}
    executor = {"cuda": {"kind": "devexec", "device": ROCM_0}, "omp": "plainexec"}
    data = recipe_data([OMP_TO_CUDA], executor, profiler=section)
    outcome = run_all(tmp_path, "source-omp", data, make_registry(log))
    assert [attempt.profile for attempt in outcome.trial(OMP_TO_CUDA).attempts] == [power_profile(1)]
    assert log.events.count("run reference") == 2 and log.events.count("start") == 1, "references are not profiled"


@pytest.mark.parametrize(
    ("stages", "executor"),
    [(COMPILE_STAGES, {"kind": "plainexec"}), (RUN_STAGES, {"kind": "none"})],
    ids=["no-stage-runs-attempts", "compile-only-executor"],
)
def test_a_profiler_with_nothing_to_profile_is_refused(
    tmp_path: Path, stages: list[str], executor: dict[str, Any]
) -> None:
    data = recipe_data([OMP_TO_CUDA], executor, profiler={"kind": "faketime"}, stages=stages)
    control_run(tmp_path, "nothing", data, Log(replies=[GOOD["cuda"]]))
    log = Log()
    message = refused(tmp_path, "nothing", data, log)
    assert "profiler" in message and BLANKET not in message, f"the reason: it would profile nothing: {message}"
    assert "profiler faketime" not in log.events


def test_a_profiler_on_a_simulator_executor_is_refused(tmp_path: Path) -> None:
    data = recipe_data([OMP_TO_CUDA], {"kind": "simexec"}, profiler={"kind": "faketime"})
    control_run(tmp_path, "simulated", data, Log(replies=[GOOD["cuda"]]))
    log = Log()
    message = refused(tmp_path, "simulated", data, log)
    assert "simulator" in message and "profiler" in message, f"simulator timing is never performance: {message}"
    assert BLANKET not in message, message
    assert "profiler faketime" not in log.events


def test_supports_power_without_takes_device_is_refused(tmp_path: Path) -> None:
    log = Log()
    section = {"kind": "nodevpower", "interval_ms": 10}
    data = recipe_data([OMP_TO_CUDA], {"kind": "devexec", "device": ROCM_0}, profiler=section)
    message = refused(tmp_path, "no-takes-device", data, log)
    assert SUPPORTS_POWER in message and TAKES_DEVICE in message and "nodevpower" in message, message
    assert "profiler nodevpower" not in log.events


# ---------------------------------------------------------------------------
# Refusals when the profiler is built, still before any directory


def test_missing_telemetry_is_refused_before_any_directory(tmp_path: Path) -> None:
    log = Log()
    section = {"kind": "notelemetry", "device": ROCM_0, "interval_ms": 10}
    data = recipe_data([OMP_TO_CUDA], {"kind": "devexec", "device": ROCM_0}, profiler=section)
    message = refused(tmp_path, "no-telemetry", data, log)
    assert "profiler.device" in message and "telemetry" in message.lower() and NO_TELEMETRY in message, message


def kfd_root(root: Path) -> Path:
    """Write a SYNTHETIC AMD host root with one KFD GPU node (render minor 128) and no hwmon files."""
    files = {
        "dev/kfd": "",
        "dev/dri/renderD128": "",
        "sys/class/kfd/kfd/topology/nodes/0/gpu_id": "0\n",
        "sys/class/kfd/kfd/topology/nodes/0/properties": "cpu_cores_count 16\n",
        "sys/class/kfd/kfd/topology/nodes/1/gpu_id": "11111\n",
        "sys/class/kfd/kfd/topology/nodes/1/properties": "gfx_target_version 90402\ndrm_render_minor 128\n",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def test_the_registered_rocm_smi_profiler_without_hwmon_is_refused_before_any_directory(tmp_path: Path) -> None:
    host = kfd_root(tmp_path / "host")
    cls = profiled("rocm_smi", "RocmSmiProfiler")
    log = Log()
    registry = make_registry(log)
    registry.register("Profiler", "rocm_smi", with_keywords(cls, root=host))
    section = {"kind": "rocm_smi", "device": ROCM_0, "interval_ms": 10}
    data = recipe_data([OMP_TO_CUDA], {"kind": "devexec", "device": ROCM_0}, profiler=section)
    message = refused(tmp_path, "no-hwmon", data, log, registry)
    assert "profiler.device" in message and "telemetry" in message.lower(), message
    assert "/sys/class/drm/renderD128" in message and str(host) not in message, message


@pytest.mark.parametrize(
    ("changes", "executed", "setting"),
    [
        pytest.param({}, ROCM_0, "profiler.interval_ms", id="no-interval"),
        pytest.param({"interval_ms": 0}, ROCM_0, "profiler.interval_ms", id="zero-interval"),
        pytest.param({"interval_ms": True}, ROCM_0, "profiler.interval_ms", id="bool-interval"),
        pytest.param({"interval_ms": 10, "device": ROCM_01}, ROCM_01, "profiler.device", id="two-gpus"),
    ],
)
def test_a_refused_profiler_setting_is_a_run_error_before_any_directory(
    tmp_path: Path, changes: dict[str, Any], executed: dict[str, Any], setting: str
) -> None:
    cls = profiled("rocm_smi", "RocmSmiProfiler")
    log = Log()
    registry = make_registry(log)
    registry.register("Profiler", "rocm_smi", with_keywords(cls, source=ConstantSource(100.0)))
    section = {"kind": "rocm_smi", "device": ROCM_0, **changes}
    data = recipe_data([OMP_TO_CUDA], {"kind": "devexec", "device": executed}, profiler=section)
    message = refused(tmp_path, "bad-setting", data, log, registry)
    assert setting in message and "rocm_smi" in message, message


def test_a_setting_a_fake_profiler_refuses_is_a_run_error_naming_it(tmp_path: Path) -> None:
    log = Log()
    data = recipe_data([OMP_TO_CUDA], {"kind": "plainexec"}, profiler={"kind": "refusing"})
    message = refused(tmp_path, "refusing", data, log)
    assert "profiler.kind" in message and "refusing" in message and REFUSED_SETTING in message, message


# ---------------------------------------------------------------------------
# A stop() result outside the contract stops the run


@pytest.mark.parametrize(("kind", "named"), [("dictstop", "dictstop"), ("powerless", SUPPORTS_POWER)])
def test_a_stop_result_outside_the_contract_is_refused(tmp_path: Path, kind: str, named: str) -> None:
    log = Log(replies=[GOOD["cuda"]])
    data = recipe_data([OMP_TO_CUDA], {"kind": "plainexec"}, profiler={"kind": kind})
    with pytest.raises(ValueError, match=named) as caught:
        run_recipe(write_recipe(tmp_path, kind, data), options(tmp_path, make_registry(log)))
    assert kind in str(caught.value), "the error names the profiler"
