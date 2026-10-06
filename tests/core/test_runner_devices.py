"""Tests for the device layer in `lassi run` (task P17.2): probe before any directory, record in provenance.

Bible: Result Record (provenance, Storage), Project Recipes (Notes),
Component Interfaces (contract rules), Agent Rules 1, 3, and 10;
plans/p17-portable.md, the planning decisions "Explicit devices (P17.2)" and
"Placement (Design Principle 9)"; plans/LESSONS.md (Audits: a refusal before
any directory needs the output built in memory first).

The contract these tests fix:

- For each binding whose component declares `takes_device`, in binding
  order (the model's first), the runner reads its device section, asks the
  component class for its framework build (an optional framework(), build
  metadata only, called on the class before anything is built), and probes
  the named device through RunOptions.probes (None: the probes lassi
  registers by kind). All of it happens before any component is built and
  before any directory exists.
- A device the host lacks is a RunError naming the key, the kind, and the
  probe's reason: a missing probe for the kind, a framework build that does
  not fit the kind (rocm needs a HIP build), a probe that refuses, or an index
  at or past the host's count. Nothing falls back to the CPU: the cpu probe
  is never asked in place of another kind.
- A framework() that returns neither a FrameworkBuild nor None is a RunError
  naming the binding. So is any error framework() raises, or a framework()
  the runner cannot call on the class, naming the device key (task P17.4).
- A component that takes a device is built with its section as given, a
  mapping: the backend as factory(model_id, device=<mapping>) and an executor
  as factory(**config).
- provenance.json holds `device_records`, the records in binding order,
  always present ([] without a device section), and `driver`, the distinct
  drivers of those records joined by "; " (null when none has one). Each
  trial's provenance copies the whole list and sdk copies the driver;
  trial.json and the Parquet trials table (provenance_device_records, JSON
  text) carry the same records.
- On a host whose GPU node exists but refuses access (alpha01's /dev/kfd,
  OQ-002), the run is refused from the framework build and file metadata:
  the ROCm probe in lassi/executors stats the node, asks access(2), and
  opens nothing.

Every component is a fake in a test Registry beside the real generate and
compile_loop stages; the fake toolchain writes a PLACEHOLDER artifact and
compiles nothing, the fake backend answers with a SYNTHETIC reply, the fake
probes answer with SYNTHETIC facts, and every framework build is SYNTHETIC.
No GPU, framework, or device node is touched. No value in this module is a
measurement.
"""

from __future__ import annotations

import builtins
import dataclasses
import importlib
import io
import json
import os
import pathlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from lassi.core import record as record_module
from lassi.core import runner as runner_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.parquet import read_run_parquet
from lassi.core.registry import Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import CompileLoopStage, GenerateStage
from lassi.core.store import TextStore, read_trial, trial_dir

# The capability literal, so this module collects before lassi.core.capabilities defines TAKES_DEVICE.
TAKES_DEVICE = "takes_device"

SUITE = "lassi-hecbench-10"
ITEM = "layout"
MODEL_ID = "scripted-fixture"
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
# SYNTHETIC bench sources and the scripted reply, which the fake toolchain builds.
OMP_SOURCE = '#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  std::printf("done\\n");\n}\n'
CUDA_SOURCE = "#include <cstdio>\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n"
GOOD_REPLY = render_file_blocks({"main.cu": "int main() { return 0; }\n"})

# SYNTHETIC framework builds (name, version, cuda, hip), in the shape torch's build metadata has.
CPU_BUILD = ("torch", "2.99.0+synthetic-cpu", None, None)
CUDA_BUILD = ("torch", "2.99.0+synthetic-cu", "99.9", None)
HIP_BUILD = ("torch", "2.99.0+synthetic-rocm", None, "9.9.99999-synthetic")
# SYNTHETIC host facts.
CPU_NAME = "SYNTHETIC CPU model 9000"
ROCM_DRIVER = "SYNTHETIC-amdgpu-6.8.5"
ROCM_RUNTIME = "SYNTHETIC-rocm-runtime"
CUDA_DRIVER = "SYNTHETIC-nvidia-999.88"
EXECUTOR_DEVICE = "SYNTHETIC device of devexec"

CPU = {"kind": "cpu"}
ROCM_0 = {"kind": "rocm", "indices": [0]}
# A fake probe's SYNTHETIC refusal.
NO_KFD = "SYNTHETIC: this host has no kfd node"


def core_devices() -> ModuleType:
    """Import lassi.core.devices, failing the test clearly while it does not exist."""
    try:
        return importlib.import_module("lassi.core.devices")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.core.devices does not exist yet (task P17.2): {error}")


def executor_probes() -> ModuleType:
    """Import lassi.executors.devices, failing the test clearly while it does not exist."""
    try:
        return importlib.import_module("lassi.executors.devices")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.executors.devices does not exist yet (task P17.2): {error}")


def framework_build(values: tuple[str, str, str | None, str | None]) -> Any:
    """Return a FrameworkBuild from (name, version, cuda, hip)."""
    name, version, cuda, hip = values
    return core_devices().FrameworkBuild(name=name, version=version, cuda=cuda, hip=hip)


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate and compile variables, point TMPDIR at a test directory, and fake git."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    answers = {"rev-parse": FAKE_COMMIT + "\n", "status": ""}
    monkeypatch.setattr(runner_module, "_git", lambda *args: answers[args[0]])


# ---------------------------------------------------------------------------
# Fake components and probes, all writing to one Log


@dataclass
class Log:
    """What the fakes saw, in order: framework() calls, probes, builds; and each component's device section."""

    events: list[str] = field(default_factory=list)
    sections: dict[str, Any] = field(default_factory=dict)

    def builds(self) -> list[str]:
        """Return the build events, in order."""
        return [event for event in self.events if event.startswith("build ")]


def device_backend(log: Log, framework: Callable[[], Any]) -> type:
    """Return an LLMBackend class "devllm" that takes a device; framework() returns what `framework` gives."""

    class DeviceBackend:
        """A scripted backend that declares takes_device and names its framework build."""

        name = "devllm"
        capabilities = frozenset({"chat", TAKES_DEVICE})

        def __init__(self, model_id: str, *, device: Mapping[str, Any]) -> None:
            """Record the build and the device section it was given."""
            log.events.append("build devllm")
            log.sections["devllm"] = device
            self.model_id = model_id

        @staticmethod
        def framework() -> Any:
            """Record the call and return the configured build metadata."""
            log.events.append("framework devllm")
            return framework()

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Return the SYNTHETIC reply that the fake toolchain builds."""
            return Completion(text=GOOD_REPLY, prompt_tokens=0, completion_tokens=0)

    return DeviceBackend


def plain_backend(log: Log) -> type:
    """Return an LLMBackend class "plainllm" that takes no device."""

    class PlainBackend:
        """A scripted backend with no device section."""

        name = "plainllm"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Record the build."""
            log.events.append("build plainllm")
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Return the SYNTHETIC reply that the fake toolchain builds."""
            return Completion(text=GOOD_REPLY, prompt_tokens=0, completion_tokens=0)

    return PlainBackend


def compile_only_executor(log: Log, takes_device: bool) -> type:
    """Return a compile-only Executor class, "devexec" when it takes a device, else "none"."""
    label = "devexec" if takes_device else "none"

    class CompileOnly:
        """Runs nothing; records its build and, when it takes one, its device section."""

        name = label
        capabilities = frozenset({"compile_only", TAKES_DEVICE} if takes_device else {"compile_only"})

        def __init__(self, **config: Any) -> None:
            """Record the build and the config the recipe gave."""
            log.events.append(f"build {label}")
            if "device" in config:
                log.sections[label] = config["device"]

        def device(self) -> str:
            """Return this fake's SYNTHETIC device name."""
            return EXECUTOR_DEVICE if takes_device else "none (compile only)"

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Fail the test: nothing runs on the compile-only path."""
            raise AssertionError("the compile-only path ran an executor")

    return CompileOnly


def fake_toolchain(log: Log) -> type:
    """Return a Toolchain class "nvcc-sm80" without PIN that writes a PLACEHOLDER artifact; it compiles nothing."""

    class FakeToolchain:
        """Writes the files and a PLACEHOLDER artifact."""

        name = "nvcc-sm80"
        capabilities = frozenset({"diagnostics"})

        def __init__(self) -> None:
            """Record the build."""
            log.events.append("build nvcc-sm80")

        def build(self, files: Mapping[str, str], workdir: Path) -> BuildResult:
            """Write every file and a PLACEHOLDER artifact."""
            for path, text in files.items():
                target = Path(workdir) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            artifact = Path(workdir) / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


@dataclass
class FakeProbe:
    """A probe of one kind with SYNTHETIC facts, or a refusal; it records each call in the Log."""

    kind: str
    log: Log
    count: int = 2
    name: str | None = None
    memory_bytes: int | None = None
    driver: str | None = None
    runtime: str | None = None
    refuse: str | None = None
    calls: int = 0

    def probe(self) -> Any:
        """Record the call; raise DeviceUnavailable with `refuse`, or return the facts."""
        devices = core_devices()
        self.calls += 1
        self.log.events.append(f"probe {self.kind}")
        if self.refuse is not None:
            raise devices.DeviceUnavailable(self.refuse)
        return devices.HostFacts(
            count=self.count, name=self.name, memory_bytes=self.memory_bytes, driver=self.driver, runtime=self.runtime
        )


def fake_probes(log: Log, **changes: Any) -> dict[str, FakeProbe]:
    """Return SYNTHETIC cpu, rocm, and cuda probes; `changes` maps a kind to a replacement probe or None (absent)."""
    probes: dict[str, FakeProbe | None] = {
        "cpu": FakeProbe("cpu", log, count=8, name=CPU_NAME, memory_bytes=4096),
        "rocm": FakeProbe("rocm", log, count=2, driver=ROCM_DRIVER, runtime=ROCM_RUNTIME),
        "cuda": FakeProbe("cuda", log, count=2, driver=CUDA_DRIVER),
    }
    probes.update(changes)
    return {kind: probe for kind, probe in probes.items() if probe is not None}


def make_registry(log: Log, framework: Callable[[], Any] = lambda: framework_build(CPU_BUILD)) -> Registry:
    """Return a test Registry: device and plain backends and executors, the fake toolchain, and the real stages."""
    registry = Registry()
    registry.register("LLMBackend", "devllm", device_backend(log, framework))
    registry.register("LLMBackend", "plainllm", plain_backend(log))
    registry.register("Executor", "devexec", compile_only_executor(log, takes_device=True))
    registry.register("Executor", "none", compile_only_executor(log, takes_device=False))
    registry.register("Toolchain", "nvcc-sm80", fake_toolchain(log))
    registry.register("Stage", "generate", GenerateStage)
    registry.register("Stage", "compile_loop", CompileLoopStage)
    return registry


# ---------------------------------------------------------------------------
# Recipes, bench sources, and runs


def recipe_data(
    model_device: Mapping[str, Any] | None = None,
    executor: Mapping[str, Any] | None = None,
    *,
    backend: str = "devllm",
    trials: int = 1,
) -> dict[str, Any]:
    """Return a compile-only recipe for the layout item, omp to cuda, with the given device sections."""
    model: dict[str, Any] = {"backend": backend, "id": MODEL_ID}
    if model_device is not None:
        model["device"] = json.loads(json.dumps(model_device))
    return {
        "extends": "base",
        "model": model,
        "llm": {"sampling": {"max_tokens": 64}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": "omp", "target": "cuda"}],
        "prompts": "p0-smoke",
        "toolchain": {"cuda": "nvcc-sm80"},
        "stages": ["generate", "compile_loop"],
        "executor": json.loads(json.dumps(executor or {"kind": "none"})),
        "trials": {"n": trials},
    }


def write_recipe(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe <directory>/<name>.yaml (ASCII, LF) and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """Write the SYNTHETIC layout sources where the suite manifest lays them out; return the bench root."""
    root = tmp_path / "bench"
    for relative, text in (("src/layout-omp/main.cpp", OMP_SOURCE), ("src/layout-cuda/main.cu", CUDA_SOURCE)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def run(
    tmp_path: Path, bench: Path, registry: Registry, data: Mapping[str, Any], probes: Mapping[str, Any], name: str
) -> Path:
    """Write the recipe `name` and run it with the given registry and probes; return the run directory."""
    options = RunOptions(
        runs_root=tmp_path / "runs-root", run_id="device-run", bench_root=bench, registry=registry, probes=probes
    )
    return run_recipe(write_recipe(tmp_path, name, data), options)


def refused(
    tmp_path: Path, bench: Path, registry: Registry, data: Mapping[str, Any], probes: Mapping[str, Any]
) -> str:
    """Run a recipe the runner must refuse; assert no directory exists under the runs root; return the message."""
    with pytest.raises(RunError) as info:
        run(tmp_path, bench, registry, data, probes, "refused")
    assert not (tmp_path / "runs-root").exists(), "a refusal comes before any directory exists"
    return str(info.value)


def trial_id(project: str) -> str:
    """Return the id of a project's first trial."""
    return f"{project}/{MODEL_ID}/{SUITE}/omp-cuda/{ITEM}/run01"


def read_ascii(path: Path) -> str:
    """Return a file's text after checking it is ASCII with LF newlines."""
    data = path.read_bytes()
    assert data.isascii() and b"\r" not in data, path
    return data.decode("ascii")


def manifest(run_dir: Path) -> dict[str, Any]:
    """Return the run's provenance.json."""
    return json.loads(read_ascii(run_dir / "provenance.json"))


def cpu_record(key: str = "model.device", values: tuple[str, str, str | None, str | None] = CPU_BUILD) -> dict:
    """Return the record the fake cpu probe gives for `key` under a framework build."""
    return {
        "key": key, "kind": "cpu", "indices": [], "name": CPU_NAME, "count": 8, "memory_bytes": 4096,
        "driver": None, "runtime": None, "framework": values[0], "framework_version": values[1],
    }


def rocm_record(key: str = "executor.device", indices: Sequence[int] = (0,)) -> dict:
    """Return the record the fake rocm probe gives for an executor without a framework build."""
    return {
        "key": key, "kind": "rocm", "indices": list(indices), "name": None, "count": 2, "memory_bytes": None,
        "driver": ROCM_DRIVER, "runtime": ROCM_RUNTIME, "framework": None, "framework_version": None,
    }


def assert_every_trial_carries(run_dir: Path, ids: Sequence[str], records: list[dict], driver: str | None) -> None:
    """Assert each trial's record, trial.json, and Parquet row hold `records` and `driver` as sdk."""
    rows = {row["trial_id"]: row for row in read_run_parquet(run_dir / "parquet")["trials"]}
    assert sorted(rows) == sorted(ids)
    for one in ids:
        trial = read_trial(trial_dir(run_dir, one), TextStore(run_dir))
        assert [record_module.to_dict(item) for item in trial.provenance.device_records] == records, one
        assert trial.provenance.sdk == driver, one
        raw = json.loads(read_ascii(trial_dir(run_dir, one) / "trial.json"))
        assert raw["provenance"]["device_records"] == records, one
        assert raw["provenance"]["sdk"] == driver, one
        assert json.loads(rows[one]["provenance_device_records"]) == records, one


# ---------------------------------------------------------------------------
# The option


def test_run_options_take_the_probes_with_none_meaning_the_registered_ones() -> None:
    assert "probes" in {spec.name for spec in dataclasses.fields(RunOptions)}
    assert RunOptions().probes is None


# ---------------------------------------------------------------------------
# Provenance: the records, the driver, and every trial's copy


def test_device_records_reach_provenance_json_and_every_trial(tmp_path: Path, bench: Path) -> None:
    log = Log()
    data = recipe_data(CPU, {"kind": "devexec", "device": ROCM_0}, trials=2)
    run_dir = run(tmp_path, bench, make_registry(log), data, fake_probes(log), "records")
    expected = [cpu_record("model.device"), rocm_record("executor.device")]
    written = manifest(run_dir)
    assert written["device_records"] == expected, "binding order: the model's section first"
    assert [list(item) for item in written["device_records"]] == [list(expected[0])] * 2, "field order"
    assert written["driver"] == ROCM_DRIVER
    assert written["device"] == EXECUTOR_DEVICE, "the executor's device() keeps its key and meaning"
    ids = [trial_id("records"), trial_id("records").replace("run01", "run02")]
    assert_every_trial_carries(run_dir, ids, expected, ROCM_DRIVER)


def test_driver_is_filled_from_the_device_records(tmp_path: Path, bench: Path) -> None:
    log = Log()
    registry = make_registry(log, framework=lambda: framework_build(HIP_BUILD))
    data = recipe_data(ROCM_0, {"kind": "devexec", "device": {"kind": "cuda", "indices": [1]}})
    run_dir = run(tmp_path, bench, registry, data, fake_probes(log), "drivers")
    written = manifest(run_dir)
    assert [item["key"] for item in written["device_records"]] == ["model.device", "executor.device"]
    model, executor = written["device_records"]
    assert (model["runtime"], model["framework"], model["framework_version"]) == (HIP_BUILD[3], "torch", HIP_BUILD[1])
    assert (executor["kind"], executor["indices"], executor["driver"]) == ("cuda", [1], CUDA_DRIVER)
    assert written["driver"] == f"{ROCM_DRIVER}; {CUDA_DRIVER}"
    assert_every_trial_carries(run_dir, [trial_id("drivers")], written["device_records"], written["driver"])
    assert f"| Driver | {ROCM_DRIVER}; {CUDA_DRIVER} |" in read_ascii(run_dir / "run.md")


def test_driver_stays_null_and_records_empty_without_device_sections(tmp_path: Path, bench: Path) -> None:
    log = Log()
    probes = fake_probes(log)
    run_dir = run(tmp_path, bench, make_registry(log), recipe_data(backend="plainllm"), probes, "plain")
    written = manifest(run_dir)
    assert (written["device_records"], written["driver"]) == ([], None)
    assert_every_trial_carries(run_dir, [trial_id("plain")], [], None)
    rows = read_run_parquet(run_dir / "parquet")["trials"]
    assert [row["provenance_device_records"] for row in rows] == ["[]"]
    assert all(probe.calls == 0 for probe in probes.values()), "no device section, no probe"


def test_a_per_language_section_records_its_key(tmp_path: Path, bench: Path) -> None:
    log = Log()
    executor = {"cuda": {"kind": "devexec", "device": CPU}}
    data = recipe_data(backend="plainllm", executor=executor)
    run_dir = run(tmp_path, bench, make_registry(log), data, fake_probes(log), "per-language")
    record = cpu_record("executor.cuda.device")
    record.update(framework=None, framework_version=None)
    assert manifest(run_dir)["device_records"] == [record]
    assert_every_trial_carries(run_dir, [trial_id("per-language")], [record], None)


def test_the_backend_is_built_with_its_device_section(tmp_path: Path, bench: Path) -> None:
    log = Log()
    data = recipe_data(CPU, {"kind": "devexec", "device": ROCM_0})
    run(tmp_path, bench, make_registry(log), data, fake_probes(log), "built")
    assert log.sections == {"devllm": CPU, "devexec": ROCM_0}, "each component gets its section as given"


def test_the_framework_and_the_probes_come_before_any_component_is_built(tmp_path: Path, bench: Path) -> None:
    log = Log()
    data = recipe_data(CPU, {"kind": "devexec", "device": ROCM_0})
    run(tmp_path, bench, make_registry(log), data, fake_probes(log), "order")
    first_build = log.events.index(log.builds()[0])
    before = log.events[:first_build]
    assert "framework devllm" in before and "probe cpu" in before and "probe rocm" in before, log.events
    assert before.index("framework devllm") < before.index("probe cpu"), "the framework build is read first"
    assert not any(event.startswith("probe ") for event in log.events[first_build:]), log.events


# ---------------------------------------------------------------------------
# Refusals before any directory, and no fallback


@pytest.mark.parametrize(
    ("probe_changes", "reason"),
    [
        pytest.param({"refuse": NO_KFD}, NO_KFD, id="refused"),
        pytest.param({"count": 0}, "0", id="no-device"),
    ],
)
def test_a_device_the_host_lacks_is_refused_before_any_directory(
    tmp_path: Path, bench: Path, probe_changes: dict[str, Any], reason: str
) -> None:
    log = Log()
    registry = make_registry(log, framework=lambda: framework_build(HIP_BUILD))
    rocm = FakeProbe("rocm", log, driver=ROCM_DRIVER, **probe_changes)
    message = refused(tmp_path, bench, registry, recipe_data(ROCM_0), fake_probes(log, rocm=rocm))
    assert "model.device" in message and "rocm" in message and reason in message, message
    assert log.builds() == [], "no component is built before the probe refuses"


def test_an_index_past_the_count_is_refused_naming_the_key(tmp_path: Path, bench: Path) -> None:
    log = Log()
    data = recipe_data(backend="plainllm", executor={"kind": "devexec", "device": {"kind": "rocm", "indices": [0, 3]}})
    message = refused(tmp_path, bench, make_registry(log), data, fake_probes(log))
    assert "executor.device" in message and "3" in message, message
    assert log.builds() == []


def test_no_cpu_fallback(tmp_path: Path, bench: Path) -> None:
    log = Log()
    registry = make_registry(log, framework=lambda: framework_build(HIP_BUILD))
    probes = fake_probes(log, rocm=FakeProbe("rocm", log, refuse="SYNTHETIC: refused"))
    refused(tmp_path, bench, registry, recipe_data(ROCM_0), probes)
    assert probes["cpu"].calls == 0, "a refused rocm device never falls back to the CPU"
    cuda_registry = make_registry(log, framework=lambda: framework_build(CUDA_BUILD))
    only_cpu = fake_probes(log, rocm=None, cuda=None)
    message = refused(tmp_path, bench, cuda_registry, recipe_data({"kind": "cuda", "indices": [0]}), only_cpu)
    assert "model.device" in message and "cuda" in message, message
    assert only_cpu["cpu"].calls == 0, "a kind with no probe never falls back to the CPU"
    assert log.builds() == []


def test_a_framework_build_refusal_comes_before_the_probe(tmp_path: Path, bench: Path) -> None:
    log = Log()
    probes = fake_probes(log)
    message = refused(tmp_path, bench, make_registry(log), recipe_data(ROCM_0), probes)
    assert "model.device" in message and "rocm" in message, message
    assert probes["rocm"].calls == 0, "a CPU build refuses kind rocm from its build metadata, before any probe"
    assert log.events == ["framework devllm"], log.events


@pytest.mark.parametrize(
    "answer",
    [pytest.param("torch 2.99.0+synthetic", id="string"), pytest.param(CPU_BUILD, id="tuple")],
)
def test_framework_must_return_a_build_or_none(tmp_path: Path, bench: Path, answer: Any) -> None:
    log = Log()
    registry = make_registry(log, framework=lambda: answer)
    message = refused(tmp_path, bench, registry, recipe_data(CPU), fake_probes(log))
    assert "devllm" in message, message
    assert log.builds() == []


def test_a_framework_of_none_is_no_framework(tmp_path: Path, bench: Path) -> None:
    log = Log()
    registry = make_registry(log, framework=lambda: None)
    run_dir = run(tmp_path, bench, registry, recipe_data(ROCM_0), fake_probes(log), "no-framework")
    (written,) = manifest(run_dir)["device_records"]
    assert (written["framework"], written["framework_version"], written["runtime"]) == (None, None, ROCM_RUNTIME)


# A failing framework()'s SYNTHETIC message (task P17.4: hf_local's raises FrameworkMissing without the extra).
FRAMEWORK_FAILURE = "SYNTHETIC: the framework build could not be read"


def raise_framework_failure() -> Any:
    """Raise the SYNTHETIC RuntimeError of a framework() that fails."""
    raise RuntimeError(FRAMEWORK_FAILURE)


class InstanceFramework:
    """An LLMBackend "devllm" that takes a device but defines framework() as an instance method; never built."""

    name = "devllm"
    capabilities = frozenset({"chat", TAKES_DEVICE})

    def __init__(self, model_id: str, *, device: Mapping[str, Any]) -> None:
        """Fail the test: the run is refused before any component is built."""
        raise AssertionError("the backend was built although the runner could not call its framework()")

    def framework(self) -> Any:
        """Return no framework; the runner calls framework() on the class, where this needs an instance."""
        return None

    def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
        """Fail the test: no request is sent."""
        raise AssertionError("a request was sent")


def test_a_framework_that_raises_is_a_run_error_naming_the_key(tmp_path: Path, bench: Path) -> None:
    # Any error framework() raises, a missing framework's among them, is a RunError naming the device key, before
    # any component is built or any directory exists (PHASE-NOTES P17, P17.2's note for P17.4).
    log = Log()
    probes = fake_probes(log)
    message = refused(tmp_path, bench, make_registry(log, framework=raise_framework_failure), recipe_data(CPU), probes)
    assert "model.device" in message and "devllm" in message and FRAMEWORK_FAILURE in message, message
    assert log.builds() == [] and probes["cpu"].calls == 0
    log = Log()
    registry = Registry()
    registry.register("LLMBackend", "devllm", InstanceFramework)
    registry.register("Executor", "none", compile_only_executor(log, takes_device=False))
    registry.register("Toolchain", "nvcc-sm80", fake_toolchain(log))
    registry.register("Stage", "generate", GenerateStage)
    registry.register("Stage", "compile_loop", CompileLoopStage)
    message = refused(tmp_path, bench, registry, recipe_data(CPU), fake_probes(log))
    assert "model.device" in message and "devllm" in message, message
    assert log.builds() == []


# ---------------------------------------------------------------------------
# alpha01's shape: the node exists but refuses access; the probe opens nothing


def opened_under(monkeypatch: pytest.MonkeyPatch, root: Path) -> list[str]:
    """Record every file opened under `root` (builtins, io, os, and pathlib opens) and return the list."""
    opened: list[str] = []
    prefix = os.path.normcase(os.path.abspath(root))

    def note(target: Any) -> None:
        if isinstance(target, int):
            return
        path = os.path.normcase(os.path.abspath(os.fspath(target)))
        if path == prefix or path.startswith(prefix + os.sep):
            opened.append(path)

    real_open, real_os_open, real_path_open = builtins.open, os.open, pathlib.Path.open

    def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        note(file)
        return real_open(file, *args, **kwargs)

    def guarded_os_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        note(path)
        return real_os_open(path, *args, **kwargs)

    def guarded_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        note(self)
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(pathlib.Path, "open", guarded_path_open)
    return opened


def alpha01_root(root: Path) -> Path:
    """Write a SYNTHETIC host root shaped like alpha01: a kfd node, two GPU topology nodes, and a driver version."""
    files = {
        "dev/kfd": "",
        "sys/class/kfd/kfd/topology/nodes/0/gpu_id": "0\n",
        "sys/class/kfd/kfd/topology/nodes/1/gpu_id": "11111\n",
        "sys/class/kfd/kfd/topology/nodes/2/gpu_id": "22222\n",
        "sys/module/amdgpu/version": "SYNTHETIC-6.8.5\n",
        "proc/cpuinfo": f"processor\t: 0\nmodel name\t: {CPU_NAME}\n\n",
        "proc/meminfo": "MemTotal:       4 kB\n",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def test_a_gpu_kind_on_an_alpha01_shaped_root_is_refused(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes_module = executor_probes()
    root = alpha01_root(tmp_path / "host")
    asked: list[Path] = []

    def deny(path: Path) -> bool:
        asked.append(Path(path))
        return False

    probes = {"rocm": probes_module.RocmProbe(root=root, access=deny), "cpu": probes_module.CpuProbe(root=root)}
    log = Log()
    registry = make_registry(log, framework=lambda: framework_build(HIP_BUILD))
    opened = opened_under(monkeypatch, root)
    message = refused(tmp_path, bench, registry, recipe_data(ROCM_0), probes)
    assert "model.device" in message and "kfd" in message, message
    assert asked == [root / "dev" / "kfd"], "the probe asks access(2) about the node"
    assert opened == [], f"the probe opened files under the host root: {opened}"
    assert log.builds() == []


def test_the_cpu_build_refuses_a_gpu_kind_on_that_root_before_any_metadata(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes_module = executor_probes()
    root = alpha01_root(tmp_path / "host")
    asked: list[Path] = []
    probes = {"rocm": probes_module.RocmProbe(root=root, access=lambda path: asked.append(Path(path)) or True)}
    log = Log()
    opened = opened_under(monkeypatch, root)
    message = refused(tmp_path, bench, make_registry(log), recipe_data(ROCM_0), probes)
    assert "model.device" in message, message
    assert (asked, opened) == ([], []), "the CPU build refuses kind rocm before the probe stats or reads anything"
