"""Tests for the device-neutral device layer in lassi/core (task P17.2).

Bible: Result Record (provenance), Project Recipes (Notes), Component
Interfaces (contract rules), Agent Rules 1, 3, and 10, Design Principle 9;
plans/p17-portable.md, the planning decisions "Explicit devices (P17.2)" and
"Placement (Design Principle 9)".

lassi.core.devices holds the parts every caller shares:

- parse_device(value, path) -> DeviceSpec reads one device section,
  `{kind, indices}`. kind is one of cpu, cuda, rocm (record.DEVICE_KINDS),
  required, with no default and no case folding. cuda and rocm need indices,
  a non-empty list of distinct integers of at least 0 (a bool is not an
  integer), kept in the order written; cpu takes no indices key at all. Any
  other key, and a value that is not a mapping, is refused. Every refusal is
  a DeviceSectionError (a ValueError) whose message names the dotted key.
- probe_device(key, spec, framework, probes) -> DeviceRecord checks that a
  probe exists for the kind, then that the framework build fits the kind
  before any probe runs (cuda needs a CUDA build without HIP, rocm a HIP
  build, since ROCm PyTorch reaches its GPUs through the cuda device type;
  cpu fits any build), then probes, then refuses an index at or past the
  host's count. It never falls back to another kind. A refusal is
  DeviceUnavailable.
- register_probe(probe, probes) fills a probe registry by kind, refusing a
  kind outside DEVICE_KINDS and a kind registered twice.
- device_driver(records) joins the distinct drivers in record order with
  "; ", or gives None.

No file under lassi/core imports an ML framework or a vendor library, and
lassi/core/devices.py reads no file (device files are read only in
lassi/executors and lassi/profilers). Every probe here is a fake and every
framework build is SYNTHETIC; no value in this module is a measurement.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.core import record

REPO = Path(__file__).resolve().parents[2]
CORE_DIR = REPO / "lassi" / "core"
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
PROJECT_NAMES = ("lassi-repro", "lassi-ee", "lassi-df", "hecbench", "qwen", "wizardcoder", "a100", "mi300x", "gpt-oss")
# The modules P17.2 adds: the device-neutral core and the host probes beside the executors (Design Principle 9).
NEW_MODULES = ("lassi.core.devices", "lassi.executors.devices")

# Top-level packages of ML frameworks and vendor device libraries that lassi/core must never import.
FORBIDDEN_IMPORTS = frozenset(
    {
        "torch", "torchvision", "transformers", "trl", "peft", "accelerate", "vllm", "sglang", "llama_cpp",
        "tensorflow", "jax", "jaxlib", "onnxruntime", "triton", "deepspeed", "cupy", "pycuda", "cuda",
        "pynvml", "nvidia_smi", "amdsmi", "rocm_smi", "pyrsmi", "pyamdgpuinfo", "habana_frameworks",
        "intel_extension_for_pytorch",
    }
)
# Host device paths that only lassi/executors and lassi/profilers may name in code (not in docstrings).
DEVICE_PATHS = (
    "/dev/kfd", "/dev/dri", "/dev/nvidia", "/sys/class/kfd", "/sys/devices/virtual/kfd", "/sys/module/amdgpu",
    "/proc/driver/nvidia", "/proc/cpuinfo", "/proc/meminfo",
)
# Calls that read a file or list a directory; lassi/core/devices.py makes none of them.
FILE_CALLS = frozenset(
    {"open", "read_text", "read_bytes", "listdir", "scandir", "iterdir", "glob", "rglob", "walk", "readlink"}
)

# SYNTHETIC framework builds, in the shape of torch.__version__, torch.version.cuda, and torch.version.hip.
CPU_BUILD = ("torch", "2.99.0+synthetic-cpu", None, None)
CUDA_BUILD = ("torch", "2.99.0+synthetic-cu", "99.9", None)
HIP_BUILD = ("torch", "2.99.0+synthetic-rocm", None, "9.9.99999-synthetic")
BOTH_BUILD = ("torch", "2.99.0+synthetic-both", "99.9", "9.9.99999-synthetic")
SYNTHETIC_DRIVER = "SYNTHETIC-driver-1.2.3"
SYNTHETIC_RUNTIME = "SYNTHETIC-runtime-4.5.6"


@pytest.fixture(scope="module")
def devices() -> ModuleType:
    """Import lassi.core.devices, failing each test clearly while the module does not exist."""
    try:
        return importlib.import_module("lassi.core.devices")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.core.devices does not exist yet (task P17.2): {error}")


def device_record_class() -> type:
    """Return record.DeviceRecord, failing the test clearly while it does not exist."""
    cls = getattr(record, "DeviceRecord", None)
    if cls is None:
        pytest.fail("lassi.core.record has no DeviceRecord (task P17.2)")
    return cls


def build(devices: ModuleType, values: tuple[str, str, str | None, str | None]) -> Any:
    """Return a FrameworkBuild from (name, version, cuda, hip)."""
    name, version, cuda, hip = values
    return devices.FrameworkBuild(name=name, version=version, cuda=cuda, hip=hip)


def facts(devices: ModuleType, **changes: Any) -> Any:
    """Return SYNTHETIC HostFacts: two devices, a name, memory, a driver, and a runtime, with the given changes."""
    values: dict[str, Any] = {
        "count": 2,
        "name": "SYNTHETIC device model",
        "memory_bytes": 1024,
        "driver": SYNTHETIC_DRIVER,
        "runtime": SYNTHETIC_RUNTIME,
    }
    values.update(changes)
    return devices.HostFacts(**values)


@dataclass
class FakeProbe:
    """A probe of one kind that returns fixed facts, or raises `error`, and counts its calls."""

    kind: str
    answer: Any = None
    error: BaseException | None = None
    calls: list[str] = field(default_factory=list)

    def probe(self) -> Any:
        """Record the call, then raise the error or return the facts."""
        self.calls.append(self.kind)
        if self.error is not None:
            raise self.error
        return self.answer


def spec(devices: ModuleType, kind: str, indices: tuple[int, ...] = ()) -> Any:
    """Return a DeviceSpec of `kind` and `indices`."""
    return devices.DeviceSpec(kind=kind, indices=indices)


# ---------------------------------------------------------------------------
# parse_device: one shape, every refusal naming the key


def test_device_kinds_are_cpu_cuda_and_rocm() -> None:
    assert getattr(record, "DEVICE_KINDS", None) == ("cpu", "cuda", "rocm")


def test_device_section_error_is_a_value_error(devices: ModuleType) -> None:
    assert issubclass(devices.DeviceSectionError, ValueError)


def test_parse_device_accepts_cpu_without_indices(devices: ModuleType) -> None:
    parsed = devices.parse_device({"kind": "cpu"}, "model.device")
    assert (parsed.kind, parsed.indices) == ("cpu", ())
    assert parsed == spec(devices, "cpu")


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        pytest.param({"kind": "cuda", "indices": [0]}, ("cuda", (0,)), id="cuda"),
        pytest.param({"kind": "rocm", "indices": [1, 0]}, ("rocm", (1, 0)), id="rocm-order-kept"),
        pytest.param({"indices": [3, 7, 2], "kind": "rocm"}, ("rocm", (3, 7, 2)), id="keys-in-any-order"),
    ],
)
def test_parse_device_accepts_gpu_kinds_with_indices(
    devices: ModuleType, section: dict[str, Any], expected: tuple[str, tuple[int, ...]]
) -> None:
    parsed = devices.parse_device(section, "executor.device")
    assert (parsed.kind, parsed.indices) == expected
    assert isinstance(parsed.indices, tuple)


def test_device_spec_is_frozen(devices: ModuleType) -> None:
    parsed = devices.parse_device({"kind": "cpu"}, "model.device")
    with pytest.raises(AttributeError):
        parsed.kind = "rocm"


def refusal(devices: ModuleType, section: Any, path: str) -> str:
    """Return the message of the DeviceSectionError parse_device raises for `section` at `path`."""
    with pytest.raises(devices.DeviceSectionError) as info:
        devices.parse_device(section, path)
    return str(info.value)


@pytest.mark.parametrize("section", [{}, {"indices": [0]}], ids=["empty", "indices-only"])
def test_parse_device_refuses_missing_kind_naming_the_key(devices: ModuleType, section: dict[str, Any]) -> None:
    assert "model.device.kind" in refusal(devices, section, "model.device")


@pytest.mark.parametrize("kind", ["xpu", "mps", "CUDA", "Rocm", "gpu", "", 1, True, None, ["cuda"]])
def test_parse_device_refuses_unknown_kind_naming_the_key(devices: ModuleType, kind: Any) -> None:
    message = refusal(devices, {"kind": kind, "indices": [0]}, "executor.device")
    assert "executor.device.kind" in message


def test_parse_device_refuses_duplicate_indices_naming_the_key(devices: ModuleType) -> None:
    message = refusal(devices, {"kind": "cuda", "indices": [1, 0, 1]}, "executor.cuda.device")
    assert "executor.cuda.device.indices" in message
    assert "1" in message, "the message names the repeated index"


@pytest.mark.parametrize(
    "section",
    [
        pytest.param({"kind": "rocm"}, id="gpu-without-indices"),
        pytest.param({"kind": "cuda", "indices": []}, id="empty-list"),
        pytest.param({"kind": "rocm", "indices": [-1]}, id="negative"),
        pytest.param({"kind": "rocm", "indices": [True]}, id="bool"),
        pytest.param({"kind": "rocm", "indices": ["0"]}, id="string-item"),
        pytest.param({"kind": "rocm", "indices": [0.5]}, id="float-item"),
        pytest.param({"kind": "rocm", "indices": [1.0]}, id="whole-float-item"),
        pytest.param({"kind": "cuda", "indices": "0"}, id="string"),
        pytest.param({"kind": "cuda", "indices": 0}, id="bare-int"),
        pytest.param({"kind": "cuda", "indices": None}, id="null"),
        pytest.param({"kind": "cuda", "indices": {"0": 0}}, id="mapping"),
    ],
)
def test_parse_device_refuses_bad_indices(devices: ModuleType, section: dict[str, Any]) -> None:
    assert "model.device.indices" in refusal(devices, section, "model.device")


@pytest.mark.parametrize("indices", [[0], [], [0, 1]], ids=["one", "empty", "two"])
def test_parse_device_refuses_indices_for_cpu(devices: ModuleType, indices: list[int]) -> None:
    assert "oracle.device.indices" in refusal(devices, {"kind": "cpu", "indices": indices}, "oracle.device")


def test_parse_device_refuses_unknown_key_and_non_mapping(devices: ModuleType) -> None:
    message = refusal(devices, {"kind": "cpu", "count": 1}, "model.device")
    assert "model.device.count" in message
    for value in ("cuda:0", ["cuda", 0], 0, None):
        message = refusal(devices, value, "executor.omp.device")
        assert "executor.omp.device" in message, value


def test_parse_device_reports_the_first_problem_with_the_path_it_was_given(devices: ModuleType) -> None:
    # The same section under two paths: each message names its own path, so P17.8's train loader can reuse it.
    section = {"kind": "rocm", "indices": [0, 0]}
    assert "train.device.indices" in refusal(devices, section, "train.device")
    assert "model.device.indices" in refusal(devices, section, "model.device")


# ---------------------------------------------------------------------------
# probe_device: the framework build first, then the probe, then the indices; never another kind


def test_probe_device_builds_the_record(devices: ModuleType) -> None:
    probe = FakeProbe("rocm", answer=facts(devices))
    hip = build(devices, HIP_BUILD)
    built = devices.probe_device("model.device", spec(devices, "rocm", (1, 0)), hip, {"rocm": probe})
    expected = device_record_class()(
        key="model.device",
        kind="rocm",
        indices=[1, 0],
        name="SYNTHETIC device model",
        count=2,
        memory_bytes=1024,
        driver=SYNTHETIC_DRIVER,
        runtime=HIP_BUILD[3],
        framework="torch",
        framework_version=HIP_BUILD[1],
    )
    assert built == expected
    assert probe.calls == ["rocm"]


def test_probe_device_takes_the_cuda_runtime_from_a_cuda_build(devices: ModuleType) -> None:
    probe = FakeProbe("cuda", answer=facts(devices, count=1))
    cuda = build(devices, CUDA_BUILD)
    built = devices.probe_device("executor.device", spec(devices, "cuda", (0,)), cuda, {"cuda": probe})
    assert (built.kind, built.indices, built.runtime) == ("cuda", [0], CUDA_BUILD[2])
    assert (built.framework, built.framework_version) == ("torch", CUDA_BUILD[1])


@pytest.mark.parametrize("values", [CPU_BUILD, CUDA_BUILD, HIP_BUILD], ids=["cpu-build", "cuda-build", "hip-build"])
def test_probe_device_accepts_cpu_with_any_framework_build(
    devices: ModuleType, values: tuple[str, str, str | None, str | None]
) -> None:
    probe = FakeProbe("cpu", answer=facts(devices, count=8, driver=None, runtime=None))
    built = devices.probe_device("model.device", spec(devices, "cpu"), build(devices, values), {"cpu": probe})
    assert (built.key, built.kind, built.indices, built.count) == ("model.device", "cpu", [], 8)
    assert built.runtime is None, "a cpu record names no GPU runtime"
    assert (built.framework, built.framework_version) == (values[0], values[1])
    assert probe.calls == ["cpu"]


def test_probe_device_refuses_rocm_without_a_hip_build_before_probing(devices: ModuleType) -> None:
    probe = FakeProbe("rocm", answer=facts(devices))
    for values in (CPU_BUILD, CUDA_BUILD):
        with pytest.raises(devices.DeviceUnavailable) as info:
            devices.probe_device("model.device", spec(devices, "rocm", (0,)), build(devices, values), {"rocm": probe})
        assert "hip" in str(info.value).lower(), info.value
    assert probe.calls == [], "the framework build is checked before any probe runs"


@pytest.mark.parametrize("values", [CPU_BUILD, HIP_BUILD, BOTH_BUILD], ids=["cpu-build", "hip-build", "cuda-and-hip"])
def test_probe_device_refuses_cuda_with_a_hip_or_cpu_build(
    devices: ModuleType, values: tuple[str, str, str | None, str | None]
) -> None:
    probe = FakeProbe("cuda", answer=facts(devices))
    with pytest.raises(devices.DeviceUnavailable):
        devices.probe_device("model.device", spec(devices, "cuda", (0,)), build(devices, values), {"cuda": probe})
    assert probe.calls == [], "the framework build is checked before any probe runs"


def test_probe_device_refuses_an_index_past_the_count(devices: ModuleType) -> None:
    probe = FakeProbe("rocm", answer=facts(devices, count=2))
    with pytest.raises(devices.DeviceUnavailable) as info:
        devices.probe_device("executor.device", spec(devices, "rocm", (0, 3)), None, {"rocm": probe})
    message = str(info.value)
    assert "3" in message and "2" in message, f"the message names the index and the count: {message}"
    with pytest.raises(devices.DeviceUnavailable):
        devices.probe_device("executor.device", spec(devices, "rocm", (2,)), None, {"rocm": probe})
    within = devices.probe_device("executor.device", spec(devices, "rocm", (1, 0)), None, {"rocm": probe})
    assert within.indices == [1, 0]


def test_probe_device_refuses_any_index_when_the_host_has_none(devices: ModuleType) -> None:
    probe = FakeProbe("cuda", answer=facts(devices, count=0))
    with pytest.raises(devices.DeviceUnavailable):
        devices.probe_device("executor.device", spec(devices, "cuda", (0,)), None, {"cuda": probe})


def test_probe_device_refuses_a_kind_without_a_probe(devices: ModuleType) -> None:
    cpu = FakeProbe("cpu", answer=facts(devices, driver=None, runtime=None))
    for kind in ("rocm", "cuda"):
        with pytest.raises(devices.DeviceUnavailable):
            devices.probe_device("model.device", spec(devices, kind, (0,)), None, {"cpu": cpu})
    with pytest.raises(devices.DeviceUnavailable):
        devices.probe_device("model.device", spec(devices, "cpu"), None, {})
    assert cpu.calls == [], "a kind the host cannot probe never falls back to the CPU"


def test_probe_device_passes_on_the_probe_refusal(devices: ModuleType) -> None:
    rocm = FakeProbe("rocm", error=devices.DeviceUnavailable("SYNTHETIC: no kfd node"))
    cpu = FakeProbe("cpu", answer=facts(devices, driver=None, runtime=None))
    with pytest.raises(devices.DeviceUnavailable) as info:
        hip = build(devices, HIP_BUILD)
        devices.probe_device("model.device", spec(devices, "rocm", (0,)), hip, {"rocm": rocm, "cpu": cpu})
    assert "SYNTHETIC: no kfd node" in str(info.value)
    assert (rocm.calls, cpu.calls) == (["rocm"], []), "nothing falls back to the CPU"


def test_probe_device_without_a_framework_takes_the_probe_runtime(devices: ModuleType) -> None:
    probe = FakeProbe("rocm", answer=facts(devices))
    built = devices.probe_device("executor.device", spec(devices, "rocm", (0,)), None, {"rocm": probe})
    assert built.runtime == SYNTHETIC_RUNTIME
    assert (built.framework, built.framework_version) == (None, None)
    known = (built.name, built.count, built.memory_bytes, built.driver)
    assert known == ("SYNTHETIC device model", 2, 1024, SYNTHETIC_DRIVER)


def test_probe_device_keeps_unknown_facts_unknown(devices: ModuleType) -> None:
    probe = FakeProbe("rocm", answer=facts(devices, name=None, memory_bytes=None, driver=None, runtime=None))
    hip = build(devices, HIP_BUILD)
    built = devices.probe_device("model.device", spec(devices, "rocm", (0,)), hip, {"rocm": probe})
    assert (built.name, built.memory_bytes, built.driver) == (None, None, None), "None means not read, never a guess"


# ---------------------------------------------------------------------------
# The probe registry and the driver


def test_register_probe_refuses_unknown_kind_and_repeat(devices: ModuleType) -> None:
    probes: dict[str, Any] = {}
    cpu = FakeProbe("cpu")
    devices.register_probe(cpu, probes)
    assert probes == {"cpu": cpu}
    with pytest.raises(ValueError):
        devices.register_probe(FakeProbe("cpu"), probes)
    for kind in ("xpu", "CUDA", ""):
        with pytest.raises(ValueError):
            devices.register_probe(FakeProbe(kind), probes)
    assert probes == {"cpu": cpu}


def test_default_probes_is_a_registry_by_kind(devices: ModuleType) -> None:
    assert isinstance(devices.DEFAULT_PROBES, dict)
    assert set(devices.DEFAULT_PROBES) <= set(record.DEVICE_KINDS)
    for kind, probe in devices.DEFAULT_PROBES.items():
        assert probe.kind == kind


def make_record(driver: str | None, key: str = "executor.device") -> Any:
    """Return a SYNTHETIC rocm DeviceRecord with `driver`."""
    return device_record_class()(
        key=key,
        kind="rocm",
        indices=[0],
        name=None,
        count=1,
        memory_bytes=None,
        driver=driver,
        runtime=None,
        framework=None,
        framework_version=None,
    )


def test_device_driver_joins_distinct_drivers(devices: ModuleType) -> None:
    assert devices.device_driver([]) is None
    assert devices.device_driver([make_record(None), make_record(None, "model.device")]) is None
    assert devices.device_driver([make_record(None), make_record("A-1.0")]) == "A-1.0"
    records = [make_record("B-2.0"), make_record(None), make_record("A-1.0"), make_record("B-2.0")]
    assert devices.device_driver(records) == "B-2.0; A-1.0", "distinct drivers in record order"


# ---------------------------------------------------------------------------
# Placement: lassi/core imports no framework and reads no device file


def imported_roots(tree: ast.AST) -> set[str]:
    """Return the top-level package of every absolute import, import_module call, and __import__ call."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in ("import_module", "__import__") and isinstance(node.args[0].value, str):
                roots.add(node.args[0].value.split(".")[0])
    return roots


def docstring_nodes(tree: ast.AST) -> set[int]:
    """Return the ids of the Constant nodes that are docstrings of the module, a class, or a function."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, *FUNCTION_NODES)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


def core_sources() -> list[Path]:
    """Return every Python file under lassi/core."""
    files = sorted(CORE_DIR.rglob("*.py"))
    assert files, f"no Python files under {CORE_DIR}"
    return files


def test_lassi_core_imports_no_ml_framework() -> None:
    hits = []
    for path in core_sources():
        roots = imported_roots(ast.parse(path.read_bytes().decode("utf-8")))
        hits += [f"{path.relative_to(REPO).as_posix()}: {root}" for root in sorted(roots & FORBIDDEN_IMPORTS)]
    assert not hits, f"lassi/core imports a framework or vendor library: {hits}"


def test_the_import_scan_sees_lazy_imports() -> None:
    source = "def f():\n    import torch.cuda\n    from amdsmi import x\n    importlib.import_module('vllm.engine')\n"
    assert imported_roots(ast.parse(source)) >= {"torch", "amdsmi", "vllm"}


def test_lassi_core_code_names_no_host_device_path() -> None:
    hits = []
    for path in core_sources():
        tree = ast.parse(path.read_bytes().decode("utf-8"))
        skip = docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
                hits += [f"{path.name}:{node.lineno} {p}" for p in DEVICE_PATHS if p in node.value]
    assert not hits, f"device files are read only in lassi/executors and lassi/profilers: {hits}"


def test_the_core_devices_module_reads_no_file() -> None:
    spec = importlib.util.find_spec("lassi.core.devices")
    if spec is None or not spec.origin:
        pytest.fail("lassi.core.devices does not exist yet (task P17.2)")
    tree = ast.parse(Path(spec.origin).read_bytes().decode("ascii"))
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            calls.add(name)
    assert not calls & FILE_CALLS, f"lassi/core/devices.py reads files: {sorted(calls & FILE_CALLS)}"
    assert "access" not in calls, "the core module checks no device node; the probes in lassi/executors do"


# ---------------------------------------------------------------------------
# Readability of the new modules


def module_source(name: str) -> str:
    """Return the ASCII source text of an importable module, failing when it does not exist or is not ASCII."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        pytest.fail(f"{name} does not exist yet (task P17.2)")
    raw = Path(spec.origin).read_bytes()
    assert raw.isascii(), f"{name} has non-ASCII source text"
    assert b"\r" not in raw, f"{name} has a CR"
    return raw.decode("ascii")


def public_defs(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    """Return public top-level classes and functions, plus the public methods of public classes."""
    found: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, *FUNCTION_NODES)) or node.name.startswith("_"):
            continue
        found.append((node.name, node))
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, FUNCTION_NODES) and not item.name.startswith("_"):
                    found.append((f"{node.name}.{item.name}", item))
    return found


def is_typed(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True when every parameter except self or cls, and the return value, are annotated."""
    args = node.args
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    params += [arg for arg in (args.vararg, args.kwarg) if arg is not None]
    params = [param for param in params if param.arg not in ("self", "cls")]
    return node.returns is not None and all(param.annotation is not None for param in params)


@pytest.mark.parametrize("name", NEW_MODULES)
def test_new_modules_are_documented_typed_and_ascii(name: str) -> None:
    source = module_source(name)
    tree = ast.parse(source)
    assert ast.get_docstring(tree), f"{name} has no module docstring"
    undocumented = [qual for qual, node in public_defs(tree) if not ast.get_docstring(node)]
    assert not undocumented, f"{name}: no docstring on {undocumented}"
    untyped = [qual for qual, node in public_defs(tree) if isinstance(node, FUNCTION_NODES) and not is_typed(node)]
    assert not untyped, f"{name}: missing type hints on {untyped}"
    assert "from __future__ import annotations" in source, name


@pytest.mark.parametrize("name", NEW_MODULES)
def test_new_modules_hold_no_project_code_and_no_new_dependency(name: str) -> None:
    source = module_source(name)
    found = [word for word in PROJECT_NAMES if word in source.lower()]
    assert not found, f"{name} names projects or hosts: {found}"
    roots = imported_roots(ast.parse(source))
    third_party = sorted(root for root in roots if root not in sys.stdlib_module_names and root != "lassi")
    assert not third_party, f"{name} imports {third_party}"
