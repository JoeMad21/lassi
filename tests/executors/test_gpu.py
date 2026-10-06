"""Tests for the gpu executor, registered as Executor "gpu" (task P17.5).

Bible: Execution Backends (gpu rows), Sandbox, Component Interfaces
(Executor, device()), Project Recipes (Notes, executor forms), Agent Rules
6, 7, and 13; plans/p17-portable.md, task P17.5, its Constraints, and the
planning decisions "One GPU, one role (P17.5)" and "Placement (Design
Principle 9)"; OQ-002 (nothing opens /dev/kfd, a /dev/dri node, or an
NVIDIA node, a probe or a test included).

The contract these tests fix:

- lassi.executors.gpu registers GpuExecutor as Executor "gpu", with the
  capabilities runs_code, sandboxed, and takes_device and the one config
  key harness. It takes no vendor key: its device section (task P17.2)
  names the vendor, kind cuda for NVIDIA and kind rocm for AMD, with the
  indices, and kind cpu is refused at construction (ValueError naming
  device.kind). It defines no framework(). The keyword settings
  hidden_roots, toolchains, and sandbox are the native executor's; `root`
  (default "/") and `access` (default access(2) for read and write, which
  opens nothing) are the host root its device files are read under and the
  check it asks about a node, as the P17.2 probes take them.
- One table per vendor (VendorTable; NVIDIA_TABLE and AMD_TABLE, TABLES by
  kind), unverified until P17.12: the fixed nodes (nvidiactl, nvidia-uvm,
  and nvidia-uvm-tools; kfd), the /sys directories bound read-only (none;
  the KFD topology), the visibility variables (CUDA_VISIBLE_DEVICES;
  ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES), and the settings
  (CUDA_DEVICE_ORDER=PCI_BUS_ID). Together they are the sandbox's
  GPU_VISIBILITY_NAMES.
- At construction it reads the host's GPU inventory under the root
  (lassi.executors.devices gpu_inventory and gpu_driver), from file
  metadata and procfs or sysfs text only, and checks (check_gpu_nodes) that
  each fixed node and each listed index's node exists and allows read and
  write. AMD: index i is the i-th KFD topology node, in numeric order of the
  node directories, whose gpu_id is not 0; its node is
  /dev/dri/renderD<drm_render_minor> and its name gfx<major><minor in
  hex><stepping in hex> from gfx_target_version. NVIDIA: index i is the
  i-th entry of proc/driver/nvidia/gpus in name order (PCI bus id order);
  its node is /dev/nvidia<Device Minor> and its name the Model line. A
  missing or refused node, an index the inventory lacks, and an index
  without a node raise DeviceUnavailable naming the host path or the index,
  never the root-joined path, and access is asked about the fixed and the
  listed nodes only.
- device() names the kind, each index with its GPU's name, and the driver
  module with its version, as one line of printable ASCII ("name not read"
  and "driver not read" for unknown values). It formats what construction
  read: it opens nothing and starts no process.
- run() runs only through the sandbox, with the native executor's rules
  (the workdir inside the runs root, the hidden roots, the argv, the output
  files, the RunResult). Its SandboxSpec's devices (a DeviceExposure) bind
  the fixed nodes and the listed indices' nodes, each at its own path, and
  the table's /sys directories; its environment is the native executor's
  (PATH, LANG, TMPDIR, OMP_NUM_THREADS) plus the table's visibility
  variables, set to the in-sandbox indices 0,...,n-1 (never the host
  indices), and its settings. Its command is built from
  DEVICE_SETUP_SCRIPT.
- Native and ttsim runs see no GPU node: their specs carry no devices and
  no visibility variable, and their commands keep SETUP_SCRIPT.

Every host root here is SYNTHETIC, written under tmp_path; the device nodes
are empty regular files, so the tests run on Windows, and procfs entries
whose real names hold ':' (PCI bus ids) are written with '-'. No device node
is opened, no sandbox or process starts (fake Sandboxes record each run),
and no value in this module is a measurement.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import importlib
import inspect
import io
import os
import pathlib
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from ttsim_fakes import FakeSandbox as TtsimSandbox
from ttsim_fakes import canned as ttsim_canned
from ttsim_fakes import make_install

import lassi.executors  # noqa: F401  (importing the package registers every executor)
from lassi.core.devices import DeviceUnavailable
from lassi.core.interfaces import Limits, RunResult
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.executors import NativeExecutor, TtsimExecutor
from lassi.executors import sandbox as sandbox_module
from lassi.executors.sandbox import SANDBOX_PATH

LIMITS = Limits(wall_s=2.5, memory_mb=256, cpus=2)
INPUTS = ["--size", "8"]
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
PROJECT_NAMES = ("lassi-repro", "lassi-ee", "lassi-df", "hecbench", "qwen", "wizardcoder", "a100", "mi300x")
CAPABILITIES = frozenset({"runs_code", "sandboxed", "takes_device"})
# The names the gpu executor alone may set (the sandbox's GPU_VISIBILITY_NAMES), and the native executor's program
# environment without its thread count.
GPU_NAMES = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
NATIVE_ENVIRONMENT = {"PATH": SANDBOX_PATH, "LANG": "C.UTF-8", "TMPDIR": "/tmp"}
# The fixed nodes of each vendor's table, relative to /dev, and the AMD table's /sys directory.
CUDA_FIXED = frozenset({"nvidiactl", "nvidia-uvm", "nvidia-uvm-tools"})
ROCM_FIXED = frozenset({"kfd"})
KFD_TOPOLOGY = "/sys/devices/virtual/kfd/kfd/topology"
AMDGPU_VERSION = "SYNTHETIC-6.12.12"
NVIDIA_VERSION = "999.88.77"
NVRM_LINE = f"NVRM version: NVIDIA UNIX x86_64 Kernel Module  {NVIDIA_VERSION}  SYNTHETIC date\n"
# SYNTHETIC KFD topology nodes: directory name -> (gpu_id, drm_render_minor, gfx_target_version). Node 0 is a CPU
# node (gpu_id 0). Numerically "10" comes after "2"; as text it comes before it, so index 1 tells the orders apart.
KFD_NODES: dict[str, tuple[str, int | None, int | None]] = {
    "0": ("0", 0, 0),
    "1": ("11111", 128, 90402),
    "2": ("22222", 129, 90010),
    "10": ("33333", 130, 110000),
}
# The names those GPU nodes decode to (major, minor in hex, stepping in hex; the decoding is unverified until P17.12).
ROCM_NAMES = ("gfx942", "gfx90a", "gfx1100")
# SYNTHETIC NVIDIA GPUs: entry name (a PCI bus id, '-' for ':') -> (Model, Device Minor), written out of name order.
NVIDIA_GPUS: dict[str, tuple[str, int | None]] = {
    "0000-81-00.0": ("SYNTHETIC GPU model B", 0),
    "0000-01-00.0": ("SYNTHETIC GPU model A", 1),
}
ROCM_0 = {"kind": "rocm", "indices": [0]}
CUDA_0 = {"kind": "cuda", "indices": [0]}


# ---------------------------------------------------------------------------
# Modules, SYNTHETIC host roots, and fakes


@pytest.fixture(scope="module")
def gpu() -> ModuleType:
    """Import lassi.executors.gpu, failing each test clearly while it does not exist."""
    try:
        return importlib.import_module("lassi.executors.gpu")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.executors.gpu does not exist yet (task P17.5): {error}")


@pytest.fixture(scope="module")
def probes() -> ModuleType:
    """Return lassi.executors.devices, where the index-to-node mapping lives beside the P17.2 probes."""
    return importlib.import_module("lassi.executors.devices")


@pytest.fixture
def sandbox() -> ModuleType:
    """Return the lassi.executors.sandbox module."""
    return sandbox_module


@pytest.fixture(autouse=True)
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make tmp_path/runs the runs root, set the scratch and home roots beside it, and unset $LASSI_TOOLCHAINS."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(runs))
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LASSI_TOOLCHAINS", raising=False)
    return runs


def attribute(module: ModuleType, name: str) -> Any:
    """Return module.<name>, failing the test clearly while task P17.5 has not added it."""
    if not hasattr(module, name):
        pytest.fail(f"{module.__name__} has no {name} yet (task P17.5)")
    return getattr(module, name)


def write_files(root: Path, files: Mapping[str, str]) -> Path:
    """Write SYNTHETIC files under `root` (relative POSIX path -> text, UTF-8, LF) and return `root`."""
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return root


def kfd_properties(minor: int | None, target: int | None) -> str:
    """Return a SYNTHETIC KFD node properties file: one "name value" line per property, as the kernel prints them."""
    lines = ["cpu_cores_count 0", "simd_count 304", "mem_banks_count 1"]
    if target is not None:
        lines.append(f"gfx_target_version {target}")
    lines += ["vendor_id 4098", "device_id 29856"]
    if minor is not None:
        lines.append(f"drm_render_minor {minor}")
    lines.append("hive_id 0")
    return "".join(f"{line}\n" for line in lines)


def rocm_root(root: Path, nodes: Mapping[str, tuple[str, int | None, int | None]] = KFD_NODES) -> Path:
    """Write a SYNTHETIC AMD host root: dev/kfd, the KFD topology nodes, a render node per GPU minor, and amdgpu."""
    files = {"dev/kfd": "", "dev/dri/card1": "", "sys/module/amdgpu/version": f"{AMDGPU_VERSION}\n"}
    for name, (gpu_id, minor, target) in nodes.items():
        base = f"sys/class/kfd/kfd/topology/nodes/{name}"
        files[f"{base}/gpu_id"] = f"{gpu_id}\n"
        files[f"{base}/properties"] = kfd_properties(minor, target)
        if gpu_id != "0" and minor:
            files[f"dev/dri/renderD{minor}"] = ""
    return write_files(root, files)


def information(model: str, minor: int | None) -> str:
    """Return a SYNTHETIC proc/driver/nvidia/gpus/<bus id>/information file in the driver's layout."""
    lines = [f"Model: \t\t {model}", "IRQ:   \t\t 99", "GPU UUID: \t GPU-SYNTHETIC", "Bus Type: \t PCIe"]
    lines.append("Bus Location: \t 0000:01:00.0")
    if minor is not None:
        lines.append(f"Device Minor: \t {minor}")
    lines.append("GPU Firmware: \t N/A")
    return "".join(f"{line}\n" for line in lines)


def cuda_root(root: Path, gpus: Mapping[str, tuple[str, int | None]] = NVIDIA_GPUS) -> Path:
    """Write a SYNTHETIC NVIDIA host root: the fixed nodes, one entry and one node per GPU, and the version file."""
    files = {f"dev/{name}": "" for name in sorted(CUDA_FIXED)}
    files["dev/nvidia-modeset"] = ""
    files["proc/driver/nvidia/version"] = NVRM_LINE + "GCC version:  SYNTHETIC\n"
    for name, (model, minor) in gpus.items():
        files[f"proc/driver/nvidia/gpus/{name}/information"] = information(model, minor)
        if minor is not None:
            files[f"dev/nvidia{minor}"] = ""
    return write_files(root, files)


def host_root(kind: str, root: Path) -> Path:
    """Write the SYNTHETIC host root of `kind` (rocm or cuda) at `root`."""
    return rocm_root(root) if kind == "rocm" else cuda_root(root)


class Access:
    """A stand-in access check: it records each path asked about, relative to `root`, and denies those in `deny`."""

    def __init__(self, root: Path, deny: Iterable[str] = ()) -> None:
        """Answer True except for the relative POSIX paths in `deny`."""
        self.root = Path(root)
        self.deny = set(deny)
        self.asked: list[str] = []

    def __call__(self, path: Path) -> bool:
        """Record the path (relative to the root, which it must lie under) and answer."""
        relative = Path(path).relative_to(self.root).as_posix()
        self.asked.append(relative)
        return relative not in self.deny


def posix(path: object) -> str:
    """Return a path as POSIX text: its as_posix() when it has one, else str()."""
    method = getattr(path, "as_posix", None)
    return method() if callable(method) else str(path)


def pairs(binds: Iterable[Any]) -> list[tuple[str, str]]:
    """Return each DeviceBind as (source, target) POSIX text."""
    return [(posix(bind.source), posix(bind.target)) for bind in binds]


def dev_relative(node: object) -> str:
    """Return a node path relative to /dev: "/dev/dri/renderD128" and "dri/renderD128" both give the latter."""
    text = posix(node)
    return text[len("/dev/") :] if text.startswith("/dev/") else text


def build(gpu: ModuleType, kind: str, indices: Sequence[int], root: Path, access: Any = None, **settings: Any) -> Any:
    """Return GpuExecutor(device={kind, indices}, root=root, access=access or Access(root), **settings)."""
    section = {"kind": kind, "indices": list(indices)}
    return gpu.GpuExecutor(device=section, root=root, access=access or Access(root), **settings)


@dataclass
class FakeSandbox:
    """A stand-in for lassi.executors.sandbox.Sandbox: it records each run, applies `effect`, and returns `result`."""

    result: Any
    effect: Callable[[Any], None] | None = None
    calls: list[tuple[Any, list[str], Limits]] = field(default_factory=list)

    def run(self, spec: Any, argv: Sequence[str], limits: Limits) -> Any:
        """Record the call, apply the effect (as a program would write files), and return the canned result."""
        self.calls.append((spec, list(argv), limits))
        if self.effect is not None:
            self.effect(spec)
        return self.result


class ExplodingSandbox:
    """A stand-in sandbox: any use of it fails the test, since naming a device runs nothing."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the sandbox was used ({name})")


def canned(**overrides: Any) -> Any:
    """Return a SandboxResult for a normal exit with status 0, with any field overridden."""
    values: dict[str, Any] = {"returncode": 0, "stdout": "", "stderr": "", "wall_s": 0.25, "hang": False}
    values.update({"killed": False, **overrides})
    return sandbox_module.SandboxResult(**values)


def make_artifact(runs: Path) -> Path:
    """Return a PLACEHOLDER artifact in a fresh build directory under the runs root; it is never executed."""
    workdir = runs / "attempt00" / "build"
    workdir.mkdir(parents=True, exist_ok=True)
    artifact = workdir / "main"
    artifact.write_bytes(b"PLACEHOLDER artifact, never executed\n")
    return artifact


def same_path(first: object, second: object) -> bool:
    """Return True when two paths name the same place once resolved."""
    return os.path.realpath(os.fspath(first)) == os.path.realpath(os.fspath(second))  # type: ignore[arg-type]


def script_of(command: list[str]) -> str:
    """Return the script element of a sandbox command: the element after "sh", "-c"."""
    start = command.index("sh")
    assert command[start + 1] == "-c", command
    return command[start + 2]


def is_under(path: str, directory: Path) -> bool:
    """Return True when the normalized `path` is `directory` or lies under it."""
    prefix = os.path.normcase(os.path.abspath(directory))
    return path == prefix or path.startswith(prefix + os.sep)


def guard_opens(monkeypatch: pytest.MonkeyPatch, roots: Sequence[Path]) -> list[str]:
    """Record every file opened under `roots`, and fail any open under a root's dev directory; return the record.

    builtins.open, io.open, os.open, and pathlib.Path.open are wrapped (Python
    3.10's pathlib opens through its own accessor). An open under <root>/dev
    raises AssertionError and is recorded too, so code that catches the error
    still fails the test.
    """
    opened: list[str] = []

    def note(target: Any) -> None:
        if isinstance(target, int):
            return
        path = os.path.normcase(os.path.abspath(os.fspath(target)))
        if any(is_under(path, root) for root in roots):
            opened.append(path)
        if any(is_under(path, root / "dev") for root in roots):
            raise AssertionError(f"a device node was opened: {path}")

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


def trap_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any process start fail the test: subprocess.Popen and os.system raise AssertionError."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"a process was started: {args!r}")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(os, "system", refuse)


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
    params += [a for a in (args.vararg, args.kwarg) if a is not None]
    params = [p for p in params if p.arg not in ("self", "cls")]
    return node.returns is not None and all(p.annotation is not None for p in params)


def module_problems(module: ModuleType) -> tuple[str, list[str], list[str]]:
    """Return a module's docstring and its public definitions without a docstring or without full type hints.

    The file must be plain ASCII with LF newlines and name no project (Agent Rule 3).
    """
    raw = Path(str(module.__file__)).read_bytes()
    assert raw.isascii() and b"\r" not in raw, module.__name__
    source = raw.decode("ascii")
    assert not [name for name in PROJECT_NAMES if name in source.lower()], "Agent Rule 3: no project in lassi/"
    tree = ast.parse(source)
    missing = [name for name, node in public_defs(tree) if not ast.get_docstring(node)]
    untyped = [name for name, node in public_defs(tree) if isinstance(node, FUNCTION_NODES) and not is_typed(node)]
    return ast.get_docstring(tree) or "", missing, untyped


# ---------------------------------------------------------------------------
# Registration, construction, and the module


def test_registered_as_executor_gpu_with_its_capabilities(gpu: ModuleType) -> None:
    entry = DEFAULT_REGISTRY.get("Executor", "gpu")
    assert entry.factory is gpu.GpuExecutor
    assert entry.capabilities == CAPABILITIES
    assert entry.config_keys == frozenset({"harness"}), "no vendor key: the device section's kind names the vendor"
    assert gpu.GpuExecutor.name == "gpu"
    assert not callable(getattr(gpu.GpuExecutor, "framework", None)), "it runs a built program on no framework"


def test_package_imports_and_re_exports_gpu(gpu: ModuleType) -> None:
    import lassi.executors as package

    assert package.gpu is gpu, "importing lassi.executors alone registers the executor"
    assert package.GpuExecutor is gpu.GpuExecutor
    assert {"GpuExecutor", "gpu"} <= set(package.__all__)
    assert '"gpu"' in (package.__doc__ or ""), "the package docstring lists every executor it registers"


def test_the_constructor_takes_the_device_section_and_keyword_settings(gpu: ModuleType, probes: ModuleType) -> None:
    params = inspect.signature(gpu.GpuExecutor).parameters
    assert set(params) == {"device", "harness", "hidden_roots", "toolchains", "sandbox", "root", "access"}
    assert all(param.kind is inspect.Parameter.KEYWORD_ONLY for param in params.values())
    assert params["device"].default is inspect.Parameter.empty, "a device is always named; there is no default"
    assert all(params[name].default is None for name in ("harness", "hidden_roots", "toolchains", "sandbox"))
    assert Path(params["root"].default) == Path("/")
    assert params["access"].default is probes._accessible, "access(2) for read and write, which opens nothing"
    assert list(inspect.signature(gpu.GpuExecutor.run).parameters) == ["self", "artifact", "inputs", "limits"]


@pytest.mark.parametrize(
    "section",
    [
        pytest.param({"kind": "cpu"}, id="cpu"),
        pytest.param({"kind": "tpu", "indices": [0]}, id="unknown-kind"),
        pytest.param({"kind": "rocm"}, id="no-indices"),
        pytest.param({"kind": "rocm", "indices": [0, 0]}, id="repeated-index"),
    ],
)
def test_a_section_that_names_no_gpu_is_refused_at_construction(
    gpu: ModuleType, tmp_path: Path, section: dict[str, Any]
) -> None:
    # Before any file is read: the root is empty, so a refusal for a missing node would be a DeviceUnavailable.
    empty = tmp_path / "empty"
    access = Access(empty)
    with pytest.raises(ValueError) as caught:
        gpu.GpuExecutor(device=section, root=empty, access=access)
    message = str(caught.value)
    assert "device" in message, message
    if section["kind"] == "cpu":
        assert "device.kind" in message and "cpu" in message, message
    assert access.asked == []


def test_one_table_per_vendor_with_nodes_sys_dirs_and_visibility_variables(gpu: ModuleType) -> None:
    tables = attribute(gpu, "TABLES")
    assert set(tables) == {"cuda", "rocm"}
    assert tables["cuda"] is gpu.NVIDIA_TABLE and tables["rocm"] is gpu.AMD_TABLE
    assert dataclasses.is_dataclass(gpu.VendorTable)
    nvidia, amd = gpu.NVIDIA_TABLE, gpu.AMD_TABLE
    assert isinstance(nvidia, gpu.VendorTable) and isinstance(amd, gpu.VendorTable)
    assert (nvidia.kind, amd.kind) == ("cuda", "rocm")
    assert {dev_relative(node) for node in nvidia.fixed_nodes} == CUDA_FIXED, "no modeset, MIG, or nvswitch node"
    assert {dev_relative(node) for node in amd.fixed_nodes} == ROCM_FIXED, "no udmabuf node"
    assert [posix(path) for path in nvidia.sys_dirs] == []
    assert [posix(path) for path in amd.sys_dirs] == [KFD_TOPOLOGY]
    assert set(nvidia.visibility) == {"CUDA_VISIBLE_DEVICES"}
    assert dict(nvidia.settings) == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, (
        "CUDA's order is the PCI order the mapping uses"
    )
    assert set(amd.visibility) == {"ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"}
    assert dict(amd.settings) == {}
    names = {*nvidia.visibility, *nvidia.settings, *amd.visibility, *amd.settings}
    assert names == set(attribute(sandbox_module, "GPU_VISIBILITY_NAMES"))


def test_the_gpu_module_is_documented_typed_ascii_and_cites_its_sources(gpu: ModuleType) -> None:
    doc, missing, untyped = module_problems(gpu)
    assert not missing, missing
    assert not untyped, untyped
    # The tables are unverified until P17.12 measures a GPU host, and each value read from upstream source cites the
    # line of code that sets or compares it (plans/p17-portable.md, Working practices).
    terms = ("unverified", "P17.12", "OQ-002", "topology.c:", "flag.h:", "flags.hpp:", "nvidia-modprobe-utils")
    for term in terms:
        assert term in doc, term


def test_the_devices_module_documents_the_index_to_node_mapping(probes: ModuleType) -> None:
    for name in ("GpuEntry", "gpu_inventory", "gpu_driver", "check_gpu_nodes"):
        attribute(probes, name)
    doc, missing, untyped = module_problems(probes)
    assert not missing, missing
    assert not untyped, untyped
    for term in ("drm_render_minor", "Device Minor", "gfx_target_version", "P17.12"):
        assert term in doc, term
    assert [item.name for item in dataclasses.fields(probes.GpuEntry)] == ["index", "node", "name"]


# ---------------------------------------------------------------------------
# The inventory: index to node and name, from file metadata and procfs or sysfs text


def test_rocm_inventory_maps_indices_to_render_nodes_in_numeric_node_order(probes: ModuleType, tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "host")
    inventory = list(attribute(probes, "gpu_inventory")("rocm", root))
    assert [entry.index for entry in inventory] == [0, 1, 2], "the CPU node (gpu_id 0) is no GPU"
    assert [dev_relative(entry.node) for entry in inventory] == ["dri/renderD128", "dri/renderD129", "dri/renderD130"]
    assert [entry.name for entry in inventory] == list(ROCM_NAMES)


def test_cuda_inventory_maps_indices_to_minors_in_bus_order(probes: ModuleType, tmp_path: Path) -> None:
    root = cuda_root(tmp_path / "host")
    inventory = list(attribute(probes, "gpu_inventory")("cuda", root))
    assert [entry.index for entry in inventory] == [0, 1]
    assert [dev_relative(entry.node) for entry in inventory] == ["nvidia1", "nvidia0"], "name order, not minor order"
    assert [entry.name for entry in inventory] == ["SYNTHETIC GPU model A", "SYNTHETIC GPU model B"]


def test_inventory_values_that_cannot_be_read_are_none(probes: ModuleType, tmp_path: Path) -> None:
    inventory = attribute(probes, "gpu_inventory")
    nodes = {"0": ("0", 0, 0), "1": ("11111", None, None), "2": ("22222", 0, 90402)}
    found = list(inventory("rocm", rocm_root(tmp_path / "rocm", nodes)))
    assert [(entry.index, entry.node, entry.name) for entry in found] == [(0, None, None), (1, None, "gfx942")]
    gpus = {"0000-01-00.0": ("SYNTHETIC GPU model A", None)}
    found = list(inventory("cuda", cuda_root(tmp_path / "cuda", gpus)))
    assert [(entry.index, entry.node, entry.name) for entry in found] == [(0, None, "SYNTHETIC GPU model A")]
    assert list(inventory("rocm", tmp_path / "empty")) == [] and list(inventory("cuda", tmp_path / "empty")) == []


def test_gpu_driver_reads_what_the_probes_read(probes: ModuleType, tmp_path: Path) -> None:
    driver = attribute(probes, "gpu_driver")
    rocm, cuda = rocm_root(tmp_path / "rocm"), cuda_root(tmp_path / "cuda")
    assert driver("rocm", rocm) == AMDGPU_VERSION == probes.RocmProbe(root=rocm, access=Access(rocm)).probe().driver
    assert driver("cuda", cuda) == NVIDIA_VERSION == probes.CudaProbe(root=cuda, access=Access(cuda)).probe().driver
    assert driver("rocm", tmp_path / "empty") is None and driver("cuda", tmp_path / "empty") is None


# ---------------------------------------------------------------------------
# Refusals at construction: a node this host lacks or refuses, or an index it lacks

# Each case: kind, indices, files removed from the SYNTHETIC root, nodes the access check denies, and the text the
# message names (a host path, or the index).
REFUSALS: dict[str, tuple[str, list[int], list[str], list[str], str]] = {
    "render-node-missing": ("rocm", [1], ["dev/dri/renderD129"], [], "/dev/dri/renderD129"),
    "render-node-denied": ("rocm", [1], [], ["dev/dri/renderD129"], "/dev/dri/renderD129"),
    "kfd-missing": ("rocm", [0], ["dev/kfd"], [], "/dev/kfd"),
    "kfd-denied": ("rocm", [0], [], ["dev/kfd"], "/dev/kfd"),
    "nvidiactl-missing": ("cuda", [1], ["dev/nvidiactl"], [], "/dev/nvidiactl"),
    "nvidia-uvm-missing": ("cuda", [0], ["dev/nvidia-uvm"], [], "/dev/nvidia-uvm"),
    "nvidia-uvm-tools-denied": ("cuda", [0], [], ["dev/nvidia-uvm-tools"], "/dev/nvidia-uvm-tools"),
    "minor-node-missing": ("cuda", [1], ["dev/nvidia0"], [], "/dev/nvidia0"),
    "rocm-index-past-the-inventory": ("rocm", [0, 3], [], [], "index 3"),
    "cuda-index-past-the-inventory": ("cuda", [2], [], [], "index 2"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_missing_or_denied_node_or_index_is_refused_naming_the_host_path(
    gpu: ModuleType, tmp_path: Path, case: str
) -> None:
    kind, indices, removed, denied, named = REFUSALS[case]
    root = host_root(kind, tmp_path / "host")
    for relative in removed:
        (root / relative).unlink()
    access = Access(root, deny=denied)
    with pytest.raises(DeviceUnavailable) as caught:
        build(gpu, kind, indices, root, access)
    message = str(caught.value)
    assert named in message, message
    assert str(root) not in message and root.as_posix() not in message, f"the root-joined path is named: {message}"
    assert all((root / relative).exists() for relative in access.asked), "access is asked only about existing nodes"


@pytest.mark.parametrize("minor", [None, 0], ids=["no-render-minor", "render-minor-zero"])
def test_an_index_without_a_node_is_refused_naming_the_index(
    gpu: ModuleType, tmp_path: Path, minor: int | None
) -> None:
    nodes = {**KFD_NODES, "2": ("22222", minor, 90010)}
    root = rocm_root(tmp_path / "host", nodes)
    with pytest.raises(DeviceUnavailable) as caught:
        build(gpu, "rocm", [1], root)
    assert "index 1" in str(caught.value), str(caught.value)
    assert build(gpu, "rocm", [0, 2], root).device().startswith("gpu (rocm): index 0 gfx942, index 2 gfx1100;")


def test_only_the_fixed_and_the_listed_nodes_are_asked_about(gpu: ModuleType, tmp_path: Path) -> None:
    rocm = rocm_root(tmp_path / "rocm")
    access = Access(rocm)
    build(gpu, "rocm", [1], rocm, access)
    assert sorted(set(access.asked)) == ["dev/dri/renderD129", "dev/kfd"], access.asked
    cuda = cuda_root(tmp_path / "cuda")
    access = Access(cuda)
    build(gpu, "cuda", [0], cuda, access)
    expected = sorted(["dev/nvidiactl", "dev/nvidia-uvm", "dev/nvidia-uvm-tools", "dev/nvidia1"])
    assert sorted(set(access.asked)) == expected, access.asked


def test_the_default_access_check_asks_access_2_for_read_and_write(
    gpu: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = rocm_root(tmp_path / "host")
    asked: list[tuple[str, int]] = []

    def access(path: Any, mode: int, *args: Any, **kwargs: Any) -> bool:
        asked.append((Path(path).relative_to(root).as_posix(), mode))
        return True

    monkeypatch.setattr(os, "access", access)
    gpu.GpuExecutor(device={"kind": "rocm", "indices": [0]}, root=root)
    assert sorted(asked) == [("dev/dri/renderD128", os.R_OK | os.W_OK), ("dev/kfd", os.R_OK | os.W_OK)]


# ---------------------------------------------------------------------------
# device(): from procfs or sysfs, starting no process


def test_device_names_each_index_its_gpu_and_the_driver(gpu: ModuleType, tmp_path: Path) -> None:
    rocm = rocm_root(tmp_path / "rocm")
    every = "index 0 gfx942, index 1 gfx90a, index 2 gfx1100"
    assert build(gpu, "rocm", [0, 1, 2], rocm).device() == f"gpu (rocm): {every}; amdgpu driver {AMDGPU_VERSION}"
    assert build(gpu, "rocm", [1], rocm).device() == f"gpu (rocm): index 1 gfx90a; amdgpu driver {AMDGPU_VERSION}"
    cuda = cuda_root(tmp_path / "cuda")
    expected = f"gpu (cuda): index 0 SYNTHETIC GPU model A; nvidia driver {NVIDIA_VERSION}"
    assert build(gpu, "cuda", [0], cuda).device() == expected


def test_device_with_unread_values_says_so_in_plain_ascii(gpu: ModuleType, tmp_path: Path) -> None:
    root = rocm_root(tmp_path / "rocm", {"0": ("0", 0, 0), "1": ("11111", 128, None)})
    (root / "sys" / "module" / "amdgpu" / "version").unlink()
    assert build(gpu, "rocm", [0], root).device() == "gpu (rocm): index 0 name not read; amdgpu driver not read"
    # A SYNTHETIC model with blanks, a tab, a letter that is not ASCII, and a BEL control character.
    model = "  SYNTHETIC \t GPU \N{GREEK SMALL LETTER MU} \x07 9000 "
    cuda = cuda_root(tmp_path / "cuda", {"0000-01-00.0": (model, 0)})
    (cuda / "proc" / "driver" / "nvidia" / "version").unlink()
    device = build(gpu, "cuda", [0], cuda).device()
    assert device == "gpu (cuda): index 0 SYNTHETIC GPU \\u03bc \\x07 9000; nvidia driver not read"
    assert device.isascii() and device.isprintable() and device == device.strip()


def test_construction_and_device_open_no_device_node_and_start_no_process(
    gpu: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rocm, cuda = rocm_root(tmp_path / "rocm"), cuda_root(tmp_path / "cuda")
    opened = guard_opens(monkeypatch, [rocm, cuda])
    trap_processes(monkeypatch)
    build(gpu, "rocm", [0, 1, 2], rocm, sandbox=ExplodingSandbox()).device()
    build(gpu, "cuda", [0, 1], cuda, sandbox=ExplodingSandbox()).device()
    for kind, root, denied in (("rocm", rocm, "dev/dri/renderD129"), ("cuda", cuda, "dev/nvidia0")):
        with pytest.raises(DeviceUnavailable):
            build(gpu, kind, [1], root, Access(root, deny=[denied]))
    nodes = [path for path in opened if is_under(path, rocm / "dev") or is_under(path, cuda / "dev")]
    assert nodes == [], f"a device node was opened: {nodes}"


def test_device_formats_what_construction_read(
    gpu: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = rocm_root(tmp_path / "rocm")
    executor = build(gpu, "rocm", [0], root, sandbox=ExplodingSandbox())
    expected = executor.device()
    opened = guard_opens(monkeypatch, [root])
    trap_processes(monkeypatch)
    assert executor.device() == expected
    assert opened == [], f"device() read files again: {opened}"


# ---------------------------------------------------------------------------
# run(): only through the sandbox, with the table's exposure and variables


def test_run_exposes_the_fixed_nodes_and_only_the_listed_indices(
    gpu: ModuleType, tmp_path: Path, runs_root: Path
) -> None:
    artifact = make_artifact(runs_root)
    rocm = rocm_root(tmp_path / "rocm")
    fake = FakeSandbox(result=canned())
    build(gpu, "rocm", [1], rocm, sandbox=fake).run(artifact, INPUTS, LIMITS)
    ((spec, argv, limits),) = fake.calls
    assert isinstance(spec, sandbox_module.SandboxSpec)
    assert isinstance(spec.devices, attribute(sandbox_module, "DeviceExposure"))
    nodes = pairs(spec.devices.nodes)
    assert nodes == [("/dev/kfd", "/dev/kfd"), ("/dev/dri/renderD129", "/dev/dri/renderD129")], "never index 0's"
    assert pairs(spec.devices.sys_dirs) == [(KFD_TOPOLOGY, KFD_TOPOLOGY)]
    command = sandbox_module.sandbox_command(spec, argv, limits)
    assert script_of(command) == attribute(sandbox_module, "DEVICE_SETUP_SCRIPT")
    cuda = cuda_root(tmp_path / "cuda")
    fake = FakeSandbox(result=canned())
    build(gpu, "cuda", [0], cuda, sandbox=fake).run(artifact, INPUTS, LIMITS)
    spec = fake.calls[0][0]
    nodes = pairs(spec.devices.nodes)
    assert {source for source, _target in nodes[:3]} == {f"/dev/{name}" for name in CUDA_FIXED}, "fixed nodes first"
    assert nodes[3:] == [("/dev/nvidia1", "/dev/nvidia1")], "index 0 is the first entry in bus order, minor 1"
    assert all(source == target for source, target in nodes), "each node is bound at its own path"
    assert pairs(spec.devices.sys_dirs) == []


@pytest.mark.parametrize(
    ("kind", "indices", "expected"),
    [
        pytest.param("rocm", [2], {"ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0"}, id="rocm-one"),
        pytest.param("rocm", [2, 0], {"ROCR_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "0,1"}, id="rocm-two"),
        pytest.param("cuda", [1], {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, id="cuda-one"),
        pytest.param("cuda", [1, 0], {"CUDA_VISIBLE_DEVICES": "0,1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, id="cuda-two"),
    ],
)
def test_run_sets_the_vendor_visibility_variables_to_the_sandbox_indices(
    gpu: ModuleType,
    tmp_path: Path,
    runs_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    indices: list[int],
    expected: dict[str, str],
) -> None:
    # Only the listed nodes are bound, so the runtime numbers the listed GPUs from 0 inside; a caller's own
    # visibility variables never reach the program.
    for name in GPU_NAMES:
        monkeypatch.setenv(name, "7")
    root = host_root(kind, tmp_path / "host")
    fake = FakeSandbox(result=canned())
    build(gpu, kind, indices, root, sandbox=fake).run(make_artifact(runs_root), INPUTS, LIMITS)
    spec = fake.calls[0][0]
    assert dict(spec.environment) == {**NATIVE_ENVIRONMENT, "OMP_NUM_THREADS": str(LIMITS.cpus), **expected}
    assert len(spec.devices.nodes) == {"rocm": 1, "cuda": 3}[kind] + len(indices)


def test_run_keeps_the_native_workdir_root_and_result_rules(gpu: ModuleType, tmp_path: Path, runs_root: Path) -> None:
    root = rocm_root(tmp_path / "rocm")
    harness, toolchains = tmp_path / "assets" / "harness", tmp_path / "toolchains"
    artifact = make_artifact(runs_root)

    def writes_output(spec: Any) -> None:
        """Write a new file in the workdir, as a program would."""
        (Path(spec.workdir) / "out.bin").write_bytes(b"SYNTHETIC output\n")

    result = canned(returncode=3, stdout="o", stderr="e", stdout_truncated=True, workdir_incomplete=True)
    fake = FakeSandbox(result=result, effect=writes_output)
    settings = {"sandbox": fake, "harness": str(harness), "toolchains": str(toolchains)}
    ran = build(gpu, "rocm", [0], root, **settings).run(artifact, INPUTS, LIMITS)
    ((spec, argv, limits),) = fake.calls
    assert same_path(spec.workdir, artifact.parent) and same_path(argv[0], artifact) and argv[1:] == INPUTS
    assert limits == LIMITS
    assert same_path(spec.harness, harness) and same_path(spec.toolchains, toolchains)
    expected_roots = [os.path.realpath(tmp_path / name) for name in ("scratch", "home", "runs")]
    assert [os.path.realpath(root) for root in spec.hidden_roots] == expected_roots
    assert isinstance(ran, RunResult)
    assert (ran.exit_code, ran.hang, ran.stdout, ran.stderr) == (3, False, "o", "e")
    assert (ran.stdout_truncated, ran.stderr_truncated, ran.workdir_incomplete) == (True, False, True)
    assert list(ran.output_files) == ["out.bin"]
    outside = tmp_path / "elsewhere" / "main"
    outside.parent.mkdir()
    outside.write_bytes(b"PLACEHOLDER artifact, never executed\n")
    fake = FakeSandbox(result=canned())
    executor = build(gpu, "rocm", [0], root, sandbox=fake)
    for artifact_path, inputs in ((Path("relative") / "main", INPUTS), (outside, INPUTS), (artifact, "--size 8")):
        with pytest.raises(ValueError):
            executor.run(artifact_path, inputs, LIMITS)
    assert fake.calls == [], "every refusal comes before the sandbox runs"


# ---------------------------------------------------------------------------
# Native and ttsim runs see no GPU node


def test_native_and_ttsim_specs_carry_no_devices(
    tmp_path: Path, runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in GPU_NAMES:
        monkeypatch.setenv(name, "0")
    artifact = make_artifact(runs_root)
    native = FakeSandbox(result=canned())
    NativeExecutor(sandbox=native).run(artifact, INPUTS, LIMITS)  # type: ignore[arg-type]
    install = make_install(tmp_path, monkeypatch)
    ttsim = TtsimSandbox(results=[ttsim_canned()])
    TtsimExecutor(toolchains=str(install.root), sandbox=ttsim).run(artifact, INPUTS, LIMITS)  # type: ignore[arg-type]
    for spec in (native.calls[0][0], ttsim.calls[0].spec):
        assert getattr(spec, "devices", None) is None, "only the gpu executor sets an exposure"
        assert not set(spec.environment or {}) & set(GPU_NAMES)
        assert script_of(sandbox_module.sandbox_command(spec, ["./main"], LIMITS)) == sandbox_module.SETUP_SCRIPT


# ---------------------------------------------------------------------------
# The recipe form: the device section names the vendor


def never_built(name: str, capabilities: Iterable[str]) -> type:
    """Return a fake component class with the class attributes the registry reads; constructing one fails."""

    def refuse(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"component {name} was constructed while loading a recipe")

    namespace = {"__doc__": f"Fake {name}; never built.", "name": name, "capabilities": frozenset(capabilities)}
    return type(f"Fake_{name}", (), {**namespace, "config_keys": frozenset(), "__init__": refuse})


def load(gpu: ModuleType, tmp_path: Path, executor: Mapping[str, Any], name: str) -> Any:
    """Write a standalone recipe with `executor` and load it against a registry holding the real gpu executor."""
    registry = Registry()
    registry.register("Executor", "gpu", gpu.GpuExecutor)
    registry.register("Executor", "plainexec", never_built("plainexec", {"runs_code", "sandboxed"}))
    registry.register("LLMBackend", "plainllm", never_built("plainllm", {"chat"}))
    registry.register("Stage", "generate", never_built("generate", ()))
    data = {
        "llm": {"sampling": {"temperature": 0.2, "top_p": 0.9}},
        "loop": {"max_corrections": 3},
        "trials": {"n": 1},
        "runs_root": "fixture-runs",
        "sandbox": {"network": False, "wall_s": 60, "mem_gb": 4},
        "report": {"trial_md": True, "parquet": True},
        "bench": {"suite": "fixture-suite", "split": "eval"},
        "directions": [{"source": "omp", "target": "cuda"}],
        "stages": ["generate"],
        "executor": dict(executor),
        "model": {"backend": "plainllm", "id": "fixture-model"},
    }
    path = tmp_path / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    return load_recipe(path, roots=[tmp_path], registry=registry)


def test_a_recipe_names_the_gpu_by_its_device_section_with_no_vendor_or_host_key(
    gpu: ModuleType, tmp_path: Path
) -> None:
    single = load(gpu, tmp_path, {"kind": "gpu", "device": ROCM_0}, "single")
    (executor,) = [item for item in single.bindings if item.interface == "Executor"]
    assert dict(executor.config) == {"device": ROCM_0}
    per_language = load(gpu, tmp_path, {"cuda": {"kind": "gpu", "device": CUDA_0}, "omp": "plainexec"}, "per-language")
    found = {item.where: dict(item.config) for item in per_language.bindings if item.interface == "Executor"}
    assert found == {"executor.cuda.kind": {"device": CUDA_0}, "executor.omp": {}}
    refusals = [
        ({"kind": "gpu"}, "executor.device"),
        ({"cuda": "gpu", "omp": "plainexec"}, "executor.cuda.device"),
        ({"kind": "gpu", "vendor": "amd", "device": ROCM_0}, "vendor"),
        ({"kind": "gpu", "host": "fixture-host", "device": ROCM_0}, "host"),
    ]
    for number, (section, named) in enumerate(refusals):
        with pytest.raises(RecipeError) as caught:
            load(gpu, tmp_path, section, f"refused-{number}")
        assert named in str(caught.value), str(caught.value)
