"""Tests for the lassi.profilers package as a whole (task P17.7): registration, imports, and module hygiene.

Bible: Design Principle 9 (vendor code stays behind interfaces; device
files and telemetry are read only in lassi/executors and lassi/profilers),
Repository Layout (profilers/), Readability Standards, Agent Rules 3 and 15;
plans/PHASE-NOTES.md, P17 (the registration rule); plans/p17-portable.md,
Constraints (OQ-002: no SMI tool starts and no GPU node opens).

The contract these tests fix:

- Importing lassi.profilers registers exactly the Profilers timing, nvml,
  and rocm_smi, and imports no vendor library: pynvml is imported only when
  an nvml source is built, so the package registers all three when pynvml
  cannot be imported at all. Importing lassi.core.runner registers them too
  (the runner binds `profiler`), and neither it nor lassi.cli imports a
  vendor library.
- At module level, every module under lassi/profilers imports only the
  standard library and lassi. Only lassi/profilers/nvml.py imports a vendor
  library, pynvml, and only inside a function; no module imports another
  vendor or framework package (the AMD SMI library among them), subprocess,
  or another module that starts processes, or calls os.system, os.popen,
  or an exec, spawn, or fork function.
- No code constant (docstrings aside) names an SMI tool: rocm-smi, amd-smi,
  rocminfo, or nvidia-smi.
- No file under lassi/core imports threading: the sampling thread lives in
  lassi/profilers/power.py.
- Every module is ASCII with LF and starts its code with `from __future__
  import annotations`; it has a module docstring, documents and type-hints
  its public classes, functions, and methods, keeps each function within
  60 lines of code after its docstring (Readability Standards: under about
  60 lines), and names no project or host (Agent Rule 3).

No value in this module is a measurement.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from profiler_fakes import profilers_module

REPO = Path(__file__).resolve().parents[2]
PROFILERS_DIR = REPO / "lassi" / "profilers"
CORE_DIR = REPO / "lassi" / "core"
EXPECTED_MODULES = ("__init__.py", "nvml.py", "power.py", "rocm_smi.py", "timing.py")
REGISTERED = ["nvml", "rocm_smi", "timing"]
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
MAX_FUNCTION_LINES = 60
# Vendor libraries and frameworks no import of lassi.profilers, the runner, or the CLI may load.
VENDOR_MODULES = ("pynvml", "amdsmi", "rocm_smi", "pyrsmi", "nvidia_smi", "pyamdgpuinfo", "torch")
# Packages no profiler module may import at all, at any level.
FORBIDDEN_ANYWHERE = frozenset(
    {"amdsmi", "rocm_smi", "pyrsmi", "nvidia_smi", "pyamdgpuinfo", "torch", "cupy", "pycuda", "cuda"}
)
# Modules that start processes; no profiler module imports one.
PROCESS_MODULES = frozenset({"subprocess", "multiprocessing", "pty", "asyncio", "concurrent"})
# os functions that start or replace processes.
PROCESS_CALLS = frozenset(
    {"system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp", "startfile"}
    | {f"exec{suffix}" for suffix in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")}
    | {f"spawn{suffix}" for suffix in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")}
)
SMI_TOOLS = ("rocm-smi", "amd-smi", "rocminfo", "nvidia-smi")
PROJECT_NAMES = ("lassi-repro", "lassi-ee", "lassi-df", "hecbench", "qwen", "wizardcoder", "a100", "mi300x", "gpt-oss")


def python_says(code: str) -> Any:
    """Run `code` in a fresh interpreter from the repository root and return the JSON it prints."""
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def profiler_sources() -> list[Path]:
    """Return every Python file under lassi/profilers, failing clearly while the package does not exist."""
    profilers_module()
    files = sorted(PROFILERS_DIR.glob("*.py"))
    assert files, f"no Python files under {PROFILERS_DIR}"
    return files


def parsed(path: Path) -> ast.Module:
    """Return the AST of an ASCII source file."""
    return ast.parse(path.read_bytes().decode("ascii"), filename=str(path))


def imports(tree: ast.AST) -> list[tuple[str, bool]]:
    """Return (top-level package, inside a function) for every import, import_module, and __import__ call."""
    found: list[tuple[str, bool]] = []

    def visit(node: ast.AST, in_function: bool) -> None:
        if isinstance(node, ast.Import):
            found.extend((alias.name.split(".")[0], in_function) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module.split(".")[0], in_function))
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in ("import_module", "__import__") and isinstance(node.args[0].value, str):
                found.append((node.args[0].value.split(".")[0], in_function))
        inside = in_function or isinstance(node, (*FUNCTION_NODES, ast.Lambda))
        for child in ast.iter_child_nodes(node):
            visit(child, inside)

    visit(tree, False)
    return found


def docstring_nodes(tree: ast.AST) -> set[int]:
    """Return the ids of the Constant nodes that are docstrings of the module, a class, or a function."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, *FUNCTION_NODES)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


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


def code_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Return the number of lines a function spans after its docstring."""
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and len(body) > 1:
        body = body[1:]
    return (node.end_lineno or node.lineno) - body[0].lineno + 1


# ---------------------------------------------------------------------------
# Registration, with and without the vendor library


def test_the_package_holds_the_profiler_modules() -> None:
    names = sorted(path.name for path in profiler_sources())
    assert set(EXPECTED_MODULES) <= set(names), names


def test_importing_lassi_profilers_registers_three_profilers_and_no_vendor_library() -> None:
    profilers_module()
    code = (
        "import json, sys\n"
        "sys.modules['pynvml'] = None\n"
        "import lassi.profilers\n"
        "from lassi.core.registry import DEFAULT_REGISTRY as registry\n"
        "names = registry.names('Profiler')\n"
        "print(json.dumps({'names': names,\n"
        "    'capabilities': {name: sorted(registry.get('Profiler', name).capabilities) for name in names},\n"
        "    'config_keys': {name: sorted(registry.get('Profiler', name).config_keys) for name in names}}))\n"
    )
    found = python_says(code)
    assert found["names"] == REGISTERED, "the package registers its profilers without the vendor library"
    assert found["capabilities"] == {
        "nvml": ["supports_power", "takes_device"],
        "rocm_smi": ["supports_power", "takes_device"],
        "timing": [],
    }
    assert found["config_keys"] == {"nvml": ["interval_ms"], "rocm_smi": ["interval_ms"], "timing": []}


def test_importing_the_runner_registers_the_profilers_and_loads_no_vendor_library() -> None:
    profilers_module()
    code = (
        "import json, sys\n"
        "import lassi.core.runner, lassi.cli\n"
        "import lassi.profilers.nvml, lassi.profilers.rocm_smi, lassi.profilers.timing, lassi.profilers.power\n"
        "from lassi.core.registry import DEFAULT_REGISTRY as registry\n"
        f"loaded = sorted(name for name in {list(VENDOR_MODULES)!r} if sys.modules.get(name) is not None)\n"
        "print(json.dumps({'names': registry.names('Profiler'), 'loaded': loaded}))\n"
    )
    found = python_says(code)
    assert set(REGISTERED) <= set(found["names"]), "importing the runner registers every profiler"
    assert found["loaded"] == [], "a vendor library is imported only when a profiler's source is built"


# ---------------------------------------------------------------------------
# Imports, processes, and SMI tools


def test_profiler_modules_import_only_stdlib_and_lassi_at_module_level() -> None:
    allowed = set(sys.stdlib_module_names) | {"lassi"}
    hits = []
    for path in profiler_sources():
        for root, in_function in imports(parsed(path)):
            if in_function and path.name == "nvml.py" and root == "pynvml":
                continue
            if root not in allowed:
                hits.append(f"{path.name}: {root} ({'in a function' if in_function else 'at module level'})")
    assert not hits, f"profiler modules import outside the standard library and lassi: {hits}"


def test_only_the_nvml_module_imports_pynvml_and_only_inside_a_function() -> None:
    found = {path.name: imports(parsed(path)) for path in profiler_sources()}
    users = sorted(name for name, entries in found.items() if any(root == "pynvml" for root, _ in entries))
    assert users == ["nvml.py"], users
    assert all(in_function for root, in_function in found["nvml.py"] if root == "pynvml"), (
        "pynvml is imported when the source is built, never when the module is imported (the registration rule)"
    )
    for name, entries in found.items():
        assert not {root for root, _ in entries} & FORBIDDEN_ANYWHERE, f"{name} imports a rejected vendor library"


def test_no_profiler_code_names_an_smi_tool_or_starts_a_process() -> None:
    hits = []
    for path in profiler_sources():
        tree = parsed(path)
        roots = {root for root, _ in imports(tree)}
        hits += [f"{path.name}: imports {root}" for root in sorted(roots & PROCESS_MODULES)]
        skip = docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
                hits += [f"{path.name}:{node.lineno} names {tool}" for tool in SMI_TOOLS if tool in node.value.lower()]
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
                if node.attr in PROCESS_CALLS:
                    hits.append(f"{path.name}:{node.lineno} uses os.{node.attr}")
    assert not hits, f"a profiler starts no process and names no SMI tool in code (OQ-002): {hits}"


def test_lassi_core_starts_no_sampling_thread() -> None:
    hits = []
    for path in sorted(CORE_DIR.rglob("*.py")):
        roots = {root for root, _ in imports(parsed(path))}
        if roots & {"threading", "_thread", "concurrent"}:
            hits.append(path.name)
    assert not hits, f"the sampling thread lives in lassi/profilers, never in lassi/core: {hits}"


# ---------------------------------------------------------------------------
# Readability and Agent Rule 3


@pytest.mark.parametrize("name", EXPECTED_MODULES)
def test_profiler_modules_are_documented_typed_ascii_lf(name: str) -> None:
    path = next((item for item in profiler_sources() if item.name == name), None)
    assert path is not None, f"lassi/profilers/{name} does not exist yet (task P17.7)"
    raw = path.read_bytes()
    assert raw.isascii() and b"\r" not in raw, f"{name} is not plain ASCII with LF"
    source = raw.decode("ascii")
    tree = ast.parse(source)
    assert ast.get_docstring(tree), f"{name} has no module docstring"
    assert "from __future__ import annotations" in source, name
    undocumented = [qual for qual, node in public_defs(tree) if not ast.get_docstring(node)]
    assert not undocumented, f"{name}: no docstring on {undocumented}"
    untyped = [qual for qual, node in public_defs(tree) if isinstance(node, FUNCTION_NODES) and not is_typed(node)]
    assert not untyped, f"{name}: missing type hints on {untyped}"
    long = [
        f"{node.name} ({code_lines(node)} lines)"
        for node in ast.walk(tree)
        if isinstance(node, FUNCTION_NODES) and code_lines(node) > MAX_FUNCTION_LINES
    ]
    assert not long, f"{name}: functions over {MAX_FUNCTION_LINES} lines: {long}"


@pytest.mark.parametrize("name", EXPECTED_MODULES)
def test_profiler_modules_name_no_project_or_host(name: str) -> None:
    path = PROFILERS_DIR / name
    if not path.exists():
        pytest.fail(f"lassi/profilers/{name} does not exist yet (task P17.7)")
    source = path.read_bytes().decode("ascii")
    found = [word for word in PROJECT_NAMES if word in source.lower()]
    assert not found, f"lassi/profilers/{name} names projects or hosts (Agent Rule 3): {found}"
    assert "projects" not in {root for root, _ in imports(ast.parse(source))}, name
